"""Core-owned model preparation: one immutable inventory for every transfer leg."""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging
import os
from pathlib import PurePosixPath
import re
import uuid

from sparkrun.core.model_distribution import (
    ModelCopyRequest,
    ModelDistributionFailed,
    ModelPullRequest,
    ModelTarget,
    record_binding,
    transfer_timeouts,
    try_model_transfer,
)
from sparkrun.models.acquisition import run_adapter
from sparkrun.models.artifacts import (
    CacheValidationReport,
    ModelArtifactError,
    ModelArtifactManifest,
    ModelArtifactRequest,
    ModelInventoryUnavailable,
    ValidatedModelBinding,
    normalize_file_patterns,
)
from sparkrun.models.host_io import ModelHostIO
from sparkrun.models.disk_space import ModelDiskSpaceError, disk_failure, require_model_space
from sparkrun.models.manifest import check_request, fetch_inventory, manifest_path, select_manifest
from sparkrun.models.snapshot_links import link_existing_blobs
from sparkrun.models.transport import builtin_copy
from sparkrun.models.transfer_cleanup import cleanup_transfer_partials
from sparkrun.utils.shell import quote

logger = logging.getLogger(__name__)
_REPAIRABLE = {"missing", "wrong_size", "checksum_mismatch"}


@dataclass(frozen=True)
class PreparedModel:
    manifest: ModelArtifactManifest
    bindings: tuple[ValidatedModelBinding, ...]


def _transfer_limits(config) -> dict:
    """``ModelCopyRequest`` timeout fields from ``model_transfer_*`` config."""
    limits = transfer_timeouts(config)
    return {"timeout": limits.timeout, "idle_timeout": limits.idle_timeout}


def _report_error(report: CacheValidationReport) -> str:
    paths = "; ".join(f.path + ": " + f.reason for f in report.failures[:10])
    return "%s (%s, %s): %s" % (report.host, report.status, report.level, paths or report.detail)


def _repair_files(report: CacheValidationReport) -> tuple[str, ...]:
    if report.complete:
        return ()
    if not report.failures or any(f.reason not in _REPAIRABLE for f in report.failures):
        raise ModelArtifactError("model cache cannot be repaired by downloading: " + _report_error(report))
    return tuple(f.path for f in report.failures)


def _transfer(request: ModelCopyRequest, *, pull=False):
    try:
        return try_model_transfer(request, pull=True) if pull else try_model_transfer(request)
    except ModelDistributionFailed as exc:
        for target in request.targets:
            reason = disk_failure(exc.failures.get(target.host, ""))
            if reason is None and len(request.targets) == 1:
                reason = disk_failure(exc)
            if reason is not None:
                raise ModelDiskSpaceError(target.host, target.cache_root, reason=reason, action="model transfer") from exc
        raise


def _saved_manifest(
    io: ModelHostIO, request: ModelArtifactRequest, locations: list[tuple[str | None, str]]
) -> ModelArtifactManifest | None:
    for host, root in locations:
        text = io.read(host, str(manifest_path(root, request)))
        if text is not None:
            manifest = ModelArtifactManifest.from_json(text)
            check_request(manifest, request)
            return manifest
    return None


def _acquire(
    io: ModelHostIO,
    host: str | None,
    root: str,
    request: ModelArtifactRequest,
    manifest: ModelArtifactManifest,
    report: CacheValidationReport,
    *,
    token: str | None,
    offline: bool,
    checksums: bool,
    config,
    operation: str,
) -> None:
    files = _repair_files(report)
    if not files:
        cleanup_transfer_partials(io, host, root, manifest, report)
        return
    if offline:
        raise ModelArtifactError("offline: required model files are unavailable: " + _report_error(report))
    require_model_space(io, manifest, [(host, root, files)])
    result = None
    if host is not None:
        pull = ModelPullRequest(
            manifest,
            host,
            root,
            (ModelTarget(host, root, host, files),),
            operation,
            config=config,
            session=io.session,
            cancelled=io.cancelled,
            token=lambda: token,
            **_transfer_limits(config),
        )
        result = _transfer(pull, pull=True)
    if result is None:
        try:
            run_adapter(io, host, request, root, token=token, manifest=manifest, files=files, config=config)
        except OSError as exc:
            if reason := disk_failure(exc):
                raise ModelDiskSpaceError(host, root, reason=reason, action="model download") from exc
            raise
    verified = io.observe(host, manifest, root, checksums=checksums)
    if not verified.complete:
        raise ModelArtifactError("model source still invalid after one repair: " + _report_error(verified))
    cleanup_transfer_partials(io, host, root, manifest, verified)
    from sparkrun.models.usage import record_use

    record_use(io, host, root, manifest, event="acquired", acquired=True)


def prepare_model(
    model: str,
    hosts: list[str],
    *,
    cache_dir: str,
    local_cache_dir: str | None = None,
    revision: str | None = None,
    projector: str | None = None,
    file_selection: str = "auto",
    allow_patterns: tuple[str, ...] | list[str] | None = None,
    ignore_patterns: tuple[str, ...] | list[str] | None = None,
    endpoint: str | None = None,
    transfer_mode: str = "local",
    transfer_hosts: list[str] | None = None,
    worker_transfer_hosts: list[str] | None = None,
    ssh_kwargs: dict | None = None,
    token: str | None = None,
    offline: bool = False,
    dry_run: bool = False,
    checksums: bool = False,
    config=None,
    session=None,
    manifest: ModelArtifactManifest | None = None,
    auto_delegated: bool = False,
    allow_cached_inventory: bool = False,
) -> PreparedModel | None:
    """Validate every target; transport and downloader exits never prove readiness.

    A supplied manifest supports explicit offline imports. Host sessions are
    borrowed when supplied. Native restore strategies skip this operation via
    their existing LaunchAssetPolicy, rather than fabricating HF artifacts.
    """
    if not hosts or dry_run:
        return None
    if len(hosts) != len(set(hosts)):
        raise ModelArtifactError("model target hosts must be unique")
    if transfer_mode not in {"local", "push", "delegated", "pull"}:
        raise ModelArtifactError("unsupported model transfer mode: " + transfer_mode)
    if transfer_hosts is not None and len(transfer_hosts) != len(hosts):
        raise ModelArtifactError("model transfer addresses must align with target hosts")
    if worker_transfer_hosts is not None and len(worker_transfer_hosts) != len(hosts) - 1:
        raise ModelArtifactError("worker transfer addresses must align with model workers")
    request = ModelArtifactRequest(
        model,
        revision,
        endpoint or os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
        projector,
        file_selection,
        normalize_file_patterns(allow_patterns, "allow_patterns"),
        normalize_file_patterns(ignore_patterns, "ignore_patterns"),
    )
    operation = uuid.uuid4().hex
    ssh = ssh_kwargs or {}
    head = hosts[0]
    with ModelHostIO(ssh, session=session) as io:
        roots = {host: io.root(host, cache_dir) for host in hosts}
        source_host = None if transfer_mode in {"local", "push"} else head
        source_root = io.root(None, local_cache_dir or cache_dir) if source_host is None else roots[head]
        if manifest is not None:
            check_request(manifest, request)
        saved = None
        if manifest is None and (allow_cached_inventory or offline or (revision is not None and re.fullmatch(r"[0-9a-f]{40}", revision))):
            saved = _saved_manifest(io, request, [(source_host, source_root), *roots.items()])
            if offline or (revision is not None and re.fullmatch(r"[0-9a-f]{40}", revision)):
                manifest = saved
        if manifest is None:
            if offline:
                raise ModelInventoryUnavailable(
                    "offline: no saved authoritative model manifest; cache is unverified. Import a manifest or prepare online first"
                )
            try:
                try:
                    inventory = (
                        fetch_inventory(request, token=token, cache=source_root)
                        if source_host is None
                        else run_adapter(io, source_host, request, source_root, token=token)
                    )
                except ModelInventoryUnavailable:
                    if not (transfer_mode == "delegated" and auto_delegated):
                        raise
                    source_host, transfer_mode = None, "push"
                    source_root = io.root(None, local_cache_dir or cache_dir)
                    inventory = fetch_inventory(request, token=token, cache=source_root)
                manifest = select_manifest(request, inventory)
            except ModelInventoryUnavailable:
                if saved is None or not allow_cached_inventory:
                    raise
                manifest = saved
                logger.warning(
                    "Cannot refresh inventory for %s; validating saved revision %s (legacy compatibility)", model, manifest.revision
                )
        from sparkrun.models.cache import auto_prune
        from sparkrun.models.usage import record_use

        # Most launches need only their target caches. Do not serialize a warm
        # launch behind an unrelated control-cache download or multi-minute copy.
        locations: dict[tuple[str | None, str], None] = dict.fromkeys(roots.items())
        auto_prune(io, locations, config=config, exclude_repos=(request.repo,))
        _prepare_locations(io, locations, request, manifest, checksums=checksums)
        reports = {host: io.observe(host, manifest, root, checksums=checksums) for host, root in roots.items()}
        bad = [host for host in hosts if not reports[host].complete]
        if bad and (source_host, source_root) not in locations and not (transfer_mode == "push" and reports[head].complete):
            # Never add the control lease while holding target leases: another
            # cold preparation acquires control first. Drop the healthy leases
            # and reacquire the full set in order, then repeat reuse/observation
            # because eviction or another preparation may have run in the gap.
            io.release_leases()
            auto_prune(io, [(source_host, source_root)], config=config, exclude_repos=(request.repo,))
            locations[source_host, source_root] = None
            _prepare_locations(io, locations, request, manifest, checksums=checksums)
            reports = {host: io.observe(host, manifest, root, checksums=checksums) for host, root in roots.items()}
            bad = [host for host in hosts if not reports[host].complete]
        logger.info(
            "Model %s pinned to %s: %d selected files, %.2f GiB",
            model,
            manifest.revision,
            len(manifest.files),
            sum(f.size for f in manifest.files) / 2**30,
        )
        if bad:
            for host in bad:
                _repair_files(reports[host])  # Do not mistake access/SSH failures for missing payload.
            if transfer_mode == "pull":
                require_model_space(io, manifest, [(host, roots[host], _repair_files(reports[host])) for host in bad])
                for host in bad:
                    current = io.observe(host, manifest, roots[host], checksums=checksums)
                    _acquire(
                        io,
                        host,
                        roots[host],
                        request,
                        manifest,
                        current,
                        token=token,
                        offline=offline,
                        checksums=checksums,
                        config=config,
                        operation=operation,
                    )
            else:
                # A valid push head is already the desired fan-out source.
                if transfer_mode == "push" and reports[head].complete:
                    source_host, source_root = head, roots[head]
                    io.lock(head, source_root, request.repo)
                source_report = io.observe(source_host, manifest, source_root, checksums=checksums)
                if offline and not source_report.complete:
                    # An existing target can seed other caches without origin
                    # egress, even when the controller/head cache is incomplete.
                    available = next((h for h in hosts if reports[h].complete), None)
                    if available is not None:
                        source_host, source_root = available, roots[available]
                        io.lock(source_host, source_root, request.repo)
                        source_report = io.observe(source_host, manifest, source_root, checksums=checksums)
                        transfer_mode = "local"  # Copy to the complete target set, including a missing head.
                require_model_space(
                    io,
                    manifest,
                    [(source_host, source_root, _repair_files(source_report))]
                    + [(host, roots[host], _repair_files(reports[host])) for host in bad],
                )
                _acquire(
                    io,
                    source_host,
                    source_root,
                    request,
                    manifest,
                    source_report,
                    token=token,
                    offline=offline,
                    checksums=checksums,
                    config=config,
                    operation=operation,
                )

                def copy_to(selected: list[str], addresses: list[str]):
                    needed = []
                    domains = {}
                    for host, address in zip(selected, addresses, strict=True):
                        report = io.observe(host, manifest, roots[host], checksums=checksums)
                        if not report.complete:
                            # Shared mounts/host aliases reuse the same held lease.
                            # One writer per file even when several hostnames see it.
                            domain = io.lock(host, roots[host], request.repo)
                            files = _repair_files(report)
                            if domain in domains:
                                index = domains[domain]
                                target = needed[index]
                                needed[index] = replace(target, required_files=tuple(sorted(set(target.required_files) | set(files))))
                            else:
                                domains[domain] = len(needed)
                                needed.append(ModelTarget(host, roots[host], address, files))
                    if not needed:
                        return
                    source_check = io.observe(source_host, manifest, source_root, checksums=checksums)
                    if not source_check.complete:
                        raise ModelArtifactError("model source changed before fan-out: " + _report_error(source_check))
                    require_model_space(io, manifest, [(t.host, t.cache_root, t.required_files) for t in needed])
                    record_use(io, source_host, source_root, manifest, event="distributed")
                    copy = ModelCopyRequest(
                        manifest,
                        source_host,
                        source_root,
                        tuple(needed),
                        operation,
                        offline=offline,
                        config=config,
                        session=io.session,
                        cancelled=io.cancelled,
                        **_transfer_limits(config),
                    )
                    outcome = _transfer(copy)
                    if outcome is None:
                        # Concurrent rsync --mkpath receivers can race on mkdir,
                        # including implied parent directories. Prepare them once
                        # under the held lease before any payload workers start.
                        for target in needed:
                            snapshot = PurePosixPath(target.cache_root) / manifest.relative_snapshot
                            parents = sorted({str((snapshot / path).parent) for path in target.required_files})
                            io.checked(
                                target.host,
                                "set -euo pipefail\nprintf '%s\\0' "
                                + " ".join(quote(path) for path in parents)
                                + " | xargs -0 -r mkdir -p --\n",
                            )
                        outcome = builtin_copy(copy, io, ssh)
                    if any(state == "failed" for state in outcome.outcomes.values()):
                        for host, error in outcome.errors.items():
                            if reason := disk_failure(error):
                                raise ModelDiskSpaceError(host, roots[host], reason=reason, action="model transfer")
                        raise ModelArtifactError(
                            "model transfer failed: " + "; ".join(h + ": " + reason for h, reason in outcome.errors.items())
                        )
                    for target in needed:
                        report = io.observe(target.host, manifest, target.cache_root, checksums=checksums)
                        if not report.complete:
                            raise ModelArtifactError("model target invalid after transfer: " + _report_error(report))

                if transfer_mode == "push" and source_host is None:
                    copy_to([head], [head])
                    source_host, source_root = head, roots[head]
                if transfer_mode in {"push", "delegated"}:
                    copy_to(hosts[1:], worker_transfer_hosts or hosts[1:])
                else:
                    copy_to(hosts, transfer_hosts or hosts)
        bindings = []
        for host in hosts:
            report = io.observe(host, manifest, roots[host], checksums=checksums)
            if not report.complete:
                raise ModelArtifactError("model cache not ready for launch: " + _report_error(report))
            cleanup_transfer_partials(io, host, roots[host], manifest, report)
            binding = ValidatedModelBinding(model, host, roots[host], manifest, report)
            record_use(io, host, roots[host], manifest, acquired=host in bad)
            bindings.append(binding)
            try:
                ref = request.revision or "main"
                if (
                    not request.has_file_patterns
                    and not re.fullmatch(r"[0-9a-f]{40}", ref)
                    and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", ref)
                    and ".." not in ref
                ):
                    # Compatibility with HF loaders and offline pin/inspection
                    # tools. Runtime bindings still use the immutable snapshot.
                    io.publish(
                        host,
                        roots[host],
                        request.repo,
                        roots[host] + "/hub/models--" + request.repo.replace("/", "--") + "/refs/" + ref,
                        manifest.revision,
                    )
            except ModelArtifactError as exc:
                logger.warning("Model validated, but its HF revision ref could not be persisted: %s", exc)
        # Publish only after every required host passed. No partial binding can
        # leak into launch when a later worker fails.
        for binding in bindings:
            record_binding(binding)
        return PreparedModel(manifest, tuple(bindings))


def _prepare_locations(io, locations, request, manifest, *, checksums):
    # Hold leases until durable workload claims are published. The nonce reuses
    # leases across shared mount/host aliases; stable ordering avoids inversion.
    for host, root in sorted(locations, key=lambda item: (item[0] or "", item[1])):
        io.lock(host, root, request.repo)
    # Persist known requirements even if a later repair fails, so offline runs
    # cannot fall back to presence heuristics. Reuse blobs before observing.
    for host, root in locations:
        try:
            _persist_manifest(io, host, root, request, manifest)
        except ModelArtifactError as exc:
            logger.warning("Could not persist expected model inventory on %s: %s", host or "control", exc)
        link_existing_blobs(io, host, root, manifest, checksums=checksums)


def _persist_manifest(io, host, root, request, manifest):
    # Store both the requested-ref lookup and immutable lookup. A later pinned
    # recipe/export can reuse inventory learned from an unpinned launch.
    contents = manifest.to_json()
    paths = {manifest_path(root, request), manifest_path(root, replace(request, revision=manifest.revision))}
    for path in sorted(paths):
        io.publish(host, root, request.repo, str(path), contents)


def validate_preplaced(
    model: str, hosts, *, model_path=None, ssh_kwargs=None, checksums=False, model_format="auto"
) -> tuple[ValidatedModelBinding, ...]:
    """Observe explicitly imported inventories beside pre-placed model paths.

    A directory carries .sparkrun-model-manifest.json; a GGUF file uses
    <filename>.sparkrun-model-manifest.json and paths relative to its parent.
    Without metadata, check basic format/readability but do not create a
    validated binding or claim a known revision or complete inventory.
    """
    from pathlib import PurePosixPath
    from sparkrun.models.observation import local_model_probe_script, observation_script, parse_observations
    from sparkrun.utils.shell import quote

    bindings = []
    with ModelHostIO(ssh_kwargs) as io:
        for host in hosts:
            path = io.root(host, model_path or model)
            directory = io.execute(host, "test -d " + quote(path)).returncode == 0
            snapshot = path if directory else str(PurePosixPath(path).parent)
            from sparkrun.models.usage import protect_preplaced

            protect_preplaced(snapshot, host, ssh_kwargs=ssh_kwargs)
            metadata = path + "/.sparkrun-model-manifest.json" if directory else path + ".sparkrun-model-manifest.json"
            text = io.read(host, metadata)
            if text is None:
                result = io.execute(host, local_model_probe_script(path, model_format), timeout=60)
                if result.returncode:
                    raise ModelArtifactError(
                        "local model preflight failed on %s at %s: %s"
                        % (host, path, result.stderr.decode(errors="replace").strip()[-1500:])
                    )
                logger.warning("Pre-placed model %s on %s is unverified: import expected metadata at %s", model, host, metadata)
                continue
            manifest = ModelArtifactManifest.from_json(text)
            if not directory and manifest.weight_path != PurePosixPath(path).name:
                raise ModelArtifactError("pre-placed GGUF manifest selects a different weight file")
            result = io.execute(host, observation_script(manifest, snapshot, checksums=checksums), timeout=1800 if checksums else 60)
            report = parse_observations(
                host, manifest, result.stdout.decode(errors="replace"), returncode=result.returncode, checksums=checksums
            )
            if not report.complete:
                raise ModelArtifactError("pre-placed model is incomplete: " + _report_error(report))
            bindings.append(ValidatedModelBinding(model, host, snapshot, manifest, report, snapshot))
    if bindings and len(bindings) != len(hosts):
        raise ModelArtifactError("pre-placed model inventory is missing on some launch hosts")
    if len({b.manifest.identity for b in bindings}) > 1:
        raise ModelArtifactError("pre-placed model differs between launch hosts")
    for binding in bindings:
        record_binding(binding)
    return tuple(bindings)
