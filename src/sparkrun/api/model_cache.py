"""Explicit model cache inspection, inventory import and targeted repair.

Inspection is read-only and offline. Resolution contacts the model origin;
repair is the only operation here that downloads model payloads. These helpers
never start/evict workloads and do not change api.materialize's contract.

.. warning::
   **Experimental.** Function signatures and the request/report types
   (``ModelArtifactRequest``, ``ModelArtifactManifest``, ``CacheValidationReport``,
   ``CacheReport``) are core types that may change in a minor release.
"""

from __future__ import annotations

from collections.abc import Callable
import os

from sparkrun.models.artifacts import (
    CacheValidationReport,
    ModelArtifactError,
    ModelArtifactManifest,
    ModelArtifactRequest,
    ModelInventoryUnavailable,
)
from sparkrun.models.host_io import ModelHostIO
from sparkrun.models.cache_reports import CacheReport, ClaimReconciliation
from sparkrun.models.manifest import check_request, fetch_inventory, manifest_path, select_manifest
from sparkrun.models.preparation import PreparedModel, prepare_model


def _ssh(ssh_kwargs, sctx):
    """SSH arguments for host IO: *sctx*'s configured ones, overridden per key.

    Without *sctx* the explicit arguments are used as given (the historical
    behaviour), so callers that already pass ``ssh_kwargs`` see no change.
    """
    if sctx is None:
        return ssh_kwargs
    from sparkrun.orchestration.primitives import build_ssh_kwargs

    return {**build_ssh_kwargs(sctx.config), **(ssh_kwargs or {})}


def resolve_manifest(request: ModelArtifactRequest, *, token: str | None = None, cache_dir: str | None = None) -> ModelArtifactManifest:
    """Resolve authoritative metadata, including a small weight index if present."""
    return select_manifest(request, fetch_inventory(request, token=token, cache=cache_dir))


def inspect_model(
    request: ModelArtifactRequest, *, cache_dir: str, hosts=("localhost",), checksums=False, ssh_kwargs=None, session=None, sctx=None
) -> tuple[CacheValidationReport, ...]:
    reports = []
    with ModelHostIO(_ssh(ssh_kwargs, sctx), session=session) as io:
        for host in hosts:
            root = io.root(host, cache_dir)
            stored = io.read(host, str(manifest_path(root, request)))
            if stored is None:
                reports.append(
                    CacheValidationReport(
                        host,
                        "",
                        "checksum" if checksums else "metadata",
                        "unverified",
                        detail="no saved manifest; resolve or import authoritative inventory first",
                    )
                )
                continue
            manifest = ModelArtifactManifest.from_json(stored)
            check_request(manifest, request)
            reports.append(io.observe(host, manifest, root, checksums=checksums))
    return tuple(reports)


def import_manifest(
    request: ModelArtifactRequest,
    manifest: ModelArtifactManifest,
    *,
    cache_dir: str,
    hosts=("localhost",),
    checksums=False,
    ssh_kwargs=None,
    session=None,
    sctx=None,
) -> tuple[CacheValidationReport, ...]:
    """Import operator-supplied expected metadata, never infer it by listing files.

    Import can record an incomplete inventory for later repair. The returned
    observations explicitly distinguish that from a complete cache.
    """
    check_request(manifest, request)
    reports = []
    with ModelHostIO(_ssh(ssh_kwargs, sctx), session=session) as io:
        for host in hosts:
            root = io.root(host, cache_dir)
            io.lock(host, root, request.repo)
            reports.append(io.observe(host, manifest, root, checksums=checksums))
            io.publish(host, root, request.repo, str(manifest_path(root, request)), manifest.to_json())
    return tuple(reports)


def import_model(
    request: ModelArtifactRequest,
    manifest: ModelArtifactManifest | None = None,
    *,
    cache_dir: str,
    hosts=("localhost",),
    checksums=False,
    fetch_manifest=False,
    offline=False,
    token: str | None = None,
    confirm_manifest: Callable[[ModelArtifactManifest], bool] | None = None,
    ssh_kwargs=None,
    session=None,
    sctx=None,
) -> tuple[CacheValidationReport, ...]:
    """Import supplied metadata, or explicitly resolve it once from the origin.

    Without a manifest, ``fetch_manifest=True`` permits origin resolution;
    ``offline=True`` always requires a supplied manifest. Origin credentials
    default to the local HF configuration. No model weights are downloaded.

    When provided, ``confirm_manifest`` reviews the resolved manifest before
    opening any target cache. Returning False cancels the import. Supplied
    manifests bypass resolution and confirmation, even with fetch enabled.
    """
    if manifest is None:
        if offline:
            raise ModelArtifactError("Offline import requires a manifest; provide --manifest or resolve one on an online machine")
        if not fetch_manifest:
            raise ModelArtifactError("Import requires a manifest or explicit origin consent with fetch_manifest=True")
        from sparkrun.core.config import resolve_hf_token

        try:
            manifest = resolve_manifest(request, token=token if token is not None else resolve_hf_token())
        except ModelInventoryUnavailable as error:
            raise ModelInventoryUnavailable(
                "%s; supply --manifest, or check origin access and HF credentials before retrying" % error
            ) from error
        check_request(manifest, request)
        if confirm_manifest is not None and not confirm_manifest(manifest):
            raise ModelArtifactError("Manifest import cancelled")
    return import_manifest(
        request,
        manifest,
        cache_dir=cache_dir,
        hosts=hosts,
        checksums=checksums,
        ssh_kwargs=ssh_kwargs,
        session=session,
        sctx=sctx,
    )


def repair_model(
    request: ModelArtifactRequest,
    *,
    cache_dir: str,
    hosts=("localhost",),
    local_cache_dir=None,
    transfer_mode="local",
    checksums=False,
    offline=False,
    dry_run=False,
    token=None,
    ssh_kwargs=None,
    config=None,
    session=None,
    manifest=None,
    sctx=None,
) -> PreparedModel | None:
    """Repair selected missing/corrupt paths and revalidate all requested hosts.

    With *sctx*, ``config`` and the SSH arguments default to its configuration
    (explicit values still win, ``ssh_kwargs`` per key).
    """
    return prepare_model(
        request.model,
        list(hosts),
        cache_dir=cache_dir,
        local_cache_dir=local_cache_dir,
        revision=request.revision,
        projector=request.projector,
        file_selection=request.file_selection,
        allow_patterns=request.allow_patterns,
        ignore_patterns=request.ignore_patterns,
        endpoint=request.endpoint,
        transfer_mode=transfer_mode,
        checksums=checksums,
        offline=offline,
        dry_run=dry_run,
        token=token,
        ssh_kwargs=_ssh(ssh_kwargs, sctx),
        config=config if config is not None or sctx is None else sctx.config,
        session=session,
        manifest=manifest,
    )


def inventory_cache(*, cache_dir, hosts=("localhost",), ssh_kwargs=None, session=None, sctx=None) -> tuple[CacheReport, ...]:
    """Read host-local inventory, usage, pins and outstanding workload claims."""
    from sparkrun.models.cache import cache_operation

    return cache_operation(cache_dir=cache_dir, hosts=hosts, ssh_kwargs=_ssh(ssh_kwargs, sctx), session=session)


def prune_cache(
    *,
    cache_dir,
    hosts=("localhost",),
    dry_run=True,
    min_age_days=7,
    max_bytes=None,
    min_free_bytes=None,
    ssh_kwargs=None,
    session=None,
    sctx=None,
) -> tuple[CacheReport, ...]:
    """Evict eligible revisions in LRU order; preview by default.

    max_bytes budgets private managed snapshot allocations, including blobs only
    that snapshot links to (evicted with it), excluding shared blobs and untracked data. min_free_bytes measures actual filesystem space.
    Without a space target, remove every eligible revision older than min_age_days.
    """
    from sparkrun.models.cache import cache_operation, age_seconds

    return cache_operation(
        cache_dir=cache_dir,
        hosts=hosts,
        ssh_kwargs=_ssh(ssh_kwargs, sctx),
        session=session,
        action="prune",
        dry_run=dry_run,
        min_age_seconds=age_seconds(min_age_days),
        max_bytes=max_bytes,
        min_free_bytes=min_free_bytes,
    )


def _require_revision(request):
    import re

    if not request.revision or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", request.revision):
        raise ModelArtifactError("a full model commit or a ref name such as main is required")


def pin_revision(
    request, *, cache_dir, hosts=("localhost",), pinned=True, ssh_kwargs=None, session=None, sctx=None
) -> tuple[CacheReport, ...]:
    """Pin/unpin an already tracked revision on selected cache hosts.

    ``request.revision`` is a full commit, or a ref such as ``main`` that each
    cache resolves through its own ``refs/`` (reports carry the commit).
    """
    from sparkrun.models.cache import cache_operation

    _require_revision(request)
    return cache_operation(
        cache_dir=cache_dir,
        hosts=hosts,
        ssh_kwargs=_ssh(ssh_kwargs, sctx),
        session=session,
        action="pin",
        dry_run=False,
        repo=request.repo,
        revision=request.revision,
        endpoint=request.endpoint,
        pinned=pinned,
    )


def reconcile_claim(
    request, *, cache_dir, claim, confirm_inactive=False, hosts=("localhost",), ssh_kwargs=None, session=None, sctx=None
) -> tuple[CacheReport, ...]:
    """Release an interrupted workload claim after explicit operator inspection.

    Normal stop releases exact claims automatically. Never infer inactivity from
    age, a dead controller, an unavailable host, or missing local job metadata.
    ``request.revision`` may be a ref, resolved per cache as in :func:`pin_revision`.
    """
    from sparkrun.models.cache import cache_operation

    _require_revision(request)
    return cache_operation(
        cache_dir=cache_dir,
        hosts=hosts,
        ssh_kwargs=_ssh(ssh_kwargs, sctx),
        session=session,
        action="reconcile",
        dry_run=False,
        repo=request.repo,
        revision=request.revision,
        endpoint=request.endpoint,
        claim=claim,
        confirm_inactive=confirm_inactive,
    )


#: Default minimum claim age for :func:`reconcile_claims`. A launch claims its
#: model before its containers exist, so a young claim with no workload may be
#: a launch still preparing (possibly from another control machine).
RECONCILE_MIN_AGE_SECONDS = 3600
#: Added to every minimum age, including zero. A claim's ``created`` is the
#: clock of the controller that launched it, which may be another machine's.
RECONCILE_CLOCK_SKEW_SECONDS = 300


def _jobs_on_unswept_executors(swept, sctx):
    """Recorded workloads launched through an executor the sweep did not query.

    An executor whose feature flag was turned off after a launch is not swept,
    so its workloads look absent. Returns ``(unswept, recorded)``: workload id
    to executor for those, and every workload id with a job record here mapped
    to the executor it records (``None`` when the record does not say).
    """
    from sparkrun.api._jobs import list_jobs

    unswept: dict[str, str] = {}
    recorded: dict[str, str | None] = {}
    for job in list_jobs(sctx=sctx):
        executor = (job.metadata or {}).get("executor")
        recorded[job.workload_id] = executor if isinstance(executor, str) and executor else None
        if executor and swept and executor not in swept:
            unswept[job.workload_id] = executor
    return unswept, recorded


def _parse_claim(claim: dict) -> tuple[str, tuple[str, ...], float] | None:
    """``(workload, hosts, created)`` of a well-formed claim record, else None."""
    workload, hosts, created = claim.get("workload"), claim.get("hosts"), claim.get("created")
    if not isinstance(workload, str) or not isinstance(hosts, list) or not all(isinstance(h, str) for h in hosts):
        return None
    # A bool is an int, and is not a timestamp.
    if not isinstance(created, (int, float)) or isinstance(created, bool):
        return None
    return workload, tuple(hosts), float(created)


def _executor_hint(name, gated):
    """*name*, plus how to make status sweep it when this controller gates it off."""
    flag = gated.get(name)
    return name if flag is None else "%s (disabled here; enable %s to sweep it)" % (name, flag)


def reconcile_claims(
    *,
    cache_dir,
    hosts=("localhost",),
    cluster=None,
    repo=None,
    dry_run=True,
    min_age_seconds=RECONCILE_MIN_AGE_SECONDS,
    ssh_kwargs=None,
    session=None,
    sctx=None,
) -> tuple[ClaimReconciliation, ...]:
    """Find workload claims whose workload is gone and release them; preview by default.

    A claim is released only when it names a workload of this distribution,
    one status sweep of *hosts* (through *cluster*'s executors, like
    ``status``) observed every host in the claim and found the workload
    running on none of them, the sweep queried the executor the claim records
    (a claim recording none: no job record here places the workload, or, with
    no record, any workload, under an executor the sweep could not query, and
    no executor in the sweep's scope is feature-gated off here), no pending
    operation on this control machine names it, and the claim is at least
    *min_age_seconds* old (plus :data:`RECONCILE_CLOCK_SKEW_SECONDS`, since the
    claim was timestamped by whichever controller launched it). Anything else
    is reported and kept. Limit to one model with *repo*.
    """
    import math
    import time

    from sparkrun.api._status import status as cluster_status
    from sparkrun.core.application_profile import get_application_profile
    from sparkrun.core.ownership import resource_namespace_of
    from sparkrun.core.pending_ops import list_pending_ops
    from sparkrun.models.cache import cache_operation, resolve_cache_locations
    from sparkrun.orchestration.executors._base import gated_off_executors

    if type(min_age_seconds) not in (int, float) or not math.isfinite(min_age_seconds) or min_age_seconds < 0:
        raise ValueError("minimum claim age must be finite and nonnegative")
    hosts = tuple(hosts)
    ssh_kwargs = _ssh(ssh_kwargs, sctx)
    locations = resolve_cache_locations(cache_dir=cache_dir, hosts=hosts, ssh_kwargs=ssh_kwargs, session=session, verify=not dry_run)
    reports = cache_operation(
        cache_dir=cache_dir, hosts=hosts, ssh_kwargs=ssh_kwargs, session=session, locations=locations, action="inventory"
    )
    claimed = [(r, e) for r in reports for e in r.entries if e.claims and (repo is None or e.repo == repo)]
    if not claimed:
        return ()
    snapshot = cluster_status(list(hosts), cluster=cluster, sctx=sctx)
    running = set(snapshot.running_cluster_ids())
    observed = set(hosts) - set(snapshot.observation_errors)
    coverage = tuple(getattr(snapshot, "coverage", ()))
    swept = {c.target.executor for c in coverage}
    scopes = {c.status_scope for c in coverage}
    # Executors this controller has turned off are registered but invisible to
    # status, so their workloads (possibly launched by another controller,
    # which leaves no record here) would look absent.
    gated = {name: flag for name, (flag, scope) in gated_off_executors().items() if name not in swept and (not scopes or scope in scopes)}
    pending = {op.get("job_cluster_id") for op in list_pending_ops()} - {"", None}
    unswept, recorded = _jobs_on_unswept_executors(swept, sctx)
    namespace = get_application_profile().resource_namespace
    now = time.time()
    results = []
    for report, entry in claimed:
        for token, claim in sorted(entry.claims.items()):
            claim = claim if isinstance(claim, dict) else {}
            parsed = _parse_claim(claim)
            workload, claim_hosts, created = parsed or ("", (), 0.0)
            claim_executor = claim.get("executor")
            unobserved = [h for h in claim_hosts if h not in observed]
            if parsed is None:
                status, reason = "unverified", "malformed claim record"
            elif resource_namespace_of(workload) != namespace:
                # Status reports only this distribution's workloads, so another
                # distribution's live workload would look absent.
                status, reason = "foreign", "workload belongs to another distribution or has an unrecognized id"
            elif workload in running:
                status, reason = "running", "workload is running"
            elif not claim_hosts or unobserved:
                status, reason = "unverified", "not observed: " + (", ".join(unobserved) or "claim names no hosts")
            elif isinstance(claim_executor, str) and claim_executor not in swept:
                status, reason = "unverified", "its executor was not swept: " + _executor_hint(claim_executor, gated)
            elif (
                not isinstance(claim_executor, str)
                and recorded.get(workload) not in swept
                and (workload in unswept or (unswept and workload not in recorded) or gated)
            ):
                # A claim that does not name its executor (written before claims
                # recorded one) may be under any of these. A job record here
                # naming a swept executor settles it: without that, executor.local
                # being off on stable would keep every older claim unverified.
                names = {unswept[workload]} if workload in unswept else {*unswept.values(), *gated}
                status, reason = (
                    "unverified",
                    "may run under an executor this sweep could not query: " + ", ".join(_executor_hint(n, gated) for n in sorted(names)),
                )
            elif workload in pending:
                status, reason = "preparing", "a pending operation on this machine names the workload"
            elif now - created < min_age_seconds + RECONCILE_CLOCK_SKEW_SECONDS:
                status, reason = "recent", "younger than the minimum age; a launch may still be preparing"
            else:
                status, reason = "orphaned", "workload absent from every claimed host"
            action = None
            if status == "orphaned":
                action = "would_release"
                if not dry_run:
                    try:
                        cache_operation(
                            cache_dir=cache_dir,
                            ssh_kwargs=ssh_kwargs,
                            session=session,
                            locations=(report.location,) if report.location is not None else None,
                            hosts=(report.host,),
                            action="reconcile",
                            dry_run=False,
                            repo=entry.repo,
                            revision=entry.revision,
                            claim=token,
                            confirm_inactive=True,
                        )
                        action = "released"
                    except (ModelArtifactError, ValueError, OSError) as exc:
                        action, reason = "failed", str(exc)
            results.append(
                ClaimReconciliation(
                    entry.repo, entry.revision, token, workload, claim_hosts, created, status, reason, action, report.location
                )
            )
    return tuple(results)


def adopt_cache(
    request: ModelArtifactRequest | None = None,
    *,
    cache_dir,
    hosts=("localhost",),
    all_models=False,
    endpoint=None,
    dry_run=True,
    offline=False,
    last_used="now",
    pinned=False,
    token=None,
    ssh_kwargs=None,
    session=None,
    sctx=None,
) -> tuple[CacheReport, ...]:
    """Start tracking snapshots downloaded outside sparkrun; preview by default.

    Adopts every untracked revision of ``request.model`` (or only
    ``request.revision``, a commit or ref), or of every model with
    *all_models*: one of the two is required, so one command cannot enroll
    everything other tools on the host also use. Online (default), each
    present file must match the model origin's inventory for that commit;
    *offline* accepts only snapshots in the Hub's blob-linked layout. Only the
    usage record is written, never a manifest. *last_used* is ``now`` (default:
    nothing becomes prune-eligible on adoption) or ``mtime`` (the snapshot's
    newest entry). *pinned* adopts and protects in one step. *endpoint* is the
    origin for *all_models* (a request carries its own); default ``HF_ENDPOINT``.

    Adopted revisions then age by sparkrun's use only: with automatic pruning
    enabled, one that other tools still use becomes a candidate (with the blobs
    only it uses) once the retention window passes. *pinned* prevents that.
    """
    from sparkrun.core.config import resolve_hf_token
    from sparkrun.models.adoption import adopt_snapshots

    if (request is None) == (not all_models):
        raise ValueError("name a model to adopt, or pass all_models (not both)")
    return adopt_snapshots(
        cache_dir=cache_dir,
        hosts=hosts,
        repo=request.repo if request is not None else None,
        revision=request.revision if request is not None else None,
        endpoint=request.endpoint if request is not None else endpoint or os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
        offline=offline,
        dry_run=dry_run,
        last_used=last_used,
        pinned=pinned,
        token=token if token is not None or offline else resolve_hf_token(),
        ssh_kwargs=_ssh(ssh_kwargs, sctx),
        session=session,
    )


def delete_model(
    request,
    *,
    cache_dir,
    hosts=("localhost",),
    dry_run=True,
    force=False,
    allowed_revisions=None,
    ssh_kwargs=None,
    session=None,
    sctx=None,
) -> tuple[CacheReport, ...]:
    """Delete one revision (``request.revision``), or every revision, of a model; preview by default.

    Unlike prune this ignores pins, retention age and tracking: naming the model
    is the authorization. Revisions with outstanding workload claims are refused
    unless *force*, and an active download or busy publication lock always
    refuses. Blobs are removed only when no other snapshot or repository in the
    cache references them. Saved manifests are kept, so ``repair`` can restore it.

    Applying previews every host first and deletes nothing anywhere if any host
    would refuse (``target_met`` false on its report); those preview reports are
    returned instead. Pass the revisions a user confirmed as *allowed_revisions*
    so a revision that appeared after the confirmation is refused, not deleted.
    """
    from sparkrun.models.cache import cache_operation, resolve_cache_locations

    ssh = _ssh(ssh_kwargs, sctx)
    # Applying: verify the grouping once and gate on a preview of exactly the
    # caches the apply will change. An unverified preview could have merged a
    # cache the verified apply splits off, which no refusal would then cover.
    locations = None if dry_run else resolve_cache_locations(cache_dir=cache_dir, hosts=hosts, ssh_kwargs=ssh, session=session, verify=True)
    options = dict(
        cache_dir=cache_dir,
        hosts=hosts,
        ssh_kwargs=ssh,
        session=session,
        locations=locations,
        action="delete",
        repo=request.repo,
        revision=request.revision,
        force=force,
        allowed=None if allowed_revisions is None else sorted(allowed_revisions),
    )
    preview = cache_operation(dry_run=True, **options)
    if dry_run or not all(report.target_met for report in preview):
        return preview
    return cache_operation(dry_run=False, **options)
