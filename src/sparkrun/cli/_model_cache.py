"""Operator-visible manifest and cache migration commands."""

from dataclasses import asdict
from functools import partial, wraps
import json
from pathlib import Path
import sys

import click

from sparkrun.api import model_cache as api
from sparkrun.models.artifacts import ModelArtifactError, ModelArtifactManifest, ModelArtifactRequest
from sparkrun.utils.cli_formatters import format_bytes, format_cache_validation_reports, format_model_cache_reports
from ._common import _get_cluster_manager, _get_context, host_options, json_option, resolve_host_context


@click.group("model-cache")
def model_cache():
    """Inspect, repair, retain and prune model caches on cluster hosts.

    Targets the hosts of --cluster, --hosts or --hosts-file, else the default
    cluster, else config default_hosts, else localhost.

    Commands that remove or release (prune, adopt, reconcile) preview by
    default and act only with --apply. delete always previews and then asks
    for confirmation (--yes skips it, --dry-run stops after the preview).
    repair only adds files, so it runs unless given --dry-run.
    """


def _resolve_targets(sctx, *, legacy_hosts, hosts, hosts_file, cluster_name, cache_dir):
    """Resolve target hosts and their HF home like every other cluster command.

    Returns ``(host_list, cache_dir, banner, cluster_name)``; *cluster_name*
    is the effective cluster (``None`` for explicit or default hosts). ``--host`` (repeatable, legacy)
    is merged into ``--hosts``. With no host source at all the control
    machine's own cache (localhost) is the target, as before clusters were
    supported. The cache dir is ``--cache-dir`` → the cluster's ``cache_dir`` →
    the head's HF home probed over SSH (what a launch would use), so a remote
    host is never pointed at the control machine's home directory.
    """
    from sparkrun.core.launcher import resolve_effective_cache_dir
    from sparkrun.orchestration.primitives import build_ssh_kwargs

    config = sctx.config
    hosts = ",".join([*legacy_hosts, *([hosts] if hosts else [])]) or None
    if not (hosts or hosts_file or cluster_name or config.default_hosts or _get_cluster_manager(sctx=sctx).get_default()):
        host_list, cluster_cache, banner, effective = ["localhost"], None, "Hosts: localhost (no cluster configured)", None
    else:
        hctx = resolve_host_context(hosts, hosts_file, cluster_name, config, sctx=sctx)
        host_list, cluster_cache, banner, effective = hctx.host_list, hctx.cluster.cache_dir, hctx.describe(), hctx.cluster_name
    cache_dir = resolve_effective_cache_dir(cache_dir or cluster_cache, host_list, build_ssh_kwargs(config), config)
    return host_list, cache_dir, banner, effective


def _target_options(function):
    """``--cluster/--hosts/--hosts-file``, the hidden legacy ``--host`` and ``--cache-dir``."""
    function = click.option("--cache-dir", default=None, help="HF_HOME on the target hosts, not its hub subdirectory.")(function)
    function = click.option("--host", "legacy_hosts", multiple=True, hidden=True, help="Target host (repeatable; legacy).")(function)
    return host_options(function)


class ByteSizeType(click.ParamType):
    """A byte count typed as ``500G`` / ``1.5TiB`` / ``500GB`` / ``1073741824``."""

    name = "size"

    def convert(self, value, param, ctx):
        from sparkrun.utils.data import parse_byte_size

        try:
            return parse_byte_size(value)
        except ValueError as error:
            self.fail(str(error), param, ctx)


BYTE_SIZE = ByteSizeType()


endpoint_option = click.option(
    "--endpoint",
    default="https://huggingface.co",
    envvar="HF_ENDPOINT",
    help="Model origin URL (defaults to HF_ENDPOINT or huggingface.co). For pin and reconcile it only guards against"
    " a mismatch with the origin recorded for the revision.",
)


def _pop_targets(kwargs):
    return {key: kwargs.pop(key) for key in ("legacy_hosts", "hosts", "hosts_file", "cluster_name", "cache_dir")}


def options(function):
    @wraps(function)
    def wrapped(ctx, model, revision, file_selection, projector, endpoint, checksums, output_json, **kwargs):
        from sparkrun.orchestration.primitives import build_ssh_kwargs

        sctx = _get_context(ctx)
        try:
            host_list, cache_dir, banner, _ = _resolve_targets(sctx, **_pop_targets(kwargs))
            if not output_json:
                click.echo(banner)
            request = ModelArtifactRequest(model, revision, endpoint, projector, file_selection)
            result = function(
                request,
                cache_dir=cache_dir,
                hosts=tuple(host_list),
                checksums=checksums,
                ssh_kwargs=build_ssh_kwargs(sctx.config),
                config=sctx.config,
                **kwargs,
            )
        except (ModelArtifactError, OSError) as error:
            raise click.ClickException(str(error)) from error
        if result is not None:
            if output_json:
                click.echo(json.dumps([asdict(report) for report in result], indent=2))
            else:
                click.echo(format_cache_validation_reports(result))
            if any(not report.complete for report in result):
                ctx.exit(1)

    for decorator in (
        json_option("Print the per-cache reports as JSON."),
        click.option("--checksum", "checksums", is_flag=True, help="Hash selected files; normal inspection checks size and readability."),
        _target_options,
        endpoint_option,
        click.option(
            "--projector",
            default=None,
            help="GGUF only: mmproj file to include, by name substring (default: chosen automatically; none to skip).",
        ),
        click.option(
            "--file-selection",
            type=click.Choice(["auto", "all", "safetensors"]),
            default="auto",
            show_default=True,
            help="auto: with a safetensors index, its shards plus non-weight files, else every file; all: every file;"
            " safetensors: like auto, but require the index.",
        ),
        click.option("--revision", default=None, help="Commit or ref to select (default: main)."),
        click.argument("model"),
    ):
        wrapped = decorator(wrapped)
    return click.pass_context(wrapped)


@model_cache.command("inspect")
@options
def inspect_cache(request, *, config, **kwargs):
    """Observe saved inventory without network access or modifying the cache."""
    return api.inspect_model(request, **kwargs)


@model_cache.command("resolve")
@click.argument("model")
@click.option("--revision", default=None, help="Commit or ref to resolve (default: main).")
@click.option(
    "--file-selection",
    type=click.Choice(["auto", "all", "safetensors"]),
    default="auto",
    show_default=True,
    help="auto: with a safetensors index, its shards plus non-weight files, else every file; all: every file;"
    " safetensors: like auto, but require the index.",
)
@click.option(
    "--projector", default=None, help="GGUF only: mmproj file to include, by name substring (default: chosen automatically; none to skip)."
)
@endpoint_option
@click.option("--cache-dir", default=None, help="HF_HOME on this control machine, used for the weight index download.")
@click.option("--output", required=True, type=click.Path(path_type=Path), help="Write the portable manifest here.")
def resolve(model, revision, file_selection, projector, endpoint, cache_dir, output):
    """Resolve origin metadata online; download only the small weight index.

    Runs on the control machine only; import the manifest onto cluster hosts.
    """
    from sparkrun.core.config import resolve_hf_cache_home, resolve_hf_token

    try:
        request = ModelArtifactRequest(model, revision, endpoint, projector, file_selection)
        manifest = api.resolve_manifest(request, token=resolve_hf_token(), cache_dir=resolve_hf_cache_home(cache_dir))
        output.write_text(manifest.to_json() + "\n")
    except (ModelArtifactError, OSError) as error:
        raise click.ClickException(str(error)) from error
    click.echo("Wrote %d files at revision %s to %s" % (len(manifest.files), manifest.revision, output))


@model_cache.command("import")
@click.option(
    "--manifest",
    "manifest_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Import this inventory offline; otherwise resolve metadata from the model origin.",
)
@click.option(
    "--offline",
    is_flag=True,
    help="Fail rather than contact the model origin when no --manifest is given (--manifest alone never does).",
)
@click.option("--yes", "-y", "assume_yes", is_flag=True, help="Allow origin-resolved metadata without asking for confirmation.")
@options
@click.pass_context
def import_cache(ctx, request, *, manifest_file, offline, assume_yes, config, **kwargs):
    """Import expected metadata and report current cache completeness.

    Prefer a reviewed --manifest for reproducible, offline imports. Otherwise,
    resolve metadata from the origin and confirm the immutable revision before
    saving it. Unattended and --json imports require --yes for origin resolution.
    No model weights are downloaded; missing files remain incomplete.
    """
    manifest = ModelArtifactManifest.from_json(manifest_file.read_text()) if manifest_file is not None else None
    if manifest is None and not offline and not assume_yes and (ctx.params["output_json"] or not _interactive()):
        raise click.ClickException("Origin resolution requires confirmation; pass --yes or supply --manifest")

    def confirm_manifest(resolved):
        click.echo(
            "Resolved %s from %s\nRevision: %s\nSelected files: %d\nTargets: %s (HF_HOME: %s)"
            % (resolved.repo, resolved.endpoint, resolved.revision, len(resolved.files), ", ".join(kwargs["hosts"]), kwargs["cache_dir"]),
            err=True,
        )
        return assume_yes or click.confirm("Save this manifest and validate the target caches?", default=False, err=True)

    return api.import_model(
        request,
        manifest,
        fetch_manifest=manifest is None,
        offline=offline,
        confirm_manifest=confirm_manifest,
        **kwargs,
    )


@model_cache.command("repair")
@click.option("--offline", is_flag=True, help="Permit cluster copies only; no origin calls or tool installation.")
@click.option("--dry-run", is_flag=True, help="Report what would be acquired and copied; change nothing.")
@click.option("--local-cache-dir", default=None, help="HF_HOME on this control machine, the copy source for local/push.")
@click.option(
    "--transfer-mode",
    type=click.Choice(["local", "push", "delegated", "pull"]),
    default="local",
    show_default=True,
    help="How files reach the hosts, as for run: via this machine (local/push), via the first host, or each host from the origin.",
)
@options
def repair(request, **kwargs):
    """Repair selected paths, then validate every target against the manifest."""
    from sparkrun.core.config import resolve_hf_token

    prepared = api.repair_model(request, token=resolve_hf_token(), **kwargs)
    if prepared is None:
        click.echo("Dry run: no model metadata acquired or caches changed.")
        return None
    return tuple(binding.report for binding in prepared.bindings)


def cache_options(action):
    """Shared host/cache options for retention commands; *action* selects the summary footer."""
    return lambda function: _cache_options(function, action)


def _cache_options(function, action):
    @wraps(function)
    def wrapped(ctx, output_json, **kwargs):
        from sparkrun.orchestration.primitives import build_ssh_kwargs

        sctx = _get_context(ctx)
        show_all = kwargs.pop("show_all", False)
        try:
            host_list, cache_dir, banner, _ = _resolve_targets(sctx, **_pop_targets(kwargs))
            if not output_json:
                click.echo(banner)
            reports = function(
                cache_dir=cache_dir,
                hosts=tuple(host_list),
                ssh_kwargs=build_ssh_kwargs(sctx.config),
                **kwargs,
            )
        except (ValueError, OSError) as exc:
            raise click.ClickException(str(exc)) from exc
        if output_json:
            click.echo(json.dumps([asdict(report) for report in reports], indent=2))
        else:
            click.echo(format_model_cache_reports(reports, action=action, show_untracked=show_all))
        if any(not r.get("target_met", True) for r in reports):
            ctx.exit(1)

    return click.pass_context(_target_options(json_option("Print the per-cache reports as JSON.")(wrapped)))


@model_cache.command("list")
@click.option("--all", "show_all", is_flag=True, help="Also list untracked snapshots (always included in --json).")
@cache_options("inventory")
def list_cache(**kwargs):
    """List cached revisions, usage, pins and workload claims without changing files.

    Untracked snapshots are reported but never eligible for pruning. Requires
    Python 3 on cache hosts; no model-origin requests or tool installation.
    """
    return api.inventory_cache(**kwargs)


@model_cache.command("prune")
@click.option("--dry-run/--apply", default=True, show_default=True, help="Preview removals, or explicitly apply the cleanup.")
@click.option(
    "--min-age-days",
    type=click.FloatRange(min=0),
    default=7,
    show_default=True,
    help="Minimum time since preparation, distribution or release.",
)
@click.option(
    "--max-bytes",
    type=BYTE_SIZE,
    help="Target budget for managed snapshots, counting blobs only they use (e.g. 500G, 1.5T; GB/TB are decimal).",
)
@click.option("--min-free-bytes", type=BYTE_SIZE, help="Target free space on the cache filesystem (e.g. 200G).")
@cache_options("prune")
def prune_cache(**kwargs):
    """Remove eligible revisions in least-recently-used order.

    Defaults to a read-only preview; --apply deletes without asking, so review
    the preview first. With no space target, selects all eligible revisions
    older than --min-age-days. Pins, workload claims, busy caches and unknown
    content are protected. Blobs another revision still uses are kept.
    """
    targets = [
        "managed snapshots <= %s" % format_bytes(kwargs["max_bytes"]) if kwargs.get("max_bytes") is not None else None,
        "free space >= %s" % format_bytes(kwargs["min_free_bytes"]) if kwargs.get("min_free_bytes") is not None else None,
    ]
    if any(targets):
        click.echo("Target: " + "; ".join(t for t in targets if t), err=True)
    return api.prune_cache(**kwargs)


@model_cache.command("pin")
@click.argument("model")
@click.option("--revision", required=True, help="Commit ID, or a ref such as main resolved in each cache.")
@click.option("--remove", is_flag=True, help="Remove the retention pin instead of setting it.")
@endpoint_option
@cache_options("pin")
def pin_cache(model, revision, remove, endpoint, **kwargs):
    """Keep a tracked revision out of cleanup on each selected cache host."""
    return api.pin_revision(ModelArtifactRequest(model, revision, endpoint=endpoint), pinned=not remove, **kwargs)


@model_cache.command("reconcile")
@click.argument("model", required=False)
@click.option("--revision", default=None, help="With --claim: commit ID, or a ref such as main.")
@click.option("--claim", default=None, help="Release this exact claim ID shown by model-cache list, instead of finding claims.")
@click.option("--confirm-inactive", is_flag=True, help="With --claim: confirm all workloads/launches sharing it have stopped.")
@click.option("--all", "reconcile_all", is_flag=True, help="Find orphaned claims on every model in the cache.")
@click.option("--dry-run/--apply", default=True, show_default=True, help="Preview, or explicitly release the orphaned claims found.")
@click.option(
    "--min-age-minutes",
    type=click.FloatRange(min=0),
    default=api.RECONCILE_MIN_AGE_SECONDS / 60,
    show_default=True,
    help="Keep younger claims, which may belong to a launch still preparing. Five minutes are added for clock skew.",
)
@endpoint_option
@click.pass_context
@_target_options
@json_option("Print the per-cache reports as JSON.")
def reconcile_cache(
    ctx, model, revision, claim, confirm_inactive, reconcile_all, dry_run, min_age_minutes, endpoint, output_json, **targets
):
    """Release workload claims left by workloads that are no longer running.

    Normal stop releases claims automatically. Given MODEL (or --all for
    every model), sweeps the target hosts' status and releases each claim
    whose workload runs on none of the hosts it claims, provided every one of
    them was reachable, no pending operation here names it, and it is older
    than --min-age-minutes. Defaults to a preview; pass --apply to release.

    With --claim, releases that one claim (MODEL --revision --claim) after you
    have inspected every host in it and pass --confirm-inactive. A dead
    controller or an old timestamp does not prove inactivity. Never deletes
    model files.
    """
    from sparkrun.orchestration.primitives import build_ssh_kwargs
    from sparkrun.utils.cli_formatters import format_claim_reconciliation

    def given(name):
        return ctx.get_parameter_source(name) is click.core.ParameterSource.COMMANDLINE

    automatic = claim is None
    if automatic:
        if bool(model) == reconcile_all:
            raise click.UsageError("Name a MODEL to reconcile, or pass --all.")
        if revision or confirm_inactive or given("endpoint"):
            raise click.UsageError("--revision, --confirm-inactive and --endpoint apply only with --claim.")
    else:
        if reconcile_all or given("dry_run") or given("min_age_minutes"):
            raise click.UsageError("--claim releases one claim; it takes no --all, --apply/--dry-run or --min-age-minutes.")
        if not (model and revision):
            raise click.UsageError("--claim needs MODEL and --revision.")
        if not confirm_inactive:
            raise click.ClickException("Inspect every host in the claim, stop its workloads, then pass --confirm-inactive")
    sctx = _get_context(ctx)
    try:
        host_list, cache_dir, banner, cluster = _resolve_targets(sctx, **targets)
        if not output_json:
            click.echo(banner)
        common = dict(cache_dir=cache_dir, hosts=tuple(host_list), ssh_kwargs=build_ssh_kwargs(sctx.config))
        if automatic:
            results = api.reconcile_claims(
                cluster=cluster,
                repo=ModelArtifactRequest(model).repo if model else None,
                dry_run=dry_run,
                min_age_seconds=min_age_minutes * 60,
                sctx=sctx,
                **common,
            )
        else:
            request = ModelArtifactRequest(model, revision, endpoint=endpoint)
            reports = api.reconcile_claim(request, claim=claim, confirm_inactive=True, **common)
    except (ModelArtifactError, ValueError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc
    if not automatic:
        if output_json:
            click.echo(json.dumps([asdict(report) for report in reports], indent=2))
        else:
            click.echo(format_model_cache_reports(reports, action="reconcile"))
        if any(not r.get("target_met", True) for r in reports):
            ctx.exit(1)
        return
    if output_json:
        click.echo(json.dumps([asdict(result) for result in results], indent=2))
    else:
        click.echo(format_claim_reconciliation(results, dry_run=dry_run))
    if any(result.action == "failed" for result in results):
        ctx.exit(1)


@model_cache.command("adopt")
@click.argument("model", required=False)
@click.option("--revision", default=None, help="Adopt only this commit (or ref, e.g. main).")
@click.option("--all", "all_models", is_flag=True, help="Adopt untracked revisions of every model in the cache.")
@click.option("--dry-run/--apply", default=True, show_default=True, help="Preview what would be adopted, or explicitly adopt it.")
@click.option(
    "--offline",
    is_flag=True,
    help="Verify without the model origin; only Hub-layout snapshots (entries linked into blobs/) qualify.",
)
@click.option(
    "--last-used",
    type=click.Choice(["now", "mtime"]),
    default="now",
    show_default=True,
    help="Start the retention clock now, or at the snapshot's newest file (may make it prune-eligible at once).",
)
@click.option("--pin", "pinned", is_flag=True, help="Also pin the adopted revisions against cleanup.")
@endpoint_option
@cache_options("adopt")
def adopt_cache(model, revision, all_models, endpoint, **kwargs):
    """Start tracking model revisions downloaded outside sparkrun.

    Writes a usage record for each untracked snapshot of MODEL (or of every
    model with --all), so list shows it and prune can manage it. Each file must
    match the model origin's inventory for that commit; partial downloads are
    fine, foreign or resized files refuse. Nothing is downloaded, moved or
    deleted. Defaults to a preview.

    Adopted revisions then age by sparkrun's use only: with automatic pruning
    on, one other tools still use becomes a cleanup candidate once the
    retention window passes. Use --pin for revisions that must stay.
    """
    if bool(model) == all_models:
        raise click.UsageError("Name a MODEL to adopt, or pass --all.")
    if all_models and revision:
        raise click.UsageError("--revision needs a MODEL.")
    request = ModelArtifactRequest(model, revision, endpoint=endpoint) if model else None
    return api.adopt_cache(request, all_models=all_models, **kwargs)


def _interactive() -> bool:
    return sys.stdin.isatty()


@model_cache.command("delete")
@click.argument("model")
@click.option("--revision", default=None, help="Delete only this commit (or ref, e.g. main); default deletes every cached revision.")
@click.option("--dry-run", is_flag=True, help="Show what would be deleted and stop.")
@click.option("--yes", "-y", "assume_yes", is_flag=True, help="Delete without asking for confirmation.")
@click.option("--force", is_flag=True, help="Also delete revisions with outstanding workload claims (likely in use).")
@click.pass_context
@_target_options
@json_option("Print the per-cache reports as JSON.")
def delete_cache(ctx, model, revision, dry_run, assume_yes, force, output_json, **targets):
    """Delete a model, or one revision of it, from the target hosts' caches.

    Unlike prune, pins, retention age and untracked status do not protect a
    revision. Outstanding workload claims do unless --force, and an active
    download always does. If any host would refuse, nothing is deleted on any
    host. Blobs are removed only when nothing else in the cache uses them.
    Saved manifests are kept, so `model-cache repair` can restore the model.
    """
    from sparkrun.orchestration.primitives import build_ssh_kwargs

    sctx = _get_context(ctx)
    try:
        request = ModelArtifactRequest(model, revision)
        host_list, cache_dir, banner, _ = _resolve_targets(sctx, **targets)
        delete = partial(
            api.delete_model,
            request,
            cache_dir=cache_dir,
            hosts=tuple(host_list),
            force=force,
            ssh_kwargs=build_ssh_kwargs(sctx.config),
        )
        reports = delete(dry_run=True)
        doomed = sum(1 for r in reports for e in r.entries if e.action == "would_delete")
        proceed = not dry_run and doomed and all(r.target_met for r in reports)
        if not output_json:
            click.echo(banner)
            click.echo(format_model_cache_reports(reports, action="delete"))
        if proceed and not assume_yes:
            if output_json or not _interactive():
                raise click.ClickException("Refusing to delete without confirmation; pass --yes")
            planned = sum(r.planned_bytes for r in reports)
            hosts = sum(len(r.location.hosts) if r.location else 1 for r in reports)
            where = "%d host(s)" % hosts if hosts == len(reports) else "%d cache(s) on %d host(s)" % (len(reports), hosts)
            click.confirm(
                "\nDelete %d revision(s) of %s from %s, freeing ~%s?" % (doomed, request.repo, where, format_bytes(planned)),
                abort=True,
            )
        if proceed:
            confirmed = {e.revision for r in reports for e in r.entries if e.action == "would_delete"}
            reports = delete(dry_run=False, allowed_revisions=confirmed)
            if not output_json:
                click.echo()
                click.echo(format_model_cache_reports(reports, action="delete"))
    except (ModelArtifactError, ValueError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc
    if output_json:
        click.echo(json.dumps([asdict(report) for report in reports], indent=2))
    if not doomed and all(r.target_met for r in reports):
        click.echo("%s%s is not cached on the target hosts." % (request.repo, "@" + revision if revision else ""), err=True)
        ctx.exit(1)
    if any(not r.target_met for r in reports):
        ctx.exit(1)
