"""``sparkrun.api.stop_all`` — discover and stop every sparkrun workload.

The discovery-driven counterpart to :func:`sparkrun.api.stop`: instead of
naming a workload, sweep a set of hosts for whatever sparkrun launched
and tear all of it down.

This is console-free like the rest of ``sparkrun.api`` — it returns a
:class:`~sparkrun.api._models.StopAllResult` describing what was found,
what was removed, and what failed; rendering and exit codes are the
caller's business.  (It previously lived inside the CLI's ``--all``
branch, which meant the GUI sidecar and any other library caller had no
way to do it and no way to inherit its fixes.)

Two failure modes are distinguished, and neither is ever silently
reported as success:

- **discovery errors** — a host that could not be queried.  It is *not*
  "nothing to stop"; it may be running containers we never saw.
- **teardown failures** — a host whose containers did not confirm gone.
  Job metadata for anything with a container there is deliberately
  retained, because the workload is still live and the metadata is how
  it is found again.
"""

from __future__ import annotations

from sparkrun.core.coordination import operation_boundary

import logging
from typing import TYPE_CHECKING

from sparkrun.api._errors import SparkrunError
from sparkrun.api._models import StopAllResult

if TYPE_CHECKING:
    from sparkrun.core.cluster_manager import ClusterDefinition, ClusterStatusResult
    from sparkrun.core.context import SparkrunContext

logger = logging.getLogger(__name__)


@operation_boundary(mutation=True)
def stop_all(
    hosts: list[str] | tuple[str, ...] | None = None,
    *,
    cluster: "str | ClusterDefinition | None" = None,
    cache_dir: str | None = None,
    ssh_kwargs: dict | None = None,
    dry_run: bool = False,
    discovered: "ClusterStatusResult | None" = None,
    sctx: "SparkrunContext | None" = None,
) -> StopAllResult:
    """Discover and stop every sparkrun workload on *hosts*.

    Args:
        hosts: Hosts to sweep.  ``None`` takes *cluster*'s hosts.  One of
            *hosts* or *cluster* is required: a teardown never picks its own
            target, so pass the default cluster's name to sweep it.
        cluster: Optional cluster name/definition for executor + transport
            resolution (forwarded to :func:`sparkrun.api.status_report`).
        cache_dir: Cache directory holding job metadata.
        ssh_kwargs: Per-key overrides for cluster-scoped SSH configuration.
            With a discovery snapshot, authentication settings may be overridden
            but an explicit different SSH user requires fresh discovery.
        dry_run: Log planned teardown; result counts are previews and
            ``result.dry_run`` is true. Discovery still
            runs for real — there is nothing to preview otherwise.
        discovered: A snapshot from :func:`sparkrun.api.status_report` to
            act on instead of re-querying.  Lets a caller that already
            displayed the discovery (the CLI) avoid a second sweep.
        sctx: Optional shared :class:`SparkrunContext`.

    Returns:
        :class:`~sparkrun.api._models.StopAllResult`.
    """
    from sparkrun.api._status import status_report
    from sparkrun.orchestration.primitives import cleanup_containers_by_host, merge_teardown_results
    from sparkrun.orchestration.teardown import parse_teardown_removed

    from sparkrun.api._resolve import resolve_cluster, scope_operation

    explicit_ssh = ssh_kwargs
    if hosts is None and cluster is None:
        raise SparkrunError("stop_all requires hosts or a cluster; a teardown never defaults to the default cluster")
    cluster_def = resolve_cluster(cluster, hosts, sctx=sctx)
    host_list = list(cluster_def.hosts) if hosts is None else list(hosts)
    sctx, ssh_kwargs = scope_operation(cluster_def, sctx=sctx, ssh_kwargs=ssh_kwargs, prepare=False)
    from sparkrun.core.config import resolve_configured_cache_dir

    cache_dir = str(resolve_configured_cache_dir(cache_dir, config=sctx.config))
    result = discovered
    if result is None:
        result = status_report(host_list, cluster=cluster, ssh_kwargs=explicit_ssh, cache_dir=cache_dir, sctx=sctx)

    if result.total_containers == 0:
        return _outcome(
            result,
            dry_run=dry_run,
            jobs_stopped=0,
            containers_removed=0,
        )

    executor_names = set(result.container_executors.values())
    targets = {c.target.executor: c.target for c in result.coverage}
    executors = {name: _resolve_teardown_executor(name, cluster, host_list, sctx, target=targets.get(name)) for name in executor_names}
    native = any(group.meta.get("native_resource") is not None for group in result.groups.values()) or any(
        entry.meta.get("native_resource") is not None for entry in result.solo_entries
    )
    control_plane = any(
        isinstance(getattr(executor, "status_scope", None), str) and executor.status_scope != "host"
        for executor in executors.values()
        if executor is not None
    )
    if native or control_plane:
        # Native resources cannot be reconstructed as cluster_id + role. Use
        # the same portable-ID teardown as individual stops and replacement.
        return _stop_discovered_workloads(result, cluster=cluster, cache_dir=cache_dir, dry_run=dry_run, sctx=sctx)

    host_containers = _containers_by_host(result)

    # Discovery is cross-executor (docker + local share the "host" scope), so
    # teardown must be too: each executor is dispatched with only the
    # workloads *it* reported, then the per-host verdicts are recombined.
    # Sending the whole set to one executor is what let a `local` workload
    # survive a "successful" stop --all.
    grouped_by_executor = _group_by_executor(host_containers, result)
    connections = {name: _discovered_transport(result, name, ssh_kwargs, explicit_ssh) for name in grouped_by_executor}
    results: dict = {}
    for executor_name, grouped in grouped_by_executor.items():
        results = merge_teardown_results(
            results,
            cleanup_containers_by_host(
                grouped,
                ssh_kwargs=connections[executor_name],
                dry_run=dry_run,
                executor=executors.get(executor_name),
            ),
        )

    failed_hosts: dict[str, str] = {}
    for host in host_containers:
        r = results.get(host)
        if r is None:
            failed_hosts[host] = "teardown did not run"
        elif not r.success:
            failed_hosts[host] = (r.stderr or r.stdout).strip() or ("exit code %d" % r.returncode)

    if dry_run:
        # Nothing was executed, so nothing failed and nothing was removed;
        # report the discovered shape as what *would* be stopped.
        return _outcome(
            result,
            dry_run=dry_run,
            jobs_stopped=len(result.groups) + len(result.solo_entries),
            containers_removed=result.total_containers,
            hosts_stopped=tuple(host_containers),
        )

    containers_removed = sum(parse_teardown_removed(r.stdout) for r in results.values())

    # Drop job metadata only for jobs with no container left behind.
    jobs_stopped = 0
    for cid, group in result.groups.items():
        if any(member_host in failed_hosts for member_host, _role, _status, _image in group.members):
            continue
        _forget_job(cid, cache_dir, hosts=[m[0] for m in group.members], ssh_kwargs=ssh_kwargs)
        jobs_stopped += 1
    for entry in result.solo_entries:
        if entry.host in failed_hosts:
            continue
        solo_cid = entry.name.removesuffix("_solo") if entry.name.endswith("_solo") else entry.name
        _forget_job(solo_cid, cache_dir, hosts=[entry.host], ssh_kwargs=ssh_kwargs)
        jobs_stopped += 1

    return _outcome(
        result,
        dry_run=dry_run,
        jobs_stopped=jobs_stopped,
        containers_removed=containers_removed,
        hosts_stopped=tuple(h for h in host_containers if h not in failed_hosts),
        hosts_failed=failed_hosts,
    )


def _outcome(discovered, *, dry_run, **counts):
    """Discovery failures survive every backend and preview return path."""
    return StopAllResult(discovered=discovered, discovery_errors=dict(discovered.errors), dry_run=dry_run, **counts)


def _discovered_transport(discovered, executor_name, defaults, explicit):
    from sparkrun.api._errors import SparkrunError
    from sparkrun.utils.data import thaw

    coverage = [c for c in discovered.coverage if c.target.executor == executor_name]
    if not coverage:
        return defaults
    if len({(c.target.destination_key, c.ssh_user) for c in coverage}) != 1:
        raise SparkrunError("Bulk teardown requires one observed destination per executor")
    observed = coverage[0]
    if explicit is not None and "ssh_user" in explicit and (explicit["ssh_user"] or None) != observed.ssh_user:
        raise SparkrunError("SSH user differs from the discovery snapshot; query the intended destination first")
    connection = thaw(observed.ssh_kwargs) if observed.ssh_kwargs is not None else {"ssh_user": observed.ssh_user}
    return {**defaults, **connection, **(explicit or {})}


def _stop_discovered_workloads(discovered, *, cluster, cache_dir, dry_run, sctx):
    from sparkrun.api._stop import stop

    jobs = {cid: list(dict.fromkeys(member[0] for member in group.members)) for cid, group in discovered.groups.items()}
    for entry in discovered.solo_entries:
        hosts = jobs.setdefault(entry.cluster_id, [])
        if entry.host not in hosts:
            hosts.append(entry.host)
    all_hosts = tuple(dict.fromkeys(host for hosts in jobs.values() for host in hosts))
    if dry_run:
        return _outcome(
            discovered,
            dry_run=dry_run,
            jobs_stopped=len(jobs),
            containers_removed=discovered.total_containers,
            hosts_stopped=all_hosts,
        )
    # A supplied discovery result owns its provider target even if defaults
    # changed or local job metadata disappeared after the query.
    native_targets = [c.target for c in discovered.coverage if c.status_scope != "host"]
    if len(native_targets) == 1:
        from dataclasses import replace
        from sparkrun.api._resolve import resolve_cluster

        target = native_targets[0]
        cluster = replace(resolve_cluster(cluster, all_hosts, sctx=sctx), executor=target.executor, executor_config=target.overrides)
    elif len(native_targets) > 1:
        raise ValueError("Native bulk teardown requires a single discovery destination")
    jobs_stopped = removed = 0
    failures = {}
    for cid, hosts in jobs.items():
        try:
            # An unscoped legacy snapshot cannot supply a provider destination.
            # With no explicit cluster, require stop-by-ID's authoritative metadata.
            outcome = stop(workload_id=cid, hosts=hosts if cluster is not None else None, cluster=cluster, cache_dir=cache_dir, sctx=sctx)
        except Exception as exc:
            failures.update({host: str(exc) for host in hosts})
            continue
        removed += outcome.containers_removed
        if outcome.success:
            jobs_stopped += 1
        else:
            detail = "; ".join(outcome.errors) or "teardown did not confirm"
            failures.update({host: detail for host in outcome.hosts_failed or hosts})
    return _outcome(
        discovered,
        dry_run=dry_run,
        jobs_stopped=jobs_stopped,
        containers_removed=removed,
        hosts_stopped=tuple(h for h in all_hosts if h not in failures),
        hosts_failed=failures,
    )


def _containers_by_host(result: "ClusterStatusResult") -> dict[str, list[str]]:
    """Map host → container names to tear down, from a discovery snapshot."""
    host_containers: dict[str, list[str]] = {}
    for cid, group in result.groups.items():
        for host, role, _status, _image in group.members:
            host_containers.setdefault(host, []).append("%s_%s" % (cid, role))
    for entry in result.solo_entries:
        host_containers.setdefault(entry.host, []).append(entry.name)
    return host_containers


def _group_by_executor(
    host_containers: dict[str, list[str]],
    result: "ClusterStatusResult",
) -> dict[str, dict[str, list[str]]]:
    """Split a host→containers map into one such map per reporting executor.

    Uses :attr:`ClusterStatusResult.container_executors`, stamped during
    discovery.  Containers with no attribution (a snapshot from an executor
    that predates the field, or a hand-built one in a test) group under ``""``
    and are torn down with the cluster's default executor — the historical
    behaviour, and the safe reading of "we don't know".
    """
    grouped: dict[str, dict[str, list[str]]] = {}
    for host, names in host_containers.items():
        for name in names:
            executor_name = result.container_executors.get((host, name), "")
            grouped.setdefault(executor_name, {}).setdefault(host, []).append(name)
    return grouped


def _resolve_teardown_executor(
    executor_name: str,
    cluster: "str | ClusterDefinition | None",
    hosts: list[str],
    sctx: "SparkrunContext | None",
    *,
    target=None,
):
    """Build the :class:`Executor` that tears down *executor_name*'s workloads.

    Resolved through the same chain the status sweep used
    (``resolve_executor`` with the name as a CLI-level override), so the
    executor's config matches the one that reported the workload — that
    matters for substrates whose teardown depends on it, e.g. the ``local``
    executor's ``pid_dir``, which is where the pidfile it must signal lives.

    Returns ``None`` — meaning "let the primitive use its default" — for an
    unattributed group or when resolution fails.  A teardown that can't
    identify its substrate should still attempt the historical one rather than
    skip the host entirely.
    """
    from sparkrun.orchestration.executor import resolve_executor

    if not executor_name:
        return None
    try:
        from sparkrun.api._resolve import resolve_cluster

        cluster_def = resolve_cluster(cluster, hosts, sctx=sctx)
        return resolve_executor(
            cluster=cluster_def,
            cli_overrides=target.overrides if target is not None else {"executor": executor_name},
            rootless=False,
            auto_user=False,
            config=sctx.config if sctx is not None else None,
            v=sctx.variables if sctx is not None else None,
        )
    except Exception:
        logger.warning(
            "Could not resolve executor %r for teardown; falling back to the default executor",
            executor_name,
            exc_info=True,
        )
        return None


def _forget_job(cluster_id: str, cache_dir: str | None, *, hosts=None, ssh_kwargs=None) -> None:
    """Remove job metadata, tolerating an already-absent record."""
    from sparkrun.orchestration.workload_cleanup import finalize_workload

    try:
        if not finalize_workload(cluster_id, hosts=hosts or (), cache_dir=cache_dir, ssh_kwargs=ssh_kwargs):
            logger.warning(
                "Kept job metadata for %s: model cache claims remain on hosts this stop did not confirm; "
                "stop it by id or run model-cache reconcile --all",
                cluster_id,
            )
    except Exception:
        logger.warning(
            "Could not release model cache claims or remove job metadata for %s; retry stop to reconcile", cluster_id, exc_info=True
        )
