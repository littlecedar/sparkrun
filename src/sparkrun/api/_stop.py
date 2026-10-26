"""``sparkrun.api.stop`` — stop a running sparkrun workload.

Two modes:

- **By cluster_id**: provide the literal ``cluster_id`` (as returned
  by :func:`sparkrun.api.run`); the API loads the job metadata, picks
  the executor that originally launched it, and stops its native controller
  resource or tears down observed container names on each host. Without
  a discovery snapshot, it falls back to candidate names.
- **By recipe+hosts+overrides**: derive the same ``cluster_id`` the
  launcher would have produced and dispatch identically.  Useful for
  ``sparkrun stop <recipe>`` semantics.

Returns a :class:`StopResult` summarizing how many containers were
removed and any per-host errors.
"""

from __future__ import annotations

from sparkrun.core.coordination import operation_boundary

import logging
from typing import TYPE_CHECKING

from sparkrun.api._errors import SparkrunError
from sparkrun.api._models import StopResult

if TYPE_CHECKING:
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.cluster_status import ClusterStatus
    from sparkrun.core.context import SparkrunContext
    from sparkrun.core.recipe import Recipe

logger = logging.getLogger(__name__)


@operation_boundary(mutation=True)
def stop(
    *,
    workload_id: str | None = None,
    recipe: "str | Recipe | None" = None,
    hosts: list[str] | tuple[str, ...] | None = None,
    overrides: dict | None = None,
    cluster: "str | ClusterDefinition | None" = None,
    cache_dir: str | None = None,
    discovered: "ClusterStatus | None" = None,
    dry_run: bool = False,
    sctx: "SparkrunContext | None" = None,
) -> StopResult:
    """Stop a running sparkrun workload.

    Either ``workload_id`` *or* (``recipe`` + a host source) is required.
    When both are provided, ``workload_id`` wins.

    Args:
        workload_id: The workload's id (``RunResult.workload_id``).
        dry_run: Resolve the target, hosts and executor exactly as a real stop
            would, then return without tearing anything down
            (``result.dry_run`` is true, ``containers_removed`` is 0).
            Discovery still contacts the hosts.
        discovered: Optional status snapshot for exact per-host container names.
            Reuses discovery during eviction instead of inferring ranks from
            the surviving host count. The snapshot must cover the requested
            hosts and match the resolved executor/destination; missing container
            detail is an error, not permission to fall back to guessed names.
        cache_dir: Explicit metadata root, otherwise the supplied or current
            application's configured cache. Resolved before job lookup.
        sctx: Optional shared :class:`SparkrunContext` for chained
            api calls (registry/cluster manager + config sharing).
    """
    from sparkrun.api._resolve import resolve_workload_target, scope_operation
    from sparkrun.orchestration.executor import resolve_executor

    from sparkrun.core.config import resolve_configured_cache_dir

    cache_dir = str(resolve_configured_cache_dir(cache_dir, config=sctx.config if sctx is not None else None))

    # Recipe form: status-driven discovery of the intent over the user's host
    # scope (a load-aware scheduler may have placed it on any of them).  Both
    # forms then reach the job through its own recorded connection (#277).
    target_spec = resolve_workload_target(
        workload_id=workload_id,
        recipe=recipe,
        hosts=hosts,
        overrides=overrides,
        cluster=cluster,
        cache_dir=cache_dir,
        operation="api.stop",
        sctx=sctx,
    )
    cluster_id, meta, target_hosts, cluster_def = target_spec[:4]

    # Provider refresh may update connection/executor settings. Resolve only
    # after that refresh; SSH transports are a no-op here.
    sctx, _ = scope_operation(cluster_def, sctx=sctx, prepare=not target_spec.transport_prepared)

    # Resolve the executor — prefer recipe-encoded selection from metadata
    # so we use the same executor that launched the workload.
    from sparkrun.core._executor_destination import metadata_executor_overrides

    cli_overrides = metadata_executor_overrides(meta)

    if not (cli_overrides or {}).get("executor"):
        # No metadata (or none naming an executor) — ask the cluster what is
        # actually running instead of defaulting.  Without this a job whose
        # metadata is gone gets the *default* executor, and a teardown aimed
        # at the wrong substrate exits 0 having done nothing: docker
        # truthfully reports no such container, so the workload survives and
        # its record is dropped as a confirmed stop.  Metadata does go
        # missing — an interrupted launch, a manually cleared cache, or the
        # very bug this guards against.
        discovered_exec = _discover_executor_name(
            cluster_id,
            target_hosts,
            cluster_def=cluster_def,
            sctx=sctx,
            discovered=discovered,
        )
        if discovered_exec:
            cli_overrides = dict(cli_overrides or {})
            cli_overrides["executor"] = discovered_exec

    executor = resolve_executor(
        cluster=cluster_def,
        cli_overrides=cli_overrides,
        rootless=False,
        auto_user=False,
        v=sctx.variables if sctx is not None else None,
        config=sctx.config if sctx is not None else None,
    )

    from sparkrun.api._resolve import bind_job_cluster

    target = executor.resolve_target(dry_run=True)
    cluster_def = bind_job_cluster(cluster_def, target, meta)
    sctx = sctx.for_cluster(cluster_def)
    if discovered is not None and discovered.coverage:
        covered = {
            host for coverage in discovered.coverage if coverage.matches(target, ssh_user=sctx.config.ssh_user) for host in coverage.hosts
        }
        if not set(target_hosts) <= covered:
            raise SparkrunError(
                "Stop discovery does not cover the requested executor destination and hosts; refresh status before teardown."
            )

    # A same-id relaunch keeps the earlier launch's model cache claims, and the
    # scheduler may have placed it elsewhere, so the record can claim hosts the
    # latest launch never touched.  Tearing the id down there too is what lets
    # those claims be released; on a host where nothing runs it is a no-op.
    # Hosts of the latest launch stay the caller's to scope (eviction stops a
    # subset on purpose), so only hosts nothing but an earlier launch names.
    claim_hosts = sorted(
        {ref["host"] for ref in (meta or {}).get("model_cache_claims") or () if isinstance(ref, dict) and isinstance(ref.get("host"), str)}
        - set((meta or {}).get("hosts") or ())
        - set(target_hosts)
    )

    if dry_run:
        logger.info("Dry run: would stop %s on %s", cluster_id, ", ".join([*target_hosts, *claim_hosts]))
        return StopResult(workload_id=cluster_id, hosts_targeted=(*target_hosts, *claim_hosts), containers_removed=0, dry_run=True)

    try:
        native_removed = executor.stop_workload(cluster_id, metadata=meta)
        if native_removed is not None and (type(native_removed) is not int or native_removed < 0):
            raise TypeError("Executor.stop_workload must return a nonnegative integer or None")
    except Exception as exc:
        logger.warning("Keeping job metadata for %s — native teardown failed: %s", cluster_id, exc)
        return StopResult(
            workload_id=cluster_id,
            hosts_targeted=tuple(target_hosts),
            containers_removed=0,
            errors=(str(exc),),
            hosts_failed=tuple(target_hosts),
        )
    warnings: list[str] = []
    if native_removed is None:
        removed_count, hosts_failed, errors = _stop_containers(
            cluster_id,
            target_hosts,
            cluster_def,
            executor,
            sctx,
            discovered=discovered,
        )
        claim_confirmed: list[str] = []
        if claim_hosts:
            # Discovery is not required here (and may not cover these hosts):
            # candidate names are removed and verified gone.  An unconfirmed
            # host only keeps its claims, so it warns rather than failing a
            # stop the workload's own hosts confirmed.
            extra_removed, extra_failed, extra_errors = _stop_containers(
                cluster_id,
                claim_hosts,
                cluster_def,
                executor,
                sctx,
                node_count=len(target_hosts) + len(claim_hosts),
            )
            removed_count += extra_removed
            claim_confirmed = [host for host in claim_hosts if host not in extra_failed]
            warnings.extend(extra_errors)
    else:
        removed_count, hosts_failed, errors = native_removed, (), []
        claim_confirmed = list(claim_hosts)

    # Cleanup persistent metadata only once teardown is confirmed
    # everywhere: while a container survives, the metadata still describes
    # a live workload and is what ``stop``/``status`` use to find it again.
    if hosts_failed:
        logger.warning(
            "Keeping job metadata for %s — teardown did not confirm on: %s",
            cluster_id,
            ", ".join(hosts_failed),
        )
    else:
        try:
            from sparkrun.orchestration.workload_cleanup import finalize_workload
            from sparkrun.orchestration.primitives import build_ssh_kwargs

            # False also means "kept for hosts this stop was scoped away
            # from", which is the caller's choice and needs no warning.
            finalize_workload(
                cluster_id, hosts=[*target_hosts, *claim_confirmed], cache_dir=cache_dir, ssh_kwargs=build_ssh_kwargs(sctx.config)
            )
            unreleased = sorted(set(claim_hosts) - set(claim_confirmed))
        except Exception:
            logger.warning(
                "Could not release model cache claims or remove job metadata for %s; retry stop to reconcile", cluster_id, exc_info=True
            )
            unreleased = [*target_hosts, *claim_hosts]
        if unreleased:
            from sparkrun.core.application_profile import render_identity_text

            # The workload is stopped, but a claim it still holds protects
            # its model revision from pruning until something releases it.
            warnings.append(
                render_identity_text(
                    "Model cache claims of %s may remain on %s and its job record was kept; retry stop, or run "
                    "'{app_command} model-cache reconcile --all' once those hosts are reachable." % (cluster_id, ", ".join(unreleased))
                )
            )

    return StopResult(
        workload_id=cluster_id,
        hosts_targeted=(*target_hosts, *claim_hosts),
        containers_removed=removed_count,
        errors=tuple(errors),
        hosts_failed=hosts_failed,
        warnings=tuple(warnings),
    )


def _stop_containers(cluster_id, target_hosts, cluster_def, executor, sctx, *, discovered=None, node_count=None):
    from sparkrun.api._resolve import scope_operation
    from sparkrun.orchestration.teardown import parse_teardown_removed

    if discovered is None:
        container_names = executor.enumerate_containers(cluster_id, node_count or len(target_hosts))
        host_containers = {host: list(container_names) for host in target_hosts}
    else:
        host_containers = _observed_container_names(cluster_id, target_hosts, executor.executor_name, discovered)

    # ``cleanup_containers_by_host`` is the shared teardown primitive: it
    # dispatches local-vs-SSH per host, verifies the workloads are actually
    # gone, and reports what it removed per host.  The executor resolved
    # above is threaded in so the teardown speaks the substrate that
    # launched this job — without it every stop emitted ``docker rm -f``,
    # which a ``local`` executor's native process truthfully answers "no
    # such container" to while continuing to serve.
    from sparkrun.orchestration.primitives import cleanup_containers_by_host

    sctx, ssh_kwargs = scope_operation(cluster_def, sctx=sctx, prepare=False)

    errors: list[str] = []
    try:
        results = cleanup_containers_by_host(
            host_containers,
            ssh_kwargs=ssh_kwargs,
            executor=executor,
        )
    except Exception as e:  # pragma: no cover - defensive; the primitive absorbs per-host failures
        errors.append(str(e))
        results = {}

    hosts_failed = tuple(host for host in target_hosts if not (results.get(host) and results[host].success))
    removed_count = sum(parse_teardown_removed(r.stdout) for r in results.values())
    for host in hosts_failed:
        r = results.get(host)
        detail = ((r.stderr or r.stdout).strip() if r else "") or "teardown did not confirm"
        errors.append("%s: %s" % (host, detail))

    return removed_count, hosts_failed, errors


def _observed_container_names(cluster_id, hosts, executor_name, discovered):
    """Use actual ranks on each host; never turn incomplete discovery into success."""
    names: dict[str, list[str]] = {host: [] for host in hosts}
    for entry in discovered.hosts:
        if entry.host not in names:
            continue
        if entry.host in discovered.errors:
            raise SparkrunError("Stop discovery failed on %s; refresh status before teardown." % entry.host)
        for workload in entry.workloads:
            if workload.cluster_id != cluster_id:
                continue
            if not workload.containers:
                raise SparkrunError("Stop discovery lacks container names for %s on %s." % (cluster_id, entry.host))
            for container in workload.containers:
                owner = container.executor or discovered.executor
                if owner and owner != executor_name:
                    raise SparkrunError("Stop discovery reports %s via another executor; use fresh stop_all discovery." % cluster_id)
                if not container.name:
                    raise SparkrunError("Stop discovery contains an unnamed container for %s." % cluster_id)
                if container.name not in names[entry.host]:
                    names[entry.host].append(container.name)
    missing = [host for host, containers in names.items() if not containers]
    if missing:
        raise SparkrunError("Stop discovery lacks container names for %s on: %s." % (cluster_id, ", ".join(missing)))
    return names


def _discover_executor_name(
    cluster_id: str,
    hosts: list[str],
    *,
    cluster_def,
    sctx: "SparkrunContext | None",
    discovered: "ClusterStatus | None" = None,
) -> str | None:
    """Return the executor currently reporting *cluster_id*, or ``None``.

    The live peer of the metadata lookup: ``api.status`` sweeps every executor
    on the cluster's substrate and stamps each container with the one that
    saw it, so a running workload can always name its own teardown mechanism
    even when nothing was written down.

    Returns ``None`` — "fall back to the resolution chain" — when the sweep
    fails or finds nothing.  A workload that isn't running needs no
    substrate-specific teardown, and a failed sweep must not block a stop.
    """
    snapshot = discovered
    if snapshot is None:
        try:
            from sparkrun.api._status import status

            snapshot = status(list(hosts), cluster=cluster_def, sctx=sctx)
        except Exception:
            logger.debug("Executor discovery sweep failed for %s", cluster_id, exc_info=True)
            return None

    names = {
        container.executor
        for entry in snapshot.hosts
        if entry.host in hosts
        for workload in entry.workloads
        if workload.cluster_id == cluster_id
        for container in workload.containers
        if container.executor
    }
    if len(names) > 1:
        raise SparkrunError("Workload %s spans multiple executors; use fresh stop_all discovery." % cluster_id)
    return next(iter(names), None)


__all__ = ["stop"]
