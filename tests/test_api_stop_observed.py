"""Eviction must tear down observed ranks through the real stop path."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

import sparkrun.api as api
from sparkrun.api._run import _evict_superseded_deployments
from sparkrun.core.cluster_manager import ClusterDefinition
from sparkrun.core.cluster_status import ClusterStatus, ContainerDetail, HostOccupancy, RunningWorkload
from sparkrun.core.status_observation import ExecutorCoverage
from sparkrun.orchestration.executor import resolve_executor
from sparkrun.orchestration.ssh import RemoteResult

INTENT = "aaaaaaaaaaaa"
PRIOR = f"sparkrun_{INTENT}_111111111111"
NEW = f"sparkrun_{INTENT}_222222222222"
FOREIGN = "sparkrun_bbbbbbbbbbbb_333333333333"


@pytest.fixture
def observed_job(monkeypatch, v):
    def create(backend="docker", ranks=(1,), fail=False, metadata=True):
        cluster = ClusterDefinition(
            "test",
            ["h0", "h1", "h2", "h3"],
            user="alice",
            executor=backend,
            executor_config={"pid_dir": "/srv/native-jobs"} if backend == "local" else {},
        )
        sctx = api.default_sctx().for_cluster(cluster)
        executor = resolve_executor(cluster=cluster, config=sctx.config, v=v, rootless=False, auto_user=False)
        target = executor.resolve_target(dry_run=True)
        names = tuple(f"{PRIOR}_node_{rank}" for rank in ranks)
        foreign = FOREIGN + "_node_1"
        workload = RunningWorkload(
            cluster_id=PRIOR,
            intent_id=INTENT,
            containers=tuple(
                ContainerDetail(name, f"node_{rank}", "Up", "image", backend) for name, rank in zip(names, ranks, strict=True)
            ),
        )
        snapshot = ClusterStatus(
            hosts=(
                HostOccupancy("h0"),
                HostOccupancy(
                    "h1",
                    workloads=(
                        workload,
                        RunningWorkload(cluster_id=FOREIGN, containers=(ContainerDetail(foreign, "node_1", "Up", "image", backend),)),
                    ),
                ),
            ),
            executor=backend,
            coverage=(ExecutorCoverage(target, "host", frozenset(cluster.hosts), frozenset(cluster.hosts), "alice"),),
        )
        meta = {
            "hosts": list(cluster.hosts),
            "executor": backend,
            "executor_config": dict(target.config),
            "executor_destination_key": target.destination_key,
            "executor_user_scoped": target.user_scoped,
            "ssh_user": "alice",
        }
        monkeypatch.setattr("sparkrun.orchestration.job_metadata.load_job_metadata", lambda *a, **k: meta if metadata else None)
        forgotten = []
        monkeypatch.setattr("sparkrun.orchestration.job_metadata.remove_job_metadata", lambda cid, **k: forgotten.append(cid))
        monkeypatch.setattr("sparkrun.orchestration.executor.query_status_for_cluster", lambda *a, **k: snapshot)
        monkeypatch.setattr("sparkrun.api._status.status", lambda *a, **k: pytest.fail("must reuse eviction discovery"))
        alive = set(names) | {foreign}
        calls = []

        def cleanup(mapping, *, executor, ssh_kwargs):
            calls.append((mapping, executor, ssh_kwargs))
            results = {}
            for host, candidates in mapping.items():
                present = alive.intersection(candidates)
                if not fail:
                    alive.difference_update(candidates)
                left = alive.intersection(candidates)
                results[host] = RemoteResult(
                    host,
                    returncode=1 if left else 0,
                    stdout=f"sparkrun_removed={0 if left else len(present)}\n",
                    stderr="worker still running" if left else "",
                )
            return results

        monkeypatch.setattr("sparkrun.orchestration.primitives.cleanup_containers_by_host", cleanup)
        return SimpleNamespace(
            cluster=cluster, sctx=sctx, snapshot=snapshot, names=names, foreign=foreign, alive=alive, calls=calls, forgotten=forgotten
        )

    return create


def evict(job):
    return _evict_superseded_deployments(
        intent_id=INTENT,
        cluster_id_for_launch=NEW,
        candidate_hosts=job.cluster.hosts,
        target_hosts=["h1"],
        cluster_def=job.cluster,
        config=job.sctx.config,
        sctx=job.sctx,
        strict=True,
    )


@pytest.mark.parametrize("backend", ["docker", "local"])
@pytest.mark.parametrize("ranks", [(1,), (1, 3)])
@pytest.mark.parametrize("metadata", [True, False])
def test_eviction_removes_observed_survivors_and_only_then_forgets_job(observed_job, backend, ranks, metadata):
    job = observed_job(backend, ranks, metadata=metadata)
    evicted, _ = evict(job)
    assert evicted == [PRIOR]
    assert job.alive == {job.foreign}
    assert job.calls[0][0] == {"h1": list(job.names)}
    assert job.calls[0][1].executor_name == backend
    assert job.calls[0][2]["ssh_user"] == "alice"
    # Other recorded hosts were not torn down by this scoped stop.
    assert job.forgotten == ([] if metadata else [PRIOR])


def test_failed_survivor_teardown_preserves_metadata_and_aborts_strict_eviction(observed_job):
    job = observed_job(fail=True)
    with pytest.raises(RuntimeError, match="not confirmed on h1"):
        evict(job)
    assert set(job.names) <= job.alive
    assert job.calls[0][0] == {"h1": list(job.names)}
    assert job.forgotten == []


def test_stop_with_discovery_reports_actual_removed_count(observed_job):
    job = observed_job(ranks=(1, 3))
    result = api.stop(workload_id=PRIOR, hosts=["h1"], cluster=job.cluster, sctx=job.sctx, discovered=job.snapshot)
    assert result.containers_removed == 2
    assert result.hosts_failed == () and result.errors == ()


@pytest.mark.parametrize("failed", [False, True])
def test_stop_releases_cache_claims_only_after_confirmed_teardown(observed_job, monkeypatch, failed):
    from unittest.mock import Mock

    job = observed_job(fail=failed)
    from sparkrun.orchestration.job_metadata import load_job_metadata

    claims = [{"claim": "exact-generation", "host": "h1"}]
    load_job_metadata(PRIOR)["model_cache_claims"] = claims
    release = Mock()
    monkeypatch.setattr("sparkrun.models.usage.release_claims", release)
    api.stop(workload_id=PRIOR, hosts=["h1"], cluster=job.cluster, sctx=job.sctx, discovered=job.snapshot)
    if failed:
        release.assert_not_called()
        assert not job.forgotten
    else:
        assert release.call_args.args == (claims,)
        assert set(release.call_args.kwargs["hosts"]) == {"h1"}
        assert release.call_args.kwargs["ssh_kwargs"]["ssh_user"] == "alice"
        assert job.forgotten == []  # Recovery metadata still covers h0, h2 and h3.


@pytest.mark.parametrize("unreachable", [False, True])
def test_stop_tears_down_and_releases_claims_an_earlier_launch_left_elsewhere(observed_job, monkeypatch, unreachable):
    """A same-id relaunch placed on other hosts keeps the earlier launch's claim (h9)."""
    from unittest.mock import Mock

    job = observed_job()
    from sparkrun.orchestration.job_metadata import load_job_metadata

    meta = load_job_metadata(PRIOR)
    meta["hosts"] = ["h1"]
    meta["model_cache_claims"] = [{"claim": "latest", "host": "h1"}, {"claim": "earlier", "host": "h9"}]
    if unreachable:

        def cleanup(mapping, *, executor, ssh_kwargs):
            job.calls.append((mapping, executor, ssh_kwargs))
            return {
                host: RemoteResult(host, returncode=255 if host == "h9" else 0, stdout="sparkrun_removed=0\n", stderr="unreachable")
                for host in mapping
            }

        monkeypatch.setattr("sparkrun.orchestration.primitives.cleanup_containers_by_host", cleanup)
    release = Mock()
    monkeypatch.setattr("sparkrun.models.usage.release_claims", release)

    result = api.stop(workload_id=PRIOR, hosts=["h1"], cluster=job.cluster, sctx=job.sctx, discovered=job.snapshot)

    # The earlier host is torn down by candidate names, which discovery need not cover.
    assert "h9" in job.calls[-1][0] and PRIOR + "_solo" in job.calls[-1][0]["h9"]
    assert result.hosts_targeted == ("h1", "h9")
    assert result.success  # the workload's own hosts confirmed
    if unreachable:
        assert set(release.call_args.kwargs["hosts"]) == {"h1"}
        assert job.forgotten == []
        assert any("model-cache reconcile --all" in w and "h9" in w for w in result.warnings)
    else:
        assert set(release.call_args.kwargs["hosts"]) == {"h1", "h9"}
        assert job.forgotten == [PRIOR]
        assert result.warnings == ()


def test_scoped_stop_does_not_widen_to_the_latest_launchs_other_hosts(observed_job, monkeypatch):
    monkeypatch.setattr("sparkrun.models.usage.release_claims", lambda claims, **kw: None)
    job = observed_job()
    from sparkrun.orchestration.job_metadata import load_job_metadata

    load_job_metadata(PRIOR)["model_cache_claims"] = [{"claim": "c", "host": host} for host in job.cluster.hosts]
    result = api.stop(workload_id=PRIOR, hosts=["h1"], cluster=job.cluster, sctx=job.sctx, discovered=job.snapshot)
    assert [list(call[0]) for call in job.calls] == [["h1"]]
    assert result.hosts_targeted == ("h1",) and result.warnings == ()


@pytest.mark.parametrize("recorded_executor", ["docker", "local", None])
def test_same_id_relaunch_releases_earlier_claims_only_where_the_sweep_saw_it_gone(observed_job, monkeypatch, recorded_executor):
    """h0 was observed without the workload; it still runs on h1; h7 was never swept."""
    job = observed_job()
    from sparkrun.orchestration.job_metadata import load_job_metadata

    meta = load_job_metadata(PRIOR)
    meta["executor"] = recorded_executor
    claims = [{"claim": "earlier", "host": host} for host in ("h0", "h1", "h7")]
    meta["model_cache_claims"] = claims
    released, forgotten = [], []
    monkeypatch.setattr("sparkrun.models.usage.release_claims", lambda refs, **kw: released.append((refs, set(kw["hosts"]))))
    monkeypatch.setattr(
        "sparkrun.orchestration.job_metadata.forget_model_cache_claims", lambda cid, refs, **kw: forgotten.append((cid, refs))
    )

    evicted, _ = _evict_superseded_deployments(
        intent_id=INTENT,
        cluster_id_for_launch=PRIOR,
        candidate_hosts=job.cluster.hosts,
        target_hosts=["h1"],
        cluster_def=job.cluster,
        config=job.sctx.config,
        sctx=job.sctx,
    )

    assert evicted == [] and job.calls == []  # the runtime replaces its own containers by name
    if recorded_executor == "docker":
        assert released == [([claims[0]], {"h0"})]
        assert forgotten == [(PRIOR, [claims[0]])]
    else:
        # An executor the sweep did not query (or none recorded) proves nothing.
        assert released == [] and forgotten == []


@pytest.mark.parametrize("mismatch", ["executor", "destination", "user", "missing_names"])
def test_unusable_observation_never_forgets_job(observed_job, mismatch):
    job = observed_job("local")
    snapshot = job.snapshot
    if mismatch == "missing_names":
        host = snapshot.hosts[1]
        snapshot = replace(snapshot, hosts=(snapshot.hosts[0], replace(host, workloads=(replace(host.workloads[0], containers=()),))))
    elif mismatch == "executor":
        host = snapshot.hosts[1]
        workload = host.workloads[0]
        wrong = replace(workload.containers[0], executor="docker")
        snapshot = replace(snapshot, hosts=(snapshot.hosts[0], replace(host, workloads=(replace(workload, containers=(wrong,)),))))
    else:
        coverage = snapshot.coverage[0]
        if mismatch == "user":
            coverage = replace(coverage, ssh_user="bob")
        else:
            coverage = replace(
                coverage, target=replace(coverage.target, destination_key="/srv/other-jobs", config={"pid_dir": "/srv/other-jobs"})
            )
        snapshot = replace(snapshot, coverage=(coverage,))
    with pytest.raises(api.SparkrunError):
        api.stop(workload_id=PRIOR, hosts=["h1"], cluster=job.cluster, sctx=job.sctx, discovered=snapshot)
    assert job.calls == []
    assert job.forgotten == []
    assert set(job.names) <= job.alive


def test_missing_metadata_uses_observed_executor_not_cluster_default(observed_job):
    job = observed_job("local", metadata=False)
    job.cluster.executor = "docker"
    evict(job)
    assert job.calls[0][1].executor_name == "local"
    assert job.alive == {job.foreign}


def test_observed_host_mapping_does_not_copy_ranks_to_other_hosts(observed_job):
    job = observed_job(ranks=(1, 3))
    other = PRIOR + "_node_2"
    job.alive.add(other)
    snapshot = replace(
        job.snapshot,
        hosts=(
            job.snapshot.hosts[1],
            HostOccupancy(
                "h2",
                workloads=(
                    RunningWorkload(
                        cluster_id=PRIOR,
                        containers=(ContainerDetail(other, "node_2", "Up", "image", "docker"),),
                    ),
                ),
            ),
        ),
    )
    result = api.stop(workload_id=PRIOR, hosts=["h1", "h2"], cluster=job.cluster, sctx=job.sctx, discovered=snapshot)
    assert job.calls[0][0] == {"h1": list(job.names), "h2": [other]}
    assert result.containers_removed == 3
    assert job.alive == {job.foreign}


def test_nonstrict_failed_eviction_retains_live_metadata(observed_job):
    job = observed_job(fail=True)
    _, observation = _evict_superseded_deployments(
        intent_id=INTENT,
        cluster_id_for_launch=NEW,
        candidate_hosts=job.cluster.hosts,
        target_hosts=["h1"],
        cluster_def=job.cluster,
        config=job.sctx.config,
        sctx=job.sctx,
    )
    assert job.forgotten == []
    assert PRIOR in observation.cluster_ids
    assert set(job.names) <= job.alive
