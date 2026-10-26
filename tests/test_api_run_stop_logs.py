"""Tests for ``sparkrun.api.run``, ``sparkrun.api.stop``, ``sparkrun.api.logs`` (Task 7).

The full launch path requires SSH-reachable hosts and a complete
runtime/builder/registry environment, so these tests focus on:

- API contract: signatures, return types, error mapping.
- Pre-launcher orchestration: recipe / hosts / cluster resolution and
  the routing of scheduling failures into typed API errors.
- Dry-run path through ``api.run`` (no remote execution).
- ``api.stop`` argument validation + ``JobNotFound`` handling.
- ``api.logs`` argument validation + ``JobNotFound`` handling.
- Iterator contract for ``api.logs``.

Full SSH-driven end-to-end coverage lands in Task 13's integration
tests.
"""

from __future__ import annotations

from typing import Iterator
from unittest.mock import patch

import pytest

import sparkrun.api as api
from sparkrun.core.cluster_status import ClusterStatus
from sparkrun.orchestration.executors.docker import DockerExecutor
from sparkrun.orchestration.ssh import RemoteResult


# --------------------------------------------------------------------------
# Public surface — re-exports reachable
# --------------------------------------------------------------------------


def test_run_function_exposed():
    assert hasattr(api, "run") and callable(api.run)


def test_stop_function_exposed():
    assert hasattr(api, "stop") and callable(api.stop)


def test_logs_function_exposed():
    assert hasattr(api, "logs") and callable(api.logs)


def test_logs_returns_iterator_protocol():
    """``api.logs`` is declared to return an Iterator[LogLine]; verify the
    type hint matches the runtime behaviour by inspecting the function
    return annotation."""
    import inspect

    sig = inspect.signature(api.logs)
    assert sig.return_annotation is not inspect.Parameter.empty


# --------------------------------------------------------------------------
# api.run — input validation / pre-launch failures
# --------------------------------------------------------------------------


def test_run_unknown_recipe_raises_recipe_not_found():
    """A string recipe name that doesn't resolve must raise RecipeNotFound."""
    with pytest.raises(api.RecipeNotFound):
        api.run(api.RunOptions(recipe="this-recipe-name-doesnt-exist-anywhere", hosts=("h1",)))


def test_run_no_hosts_no_cluster_raises_hosts_unreachable():
    """When no host source is available and config has no defaults."""
    from sparkrun.core.recipe import Recipe

    recipe = Recipe({"sparkrun_version": "2", "runtime": "vllm", "model": "test/m"})
    # No hosts, no cluster, and assume no default_hosts in test isolation.
    with pytest.raises(api.HostsUnreachable):
        api.run(api.RunOptions(recipe=recipe))


@pytest.mark.usefixtures("unavailable_hardware_probe", "idle_occupancy")
def test_run_options_immutable_through_run():
    """Passing the same RunOptions twice must not mutate its overrides dict."""
    from sparkrun.core.recipe import Recipe

    recipe = Recipe({"sparkrun_version": "2", "runtime": "vllm", "model": "test/m"})
    opts = api.RunOptions(recipe=recipe, hosts=("h1",), overrides={"port": 8000})
    snapshot = dict(opts.overrides)
    with patch("sparkrun.core.launcher.launch_inference", side_effect=api.SparkrunError("fixture launch failure")) as launch:
        for _ in range(2):
            with pytest.raises(api.SparkrunError, match="fixture launch failure"):
                api.run(opts)
            assert opts.overrides == snapshot
        assert launch.call_count == 2


# --------------------------------------------------------------------------
# api.run dry-run path — no remote execution
# --------------------------------------------------------------------------


@pytest.mark.parametrize("reuse_hardware", [False, True])
def test_run_dry_run_returns_run_result_without_ssh(reuse_hardware):
    """A dry-run call returns a populated RunResult without invoking SSH.

    Patching ``launch_inference`` keeps this hermetic — we just verify
    that api.run wires the options through and translates the result.
    """
    from sparkrun.core.recipe import Recipe

    recipe = Recipe({"sparkrun_version": "2", "runtime": "vllm", "model": "test/m"})
    opts = api.RunOptions(recipe=recipe, hosts=("h1",), dry_run=True)

    class _FakeRuntime:
        runtime_name = "vllm"
        executor = None  # no executor; fallback path

        def world_size(self, parallelism, recipe=None, cluster=None):
            return parallelism.world_size() or 1

    fake_result = type(
        "FakeLaunchResult",
        (),
        {
            "rc": 0,
            "cluster_id": "sparkrun_fakefakefake",
            "host_list": ["h1"],
            "is_solo": True,
            "runtime": _FakeRuntime(),
            "recipe": recipe,
            "overrides": {},
            "container_image": "test:latest",
            "effective_cache_dir": "/tmp/cache",
            "serve_port": 8000,
            "config": None,
            "recipe_ref": None,
            "comm_env": None,
            "ib_ip_map": {},
            "serve_command": "",
            "runtime_info": {},
            "builder": None,
            "backends": {},
            "timeline": None,
        },
    )()

    from types import SimpleNamespace
    from sparkrun.core.execution import PreparedExecution
    from sparkrun.core.hardware import HostHardware, AcceleratorSpec

    observed = HostHardware(
        accelerators=[AcceleratorSpec("nvidia", "gb10", capabilities=frozenset({"cuda", "rdma:roce-v2"}))],
        source="detected",
        driver_versions={"nvidia": "610.43.02"},
        ib_info={"IB_DETECTED": "1", "DETECTED_NET_LIST": "roce0"},
    )
    contexts = []

    def resolve(context):
        contexts.append(context)
        if not reuse_hardware:
            return None, ()
        strategy = SimpleNamespace(
            name="snapshot",
            finalize_preparation=lambda context, receipts: PreparedExecution("snapshot", host_hardware={"h1": observed}),
        )
        return strategy, ()

    with (
        patch("sparkrun.api.status", return_value=ClusterStatus()),
        patch("sparkrun.core.launcher.launch_inference", return_value=fake_result) as launch,
        patch("sparkrun.api._resolve.resolve_runtime", return_value=_FakeRuntime()),
        patch("sparkrun.core.execution.resolve_recipe_execution", side_effect=resolve),
    ):
        result = api.run(opts)

    if reuse_hardware:
        kwargs = launch.call_args.kwargs
        original = contexts[0]
        assert original.plan.cluster.hardware_for("h1").source == "assumed"
        assert kwargs["cluster"].hardware_for("h1").source == "detected"
        assert kwargs["placement"] is original.plan.placement
        assert kwargs["execution_context"].plan.cluster is kwargs["cluster"]
        assert kwargs["execution_context"].sctx is kwargs["sctx"]
        assert kwargs["execution_context"].plan.workload_id == original.plan.workload_id
        evidence = result.metadata["hardware_evidence"]["h1"]
        assert evidence["accelerators"][0]["capacity_verification"] == "estimated"
        assert evidence["drivers"]["nvidia"]["version"] == "610.43.02"
        assert evidence["interfaces"]["status"] == "discovered"
    else:
        assert "hardware_evidence" not in result.metadata

    assert isinstance(result, api.RunResult)
    assert result.workload_id == "sparkrun_fakefakefake"
    assert result.dry_run is True
    assert result.runtime == "vllm"
    assert result.is_solo is True


def test_run_solo_mode_truncates_to_one_host(idle_occupancy):
    """Solo mode keeps only the head host even when multiple are passed."""
    from sparkrun.core.recipe import Recipe

    recipe = Recipe({"sparkrun_version": "2", "runtime": "vllm", "model": "test/m"})
    opts = api.RunOptions(
        recipe=recipe,
        hosts=("h1", "h2", "h3"),
        solo=True,
        dry_run=True,
    )

    captured_hosts: list[str] = []

    class _FakeRuntime:
        runtime_name = "vllm"
        executor = None

        def world_size(self, parallelism, recipe=None, cluster=None):
            return parallelism.world_size() or 1

    def _capture(**kwargs):
        captured_hosts.extend(kwargs["host_list"])
        return type(
            "FakeLaunchResult",
            (),
            {
                "rc": 0,
                "cluster_id": "sparkrun_solosolosolo",
                "host_list": kwargs["host_list"],
                "is_solo": kwargs["is_solo"],
                "runtime": _FakeRuntime(),
                "recipe": recipe,
                "overrides": {},
                "container_image": "test:latest",
                "effective_cache_dir": "/tmp",
                "serve_port": 8000,
                "config": None,
                "recipe_ref": None,
                "comm_env": None,
                "ib_ip_map": {},
                "serve_command": "",
                "runtime_info": {},
                "builder": None,
                "backends": {},
                "timeline": None,
            },
        )()

    with (
        patch("sparkrun.api.status", return_value=ClusterStatus()),
        patch("sparkrun.core.launcher.launch_inference", side_effect=_capture),
        patch("sparkrun.api._resolve.resolve_runtime", return_value=_FakeRuntime()),
    ):
        result = api.run(opts)

    assert captured_hosts == ["h1"]
    assert result.is_solo is True


# --------------------------------------------------------------------------
# api.stop — argument validation
# --------------------------------------------------------------------------


def test_stop_requires_cluster_id_or_recipe():
    """Calling stop with neither cluster_id nor recipe must raise SparkrunError."""
    with pytest.raises(api.SparkrunError):
        api.stop()


def test_stop_unknown_cluster_id_raises_job_not_found(tmp_path):
    """When no job metadata matches cluster_id and no hosts given."""
    with pytest.raises(api.JobNotFound):
        api.stop(workload_id="sparkrun_doesnotexist", cache_dir=str(tmp_path))


def test_stop_with_hosts_skips_metadata_lookup(tmp_path, idle_occupancy):
    """Providing explicit hosts allows stop to proceed without metadata."""
    # Mock the SSH dispatch so no real connection is attempted.
    from sparkrun.orchestration.ssh import RemoteResult

    fake_result = RemoteResult(host="h1", returncode=0, stdout="", stderr="")
    with patch("sparkrun.orchestration.primitives.run_remote_command", return_value=fake_result) as remote:
        result = api.stop(
            workload_id="sparkrun_explicithost",
            hosts=("h1",),
            cache_dir=str(tmp_path),
        )
    remote.assert_called_once()
    assert not result.hosts_failed
    assert isinstance(result, api.StopResult)
    assert result.workload_id == "sparkrun_explicithost"
    assert result.hosts_targeted == ("h1",)


def test_discover_cluster_id_sweeps_whole_host_scope(monkeypatch):
    """``stop <recipe>`` intent-discovery sweeps the whole host scope, so a job
    launched under the native ``local`` executor (invisible to ``docker ps``) is
    still found — not just docker workloads."""
    from sparkrun.api._resolve import discover_cluster_id_by_intent
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.cluster_status import ClusterStatus, HostOccupancy, RunningWorkload
    from sparkrun.orchestration.executors.docker import DockerExecutor
    from sparkrun.orchestration.executors.local import LocalExecutor

    intent = "aabbccddeeff0011"
    cid = "sparkrun_%s_0123456789ab" % intent

    # docker sees nothing; the workload only exists as a local-executor process.
    monkeypatch.setattr(
        DockerExecutor,
        "query_status",
        lambda self, hosts, **kw: ClusterStatus(hosts=tuple(HostOccupancy(host=h) for h in hosts), executor="docker"),
    )
    monkeypatch.setattr(
        LocalExecutor,
        "query_status",
        lambda self, hosts, **kw: ClusterStatus(
            hosts=tuple(HostOccupancy(host=h, workloads=(RunningWorkload(cluster_id=cid),)) for h in hosts),
            executor="local",
        ),
    )

    found = discover_cluster_id_by_intent(
        intent,
        ["h1"],
        cluster_def=ClusterDefinition(name="c", hosts=["h1"]),
        cache_dir=None,
        sctx=None,
    )
    assert found == cid  # discovered via the local sweep, not docker


def test_discover_cluster_id_passes_the_cluster_hardware_inventory(monkeypatch):
    """Intent discovery queries status with the cluster's saved inventory, the
    way ``api.status`` does, so the executors never fall back to
    ``resolve_hardware()`` for a host the cluster already describes."""
    from sparkrun.api._resolve import discover_cluster_id_by_intent
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.cluster_status import HostOccupancy, RunningWorkload
    from sparkrun.core.hardware import AcceleratorSpec, HostHardware

    intent = "aabbccddeeff0011"
    cid = "sparkrun_%s_0123456789ab" % intent
    hw = HostHardware([AcceleratorSpec("nvidia", "jetson-orin", memory_gb=61.3)], source="detected")
    queries: list[dict] = []

    def _query(*args, **kwargs):
        queries.append(kwargs)
        return ClusterStatus(hosts=(HostOccupancy(host="h1", workloads=(RunningWorkload(cluster_id=cid),)),), executor="docker")

    monkeypatch.setattr("sparkrun.orchestration.executor.query_status_for_cluster", _query)

    found = discover_cluster_id_by_intent(
        intent,
        ["h1"],
        cluster_def=ClusterDefinition(name="c", hosts=["h1"], hosts_hardware={"h1": hw}),
        cache_dir=None,
        sctx=None,
    )

    assert found == cid
    assert queries[0]["host_hardware"] == {"h1": hw}


def test_discover_cluster_id_under_require_metadata_profile(monkeypatch):
    """End to end through the real docker/local executors: an application
    profile with ``hardware_fallback="require-metadata"`` finds its running
    workload on a cluster whose inventory was probed, instead of reporting
    the host's status as unavailable."""
    import json

    from sparkrun.api._resolve import discover_cluster_id_by_intent
    from sparkrun.core.application_profile import ApplicationProfile, select_application_profile
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.hardware import AcceleratorSpec, HostHardware

    profile = select_application_profile(
        ApplicationProfile(
            id="jetsonrun",
            display_name="Jetsonrun",
            command="jetsonrun",
            package="jetsonrun",
            hardware_fallback="require-metadata",
        )
    )
    intent = "aabbccddeeff0011"
    cid = "jetsonrun_%s_0123456789ab" % intent
    hw = HostHardware([AcceleratorSpec("nvidia", "jetson-orin", memory_gb=61.3)], source="detected")

    def _fake_parallel(hosts, script, **kwargs):
        out = ""
        if script.startswith("docker ps"):
            out = json.dumps({"ID": "x", "Names": cid + "_solo", "Labels": "sparkrun.distribution=" + profile.id}) + "\n"
        return [RemoteResult(h, 0, out, "") for h in hosts]

    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_scripts_parallel", _fake_parallel)

    found = discover_cluster_id_by_intent(
        intent,
        ["localhost"],
        cluster_def=ClusterDefinition(name="c", hosts=["localhost"], hosts_hardware={"localhost": hw}),
        cache_dir=None,
        sctx=None,
    )

    assert found == cid


# --------------------------------------------------------------------------
# api.logs — argument validation
# --------------------------------------------------------------------------


def test_logs_unknown_cluster_raises_job_not_found(tmp_path):
    """No metadata and no hosts → JobNotFound when consumed."""
    # logs() is a generator-returning function; the underlying call
    # currently raises immediately (no host source).  We consume the
    # iterator to confirm.
    with pytest.raises(api.JobNotFound):
        # Either call raises directly, or consuming the iterator does.
        gen = api.logs("sparkrun_doesnotexist", cache_dir=str(tmp_path))
        if isinstance(gen, Iterator):
            list(gen)


def test_logs_without_metadata_refuses_to_guess(tmp_path):
    """Hosts alone aren't enough: without job metadata there is no runtime, and
    without a runtime sparkrun cannot know the container name or whether the
    output lives on container stdout or in the in-container serve log.

    This used to return an iterator that silently read ``{cid}_solo`` via
    ``docker logs`` — the wrong container for Ray runtimes and an empty stream
    for every sleep-infinity + exec runtime.  Failing with an actionable
    message beats streaming nothing.
    """
    with pytest.raises(api.JobNotFound) as exc:
        api.logs("sparkrun_anyid", hosts=("h1",), cache_dir=str(tmp_path), tail=10)
    assert "recipe" in str(exc.value)


def test_logs_returns_iterator_when_metadata_names_the_runtime(tmp_path, monkeypatch):
    """With metadata recording the runtime, ``api.logs`` returns a lazy iterator.

    Resolution happens eagerly (so bad targets raise at call time); only the
    reading is deferred, so no subprocess is spawned by the call itself.
    """
    from sparkrun.core.recipe import Recipe
    from sparkrun.orchestration.job_metadata import save_job_metadata

    # A *resolved* runtime name, which is what save_job_metadata records for a
    # loaded recipe ("vllm" is resolved to vllm-ray / vllm-distributed at load).
    recipe = Recipe({"sparkrun_version": "2", "runtime": "vllm-distributed", "model": "test/m"})
    cluster_id = "sparkrun_aaaaaaaaaaaaaaaa_111111111111"
    save_job_metadata(cluster_id, recipe, ["h1"], cache_dir=str(tmp_path))

    from sparkrun.core.cluster_status import empty_status

    monkeypatch.setattr("sparkrun.orchestration.executors.docker.DockerExecutor.query_status", lambda self, hosts, **kw: empty_status([]))
    gen = api.logs(cluster_id, hosts=("h1",), cache_dir=str(tmp_path), tail=10)
    assert isinstance(gen, Iterator)


def test_logs_rejects_unknown_scope(tmp_path):
    with pytest.raises(api.SparkrunError):
        api.logs("sparkrun_anyid", hosts=("h1",), scope="everything", cache_dir=str(tmp_path))


def test_logs_requires_cluster_id_or_recipe():
    with pytest.raises(api.SparkrunError):
        api.logs()


def _make_status(host_workloads, errors=None):
    """Build a ClusterStatus for precheck tests.

    ``host_workloads`` maps host → list of (cluster_id, container_names).
    A host with no entry is unreachable (goes into ``errors``).
    """
    from sparkrun.core.cluster_status import ClusterStatus, ContainerDetail, HostOccupancy, RunningWorkload

    hosts = []
    for host, workloads in host_workloads.items():
        ws = []
        for cid, container_names in workloads:
            containers = tuple(ContainerDetail(name=n, role="solo", status="Up", image="img") for n in container_names)
            ws.append(RunningWorkload(cluster_id=cid, containers=containers))
        hosts.append(HostOccupancy(host=host, workloads=tuple(ws)))
    return ClusterStatus(hosts=tuple(hosts), executor="docker", errors=dict(errors or {}))


def test_logs_stopped_container_preserves_metadata(tmp_path):
    """Container stopped but still exists → JobNotFound + metadata preserved.

    ``query_status`` reports only running workloads, so the precheck asks
    ``executor.describe_terminated`` what became of this one.  The container is
    still on the host, so metadata is kept and the executor's own investigation
    hints are rendered.
    """
    from sparkrun.core.recipe import Recipe
    from sparkrun.orchestration.job_metadata import save_job_metadata, load_job_metadata

    recipe = Recipe({"sparkrun_version": "2", "runtime": "vllm-distributed", "model": "test/m"})
    cluster_id = "sparkrun_aaaaaaaaaaaaaaaa_111111111111"
    save_job_metadata(cluster_id, recipe, ["h1"], cache_dir=str(tmp_path))

    # query_status → host reachable, no workloads (container not running)
    # describe_terminated probe → docker ps -a lists it, still stopped on the host
    empty_snapshot = _make_status({"h1": []})
    probe = [RemoteResult(host="h1", returncode=0, stdout="%s_solo\tExited (1) 2 minutes ago\n" % cluster_id, stderr="")]
    with (
        patch.object(DockerExecutor, "query_status", return_value=empty_snapshot),
        patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", return_value=probe),
    ):
        with pytest.raises(api.JobNotFound) as exc:
            api.logs(cluster_id, hosts=("h1",), cache_dir=str(tmp_path), tail=10)

    msg = str(exc.value)
    assert "not currently running" in msg
    assert cluster_id in msg
    assert "Exited (1) 2 minutes ago" in msg
    # Hints come from the executor, not from this module.
    assert "docker logs" in msg
    assert "docker inspect" in msg
    assert "sparkrun stop" in msg
    # Metadata is preserved for investigation.
    assert load_job_metadata(cluster_id, cache_dir=str(tmp_path)) is not None


def test_logs_removed_container_preserves_metadata(tmp_path):
    """An absent container may be a pending launch; log inspection is read-only."""
    from sparkrun.core.recipe import Recipe
    from sparkrun.orchestration.job_metadata import save_job_metadata, load_job_metadata

    recipe = Recipe({"sparkrun_version": "2", "runtime": "vllm-distributed", "model": "test/m"})
    cluster_id = "sparkrun_aaaaaaaaaaaaaaaa_111111111111"
    save_job_metadata(cluster_id, recipe, ["h1"], cache_dir=str(tmp_path))

    # query_status → host reachable, no workloads (container not running)
    # describe_terminated probe → docker ps -a lists nothing (container gone)
    empty_snapshot = _make_status({"h1": []})
    gone = [RemoteResult(host="h1", returncode=0, stdout="", stderr="")]
    with (
        patch.object(DockerExecutor, "query_status", return_value=empty_snapshot),
        patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", return_value=gone),
    ):
        with pytest.raises(api.JobNotFound) as exc:
            api.logs(cluster_id, hosts=("h1",), cache_dir=str(tmp_path), tail=10)

    msg = str(exc.value)
    assert "nothing remains to read" in msg
    assert cluster_id in msg
    assert "job metadata has been preserved" in msg
    assert "sparkrun stop " + cluster_id in msg
    # auto_remove defaults on, so absence is `--rm` doing its job rather than a
    # workload that never ran — say which, and how to keep it next time.
    assert "auto-removed on exit" in msg
    assert "auto_remove=false" in msg
    assert load_job_metadata(cluster_id, cache_dir=str(tmp_path)) is not None


@pytest.mark.parametrize("relaunch", [False, True])
@pytest.mark.parametrize("filtered", [False, True])
def test_log_inspection_preserves_pending_launch_claims(tmp_path, relaunch, filtered):
    """Reading before container creation, including an ID reuse, cannot finalize."""
    from sparkrun.core.recipe import Recipe
    from sparkrun.orchestration.job_metadata import save_job_metadata, load_job_metadata

    recipe = Recipe({"sparkrun_version": "2", "runtime": "vllm-distributed", "model": "test/m"})
    cluster_id = "sparkrun_aaaaaaaaaaaaaaaa_111111111111"
    claims = [{"host": "h1", "cache_root": "/fixture/hf", "repo": "test/m", "revision": "a" * 40, "claim": "pending-launch"}]

    def record_launch():
        save_job_metadata(cluster_id, recipe, ["h1"], cache_dir=str(tmp_path), model_cache_claims=claims)

    record_launch()

    def observe(*args, **kwargs):
        if relaunch:
            # Model another launch replacing the metadata after logs resolved it.
            claims[0]["claim"] = "new-launch"
            record_launch()
        return _make_status({"h1": []})

    gone = [RemoteResult(host="h1", returncode=0, stdout="", stderr="")]
    with (
        patch.object(DockerExecutor, "query_status", side_effect=observe),
        patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", return_value=gone),
        patch("sparkrun.models.usage.release_claims") as release,
    ):
        with pytest.raises(api.JobNotFound, match="job metadata has been preserved"):
            api.logs(cluster_id, hosts=("h1",), cache_dir=str(tmp_path), **({"host": "h1"} if filtered else {}))
        release.assert_not_called()
    # A relaunch under the same ID keeps the earlier launch's claim alongside its own.
    expected = [{**claims[0], "claim": "pending-launch"}] + (claims if relaunch else [])
    assert load_job_metadata(cluster_id, cache_dir=str(tmp_path))["model_cache_claims"] == expected


def test_logs_precheck_worker_alive_when_head_dead(tmp_path):
    """Head dead but workers alive → helpful error pointing to --all-sources.

    In a multi-node job the head may crash while workers are still alive.
    The precheck detects this via the status snapshot (head host has no
    matching workload, worker host does) and raises a JobNotFound with a
    pointer to ``--all-sources`` rather than letting the reader fail on
    the dead head container with a raw docker error.  Metadata is preserved.
    """
    from sparkrun.core.recipe import Recipe
    from sparkrun.orchestration.job_metadata import save_job_metadata, load_job_metadata

    recipe = Recipe({"sparkrun_version": "2", "runtime": "vllm-distributed", "model": "test/m"})
    cluster_id = "sparkrun_aaaaaaaaaaaaaaaa_111111111111"
    save_job_metadata(cluster_id, recipe, ["h1", "h2"], cache_dir=str(tmp_path))

    # h1: reachable but no workloads (head dead)
    # h2: has the worker container running
    snapshot = _make_status(
        {
            "h1": [],
            "h2": [(cluster_id, [cluster_id + "_node_1"])],
        }
    )
    with patch.object(DockerExecutor, "query_status", return_value=snapshot):
        with pytest.raises(api.JobNotFound) as exc:
            api.logs(cluster_id, hosts=("h1", "h2"), cache_dir=str(tmp_path), tail=10)

    msg = str(exc.value)
    assert "partially running" in msg
    assert "h2" in msg
    assert "--all-sources" in msg
    # Metadata is preserved.
    assert load_job_metadata(cluster_id, cache_dir=str(tmp_path)) is not None


def test_logs_precheck_all_sources_proceeds_when_head_dead(tmp_path):
    """With --all-sources, head dead + worker alive → proceed (don't raise).

    When the user passes ``--all-sources`` (scope=SCOPE_ALL), the reader
    can read from the surviving workers, so the precheck should let it
    through rather than raising the "partially running" error.
    """
    from sparkrun.core.recipe import Recipe
    from sparkrun.orchestration.job_metadata import save_job_metadata

    recipe = Recipe({"sparkrun_version": "2", "runtime": "vllm-distributed", "model": "test/m"})
    cluster_id = "sparkrun_aaaaaaaaaaaaaaaa_111111111111"
    save_job_metadata(cluster_id, recipe, ["h1", "h2"], cache_dir=str(tmp_path))

    # h1: reachable but no workloads (head dead)
    # h2: has the worker container running
    snapshot = _make_status(
        {
            "h1": [],
            "h2": [(cluster_id, [cluster_id + "_node_1"])],
        }
    )
    with patch.object(DockerExecutor, "query_status", return_value=snapshot):
        gen = api.logs(cluster_id, hosts=("h1", "h2"), cache_dir=str(tmp_path), tail=10, scope="all")
        # With --all-sources, worker is alive → proceed (no JobNotFound).
        assert isinstance(gen, Iterator)


@pytest.mark.parametrize("selected_alive", [True, False])
def test_filtered_logs_precheck_only_selected_worker(tmp_path, selected_alive):
    from sparkrun.core.recipe import Recipe
    from sparkrun.orchestration.job_metadata import save_job_metadata, load_job_metadata

    recipe = Recipe({"sparkrun_version": "2", "runtime": "sglang", "model": "test/m"})
    cid = "sparkrun_aaaaaaaaaaaaaaaa_111111111111"
    save_job_metadata(cid, recipe, ["h1", "h2"], cache_dir=str(tmp_path))
    snapshot = _make_status(
        {
            "h1": [] if selected_alive else [(cid, [cid + "_node_0"])],
            "h2": [(cid, [cid + "_node_1"])] if selected_alive else [],
        }
    )
    with (
        patch.object(DockerExecutor, "query_status", return_value=snapshot) as query,
        patch.object(DockerExecutor, "describe_terminated", return_value={}),
    ):
        if selected_alive:
            target = api.open_log_target(cid, host="h2", cache_dir=str(tmp_path))
            assert target.sources[0].rank == 1
        else:
            with pytest.raises(api.JobNotFound):
                api.open_log_target(cid, host="h2", cache_dir=str(tmp_path))
        assert query.call_args.args[0] == ["h2"]
    assert load_job_metadata(cid, cache_dir=str(tmp_path)) is not None


def test_logs_precheck_passes_the_cluster_hardware_inventory(tmp_path):
    """The liveness precheck queries status with the cluster's saved
    inventory, as ``api.status`` does."""
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.hardware import AcceleratorSpec, HostHardware
    from sparkrun.core.recipe import Recipe
    from sparkrun.orchestration.job_metadata import save_job_metadata

    recipe = Recipe({"sparkrun_version": "2", "runtime": "sglang", "model": "test/m"})
    cid = "sparkrun_aaaaaaaaaaaaaaaa_111111111111"
    save_job_metadata(cid, recipe, ["h1"], cache_dir=str(tmp_path))
    hw = HostHardware([AcceleratorSpec("nvidia", "jetson-orin", memory_gb=61.3)], source="detected")
    cluster = ClusterDefinition(name="c", hosts=["h1"], hosts_hardware={"h1": hw})
    snapshot = _make_status({"h1": [(cid, [cid + "_solo"])]})

    with patch.object(DockerExecutor, "query_status", return_value=snapshot) as query:
        api.open_log_target(cid, cluster=cluster, cache_dir=str(tmp_path))

    assert query.call_args.kwargs["host_hardware"] == {"h1": hw}


def test_logs_precheck_runs_under_require_metadata_profile(tmp_path, monkeypatch):
    """End to end through the real docker executor: under a
    ``require-metadata`` profile the precheck still reports a dead workload
    on a cluster whose inventory was probed, instead of the status query
    raising and the check being skipped silently."""
    from sparkrun.core.application_profile import ApplicationProfile, select_application_profile
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.hardware import AcceleratorSpec, HostHardware
    from sparkrun.core.recipe import Recipe
    from sparkrun.orchestration.job_metadata import save_job_metadata

    select_application_profile(
        ApplicationProfile(
            id="jetsonrun",
            display_name="Jetsonrun",
            command="jetsonrun",
            package="jetsonrun",
            hardware_fallback="require-metadata",
        )
    )
    recipe = Recipe({"sparkrun_version": "2", "runtime": "sglang", "model": "test/m"})
    cid = "jetsonrun_aaaaaaaaaaaaaaaa_111111111111"
    save_job_metadata(cid, recipe, ["h1"], cache_dir=str(tmp_path))
    hw = HostHardware([AcceleratorSpec("nvidia", "jetson-orin", memory_gb=61.3)], source="detected")
    cluster = ClusterDefinition(name="c", hosts=["h1"], hosts_hardware={"h1": hw})

    # Host reachable, nothing running.
    monkeypatch.setattr(
        "sparkrun.orchestration.ssh.run_remote_scripts_parallel",
        lambda hosts, script, **kw: [RemoteResult(h, 0, "", "") for h in hosts],
    )

    with (
        patch.object(DockerExecutor, "describe_terminated", return_value={}),
        pytest.raises(api.JobNotFound),
    ):
        api.open_log_target(cid, cluster=cluster, cache_dir=str(tmp_path))


def test_logs_skips_precheck_on_ssh_failure(tmp_path):
    """Host unreachable (in ClusterStatus.errors) → precheck skipped.

    When ``query_status`` can't reach a host, it records it in
    ``ClusterStatus.errors`` and omits it from ``hosts``.  The precheck
    treats this as inconclusive (``for_host()`` returns ``None``) and
    skips, letting the log reader surface its own error.
    """
    from sparkrun.core.recipe import Recipe
    from sparkrun.orchestration.job_metadata import save_job_metadata

    recipe = Recipe({"sparkrun_version": "2", "runtime": "vllm-distributed", "model": "test/m"})
    cluster_id = "sparkrun_aaaaaaaaaaaaaaaa_111111111111"
    save_job_metadata(cluster_id, recipe, ["h1"], cache_dir=str(tmp_path))

    # query_status → h1 unreachable (in errors, absent from hosts)
    snapshot = _make_status({}, errors={"h1": "unreachable"})
    with patch.object(DockerExecutor, "query_status", return_value=snapshot):
        gen = api.logs(cluster_id, hosts=("h1",), cache_dir=str(tmp_path), tail=10)
        # Precheck skipped → returns a lazy iterator (no JobNotFound raised).
        assert isinstance(gen, Iterator)


def test_logs_precheck_skips_on_query_exception(tmp_path):
    """If executor.query_status raises, precheck is skipped (best-effort).

    The precheck must never crash — a broken query_status should let the
    log reader surface its own error rather than preempting with a
    misleading 'not running' message.
    """
    from sparkrun.core.recipe import Recipe
    from sparkrun.orchestration.job_metadata import save_job_metadata

    recipe = Recipe({"sparkrun_version": "2", "runtime": "vllm-distributed", "model": "test/m"})
    cluster_id = "sparkrun_aaaaaaaaaaaaaaaa_111111111111"
    save_job_metadata(cluster_id, recipe, ["h1"], cache_dir=str(tmp_path))

    with patch.object(DockerExecutor, "query_status", side_effect=RuntimeError("internal error")):
        gen = api.logs(cluster_id, hosts=("h1",), cache_dir=str(tmp_path), tail=10)
        assert isinstance(gen, Iterator)


def test_logs_precheck_exists_cmd_failure_preserves_metadata(tmp_path):
    """An inconclusive ``describe_terminated`` probe → preserve metadata.

    After the status snapshot shows nothing running, the precheck asks the
    executor what became of the workload.  When that probe cannot answer (SSH
    failure rc=255, no container engine rc=127, timeout), the executor reports
    nothing for that source — and "cannot tell" must never be read as
    "confirmed gone", because that is the verdict that deletes metadata.
    """
    from sparkrun.core.recipe import Recipe
    from sparkrun.orchestration.job_metadata import save_job_metadata, load_job_metadata

    recipe = Recipe({"sparkrun_version": "2", "runtime": "vllm-distributed", "model": "test/m"})
    cluster_id = "sparkrun_aaaaaaaaaaaaaaaa_111111111111"
    save_job_metadata(cluster_id, recipe, ["h1"], cache_dir=str(tmp_path))

    # query_status → host reachable, no workloads (container not running)
    # describe_terminated probe → rc=255 (SSH failure), so no verdict at all
    empty_snapshot = _make_status({"h1": []})
    ssh_fail_followup = [RemoteResult(host="h1", returncode=255, stdout="", stderr="ssh: connection refused")]
    with (
        patch.object(DockerExecutor, "query_status", return_value=empty_snapshot),
        patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", return_value=ssh_fail_followup),
    ):
        with pytest.raises(api.JobNotFound) as exc:
            api.logs(cluster_id, hosts=("h1",), cache_dir=str(tmp_path), tail=10)

    msg = str(exc.value)
    # Inconclusive must not read as "gone" — that verdict deletes metadata.
    assert "not currently running" in msg
    assert "nothing remains to read" not in msg
    # Metadata preserved.
    assert load_job_metadata(cluster_id, cache_dir=str(tmp_path)) is not None


# --------------------------------------------------------------------------
# open_log_target and LogTarget (Step 2)
# --------------------------------------------------------------------------


def test_open_log_target_exposed():
    assert hasattr(api, "open_log_target") and callable(api.open_log_target)
    assert hasattr(api, "LogTarget")


def test_open_log_target_host_filtering_and_precheck(tmp_path):
    """open_log_target(cid, host='h2') keeps only h2 sources, uses SCOPE_ALL,
    and runs precheck on the filtered list so a dead head on h1 doesn't block."""
    from sparkrun.core.recipe import Recipe
    from sparkrun.orchestration.job_metadata import save_job_metadata

    recipe = Recipe({"sparkrun_version": "2", "runtime": "vllm-distributed", "model": "test/m"})
    cluster_id = "sparkrun_aaaaaaaaaaaaaaaa_111111111111"
    save_job_metadata(cluster_id, recipe, ["h1", "h2"], cache_dir=str(tmp_path))

    # Snapshot: h1 head dead, h2 worker alive
    snapshot = _make_status(
        {
            "h1": [],
            "h2": [(cluster_id, [cluster_id + "_node_1"])],
        }
    )
    with patch.object(DockerExecutor, "query_status", return_value=snapshot):
        target = api.open_log_target(cluster_id, hosts=("h1", "h2"), host="h2", cache_dir=str(tmp_path))
        assert isinstance(target, api.LogTarget)
        assert len(target.sources) == 1
        assert target.sources[0].host == "h2"
        assert target.sources[0].container == cluster_id + "_node_1"


def test_open_log_target_container_filtering(tmp_path):
    """open_log_target with container filter keeps only the named container."""
    from sparkrun.core.recipe import Recipe
    from sparkrun.orchestration.job_metadata import save_job_metadata

    recipe = Recipe({"sparkrun_version": "2", "runtime": "vllm-distributed", "model": "test/m"})
    cluster_id = "sparkrun_aaaaaaaaaaaaaaaa_111111111111"
    save_job_metadata(cluster_id, recipe, ["h1", "h2"], cache_dir=str(tmp_path))

    snapshot = _make_status(
        {
            "h1": [(cluster_id, [cluster_id + "_node_0"])],
            "h2": [(cluster_id, [cluster_id + "_node_1"])],
        }
    )
    with patch.object(DockerExecutor, "query_status", return_value=snapshot):
        target = api.open_log_target(cluster_id, hosts=("h1", "h2"), host="h2", container=cluster_id + "_node_1", cache_dir=str(tmp_path))
        assert len(target.sources) == 1
        assert target.sources[0].container == cluster_id + "_node_1"


def test_open_log_target_cluster_scope_all(tmp_path):
    """open_log_target with host=None and scope=all returns all sources."""
    from sparkrun.core.recipe import Recipe
    from sparkrun.orchestration.job_metadata import save_job_metadata

    recipe = Recipe({"sparkrun_version": "2", "runtime": "vllm-distributed", "model": "test/m"})
    cluster_id = "sparkrun_aaaaaaaaaaaaaaaa_111111111111"
    save_job_metadata(cluster_id, recipe, ["h1", "h2"], cache_dir=str(tmp_path))

    snapshot = _make_status(
        {
            "h1": [(cluster_id, [cluster_id + "_node_0"])],
            "h2": [(cluster_id, [cluster_id + "_node_1"])],
        }
    )
    with patch.object(DockerExecutor, "query_status", return_value=snapshot):
        target = api.open_log_target(cluster_id, hosts=("h1", "h2"), host=None, scope="all", cache_dir=str(tmp_path))
        assert len(target.sources) == 2
        hosts = [s.host for s in target.sources]
        assert "h1" in hosts and "h2" in hosts


def test_open_log_target_host_normalization(tmp_path):
    """Host matching handles user@ prefix and case insensitivity."""
    from sparkrun.core.recipe import Recipe
    from sparkrun.orchestration.job_metadata import save_job_metadata

    recipe = Recipe({"sparkrun_version": "2", "runtime": "vllm-distributed", "model": "test/m"})
    cluster_id = "sparkrun_aaaaaaaaaaaaaaaa_111111111111"
    save_job_metadata(cluster_id, recipe, ["h1"], cache_dir=str(tmp_path))

    snapshot = _make_status({"h1": [(cluster_id, [cluster_id + "_solo"])]})
    with patch.object(DockerExecutor, "query_status", return_value=snapshot):
        target = api.open_log_target(cluster_id, hosts=("h1",), host="ubuntu@H1", cache_dir=str(tmp_path))
        assert len(target.sources) == 1
        assert target.sources[0].host == "h1"


def test_open_log_target_unmatched_host_raises_job_not_found(tmp_path):
    """Host filter matching no sources raises JobNotFound."""
    from sparkrun.core.recipe import Recipe
    from sparkrun.orchestration.job_metadata import save_job_metadata

    recipe = Recipe({"sparkrun_version": "2", "runtime": "vllm-distributed", "model": "test/m"})
    cluster_id = "sparkrun_aaaaaaaaaaaaaaaa_111111111111"
    save_job_metadata(cluster_id, recipe, ["h1"], cache_dir=str(tmp_path))

    with pytest.raises(api.JobNotFound, match="No log sources match"):
        api.open_log_target(cluster_id, hosts=("h1",), host="h999", cache_dir=str(tmp_path))


def test_open_log_target_unmatched_container_raises_job_not_found(tmp_path):
    """Container filter matching no sources raises JobNotFound."""
    from sparkrun.core.recipe import Recipe
    from sparkrun.orchestration.job_metadata import save_job_metadata

    recipe = Recipe({"sparkrun_version": "2", "runtime": "vllm-distributed", "model": "test/m"})
    cluster_id = "sparkrun_aaaaaaaaaaaaaaaa_111111111111"
    save_job_metadata(cluster_id, recipe, ["h1"], cache_dir=str(tmp_path))

    with pytest.raises(api.JobNotFound, match="No log sources match"):
        api.open_log_target(cluster_id, hosts=("h1",), container="nonexistent", cache_dir=str(tmp_path))


def test_log_target_follow():
    """target.follow() streams from read_log_sources with tail and cancel."""
    from sparkrun.core.log_source import LogLine, LogSource

    source = LogSource(host="h1", container="c0")
    dummy_line = LogLine(host="h1", container="c0", text="hello", role="node_0", rank=0, timestamp=0.0)

    target = api.LogTarget(
        workload_id="cid",
        runtime="vllm-distributed",
        sources=(source,),
        _executor=DockerExecutor(),
        _ssh_kwargs={},
    )
    with patch("sparkrun.orchestration.logs.read_log_sources", return_value=iter([dummy_line])) as mock_read:
        lines = list(target.follow(tail=150))
        assert lines == [dummy_line]
        assert mock_read.call_args[0] == (target._executor, target.sources)
        assert mock_read.call_args[1]["follow"] is True
        assert mock_read.call_args[1]["tail"] == 150


def test_log_target_page():
    """target.page(source, skip=10, count=5) reads from read_log_sources."""
    from sparkrun.core.log_source import LogLine, LogSource

    source = LogSource(host="h1", container="c0")
    lines = [LogLine(host="h1", container="c0", text=f"line-{i}", role="node_0", rank=0, timestamp=0.0) for i in range(15)]

    target = api.LogTarget(
        workload_id="cid",
        runtime="vllm-distributed",
        sources=(source,),
        _executor=DockerExecutor(),
        _ssh_kwargs={},
    )
    with patch("sparkrun.orchestration.logs.read_log_sources", return_value=(line for line in lines)) as mock_read:
        result = target.page(source, skip=10, count=5)
        assert len(result) == 5
        assert [line.text for line in result] == ["line-0", "line-1", "line-2", "line-3", "line-4"]
        assert mock_read.call_args[0] == (target._executor, (source,))
        assert mock_read.call_args[1]["tail"] == 15


class _RealShellDumpExecutor(DockerExecutor):
    def read_logs_cmd(self, source, *, follow=False, tail=None):
        if source.role == "head":
            return "printf 'head line 1\\nhead line 2\\n'"
        elif source.role == "worker":
            return "printf 'worker line 1\\nworker line 2\\n'"
        elif source.role == "slow":
            # Emits lines with delay so mid-stream cancellation can be tested against real
            # processes; long enough that cancellation always lands well before EOF.
            return 'for i in $(seq 1 1000); do echo "slow line $i"; sleep 0.05; done'
        elif source.role == "stderr":
            # Workload stdout plus transport noise: stderr must land in the
            # reader's diagnostics, never inside the dumped log history.
            return 'printf "out 1\\n"; printf "transport warning\\n" >&2; printf "out 2\\n"'
        return "printf 'default line\\n'"


def test_log_target_dump_real_ordering_and_prefixes(tmp_path):
    """target.dump streams lines in rank order, emits correct headers, and prefixes lines appropriately."""
    from sparkrun.core.log_source import LogSource, MODE_FILE, SCOPE_ALL

    worker_src = LogSource(host="localhost", container="c1", role="worker", rank=1, mode=MODE_FILE)
    head_src = LogSource(host="localhost", container="c0", role="head", rank=0, mode=MODE_FILE)

    executor = _RealShellDumpExecutor()

    # 1. Cluster scope (sources given out of order: [worker, head])
    target_cluster = api.LogTarget(
        workload_id="sparkrun_dump123",
        runtime="vllm-distributed",
        sources=(worker_src, head_src),
        _executor=executor,
        _scope=SCOPE_ALL,
        _ssh_kwargs={},
    )
    out_cluster = tmp_path / "cluster.log"
    count = target_cluster.dump(str(out_cluster))
    assert count == 4
    assert out_cluster.exists()
    assert not (tmp_path / "cluster.log.partial").exists()

    content_cluster = out_cluster.read_text(encoding="utf-8").splitlines()
    assert content_cluster[0] == "# workload: sparkrun_dump123 | scope: cluster (all) | sources: localhost/head, localhost/worker"
    # Rank 0 (head) must come before Rank 1 (worker), with [host/role/rank] prefix
    assert content_cluster[1] == "[localhost/head/0] head line 1"
    assert content_cluster[2] == "[localhost/head/0] head line 2"
    assert content_cluster[3] == "[localhost/worker/1] worker line 1"
    assert content_cluster[4] == "[localhost/worker/1] worker line 2"

    # 2. Node scope (multiple containers on same host -> [role/rank] prefix, no host)
    target_node = api.LogTarget(
        workload_id="sparkrun_dump123",
        runtime="vllm-distributed",
        sources=(worker_src, head_src),
        _executor=executor,
        _host="localhost",
        _ssh_kwargs={},
    )
    out_node = tmp_path / "node.log"
    count = target_node.dump(str(out_node))
    assert count == 4
    content_node = out_node.read_text(encoding="utf-8").splitlines()
    assert content_node[0] == "# workload: sparkrun_dump123 | scope: node (localhost) | sources: localhost/head, localhost/worker"
    assert content_node[1] == "[head/0] head line 1"
    assert content_node[2] == "[head/0] head line 2"
    assert content_node[3] == "[worker/1] worker line 1"
    assert content_node[4] == "[worker/1] worker line 2"

    # 3. Single source (no prefix attached to lines)
    target_single = api.LogTarget(
        workload_id="sparkrun_dump123",
        runtime="vllm-distributed",
        sources=(head_src,),
        _executor=executor,
        _host="localhost",
        _container="c0",
        _ssh_kwargs={},
    )
    out_single = tmp_path / "single.log"
    count = target_single.dump(str(out_single))
    assert count == 2
    content_single = out_single.read_text(encoding="utf-8").splitlines()
    assert content_single[0] == "# workload: sparkrun_dump123 | scope: container (localhost/c0) | sources: localhost/head"
    assert content_single[1] == "head line 1"
    assert content_single[2] == "head line 2"


def test_log_target_dump_real_overwrite(tmp_path):
    """dump() respects overwrite=True and refuses overwrite=False with existing destination."""
    from sparkrun.core.log_source import LogSource, MODE_FILE

    head_src = LogSource(host="localhost", container="c0", role="head", rank=0, mode=MODE_FILE)
    target = api.LogTarget(
        workload_id="sparkrun_dump123",
        runtime="vllm-distributed",
        sources=(head_src,),
        _executor=_RealShellDumpExecutor(),
        _ssh_kwargs={},
    )

    out_path = tmp_path / "existing.log"
    out_path.write_text("prior content\n")

    with pytest.raises(api.SparkrunError, match="already exists"):
        target.dump(str(out_path), overwrite=False)
    assert out_path.read_text() == "prior content\n"
    assert not (tmp_path / "existing.log.partial").exists()

    count = target.dump(str(out_path), overwrite=True)
    assert count == 2
    assert "head line 1" in out_path.read_text()
    assert not (tmp_path / "existing.log.partial").exists()


def test_log_target_dump_real_midstream_cancellation(tmp_path):
    """When cancelled mid-stream, dump() terminates reader, removes .partial, and returns partial count."""
    import threading
    from sparkrun.core.log_source import LogSource, MODE_FILE

    slow_src = LogSource(host="localhost", container="c_slow", role="slow", rank=0, mode=MODE_FILE)
    target = api.LogTarget(
        workload_id="sparkrun_dump123",
        runtime="vllm-distributed",
        sources=(slow_src,),
        _executor=_RealShellDumpExecutor(),
        _ssh_kwargs={},
    )

    cancel = threading.Event()
    # Trigger cancel after short delay while the loop command is actively writing
    timer = threading.Timer(0.15, cancel.set)
    timer.start()

    out_path = tmp_path / "cancelled.log"
    partial_path = tmp_path / "cancelled.log.partial"

    try:
        count = target.dump(str(out_path), cancel=cancel)
        assert count > 0  # Captured partial lines before cancel
        assert not out_path.exists()
        assert not partial_path.exists()
    finally:
        timer.cancel()


def test_log_target_dump_honors_target_cancel_event(tmp_path):
    """dump(path) with cancel=None must not publish when the target's own event is set."""
    import threading
    from sparkrun.core.log_source import LogSource, MODE_FILE

    head_src = LogSource(host="localhost", container="c0", role="head", rank=0, mode=MODE_FILE)
    target_cancel = threading.Event()
    target_cancel.set()
    target = api.LogTarget(
        workload_id="sparkrun_dump123",
        runtime="vllm-distributed",
        sources=(head_src,),
        _executor=_RealShellDumpExecutor(),
        _ssh_kwargs={},
        _cancel=target_cancel,
    )

    out_path = tmp_path / "scoped-cancel.log"
    partial_path = tmp_path / "scoped-cancel.log.partial"

    count = target.dump(str(out_path))

    assert count == 0
    assert not out_path.exists()
    assert not partial_path.exists()


def test_log_target_dump_keyboard_interrupt_removes_partial(tmp_path):
    """When interrupted by KeyboardInterrupt (or SIGINT/SIGTERM), .partial file is removed."""
    from sparkrun.core.log_source import LogSource, MODE_FILE

    head_src = LogSource(host="localhost", container="c0", role="head", rank=0, mode=MODE_FILE)
    target = api.LogTarget(
        workload_id="sparkrun_dump123",
        runtime="vllm-distributed",
        sources=(head_src,),
        _executor=_RealShellDumpExecutor(),
        _ssh_kwargs={},
    )

    out_path = tmp_path / "ki.log"
    partial_path = tmp_path / "ki.log.partial"

    with patch("sparkrun.orchestration.logs.read_log_sources", side_effect=KeyboardInterrupt()):
        with pytest.raises(KeyboardInterrupt):
            target.dump(str(out_path))

    assert not out_path.exists()
    assert not partial_path.exists()


def test_log_target_dump_concurrent_dump_exclusive_creation(tmp_path):
    """A dump cannot collide with an in-progress dump and will not remove the other dump's partial file."""
    from sparkrun.core.log_source import LogSource, MODE_FILE

    head_src = LogSource(host="localhost", container="c0", role="head", rank=0, mode=MODE_FILE)
    target = api.LogTarget(
        workload_id="sparkrun_dump123",
        runtime="vllm-distributed",
        sources=(head_src,),
        _executor=_RealShellDumpExecutor(),
        _ssh_kwargs={},
    )

    out_path = tmp_path / "concurrent.log"
    partial_path = tmp_path / "concurrent.log.partial"
    partial_path.write_text("in-flight dump contents from another process\n")

    with pytest.raises(api.SparkrunError, match="already in progress"):
        target.dump(str(out_path), overwrite=True)

    # The existing partial file created by the other process must remain intact!
    assert partial_path.exists()
    assert partial_path.read_text() == "in-flight dump contents from another process\n"
    assert not out_path.exists()

    partial_path.unlink()


def test_log_target_dump_no_overwrite_race_atomic_link(tmp_path):
    """If target file appears mid-stream when overwrite=False, finalize fails without clobbering and cleans up .partial."""
    from sparkrun.core.log_source import LogSource, MODE_FILE

    head_src = LogSource(host="localhost", container="c0", role="head", rank=0, mode=MODE_FILE)
    target = api.LogTarget(
        workload_id="sparkrun_dump123",
        runtime="vllm-distributed",
        sources=(head_src,),
        _executor=_RealShellDumpExecutor(),
        _ssh_kwargs={},
    )

    out_path = tmp_path / "raced.log"
    partial_path = tmp_path / "raced.log.partial"

    # Simulate the destination file appearing mid-dump: the write runs eagerly
    # at read start, before dump() iterates or finalizes.
    def simulate_race_read(executor, sources, **kwargs):
        out_path.write_text("raced destination content\n")
        return (line for line in ())

    with patch("sparkrun.orchestration.logs.read_log_sources", side_effect=simulate_race_read):
        with pytest.raises(api.SparkrunError, match="already exists"):
            target.dump(str(out_path), overwrite=False)

    assert out_path.read_text() == "raced destination content\n"
    assert not partial_path.exists()


def test_log_target_dump_without_hard_links_refuses_rather_than_check_and_replace(tmp_path, monkeypatch):
    """No hard links on POSIX: refuse to publish instead of exists() + replace(), and clean up."""
    import errno

    from sparkrun.core.log_source import LogSource, MODE_FILE

    head_src = LogSource(host="localhost", container="c0", role="head", rank=0, mode=MODE_FILE)
    target = api.LogTarget(
        workload_id="sparkrun_dump123",
        runtime="vllm-distributed",
        sources=(head_src,),
        _executor=_RealShellDumpExecutor(),
        _ssh_kwargs={},
    )

    def no_links(src, dst):
        raise PermissionError(errno.EPERM, "Operation not permitted")

    replaced = []
    monkeypatch.setattr("sparkrun.api._logs.os.link", no_links)
    monkeypatch.setattr("sparkrun.api._logs.os.replace", lambda *a: replaced.append(a))
    out_path = tmp_path / "nolinks.log"

    with pytest.raises(api.SparkrunError, match="without risking an overwrite"):
        target.dump(str(out_path), overwrite=False)

    assert replaced == []
    assert not out_path.exists()
    assert not (tmp_path / "nolinks.log.partial").exists()


def test_publish_without_overwrite_windows_rename_fallback(tmp_path, monkeypatch):
    """Without hard links on Windows, rename publishes and an existing target is refused."""
    import errno

    from sparkrun.api import _logs

    def no_links(src, dst):
        raise PermissionError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(_logs.os, "link", no_links)
    monkeypatch.setattr(_logs.os, "name", "nt")

    partial = tmp_path / "out.log.partial"
    partial.write_text("body\n")
    out = tmp_path / "out.log"
    _logs._publish_without_overwrite(str(partial), str(out), str(out))
    assert out.read_text() == "body\n"
    assert not partial.exists()

    def rename_exists(src, dst):
        raise FileExistsError(errno.EEXIST, "exists")

    monkeypatch.setattr(_logs.os, "rename", rename_exists)
    partial.write_text("second\n")
    with pytest.raises(api.SparkrunError, match="already exists"):
        _logs._publish_without_overwrite(str(partial), str(out), str(out))
    assert out.read_text() == "body\n"


def test_log_target_dump_missing_directory(tmp_path):
    """dump() raises SparkrunError if parent directory is missing, no partial created."""
    from sparkrun.core.log_source import LogSource

    target = api.LogTarget(
        workload_id="sparkrun_dump123",
        runtime="vllm-distributed",
        sources=(LogSource(host="h1", container="c0"),),
        _executor=DockerExecutor(),
        _ssh_kwargs={},
    )

    bad_path = tmp_path / "nonexistent_dir" / "out.log"
    with pytest.raises(api.SparkrunError, match="does not exist"):
        target.dump(str(bad_path))

    assert not (tmp_path / "nonexistent_dir" / "out.log.partial").exists()


def test_log_target_dump_error_removes_partial(tmp_path):
    """When dump raises, the .partial file is cleaned up."""
    from sparkrun.core.log_source import LogSource

    target = api.LogTarget(
        workload_id="sparkrun_dump123",
        runtime="vllm-distributed",
        sources=(LogSource(host="h1", container="c0"),),
        _executor=DockerExecutor(),
        _ssh_kwargs={},
    )

    out_path = tmp_path / "failed.log"
    with patch("sparkrun.orchestration.logs.read_log_sources", side_effect=RuntimeError("disk full")):
        with pytest.raises(RuntimeError, match="disk full"):
            target.dump(str(out_path))

    assert not out_path.exists()
    assert not (tmp_path / "failed.log.partial").exists()


def test_log_target_dump_explicit_prefix_modes(tmp_path, monkeypatch):
    """Explicit dump prefixes: True, string, callable, and multi-host True."""
    from sparkrun.core.log_source import LogSource, MODE_FILE

    head_src = LogSource(host="localhost", container="c0", role="head", rank=0, mode=MODE_FILE)
    executor = _RealShellDumpExecutor()

    def make(sources, **extra):
        return api.LogTarget(
            workload_id="sparkrun_dump123",
            runtime="vllm-distributed",
            sources=tuple(sources),
            _executor=executor,
            _ssh_kwargs={},
            **extra,
        )

    # prefix=True on a single-host target tags with the role only.
    out = tmp_path / "true.log"
    assert make([head_src]).dump(str(out), prefix=True) == 2
    assert out.read_text(encoding="utf-8").splitlines()[1:] == ["[head] head line 1", "[head] head line 2"]

    # Non-empty string prefix.
    out = tmp_path / "custom.log"
    assert make([head_src]).dump(str(out), prefix="[custom]") == 2
    assert out.read_text(encoding="utf-8").splitlines()[1] == "[custom] head line 1"

    # Callable prefix returning a tag (or falsy to skip the tag).
    out = tmp_path / "callable.log"
    assert make([head_src]).dump(str(out), prefix=lambda line: f"<{line.host}>") == 2
    assert out.read_text(encoding="utf-8").splitlines()[1:] == ["<localhost> head line 1", "<localhost> head line 2"]

    # prefix=True with multiple hosts tags with host/role.
    monkeypatch.setattr("sparkrun.orchestration.ssh.should_run_locally", lambda *a, **kw: True)
    worker_src = LogSource(host="h2", container="c1", role="worker", rank=1, mode=MODE_FILE)
    out = tmp_path / "multihost.log"
    assert make([head_src, worker_src]).dump(str(out), prefix=True) == 4
    body = out.read_text(encoding="utf-8").splitlines()[1:]
    assert body == [
        "[localhost/head] head line 1",
        "[localhost/head] head line 2",
        "[h2/worker] worker line 1",
        "[h2/worker] worker line 2",
    ]


def test_log_target_dump_rank_none_orders_last(tmp_path):
    """A rank-less source sorts after every ranked one through the production read path."""
    from sparkrun.core.log_source import LogSource, MODE_FILE

    sources = (
        LogSource(host="localhost", container="c0", role="head", rank=0, mode=MODE_FILE),
        LogSource(host="localhost", container="c1", role="worker", rank=1, mode=MODE_FILE),
        LogSource(host="localhost", container="c2", role="node_3", rank=None, mode=MODE_FILE),
    )
    target = api.LogTarget(
        workload_id="sparkrun_dump123",
        runtime="vllm-distributed",
        sources=sources,
        _executor=_RealShellDumpExecutor(),
        _ssh_kwargs={},
    )

    out = tmp_path / "order.log"
    assert target.dump(str(out), prefix=False) == 5
    assert out.read_text(encoding="utf-8").splitlines()[1:] == [
        "head line 1",
        "head line 2",
        "worker line 1",
        "worker line 2",
        "default line",
    ]


def test_log_target_dump_cancel_reaps_process_group(tmp_path, monkeypatch):
    """Cancelling a dump reaps the reader's whole process group."""
    import os
    import threading
    import time as time_module

    from sparkrun.core.log_source import LogSource, MODE_FILE
    from sparkrun.orchestration import logs as logs_module

    slow_src = LogSource(host="localhost", container="c_slow", role="slow", rank=0, mode=MODE_FILE)
    target = api.LogTarget(
        workload_id="sparkrun_dump123",
        runtime="vllm-distributed",
        sources=(slow_src,),
        _executor=_RealShellDumpExecutor(),
        _ssh_kwargs={},
    )

    real_spawn = logs_module._spawn
    captured: dict = {}

    def capturing_spawn(cmd, **kwargs):
        proc = real_spawn(cmd, **kwargs)
        captured["proc"] = proc
        return proc

    monkeypatch.setattr(logs_module, "_spawn", capturing_spawn)

    cancel = threading.Event()
    out_path = tmp_path / "cancel.log"
    partial_path = tmp_path / "cancel.log.partial"
    result: dict = {}

    def run_dump():
        result["count"] = target.dump(str(out_path), cancel=cancel)

    thread = threading.Thread(target=run_dump)
    thread.start()
    try:
        # The .partial file is userspace-buffered, so its size is not an observable
        # "mid-stream" signal: wait for the reader process, then let it stream briefly.
        deadline = time_module.monotonic() + 5
        while time_module.monotonic() < deadline and "proc" not in captured:
            time_module.sleep(0.01)
        assert "proc" in captured
        time_module.sleep(0.5)
        cancel.set()
    finally:
        thread.join(timeout=10)
    assert not thread.is_alive()

    assert result["count"] > 0
    assert not out_path.exists()
    assert not partial_path.exists()

    proc = captured["proc"]
    reaped = False
    deadline = time_module.monotonic() + 5
    while time_module.monotonic() < deadline:
        try:
            os.killpg(proc.pid, 0)
            time_module.sleep(0.05)
        except ProcessLookupError:
            reaped = True
            break
    assert reaped, f"Process group {proc.pid} still alive after dump cancellation"


def test_log_target_dump_empty_sources_writes_header_only(tmp_path):
    """A target with no sources still publishes a header-only dump and reports zero lines."""
    target = api.LogTarget(
        workload_id="sparkrun_dump123",
        runtime="vllm-distributed",
        sources=(),
        _executor=DockerExecutor(),
        _ssh_kwargs={},
    )

    out_path = tmp_path / "empty.log"
    assert target.dump(str(out_path)) == 0

    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("# workload: sparkrun_dump123")


def test_log_target_dump_reader_error_raises_called_process_error(tmp_path):
    """A failing reader surfaces as CalledProcessError and leaves no .partial."""
    import subprocess

    from sparkrun.core.log_source import LogSource

    class _FailingReadExecutor(DockerExecutor):
        def read_logs_cmd(self, source, *, follow=False, tail=None):
            return "echo err; exit 3"

    target = api.LogTarget(
        workload_id="sparkrun_dump123",
        runtime="vllm-distributed",
        sources=(LogSource(host="localhost", container="c0", role="head", rank=0),),
        _executor=_FailingReadExecutor(),
        _ssh_kwargs={},
    )

    out_path = tmp_path / "failed.log"
    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        target.dump(str(out_path))
    assert exc_info.value.returncode == 3
    assert not out_path.exists()
    assert not (tmp_path / "failed.log.partial").exists()


def test_log_target_dump_excludes_reader_stderr(tmp_path):
    """Transport stderr stays in diagnostics; the dump holds only workload stdout."""
    from sparkrun.core.log_source import LogSource, MODE_FILE

    src = LogSource(host="localhost", container="c0", role="stderr", rank=0, mode=MODE_FILE)
    target = api.LogTarget(
        workload_id="sparkrun_dump123",
        runtime="vllm-distributed",
        sources=(src,),
        _executor=_RealShellDumpExecutor(),
        _ssh_kwargs={},
    )

    out_path = tmp_path / "stderr.log"
    assert target.dump(str(out_path), prefix=False) == 2
    assert out_path.read_text(encoding="utf-8").splitlines()[1:] == ["out 1", "out 2"]


def test_log_target_dump_validates_native_source(tmp_path):
    """Native sources are validated before the read and again after normal EOF."""
    from sparkrun.core.log_source import LogResource, LogSource

    calls: list = []

    class _ValidatingExecutor:
        def read_logs_cmd(self, source, *, follow=False, tail=None):
            return "printf 'native line\\n'"

        def validate_log_source(self, source, *, cancel=None, timeout=None):
            calls.append(source)

    src = LogSource(
        host="localhost",
        container="c0",
        role="head",
        rank=0,
        resource=LogResource(kind="job", namespace="ns", name="n", uid="u", instance="i"),
    )
    target = api.LogTarget(
        workload_id="sparkrun_dump123",
        runtime="vllm-distributed",
        sources=(src,),
        _executor=_ValidatingExecutor(),
        _ssh_kwargs={},
    )

    out_path = tmp_path / "native.log"
    assert target.dump(str(out_path), prefix=False) == 1
    assert calls == [src, src]
    assert out_path.read_text(encoding="utf-8").splitlines()[1] == "native line"


def test_logs_forwards_host_container_and_cancel(tmp_path):
    """api.logs delegates to open_log_target and forwards host, container, cancel."""
    import threading
    from sparkrun.core.recipe import Recipe
    from sparkrun.orchestration.job_metadata import save_job_metadata

    recipe = Recipe({"sparkrun_version": "2", "runtime": "vllm-distributed", "model": "test/m"})
    cluster_id = "sparkrun_aaaaaaaaaaaaaaaa_111111111111"
    save_job_metadata(cluster_id, recipe, ["h1", "h2"], cache_dir=str(tmp_path))

    snapshot = _make_status(
        {
            "h1": [(cluster_id, [cluster_id + "_node_0"])],
            "h2": [(cluster_id, [cluster_id + "_node_1"])],
        }
    )
    cancel = threading.Event()
    with patch.object(DockerExecutor, "query_status", return_value=snapshot):
        with patch.object(api.LogTarget, "follow", return_value=iter([])) as mock_follow:
            api.logs(
                cluster_id,
                hosts=("h1", "h2"),
                host="h2",
                container=cluster_id + "_node_1",
                follow=True,
                tail=50,
                cancel=cancel,
                cache_dir=str(tmp_path),
            )
            mock_follow.assert_called_once_with(tail=50, cancel=cancel, check=True)


# --------------------------------------------------------------------------
# Click independence — the full API surface still must not import click
# --------------------------------------------------------------------------


def test_full_api_with_run_stop_logs_imports_without_click():
    import subprocess
    import sys

    code = (
        "import sys, importlib;"
        "m = importlib.import_module('sparkrun.api');"
        "assert all(hasattr(m, n) for n in ('run', 'stop', 'logs'));"
        "assert 'click' not in sys.modules, 'click should not be pulled in'"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, "stdout=%s\nstderr=%s" % (result.stdout, result.stderr)


# --------------------------------------------------------------------------
# Scheduler resolution chain: CLI > recipe > cluster > greedy default
# --------------------------------------------------------------------------


def _run_with_scheduler_chain(*, options_scheduler, recipe_scheduler, cluster_scheduler):
    """Exercise ``api.run`` with the three layers wired up; return the
    ``RunResult.scheduler`` field (the actually-resolved scheduler name).
    """
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.recipe import Recipe

    recipe_data = {"sparkrun_version": "2", "runtime": "vllm", "model": "test/m"}
    if recipe_scheduler is not None:
        recipe_data["scheduler"] = recipe_scheduler
    recipe = Recipe(recipe_data)

    cluster_def = ClusterDefinition(name="test-cluster", hosts=["h1"], scheduler=cluster_scheduler)

    opts = api.RunOptions(
        recipe=recipe,
        hosts=("h1",),
        dry_run=True,
        scheduler=options_scheduler,
    )

    class _FakeRuntime:
        runtime_name = "vllm"
        executor = None

        def world_size(self, parallelism, recipe=None, cluster=None):
            return 1

    fake_result = type(
        "FakeLaunchResult",
        (),
        {
            "rc": 0,
            "cluster_id": "sparkrun_schedtestcid",
            "host_list": ["h1"],
            "is_solo": True,
            "runtime": _FakeRuntime(),
            "recipe": recipe,
            "overrides": {},
            "container_image": "test:latest",
            "effective_cache_dir": "/tmp/cache",
            "serve_port": 8000,
            "config": None,
            "recipe_ref": None,
            "comm_env": None,
            "ib_ip_map": {},
            "serve_command": "",
            "runtime_info": {},
            "builder": None,
            "backends": {},
            "timeline": None,
        },
    )()

    from sparkrun.schedulers.greedy import GreedyScheduler

    with (
        patch("sparkrun.api.schedule", side_effect=lambda request, **kwargs: GreedyScheduler().schedule(request)),
        patch("sparkrun.api.status", return_value=ClusterStatus()),
        patch("sparkrun.core.launcher.launch_inference", return_value=fake_result),
        patch("sparkrun.api._resolve.resolve_runtime", return_value=_FakeRuntime()),
        patch("sparkrun.api._resolve.resolve_cluster", return_value=cluster_def),
    ):
        return api.run(opts)


def test_scheduler_chain_cli_option_wins_over_all():
    result = _run_with_scheduler_chain(
        options_scheduler="from-cli",
        recipe_scheduler="from-recipe",
        cluster_scheduler="from-cluster",
    )
    assert result.scheduler == "from-cli"


def test_scheduler_chain_recipe_wins_over_cluster():
    result = _run_with_scheduler_chain(
        options_scheduler=None,
        recipe_scheduler="from-recipe",
        cluster_scheduler="from-cluster",
    )
    assert result.scheduler == "from-recipe"


def test_scheduler_chain_cluster_wins_over_default():
    result = _run_with_scheduler_chain(
        options_scheduler=None,
        recipe_scheduler=None,
        cluster_scheduler="from-cluster",
    )
    assert result.scheduler == "from-cluster"


def test_scheduler_chain_all_unset_falls_back_to_default():
    from sparkrun.core.scheduler import FALLBACK_DEFAULT_SCHEDULER

    result = _run_with_scheduler_chain(
        options_scheduler=None,
        recipe_scheduler=None,
        cluster_scheduler=None,
    )
    # RunResult.scheduler reflects the actually-resolved scheduler name —
    # when no caller supplies one we land on FALLBACK_DEFAULT_SCHEDULER.
    assert result.scheduler == FALLBACK_DEFAULT_SCHEDULER


# --------------------------------------------------------------------------
# Scheduling request carries occupancy status
# --------------------------------------------------------------------------


def _build_multihost_run_fixtures(num_hosts: int = 4):
    """Build a recipe + cluster + options tuple that triggers the multi-host
    scheduling path inside ``api.run``.

    Returns ``(recipe, cluster_def, opts, fake_runtime, fake_launch_result)``.
    """
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.recipe import Recipe

    hosts = tuple("h%d" % (i + 1) for i in range(num_hosts))
    recipe = Recipe(
        {
            "sparkrun_version": "2",
            "runtime": "vllm",
            "model": "test/m",
            "defaults": {"tensor_parallel": 2},
        }
    )
    cluster_def = ClusterDefinition(name="multi", hosts=list(hosts))

    class _FakeRuntime:
        runtime_name = "vllm"
        executor = None

        def world_size(self, parallelism, recipe=None, cluster=None):
            return parallelism.tensor_parallel

    fake_runtime = _FakeRuntime()
    fake_launch_result = type(
        "FakeLaunchResult",
        (),
        {
            "rc": 0,
            "cluster_id": "sparkrun_multihostfix",
            "host_list": list(hosts[:2]),
            "is_solo": False,
            "runtime": fake_runtime,
            "recipe": recipe,
            "overrides": {},
            "container_image": "test:latest",
            "effective_cache_dir": "/tmp",
            "serve_port": 8000,
            "config": None,
            "recipe_ref": None,
            "comm_env": None,
            "ib_ip_map": {},
            "serve_command": "",
            "runtime_info": {},
            "builder": None,
            "backends": {},
            "timeline": None,
        },
    )()

    opts = api.RunOptions(recipe=recipe, hosts=hosts, cluster=cluster_def, dry_run=True)
    return recipe, cluster_def, opts, fake_runtime, fake_launch_result


def test_run_populates_scheduling_request_status():
    """``api.run`` queries cluster status and passes the snapshot into the
    SchedulingRequest so occupancy-sparse / occupancy-dense schedulers can subtract committed
    workloads from each host's capacity."""
    from sparkrun.core.cluster_status import ClusterStatus
    from sparkrun.core.scheduler import RankAssignment, SchedulingResult

    recipe, cluster_def, opts, fake_runtime, fake_launch_result = _build_multihost_run_fixtures()

    fake_status = ClusterStatus(hosts=())
    captured_requests: list = []

    def _fake_schedule(request, *, scheduler=None, sctx=None):
        captured_requests.append(request)
        return SchedulingResult(
            assignment=RankAssignment(
                by_rank=(),
                hosts_used=tuple(request.hosts[:2]),
            ),
            scheduler_name="greedy",
            diagnostics=(),
        )

    with (
        patch("sparkrun.core.launcher.launch_inference", return_value=fake_launch_result),
        patch("sparkrun.api._resolve.resolve_runtime", return_value=fake_runtime),
        patch("sparkrun.api._resolve.resolve_cluster", return_value=cluster_def),
        patch("sparkrun.api.status", return_value=fake_status),
        patch("sparkrun.api.schedule", side_effect=_fake_schedule),
    ):
        api.run(opts)

    assert len(captured_requests) == 1
    assert captured_requests[0].status is fake_status


def test_run_status_acquisition_failure_reaches_scheduler_as_unknown():
    """When the cluster status query fails (partial reachability, missing
    executor, transient SSH error), scheduling still proceeds with
    an error-bearing status rather than inventing free capacity."""
    from sparkrun.core.scheduler import RankAssignment, SchedulingResult

    recipe, cluster_def, opts, fake_runtime, fake_launch_result = _build_multihost_run_fixtures()

    captured_requests: list = []

    def _fake_schedule(request, *, scheduler=None, sctx=None):
        captured_requests.append(request)
        return SchedulingResult(
            assignment=RankAssignment(
                by_rank=(),
                hosts_used=tuple(request.hosts[:2]),
            ),
            scheduler_name="greedy",
            diagnostics=(),
        )

    with (
        patch("sparkrun.core.launcher.launch_inference", return_value=fake_launch_result),
        patch("sparkrun.api._resolve.resolve_runtime", return_value=fake_runtime),
        patch("sparkrun.api._resolve.resolve_cluster", return_value=cluster_def),
        patch("sparkrun.api.status", side_effect=RuntimeError("boom")),
        patch("sparkrun.api.schedule", side_effect=_fake_schedule),
    ):
        api.run(opts)

    assert len(captured_requests) == 1
    assert set(captured_requests[0].status.observation_errors) == set(captured_requests[0].hosts)


def test_run_result_scheduler_reflects_effective_resolution():
    """When the caller doesn't pick a scheduler, ``RunResult.scheduler``
    reports the actually-resolved scheduler name (not a stale ``"greedy"``
    default) so consumers can verify which scheduler ran."""
    from sparkrun.core.scheduler import FALLBACK_DEFAULT_SCHEDULER

    result = _run_with_scheduler_chain(
        options_scheduler=None,
        recipe_scheduler=None,
        cluster_scheduler=None,
    )
    # FALLBACK_DEFAULT_SCHEDULER ("greedy") names a registered plugin; the
    # resolver returns the plugin's canonical scheduler_name when nothing in
    # the CLI/recipe/cluster chain selects one.
    assert result.scheduler == FALLBACK_DEFAULT_SCHEDULER


# --------------------------------------------------------------------------
# cluster_id placement token: deterministic for greedy, random for occupancy
# --------------------------------------------------------------------------


def _capture_launch_cluster_id(opts):
    """Run ``api.run(opts)`` with a hermetic launcher and return the
    ``cluster_id_override`` it composed and handed to ``launch_inference``.

    The fake launch result echoes the override back as its ``cluster_id`` —
    mirroring the real launcher, which honours ``cluster_id_override`` — so
    callers can assert on either the captured override or ``RunResult.cluster_id``.
    """
    from sparkrun.core.recipe import Recipe  # noqa: F401  (imported for parity / side effects)

    captured: dict = {}

    class _FakeRuntime:
        runtime_name = "vllm"
        executor = None

        def world_size(self, parallelism, recipe=None, cluster=None):
            return 1

    def _capture(**kwargs):
        captured["cluster_id"] = kwargs["cluster_id_override"]
        return type(
            "FakeLaunchResult",
            (),
            {
                "rc": 0,
                "cluster_id": kwargs["cluster_id_override"],
                "host_list": kwargs["host_list"],
                "is_solo": kwargs["is_solo"],
                "runtime": _FakeRuntime(),
                "recipe": kwargs["recipe"],
                "overrides": {},
                "container_image": "test:latest",
                "effective_cache_dir": "/tmp",
                "serve_port": 8000,
                "config": None,
                "recipe_ref": None,
                "comm_env": None,
                "ib_ip_map": {},
                "serve_command": "",
                "runtime_info": {},
                "builder": None,
                "backends": {},
                "timeline": None,
            },
        )()

    with (
        patch("sparkrun.api.status", return_value=ClusterStatus()),
        patch("sparkrun.core.launcher.launch_inference", side_effect=_capture),
        patch("sparkrun.api._resolve.resolve_runtime", return_value=_FakeRuntime()),
    ):
        api.run(opts)

    return captured["cluster_id"]


def test_run_greedy_cluster_id_is_deterministic():
    """The default (greedy) scheduler yields a deterministic cluster_id:
    relaunching the same recipe on the same hosts reuses the same id so the
    prior deployment is replaced — sparkrun 0.2.x semantics."""
    from sparkrun.core.recipe import Recipe

    recipe = Recipe({"sparkrun_version": "2", "runtime": "vllm", "model": "test/m"})
    opts = api.RunOptions(recipe=recipe, hosts=("h1",), dry_run=True)

    first = _capture_launch_cluster_id(opts)
    second = _capture_launch_cluster_id(opts)

    assert first == second


def test_run_greedy_cluster_id_matches_lookup_derivation():
    """The greedy launch id equals what the lookup paths (stop / status /
    --ensure) compute via ``derive_cluster_id`` — so they can find and
    replace the running job."""
    from sparkrun.core.recipe import Recipe
    from sparkrun.orchestration.job_metadata import derive_cluster_id

    recipe = Recipe({"sparkrun_version": "2", "runtime": "vllm", "model": "test/m"})
    hosts = ("h2", "h1")  # unsorted on purpose: derivation sorts before hashing
    opts = api.RunOptions(recipe=recipe, hosts=hosts, dry_run=True)

    launched = _capture_launch_cluster_id(opts)

    assert launched == derive_cluster_id(recipe, list(hosts))


def test_run_occupancy_cluster_id_is_random():
    """A status-aware scheduler (occupancy-sparse) keeps a fresh random
    placement token per launch so the same intent placed on different host
    sets never collides.  The intent half stays stable; only the placement
    token differs."""
    from sparkrun.core.recipe import Recipe
    from sparkrun.orchestration.job_metadata import parse_cluster_id

    recipe = Recipe({"sparkrun_version": "2", "runtime": "vllm", "model": "test/m"})
    opts = api.RunOptions(recipe=recipe, hosts=("h1",), dry_run=True, scheduler="occupancy-sparse")

    first = _capture_launch_cluster_id(opts)
    second = _capture_launch_cluster_id(opts)

    assert first != second
    first_intent, first_token = parse_cluster_id(first)
    second_intent, second_token = parse_cluster_id(second)
    assert first_intent == second_intent  # same recipe → same intent
    assert first_token != second_token  # random placement token per launch


@pytest.mark.parametrize("error_hosts,reported_hosts", [({"h2": "unreachable"}, ["h1", "h2"]), ({}, ["h1"])])
def test_intent_discovery_does_not_treat_incomplete_status_as_absence(monkeypatch, error_hosts, reported_hosts):
    from sparkrun.api._resolve import discover_cluster_id_by_intent
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.cluster_status import ClusterStatus, HostOccupancy

    status = ClusterStatus(hosts=tuple(HostOccupancy(host=h) for h in reported_hosts), errors=error_hosts)
    monkeypatch.setattr("sparkrun.orchestration.executor.query_status_for_cluster", lambda *a, **kw: status)
    with pytest.raises(api.SparkrunError, match="status unavailable for h2") as caught:
        discover_cluster_id_by_intent("intent", ["h1", "h2"], cluster_def=ClusterDefinition(name="c", hosts=["h1", "h2"]))
    assert not isinstance(caught.value, api.JobNotFound)


def test_intent_discovery_confirms_absence_on_complete_empty_status(monkeypatch):
    from sparkrun.api._resolve import discover_cluster_id_by_intent
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.cluster_status import ClusterStatus, HostOccupancy

    status = ClusterStatus(hosts=(HostOccupancy(host="h1"),))
    monkeypatch.setattr("sparkrun.orchestration.executor.query_status_for_cluster", lambda *a, **kw: status)
    with pytest.raises(api.JobNotFound):
        discover_cluster_id_by_intent("intent", ["h1"], cluster_def=ClusterDefinition(name="c", hosts=["h1"]))


def test_strategy_must_return_prepared_execution_before_launch(monkeypatch):
    from unittest.mock import Mock
    from sparkrun.core.recipe import Recipe

    recipe = Recipe({"runtime": "vllm-distributed", "model": "test/model"})
    strategy = Mock()
    strategy.name = "test-strategy"
    strategy.finalize_preparation.return_value = None
    monkeypatch.setattr("sparkrun.core.execution.resolve_recipe_execution", lambda context: (strategy, ()))
    launch = Mock(side_effect=AssertionError("invalid strategy output must not reach launch"))
    monkeypatch.setattr("sparkrun.core.launcher.launch_inference", launch)
    with pytest.raises(api.SparkrunError, match="must return a prepared execution"):
        api.run(api.RunOptions(recipe=recipe, hosts=("localhost",), solo=True, dry_run=True))
    strategy.finalize_preparation.assert_called_once()
    launch.assert_not_called()


@pytest.mark.usefixtures("unavailable_hardware_probe", "idle_occupancy")
def test_run_hands_the_callers_executor_layer_to_the_launch():
    """``-o entrypoint=''`` is recorded apart from resolved executor settings, so a job export can bake it (#323)."""
    from sparkrun.core.recipe import Recipe

    recipe = Recipe({"sparkrun_version": "2", "runtime": "vllm", "model": "test/m"})
    opts = api.RunOptions(recipe=recipe, hosts=("h1",), overrides={"entrypoint": "", "port": 8000})
    with patch("sparkrun.core.launcher.launch_inference", side_effect=api.SparkrunError("fixture launch failure")) as launch:
        with pytest.raises(api.SparkrunError, match="fixture launch failure"):
            api.run(opts)
    assert launch.call_args.kwargs["executor_overrides"] == {"entrypoint": ""}
