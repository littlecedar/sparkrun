"""Concurrent preparations use real cache locks, tiny payloads, and local hosts."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading

import pytest

pytest.importorskip("fcntl")

from sparkrun.core.model_distribution import ModelTransferResult
from sparkrun.models import _cache_agent, _publication, preparation, usage
from sparkrun.models.host_io import ModelHostIO
from test_model_artifacts import cache, DATA
from test_model_cache_retention import tracked
from test_model_publication_leases import wait_until


@pytest.fixture
def local_hosts(tmp_path, monkeypatch):
    locations = {host: tmp_path / (host or "control") for host in (None, "first", "second", "localhost")}

    class LocalHosts(ModelHostIO):
        def root(self, host, path):
            return str(locations[host])

        def execute(self, host, script, **kwargs):
            return super().execute(None, script, **kwargs)

        def lock(self, host, root, repo):
            return super().lock(None, root, repo)

    monkeypatch.setattr(preparation, "ModelHostIO", LocalHosts)
    return locations, LocalHosts


def copy_files(request):
    source = Path(request.source_cache_root) / request.manifest.relative_snapshot
    for target in request.targets:
        snapshot = Path(target.cache_root) / request.manifest.relative_snapshot
        for name in target.required_files:
            path = snapshot / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((source / name).read_bytes())
    return ModelTransferResult(request.manifest.identity, {target.host: "complete" for target in request.targets})


@pytest.mark.parametrize("mode", ["local", "push"])
def test_warm_launch_finishes_during_cold_copy_and_cold_contender_waits(local_hosts, monkeypatch, caplog, mode):
    locations, _ = local_hosts
    for host, root in locations.items():
        manifest, snapshot, _ = tracked(root)
        if host in {"first", "second"}:
            (snapshot / "model.safetensors").unlink()
    copying, release = threading.Event(), threading.Event()
    monkeypatch.setattr(_publication, "_WAIT_INTERVAL", 0.05)
    caplog.set_level(_publication.PROGRESS, logger=_publication.__name__)

    def transfer(request, **kwargs):
        assert not request.cancelled()
        if request.targets[0].host == "first":
            copying.set()
            assert release.wait(30)
        return copy_files(request)

    monkeypatch.setattr(preparation, "try_model_transfer", transfer)

    def prepare(host):
        with usage.usage_operation():
            if host == "localhost":
                usage.begin_workload("warm-launch", {})
            result = preparation.prepare_model(
                "org/model", [host], cache_dir="/hf", manifest=manifest, transfer_mode=mode, offline=True, checksums=True
            )
            if host == "localhost":
                assert usage.workload_claims()
            return result

    with ThreadPoolExecutor(3) as pool:
        first = pool.submit(prepare, "first")
        try:
            assert copying.wait(10)
            # Source and target remain protected, including force-delete, for
            # the entire provider callback while the warm launch can proceed.
            for host in (None, "first"):
                root = str(locations[host])
                assert _cache_agent.run(root, action="prune", dry_run=False, min_age_seconds=0)["entries"][0]["reason_code"] == "cache_busy"
                assert (
                    _cache_agent.run(root, action="delete", repo="org/model", force=True, dry_run=False)["entries"][0]["reason_code"]
                    == "cache_busy"
                )
            warm = pool.submit(prepare, "localhost").result(timeout=10)
            assert warm.bindings[0].report.complete
            assert not first.done()
            second = pool.submit(prepare, "second")
            wait_until(lambda: f"({locations[None]}); held by controller" in caplog.text)
            assert not second.done()
        finally:
            release.set()
        assert first.result(timeout=10).bindings[0].report.complete
        assert second.result(timeout=10).bindings[0].report.complete
    for root in locations.values():
        for name, data in DATA.items():
            assert (root / manifest.relative_snapshot / name).read_bytes() == data


def test_targets_are_rechecked_after_releasing_leases_to_add_source(local_hosts, monkeypatch):
    locations, local_io = local_hosts
    for host in (None, "first", "second"):
        manifest, snapshot, _ = tracked(locations[host])
        if host == "second":
            (snapshot / "model.safetensors").unlink()
    # One cache reached through two target names must still have one writer.
    locations["alias"] = locations["first"]
    released, copied = [], []

    class ChangedBetweenPhases(local_io):
        def release_leases(self):
            super().release_leases()
            released.append(True)
            report = _cache_agent.run(str(locations["first"]), action="prune", dry_run=False, min_age_seconds=0)
            assert report["entries"][0]["action"] == "removed"
            cache(locations["second"], manifest)  # Another preparation finished.

    def transfer(request, **kwargs):
        copied.extend(target.host for target in request.targets)
        return copy_files(request)

    monkeypatch.setattr(preparation, "ModelHostIO", ChangedBetweenPhases)
    monkeypatch.setattr(preparation, "try_model_transfer", transfer)
    result = preparation.prepare_model(
        "org/model", ["first", "alias", "second"], cache_dir="/hf", manifest=manifest, offline=True, checksums=True
    )
    assert released == [True]
    assert copied == ["first"]
    assert all(binding.report.complete for binding in result.bindings)


@pytest.mark.parametrize("scenario", ["reuse-blobs", "push-from-head"])
def test_target_reuse_does_not_wait_for_busy_control_cache(local_hosts, monkeypatch, scenario):
    locations, _ = local_hosts
    manifest, first, _ = tracked(locations["first"])
    targets = ["first"]
    copied = []
    if scenario == "reuse-blobs":
        # Only HF blobs exist: the reuse pass must run before deciding whether
        # the control source is needed, even though observation would be cold.
        for path in first.iterdir():
            path.unlink()
        cache(locations["first"], manifest, shared=True)
        for path in first.iterdir():
            path.unlink()
    else:
        targets.append("second")

    def transfer(request, **kwargs):
        copied.append((request.source_host, [target.host for target in request.targets]))
        return copy_files(request)

    monkeypatch.setattr(preparation, "try_model_transfer", transfer)
    with ModelHostIO() as blocker, ThreadPoolExecutor(1) as pool:
        blocker.lock(None, str(locations[None]), "org/model")
        future = pool.submit(
            preparation.prepare_model, "org/model", targets, cache_dir="/hf", manifest=manifest, offline=True, transfer_mode="push"
        )
        try:
            result = future.result(timeout=10)
        finally:
            blocker.release_leases()
    assert all(binding.report.complete for binding in result.bindings)
    assert copied == ([("first", ["second"])] if scenario == "push-from-head" else [])
