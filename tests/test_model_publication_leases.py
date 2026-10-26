"""Real lease loss, lock-owned publication and cancellation; temporary data only."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import base64
import os
import signal
import threading
import time
from unittest.mock import Mock

import pytest

pytest.importorskip("fcntl")

from sparkrun.models import _cache_agent, _publication, usage
from sparkrun.models.artifacts import ModelArtifactError
from sparkrun.models.host_io import ModelHostIO
from sparkrun.models.preparation import prepare_model
from sparkrun.transports.session import SshHostSession
from sparkrun.utils.shell import quote
from test_model_cache_retention import tracked


def kill_holder(io):
    holders = tuple(io._owners.values())
    assert holders
    for owner in holders:
        os.killpg(owner.process.process.pid, signal.SIGKILL)
        owner.process.wait(timeout=5)


@pytest.mark.parametrize("alias", ["localhost", "127.0.0.1"])
def test_lost_lease_is_terminal_including_shared_aliases(tmp_path, alias):
    tracked(tmp_path)
    with pytest.raises(ModelArtifactError, match="lease lost"), ModelHostIO() as io:
        io.lock("localhost", str(tmp_path), "org/model")
        kill_holder(io)
        assert io.cancelled()
        with pytest.raises(ModelArtifactError, match="lease lost"):
            io.lock(alias, str(tmp_path), "org/model")
        with pytest.raises(ModelArtifactError, match="lease lost"):
            io.release_leases()
        with pytest.raises(ModelArtifactError, match="lease lost"):
            io.publish(alias, str(tmp_path), "org/model", str(tmp_path / "invalid"), "must not publish")
        assert not (tmp_path / "invalid").exists()
        assert _cache_agent.run(str(tmp_path), action="prune", dry_run=False, min_age_seconds=0)["entries"][0]["action"] == "removed"
        # Even catching an intermediate error cannot turn this into success.


def test_preparation_never_publishes_binding_after_lease_loss(tmp_path, monkeypatch):
    manifest, snapshot, _ = tracked(tmp_path)
    original = usage.record_use
    published = Mock()
    monkeypatch.setattr("sparkrun.models.preparation.record_binding", published)

    def lose_before_claim(io, host, root, artifact, **kwargs):
        kill_holder(io)
        report = _cache_agent.run(str(tmp_path), action="prune", dry_run=False, min_age_seconds=0)
        assert report["entries"][0]["action"] == "removed"
        return original(io, host, root, artifact, **kwargs)

    monkeypatch.setattr(usage, "record_use", lose_before_claim)
    with usage.usage_operation():
        usage.begin_workload("fixture", {})
        with pytest.raises(ModelArtifactError, match="lease lost"):
            prepare_model("org/model", ["localhost"], cache_dir=str(tmp_path), manifest=manifest, offline=True)
        assert not usage.workload_claims()
    assert not snapshot.exists()
    published.assert_not_called()


def test_loss_cancels_inflight_copy_without_closing_borrowed_session(tmp_path):
    ready = threading.Event()
    session = SshHostSession()
    unrelated = session.open_process("localhost", ["sleep", "60"])
    try:
        with pytest.raises(ModelArtifactError, match="lease lost"), ModelHostIO(session=session) as io:
            io.lock("localhost", str(tmp_path), "org/model")
            with ThreadPoolExecutor(1) as pool:
                future = pool.submit(io.execute, "localhost", "printf ready; sleep 60", timeout=10, stdout=lambda _: ready.set())
                assert ready.wait(5)
                kill_holder(io)
                with pytest.raises(ModelArtifactError, match="lease lost"):
                    future.result(timeout=5)
            assert unrelated.poll() is None
            assert not session._closed
        assert unrelated.poll() is None
        assert session.execute("localhost", ["true"]).returncode == 0
    finally:
        session.close()


def test_publication_child_keeps_flock_if_holder_dies(tmp_path):
    _, snapshot, _ = tracked(tmp_path)
    ready, release, written = (tmp_path / name for name in ("ready", "release", "written"))
    with pytest.raises(ModelArtifactError, match="lease lost"), ModelHostIO() as io:
        nonce = io.lock("localhost", str(tmp_path), "org/model")
        holder = io._owners[nonce].process
        # Pause a real command inside the publication channel. FD ownership must
        # continue in this child even if only the supervising holder exits.
        script = (
            f"touch {quote(str(ready))}; while [ ! -e {quote(str(release))} ]; do sleep .01; done; printf done > {quote(str(written))}\n"
        )
        holder.stdin.write(base64.b64encode(script.encode()) + b"\n")
        holder.stdin.flush()
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists()
        os.kill(holder.process.pid, signal.SIGKILL)
        holder.wait(timeout=5)
        report = _cache_agent.run(str(tmp_path), action="prune", dry_run=False, min_age_seconds=0)
        assert report["entries"][0]["reason_code"] == "cache_busy"
        assert snapshot.exists()
        release.touch()
        while not written.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert written.read_text() == "done"


def test_publication_translates_shared_mount_alias_and_preserves_text(tmp_path):
    cache = tmp_path / "actual"
    cache.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(cache, target_is_directory=True)
    name = "metadata 'quoted'\n.json"
    contents = 'é\n{"value": "$HOME `false`"}\n' * 10000
    with ModelHostIO() as io:
        first = io.lock("localhost", str(cache), "org/model")
        assert io.lock("127.0.0.1", str(alias), "org/model") == first
        io.publish("127.0.0.1", str(alias), "org/model", str(alias / name), contents)
        assert (cache / name).read_text(encoding="utf-8") == contents
        assert len(io._owners) == 1


def test_provider_gets_lease_loss_cancellation(tmp_path, monkeypatch):
    from sparkrun.models import preparation
    from sparkrun.core.model_distribution import ModelTransferResult

    manifest, snapshot, _ = tracked(tmp_path)
    (snapshot / "model.safetensors").unlink()
    session = SshHostSession()
    observed = []

    def transfer(request, **kwargs):
        assert not request.cancelled()
        for holder in tuple(session._processes):
            os.killpg(holder.pid, signal.SIGKILL)
            holder.wait(timeout=5)
        observed.append(request.cancelled())
        return ModelTransferResult(manifest.identity, {"localhost": "complete"})

    monkeypatch.setattr(preparation, "try_model_transfer", transfer)
    try:
        with pytest.raises(ModelArtifactError, match="lease lost"):
            prepare_model("org/model", ["localhost"], cache_dir=str(tmp_path), manifest=manifest, transfer_mode="pull", session=session)
        assert observed == [True]
        assert not session._closed
    finally:
        session.close()


def test_local_acquisition_discards_staged_file_after_cancellation(tmp_path, monkeypatch):
    from pathlib import Path
    from sparkrun.models.acquisition import repair_snapshot_files

    manifest, snapshot, _ = tracked(tmp_path)
    original = (snapshot / "model.safetensors").read_bytes()
    cancelled = threading.Event()

    def download(repo, name, **kwargs):
        target = Path(kwargs["local_dir"]) / name
        target.write_bytes(b"replacement")
        cancelled.set()
        return str(target)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    with pytest.raises(RuntimeError, match="acquisition cancelled"):
        repair_snapshot_files(
            manifest.repo, manifest.revision, str(tmp_path), manifest.endpoint, None, ("model.safetensors",), cancelled=cancelled.is_set
        )
    assert (snapshot / "model.safetensors").read_bytes() == original
    assert not list(snapshot.parent.glob(".sparkrun-repair-*"))


def test_lease_supervision_preserves_large_metadata_reads(tmp_path):
    # Using the transfer supervisor for finite host IO must not apply its tail
    # retention cap to full manifests or usage metadata.
    contents = "x" * (17 * 1024**2)
    path = tmp_path / "large.json"
    path.write_text(contents)
    with ModelHostIO() as io:
        io.lock("localhost", str(tmp_path), "org/model")
        assert io.read("localhost", str(path)) == contents


def wait_until(predicate):
    deadline = time.monotonic() + 5
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert predicate()


def test_contended_acquisition_reports_holder_and_keeps_waiting(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(_publication, "_WAIT_INTERVAL", 0.05)
    caplog.set_level(_publication.PROGRESS, logger=_publication.__name__)
    with ModelHostIO() as holder, ThreadPoolExecutor(1) as pool:
        holder.lock(None, str(tmp_path), "org/model")

        def acquire():
            with ModelHostIO() as contender:
                contender.lock(None, str(tmp_path), "org/model")
                contender.publish(None, str(tmp_path), "org/model", str(tmp_path / "published"), "ready")

        future = pool.submit(acquire)
        try:
            wait_until(lambda: caplog.text.count("Waiting for model cache publication lease") >= 3)
            assert not future.done()
            assert f"pid {os.getpid()} operation {holder._operation}" in caplog.text
        finally:
            holder.release_leases()
        future.result(timeout=5)
    assert (tmp_path / "published").read_text() == "ready"


def test_acquisition_failure_is_not_reported_as_lost_lease(tmp_path):
    root = tmp_path / "not-a-directory"
    root.write_text("file")
    with closing(SshHostSession()) as session:
        with pytest.raises(ModelArtifactError, match="could not acquire") as exc:
            _publication.PublicationLease(session, None, str(root), str(root / "repo.lock"), "nonce")
        assert "lease lost" not in str(exc.value)
        assert not session._processes


def test_waiting_acquisition_is_cancellable_without_affecting_holder(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(_publication, "_WAIT_INTERVAL", 0.05)
    caplog.set_level(_publication.PROGRESS, logger=_publication.__name__)
    cancel = threading.Event()

    def interrupt():
        if cancel.is_set():
            raise KeyboardInterrupt

    with ModelHostIO() as holder, closing(SshHostSession()) as session, ThreadPoolExecutor(1) as pool:
        nonce = holder.lock(None, str(tmp_path), "org/model")
        path = tmp_path / _cache_agent.STATE / "org--model.lock"
        future = pool.submit(_publication.PublicationLease, session, None, str(tmp_path), str(path), "contender", check=interrupt)
        try:
            wait_until(lambda: "Waiting for model cache publication lease" in caplog.text)
        finally:
            cancel.set()
        with pytest.raises(KeyboardInterrupt):
            future.result(timeout=5)
        holder.check_leases()
        assert holder.lock("localhost", str(tmp_path), "org/model") == nonce
        assert not session._processes
        assert session.execute("localhost", ["true"]).returncode == 0


def test_waiting_acquisition_supervises_previously_acquired_leases(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(_publication, "_WAIT_INTERVAL", 0.05)
    caplog.set_level(_publication.PROGRESS, logger=_publication.__name__)
    with ModelHostIO() as blocker, ThreadPoolExecutor(1) as pool:
        blocker.lock(None, str(tmp_path / "second"), "org/model")
        with pytest.raises(ModelArtifactError, match="lease lost"), ModelHostIO() as contender:
            contender.lock(None, str(tmp_path / "first"), "org/model")
            future = pool.submit(contender.lock, None, str(tmp_path / "second"), "org/model")
            try:
                wait_until(lambda: "Waiting for model cache publication lease" in caplog.text)
                kill_holder(contender)
                with pytest.raises(ModelArtifactError, match="lease lost"):
                    future.result(timeout=5)
                assert not contender.local._processes - {owner.process.process for owner in contender._owners.values()}
            finally:
                blocker.release_leases()


def test_alias_that_cannot_yet_see_our_nonce_reuses_the_lease(tmp_path, monkeypatch):
    # On NFS the second name may read the lock file before the holder's nonce
    # is visible there; it then flocks against our own lease. It must
    # recognize the holder as itself instead of waiting forever.
    monkeypatch.setattr(_publication, "_WAIT_INTERVAL", 0.05)
    cache = tmp_path / "actual"
    cache.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(cache, target_is_directory=True)
    with ModelHostIO() as io:
        first = io.lock("localhost", str(cache), "org/model")
        assert (cache / _cache_agent.STATE / "org--model.lock").read_text().splitlines()[0] == first
        original = io.read
        monkeypatch.setattr(io, "read", lambda host, path: None if path.endswith(".lock") else original(host, path))
        assert io.lock("127.0.0.1", str(alias), "org/model") == first
        assert len(io._owners) == 1
        io.publish("127.0.0.1", str(alias), "org/model", str(alias / "published"), "ready")
    assert (cache / "published").read_text() == "ready"


def test_busy_lease_wait_gives_up_after_the_configured_ceiling(tmp_path, monkeypatch):
    monkeypatch.setattr(_publication, "_WAIT_INTERVAL", 0.05)
    monkeypatch.setattr("sparkrun.core.model_distribution.lease_timeout", lambda config=None: 0.2)
    with ModelHostIO() as holder:
        holder.lock(None, str(tmp_path), "org/model")
        with ModelHostIO() as contender:
            with pytest.raises(ModelArtifactError, match="model_cache_lease_timeout_s") as exc:
                contender.lock(None, str(tmp_path), "org/model")
            assert f"operation {holder._operation}" in str(exc.value)
            assert not contender._owners
        holder.check_leases()


def test_lease_timeout_setting():
    from sparkrun.core.cluster_manager import ClusterDefinition, ClusterDistributionConfig, ResourceDistributionPrefs
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.model_distribution import DEFAULT_MODEL_CACHE_LEASE_TIMEOUT_S, lease_timeout

    assert lease_timeout(SparkrunConfig()) == DEFAULT_MODEL_CACHE_LEASE_TIMEOUT_S
    assert lease_timeout({"model_cache_lease_timeout_s": 0}) is None
    assert lease_timeout({"model_cache_lease_timeout_s": 90}) == 90
    for bad in (-1, float("inf"), "1h", True):
        with pytest.raises(ModelArtifactError, match="model_cache_lease_timeout_s"):
            lease_timeout({"model_cache_lease_timeout_s": bad})
    prefs = ResourceDistributionPrefs.from_dict({"lease_timeout_s": 600})
    assert prefs.to_dict() == {"lease_timeout_s": 600} and not prefs.is_default()
    cluster = ClusterDefinition(name="c", hosts=["a"], distribution=ClusterDistributionConfig(model=prefs))
    assert lease_timeout(SparkrunConfig().for_cluster(cluster)) == 600
