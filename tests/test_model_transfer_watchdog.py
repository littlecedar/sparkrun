"""Model transfers are bounded by progress, not by elapsed time (#324)."""

from dataclasses import replace
import json
import shlex
import shutil
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from sparkrun.core import model_distribution
from sparkrun.core.config import SparkrunConfig
from sparkrun.core.cluster_manager import ClusterDefinition, ClusterDistributionConfig, ResourceDistributionPrefs
from sparkrun.core.model_distribution import (
    DEFAULT_MODEL_TRANSFER_IDLE_TIMEOUT_S,
    PROVIDER_FALLBACK_TIMEOUT_S,
    ModelCopyRequest,
    ModelTarget,
    ModelTransferResult,
    transfer_timeouts,
)
from sparkrun.models import _stream, acquisition, transport
from sparkrun.models._copy_progress import CopyProgress
from sparkrun.models._stream import ModelTransferStalled, ProgressWatch, execute_stream
from sparkrun.models.artifacts import ModelArtifactError
from sparkrun.models.host_io import ModelHostIO
from sparkrun.transports.session import HostCommandResult
from test_model_artifacts import artifact

POSIX = pytest.mark.skipif(sys.platform == "win32" or not shutil.which("bash"), reason="requires POSIX bash")
SHARD = "model-00001-of-00002.safetensors"
DATA = {SHARD: b"x" * 4096}


def request(**kwargs):
    manifest = artifact(DATA)
    return ModelCopyRequest(manifest, None, "/source", (ModelTarget("a", "/a", "ib-a", tuple(DATA)),), "op", **kwargs)


class FakeClock:
    """Simulated time: each supervisor poll advances it by *step* seconds."""

    def __init__(self, step):
        self.step = step
        self.value = 0.0
        self.lock = threading.Lock()

    def now(self):
        with self.lock:
            return self.value

    def sleep(self, _seconds):
        with self.lock:
            self.value += self.step
        time.sleep(0.002)


class FakeProcess:
    """rsync stand-in printing --info=progress2 lines on a simulated clock."""

    def __init__(self, clock, *, end, advancing):
        self.clock, self.end, self.advancing = clock, end, advancing
        self.cancelled = False
        self.stdin = SimpleNamespace(write=len, close=lambda: None)
        self.stdout = SimpleNamespace(read=self._read)
        self.stderr = SimpleNamespace(read=lambda _size: b"")

    def _done(self):
        return self.cancelled or self.clock.now() >= self.end

    def _read(self, _size):
        time.sleep(0.0005)
        if self._done():
            return b""
        count = int(self.clock.now() * 1000) if self.advancing else 1000
        return b"\r %d 1%% 20.7MB/s 0:00:01\r" % count

    def poll(self):
        return 0 if self._done() else None

    def wait(self):
        return 0

    def cancel(self):
        self.cancelled = True

    def close(self):
        self.cancel()


class FakeSession:
    def __init__(self, process):
        self.process = process

    def open_process(self, host, arguments):
        return self.process


def _stream_with(monkeypatch, clock, process, idle_timeout):
    monkeypatch.setattr(_stream, "time", SimpleNamespace(monotonic=clock.now, sleep=clock.sleep))
    req = request()
    progress = CopyProgress(replace(req, feedback=None), {0: (req.targets[0], 2**40)})
    watch = ProgressWatch(clock=clock.now)
    return execute_stream(
        FakeSession(process),
        "a",
        "rsync",
        timeout=None,
        cancelled=lambda: False,
        stdout=progress.parser(0, watch),
        idle_timeout=idle_timeout,
        watch=watch,
    )


def test_slow_transfer_that_keeps_advancing_outlives_two_hours(monkeypatch):
    clock = FakeClock(step=60)
    process = FakeProcess(clock, end=3 * 60 * 60, advancing=True)
    result = _stream_with(monkeypatch, clock, process, idle_timeout=DEFAULT_MODEL_TRANSFER_IDLE_TIMEOUT_S)
    assert result.returncode == 0
    assert clock.now() >= 3 * 60 * 60
    assert not process.cancelled or clock.now() >= process.end


def test_transfer_whose_bytes_stop_advancing_is_cancelled(monkeypatch):
    # Output keeps arriving (a repeated, unchanged progress line); that is not progress.
    clock = FakeClock(step=60)
    process = FakeProcess(clock, end=float("inf"), advancing=False)
    with pytest.raises(ModelTransferStalled) as raised:
        _stream_with(monkeypatch, clock, process, idle_timeout=DEFAULT_MODEL_TRANSFER_IDLE_TIMEOUT_S)
    assert raised.value.idle >= DEFAULT_MODEL_TRANSFER_IDLE_TIMEOUT_S
    assert raised.value.transferred == 1000
    assert clock.now() < 2 * DEFAULT_MODEL_TRANSFER_IDLE_TIMEOUT_S
    assert process.cancelled


def test_watch_resets_only_on_growing_count():
    now = [0.0]
    watch = ProgressWatch(clock=lambda: now[0])
    now[0] = 10
    watch.advance(0)
    assert watch.idle() == 10
    watch.advance(5)
    assert watch.idle() == 0 and watch.transferred == 5
    now[0] = 20
    watch.advance(5)
    watch.advance(3)
    assert watch.idle() == 10 and watch.transferred == 5


def test_idle_timeout_requires_a_watch():
    with pytest.raises(ValueError, match="progress watch"):
        execute_stream(FakeSession(None), "a", "", timeout=None, cancelled=lambda: False, idle_timeout=1)


@POSIX
def test_real_stream_survives_past_idle_interval_while_advancing():
    # Runs ~3s against a 1s idle interval: well past it in total, while every
    # gap (and the process start, on a loaded CI runner) stays well inside it.
    script = "for i in $(seq 1 30); do printf '\\r %d 1%% 1MB/s\\r' $((i * 100)); sleep 0.1; done\n"
    watch = ProgressWatch()
    req = request()
    progress = CopyProgress(replace(req, feedback=None), {0: (req.targets[0], 10**6)})
    with ModelHostIO() as io:
        result = io.execute(
            None, script, timeout=None, cancelled=lambda: False, stdout=progress.parser(0, watch), idle_timeout=1.0, watch=watch
        )
    assert result.returncode == 0
    assert watch.transferred == 3000


@POSIX
def test_real_stream_stall_kills_the_process():
    script = "while true; do printf '\\r 100 1%% 1MB/s\\r'; sleep 0.05; done\n"
    watch = ProgressWatch()
    req = request()
    progress = CopyProgress(replace(req, feedback=None), {0: (req.targets[0], 10**6)})
    started = time.monotonic()
    with ModelHostIO() as io:
        with pytest.raises(ModelTransferStalled, match="no progress"):
            io.execute(None, script, timeout=None, cancelled=lambda: False, stdout=progress.parser(0, watch), idle_timeout=0.4, watch=watch)
    assert time.monotonic() - started < 10
    assert watch.transferred == 100


@pytest.fixture
def simulated(monkeypatch):
    monkeypatch.setattr(transport, "copy_script", lambda req, target, ssh, **kw: json.dumps([target.host, target.required_files]))


def test_builtin_copy_has_no_default_deadline_and_resumes_a_stall(simulated):
    calls = []

    def execute(host, script, *, timeout, idle_timeout, watch, stdout, **kwargs):
        calls.append((timeout, idle_timeout))
        if len(calls) == 1:
            stdout(b"\r 2048 50% 1MB/s\r")
            raise ModelTransferStalled(600, watch.transferred)
        return HostCommandResult(host, 0)

    result = transport.builtin_copy(request(), SimpleNamespace(execute=execute), {})
    assert result.outcomes == {"a": "complete"}
    assert calls == [(None, DEFAULT_MODEL_TRANSFER_IDLE_TIMEOUT_S)] * 2


def test_builtin_copy_reports_a_persistent_stall(simulated):
    calls = []

    def execute(host, script, *, watch, stdout, **kwargs):
        calls.append(script)
        stdout(b"\r 1024 25% 1MB/s\r")
        raise ModelTransferStalled(612, watch.transferred)

    result = transport.builtin_copy(request(), SimpleNamespace(execute=execute), {})
    assert result.outcomes == {"a": "failed"}
    assert len(calls) == 2  # one stall retry by default
    error = result.errors["a"]
    assert "stalled: no progress for 612s" in error
    assert SHARD in error


def test_stall_retries_are_configurable(simulated):
    calls = []

    def execute(host, script, *, watch, **kwargs):
        calls.append(script)
        raise ModelTransferStalled(600, 0)

    result = transport.builtin_copy(request(config={"model_transfer_stall_retries": 3}), SimpleNamespace(execute=execute), {})
    assert result.outcomes == {"a": "failed"}
    assert len(calls) == 4


def test_rsync_own_timeout_exit_is_a_stall(simulated):
    codes = [30, 0]

    def execute(host, script, **kwargs):
        return HostCommandResult(host, codes.pop(0), stderr=b"rsync error: timeout in data send/receive (code 30)")

    result = transport.builtin_copy(request(), SimpleNamespace(execute=execute), {})
    assert result.outcomes == {"a": "complete"}
    assert codes == []


def test_explicit_ceiling_still_applies(simulated):
    def execute(host, script, *, timeout, cancelled, **kwargs):
        assert timeout is not None and timeout <= 0.2
        end = time.monotonic() + timeout
        while time.monotonic() < end and not cancelled():
            time.sleep(0.01)
        raise ModelArtifactError("model transfer deadline exceeded")

    result = transport.builtin_copy(request(timeout=0.2, config={"model_transfer_stall_retries": 0}), SimpleNamespace(execute=execute), {})
    assert result.outcomes == {"a": "failed"}
    assert "deadline exceeded" in result.errors["a"]


def test_copy_script_carries_rsync_idle_timeout():
    req = request()
    argv = shlex.split(transport.copy_script(req, req.targets[0], {}).splitlines()[-1])
    assert "--timeout=600" in argv
    argv = shlex.split(transport.copy_script(replace(req, idle_timeout=90.5), req.targets[0], {}).splitlines()[-1])
    assert "--timeout=91" in argv


def test_transfer_timeouts_defaults_and_validation():
    assert transfer_timeouts({}) == model_distribution.TransferTimeouts(600, None, 1)
    assert transfer_timeouts({"model_transfer_timeout_s": 3600, "model_transfer_idle_timeout_s": 120}).timeout == 3600
    for key, value in (
        ("model_transfer_idle_timeout_s", 0),
        ("model_transfer_idle_timeout_s", -5),
        ("model_transfer_idle_timeout_s", "600"),
        ("model_transfer_idle_timeout_s", True),
        ("model_transfer_timeout_s", -1),
        ("model_transfer_timeout_s", float("inf")),
        ("model_transfer_stall_retries", -1),
        ("model_transfer_stall_retries", 1.5),
    ):
        with pytest.raises(ModelArtifactError, match=key):
            transfer_timeouts({key: value})


def test_request_validation():
    assert request().timeout is None
    with pytest.raises(ModelArtifactError, match="idle timeout"):
        request(idle_timeout=0)
    with pytest.raises(ModelArtifactError, match="finite"):
        request(timeout=float("inf"))


def test_cluster_distribution_overrides_reach_transfer_settings():
    prefs = ResourceDistributionPrefs.from_dict({"idle_timeout_s": 300, "timeout_s": 7200, "stall_retries": 2})
    assert prefs.to_dict() == {"idle_timeout_s": 300, "timeout_s": 7200, "stall_retries": 2}
    assert not prefs.is_default()
    cluster = ClusterDefinition(name="c", hosts=["a"], distribution=ClusterDistributionConfig(model=prefs))
    scoped = SparkrunConfig().for_cluster(cluster)
    assert transfer_timeouts(scoped) == model_distribution.TransferTimeouts(300, 7200, 2)


def test_provider_without_ceiling_receives_large_finite_timeout(monkeypatch):
    seen = []

    class Provider:
        def copy(self, req):
            seen.append(req.timeout)
            return ModelTransferResult(req.manifest.identity, {"a": "complete"})

    monkeypatch.setitem(model_distribution._PROVIDERS, "example", Provider())
    req = request(config={"model_distribution_provider": "example"})
    model_distribution.try_model_transfer(req)
    model_distribution.try_model_transfer(replace(req, timeout=900))
    assert seen == [PROVIDER_FALLBACK_TIMEOUT_S, 900]


def test_repair_download_stall_names_host_and_progress(monkeypatch):
    from test_hub_adapter import _tiny_manifest

    req, selected = _tiny_manifest()

    def execute(host, script, *, timeout, idle_timeout, watch, stdout, **kwargs):
        assert timeout is None and idle_timeout == 600
        stdout(b"SPARKRUN_MODEL_ADAPTER_READY\nSPARKRUN_MODEL_PROGRESS 2147483648 4294967296\n")
        raise ModelTransferStalled(601, watch.transferred)

    with pytest.raises(ModelArtifactError, match=r"repair on h1 stalled: no download progress for 601s \(2.00 GiB"):
        acquisition.run_adapter(
            SimpleNamespace(execute=execute), "h1", req, "/cache", token=None, manifest=selected, files=("config.json",)
        )


def test_repair_bootstrap_stall_is_named_as_such():
    from test_hub_adapter import _tiny_manifest

    req, selected = _tiny_manifest()

    def execute(host, script, *, watch, **kwargs):
        raise ModelTransferStalled(600, 0)

    with pytest.raises(ModelArtifactError, match="download adapter .* did not start within 600s"):
        acquisition.run_adapter(
            SimpleNamespace(execute=execute), "h1", req, "/cache", token=None, manifest=selected, files=("config.json",)
        )


def test_adapter_ready_restarts_the_idle_clock():
    now = [0.0]
    watch = ProgressWatch(clock=lambda: now[0])
    progress = acquisition.RepairProgress("h1", watch=watch)
    now[0] = 500
    progress.feed(b"SPARKRUN_MODEL_ADAPTER_READY\n")
    assert progress.ready and watch.idle() == 0


def test_watch_ignores_finalize_and_unarmed_windows():
    now = [0.0]
    complete = ProgressWatch(expected=100, clock=lambda: now[0])
    complete.advance(100)
    now[0] = 10_000
    assert complete.idle() == 0  # receiver flushing/closing after the last byte
    resume = ProgressWatch(armed=False, clock=lambda: now[0])
    now[0] = 20_000
    assert resume.idle() == 0  # checksumming the partial file
    resume.advance(1)
    now[0] = 20_050
    assert resume.idle() == 50


def test_resume_attempt_watch_waits_for_first_byte(simulated):
    watches = []

    def execute(host, script, *, watch, **kwargs):
        watches.append((watch.armed, watch.expected))
        if len(watches) == 1:
            raise ModelTransferStalled(600, 0)
        return HostCommandResult(host, 0)

    assert transport.builtin_copy(request(), SimpleNamespace(execute=execute), {}).outcomes == {"a": "complete"}
    assert watches == [(True, 4096), (False, 4096)]
