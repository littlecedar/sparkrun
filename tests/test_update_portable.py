"""Destination-aware installation and native tool entry point regressions."""

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

from click.testing import CliRunner
import pytest

from sparkrun.cli import _self_update as update
from sparkrun.core import runtime_lease as leases
from sparkrun.core.config import SparkrunConfig
from sparkrun.core.process_identity import process_identity
from sparkrun.utils.fs import file_lock


@pytest.mark.parametrize("platform,directory,suffix", [("linux", "bin", ""), ("darwin", "bin", ""), ("win32", "Scripts", ".exe")])
def test_application_executables_follow_target_platform(monkeypatch, tmp_path, platform, directory, suffix):
    from test_application_profiles import alternate

    alternate()
    prefix = tmp_path / "tool with spaces"
    monkeypatch.setattr(update, "sys", SimpleNamespace(prefix=str(tmp_path / "uvx"), platform=platform))
    monkeypatch.setattr(
        update.subprocess, "run", Mock(return_value=SimpleNamespace(returncode=0, stdout=f"alternate-test v1 ({prefix})\n"))
    )
    assert update.installed_tool_executable("uv") == prefix / directory / ("alternate-test" + suffix)
    assert update.owning_executable() == str(tmp_path / "uvx" / directory / ("alternate-test" + suffix))
    assert update.owning_interpreter() == str(tmp_path / "uvx" / directory / ("python" + suffix))
    assert update.installation_prefix("uv") == prefix
    assert not update.is_uv_tool_install("uv")


def test_first_install_uses_owning_distribution_destination(monkeypatch, tmp_path):
    from sparkrun.core.application_profile import ApplicationProfile, select_application_profile

    select_application_profile(ApplicationProfile(id="alternate", display_name="Alternate", command="alternate", package="Acme_Tool"))
    monkeypatch.setattr(
        update.subprocess,
        "run",
        Mock(
            side_effect=[
                SimpleNamespace(returncode=0, stdout=""),
                SimpleNamespace(returncode=0, stdout=str(tmp_path) + "\n"),
            ]
        ),
    )
    assert update.installation_prefix("uv") == tmp_path / "acme-tool"


@pytest.mark.parametrize(
    "failure", [OSError("missing uv"), subprocess.TimeoutExpired("uv", 15), SimpleNamespace(returncode=1, stdout="", stderr="locked")]
)
def test_destination_discovery_failure_does_not_authorize_install(monkeypatch, failure):
    run = Mock(side_effect=failure) if isinstance(failure, Exception) else Mock(return_value=failure)
    monkeypatch.setattr(update.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="destination uv environment"):
        update.installation_prefix("uv")
    assert run.call_count == 1


def test_forced_install_from_uvx_honors_destination_worker_lease(monkeypatch, tmp_path):
    from sparkrun.cli import main
    from sparkrun.cli._setup import _commands

    destination = tmp_path / "installed tool"
    environment = leases.environment_state(destination)
    (environment / "runtime-leases").mkdir(exist_ok=True)
    record = environment / "runtime-leases" / "test-worker.json"
    monkeypatch.setattr(update, "sys", SimpleNamespace(prefix=str(tmp_path / "uvx"), platform=sys.platform))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("shutil.which", lambda name: "uv" if name == "uv" else None)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    installs = []
    try:
        identity = process_identity(child.pid)
        assert identity
        record.write_text(json.dumps({"process": identity, "kind": "worker", "config": str(tmp_path / "other/config.yaml")}))

        def run(argv, **kwargs):
            if argv[1:3] == ["tool", "list"]:
                return SimpleNamespace(returncode=0, stdout=f"sparkrun v0.4.0 ({destination})\n")
            if argv[1:3] == ["tool", "install"]:
                # The same environment lock must cover the entire replacement.
                with pytest.raises(BlockingIOError), file_lock(environment / "runtime.lock", blocking=False):
                    pass
                installs.append(argv)
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            raise AssertionError(argv)

        args = ["setup", "install", "--alpha", "--no-update-registries", "--shell", "bash"]
        with patch("subprocess.run", side_effect=run), patch.object(_commands, "_install_completion", Mock()):
            result = CliRunner().invoke(main, args)
            assert result.exit_code != 0
            assert "draining workers" in result.output
            assert not installs
            child.terminate()
            child.wait(timeout=10)
            result = CliRunner().invoke(main, args)
            assert result.exit_code == 0, result.output
            assert len(installs) == 1
            assert not record.exists()  # Dead leases do not prevent a retry.
    finally:
        if child.poll() is None:
            child.terminate()
        child.wait(timeout=10)
        record.unlink(missing_ok=True)


def test_native_entry_point_with_spaces_can_be_executed(tmp_path):
    """Install a dependency-free fixture wheel; native Windows executes its .exe."""
    import venv
    import shutil
    import zipfile

    prefix = tmp_path / "environment with spaces"
    venv.EnvBuilder().create(prefix)
    interpreter = update.environment_executable(prefix, "python")
    wheel = tmp_path / "entry_probe-1.0-py3-none-any.whl"
    info = "entry_probe-1.0.dist-info/"
    files = {
        "entry_probe.py": "def main():\n    print(42)\n",
        info + "METADATA": "Metadata-Version: 2.1\nName: entry-probe\nVersion: 1.0\n",
        info + "WHEEL": "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        info + "entry_points.txt": "[console_scripts]\nalternate-command = entry_probe:main\n",
    }
    files[info + "RECORD"] = "".join(f"{name},,\n" for name in [*files, info + "RECORD"])
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    uv = shutil.which("uv")
    assert uv, "uv is required for native tool entry point tests"
    subprocess.run(
        [uv, "pip", "install", "--offline", "--no-deps", "--python", str(interpreter), str(wheel)],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    command = update.environment_executable(prefix, "alternate-command")
    result = subprocess.run([str(command)], capture_output=True, text=True, check=True, timeout=10)
    assert result.stdout.strip() == "42"


def test_post_install_registry_failure_is_nonfatal(monkeypatch, tmp_path):
    from sparkrun.cli import main
    from sparkrun.cli._setup import _commands

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("shutil.which", lambda name: "uv" if name == "uv" else None)
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        if argv[1:3] == ["tool", "list"]:
            return SimpleNamespace(returncode=0, stdout=f"sparkrun v1 ({tmp_path / 'installed'})\n")
        if argv[1:3] == ["registry", "update"]:
            raise FileNotFoundError("entry point missing")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with patch("subprocess.run", side_effect=run), patch.object(_commands, "_install_completion", Mock()):
        result = CliRunner().invoke(main, ["setup", "install", "--shell", "bash"])
    assert result.exit_code == 0, result.output
    assert "registry update failed or timed out (non-fatal)" in result.output
    argv, kwargs = next(c for c in calls if c[0][1:3] == ["registry", "update"])
    assert Path(argv[0]) == update.environment_executable(tmp_path / "installed", "sparkrun")
    assert kwargs["env"]["SPARKRUN_APPLICATION_CONFIG"] == str(SparkrunConfig().config_path)
