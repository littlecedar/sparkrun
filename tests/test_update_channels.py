"""Tests for update-channel selection, version display, and self-update wiring."""

from __future__ import annotations

from unittest import mock

import click
import pytest
from click.testing import CliRunner

from sparkrun.cli import main
from sparkrun.cli._self_update import (
    owning_executable,
    channel_from_flags,
    describe_change,
    identity_changed,
    update_argv,
)
from sparkrun.core import channels
from sparkrun.core.config import SparkrunConfig
from sparkrun.core.version import display_version


# --- channel vocabulary ---


def test_normalize_channel_defaults_and_aliases():
    assert channels.normalize_channel(None) == "stable"
    assert channels.normalize_channel("") == "stable"
    assert channels.normalize_channel("bogus") == "stable"
    assert channels.normalize_channel("BETA") == "beta"
    assert channels.normalize_channel("yolo") == "alpha"
    assert channels.normalize_channel("alpha") == "alpha"


def test_channel_requirements_map():
    assert channels.channel_requirement("stable") == "sparkrun"
    assert channels.channel_requirement("beta").endswith("@develop")
    assert channels.channel_requirement("alpha").endswith("@develop-next")
    assert channels.channel_requirement("yolo").endswith("@develop-next")


def test_is_git_channel():
    assert not channels.is_git_channel("stable")
    assert channels.is_git_channel("beta")
    assert channels.is_git_channel("alpha")


def test_channel_suffix():
    assert channels.channel_suffix("stable") == ""
    assert channels.channel_suffix("beta") == "-beta"
    assert channels.channel_suffix("alpha") == "-alpha"


# --- config accessors ---


def test_config_channel_defaults_stable(tmp_path):
    assert SparkrunConfig(tmp_path / "config.yaml").self_update_channel == "stable"


def test_config_set_channel_persists(tmp_path):
    path = tmp_path / "config.yaml"
    SparkrunConfig(path).set_self_update_channel("beta")
    reloaded = SparkrunConfig(path)
    assert reloaded.self_update_channel == "beta"
    assert reloaded.get("self_update.source") == "git"
    assert reloaded.get("self_update.requirement").endswith("@develop")


def test_config_set_channel_yolo_normalizes_to_alpha(tmp_path):
    path = tmp_path / "config.yaml"
    SparkrunConfig(path).set_self_update_channel("yolo")
    assert SparkrunConfig(path).self_update_channel == "alpha"


def test_config_unknown_channel_reads_stable(tmp_path):
    path = tmp_path / "config.yaml"
    config = SparkrunConfig(path)
    config.set("self_update.channel", "wat")
    config.save()
    assert SparkrunConfig(path).self_update_channel == "stable"


# --- version display ---


def test_display_version_stable_no_suffix(tmp_path):
    assert display_version(SparkrunConfig(tmp_path / "config.yaml"), base="0.2.40") == "0.2.40"


def test_display_version_beta_with_commit(tmp_path, monkeypatch):
    config = SparkrunConfig(tmp_path / "config.yaml")
    config.set_self_update_channel("beta")
    monkeypatch.setattr("sparkrun.core.version.installed_commit", lambda: "1a2b3c4d5e6f")
    assert display_version(config, base="0.2.40") == "0.2.40-beta+g1a2b3c4"


def test_display_version_alpha_without_commit(tmp_path, monkeypatch):
    config = SparkrunConfig(tmp_path / "config.yaml")
    config.set_self_update_channel("alpha")
    monkeypatch.setattr("sparkrun.core.version.installed_commit", lambda: None)
    assert display_version(config, base="0.3.0") == "0.3.0-alpha"


# --- flag resolution + change messaging ---


def test_channel_from_flags_none():
    assert channel_from_flags(False, False, False, False) is None


def test_channel_from_flags_yolo_is_alpha():
    assert channel_from_flags(False, False, False, True) == "alpha"


def test_channel_from_flags_alpha_and_yolo_agree():
    assert channel_from_flags(False, False, True, True) == "alpha"


def test_channel_from_flags_conflict_raises():
    with pytest.raises(click.ClickException):
        channel_from_flags(True, True, False, False)


def test_update_argv_stable_uses_upgrade():
    assert update_argv("uv", "stable") == ["uv", "tool", "upgrade", "sparkrun"]


def test_update_argv_git_uses_force_install():
    argv = update_argv("uv", "beta")
    assert argv[1:3] == ["tool", "install"] and argv[-1] == "--force" and argv[3].endswith("@develop")


def test_describe_change_stable_same():
    assert "already the latest" in describe_change("stable", ("0.2.40", None), ("0.2.40", None))


def test_describe_change_stable_updated():
    assert describe_change("stable", ("0.2.39", None), ("0.2.40", None)) == "sparkrun updated: 0.2.39 -> 0.2.40"


def test_describe_change_git_same_commit():
    msg = describe_change("beta", ("0.2.40", "abcdef123456"), ("0.2.40", "abcdef123456"))
    assert "already on the latest commit" in msg and "(abcdef1)" in msg


def test_identity_changed_stable_same_version_is_noop():
    assert identity_changed("stable", ("0.3.5", None), ("0.3.5", None)) is False


def test_identity_changed_stable_new_version():
    assert identity_changed("stable", ("0.3.5", None), ("0.3.6", None)) is True


def test_identity_changed_git_same_commit_is_noop():
    # Git channels reinstall with --force every run, so an unchanged commit is a no-op
    # even though uv did real work.
    assert identity_changed("beta", ("0.3.5", "aaaaaaa1"), ("0.3.5", "aaaaaaa1")) is False


def test_identity_changed_git_new_commit():
    assert identity_changed("beta", ("0.3.5", "aaaaaaa1"), ("0.3.5", "bbbbbbb2")) is True


def test_identity_changed_git_ignores_static_version():
    # A git channel's pyproject version is static, so equal versions say nothing.
    assert identity_changed("alpha", ("0.3.5", "aaaaaaa1"), ("0.3.5", "bbbbbbb2")) is True


def test_identity_changed_is_false_when_new_identity_is_indeterminate():
    # `setup version --json` failed; never claim an upgrade we cannot evidence.
    assert identity_changed("stable", ("0.3.5", None), (None, None)) is False
    assert identity_changed("beta", ("0.3.5", "aaaaaaa1"), (None, None)) is False


def test_describe_change_git_new_commit():
    msg = describe_change("alpha", ("0.3.0", "aaaaaaa1"), ("0.3.0", "bbbbbbb2"))
    assert msg == "sparkrun alpha updated: commit aaaaaaa -> bbbbbbb"


# --- CLI integration ---


def _isolate_config(monkeypatch, tmp_path):
    import sparkrun.core.config as config_module

    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_DIR", tmp_path / "config")


def _run(
    monkeypatch,
    tmp_path,
    args,
    *,
    stored=None,
    new_json='{"version": "9.9.9", "channel": "x", "commit": "deadbeefcafe"}',
    core_rc=0,
    uv_tool=True,
):
    _isolate_config(monkeypatch, tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    if stored:
        SparkrunConfig(tmp_path / "config" / "config.yaml").set_self_update_channel(stored)
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[1:3] == ["tool", "list"]:
            import sys

            return mock.Mock(returncode=0, stdout="sparkrun v1 (%s)\n" % (sys.prefix if uv_tool else "/elsewhere"), stderr="")
        if cmd[1:3] in (["tool", "upgrade"], ["tool", "install"]):
            return mock.Mock(returncode=core_rc, stdout="", stderr="resolver exploded" if core_rc else "")
        if "sparkrun.core.plugin_update" in cmd:
            return mock.Mock(returncode=0, stdout='{"protocol":1,"host":{"core_version":"0.4.0"},"plugins":[]}', stderr="")
        if cmd == __import__("sparkrun.cli._self_update", fromlist=["identity_probe_argv"]).identity_probe_argv():
            return mock.Mock(returncode=0, stdout=new_json, stderr="")
        return mock.Mock(returncode=0, stdout="", stderr="")

    with mock.patch("subprocess.run", side_effect=fake_run):
        result = CliRunner().invoke(main, args)
    return result, calls


def _stored_channel(tmp_path):
    return SparkrunConfig(tmp_path / "config" / "config.yaml").self_update_channel


def test_top_update_beta_uses_git_and_persists(monkeypatch, tmp_path):
    result, calls = _run(monkeypatch, tmp_path, ["update", "--beta"])
    assert result.exit_code == 0, result.output
    installs = [c for c in calls if c[1:3] == ["tool", "install"]]
    assert installs and installs[0][3].endswith("@develop") and "--force" in installs[0]
    assert _stored_channel(tmp_path) == "beta"
    assert "Updating recipe registries" in result.output


def test_top_update_no_flag_uses_stored_alpha(monkeypatch, tmp_path):
    result, calls = _run(monkeypatch, tmp_path, ["update"], stored="alpha")
    assert result.exit_code == 0, result.output
    installs = [c for c in calls if c[1:3] == ["tool", "install"]]
    assert installs and installs[0][3].endswith("@develop-next")


def test_top_update_stable_uses_uv_upgrade(monkeypatch, tmp_path):
    result, calls = _run(monkeypatch, tmp_path, ["update"])
    assert result.exit_code == 0, result.output
    assert any(c[1:3] == ["tool", "upgrade"] for c in calls)


def test_top_update_switch_to_stable_warns_downgrade(monkeypatch, tmp_path):
    result, calls = _run(monkeypatch, tmp_path, ["update", "--stable"], stored="alpha")
    assert result.exit_code == 0, result.output
    assert "may downgrade" in result.output
    installs = [c for c in calls if c[1:3] == ["tool", "install"]]
    assert installs and installs[0][3] == "sparkrun"
    assert _stored_channel(tmp_path) == "stable"


def test_setup_update_beta_uses_git(monkeypatch, tmp_path):
    result, calls = _run(monkeypatch, tmp_path, ["setup", "update", "--beta"])
    assert result.exit_code == 0, result.output
    installs = [c for c in calls if c[1:3] == ["tool", "install"]]
    assert installs and installs[0][3].endswith("@develop")
    assert _stored_channel(tmp_path) == "beta"


def test_build_update_event_includes_channel():
    from sparkrun.telemetry.events import build_update_event

    event = build_update_event(
        command="sparkrun update",
        old_version="0.3.0",
        new_version="0.3.0",
        upgraded=True,
        registries=[],
        channel="alpha",
        requested_channel="alpha",
    )
    assert event["channel"] == "alpha"
    assert event["requested_channel"] == "alpha"


def test_build_update_event_omits_channel_when_none():
    from sparkrun.telemetry.events import build_update_event

    event = build_update_event(
        command="sparkrun update",
        old_version="0.3.0",
        new_version="0.3.0",
        upgraded=False,
        registries=[],
    )
    assert "channel" not in event
    assert "requested_channel" not in event


def _captured_update_event(monkeypatch, tmp_path, args, *, new_json, stored=None):
    """Run an update command and return the telemetry event it built."""
    import sparkrun.telemetry.emit as emit_mod

    captured = {}

    def capture(config, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(emit_mod, "emit_update_event", capture)
    result, calls = _run(monkeypatch, tmp_path, args, stored=stored, new_json=new_json)
    return result, calls, captured


def test_update_noop_does_not_report_a_successful_self_upgrade(monkeypatch, tmp_path):
    """`uv tool upgrade` exits 0 on "Nothing to upgrade" — that is not an upgrade.

    Regression: keying `upgraded` off the exit code reported every no-op update as a
    successful self-upgrade, which made `self_upgrade_succeeded` useless as a signal.
    """
    from sparkrun.core.version import installed_identity

    current_version = installed_identity()[0]
    same = '{"version": "%s", "channel": "stable", "commit": null}' % current_version

    _, _, event = _captured_update_event(monkeypatch, tmp_path, ["update"], new_json=same)

    assert event["upgraded"] is False
    assert event["self_upgrade_attempted"] is True


def test_update_noop_refreshes_registries_in_fresh_process(monkeypatch, tmp_path):
    """A no-op still audits plugins and refreshes registries outside the recovery process."""
    from sparkrun.core.version import installed_identity

    current_version = installed_identity()[0]
    same = '{"version": "%s", "channel": "stable", "commit": null}' % current_version

    _, calls, _ = _captured_update_event(monkeypatch, tmp_path, ["update"], new_json=same)

    assert calls.count([owning_executable(), "registry", "update"]) == 1


def test_update_with_a_real_upgrade_still_reports_and_shells_out(monkeypatch, tmp_path):
    """The changed path is unaffected: report the upgrade and re-exec for registries."""
    changed = '{"version": "99.9.9", "channel": "stable", "commit": null}'

    _, calls, event = _captured_update_event(monkeypatch, tmp_path, ["update"], new_json=changed)

    assert event["upgraded"] is True
    assert [owning_executable(), "registry", "update"] in calls


def test_top_update_survives_telemetry_failure(monkeypatch, tmp_path):
    """A telemetry emit failure (e.g. a cross-version downgrade) must not crash update."""
    import sparkrun.telemetry.emit as emit_mod

    def boom(*args, **kwargs):
        raise RuntimeError("telemetry shape mismatch across versions")

    monkeypatch.setattr(emit_mod, "emit_update_event", boom)
    result, _ = _run(monkeypatch, tmp_path, ["update", "--beta"])
    assert result.exit_code == 0, result.output


def test_setup_install_alpha_uses_git_and_persists(monkeypatch, tmp_path):
    _isolate_config(monkeypatch, tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return mock.Mock(returncode=0, stdout=str(tmp_path / "uv-tools") if cmd[1:3] == ["tool", "dir"] else "", stderr="")

    with (
        mock.patch("subprocess.run", side_effect=fake_run),
        mock.patch("sparkrun.cli._setup._commands._install_completion", mock.MagicMock()),
    ):
        result = CliRunner().invoke(main, ["setup", "install", "--alpha", "--no-update-registries", "--shell", "bash"])
    assert result.exit_code == 0, result.output
    installs = [c for c in calls if c[1:3] == ["tool", "install"]]
    assert installs and installs[0][3].endswith("@develop-next")
    assert _stored_channel(tmp_path) == "alpha"


@pytest.mark.parametrize("route", [["update"], ["setup", "update", "--no-update-registries"]])
@pytest.mark.parametrize("optout", [False, True])
def test_both_update_routes_use_fresh_plugin_reconciliation(monkeypatch, tmp_path, route, optout):
    result, calls = _run(monkeypatch, tmp_path, [*route, *(["--no-update-plugins"] if optout else [])])
    assert result.exit_code == 0, result.output
    helpers = [c for c in calls if "sparkrun.core.plugin_update" in c]
    assert len(helpers) == 1
    assert helpers[0][1:5] == ["-m", "sparkrun.core.plugin_update", "--protocol", "1"]
    assert ("--no-update-plugins" in helpers[0]) is optout


@pytest.mark.parametrize("route", [["update"], ["setup", "update", "--no-update-registries"]])
def test_unverified_core_update_audits_without_repairs(monkeypatch, tmp_path, route):
    result, calls = _run(monkeypatch, tmp_path, route, new_json="not JSON")
    assert result.exit_code == 0, result.output
    helper = next(c for c in calls if "sparkrun.core.plugin_update" in c)
    assert "--no-update-plugins" in helper


BOTH_ROUTES = [["update"], ["setup", "update"]]


@pytest.mark.parametrize("route", BOTH_ROUTES)
def test_failed_core_upgrade_exits_nonzero_on_both_routes(monkeypatch, tmp_path, route):
    """A failed upgrade is a failure: audit plugins, refresh registries, then exit 1."""
    result, calls = _run(monkeypatch, tmp_path, route, core_rc=1)
    assert result.exit_code == 1, result.output
    assert "upgrade failed: resolver exploded" in result.output
    helper = next(c for c in calls if "sparkrun.core.plugin_update" in c)
    assert "--no-update-plugins" in helper
    assert [owning_executable(), "registry", "update"] in calls
    assert _stored_channel(tmp_path) == "stable"


@pytest.mark.parametrize("route", BOTH_ROUTES)
def test_non_uv_install_skips_the_core_on_both_routes(monkeypatch, tmp_path, route):
    """pip / pipx / editable installs skip the self-upgrade; plugins and registries still update."""
    result, calls = _run(monkeypatch, tmp_path, route, uv_tool=False)
    assert result.exit_code == 0, result.output
    assert "skipping self-upgrade" in result.output
    assert not [c for c in calls if c[1:3] in (["tool", "upgrade"], ["tool", "install"])]
    helper = next(c for c in calls if "sparkrun.core.plugin_update" in c)
    assert "--no-update-plugins" not in helper
    assert [owning_executable(), "registry", "update"] in calls


@pytest.mark.parametrize("route", BOTH_ROUTES)
def test_both_routes_accept_no_update_registries(monkeypatch, tmp_path, route):
    result, calls = _run(monkeypatch, tmp_path, [*route, "--no-update-registries"])
    assert result.exit_code == 0, result.output
    assert [owning_executable(), "registry", "update"] not in calls


@pytest.mark.parametrize("route", BOTH_ROUTES)
def test_both_routes_report_update_telemetry_under_their_own_name(monkeypatch, tmp_path, route):
    changed = '{"version": "99.9.9", "channel": "stable", "commit": null}'
    _, _, event = _captured_update_event(monkeypatch, tmp_path, route, new_json=changed)
    assert event["command"] == "sparkrun " + " ".join(route)
    assert event["upgraded"] is True and event["self_upgrade_attempted"] is True
