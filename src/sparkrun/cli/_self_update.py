"""Update-channel self-update orchestration for the sparkrun CLI.

Wraps uv-tool install/upgrade behavior for the stable/beta/alpha channels and
provides identity-based update detection: PyPI stable compares the package
version, while git channels compare the resolved commit (their `pyproject`
version is static, so a version-string compare would never detect a change).

The uv argv here is validated by the Phase 0 spike: `uv tool install
"<git req>" --force` re-resolves a mutable branch to its newest commit and
rebuilds even when the version is unchanged.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import re
import sys
from pathlib import Path

from sparkrun.core.application_profile import get_application_profile

import click

from sparkrun.core.channels import (
    CHANNEL_STABLE,
    channel_requirement,
    is_git_channel,
    normalize_channel,
)
from sparkrun.core.version import installed_identity


def resolve_uv() -> str | None:
    """Return the uv executable path, or None if uv is not installed."""
    return shutil.which("uv")


def environment_executable(prefix: str | Path, command: str) -> Path:
    """Resolve an entry point inside a Python tool environment on this OS."""
    return Path(prefix) / ("Scripts" if sys.platform == "win32" else "bin") / (command + ".exe" if sys.platform == "win32" else command)


def owning_executable() -> str:
    """The executable inside the running tool environment, independent of PATH."""
    return str(environment_executable(sys.prefix, get_application_profile().command))


def installed_tool_executable(uv: str, *, strict: bool = False) -> Path | None:
    """Resolve the owning installed tool, including after an install from uvx."""
    try:
        check = subprocess.run([uv, "tool", "list", "--show-paths"], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as exc:
        if strict:
            raise RuntimeError("Cannot locate the destination uv environment") from exc
        return None
    if check.returncode:
        if strict:
            raise RuntimeError("Cannot locate the destination uv environment: " + check.stderr.strip())
        return None
    expected = re.sub(r"[-_.]+", "-", get_application_profile().package).lower()
    for line in check.stdout.splitlines():
        match = re.fullmatch(r"([^ ]+) v[^ ]+ \((.+)\)", line.strip())
        if match and re.sub(r"[-_.]+", "-", match[1]).lower() == expected:
            return environment_executable(Path(match[2]).resolve(), get_application_profile().command)
    return None


def installation_prefix(uv: str) -> Path:
    """Resolve the destination, including a first install or a caller in uvx.

    Discovery failures cannot authorize a forced replacement. The package name
    is the normalized owning distribution, not its possibly different command.
    """
    installed = installed_tool_executable(uv, strict=True)
    if installed is not None:
        return installed.parent.parent
    try:
        result = subprocess.run([uv, "tool", "dir"], capture_output=True, text=True, timeout=15)
        path = Path(result.stdout.strip())
        if result.returncode or not path.is_absolute():
            raise RuntimeError("Cannot locate the destination uv tools directory")
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("Cannot locate the destination uv tools directory") from exc
    package = re.sub(r"[-_.]+", "-", get_application_profile().package).lower()
    return (path / package).resolve()


def is_uv_tool_install(uv: str) -> bool:
    """Require an exact owning package AND the current interpreter environment."""
    executable = installed_tool_executable(uv)
    return executable is not None and executable.parent.parent == Path(sys.prefix).resolve()


def channel_from_flags(stable: bool, beta: bool, alpha: bool, yolo: bool) -> str | None:
    """Return the requested canonical channel from mutually-exclusive flags.

    Returns None when no flag is set. Raises ``click.ClickException`` if flags
    select conflicting channels (``--yolo`` and ``--alpha`` do not conflict —
    both normalize to ``alpha``).
    """
    requested = {
        normalize_channel(chan)
        for chan, enabled in (
            (CHANNEL_STABLE, stable),
            ("beta", beta),
            ("alpha", alpha),
            ("yolo", yolo),
        )
        if enabled
    }
    if not requested:
        return None
    if len(requested) > 1:
        raise click.ClickException("Choose only one of --stable, --beta, --alpha, --yolo.")
    return requested.pop()


def install_argv(uv: str, channel: str) -> list[str]:
    """Build the uv argv to install or switch to a channel."""
    return [uv, "tool", "install", channel_requirement(channel), "--force"]


def update_argv(uv: str, channel: str) -> list[str]:
    """Build the uv argv to update within a channel.

    Stable uses ``uv tool upgrade``; git channels reinstall with ``--force`` so
    the mutable branch re-resolves to its newest commit.
    """
    if not is_git_channel(channel) and channel_requirement(channel) == get_application_profile().package:
        return [uv, "tool", "upgrade", get_application_profile().package]
    return install_argv(uv, channel)


def owning_interpreter() -> str:
    """Fresh Python from the tool environment, even after uv replaces it."""
    return str(environment_executable(sys.prefix, "python"))


def identity_probe_argv() -> list[str]:
    # Stdlib metadata only: older hosts may load plugins even for `setup version`.
    # Pass the package as data, never interpolate it into executable code.
    script = """import importlib.metadata as m, json, sys
try:
    d = m.distribution(sys.argv[1])
    raw = d.read_text('direct_url.json')
    try:
        commit = (json.loads(raw or '{}').get('vcs_info') or {}).get('commit_id')
    except (ValueError, TypeError, AttributeError):
        commit = None
    print(json.dumps({'version': d.version, 'commit': commit}))
except m.PackageNotFoundError:
    print(json.dumps({'version': None, 'commit': None}))
"""
    return [owning_interpreter(), "-c", script, get_application_profile().package]


def new_binary_identity() -> tuple[str | None, str | None]:
    """Read the resulting application's identity without importing host/plugin code."""
    try:
        res = subprocess.run(identity_probe_argv(), capture_output=True, text=True, timeout=15)
        if res.returncode != 0:
            return None, None
        data = json.loads(res.stdout.strip() or "{}")
        version, commit = data.get("version"), data.get("commit")
        return version if isinstance(version, str) else None, commit if isinstance(commit, str) else None
    except (OSError, ValueError, AttributeError, subprocess.TimeoutExpired):
        return None, None


def identity_changed(
    channel: str,
    old: tuple[str | None, str | None],
    new: tuple[str | None, str | None],
) -> bool:
    """Return whether the install actually changed, by the same rule as `describe_change`.

    A zero exit from uv means the command worked, not that anything was installed:
    ``uv tool upgrade`` prints "Nothing to upgrade" and exits 0, and git channels
    reinstall with ``--force`` on every run regardless of whether the branch moved.
    So the identity is the only evidence an upgrade happened.

    An indeterminate new identity (the ``setup version --json`` probe failed) returns
    False: callers use this to claim an upgrade in telemetry and to decide whether the
    running process holds stale code, and both are better off assuming nothing changed
    than reporting an upgrade that may not have occurred.
    """
    old_version, old_commit = old
    new_version, new_commit = new
    if is_git_channel(channel):
        if new_commit is None:
            return False
        return old_commit != new_commit
    if new_version is None:
        return False
    return old_version != new_version


def describe_change(channel: str, old: tuple[str | None, str | None], new: tuple[str | None, str | None]) -> str:
    """Return a human message describing the version/commit change after update."""
    old_version, old_commit = old
    new_version, new_commit = new
    if is_git_channel(channel):
        if new_commit is None:
            return f"{get_application_profile().command} %s updated (could not determine new commit)." % channel
        if old_commit == new_commit:
            return f"{get_application_profile().command} %s is already on the latest commit (%s)." % (channel, new_commit[:7])
        old_disp = old_commit[:7] if old_commit else "unknown"
        return f"{get_application_profile().command} %s updated: commit %s -> %s" % (channel, old_disp, new_commit[:7])
    # stable / PyPI
    if new_version is None:
        return f"{get_application_profile().command} updated (could not determine new version)."
    if old_version == new_version:
        return f"{get_application_profile().command} %s is already the latest version." % new_version
    return f"{get_application_profile().command} updated: %s -> %s" % (old_version, new_version)


def warn_if_downgrade(current_channel: str, requested_channel: str) -> None:
    """Warn when switching from a git channel to stable, which may downgrade."""
    if requested_channel == CHANNEL_STABLE and is_git_channel(current_channel):
        click.echo(
            "Note: switching to the stable channel may downgrade from a preview build (stable tracks the latest PyPI release).",
            err=True,
        )


def capture_old_identity() -> tuple[str, str | None]:
    """Return the currently-running build's ``(version, commit)`` identity."""
    return installed_identity()


def reconcile_plugin_updates(config_path, *, update: bool = True) -> bool:
    """Run the versioned management helper in the actual resulting interpreter.

    Its code and package metadata belong to the new host. Never fall back to
    mutating plugins through this process's stale modules after a core install.
    Returns whether plugin generations changed.
    """
    import os
    from sparkrun.core.application_profile import child_environment

    args = [owning_interpreter(), "-m", "sparkrun.core.plugin_update", "--protocol", "1"]
    if not update:
        args.append("--no-update-plugins")
    try:
        result = subprocess.run(args, env={**os.environ, **child_environment(config_path=config_path)}, capture_output=True, text=True)
        if result.returncode:
            raise ValueError(result.stderr.strip() or "host does not support plugin reconciliation")
        report = json.loads(result.stdout)
        if report.get("protocol") != 1 or not isinstance(report.get("plugins"), list) or not isinstance(report.get("host"), dict):
            raise ValueError("host returned an unsupported plugin reconciliation response")
    except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError) as exc:
        click.echo("Warning: plugin compatibility audit incomplete; no unverified repairs attempted: %s" % exc, err=True)
        return False
    changed = False
    if report.get("warning"):
        click.echo("Warning: " + report["warning"], err=True)
    for default in report.get("defaults", []):
        if default["status"] == "failed":
            click.echo("Warning: default plugin %s installation failed: %s" % (default["name"], default["error"]), err=True)
    click.echo("Checking plugin compatibility against Sparkrun %s..." % report["host"].get("core_version", "unknown"))
    for row in report["plugins"]:
        outcome = row["outcome"]
        changed = changed or bool(row.get("update", {}).get("history_pruned"))
        compatibility = row.get("compatibility_after", row["compatibility"])
        if outcome in {"updated", "installed", "rolled back"}:
            changed = True
            updated = row["update"]
            click.echo("  %s: %s %s @ %s" % (row["name"], outcome, updated.get("version", ""), updated.get("revision", "")[:12]))
        elif row.get("managed"):
            click.echo("  %s: %s" % (row["name"], outcome))
        if compatibility["status"] != "compatible":
            click.echo(
                "Warning: %s compatibility %s: %s" % (row["name"], compatibility["status"], "; ".join(compatibility["reasons"])), err=True
            )
        if row.get("error"):
            click.echo("Warning: %s update failed; installed generation retained: %s" % (row["name"], row["error"]), err=True)
        if row.get("update", {}).get("warning"):
            click.echo("Warning: %s" % row["update"]["warning"], err=True)
    if not update:
        click.echo("Plugin fetches and updates skipped (--no-update-plugins or failed core update).")
    if changed:
        click.echo("Restart long-running plugin consumers; retained history may have been pruned.")
    return changed


def update_registries_after_core(config_path, *, executable=None) -> None:
    """Attempt registry refresh without importing plugins into the recovery command."""
    import os
    from sparkrun.core.application_profile import child_environment

    click.echo()
    click.echo("Updating recipe registries...")
    try:
        result = subprocess.run(
            [str(executable) if executable is not None else owning_executable(), "registry", "update"],
            env={**os.environ, **child_environment(config_path=config_path)},
            capture_output=False,
            timeout=300,
        )
        if result.returncode:
            click.echo("Warning: registry update failed (non-fatal).", err=True)
    except (OSError, subprocess.TimeoutExpired) as exc:
        click.echo("Warning: registry update failed or timed out (non-fatal): %s" % exc, err=True)


def run_update(config, *, command: str, requested: str | None, update_plugins: bool = True, update_registries: bool = True) -> int:
    """Shared body of ``update`` and ``setup update``; returns the exit status.

    Three steps, each in the resulting environment rather than this process:
    the core self-upgrade (uv tool installs only; any other install method is
    skipped, not refused, so plugins and registries still refresh), the plugin
    reconciliation, and the registry refresh. Plugins are repaired only when
    the core is known: no upgrade attempted, or the new binary's identity
    verified. A failed core upgrade still audits plugins and refreshes
    registries, then exits 1.
    """
    # Import telemetry NOW, before a possible cross-version downgrade (e.g. a
    # 0.3.x -> 0.2.x switch) overwrites the on-disk package mid-run and makes
    # the module unavailable. Keeping it resident lets the send still succeed.
    try:
        from sparkrun.telemetry.emit import emit_update_event
    except Exception:
        emit_update_event = None

    app = get_application_profile().command
    current = config.self_update_channel
    channel = requested if requested is not None else current
    switching = requested is not None and requested != current

    old_identity = capture_old_identity()
    old_version = old_identity[0]
    new_version: str | None = old_version
    self_upgrade_attempted = False
    core_identity_verified = True
    core_update_failed = False
    upgraded = False

    uv = resolve_uv()
    if uv and is_uv_tool_install(uv):
        self_upgrade_attempted = True
        if switching:
            warn_if_downgrade(current, channel)
            click.echo("Switching to the %s channel..." % channel)
        click.echo("Checking for %s updates (current: %s)..." % (app, old_version))
        from sparkrun.core.runtime_lease import guarded_update

        try:
            result = guarded_update(
                config,
                subprocess.run,
                install_argv(uv, channel) if switching else update_argv(uv, channel),
                capture_output=True,
                text=True,
            )
        except RuntimeError as exc:
            raise click.ClickException(str(exc)) from exc
        if result.returncode == 0:
            config.set_self_update_channel(channel)
            # The running process still has the old module cached, so ask the
            # newly installed binary for its identity (version for stable,
            # commit for git channels).
            new_identity = new_binary_identity()
            core_identity_verified = bool(new_identity[0])
            new_version = new_identity[0] or new_version
            click.echo(describe_change(channel, old_identity, new_identity))
            # A zero exit only says uv ran: `uv tool upgrade` exits 0 on "Nothing to
            # upgrade", and git channels reinstall with --force every time. Compare
            # identities for reporting/telemetry.
            upgraded = identity_changed(channel, old_identity, new_identity)
        else:
            core_update_failed = True
            click.echo("Error: %s upgrade failed: %s" % (app, result.stderr.strip()), err=True)
    elif uv:
        click.echo("%s not installed via uv tool — skipping self-upgrade." % app)
    else:
        click.echo("uv not found — skipping self-upgrade.")

    reconcile_plugin_updates(config.config_path, update=update_plugins and not core_update_failed and core_identity_verified)

    # Registry refresh also needs the resulting environment and may load plugins.
    # Keep that execution out of the recovery command, including no-op updates.
    if update_registries:
        update_registries_after_core(config.config_path)

    if emit_update_event is not None:
        try:
            emit_update_event(
                config,
                command=command,
                old_version=old_version,
                new_version=new_version,
                upgraded=upgraded,
                self_upgrade_attempted=self_upgrade_attempted,
                channel=channel,
                requested_channel=requested,
            )
        except Exception:  # best-effort: telemetry must never break an update
            import logging

            logging.getLogger(__name__).debug("Update telemetry was not sent", exc_info=True)
    return 1 if core_update_failed else 0
