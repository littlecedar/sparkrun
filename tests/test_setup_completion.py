"""Tests for ``sparkrun setup completion`` snippet generation.

Covers issue #198: the completion snippet must embed the absolute,
shell-resolved path to ``sparkrun`` so it is robust against PATH ordering
in the user's shell rc file, rather than the bare ``sparkrun`` command.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest
from click.testing import CliRunner

from sparkrun.cli import main


@pytest.fixture
def rc_file(tmp_path: Path):
    """Redirect the shell rc file to a temp path for the duration of a test."""
    target = tmp_path / "rc"
    with mock.patch("sparkrun.cli._setup._commands._shell_rc_file", return_value=target):
        yield target


@mock.patch("shutil.which", return_value="/home/u/.local/bin/sparkrun")
def test_bash_embeds_absolute_path(mock_which, rc_file: Path):
    result = CliRunner().invoke(main, ["setup", "completion", "--shell", "bash"])
    assert result.exit_code == 0, result.output

    contents = rc_file.read_text()
    assert "_SPARKRUN_COMPLETE=bash_source /home/u/.local/bin/sparkrun)" in contents
    # Must not fall back to the bare command when the path resolves.
    assert "=bash_source sparkrun)" not in contents


@mock.patch("shutil.which", return_value="/home/u/.local/bin/sparkrun")
def test_zsh_embeds_absolute_path(mock_which, rc_file: Path):
    result = CliRunner().invoke(main, ["setup", "completion", "--shell", "zsh"])
    assert result.exit_code == 0, result.output
    assert "_SPARKRUN_COMPLETE=zsh_source /home/u/.local/bin/sparkrun)" in rc_file.read_text()


@mock.patch("shutil.which", return_value="/home/u/.local/bin/sparkrun")
def test_fish_embeds_absolute_path(mock_which, rc_file: Path):
    result = CliRunner().invoke(main, ["setup", "completion", "--shell", "fish"])
    assert result.exit_code == 0, result.output
    assert "_SPARKRUN_COMPLETE=fish_source /home/u/.local/bin/sparkrun | source" in rc_file.read_text()


@mock.patch("shutil.which", return_value=None)
def test_falls_back_to_bare_command_when_unresolved(mock_which, rc_file: Path):
    result = CliRunner().invoke(main, ["setup", "completion", "--shell", "bash"])
    assert result.exit_code == 0, result.output
    assert "_SPARKRUN_COMPLETE=bash_source sparkrun)" in rc_file.read_text()


@mock.patch("shutil.which", return_value="/opt/my tools/sparkrun")
def test_path_with_spaces_is_quoted(mock_which, rc_file: Path):
    result = CliRunner().invoke(main, ["setup", "completion", "--shell", "bash"])
    assert result.exit_code == 0, result.output
    # shlex.quote wraps the space-containing path in single quotes.
    assert "_SPARKRUN_COMPLETE=bash_source '/opt/my tools/sparkrun')" in rc_file.read_text()


@pytest.mark.parametrize("shell", ["bash", "zsh", "fish"])
@pytest.mark.parametrize("existing", [False, True])
def test_setup_install_embeds_the_installed_tool_not_the_uvx_environment(tmp_path, rc_file: Path, shell, existing):
    """``uvx sparkrun setup install`` runs with uvx's throwaway environment first on PATH.

    Completion must name the tool it just installed. The PATH lookup embedded the
    uv cache copy, which ``uv cache clean`` deletes, breaking every new shell.
    """
    uvx_copy = str(tmp_path / ".cache/uv/archive-v0/abc/bin/sparkrun")
    installed = tmp_path / "my tools/sparkrun"
    if existing:
        rc_file.write_text(_snippet(shell, "/home/u/.cache/uv/archive-v0/deleted/bin/sparkrun") + "\n")

    def which(name):
        return "/usr/bin/uv" if name == "uv" else uvx_copy

    def run(argv, **kwargs):
        stdout = "sparkrun v0.4.1 (%s)\n" % installed if argv[1:3] == ["tool", "list"] else ""
        return mock.Mock(returncode=0, stdout=stdout, stderr="")

    with mock.patch("shutil.which", side_effect=which), mock.patch("subprocess.run", side_effect=run):
        result = CliRunner().invoke(main, ["setup", "install", "--no-update-registries", "--shell", shell])
    assert result.exit_code == 0, result.output

    contents = rc_file.read_text()
    from sparkrun.cli._self_update import environment_executable
    from sparkrun.utils.shell import quote

    assert _snippet(shell, quote(str(environment_executable(installed, "sparkrun")))) in contents
    assert uvx_copy not in contents
    assert "archive-v0" not in contents
    assert contents.count("_SPARKRUN_COMPLETE") == 1


def _snippet(shell, executable):
    if shell == "fish":
        return "_SPARKRUN_COMPLETE=fish_source %s | source" % executable
    return 'eval "$(_SPARKRUN_COMPLETE=%s_source %s)"' % (shell, executable)


@pytest.mark.parametrize("shell", ["bash", "zsh", "fish"])
@pytest.mark.parametrize("old_executable", ["sparkrun", "/old/bin/sparkrun", "'/old tools/sparkrun'"])
def test_repair_updates_generated_entries_in_place_and_is_idempotent(rc_file, shell, old_executable):
    from sparkrun.cli._setup._commands import _install_completion

    prefix = "# user settings\n# sparkrun tab-completion\n"
    suffix = '\necho "keep this setting"\n' + _snippet(shell, "/opt/other-tool").replace("_SPARKRUN_COMPLETE", "_OTHER_COMPLETE")
    # Preserve indentation and surrounding settings, including a missing final newline.
    rc_file.write_text(prefix + "  " + _snippet(shell, old_executable) + suffix)

    _install_completion(shell, rc_file, "/new tools/sparkrun", replace_existing=True)
    expected = prefix + "  " + _snippet(shell, "'/new tools/sparkrun'") + suffix
    assert rc_file.read_text() == expected

    # A second install should neither append nor rewrite the already-correct file.
    with mock.patch.object(Path, "write_text", side_effect=AssertionError("unnecessary rewrite")):
        _install_completion(shell, rc_file, "/new tools/sparkrun", replace_existing=True)
    assert rc_file.read_text() == expected


@pytest.mark.parametrize(
    "custom",
    [
        'eval "$(_SPARKRUN_COMPLETE=bash_source wrapper --custom)"',
        'eval "$(_SPARKRUN_COMPLETE=bash_source sparkrun; echo keep-me)"',
        'eval "$(_SPARKRUN_COMPLETE=bash_source $CUSTOM_COMMAND)"',
        'eval "$(_SPARKRUN_COMPLETE=bash_source \'unterminated)"',
    ],
)
def test_repair_preserves_custom_shell_code(rc_file, custom):
    from sparkrun.cli._setup._commands import _install_completion

    generated = _snippet("bash", "/old/bin/sparkrun")
    rc_file.write_text(custom + "\n" + generated + "\n")
    _install_completion("bash", rc_file, "/new/bin/sparkrun", replace_existing=True)
    assert rc_file.read_text() == custom + "\n" + _snippet("bash", "/new/bin/sparkrun") + "\n"


@mock.patch("shutil.which", return_value="/home/u/.local/bin/sparkrun")
def test_idempotent_when_already_configured(mock_which, rc_file: Path):
    rc_file.write_text('# existing\neval "$(_SPARKRUN_COMPLETE=bash_source sparkrun)"\n')
    before = rc_file.read_text()
    result = CliRunner().invoke(main, ["setup", "completion", "--shell", "bash"])
    assert result.exit_code == 0, result.output
    assert "already configured" in result.output
    assert rc_file.read_text() == before  # not appended twice


@pytest.mark.parametrize("shell", ["bash", "zsh", "fish"])
def test_standalone_completion_does_not_replace_installed_tool_with_uvx(rc_file, shell):
    before = _snippet(shell, "/home/u/.local/bin/sparkrun") + "\n"
    rc_file.write_text(before)
    with mock.patch("shutil.which", return_value="/home/u/.cache/uv/archive-v0/temporary/bin/sparkrun"):
        result = CliRunner().invoke(main, ["setup", "completion", "--shell", shell])
    assert result.exit_code == 0, result.output
    assert rc_file.read_text() == before
