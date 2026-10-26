"""Default provisioning uses the real offline Git lifecycle and persistent policy."""

from contextlib import contextmanager
from dataclasses import replace
import json
import os
import subprocess
import sys

import pytest
from click.testing import CliRunner

from sparkrun.application import ApplicationProfile, DefaultPlugin
from sparkrun.core.config import SparkrunConfig
from sparkrun.core.plugin_defaults import default_plugins, reconcile_default_plugins, suspend_default_installation
from sparkrun.core.plugin_lifecycle import change_plugin
from sparkrun.core.plugin_inventory import list_plugins
from sparkrun.core.plugin_manager import PluginManager
from sparkrun.core.plugin_policy import PluginSource, plugin_state
from test_managed_plugins import manager as manager, repository as repository, state  # noqa: F401


@pytest.fixture
def catalog(repository, monkeypatch):
    entry = DefaultPlugin("managedfixture", repository[0].as_uri(), "sparkrun.plugins.managedfixture", trusted=True)
    monkeypatch.setattr("sparkrun.core.plugin_defaults.DEFAULT_PLUGINS", (entry,))
    monkeypatch.delenv("SPARKRUN_NO_EXTERNAL_PLUGINS", raising=False)
    monkeypatch.delenv("SPARKRUN_NO_DEFAULT_PLUGINS", raising=False)
    return entry


def sync(manager, **kwargs):
    return reconcile_default_plugins(manager.config, manager=manager, **kwargs)


def test_profile_catalog_is_frozen_and_does_not_inherit_sparkrun(catalog):
    profile = ApplicationProfile("example", "Example", "example", "example")
    assert default_plugins(profile) == ()
    entries = [catalog]
    custom = replace(profile, default_plugins=entries)
    entries.clear()
    assert default_plugins(custom) == (catalog,)
    with pytest.raises(ValueError, match="Duplicate"):
        replace(profile, default_plugins=(catalog, catalog))
    with pytest.raises(ValueError, match="explicitly"):
        replace(profile, default_plugins=None)


@pytest.mark.parametrize("changes", [{"name": "NonCanonical"}, {"module": "os.path"}, {"ref": "--upload-pack=bad"}, {"trusted": "yes"}])
def test_invalid_catalog_declarations_fail_early(catalog, changes):
    with pytest.raises((ValueError, TypeError)):
        replace(catalog, **changes)


def test_preview_never_fetches_executes_or_creates_store(catalog, manager, monkeypatch):
    monkeypatch.setattr(manager, "acquire", lambda *a: pytest.fail("preview fetched"))
    assert sync(manager, dry_run=True)[0]["status"] == "planned"
    assert not manager.store.root.exists()
    with suspend_default_installation():
        assert sync(manager)[0]["status"] == "skipped"
    assert not manager.store.root.exists()


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("plugins.enabled", {"managedfixture": False}),
        ("plugins.defaults.enabled", False),
        ("plugins.defaults.suppressed", ["managedfixture"]),
        ("features", {"core.external_plugins": False}),
    ],
)
def test_opt_out_prevents_acquisition(catalog, manager, monkeypatch, setting, value):
    manager.config.set(setting, value)
    manager.config.save()
    before = manager.config.config_path.read_bytes()
    monkeypatch.setattr(manager, "acquire", lambda *a: pytest.fail("disabled default fetched"))
    assert sync(manager)[0]["status"] == "skipped"
    assert manager.config.config_path.read_bytes() == before
    assert not manager.store.ledger.exists()


@pytest.mark.parametrize(
    "variable", ["SPARKRUN_NO_EXTERNAL_PLUGINS", "SPARKRUN_NO_DEFAULT_PLUGINS", "SPARKRUN_FEATURE_CORE_EXTERNAL_PLUGINS"]
)
def test_environment_opt_out_prevents_acquisition(catalog, manager, monkeypatch, variable):
    monkeypatch.setenv(variable, "0" if variable.startswith("SPARKRUN_FEATURE") else "1")
    monkeypatch.setattr(manager, "acquire", lambda *a: pytest.fail("disabled default fetched"))
    assert sync(manager)[0]["status"] == "skipped"
    assert not manager.store.root.exists()


def test_untrusted_catalog_entry_never_fetches(catalog, manager, monkeypatch):
    monkeypatch.setattr("sparkrun.core.plugin_defaults.DEFAULT_PLUGINS", (replace(catalog, trusted=False),))
    monkeypatch.setattr(manager, "acquire", lambda *a: pytest.fail("untrusted default fetched"))
    assert "trust required" in sync(manager)[0]["reason"]


def test_default_install_preserves_directory_gate_and_is_idempotent(catalog, manager, monkeypatch, tmp_path):
    manager.config.set("plugins.paths", [str(tmp_path / "untrusted")])
    manager.config.save()
    before = manager.config.config_path.read_bytes()
    assert sync(manager)[0]["status"] == "installed"
    record = state(manager)
    assert record["trust"]["default_managed"] is True
    assert manager.config.config_path.read_bytes() == before
    config = SparkrunConfig(manager.config.config_path)
    assert not config.is_feature_enabled("core.external_plugins")
    from sparkrun.core.managed_plugins import managed_enabled

    assert managed_enabled(record, config)
    assert not managed_enabled({**record, "trust": {"granted": True}}, config)
    monkeypatch.setattr(manager, "acquire", lambda *a: pytest.fail("present default fetched"))
    assert sync(manager)[0]["status"] == "present"


def test_removal_suppresses_reinstall_and_enable_restores_it(catalog, manager, monkeypatch):
    assert sync(manager)[0]["status"] == "installed"
    manager.remove(catalog.name)
    config = SparkrunConfig(manager.config.config_path)
    assert config.get("plugins.defaults.suppressed") == [catalog.name]
    with monkeypatch.context() as patch:
        patch.setattr(manager, "acquire", lambda *a: pytest.fail("removed default fetched"))
        assert sync(manager)[0]["reason"] == "removed by user"
    change_plugin(config, catalog.name, True)
    assert sync(manager)[0]["status"] == "installed"
    change_plugin(config, catalog.name, False)
    assert not next(p for p in list_plugins(config) if p.managed).enabled


def test_missing_default_can_be_disabled_before_first_install(catalog, manager, monkeypatch):
    monkeypatch.setattr(manager, "acquire", lambda *a: pytest.fail("disabled default fetched"))
    change_plugin(manager.config, catalog.name, False)
    assert sync(manager)[0]["status"] == "skipped"
    assert not manager.store.directory.exists()


def test_enabling_missing_default_does_not_override_global_disable(catalog, manager):
    from sparkrun.core.plugin_manifest import PluginError

    manager.config.set("features", {"core.external_plugins": False})
    manager.config.set("plugins.defaults.suppressed", [catalog.name])
    manager.config.save()
    before = manager.config.config_path.read_bytes()
    with pytest.raises(PluginError, match="external plugin loading disabled"):
        change_plugin(manager.config, catalog.name, True)
    assert manager.config.config_path.read_bytes() == before


def test_existing_user_source_and_pin_win(catalog, repository, manager, monkeypatch):
    root, _commit, git = repository
    git("tag", "v1")
    manager.add(str(root), ref="refs/tags/v1", trust=True)
    before = state(manager)
    monkeypatch.setattr("sparkrun.core.plugin_defaults.DEFAULT_PLUGINS", (replace(catalog, source="https://example.invalid/other"),))
    monkeypatch.setattr(manager, "acquire", lambda *a: pytest.fail("existing provider replaced"))
    assert sync(manager)[0]["status"] == "present"
    assert state(manager) == before
    assert not state(manager)["trust"].get("default_managed")


def test_existing_unmanaged_provider_is_preserved(catalog, manager, monkeypatch):
    from sparkrun.core.plugin_inventory import PluginInfo

    plugin = PluginInfo(catalog.name, "installed", catalog.module, False, False)
    monkeypatch.setattr("sparkrun.core.plugin_inventory.list_plugins", lambda config: [plugin])
    monkeypatch.setattr(manager, "acquire", lambda *a: pytest.fail("existing provider replaced"))
    assert sync(manager)[0]["status"] == "present"
    assert not manager.store.root.exists()


def test_identity_mismatch_and_failure_do_not_block_other_defaults(catalog, manager, monkeypatch):
    wrong = replace(catalog, name="wrong", module="wrong")
    monkeypatch.setattr("sparkrun.core.plugin_defaults.DEFAULT_PLUGINS", (wrong, catalog))
    rows = sync(manager)
    assert rows[0]["status"] == "failed"
    assert "identity" in rows[0]["error"]
    assert rows[1]["status"] == "installed"
    assert set(manager.store.snapshot()[1]) == {catalog.name}


def test_concurrent_disable_during_fetch_wins(catalog, manager, monkeypatch):
    acquire = manager.acquire

    @contextmanager
    def disable_during_fetch(*args):
        with acquire(*args) as candidate:
            change_plugin(manager.config, catalog.name, False)
            yield candidate

    monkeypatch.setattr(manager, "acquire", disable_during_fetch)
    assert sync(manager)[0]["status"] == "skipped"
    assert not manager.store.ledger.exists()


def test_updates_keep_default_trust_without_enabling_directories(catalog, repository, manager):
    assert sync(manager)[0]["status"] == "installed"
    repository[1]("1.1")
    assert manager.update(catalog.name)["status"] == "updated"
    assert state(manager)["trust"]["default_managed"] is True
    assert SparkrunConfig(manager.config.config_path).feature_override("core.external_plugins") is None
    manager.rollback(catalog.name)
    assert state(manager)["trust"]["default_managed"] is True


def test_core_update_seeds_defaults_but_opt_out_is_read_only(catalog, manager, monkeypatch):
    from sparkrun.core.plugin_compatibility import HostSnapshot
    from sparkrun.core.plugin_update import reconcile_plugins

    monkeypatch.setattr(HostSnapshot, "current", classmethod(lambda cls: manager.host))
    with monkeypatch.context() as patch:
        patch.setattr(PluginManager, "acquire", lambda *a: pytest.fail("audit installed a default"))
        assert reconcile_plugins(manager.config, update=False)["defaults"] == []
    assert not manager.store.root.exists()
    report = reconcile_plugins(manager.config)
    assert report["defaults"][0]["status"] == "installed"
    assert next(row for row in report["plugins"] if row["name"] == catalog.name)["outcome"] == "installed"


def test_sync_cli_preview_and_setup_alias(catalog, manager, monkeypatch):
    from sparkrun.cli import main

    monkeypatch.setenv("SPARKRUN_CONFIG_DIR", str(manager.store.root))
    monkeypatch.setattr(PluginManager, "acquire", lambda *a: pytest.fail("preview fetched"))
    runner = CliRunner()
    for prefix in (["plugins"], ["setup", "plugins"]):
        result = runner.invoke(main, [*prefix, "sync", "--dry-run", "--json"])
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)[0]["status"] == "planned"
    assert not manager.store.root.exists()


def test_default_loads_in_fresh_process_without_loading_directory_plugins(catalog, repository, manager, tmp_path):
    _root, commit, _git = repository
    commit(
        body="""
SPARKRUN_PLUGIN_MANIFEST = {
    "api": 1,
    "features": {"fixture.capability": {"description": "Fixture", "default": True}},
}
from sparkrun.plugins import RegistryEntry, register_default_registry
def register(v):
    register_default_registry(
        RegistryEntry(name="default-fixture-recipes", url="https://example.invalid/recipes", subpath="recipes", trusted=True), owner=__name__
    )
"""
    )
    directory = tmp_path / "directories"
    directory.mkdir()
    (directory / "untrusted.py").write_text('raise AssertionError("directory plugin executed")\n')
    manager.config.set("plugins.paths", [str(directory)])
    manager.config.save()
    result = sync(manager)[0]
    assert result["status"] == "installed", result.get("error", result)
    from sparkrun.core.application_profile import child_environment

    script = """
import sys
from sparkrun.application import initialize
from sparkrun.core.features import is_feature_enabled
from sparkrun.core.plugin_inventory import list_plugins
sctx = initialize()
assert 'untrusted' not in sys.modules
assert not is_feature_enabled('core.external_plugins', config=sctx.config)
assert is_feature_enabled('fixture.capability', config=sctx.config)
row = next(p for p in list_plugins(sctx.config) if p.name == 'managedfixture')
assert row.loaded and row.enabled and row.source == 'default-managed'
from sparkrun.core.registry_defaults import iter_declared_registries
entry = next(d for d in iter_declared_registries() if d.entry.name == 'default-fixture-recipes')
assert entry.tier.value == 'out-of-tree'
assert entry.effective_entry().trusted is False
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, **child_environment(config_path=manager.config.config_path)},
        capture_output=True,
        text=True,
        timeout=40,
    )
    assert result.returncode == 0, result.stderr


def test_default_loading_honors_explicit_global_disable(catalog, manager):
    config = manager.config
    assert plugin_state(catalog.name, config, source=PluginSource.DEFAULT_MANAGED).enabled
    config.set("features", {"core.external_plugins": False})
    assert not plugin_state(catalog.name, config, source=PluginSource.DEFAULT_MANAGED).enabled


def test_legacy_loading_alias_uses_default_managed_policy(manager, monkeypatch):
    from sparkrun.core.features import is_feature_enabled
    from sparkrun.core.plugin_identity import PluginIdentity, reserve_identities

    monkeypatch.delenv("SPARKRUN_NO_EXTERNAL_PLUGINS", raising=False)
    reserve_identities([PluginIdentity("coldsnap", "coldsnap", ("default-managed", "coldsnap", "/fixture"))])
    assert is_feature_enabled("plugins.coldsnap", config=manager.config)
    manager.config.set("features", {"plugins.coldsnap": False})
    assert not is_feature_enabled("plugins.coldsnap", config=manager.config)


@pytest.mark.parametrize("field", ["source", "repository"])
def test_default_source_mismatch_is_rejected_before_execution(catalog, manager, monkeypatch, field):
    acquire = manager.acquire

    @contextmanager
    def wrong_source(*args):
        with acquire(*args) as candidate:
            if field == "source":
                candidate = replace(candidate, source={**candidate.source, "url": "https://example.invalid/other"})
            else:
                candidate = replace(candidate, metadata=replace(candidate.metadata, repository="https://example.invalid/other"))
            yield candidate

    monkeypatch.setattr(manager, "acquire", wrong_source)
    monkeypatch.setattr("sparkrun.core.plugin_manager.validate_candidate", lambda *a: pytest.fail("mismatched source executed"))
    row = sync(manager)[0]
    assert row["status"] == "failed", row
    assert not manager.store.ledger.exists()


def test_scoped_trust_still_requires_consent_for_changed_declarations(catalog, repository, manager):
    from sparkrun.core.plugin_manifest import PluginError

    assert sync(manager)[0]["status"] == "installed"
    before = state(manager)
    root, _commit, git = repository
    manifest = root / "plugin.toml"
    manifest.write_text(manifest.read_text().replace(root.as_uri(), "https://example.invalid/changed"))
    git("add", ".")
    git("commit", "-m", "changed repository declaration")
    with pytest.raises(PluginError, match="Source trust is required"):
        manager.update(catalog.name)
    assert state(manager) == before


@pytest.fixture
def profile_process(catalog, manager, tmp_path):
    """A real importable alternate profile, inherited by validation children."""
    (tmp_path / "default_profile.py").write_text(
        "from sparkrun.application import ApplicationProfile, DefaultPlugin\n"
        "PROFILE = ApplicationProfile(id='default-fixture', display_name='Default Fixture', "
        "command='default-fixture', package='default-fixture', profile_ref='default_profile:PROFILE', "
        f"default_plugins=(DefaultPlugin({catalog.name!r}, {catalog.source!r}, {catalog.module!r}, trusted=True),))\n"
    )
    env = {
        **os.environ,
        "PYTHONPATH": str(tmp_path) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        "SPARKRUN_APPLICATION_PROFILE": "default_profile:PROFILE",
        "SPARKRUN_APPLICATION_CONFIG": str(manager.config.config_path),
        "DEFAULT_FIXTURE_CONFIG_DIR": str(manager.store.root),
        "DEFAULT_FIXTURE_CACHE_DIR": str(tmp_path / "cache"),
        "DEFAULT_FIXTURE_STATE_DIR": str(tmp_path / "state"),
    }

    def run(script, **extra_env):
        result = subprocess.run([sys.executable, "-c", script], env={**env, **extra_env}, capture_output=True, text=True, timeout=45)
        assert result.returncode == 0, result.stderr
        return result

    return run


def test_first_initialization_installs_profile_default_once(catalog, manager, profile_process, repository):
    repository[1](
        body="""
SPARKRUN_PLUGIN_MANIFEST = {
    "api": 1,
    "features": {"fixture.capability": {"description": "Fixture", "default": True}},
    "hooks": {"plugin.install": "sparkrun.plugins.managedfixture:installed"},
}
def installed(context):
    from sparkrun.plugins import LifecyclePlan
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.features import is_feature_enabled
    config = SparkrunConfig()
    assert is_feature_enabled("fixture.capability", config=config)
    assert not is_feature_enabled("core.external_plugins", config=config)
    return LifecyclePlan(settings={"plugins.managedfixture.hook_seen": True})
"""
    )
    profile_process("""
from sparkrun.application import initialize
from sparkrun.core.plugin_inventory import list_plugins
sctx = initialize()
plugin = next(p for p in list_plugins(sctx.config) if p.name == 'managedfixture')
assert plugin.loaded and plugin.source == 'default-managed'
assert not sctx.config.is_feature_enabled('core.external_plugins')
assert sctx.config.get('plugins.managedfixture.hook_seen') is True
assert initialize().variables is sctx.variables
""")
    assert manager.store.ledger.exists()
    # A second process loads its existing generation without acquisition.
    profile_process("""
from sparkrun.core.plugin_manager import PluginManager
def fail(*args):
    raise AssertionError('existing default fetched')
PluginManager.acquire = fail
from sparkrun.application import initialize
from sparkrun.core.plugin_inventory import list_plugins
sctx = initialize()
assert next(p for p in list_plugins(sctx.config) if p.name == 'managedfixture').loaded
""")


def test_initialization_survives_default_install_failure(catalog, manager, profile_process):
    result = profile_process("""
from sparkrun.core.plugin_manager import PluginManager
def fail(*args):
    raise OSError('offline fixture')
PluginManager.acquire = fail
from sparkrun.application import initialize
assert initialize().variables is not None
""")
    assert "offline fixture" in result.stderr
    assert not manager.store.ledger.exists()


def test_completion_and_listing_do_not_provision_defaults(catalog, manager, profile_process):
    marker = manager.store.root.parent / "unexpected-fetch"
    guard = f"""
from pathlib import Path
from sparkrun.core.plugin_manager import PluginManager
def fail(*args):
    Path({str(marker)!r}).write_text('fetch attempted')
    raise AssertionError('read-only command fetched a default')
PluginManager.acquire = fail
from sparkrun.application import run_cli
"""
    result = profile_process(guard + "run_cli(args=['plugins', 'list', '--json'])")
    assert not any(row["name"] == catalog.name for row in json.loads(result.stdout))
    assert not manager.store.root.exists()
    result = profile_process(
        guard + "run_cli()",
        _DEFAULT_FIXTURE_COMPLETE="bash_complete",
        COMP_WORDS="default-fixture pl",
        COMP_CWORD="1",
    )
    assert "plugins" in result.stdout
    assert "read-only command fetched" not in result.stderr
    assert not manager.store.ledger.exists()
    assert not marker.exists()


def test_update_all_installs_missing_defaults_once(catalog, manager, monkeypatch):
    from sparkrun.cli import main

    monkeypatch.setenv("SPARKRUN_CONFIG_DIR", str(manager.store.root))
    monkeypatch.setattr("sparkrun.cli._plugins._manager", lambda ctx: manager)
    monkeypatch.setattr(manager, "update", lambda *a, **kw: pytest.fail("new default updated twice"))
    result = CliRunner().invoke(main, ["plugins", "update", "--all"])
    assert result.exit_code == 0, result.output
    assert "managedfixture: installed" in result.output
    assert state(manager)["trust"]["default_managed"] is True


def test_sync_reports_failure_with_json_and_nonzero_exit(catalog, manager, monkeypatch):
    from sparkrun.cli import main

    monkeypatch.setenv("SPARKRUN_CONFIG_DIR", str(manager.store.root))

    def offline(*args):
        raise OSError("offline fixture")

    monkeypatch.setattr(PluginManager, "acquire", offline)
    result = CliRunner().invoke(main, ["plugins", "sync", "--json"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)[0]["error"] == "offline fixture"
    assert not manager.store.ledger.exists()
