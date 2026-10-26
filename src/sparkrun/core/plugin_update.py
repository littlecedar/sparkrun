"""Shared plugin reconciliation for both core-update entry points."""

from __future__ import annotations

import json
import sys

from sparkrun.core.plugin_compatibility import HostSnapshot, evaluate_plugin_compatibility, metadata_for_plugin
from sparkrun.core.plugin_inventory import list_plugins
from sparkrun.core.plugin_manager import PluginManager
from sparkrun.core.plugin_manifest import PluginError


def reconcile_plugins(config, *, update: bool = True) -> dict:
    host = HostSnapshot.current()
    manager = PluginManager(config, host=host)
    rows = []
    requested_update = update
    update = update and bool(host.core_version)
    if update:
        with manager.store.mutation():
            pass
    from sparkrun.core.plugin_defaults import reconcile_default_plugins

    defaults = reconcile_default_plugins(config, manager=manager) if update else []
    installed = {row["name"]: row for row in defaults if row["status"] == "installed"}
    inventory = list_plugins(config)
    for plugin in inventory:
        before = evaluate_plugin_compatibility(metadata_for_plugin(plugin), host)
        row = {
            "name": plugin.name,
            "source": plugin.source,
            "managed": plugin.managed,
            "enabled": plugin.enabled,
            "compatibility": before.to_dict(),
            "outcome": "unmanaged" if not plugin.managed else "availability unchecked",
        }
        if plugin.name in installed:
            row.update(update=installed[plugin.name], outcome="installed")
        elif update and plugin.managed and host.core_version:
            try:
                result = manager.update(plugin.name, automatic=True)
                row["update"] = result
                row["outcome"] = result["status"]
                _config, records = manager.store.snapshot()
                if plugin.name in records:
                    row["compatibility_after"] = evaluate_plugin_compatibility(records[plugin.name]["metadata"], host).to_dict()
            except (PluginError, OSError) as exc:
                row.update(outcome="failed", error=str(exc))
        rows.append(row)
    report = {"protocol": 1, "host": host.identity(), "update_plugins": update, "plugins": rows, "defaults": defaults}
    if requested_update and not update:
        report["warning"] = "Plugin compatibility audit incomplete: host core identity is unknown; no repairs attempted"
    if update:
        from sparkrun.utils.fs import atomic_private_write

        root = config.config_path.parent
        root.mkdir(parents=True, exist_ok=True)
        atomic_private_write(root / "plugins-update-report.json", json.dumps(report, indent=2))
    return report


def main():
    from sparkrun.core.application_profile import initialize_child_application_profile
    from sparkrun.core.config import SparkrunConfig

    initialize_child_application_profile()
    if sys.argv[1:3] != ["--protocol", "1"]:
        raise PluginError("Unsupported plugin reconciliation protocol")
    print(json.dumps(reconcile_plugins(SparkrunConfig(), update="--no-update-plugins" not in sys.argv)))


if __name__ == "__main__":
    main()
