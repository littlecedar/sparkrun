"""Isolated hook runner, including installation of missing application defaults."""

from __future__ import annotations
import importlib
import json
from pathlib import Path
import sys
from dataclasses import asdict


def main():
    from sparkrun.core.application_profile import initialize_child_application_profile
    from sparkrun.core.config import SparkrunConfig
    import sparkrun.core.config as configuration
    from sparkrun.core.bootstrap import init_sparkrun
    from sparkrun.core.plugin_manifest import PluginMetadata, PluginError
    from sparkrun.core.external_plugins import loaded_plugin_module
    from sparkrun.plugin_lifecycle import LifecycleContext, LifecyclePlan
    import sparkrun.core.managed_plugins as managed

    initialize_child_application_profile()
    request = json.loads(Path(sys.argv[1]).read_text())
    configuration._lifecycle_snapshot = (Path(request["config_path"]).resolve(), request["config"])
    if request.get("metadata"):
        metadata = PluginMetadata.from_dict(request["metadata"])
        record = {"metadata": metadata.to_dict()}
        if request.get("default_managed") is True:
            record["trust"] = {"granted": True, "default_managed": True}
        managed._validation_candidate = (record, Path(request["package"]))
    config = SparkrunConfig(request["config_path"])
    init_sparkrun(config=config)
    if "inspect_feature" in request:
        from sparkrun.core.features import get_feature, FEATURE_OWNERS

        name = request["inspect_feature"]
        owner = FEATURE_OWNERS.get(name)
        Path(request["result"]).write_text(
            json.dumps({"known": get_feature(name) is not None, "owner": owner.plugin_id if owner else None})
        )
        return
    module = loaded_plugin_module(request["module"])
    if module is None:
        raise PluginError("Plugin did not load; inspect compatibility and loading policy")
    result = LifecyclePlan()
    if request.get("handler"):
        package, function = request["handler"].split(":")
        handler = getattr(importlib.import_module(package), function)
        context = LifecycleContext(**request["context"])
        result = handler(context) or LifecyclePlan()
        if not isinstance(result, LifecyclePlan):
            raise PluginError("Lifecycle hooks must return LifecyclePlan or None")
    if result.proxy:
        from sparkrun.plugin_lifecycle import select_gateway

        gateway = result.proxy.get("gateway")
        if not isinstance(gateway, str):
            raise PluginError("Gateway selection must name a provider")
        select_gateway(LifecycleContext(**request["context"]), gateway)
    Path(request["result"]).write_text(json.dumps(asdict(result)))


if __name__ == "__main__":
    main()
