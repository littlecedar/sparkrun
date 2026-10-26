"""Shared management service for explicit plugin and feature actions."""

from __future__ import annotations
from copy import deepcopy
from contextlib import contextmanager
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from sparkrun.core.plugin_manifest import PluginError, read_declarations, module_parts
from sparkrun.core.plugin_policy import plugin_state, set_plugin_override, LEGACY_PLUGIN_FLAGS
from sparkrun.core.plugin_identity import normalize_plugin_id
from sparkrun.plugin_lifecycle import LifecycleContext, LifecyclePlan


def plugin_details(plugin):
    from sparkrun.core.plugin_metadata import plugin_origin

    metadata = plugin.management.get("metadata") if plugin.managed else None
    module = plugin.module.split(":")[0]
    if metadata:
        package = plugin.path / "payload" / Path(*module_parts(module))
        declarations = metadata.get("declarations", {})
    else:
        origin = plugin_origin(module, plugin.path)
        package = origin.parent if origin else None
        declarations = read_declarations(origin, module)
    return {
        "module": module,
        "package": str(package) if package else None,
        "metadata": metadata,
        "declarations": declarations,
        "default_managed": plugin.managed and plugin.source == "default-managed",
    }


def validate_activation(plugin, config):
    from sparkrun.core.plugin_compatibility import HostSnapshot, evaluate_plugin_compatibility, metadata_for_plugin
    from sparkrun.core.plugin_store import PluginStore

    if plugin.managed:
        PluginStore(config).receipt(plugin.name, plugin.management["active_generation"])
    compatibility = evaluate_plugin_compatibility(metadata_for_plugin(plugin), HostSnapshot.current())
    if compatibility.status == "incompatible":
        raise PluginError("Plugin %s is incompatible: %s" % (plugin.name, "; ".join(compatibility.reasons)))


def _plugins(config):
    from sparkrun.core.plugin_inventory import list_plugins

    return list_plugins(config)


def find_plugin(config, selector):
    identifier = normalize_plugin_id(selector)
    matches = [p for p in _plugins(config) if p.plugin_id == identifier or p.name == selector]
    if len(matches) != 1:
        raise PluginError("No unique plugin matches %r; use sparkrun plugins list" % selector)
    return matches[0]


def _run_worker(config, request, *, timeout=60):
    """Execute trusted plugin discovery/planning without importing it in the CLI."""
    from sparkrun.core.application_profile import child_environment
    from sparkrun.utils.fs import atomic_private_write

    with tempfile.TemporaryDirectory(prefix="sparkrun-lifecycle-") as directory:
        root = Path(directory)
        result_path = root / "result.json"
        request = {**request, "config": config._data, "config_path": str(config.config_path), "result": str(result_path)}
        request_path = root / "request.json"
        atomic_private_write(request_path, json.dumps(request))
        env = {
            **os.environ,
            **child_environment(config_path=config.config_path),
            "PYTHONDONTWRITEBYTECODE": "1",
            "SPARKRUN_NO_TELEMETRY": "1",
            "SPARKRUN_NO_DEFAULT_PLUGINS": "1",
        }
        try:
            result = subprocess.run(
                [sys.executable, "-m", "sparkrun.core.plugin_lifecycle_worker", str(request_path)],
                capture_output=True,
                stdin=subprocess.DEVNULL,
                text=True,
                env=env,
                timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise PluginError("Plugin lifecycle validation failed or timed out: %s" % exc) from exc
        if result.returncode or not result_path.exists():
            raise PluginError("Plugin lifecycle failed: " + (result.stderr or result.stdout or "no result")[-4000:])
        try:
            return json.loads(result_path.read_text())
        except (ValueError, OSError) as exc:
            raise PluginError("Invalid plugin lifecycle result: %s" % exc) from exc


def run_hook(config, plugin_id, details, event, *, feature=None, previous=None, current=None, proxy=None, validate=False, timeout=60):
    from sparkrun.feedback import get_feedback

    plugin_id = normalize_plugin_id(plugin_id)
    handler = details["declarations"].get("hooks", {}).get(event)
    if not handler and not validate:
        return LifecyclePlan()
    if proxy is None:
        from sparkrun.proxy.config import ProxyConfig

        proxy = ProxyConfig(config.config_path.parent / "proxy.yaml")._data
    context = LifecycleContext(event, plugin_id, deepcopy(config._data), deepcopy(proxy or {}), feature, previous, current)
    raw = _run_worker(config, {**details, "handler": handler, "context": asdict(context)}, timeout=timeout)
    try:
        plan = LifecyclePlan(**raw)
        if not isinstance(plan.settings, dict) or not isinstance(plan.proxy, dict):
            raise ValueError("settings/proxy must be mappings")
        if not isinstance(plan.messages, (list, tuple)) or any(not isinstance(m, str) for m in plan.messages):
            raise ValueError("messages must be strings")
        if any(not isinstance(k, str) or not k.startswith("plugins." + plugin_id + ".") for k in plan.settings):
            raise ValueError("hooks may only change their own plugin settings")
        if plan.settings and (plugin_id in {"enabled", "paths", "defaults"} or any(not all(k.split(".")) for k in plan.settings)):
            raise ValueError("hooks cannot write reserved or empty plugin setting paths")
        if plan.proxy and (event != "feature.enable" or set(plan.proxy) != {"gateway"}):
            raise ValueError("only explicit feature enable may select a gateway")
        if plan.proxy and (not isinstance(plan.proxy["gateway"], str) or feature not in details["declarations"].get("features", {})):
            raise ValueError("gateway selection requires an owned feature")
    except (TypeError, ValueError) as exc:
        raise PluginError("Invalid plugin lifecycle plan: %s" % exc) from exc
    get_feedback("plugins." + plugin_id).report("Planned " + event, level="info")
    return plan


def apply_plan(config, plan):
    for key, value in plan.settings.items():
        config.set(key, value)


def _result(name, status, plan, warning=None):
    from sparkrun.feedback import get_feedback

    for message in plan.messages:
        get_feedback("plugins." + name).report(message, level="info")
    return {"name": name, "status": status, "messages": list(plan.messages), "warning": warning, "restart_required": True}


@contextmanager
def management_transaction(store, dry_run):
    if dry_run:
        config, records = store.snapshot()
        yield config, records, (None, None)
    else:
        with store.mutation() as state:
            yield state


def change_plugin(config, selector, enabled, *, dry_run=False):
    from sparkrun.core.plugin_store import PluginStore

    store = PluginStore(config)
    with management_transaction(store, dry_run) as (config, records, baseline):
        from sparkrun.core.plugin_defaults import default_plugins, suppress_default

        identifier = normalize_plugin_id(selector)
        default = next((entry for entry in default_plugins(config.profile) if entry.name == identifier), None)
        if default is not None and not any(p.plugin_id == identifier for p in _plugins(config)):
            state = plugin_state(identifier, config, source="default-managed")
            if not enabled and state.required:
                raise PluginError("Plugin %s is required by this application and cannot be disabled" % identifier)
            set_plugin_override(config, identifier, enabled)
            if enabled:
                state = plugin_state(identifier, config, source="default-managed")
                if not state.enabled:
                    raise PluginError("Cannot enable %s: %s" % (identifier, state.reason))
                suppress_default(config, identifier, False)
            if dry_run:
                return {"name": identifier, "status": "planned", "enabled": enabled}
            warning = store.commit(records, config, baseline)
            return {
                "name": identifier,
                "status": "enabled" if enabled else "disabled",
                "messages": ["Run plugins sync or initialize the application to install this default."] if enabled else [],
                "warning": warning,
            }
        plugin = find_plugin(config, selector)
        identifier = plugin.plugin_id or normalize_plugin_id(plugin.name)
        current = plugin_state(identifier, config, source=plugin.source)
        if not enabled and current.required:
            raise PluginError("Plugin %s is required by this application and cannot be disabled" % identifier)
        details_warning = None
        try:
            details = plugin_details(plugin)
        except (PluginError, OSError, ValueError) as exc:
            if enabled:
                raise
            details_warning = "Plugin metadata unavailable; disabled without hooks: " + str(exc)
            details = {"declarations": {}}
        if enabled:
            validate_activation(plugin, config)
        before = deepcopy(config._data)
        set_plugin_override(config, identifier, enabled)
        if enabled:
            suppress_default(config, identifier, False)
        state = plugin_state(identifier, config, source=plugin.source)
        if enabled and not state.enabled:
            raise PluginError("Cannot enable %s: %s" % (identifier, state.reason))
        if dry_run:
            return {"name": identifier, "status": "planned", "enabled": enabled}
        hook_config = config
        if not enabled:
            from sparkrun.core.config import SparkrunConfig

            hook_config = SparkrunConfig(config.config_path)
            hook_config._data = before
        warning = details_warning
        plan = LifecyclePlan()
        try:
            if plugin.managed and details["declarations"].get("hooks", {}).get("plugin.disable"):
                store.receipt(plugin.name, plugin.management["active_generation"])
            plan = run_hook(hook_config, identifier, details, "plugin.enable" if enabled else "plugin.disable", validate=enabled)
        except PluginError as exc:
            if enabled:
                raise
            warning = "Disabled without completing its hook: " + str(exc)
        apply_plan(config, plan)
        cleanup = store.commit(records, config, baseline)
        return _result(identifier, "enabled" if enabled else "disabled", plan, warning or cleanup)


def change_feature(config, name, enabled, *, dry_run=False):
    from sparkrun.core.features import get_feature, _raw_feature_enabled, _env_override
    from sparkrun.core.plugin_store import PluginStore
    from sparkrun.proxy.config import ProxyConfig

    aliases = {flag: plugin for plugin, flag in LEGACY_PLUGIN_FLAGS.items()}
    if name in aliases:
        if enabled is None:
            raise PluginError("This is a plugin loading alias; use sparkrun plugins enable/disable " + aliases[name])
        return change_plugin(config, aliases[name], enabled, dry_run=dry_run)
    store = PluginStore(config)
    with management_transaction(store, dry_run) as (config, records, baseline):
        owners = []
        flag = get_feature(name)
        from sparkrun.core.features import FEATURE_OWNERS

        candidates = _plugins(config) if flag is None or name in FEATURE_OWNERS else []
        for plugin in candidates:
            try:
                details = plugin_details(plugin)
            except (PluginError, OSError, ValueError):
                continue  # An unrelated broken declaration cannot block recovery.
            if name in details["declarations"].get("features", {}):
                owners.append((plugin, details))
        if len(owners) > 1:
            raise PluginError("Feature %s is declared by multiple plugins" % name)
        flag = get_feature(name)
        # Already loaded dynamic declarations have ownership too.
        if not owners and flag is not None:
            from sparkrun.core.features import FEATURE_OWNERS

            identity = FEATURE_OWNERS.get(name)
            if identity:
                plugin = find_plugin(config, identity.plugin_id)
                owners.append((plugin, plugin_details(plugin)))
        if not owners and flag is None:
            discovered = _run_worker(config, {"inspect_feature": name})
            if discovered.get("owner"):
                plugin = find_plugin(config, discovered["owner"])
                owners.append((plugin, plugin_details(plugin)))
            if not discovered.get("known"):
                raise PluginError("Unknown feature %r. Its plugin may need installation or enabling." % name)
        if owners:
            plugin, details = owners[0]
            identifier = plugin.plugin_id or normalize_plugin_id(plugin.name)
            if enabled:
                validate_activation(plugin, config)
            if not plugin.enabled:
                raise PluginError(
                    "Feature belongs to disabled plugin %s. Enable it with sparkrun plugins enable %s" % (identifier, identifier)
                )
        else:
            identifier, details = "core", None
        features = deepcopy(config._data.get("features", {}))
        if enabled is None:
            features.pop(name, None)
        else:
            features[name] = enabled
        config.set("features", features)
        if enabled is True and _env_override(name, os.environ) is False:
            raise PluginError("Feature %s is disabled by an environment override; clear it before enabling" % name)
        if name.startswith("gateway.") and enabled and not _raw_feature_enabled("gateway", config=config):
            raise PluginError("Gateway support is disabled; enable it with sparkrun setup features enable gateway")
        if dry_run:
            return {"name": name, "status": "planned", "enabled": enabled}
        proxy = ProxyConfig(store.root / "proxy.yaml")
        proxy_before = store.read_proxy()
        plan = LifecyclePlan()
        if details and enabled is not None:
            plan = run_hook(
                config,
                identifier,
                details,
                "feature.enable" if enabled else "feature.disable",
                feature=name,
                proxy=proxy._data,
                validate=bool(enabled),
            )
        apply_plan(config, plan)
        warning = store.commit(records, config, baseline, proxy_updates=plan.proxy, proxy_baseline=proxy_before)
        return _result(name, "reset" if enabled is None else "enabled" if enabled else "disabled", plan, warning)
