"""Public plugin lifecycle planning API, version 1.

Handlers run for management actions, in a bounded child process.
They return configuration changes, never save files, start services or prompt.
Core validates and commits the plan. Repeated explicit enable requests invoke
handlers again; handlers must be idempotent. Startup only runs installation
hooks when provisioning a missing default, never for already installed plugins.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

PLUGIN_LIFECYCLE_API_VERSION = 1


@dataclass(frozen=True)
class LifecycleContext:
    event: str
    plugin_id: str
    config: dict[str, Any]
    proxy_config: dict[str, Any]
    feature: str | None = None
    previous: dict[str, Any] | None = None
    current: dict[str, Any] | None = None


@dataclass(frozen=True)
class LifecyclePlan:
    # Plugin-owned settings, expressed as dotted paths below plugins.<id>.
    settings: dict[str, Any] = field(default_factory=dict)
    proxy: dict[str, Any] = field(default_factory=dict)
    messages: tuple[str, ...] = ()


def select_gateway(context: LifecycleContext, name: str) -> LifecyclePlan:
    """Plan explicit gateway selection after validating its registered provider."""
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.features import is_feature_enabled
    from sparkrun.proxy.gateway import gateway_class, gateway_feature_flag, _GATEWAY_OWNERS
    from sparkrun.core.plugin_manifest import PluginError

    if context.event != "feature.enable" or context.feature is None or context.feature != gateway_feature_flag(name):
        raise PluginError("Gateway selection requires explicitly enabling its declared feature")
    owner = _GATEWAY_OWNERS.get(name)
    if getattr(owner, "plugin_id", None) != context.plugin_id:
        raise PluginError("A plugin can only select its own gateway provider")
    config = SparkrunConfig()
    if not is_feature_enabled("gateway", config=config):
        raise PluginError("Gateway support is disabled; enable it with sparkrun setup features enable gateway")
    if not is_feature_enabled(context.feature, config=config):
        raise PluginError("Gateway feature is blocked by its plugin or an environment override")
    gateway_class(name)
    return LifecyclePlan(proxy={"gateway": name}, messages=("Selected gateway: " + name + ". Running processes are unchanged.",))
