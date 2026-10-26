"""Version 1 of the application profile contract. Importing this module performs no I/O.

A profile is trusted, installed Python data, selected before any application
settings are read. Integration installation and registry trust remain separate.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any
from sparkrun.utils.data import thaw as thaw
from sparkrun.core.plugin_defaults import DefaultPlugin

APPLICATION_PROFILE_API_VERSION = 1
_NAME = re.compile(r"[a-z][a-z0-9-]{0,47}\Z")


def freeze(value: Any) -> Any:
    """Validate and freeze profile data without widening its scalar policy."""
    from sparkrun.utils.data import freeze as freeze_data

    return freeze_data(value, label="Profile data")


@dataclass(frozen=True)
class UpdateSource:
    requirement: str
    strategy: str = "package"

    def __post_init__(self):
        if self.strategy not in {"package", "git"} or not self.requirement or self.requirement.startswith("-"):
            raise ValueError("Update source requires a requirement and package/git strategy")


@dataclass(frozen=True)
class ApplicationProfile:
    id: str
    display_name: str
    command: str
    package: str
    description: str = "Launch and manage inference workloads."
    documentation_url: str | None = None
    support_url: str | None = None
    config_namespace: str = ""
    cache_namespace: str = ""
    state_namespace: str = ""
    resource_namespace: str = ""
    env_prefix: str = ""
    env_aliases: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    defaults: Mapping[str, Any] = field(default_factory=dict)
    integrations: tuple[str, ...] = ()
    required_integrations: tuple[str, ...] = ()
    feature_defaults: Mapping[str, bool] = field(default_factory=dict)
    feature_channel_defaults: Mapping[str, Mapping[str, bool]] = field(default_factory=dict)
    # Replacement catalog, never appended to Sparkrun's defaults. None selects
    # the built-in Sparkrun policy; () ships no registries. User entries still win.
    registries: tuple[Mapping[str, Any], ...] | None = ()
    bootstrap_registry_urls: tuple[str, ...] = ()
    default_channel: str = "stable"
    update_sources: Mapping[str, UpdateSource] = field(default_factory=dict)
    # None preserves Sparkrun's historical coupling to its update channel.
    feature_channel: str | None = "stable"
    telemetry_enabled: bool = False
    telemetry_endpoint: str | None = None
    telemetry_key: str = ""
    hardware_fallback: str = "require-metadata"
    # Importable profile reference propagated to child Python processes.
    profile_ref: str | None = None
    # None selects Sparkrun's catalog; alternate applications default to none.
    default_plugins: tuple[DefaultPlugin, ...] | None = ()

    def __post_init__(self):
        if self.default_plugins is not None:
            if not isinstance(self.default_plugins, (tuple, list)) or any(
                not isinstance(entry, DefaultPlugin) for entry in self.default_plugins
            ):
                raise TypeError("default_plugins must be a sequence of DefaultPlugin entries")
            for attr in ("name", "module"):
                values = [getattr(entry, attr) for entry in self.default_plugins]
                if len(values) != len(set(values)):
                    raise ValueError("Duplicate default plugin %s" % attr)
            object.__setattr__(self, "default_plugins", tuple(self.default_plugins))
        elif self.id != "sparkrun":
            raise ValueError("Alternate application profiles must declare their default plugin policy explicitly")
        for attr in ("id", "command", "config_namespace", "cache_namespace", "state_namespace", "resource_namespace"):
            value = getattr(self, attr)
            if attr.endswith("_namespace") and value == "":
                value = self.id
                object.__setattr__(self, attr, value)
            if not isinstance(value, str) or not _NAME.fullmatch(value):
                raise ValueError("Invalid application profile %s: %r" % (attr, value))
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", self.package):
            raise ValueError("Invalid application package: %r" % self.package)
        if not isinstance(self.env_prefix, str):
            raise ValueError("Invalid environment prefix: %r" % self.env_prefix)
        prefix = self.env_prefix or self.id.upper().replace("-", "_")
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", prefix):
            raise ValueError("Invalid environment prefix: %r" % prefix)
        object.__setattr__(self, "env_prefix", prefix)
        if not isinstance(self.display_name, str) or not self.display_name.strip():
            raise ValueError("display_name must be nonempty")
        for attr in ("defaults", "feature_defaults", "feature_channel_defaults", "env_aliases"):
            if not isinstance(getattr(self, attr), Mapping):
                raise TypeError("%s must be a mapping" % attr)
        if any(type(value) is not bool for value in self.feature_defaults.values()):
            raise TypeError("feature_defaults must contain booleans")
        for channel, defaults in self.feature_channel_defaults.items():
            if not isinstance(channel, str) or not _NAME.fullmatch(channel):
                raise ValueError("Invalid application feature channel: %r" % channel)
            if not isinstance(defaults, Mapping):
                raise TypeError("feature_channel_defaults must map channels to flag mappings")
            if any(type(value) is not bool for value in defaults.values()):
                raise TypeError("feature_channel_defaults must contain booleans")
        for setting, aliases in self.env_aliases.items():
            if not isinstance(setting, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]*", setting):
                raise ValueError("env_aliases keys must be setting suffixes")
            if not isinstance(aliases, (tuple, list)) or any(not isinstance(alias, str) for alias in aliases):
                raise TypeError("env_aliases values must be sequences of variable names")
            if any(not re.fullmatch(r"[A-Z][A-Z0-9_]*", alias) for alias in aliases):
                raise ValueError("Invalid env_aliases variable name")
        for attr in ("defaults", "feature_defaults", "feature_channel_defaults", "env_aliases"):
            object.__setattr__(self, attr, freeze(getattr(self, attr)))
        if self.registries is not None:
            if not isinstance(self.registries, (tuple, list)):
                raise TypeError("registries must be a sequence of registry mappings")
            names = set()
            for entry in self.registries:
                if not isinstance(entry, Mapping):
                    raise TypeError("registries must contain registry mappings")
                name, url = entry.get("name"), entry.get("url")
                if not isinstance(name, str) or not name or not isinstance(url, str) or not url.strip():
                    raise ValueError("Each profile registry requires a name and URL")
                if name in names:
                    raise ValueError("Duplicate application profile registry: %r" % name)
                names.add(name)
            object.__setattr__(self, "registries", freeze(self.registries))
        for attr in ("integrations", "required_integrations", "bootstrap_registry_urls"):
            values = getattr(self, attr)
            if not isinstance(values, (tuple, list)) or any(not isinstance(value, str) for value in values):
                raise TypeError("%s must be a sequence of strings" % attr)
            for value in values:
                if not value.strip() or (attr != "bootstrap_registry_urls" and not _NAME.fullmatch(value)):
                    raise ValueError("Invalid %s entry: %r" % (attr, value))
            object.__setattr__(self, attr, tuple(values))
        sources = dict(self.update_sources)
        if any(not isinstance(s, UpdateSource) for s in sources.values()):
            raise TypeError("update_sources values must be UpdateSource instances")
        for channel, source in sources.items():
            if not isinstance(channel, str) or not _NAME.fullmatch(channel):
                raise ValueError("Invalid update channel")
            requirement_name = re.match(r"([A-Za-z0-9][A-Za-z0-9._-]*)(?=\s|\[|[<>=!~@]|$)", source.requirement)

            def normalize(name):
                return re.sub(r"[-_.]+", "-", name).lower()

            if not requirement_name or normalize(requirement_name[1]) != normalize(self.package):
                raise ValueError("Update source must name the owning distribution package %r" % self.package)
            if "\n" in source.requirement or "\r" in source.requirement:
                raise ValueError("Update requirement must be a single line")
        if sources and self.default_channel not in sources:
            raise ValueError("Default update channel is not supported")
        unsupported = self.feature_channel_defaults.keys() - (sources.keys() | {self.default_channel})
        if unsupported:
            raise ValueError("Feature defaults refer to unsupported application channels: %s" % ", ".join(sorted(unsupported)))
        object.__setattr__(self, "update_sources", MappingProxyType(sources))
        if not isinstance(self.hardware_fallback, str) or not self.hardware_fallback.strip():
            raise ValueError("Invalid hardware fallback policy")
        if self.registries is None and self.id != "sparkrun":
            raise ValueError("Alternate application profiles must declare their registry policy explicitly")
        if self.profile_ref and not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*", self.profile_ref):
            raise ValueError("profile_ref must be an importable module:attribute")


SPARKRUN = ApplicationProfile(
    id="sparkrun",
    display_name="Sparkrun",
    command="sparkrun",
    package="sparkrun",
    description="sparkrun — Launch inference workloads on NVIDIA DGX Spark systems.",
    documentation_url="https://sparkrun.dev",
    support_url="https://github.com/spark-arena/sparkrun/issues",
    registries=None,
    default_plugins=None,
    feature_channel=None,
    telemetry_enabled=True,
    telemetry_endpoint="https://telemetry.sparkrun.dev/",
    telemetry_key="sparkrun-telemetry-v1",
    hardware_fallback="dgx-spark",
    profile_ref="sparkrun.core.application_profile:SPARKRUN",
    update_sources={
        "stable": UpdateSource("sparkrun"),
        "beta": UpdateSource("sparkrun @ git+https://github.com/spark-arena/sparkrun@develop", "git"),
        "alpha": UpdateSource("sparkrun @ git+https://github.com/spark-arena/sparkrun@develop-next", "git"),
    },
)

_active: ApplicationProfile | None = None


def select_application_profile(profile: ApplicationProfile | None = None) -> ApplicationProfile:
    global _active
    if profile is None:
        profile = _active or SPARKRUN
    if not isinstance(profile, ApplicationProfile):
        raise TypeError("Expected a ApplicationProfile")
    if _active is not None and _active != profile:
        raise RuntimeError("Application profile %r is already selected; cannot switch to %r in this process" % (_active.id, profile.id))
    _active = profile
    return profile


def get_application_profile() -> ApplicationProfile:
    if _active is None and os.environ.get("SPARKRUN_APPLICATION_PROFILE"):
        return initialize_child_application_profile()
    return select_application_profile()


def _reset_application_profile_for_tests() -> None:
    global _active
    _active = None


def env_name(setting: str) -> str:
    return "%s_%s" % (get_application_profile().env_prefix, setting)


def product_env(setting: str, default: Any = None, *, environ: Mapping[str, str] | None = None) -> Any:
    environ = os.environ if environ is None else environ
    for name in (env_name(setting), *get_application_profile().env_aliases.get(setting, ())):
        if name in environ:
            return environ[name]
    return default


def product_path(kind: str) -> Path:
    profile = get_application_profile()
    override = product_env(kind.upper() + "_DIR")
    if override is not None:
        return Path(override).expanduser()
    # Preserve the historical roots: adding application profiles is not an XDG migration.
    parent = ".cache" if kind == "cache" else ".config"
    return Path.home() / parent / getattr(profile, kind + "_namespace")


def resource_name(suffix: str = "") -> str:
    return get_application_profile().resource_namespace + suffix


def child_environment(*, config_path: str | Path | None = None) -> dict[str, str]:
    """Environment variables that bind a child process to this application.

    Merge the result over the child's environment so a subprocess that calls
    :func:`sparkrun.application.initialize` (or ``run_cli``) selects the same
    application profile and, when *config_path* is given, the same
    configuration file. Returns only the binding variables, not the parent's
    whole environment.

    Raises:
        RuntimeError: The selected profile has no ``profile_ref`` to export.
    """
    profile = get_application_profile()
    if not profile.profile_ref:
        raise RuntimeError("Application profile %r must provide profile_ref for child execution" % profile.id)
    environment = {"SPARKRUN_APPLICATION_PROFILE": profile.profile_ref}
    if config_path is not None:
        environment["SPARKRUN_APPLICATION_CONFIG"] = str(Path(config_path).expanduser().absolute())
    return environment


def child_config_path() -> Path | None:
    """Optional same-controller config; ignored when its profile does not match."""
    if os.environ.get("SPARKRUN_APPLICATION_PROFILE") != get_application_profile().profile_ref:
        return None
    value = os.environ.get("SPARKRUN_APPLICATION_CONFIG")
    return Path(value) if value else None


def initialize_child_application_profile(reference: str | None = None) -> ApplicationProfile:
    import importlib

    reference = reference or os.environ.get("SPARKRUN_APPLICATION_PROFILE")
    if reference:
        module, attr = reference.split(":", 1)
        return select_application_profile(getattr(importlib.import_module(module), attr))
    return select_application_profile()


def remote_cache_path(suffix: str = "", *, home: str = "$HOME") -> str:
    """A remote-home-relative default, for generated shell and transfer contexts.

    When a concrete absolute path is needed, use probe_remote_sparkrun_cache:
    only the target host can resolve its own home and cache environment.
    """
    root = home + "/.cache/" + get_application_profile().cache_namespace
    return root + ("/" + suffix.lstrip("/") if suffix else "")


def render_identity_text(text: str) -> str:
    """Render explicit product tokens in human-facing command text."""
    profile = get_application_profile()
    return text.replace("{app_command}", profile.command).replace("{resource_namespace}", profile.resource_namespace)
