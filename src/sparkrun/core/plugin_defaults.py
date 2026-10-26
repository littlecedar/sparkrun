"""Curated, independently released plugins supplied by an application.

Catalog entries are trusted application policy, never declarations accepted
from plugin code. Reconciliation only fills gaps; the managed store owns all
subsequent updates and user-selected sources and pins.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from contextvars import ContextVar
import os
import re
from urllib.parse import urlsplit


@dataclass(frozen=True)
class DefaultPlugin:
    """Expected identity and acquisition policy for one optional integration.

    ``trusted=True`` authorizes executing this source and future compatible
    updates. The catalog never changes an existing installation's source/ref.
    """

    name: str
    source: str
    module: str
    ref: str = "main"
    trusted: bool = False

    def __post_init__(self):
        from sparkrun.core.plugin_identity import normalize_plugin_id
        from sparkrun.core.plugin_manifest import module_parts, normalize_source

        if normalize_plugin_id(self.name) != self.name:
            raise ValueError("Default plugin names must be canonical: %r" % self.name)
        module_parts(self.module)
        # Local file URLs are useful for offline distributions and tests. Do not
        # probe the filesystem while importing an application profile.
        if isinstance(self.source, str) and self.source.startswith("file:///"):
            parsed = urlsplit(self.source)
            if parsed.netloc or parsed.query or parsed.fragment or not parsed.path.strip("/"):
                raise ValueError("Default plugin file URL must identify a local repository without query or fragment")
        else:
            normalize_source(self.source)
        if not isinstance(self.ref, str) or not re.fullmatch(r"[A-Za-z0-9_./-]+", self.ref) or self.ref.startswith("-") or ".." in self.ref:
            raise ValueError("Invalid default plugin Git ref: %r" % self.ref)
        if type(self.trusted) is not bool:
            raise TypeError("Default plugin trusted must be a boolean")


# Adding an entry is an explicit release policy decision, including its trust.
DEFAULT_PLUGINS: tuple[DefaultPlugin, ...] = ()
_INSTALL_DEFAULTS = ContextVar("install_default_plugins", default=True)


@contextmanager
def suspend_default_installation():
    """Keep completion and other inspection contexts free of installation."""
    token = _INSTALL_DEFAULTS.set(False)
    try:
        yield
    finally:
        _INSTALL_DEFAULTS.reset(token)


def default_plugins(profile=None) -> tuple[DefaultPlugin, ...]:
    from sparkrun.core.application_profile import get_application_profile

    profile = profile or get_application_profile()
    return DEFAULT_PLUGINS if profile.default_plugins is None else profile.default_plugins


def suppressed_defaults(config) -> set[str]:
    from sparkrun.core.plugin_identity import normalize_plugin_id
    from sparkrun.core.plugin_manifest import PluginError

    values = config.get("plugins.defaults.suppressed", [])
    if not isinstance(values, (list, tuple)) or any(not isinstance(value, str) for value in values):
        raise PluginError("plugins.defaults.suppressed must be a list of plugin IDs")
    return {normalize_plugin_id(value) for value in values}


def suppress_default(config, name: str, suppressed: bool) -> None:
    names = suppressed_defaults(config)
    before = set(names)
    if suppressed:
        names.add(name)
    else:
        names.discard(name)
    if names != before:
        config.set("plugins.defaults.suppressed", sorted(names))


def default_skip_reason(entry: DefaultPlugin, config) -> str | None:
    from sparkrun.core.plugin_manifest import PluginError
    from sparkrun.core.plugin_policy import PluginSource, plugin_state

    enabled = config.get("plugins.defaults.enabled", True)
    if type(enabled) is not bool:
        raise PluginError("plugins.defaults.enabled must be a boolean")
    if not enabled or not _INSTALL_DEFAULTS.get() or os.environ.get("SPARKRUN_NO_DEFAULT_PLUGINS", "").lower() in {"1", "true", "yes"}:
        return "default plugin installation disabled"
    if entry.name in suppressed_defaults(config):
        return "removed by user"
    state = plugin_state(entry.name, config, source=PluginSource.DEFAULT_MANAGED)
    if not state.enabled:
        return state.reason
    if not entry.trusted:
        return "source trust required; install explicitly with plugins add"
    return None


def validate_default_candidate(entry: DefaultPlugin, candidate) -> None:
    from sparkrun.core.plugin_manifest import PluginError, normalize_source, repository_identity

    metadata = candidate.metadata
    if metadata.name != entry.name or metadata.module != entry.module:
        raise PluginError("Default plugin identity does not match the catalog declaration")
    expected = normalize_source(entry.source, local=True)
    if candidate.source.get("url") != expected or normalize_source(candidate.source.get("fetch_url", expected), local=True) != expected:
        raise PluginError("Default plugin source does not match the catalog declaration")
    if repository_identity(metadata.repository) != repository_identity(expected):
        raise PluginError("Default plugin manifest declares a different repository")


def reconcile_default_plugins(config, *, dry_run: bool = False, manager=None) -> list[dict]:
    """Install missing defaults, isolating failures and preserving existing providers.

    A dry run reads local metadata only. Installation rechecks policy under the
    manager's mutation lock, so a concurrent disable/remove wins over a fetch.
    """
    from sparkrun.core.plugin_inventory import list_plugins
    from sparkrun.core.plugin_manager import PluginManager

    entries = default_plugins(config.profile)
    if not entries:
        return []
    manager = manager or PluginManager(config)
    rows = []
    for entry in entries:
        row = {"name": entry.name, "source": entry.source, "ref": entry.ref}
        try:
            current, records = manager.store.snapshot()
            if entry.name in records:
                row.update(status="present")
            elif reason := default_skip_reason(entry, current):
                row.update(status="skipped", reason=reason)
            else:
                providers = list_plugins(current)
                if any(p.plugin_id == entry.name or p.module.split(":")[0] == entry.module for p in providers):
                    row.update(status="present", reason="existing unmanaged provider preserved")
                elif dry_run:
                    row.update(status="planned")
                else:
                    with manager.acquire(entry.source, entry.ref) as candidate:
                        row.update(manager.install(candidate, _default=entry))
        except (OSError, ValueError, RuntimeError) as exc:
            row.update(status="failed", error=str(exc))
        rows.append(row)
    return rows
