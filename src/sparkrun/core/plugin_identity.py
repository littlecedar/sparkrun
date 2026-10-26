"""Loader-owned identities for plugin contributions; no server dependencies."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
import re
import tomllib

from sparkrun.core.registration import enlist_registry_state


def normalize_plugin_id(name: object) -> str:
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,99}", name) or not name[-1].isalnum() or ".." in name:
        raise ValueError("Invalid plugin identity: %r" % name)
    result = re.sub(r"[_.-]+", "-", name.lower())
    if result in {"core", "plugins"}:
        raise ValueError("Reserved plugin identity: %s" % result)
    return result


@dataclass(frozen=True)
class PluginIdentity:
    plugin_id: str
    module: str
    source: tuple[str, str, str | None]

    def __post_init__(self):
        if normalize_plugin_id(self.plugin_id) != self.plugin_id or not self.module:
            raise ValueError("Plugin identity must be canonical and identify its module")


_IDENTITIES: dict[str, PluginIdentity] = {}
enlist_registry_state(globals(), "_IDENTITIES")
_CURRENT: ContextVar[PluginIdentity | str | None] = ContextVar("sparkrun_plugin_identity", default=None)


def current_registration_owner():
    return _CURRENT.get()


# plugin_id -> conflict message. Deliberately not enlisted for rollback: once
# two sources claim one ID, neither may load in this process, whatever rolls back.
_CONTESTED: dict[str, str] = {}


def _contest(plugin_id, claimants):
    from sparkrun.core.registration import PluginConflictError, record_plugin_conflict, record_plugin_load_failure

    message = _CONTESTED.get(plugin_id) or "Plugin identity %r is claimed by %s" % (
        plugin_id,
        " and ".join(repr(identity.source) for identity in claimants),
    )
    _CONTESTED[plugin_id] = message
    record_plugin_conflict(message)
    error = PluginConflictError(message)
    for identity in claimants:
        record_plugin_load_failure(identity.source, error)
    return message


def preflight_identities(identities):
    """Check the whole selected set before reserving any identity or importing code.

    Strict: raises ``PluginConflictError`` on the first contested ID. Loaders
    use :func:`reserve_identities`, which skips only the contested claims.
    """
    from sparkrun.core.registration import PluginConflictError

    rejected = reserve_identities(identities)
    if rejected:
        raise PluginConflictError(_CONTESTED[next(iter(rejected)).plugin_id])


def reserve_identities(identities) -> frozenset[PluginIdentity]:
    """Reserve every uncontested identity; return the claims that were rejected.

    Two sources claiming one plugin ID cannot be arbitrated by import order, so
    every claimant of a contested ID is rejected (an identity reserved earlier
    in this process keeps its reservation). Each conflict is recorded as a load
    failure for every claimant and as a launch-blocking conflict
    (``require_integrations``), so the rest of the plugins still load.
    """
    identities = list(dict.fromkeys(identities))
    claims: dict[str, list[PluginIdentity]] = {}
    for identity in identities:
        claims.setdefault(identity.plugin_id, []).append(identity)
    rejected: set[PluginIdentity] = set()
    for plugin_id, claimants in claims.items():
        previous = _IDENTITIES.get(plugin_id)
        rivals = list(dict.fromkeys([*([previous] if previous is not None else []), *claimants]))
        if len(rivals) > 1 or plugin_id in _CONTESTED:
            _contest(plugin_id, rivals)
            rejected.update(identity for identity in claimants if identity != previous)
    for identity in identities:
        if identity not in rejected:
            _IDENTITIES.setdefault(identity.plugin_id, identity)
    return frozenset(rejected)


@contextmanager
def plugin_registration(identity: PluginIdentity):
    previous = _CURRENT.get()
    if previous is not None and previous != identity:
        raise ValueError("Cannot change plugin ownership during registration")
    preflight_identities([identity])
    token = _CURRENT.set(identity)
    try:
        yield identity
    finally:
        _CURRENT.reset(token)


@contextmanager
def core_registration():
    if _CURRENT.get() not in (None, "core"):
        raise ValueError("Plugins cannot register core operations")
    token = _CURRENT.set("core")
    try:
        yield
    finally:
        _CURRENT.reset(token)


@contextmanager
def application_daemon_registration():
    """Bind embedded application operations to the active profile's stable namespace."""
    from sparkrun.core.application_profile import get_application_profile
    from sparkrun.core.registration import registry_transaction

    profile = get_application_profile()
    identity = PluginIdentity(normalize_plugin_id("app-" + profile.id), profile.package, ("application", profile.id, profile.package))
    with registry_transaction(), plugin_registration(identity):
        yield identity


def discovered_identity(name, module, source, *, origin=None, match_name=False):
    """Read nearby static metadata only when it names this exact module."""
    plugin_id = normalize_plugin_id(name)
    if origin:
        root = Path(origin).resolve().parent
        # Package-local manifests and conventional repo/{src/,}package layouts.
        for parent in (root, *list(root.parents)[:3]):
            path = parent / "plugin.toml"
            if not path.is_file():
                continue
            data = tomllib.loads(path.read_text(encoding="utf-8"))
            if data.get("module") != module:
                continue
            declared = normalize_plugin_id(data.get("name"))
            if match_name and declared != plugin_id:
                raise ValueError("Plugin manifest identity %r disagrees with entry point %r" % (declared, name))
            plugin_id = declared
            break
    return PluginIdentity(plugin_id, module, source)


def registered_plugin_identity(module_name):
    return next((identity for identity in _IDENTITIES.values() if identity.module == module_name), None)


def registered_plugin_by_id(plugin_id):
    return _IDENTITIES.get(normalize_plugin_id(plugin_id))


def clear_plugin_identities():
    """Test isolation, matching clear_loaded_plugin_modules."""
    _IDENTITIES.clear()
    _CONTESTED.clear()


def preflight_plugin_sources(v, config=None) -> frozenset[PluginIdentity]:
    """Inspect every selected source before the first cross-cutting plugin import.

    Never raises for one bad source: a source whose discovery fails (malformed
    ``integrations:``, an unreadable store) is recorded and contributes nothing,
    and contested identities are rejected without blocking the others. Returns
    the rejected identities.
    """
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.in_tree_plugins import in_tree_plugin_identities
    from sparkrun.core.external_plugins import configured_plugin_paths, directory_plugin_identities
    from sparkrun.core.installed_plugins import _discover_installed_plugins, installed_plugin_identities
    from sparkrun.core.managed_plugins import managed_plugin_identities, validating_candidate
    from sparkrun.core.registration import record_plugin_load_failure
    import logging

    config = config or SparkrunConfig()
    sources = (
        ("in-tree", lambda: in_tree_plugin_identities(v)),
        ("external", lambda: directory_plugin_identities(configured_plugin_paths(v), config=config)),
        ("managed", lambda: managed_plugin_identities(config)),
        ("installed", lambda: installed_plugin_identities(_discover_installed_plugins(config)[0])),
    )
    selected = []
    for kind, discover in sources:
        try:
            selected.extend(discover().values())
        except Exception as exc:  # noqa: BLE001 - one source must not stop the others or bootstrap
            if validating_candidate():
                raise
            record_plugin_load_failure((kind, "<discovery>", None), exc)
            logging.getLogger(__name__).warning("Skipping %s plugins: %s", kind, exc)
    return reserve_identities(selected)
