"""Load explicit managed package origins, without extending the host namespace."""

from __future__ import annotations

from functools import partial
import importlib
from importlib.machinery import PathFinder
import importlib.util
import logging
from pathlib import Path
import sys

from sparkrun.core.plugin_manifest import PluginError, PluginMetadata, module_parts
from sparkrun.core.registration import load_and_register_plugin
from sparkrun.core.registry_defaults import DeclarationTier

logger = logging.getLogger(__name__)
# Set only by the isolated validation helper, never by normal startup.
_validation_candidate: tuple[dict, Path] | None = None


def physical_origin(module: str) -> Path | None:
    """Find physical providers without importing any candidate plugin."""
    parts = module.split(".")
    search = None
    spec = None
    for index in range(len(parts)):
        spec = PathFinder.find_spec(".".join(parts[: index + 1]), search)
        if spec is None:
            return None
        search = spec.submodule_search_locations
        if index < len(parts) - 1 and search is None:
            return None
    return Path(spec.origin) if spec and spec.origin else None


def check_collisions(metadata: PluginMetadata, records: dict, config, *, replacing: str | None = None):
    from sparkrun.core.external_plugins import iter_plugin_module_names
    from sparkrun.core.installed_plugins import discover_installed_plugins
    from sparkrun.core.in_tree_plugins import iter_in_tree_plugin_names

    for name, record in records.items():
        if name != replacing and (name == metadata.name or record["metadata"]["module"] == metadata.module):
            raise PluginError("Plugin %s conflicts with managed %s (%s)" % (metadata.name, name, record["metadata"]["module"]))
    origin = physical_origin(metadata.module)
    if origin is not None:
        raise PluginError("Module %s already belongs to %s; remove the existing installation first" % (metadata.module, origin))
    for name in iter_in_tree_plugin_names():
        if name == metadata.name:
            raise PluginError("Plugin name %s already belongs to the bundled/installed namespace" % name)
    for path in config.external_plugin_paths:
        names = iter_plugin_module_names(path)
        if metadata.name in names or metadata.module.split(".")[0] in names:
            raise PluginError("Plugin %s conflicts with legacy plugin path %s" % (metadata.name, path))
    for installed in discover_installed_plugins(config):
        if installed.package and (installed.name == metadata.name or installed.module == metadata.module):
            raise PluginError("Plugin %s conflicts with installed integration %s (%s)" % (metadata.name, installed.name, installed.package))


def _import_package(module: str, package: Path):
    parts = module_parts(module)
    origin = (package / "__init__.py").resolve()
    previous = sys.modules.get(module)
    if previous is not None:
        previous_file = getattr(previous, "__file__", None)
        if previous_file and Path(previous_file).resolve() == origin:
            return previous
        raise PluginError("Module %s is already loaded from a different origin" % module)
    parent = importlib.import_module(".".join(parts[:-1])) if len(parts) > 1 else None
    spec = importlib.util.spec_from_file_location(module, origin, submodule_search_locations=[str(package.resolve())])
    if spec is None or spec.loader is None:
        raise PluginError("Cannot create package spec for %s" % module)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[module] = loaded
    if parent is not None:
        setattr(parent, parts[-1], loaded)
    spec.loader.exec_module(loaded)
    return loaded


def load_managed_package(metadata: PluginMetadata, package: Path, v, *, source_path: Path, source_kind: str = "external"):
    from sparkrun.core.plugin_identity import PluginIdentity, normalize_plugin_id

    prefix = metadata.module + "."
    before = {name: value for name, value in sys.modules.items() if name == metadata.module or name.startswith(prefix)}
    parent_name, _, child = metadata.module.rpartition(".")
    parent = sys.modules.get(parent_name)
    old_attribute = getattr(parent, child, None) if parent else None
    try:
        return load_and_register_plugin(
            partial(_import_package, metadata.module, package),
            v,
            tier=DeclarationTier.OUT_OF_TREE,
            source=(source_kind, metadata.module, str(source_path)),
            identity=PluginIdentity(normalize_plugin_id(metadata.name), metadata.module, (source_kind, metadata.module, str(source_path))),
            declarations=metadata.declarations,
            application_profile_api=metadata.application_profile_api,
        )
    except BaseException:
        for name in list(sys.modules):
            if (name == metadata.module or name.startswith(prefix)) and name not in before:
                del sys.modules[name]
        parent = sys.modules.get(parent_name)
        if parent is not None:
            if old_attribute is None:
                parent.__dict__.pop(child, None)
            else:
                setattr(parent, child, old_attribute)
        raise


def validating_candidate() -> bool:
    """True inside a child process validating a managed plugin candidate.

    Validation reports through initialization failure, so the loaders' usual
    "log and continue" containment must not apply there.
    """
    return _validation_candidate is not None


def managed_source(record: dict):
    from sparkrun.core.plugin_policy import PluginSource

    trust = record.get("trust", {})
    if trust.get("granted") is True and trust.get("default_managed") is True:
        return PluginSource.DEFAULT_MANAGED
    return PluginSource.EXTERNAL


def _has_enabled_default_managed_plugin(config) -> bool:
    """Narrow exception to the default-off directory gate for curated payloads.

    Only a persisted, source-trusted managed record can qualify. The ordinary
    external policy remains unchanged and explicit global opt-outs still win.
    """
    from sparkrun.core.plugin_store import PluginStore
    from sparkrun.core.plugin_policy import PluginSource

    try:
        config, records = PluginStore(config).snapshot()
    except PluginError:
        return False
    return any(managed_source(record) is PluginSource.DEFAULT_MANAGED and managed_enabled(record, config) for record in records.values())


def managed_enabled(record: dict, config) -> bool:
    from sparkrun.core.plugin_policy import plugin_state

    metadata = record["metadata"]
    legacy = metadata.get("feature") if not metadata.get("declarations") else None
    return plugin_state(metadata["name"], config, source=managed_source(record), legacy_feature=legacy).enabled


def load_managed_plugins(v, *, config=None):
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.external_plugins import _external_plugins_disabled
    from sparkrun.core.features import is_feature_enabled
    from sparkrun.core.plugin_store import PluginStore

    config = config or SparkrunConfig()
    if _validation_candidate is None and (_external_plugins_disabled() or not is_feature_enabled("core.external_plugins", config=config)):
        if not _has_enabled_default_managed_plugin(config):
            return []
    store = PluginStore(config)
    try:
        config, records = store.snapshot()
    except PluginError as exc:
        if _validation_candidate:
            raise
        logger.warning("Cannot read managed plugin state: %s", exc)
        return []
    if _validation_candidate:
        candidate, _path = _validation_candidate
        records[candidate["metadata"]["name"]] = candidate
    from sparkrun.core.plugin_identity import reserve_identities

    # Contested IDs are recorded here and refused again at registration.
    reserve_identities(managed_plugin_identities(config, records).values())
    loaded = []
    # Preflight the entire managed set before importing any managed provider.
    rejected = {}
    for name, record in records.items():
        try:
            check_collisions(PluginMetadata.from_dict(record["metadata"]), records, config, replacing=name)
        except PluginError as exc:
            rejected[name] = str(exc)
    for name, record in records.items():
        validating = _validation_candidate is not None and record is _validation_candidate[0]
        if not validating and not managed_enabled(record, config):
            continue
        try:
            if name in rejected:
                raise PluginError(rejected[name])
            metadata = PluginMetadata.from_dict(record["metadata"])
            if validating and _validation_candidate is not None:
                package = _validation_candidate[1]
                source = package
            else:
                receipt = store.receipt(name, record["active_generation"])
                if receipt["metadata"] != record["metadata"]:
                    raise PluginError("Plugin ledger and active receipt disagree")
                source = store.generation_path(name, record["active_generation"])
                package = source / "payload" / Path(*module_parts(metadata.module))
            load_managed_package(metadata, package, v, source_path=source, source_kind=managed_source(record).value)
            loaded.append(name)
        except Exception as exc:
            if validating:
                raise
            from sparkrun.core.registration import record_plugin_load_failure

            source = store.generation_path(name, record["active_generation"])
            record_plugin_load_failure((managed_source(record).value, record["metadata"]["module"], str(source)), exc)
            logger.warning("Cannot load managed plugin %s: %s", name, exc)
    return loaded


def managed_plugin_identities(config, records=None):
    from sparkrun.core.external_plugins import _external_plugins_disabled
    from sparkrun.core.features import is_feature_enabled
    from sparkrun.core.plugin_identity import PluginIdentity, normalize_plugin_id
    from sparkrun.core.plugin_store import PluginStore

    if _validation_candidate is None and (_external_plugins_disabled() or not is_feature_enabled("core.external_plugins", config=config)):
        if not _has_enabled_default_managed_plugin(config):
            return {}
    store = PluginStore(config)
    if records is None:
        try:
            config, records = store.snapshot()
        except PluginError:
            if _validation_candidate:
                raise
            return {}
    records = dict(records)
    if _validation_candidate:
        candidate, _ = _validation_candidate
        records[candidate["metadata"]["name"]] = candidate
    identities = {}
    from sparkrun.core.features import declare_plugin_features

    for name, record in records.items():
        validating = _validation_candidate is not None and record is _validation_candidate[0]
        source = (
            _validation_candidate[1] if validating and _validation_candidate else store.generation_path(name, record["active_generation"])
        )
        metadata = record["metadata"]
        module = metadata["module"]
        try:
            identity = PluginIdentity(normalize_plugin_id(name), module, (managed_source(record).value, module, str(source)))
            declare_plugin_features(identity, metadata.get("declarations", {}))
            if validating or managed_enabled(record, config):
                identities[name] = identity
        except ValueError as exc:
            from sparkrun.core.registration import record_plugin_load_failure

            record_plugin_load_failure((managed_source(record).value, module, str(source)), exc)
    return identities
