"""Managed plugin lifecycle; transport, validation, and activation stay separate."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from sparkrun.core.plugin_compatibility import HostSnapshot, evaluate_plugin_compatibility
from sparkrun.core.plugin_manifest import PluginError, PluginMetadata, module_parts, repository_identity
from sparkrun.core.plugin_sources import PluginCandidate, acquire_git
from sparkrun.core.plugin_store import PluginStore, retention
from sparkrun.core.plugin_validation import validate_candidate

Consent = Callable[[dict], bool]


class PluginManager:
    def __init__(self, config, *, host: HostSnapshot | None = None):
        self.config = config
        self.store = PluginStore(config)
        self.host = host or HostSnapshot.current()

    def acquire(self, source: str, ref: str = "main"):
        if source.startswith("pypi:") or source.endswith(".whl"):
            raise PluginError("This release supports Git sources; wheel/PyPI acquisition is a separate planned phase")
        return acquire_git(source, ref)

    def _compatible(self, metadata: PluginMetadata):
        result = evaluate_plugin_compatibility(metadata.to_dict(), self.host)
        if result.status != "compatible":
            raise PluginError("%s %s: %s. Existing installation unchanged." % (metadata.name, result.status, "; ".join(result.reasons)))
        return result

    @staticmethod
    def _consent(candidate: PluginCandidate, previous: dict | None, config, *, trust, enable, consent, legacy_consent):
        from sparkrun.core.features import is_feature_enabled

        metadata = candidate.metadata
        changed_scope = previous is None or any(
            previous["metadata"].get(k) != metadata.to_dict().get(k) for k in ("module", "distribution", "source", "feature", "repository")
        )
        changed_scope = changed_scope or (previous is not None and previous["source"].get("url") != candidate.source.get("url"))
        if previous is not None:
            changed_scope = changed_scope or previous["source"].get("fetch_url", previous["source"].get("url")) != candidate.source.get(
                "fetch_url", candidate.source.get("url")
            )
        enable_external = previous is None and not is_feature_enabled("core.external_plugins", config=config)
        legacy_paths = config.external_plugin_paths if enable_external else []
        details = {
            "name": metadata.name,
            "version": metadata.version,
            "module": metadata.module,
            "source": candidate.source,
            "revision": candidate.revision,
            "enable_external": enable_external,
            "enable_plugin": metadata.name if enable else None,
            "enable_feature": None,
            "legacy_paths": [str(p) for p in legacy_paths],
        }
        try:
            details["repository_mismatch"] = repository_identity(metadata.repository) != repository_identity(
                candidate.source.get("url", "")
            )
        except PluginError:
            details["repository_mismatch"] = True
        if changed_scope and not trust and not (consent and consent(details)):
            raise PluginError("Source trust is required; review the source and pass --trust for noninteractive installation")
        if legacy_paths and not (legacy_consent and legacy_consent(details)):
            raise PluginError(
                "Enabling external plugins also activates existing plugins.paths; review and enable those paths explicitly first"
            )
        return enable_external

    def install(
        self,
        candidate: PluginCandidate,
        *,
        selector: str | None = None,
        trust: bool = False,
        enable: bool = False,
        dry_run: bool = False,
        consent: Consent | None = None,
        legacy_consent: Consent | None = None,
        _default=None,
    ) -> dict:
        from sparkrun.core.managed_plugins import check_collisions

        self._compatible(candidate.metadata)
        if dry_run:
            config, records = self.store.snapshot()
            retention(config)
            check_collisions(candidate.metadata, records, config, replacing=selector)
            return {
                "name": candidate.metadata.name,
                "version": candidate.metadata.version,
                "revision": candidate.revision,
                "status": "planned",
                "validation": "load validation and trust pending",
            }
        with self.store.mutation() as (config, records, baseline):
            name = candidate.metadata.name
            if _default is not None:
                from sparkrun.core.plugin_defaults import default_plugins, default_skip_reason, validate_default_candidate

                if _default not in default_plugins(config.profile):
                    raise PluginError("Default plugin is not declared by this application")
                validate_default_candidate(_default, candidate)
                if selector is not None:
                    raise PluginError("Default reconciliation cannot replace an installed plugin")
                if name in records:
                    return {"name": name, "status": "present"}
                if reason := default_skip_reason(_default, config):
                    return {"name": name, "status": "skipped", "reason": reason}
            previous = None
            if selector:
                selected, previous = self.store.select(records, selector)
                if selected != name:
                    raise PluginError("Update changes plugin identity from %s to %s; add the new plugin explicitly" % (selected, name))
            elif name in records:
                raise PluginError("Plugin %s is already installed; use sparkrun plugins update %s" % (name, name))
            check_collisions(candidate.metadata, records, config, replacing=name if previous else None)
            if previous and previous["source"].get("ref_kind") == "tag":
                same_tag = all(
                    previous["source"].get(key) == candidate.source.get(key) for key in ("kind", "url", "fetch_url", "ref", "ref_kind")
                )
                pinned = previous["source"].get("pinned_revision", previous["revision"])
                if same_tag and pinned != candidate.revision:
                    raise PluginError("Pinned tag moved; existing installation unchanged")
            external = False
            if _default is None:
                external = self._consent(
                    candidate, previous, config, trust=trust, enable=enable, consent=consent, legacy_consent=legacy_consent
                )
            if previous and previous["revision"] == candidate.revision and previous["metadata"] == candidate.metadata.to_dict():
                self.store.receipt(name, previous["active_generation"])
                # Source tracking changes still persist; no duplicate generation.
                records[name]["source"] = candidate.source
                warning = self.store.commit(records, config, baseline)
                return {
                    "name": name,
                    "status": "unchanged",
                    "version": candidate.metadata.version,
                    "revision": candidate.revision,
                    "history_pruned": self.store.history_pruned,
                    "warning": warning,
                }
            default_managed = _default is not None or bool(previous and previous["trust"].get("default_managed") is True)
            validation_options = {"default_managed": True} if default_managed else {}
            validate_candidate(candidate.metadata, candidate.package_path, config.config_path, **validation_options)
            identifier, receipt = None, None
            if previous:
                for old in previous["previous_generations"]:
                    retained = self.store.receipt(name, old)
                    if retained["revision"] == candidate.revision and retained["metadata"] == candidate.metadata.to_dict():
                        identifier, receipt = old, retained
                        break
            if identifier is None:
                identifier, receipt = self.store.publish(candidate)
            history = [previous["active_generation"], *previous["previous_generations"]] if previous else []
            history = [old for old in history if old != identifier]
            records[name] = {
                "metadata": candidate.metadata.to_dict(),
                "source": candidate.source,
                "revision": candidate.revision,
                "active_generation": identifier,
                "previous_generations": history,
                "trust": {"granted": True, "scope": "source", "granted_at": datetime.now(timezone.utc).isoformat()},
            }
            if default_managed:
                records[name]["trust"]["default_managed"] = True
            if previous is None and _default is None:
                from sparkrun.core.plugin_defaults import suppress_default

                suppress_default(config, name, False)
            features = deepcopy(config._data.get("features", {}))
            if not isinstance(features, dict):
                features = {}
            if external:
                features["core.external_plugins"] = True
            if external:
                config.set("features", features)
            from sparkrun.core.plugin_policy import set_plugin_override
            from sparkrun.core.plugin_lifecycle import run_hook, apply_plan

            if enable:
                set_plugin_override(config, name, True)
            details = {
                "module": candidate.metadata.module,
                "package": str(candidate.package_path),
                "metadata": candidate.metadata.to_dict(),
                "declarations": candidate.metadata.declarations,
                "default_managed": default_managed,
            }
            plan = run_hook(
                config, name, details, "plugin.update" if previous else "plugin.install", previous=previous, current=records[name]
            )
            apply_plan(config, plan)
            messages = list(plan.messages)
            if enable:
                plan = run_hook(config, name, details, "plugin.enable", current=records[name])
                apply_plan(config, plan)
                messages.extend(plan.messages)
            warning = self.store.commit(records, config, baseline)
            return {
                "name": name,
                "status": "updated" if previous else "installed",
                "messages": messages,
                "version": candidate.metadata.version,
                "revision": candidate.revision,
                "generation": identifier,
                "history_pruned": self.store.history_pruned,
                "warning": warning,
            }

    def add(self, source: str, *, ref: str = "main", **options) -> dict:
        with self.acquire(source, ref) as candidate:
            return self.install(candidate, **options)

    def update(self, selector: str, *, ref: str | None = None, automatic: bool = False, **options) -> dict:
        # Recover before fetching, so a broken activation can be repaired offline.
        if not options.get("dry_run"):
            with self.store.mutation():
                pass
        config, records = self.store.snapshot()
        name, record = self.store.select(records, selector)
        source = record["source"]
        if ref and source["kind"] != "git":
            raise PluginError("--ref is only supported for Git plugins")
        pinned_revision = source.get("pinned_revision", record["revision"])
        pinned_policy = source.get("ref_kind") in {"commit", "tag"}
        if not ref and pinned_policy and record["revision"] == pinned_revision:
            # Tags are checked explicitly for movement; commits/artifacts need no network.
            if source["ref_kind"] != "tag" or automatic:
                if not options.get("dry_run"):
                    with self.store.mutation() as (cfg, latest, baseline):
                        warning = self.store.commit(latest, cfg, baseline)
                else:
                    warning = None
                return {
                    "name": name,
                    "status": "pinned",
                    "revision": record["revision"],
                    "history_pruned": getattr(self.store, "history_pruned", False),
                    "warning": warning,
                }
        target = source.get("fetch_url", source["url"])
        # Rollback changes the active revision, never the source's saved pin.
        # Automatic reconciliation reapplies that exact pin without following a tag.
        restoring_pin = not ref and pinned_policy and record["revision"] != pinned_revision and automatic
        target_ref = pinned_revision if restoring_pin else ref or source.get("ref", "main")
        with self.acquire(target, target_ref) as candidate:
            if restoring_pin:
                candidate = replace(candidate, source=source)
            return self.install(candidate, selector=name, **options)

    def rollback(self, selector: str, *, dry_run: bool = False) -> dict:
        if dry_run:
            config, records = self.store.snapshot()
            name, record = self.store.select(records, selector)
            if retention(config) < 2 or not record["previous_generations"]:
                raise PluginError("No retained generation is available for rollback")
            receipt = self.store.receipt(name, record["previous_generations"][0])
            self._compatible(PluginMetadata.from_dict(receipt["metadata"]))
            return {"name": name, "status": "planned", "revision": receipt["revision"]}
        with self.store.mutation() as (config, records, baseline):
            name, record = self.store.select(records, selector)
            if retention(config) < 2 or not record["previous_generations"]:
                raise PluginError("No retained generation is available for rollback (retain_generations must be at least 2)")
            target, *history = record["previous_generations"]
            receipt = self.store.receipt(name, target)
            metadata = PluginMetadata.from_dict(receipt["metadata"])
            self._compatible(metadata)
            package = self.store.generation_path(name, target) / "payload" / Path(*module_parts(metadata.module))
            validation_options = {"default_managed": True} if record["trust"].get("default_managed") is True else {}
            validate_candidate(metadata, package, config.config_path, **validation_options)
            previous = deepcopy(record)
            record.update(
                metadata=receipt["metadata"],
                revision=receipt["revision"],
                active_generation=target,
                previous_generations=[record["active_generation"], *history],
            )
            records[name] = record
            from sparkrun.core.plugin_lifecycle import run_hook, apply_plan

            details = {
                "module": metadata.module,
                "package": str(package),
                "metadata": metadata.to_dict(),
                "declarations": metadata.declarations,
                "default_managed": record["trust"].get("default_managed") is True,
            }
            plan = run_hook(config, name, details, "plugin.rollback", previous=previous, current=record)
            apply_plan(config, plan)
            warning = self.store.commit(records, config, baseline)
            return {
                "name": name,
                "status": "rolled back",
                "messages": list(plan.messages),
                "version": metadata.version,
                "revision": receipt["revision"],
                "history_pruned": getattr(self.store, "history_pruned", False),
                "warning": warning,
            }

    def remove(self, selector: str, *, dry_run: bool = False) -> dict:
        if dry_run:
            _config, records = self.store.snapshot()
            name, _record = self.store.select(records, selector)
            return {"name": name, "status": "planned removal"}
        with self.store.mutation() as (config, records, baseline):
            name, record = self.store.select(records, selector)
            from sparkrun.core.plugin_lifecycle import run_hook, apply_plan
            from sparkrun.core.plugin_policy import PluginSource, plugin_state

            if plugin_state(name, config, source=PluginSource.EXTERNAL).required:
                raise PluginError("Plugin %s is required by this application" % name)
            warning = None
            messages = []
            try:
                self.store.receipt(name, record["active_generation"])
                metadata = PluginMetadata.from_dict(record["metadata"])
                package = self.store.generation_path(name, record["active_generation"]) / "payload" / Path(*module_parts(metadata.module))
                details = {
                    "module": metadata.module,
                    "package": str(package),
                    "metadata": metadata.to_dict(),
                    "declarations": metadata.declarations,
                    "default_managed": record["trust"].get("default_managed") is True,
                }
                plan = run_hook(config, name, details, "plugin.remove", previous=record)
                apply_plan(config, plan)
                messages = list(plan.messages)
            except PluginError as exc:
                warning = "Removed without completing its hook: " + str(exc)
            del records[name]
            from sparkrun.core.plugin_defaults import suppress_default

            suppress_default(config, name, True)
            cleanup = self.store.commit(records, config, baseline)
            warning = warning or cleanup
            return {
                "name": name,
                "status": "removed",
                "messages": messages,
                "history_pruned": getattr(self.store, "history_pruned", False),
                "warning": warning,
            }
