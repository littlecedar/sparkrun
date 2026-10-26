"""Tests for the sparkrun.api recipe catalog surface (search_recipes)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from sparkrun.api import InvalidRegistryFilter, RecipeSummary, resolve_recipe_filter, search_recipes


def _write_recipe(recipe_dir: Path, stem: str, *, model: str, runtime: str = "vllm", **extra):
    recipe_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "sparkrun_version": "2",
        "name": stem,
        "description": f"{stem} description",
        "model": model,
        "runtime": runtime,
        "container": "scitrera/dgx-spark-vllm:latest",
        **extra,
    }
    with open(recipe_dir / f"{stem}.yaml", "w") as f:
        yaml.dump(data, f)


@pytest.fixture
def catalog(tmp_path: Path, monkeypatch):
    """Two enabled registries (one hidden), one disabled, no CWD recipes."""
    config_root = tmp_path / "config"
    cache_root = tmp_path / "cache" / "registries"
    config_root.mkdir(parents=True)
    cache_root.mkdir(parents=True)

    import sparkrun.core.config

    monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)
    monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CACHE_DIR", tmp_path / "cache")

    with open(config_root / "registries.yaml", "w") as f:
        yaml.dump(
            {
                "registries": [
                    {"name": "community", "url": "https://example.com/c", "subpath": "recipes", "enabled": True},
                    {"name": "atlas", "url": "https://example.com/a", "subpath": "recipes", "enabled": True, "visible": False},
                    {"name": "retired", "url": "https://example.com/r", "subpath": "recipes", "enabled": False},
                ]
            },
            f,
        )

    for name in ("community", "atlas", "retired"):
        (cache_root / name / ".git").mkdir(parents=True)

    _write_recipe(
        cache_root / "community" / "recipes",
        "shared-recipe",
        model="Qwen/Qwen3-8B",
        defaults={"tensor_parallel": 2, "gpu_memory_utilization": 0.85},
        min_nodes=2,
    )
    _write_recipe(cache_root / "community" / "recipes", "community-only", model="mistralai/Mistral-7B-v0.1", runtime="sglang")
    _write_recipe(cache_root / "atlas" / "recipes", "shared-recipe", model="Qwen/Qwen3-8B")
    _write_recipe(cache_root / "retired" / "recipes", "retired-recipe", model="Qwen/Qwen3-1.7B")

    # CWD recipes are discovered relative to the process cwd; keep it clean and
    # let individual tests opt in via the `cwd_recipe` fixture.
    monkeypatch.chdir(tmp_path / "empty-cwd")
    return cache_root


@pytest.fixture(autouse=True)
def _empty_cwd(tmp_path: Path):
    (tmp_path / "empty-cwd").mkdir(exist_ok=True)


@pytest.fixture
def cwd_recipe(tmp_path: Path, monkeypatch):
    """A recipe sitting in the working directory."""
    work = tmp_path / "work"
    work.mkdir()
    _write_recipe(work, "local-qwen", model="Qwen/Qwen3-1.7B")
    monkeypatch.chdir(work)
    return work


def _names(results):
    return [r.name for r in results]


class TestSearchRecipes:
    def test_returns_typed_summaries(self, catalog):
        results = search_recipes("community-only")
        assert len(results) == 1
        entry = results[0]
        assert isinstance(entry, RecipeSummary)
        assert entry.name == "@community/community-only"
        assert entry.file == "community-only"
        assert entry.registry == "community"
        assert entry.runtime == "sglang"
        assert entry.model == "mistralai/Mistral-7B-v0.1"

    def test_parses_recipe_defaults(self, catalog):
        (entry,) = [r for r in search_recipes("shared-recipe", registry="community")]
        assert entry.tensor_parallel == 2
        assert entry.gpu_memory_utilization == 0.85
        assert entry.min_nodes == 2

    def test_absent_defaults_are_none(self, catalog):
        (entry,) = [r for r in search_recipes("community-only")]
        assert entry.tensor_parallel is None
        assert entry.gpu_memory_utilization is None
        assert entry.min_nodes == 1

    def test_to_dict_round_trips_summary_shape(self, catalog):
        (entry,) = [r for r in search_recipes("community-only")]
        raw = entry.to_dict()
        # The legacy recipe-summary mapping the CLI formatters consume.
        assert raw["name"] == "@community/community-only"
        assert raw["file"] == "community-only"
        assert raw["registry"] == "community"
        assert "path" in raw and "model" in raw and "runtime" in raw

    def test_empty_query_matches_everything_visible(self, catalog):
        names = _names(search_recipes())
        assert "@community/shared-recipe" in names
        assert "@community/community-only" in names
        # atlas is hidden, retired is disabled
        assert "@atlas/shared-recipe" not in names
        assert not any("retired" in n for n in names)

    def test_include_hidden(self, catalog):
        names = _names(search_recipes(include_hidden=True))
        assert "@atlas/shared-recipe" in names

    def test_disabled_registry_never_listed(self, catalog):
        names = _names(search_recipes(include_hidden=True))
        assert not any("retired" in n for n in names)

    def test_runtime_filter(self, catalog):
        names = _names(search_recipes(runtime="sglang"))
        assert names == ["@community/community-only"]

    def test_search_matches_inherited_metadata_description(self, catalog):
        recipes = catalog / "community" / "recipes"
        (recipes / "base.yaml").write_text(
            "model: org/base\nruntime: sglang\nmetadata:\n  description: >-\n    Experimental engine\n    with RoCEnante.\n"
        )
        (recipes / "variant.yaml").write_text("include: base\nmodel: org/variant\n")
        results = search_recipes("RoCEnante")
        assert {r.name for r in results} == {"@community/base", "@community/variant"}
        assert all(r.description == "Experimental engine with RoCEnante." for r in results)


class TestRegistryScope:
    def test_scope_shorthand(self, catalog):
        names = _names(search_recipes("@community"))
        assert sorted(names) == ["@community/community-only", "@community/shared-recipe"]

    def test_scope_trailing_slash(self, catalog):
        assert _names(search_recipes("@community/")) == _names(search_recipes("@community"))

    def test_scope_with_query(self, catalog):
        assert _names(search_recipes("@community/shared")) == ["@community/shared-recipe"]

    def test_scope_equals_explicit_registry(self, catalog):
        assert _names(search_recipes("@community")) == _names(search_recipes(registry="community"))

    def test_registry_filter_implies_include_hidden(self, catalog):
        """A hidden registry is reachable by naming it, without include_hidden."""
        assert _names(search_recipes("@atlas")) == ["@atlas/shared-recipe"]
        assert _names(search_recipes(registry="atlas")) == ["@atlas/shared-recipe"]

    def test_bare_at_is_a_plain_query(self, catalog):
        # Every qualified name contains '@', so this matches all visible recipes.
        assert len(search_recipes("@")) == 2

    def test_unknown_registry(self, catalog):
        with pytest.raises(InvalidRegistryFilter) as exc:
            search_recipes("@comunity")
        assert exc.value.reason == "unknown"
        assert exc.value.registry == "comunity"
        assert "community" in exc.value.available

    def test_unknown_registry_explicit(self, catalog):
        with pytest.raises(InvalidRegistryFilter) as exc:
            search_recipes(registry="comunity")
        assert exc.value.reason == "unknown"

    def test_disabled_registry(self, catalog):
        with pytest.raises(InvalidRegistryFilter) as exc:
            search_recipes("@retired")
        assert exc.value.reason == "disabled"

    def test_conflicting_scope(self, catalog):
        with pytest.raises(InvalidRegistryFilter) as exc:
            search_recipes("@community/qwen", registry="atlas")
        assert exc.value.reason == "conflict"

    def test_redundant_scope_is_allowed(self, catalog):
        assert _names(search_recipes("@community/shared", registry="community")) == ["@community/shared-recipe"]

    def test_resolve_recipe_filter(self, catalog):
        assert resolve_recipe_filter("@community/qwen") == ("community", "qwen")
        assert resolve_recipe_filter("@community") == ("community", None)
        assert resolve_recipe_filter("qwen") == (None, "qwen")
        assert resolve_recipe_filter(None) == (None, None)


class TestUniqueNames:
    def test_default_keeps_every_registry_copy(self, catalog):
        names = _names(search_recipes("shared", include_hidden=True))
        assert names == ["@community/shared-recipe", "@atlas/shared-recipe"]

    def test_unique_names_collapses_by_stem(self, catalog):
        names = _names(search_recipes("shared", include_hidden=True, unique_names=True))
        assert names == ["@community/shared-recipe"]

    def test_same_stem_variants_within_a_registry_are_kept(self, catalog):
        """Sibling dirs holding same-stem variants are different recipes.

        Their catalog names include enough path context to select either one.
        """
        _write_recipe(catalog / "community" / "recipes" / "4x-cluster", "shared-recipe", model="Qwen/Qwen3-8B", min_nodes=4)

        results = [r for r in search_recipes("shared-recipe", registry="community")]
        assert len(results) == 2
        assert {r.min_nodes for r in results} == {2, 4}
        assert len({r.path for r in results}) == 2

        assert {r.name for r in results} == {"@community/shared-recipe", "@community/4x-cluster/shared-recipe"}
        assert len(search_recipes("shared-recipe", registry="community", unique_names=True)) == 2


class TestLocalRecipes:
    def test_cwd_recipe_listed_first(self, catalog, cwd_recipe):
        results = search_recipes()
        assert results[0].name == "local-qwen"
        assert results[0].registry is None

    def test_cwd_recipe_matches_query(self, catalog, cwd_recipe):
        assert _names(search_recipes("local-qwen")) == ["local-qwen"]

    def test_cwd_recipe_excluded_by_include_local(self, catalog, cwd_recipe):
        assert "local-qwen" not in _names(search_recipes(include_local=False))

    def test_cwd_recipe_excluded_by_registry_filter(self, catalog, cwd_recipe):
        """A local recipe belongs to no registry, so a registry filter drops it."""
        assert "local-qwen" not in _names(search_recipes("@community"))

    def test_local_recipe_wins_unique_names(self, catalog, tmp_path, monkeypatch):
        """A working-directory recipe shadows a registry one of the same name."""
        work = tmp_path / "shadow"
        work.mkdir()
        _write_recipe(work, "shared-recipe", model="local/override")
        monkeypatch.chdir(work)

        (entry,) = [r for r in search_recipes("shared-recipe", unique_names=True)]
        assert entry.registry is None
        assert entry.model == "local/override"


class TestInitialization:
    def test_ensure_initialized_failure_is_not_fatal(self, catalog, monkeypatch):
        """A registry that cannot be cloned degrades to what is already cached."""
        from sparkrun.core.registry import RegistryManager

        def _boom(self, *a, **kw):
            raise RuntimeError("no network")

        monkeypatch.setattr(RegistryManager, "ensure_initialized", _boom)
        assert _names(search_recipes("community-only")) == ["@community/community-only"]

    def test_ensure_initialized_can_be_skipped(self, catalog, monkeypatch):
        calls = []

        from sparkrun.core.registry import RegistryManager

        monkeypatch.setattr(RegistryManager, "ensure_initialized", lambda self, *a, **kw: calls.append(1))
        search_recipes("community-only", ensure_initialized=False)
        assert calls == []
