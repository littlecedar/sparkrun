"""Registry catalog names are stable, selectable, and shared by every consumer."""

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from sparkrun import api
from sparkrun.cli._common import RECIPE_NAME
from sparkrun.core.recipe import find_recipe, list_recipes
from sparkrun.core.registry import RegistryEntry, iter_asset_files, link_directory


STEM = "deepseek-v4.1-flash"
THREE = f"@eugr/3x-spark-cluster/{STEM}"
FOUR = f"@eugr/4x-spark-cluster/{STEM}"


def _write(path, *, model="org/model", runtime="sglang", nodes=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump({"recipe_version": "2", "model": model, "runtime": runtime, "container": "test/image", "min_nodes": nodes})
    )
    return path


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    sctx = api.default_sctx()
    manager = sctx.registry_manager
    entry = RegistryEntry(name="eugr", url="https://example.invalid/eugr.git", subpath="recipes")
    manager._save_registries([entry], pending_bootstrap_urls=[])
    root = manager.cache_root / "eugr" / "recipes"
    _write(root / "3x-spark-cluster" / f"{STEM}.yaml", model="org/three", nodes=3)
    _write(root / "4x-spark-cluster" / f"{STEM}.yaml", model="org/four", runtime="vllm", nodes=4)
    _write(root / "nested" / "unique.yaml")
    (root.parent / ".git").mkdir()
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    return sctx, manager, entry, root


@pytest.mark.parametrize("root_copy", [False, True])
def test_catalog_names_round_trip_and_respect_extension_precedence(catalog, root_copy):
    _, manager, entry, root = catalog
    _write(root / "3x-spark-cluster" / f"{STEM}.yml", model="org/shadowed")
    _write(root / ".hidden" / "unique.yaml")
    if root_copy:
        _write(root / f"{STEM}.yaml", model="org/root")
    rows = manager.search_recipes("")
    expected = {THREE, FOUR, "@eugr/unique"} | ({f"@eugr/{STEM}"} if root_copy else set())
    assert {row["name"] for row in rows} == expected
    assert "org/shadowed" not in {row["model"] for row in rows}
    for row in rows:
        path = Path(row["path"])
        assert find_recipe(row["name"], registry_manager=manager) == path
        assert manager.catalog_recipe_name(entry, path) == row["name"]
        assert row["file"] == path.stem


def test_search_and_runtime_filters_keep_inventory_names(catalog):
    sctx, _, _, _ = catalog
    for query, runtime in [("3x-spark-cluster", None), ("org/three", None), (STEM, "sglang"), (THREE, None)]:
        rows = api.search_recipes(query, runtime=runtime, ensure_initialized=False, sctx=sctx)
        assert [row.name for row in rows] == [THREE]
        assert rows[0].file == STEM


def test_paginated_catalog_and_details_share_names_and_identity(catalog):
    sctx, _, _, root = catalog
    first = api.catalog_recipes(STEM, limit=1, sctx=sctx)
    second = api.catalog_recipes(STEM, offset=1, limit=1, sctx=sctx)
    assert first["total"] == second["total"] == 2
    assert first["recipes"][0]["name"] == THREE
    assert second["recipes"][0]["name"] == FOUR
    row = first["recipes"][0]
    narrowed = api.catalog_recipes("3x-spark-cluster", sctx=sctx)["recipes"]
    assert narrowed == [row]
    detail = api.get_recipe_details(row["reference"], sctx=sctx)
    assert detail["name"] == THREE
    assert detail["reference"] == row["reference"]
    assert detail["source_path"] == str(root / "3x-spark-cluster" / f"{STEM}.yaml")
    assert api.get_recipe_details(THREE, sctx=sctx)["reference"] == row["reference"]

    # Removing the collision changes only the catalog label, not source identity
    # or the recipe's serving fingerprint. No registry-update hook is involved.
    (root / "4x-spark-cluster" / f"{STEM}.yaml").unlink()
    updated = api.get_recipe_details(row["reference"], sctx=sctx)
    assert updated["name"] == f"@eugr/{STEM}"
    assert updated["reference"] == detail["reference"]
    assert updated["recipe_revision"] == detail["recipe_revision"]
    assert api.catalog_recipes(STEM, sctx=sctx)["recipes"][0]["name"] == updated["name"]


def test_catalog_names_follow_added_collisions_without_update(catalog):
    _, manager, entry, root = catalog
    path = root / "nested" / "unique.yaml"
    assert manager.catalog_recipe_name(entry, path) == "@eugr/unique"
    sibling = _write(root / "other" / "unique.yaml")
    assert {row["name"] for row in manager.search_recipes("unique")} == {"@eugr/nested/unique", "@eugr/other/unique"}
    assert manager.catalog_recipe_name(entry, path) == "@eugr/nested/unique"
    sibling.unlink()
    assert [row["name"] for row in manager.search_recipes("unique")] == ["@eugr/unique"]


def test_invalid_sibling_still_requires_resolver_disambiguation(catalog):
    _, manager, _, root = catalog
    (root / "4x-spark-cluster" / f"{STEM}.yaml").write_text("[not a recipe]\n")
    assert [row["name"] for row in manager.search_recipes(STEM)] == [THREE]
    assert find_recipe(THREE, registry_manager=manager) == root / "3x-spark-cluster" / f"{STEM}.yaml"


def test_unique_listing_keeps_variants_and_short_name_precedence(catalog):
    sctx, manager, entry, root = catalog
    other = RegistryEntry(name="other", url="https://example.invalid/other.git", subpath="recipes")
    manager._save_registries([entry, other])
    _write(manager.cache_root / "other" / "recipes" / "unique.yaml", model="org/other")
    local = _write(Path.cwd() / "unique.yaml", model="org/local")
    rows = api.search_recipes(unique_names=True, ensure_initialized=False, sctx=sctx)
    assert {row.name for row in rows} == {"unique", THREE, FOUR}
    assert next(row for row in rows if row.name == "unique").model == "org/local"

    # A requested subtree must still be named against the full registry, and
    # the later full-root visit must reuse the same catalog enumeration.
    with patch.object(manager, "_list_dir_recipes", wraps=manager._list_dir_recipes) as enumerator:
        legacy = list_recipes(search_paths=[root / "3x-spark-cluster"], local_files=[local], registry_manager=manager)
    assert {row["name"] for row in legacy} == {"unique", THREE, FOUR}
    assert [call.args[1] for call in enumerator.call_args_list].count("eugr") == 1


@pytest.mark.parametrize(
    "prefix, expected",
    [
        ("", {"unique", THREE, FOUR}),
        ("deepseek", {THREE, FOUR}),
        ("@eugr", {"@eugr/unique", THREE, FOUR}),
        ("@eugr/", {"@eugr/unique", THREE, FOUR}),
        ("@eugr/deepseek", {THREE, FOUR}),
        ("@eugr/3x-spark-cluster/", {THREE}),
    ],
)
def test_completion_offers_selectable_variants(catalog, monkeypatch, prefix, expected):
    sctx, manager, _, _ = catalog
    monkeypatch.setattr("sparkrun.cli._common._get_config_and_registry", lambda: (sctx.config, manager))
    items = RECIPE_NAME.shell_complete(None, None, prefix)
    assert {item.value for item in items} == expected
    assert len(items) == len(expected)
    for item in items:
        assert find_recipe(item.value, registry_manager=manager).is_file()


def test_naming_reuses_listing_scan_and_detail_naming_does_not_parse_yaml(catalog, monkeypatch):
    _, manager, entry, root = catalog
    with patch("sparkrun.core.registry.iter_asset_files", wraps=iter_asset_files) as scan:
        assert len(manager.search_recipes("")) == 3
    assert scan.call_count == 1
    monkeypatch.setattr("sparkrun.core.recipe.recipe_summary", lambda *a, **kw: pytest.fail("detail naming parsed a recipe"))
    monkeypatch.setattr("sparkrun.core.registry.read_yaml", lambda *a, **kw: pytest.fail("detail naming loaded registry inventory"))
    assert manager.catalog_recipe_name(entry, root / "3x-spark-cluster" / f"{STEM}.yaml") == THREE


def test_detail_names_keep_selected_registry_through_shared_cache_alias(catalog):
    sctx, manager, entry, root = catalog
    alias = RegistryEntry(name="alias", url=entry.url, subpath="recipes")
    manager._save_registries([entry, alias])
    link_directory(manager.cache_root / "alias", root.parent)
    row = api.catalog_recipes("3x-spark-cluster", registry="alias", sctx=sctx)["recipes"][0]
    expected = f"@alias/3x-spark-cluster/{STEM}"
    assert row["name"] == expected
    assert api.get_recipe_details(row["reference"], sctx=sctx)["name"] == expected
    assert manager.catalog_recipe_name(alias, Path(row["source_path"])) == expected
