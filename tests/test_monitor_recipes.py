"""Tests for cli/_monitor_recipes.py: the recipe picker and its parameter modal."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from textual.app import App
from textual.color import Color
from textual.containers import ItemGrid, Vertical
from textual.widgets import Button, Checkbox, Input, OptionList, ProgressBar, Select, Static

from sparkrun.api import HostsUnreachable, InsufficientCapacity, RecipeSummary, SparkrunError
from sparkrun.cli._monitor_recipes import RecipeLaunch, RecipePickerScreen, RunParamsScreen, RuntimeChecks, RuntimeChecksScreen
from sparkrun.core.recipe import Recipe
from sparkrun.core.registry import RegistryManager
from sparkrun.core.scheduler import CapacityRejection
from sparkrun.models.fit import FitResult, HostFitDetail

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _summary(
    name: str, *, model: str = "org/model", runtime: str = "vllm-ray", registry: str | None = "community", description: str | None = None
) -> RecipeSummary:
    stem = name.rsplit("/", 1)[-1]
    entry: dict[str, Any] = {
        "name": name,
        "file": stem,
        "path": f"/tmp/{stem}.yaml",
        "model": model,
        "runtime": runtime,
        "description": f"{stem} description" if description is None else description,
        "min_nodes": 1,
        "tp": "",
        "gpu_mem": "",
    }
    if registry:
        entry["registry"] = registry
    return RecipeSummary.from_summary(entry)


class _PickerTestApp(App):
    """Pushes the picker on mount and records its dismissed result."""

    def __init__(self, screen: RecipePickerScreen, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.picker = screen
        self.result: Any = "unset"

    def on_mount(self) -> None:
        self.push_screen(self.picker, callback=self._on_result)

    def _on_result(self, res: Any) -> None:
        self.result = res


def _recipe_yaml(tmp_path) -> str:
    path = tmp_path / "demo.yaml"
    path.write_text(
        'recipe_version: "2"\nmodel: test/model\nruntime: sglang\ncontainer: test/image:latest\n'
        "defaults:\n  n_gpu_layers: 99\n  attention_backend: triton\n  pipeline_parallel: 2\n"
        'env:\n  VLLM_USE_V1: "1"\n'
    )
    return str(path)


async def _wait_loaded(pilot, picker: RecipePickerScreen) -> None:
    for _ in range(40):
        if picker._loaded:
            return
        await pilot.pause(0.05)
    raise AssertionError("recipe catalog never loaded")


def _fake_plan(
    host_list: tuple[str, ...] = ("h1", "h2"),
    scheduler: str = "greedy",
    notes: tuple[str, ...] = (),
    cluster: Any = None,
) -> Any:
    return lambda _options, **_kwargs: SimpleNamespace(host_list=host_list, scheduler=scheduler, notes=notes, cluster=cluster)


def _fit_result(per_host: dict[str, bool]) -> FitResult:
    details = {
        host: HostFitDetail(
            host=host,
            ranks_assigned=1,
            vram_per_rank_gb=10.0,
            accelerator_memory_gb=20.0,
            headroom_gb=10.0 if ok else -5.0,
            ok=ok,
        )
        for host, ok in per_host.items()
    }
    return FitResult(ok=all(per_host.values()), per_host=details)


def _capacity(entry_by_host: dict[str, tuple[bool, int | None]]) -> dict[str, Any]:
    """A catalog capacity probe result: host → (reachable, free_slots)."""
    return {
        "cluster": "lab",
        "observed_at": 0.0,
        "advisory": True,
        "hosts": [
            {"host": host, "reachable": reachable, "free_slots": free, "used_slots": None, "workloads": None}
            for host, (reachable, free) in entry_by_host.items()
        ],
    }


def _open(monkeypatch, summaries, tmp_path) -> _PickerTestApp:
    monkeypatch.setattr("sparkrun.cli._monitor_recipes.search_recipes", lambda **_kwargs: list(summaries))
    recipe = Recipe.load(_recipe_yaml(tmp_path))
    monkeypatch.setattr("sparkrun.cli._monitor_recipes.resolve_recipe", lambda _path, **_kwargs: recipe)
    # The checks interstitial asks api.plan for the scheduler's placement;
    # never touch hosts here. Its fake plan carries no cluster, so the VRAM fit
    # step is skipped too (every host stays selectable). The capacity probe is
    # likewise stubbed to an empty report (availability unknown).
    monkeypatch.setattr("sparkrun.cli._monitor_recipes.plan", _fake_plan())
    monkeypatch.setattr("sparkrun.cli._monitor_recipes.catalog_cluster_capacity", lambda *_args, **_kwargs: _capacity({}))
    return _PickerTestApp(RecipePickerScreen(cluster="lab", hosts=("h1", "h2")))


async def _wait_for_screen(pilot, picker: RecipePickerScreen, screen_type: type, *, tries: int = 80):
    for _ in range(tries):
        if isinstance(picker.app.screen, screen_type):
            return picker.app.screen
        await pilot.pause(0.05)
    raise AssertionError(f"{screen_type.__name__} never opened")


async def _activate_first(pilot, picker: RecipePickerScreen) -> None:
    await _wait_loaded(pilot, picker)
    await pilot.pause(0.05)
    listed = picker.query_one("#recipe-list", OptionList)
    listed.focus()
    listed.highlighted = 0
    await pilot.press("enter")


async def _open_modal(pilot, picker: RecipePickerScreen) -> RunParamsScreen:
    # The checks modal is progress-only: completion dismisses it with the
    # RuntimeChecks and the picker pushes the parameter modal in the same flow.
    await _activate_first(pilot, picker)
    modal = await _wait_for_screen(pilot, picker, RunParamsScreen)
    assert isinstance(modal, RunParamsScreen)
    return modal


async def _wait_until(pilot, predicate, *, tries: int = 200) -> None:
    for _ in range(tries):
        if predicate():
            return
        await pilot.pause(0.05)
    raise AssertionError("condition never met")


async def _wait_result(pilot, app: _PickerTestApp) -> Any:
    # Run re-queries the scheduler before launching, so the result arrives one
    # modal later than the click.
    for _ in range(200):
        if app.result != "unset":
            return app.result
        await pilot.pause(0.05)
    raise AssertionError(f"launch result never arrived: screen={type(app.screen).__name__} result={app.result!r}")


def test_picker_loads_catalog_and_shows_metadata(monkeypatch, tmp_path):
    summaries = [_summary("@community/alpha"), _summary("@community/beta"), _summary("@atlas/gamma", runtime="sglang", registry="atlas")]
    app = _open(monkeypatch, summaries, tmp_path)

    async def _run():
        async with app.run_test(size=(120, 32)) as pilot:
            picker = app.picker
            await _wait_loaded(pilot, picker)
            await pilot.pause(0.05)
            listed = picker.query_one("#recipe-list", OptionList)
            assert listed.option_count == 3
            assert [s.name for s in picker._shown] == ["@community/alpha", "@community/beta", "@atlas/gamma"]
            assert "@community/alpha" in str(picker.query_one("#recipe-meta-head", Static).content)
            # facets: All + two runtimes + two registries
            filters = picker.query_one("#recipe-filters", OptionList)
            assert filters.option_count == 1 + 2 + 2

    asyncio.run(_run())


def test_picker_search_filters_live(monkeypatch, tmp_path):
    summaries = [_summary("@community/qwen-a", model="Qwen/Qwen3"), _summary("@community/llama-b", model="meta/Llama")]
    app = _open(monkeypatch, summaries, tmp_path)

    async def _run():
        async with app.run_test(size=(120, 32)) as pilot:
            picker = app.picker
            await _wait_loaded(pilot, picker)
            await pilot.pause(0.05)
            search = picker.query_one("#recipe-search", Input)
            search.value = "qwen"
            await pilot.pause(0.1)
            assert [s.name for s in picker._shown] == ["@community/qwen-a"]
            search.value = ""
            await pilot.pause(0.1)
            assert len(picker._shown) == 2

    asyncio.run(_run())


@pytest.mark.parametrize(
    "family, expected",
    [("vllm", ["base", "ray", "distributed", "eugr"]), ("sglang", ["sglang"]), ("llama-cpp", ["llama"]), ("custom-engine", ["unknown"])],
)
def test_picker_facet_filters_by_runtime_family(monkeypatch, tmp_path, family, expected):
    summaries = [
        _summary("@community/base", runtime="vllm"),
        _summary("@community/ray", runtime="VLLM-Ray"),
        _summary("@community/distributed", runtime="vllm-distributed"),
        _summary("@community/eugr", runtime="eugr-vllm"),
        _summary("@community/sglang", runtime="sglang"),
        _summary("@community/llama", runtime="llama-cpp"),
        _summary("@community/unknown", runtime="custom-engine"),
    ]
    app = _open(monkeypatch, summaries, tmp_path)

    async def _run():
        async with app.run_test(size=(120, 32)) as pilot:
            picker = app.picker
            await _wait_loaded(pilot, picker)
            await pilot.pause(0.05)
            filters = picker.query_one("#recipe-filters", OptionList)
            assert [opt.id for opt in filters.options if (opt.id or "").startswith("runtime:")] == [
                "runtime:custom-engine",
                "runtime:llama-cpp",
                "runtime:sglang",
                "runtime:vllm",
            ]
            runtime_idx = next(i for i, opt in enumerate(filters.options) if opt.id == f"runtime:{family}")
            filters.highlighted = runtime_idx
            await pilot.pause(0.1)
            assert picker._filter == f"runtime:{family}"
            assert [s.name for s in picker._shown] == [f"@community/{stem}" for stem in expected]

    asyncio.run(_run())


def test_picker_uses_runtime_family_from_shared_plugin_context(monkeypatch, tmp_path):
    app = _open(monkeypatch, [_summary("@community/plugin", runtime="my-plugin")], tmp_path)
    variables = object()
    app.picker.sctx = SimpleNamespace(variables=variables)

    def runtime(name, *, v):
        assert name == "my-plugin"
        assert v is variables
        return SimpleNamespace(get_family=lambda: "sglang")

    monkeypatch.setattr("sparkrun.cli._monitor_recipes.get_runtime", runtime)

    async def _run():
        async with app.run_test(size=(120, 32)) as pilot:
            picker = app.picker
            await _wait_loaded(pilot, picker)
            await pilot.pause()
            filters = picker.query_one("#recipe-filters", OptionList)
            assert "runtime:my-plugin" not in [opt.id for opt in filters.options]
            filters.highlighted = next(i for i, opt in enumerate(filters.options) if opt.id == "runtime:sglang")
            await pilot.pause()
            assert [s.name for s in picker._shown] == ["@community/plugin"]

    asyncio.run(_run())


@pytest.mark.parametrize("width", [60, 120])
def test_picker_all_registries_toggle_searches_hidden_enabled_catalog(monkeypatch, tmp_path, width, v):
    """Use the real catalog: hidden is opt-in, disabled is always excluded."""
    manager = RegistryManager(tmp_path / "config", tmp_path / "cache")
    registries = [
        {"name": "community", "visible": True, "enabled": True},
        {"name": "archive", "visible": False, "enabled": True},
        {"name": "retired", "visible": False, "enabled": False},
    ]
    for entry in registries:
        entry.update(url=f"https://example.com/{entry['name']}", subpath="recipes")
        cache = manager.cache_root / entry["name"]
        (cache / ".git").mkdir(parents=True)
        recipes = cache / "recipes"
        recipes.mkdir()
        (recipes / "qwen.yaml").write_text(
            'recipe_version: "2"\nmodel: Qwen/Qwen3\nruntime: vllm-distributed\ncontainer: test/image:latest\n'
        )
    (manager.config_root / "registries.yaml").write_text(yaml.safe_dump({"registries": registries}))
    monkeypatch.chdir(tmp_path)
    app = _PickerTestApp(RecipePickerScreen(cluster="lab", hosts=("h1",), sctx=SimpleNamespace(registry_manager=manager, variables=v)))

    async def _run():
        async with app.run_test(size=(width, 32)) as pilot:
            picker = app.picker
            await _wait_loaded(pilot, picker)
            await pilot.pause()
            toggle = picker.query_one("#recipe-all-registries", Checkbox)
            assert not toggle.value
            assert 0 <= toggle.region.x < toggle.region.right <= width
            assert [s.name for s in picker._shown] == ["@community/qwen"]
            filters = picker.query_one("#recipe-filters", OptionList)
            assert "registry:archive" not in [opt.id for opt in filters.options]
            search = picker.query_one("#recipe-search", Input)
            search.value = "qwen"
            filters.highlighted = next(i for i, opt in enumerate(filters.options) if opt.id == "runtime:vllm")
            await pilot.pause()

            await pilot.click("#recipe-all-registries")
            await _wait_until(pilot, lambda: picker._loaded and len(picker._shown) == 2)
            assert picker._filter == "runtime:vllm"
            assert search.value == "qwen"
            assert [s.name for s in picker._shown] == ["@community/qwen", "@archive/qwen"]
            assert "registry:archive" in [opt.id for opt in filters.options]
            assert "registry:retired" not in [opt.id for opt in filters.options]

            filters.highlighted = next(i for i, opt in enumerate(filters.options) if opt.id == "registry:archive")
            await pilot.pause()
            assert [s.name for s in picker._shown] == ["@archive/qwen"]
            await pilot.click("#recipe-all-registries")
            await _wait_until(pilot, lambda: picker._loaded and picker._filter == "all")
            assert search.value == "qwen"
            assert [s.name for s in picker._shown] == ["@community/qwen"]
            assert "registry:archive" not in [opt.id for opt in filters.options]

    asyncio.run(_run())


@pytest.mark.parametrize("stale_failure", [False, True])
def test_picker_registry_toggle_ignores_obsolete_loads(monkeypatch, tmp_path, stale_failure):
    """An old default-scope request cannot overwrite a newer one after two toggles."""
    app = _open(monkeypatch, [], tmp_path)
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    calls = []

    def search(*, include_hidden, **_kwargs):
        first = not calls
        calls.append(include_hidden)
        if first:
            started.set()
            try:
                assert release.wait(10)
                if stale_failure:
                    raise RuntimeError("obsolete catalog failure")
                return [_summary("@community/stale")]
            finally:
                finished.set()
        return (
            [_summary("@community/fresh"), _summary("@archive/hidden", registry="archive")]
            if include_hidden
            else [_summary("@community/fresh")]
        )

    monkeypatch.setattr("sparkrun.cli._monitor_recipes.search_recipes", search)
    notifications = []
    monkeypatch.setattr(app, "notify", lambda message, **_kwargs: notifications.append(message))

    async def _run():
        async with app.run_test(size=(120, 32)) as pilot:
            try:
                picker = app.picker
                await _wait_until(pilot, started.is_set)
                await pilot.click("#recipe-all-registries")
                await _wait_until(pilot, lambda: picker._loaded and len(picker._shown) == 2)
                await pilot.click("#recipe-all-registries")
                await _wait_until(pilot, lambda: picker._loaded and len(picker._shown) == 1)
                release.set()
                await _wait_until(pilot, finished.is_set)
                await pilot.pause()
                assert calls == [False, True, False]
                assert [s.name for s in picker._shown] == ["@community/fresh"]
                assert "registry:archive" not in [opt.id for opt in picker.query_one("#recipe-filters", OptionList).options]
                assert not notifications
            finally:
                release.set()

    asyncio.run(_run())


def test_picker_enter_opens_params_and_cancel_refocuses(monkeypatch, tmp_path):
    app = _open(monkeypatch, [_summary("@community/alpha")], tmp_path)

    async def _run():
        async with app.run_test(size=(120, 50)) as pilot:
            picker = app.picker
            modal = await _open_modal(pilot, picker)
            assert modal.summary.name == "@community/alpha"

            await pilot.press("escape")
            await pilot.pause(0.1)
            assert not isinstance(app.screen, RunParamsScreen)
            assert app.screen is picker
            assert app.focused is picker.query_one("#recipe-list", OptionList)

    asyncio.run(_run())


def test_picker_cancel_button_refocuses_recipe_list(monkeypatch, tmp_path):
    app = _open(monkeypatch, [_summary("@community/alpha")], tmp_path)

    async def _run():
        async with app.run_test(size=(120, 50)) as pilot:
            picker = app.picker
            await _open_modal(pilot, picker)
            await pilot.click("#param-cancel")
            await pilot.pause(0.1)
            assert app.screen is picker
            assert app.focused is picker.query_one("#recipe-list", OptionList)

    asyncio.run(_run())


def test_params_run_builds_recipe_launch(monkeypatch, tmp_path):
    app = _open(monkeypatch, [_summary("@community/alpha")], tmp_path)

    async def _run():
        async with app.run_test(size=(120, 50)) as pilot:
            picker = app.picker
            modal = await _open_modal(pilot, picker)
            modal.query_one("#param-tp", Input).value = "2"
            modal.query_one("#param-advanced-toggle", Checkbox).value = True
            await pilot.pause(0.1)
            modal.query_one("#param-advanced", Input).value = "max_num_seqs=256 env.VLLM_TEST=1"
            await pilot.click("#param-run")
            launch = await _wait_result(pilot, app)

            assert isinstance(launch, RecipeLaunch)
            assert launch.label == "@community/alpha"
            assert launch.options.overrides == {"tensor_parallel": 2, "max_num_seqs": 256}
            recipe = launch.options.recipe
            assert isinstance(recipe, Recipe)
            assert recipe.env["VLLM_TEST"] == "1"
            assert launch.options.cluster == "lab"
            assert launch.options.hosts == ("h1", "h2")
            assert not isinstance(app.screen, RecipePickerScreen)
            # No hooks in this recipe → no authorization checkbox, trust stays off.
            assert launch.options.trust is False

    asyncio.run(_run())


def test_params_validation_keeps_modal_open(monkeypatch, tmp_path):
    app = _open(monkeypatch, [_summary("@community/alpha")], tmp_path)

    async def _run():
        async with app.run_test(size=(120, 50)) as pilot:
            picker = app.picker
            modal = await _open_modal(pilot, picker)

            modal.query_one("#param-tp", Input).value = "abc"
            await pilot.click("#param-run")
            await pilot.pause(0.1)
            assert isinstance(app.screen, RunParamsScreen)
            assert "integer" in str(modal.query_one("#param-error", Static).content)

            modal.query_one("#param-tp", Input).value = ""
            modal.query_one("#param-advanced-toggle", Checkbox).value = True
            await pilot.pause(0.1)
            modal.query_one("#param-advanced", Input).value = "notakeyvalue"
            await pilot.pause(0.1)
            await pilot.click("#param-run")
            await pilot.pause(0.1)
            assert isinstance(app.screen, RunParamsScreen)
            assert "key=value" in str(modal.query_one("#param-error", Static).content)
            assert app.result == "unset"

    asyncio.run(_run())


def test_picker_escape_dismisses_with_none(monkeypatch, tmp_path):
    app = _open(monkeypatch, [_summary("@community/alpha")], tmp_path)

    async def _run():
        async with app.run_test(size=(120, 32)) as pilot:
            await _wait_loaded(pilot, app.picker)
            await pilot.pause(0.05)
            await pilot.press("escape")
            await pilot.pause(0.1)
            assert app.result is None
            assert not isinstance(app.screen, RecipePickerScreen)

    asyncio.run(_run())


def test_params_trust_checkbox_authorizes_untrusted_hooks(monkeypatch, tmp_path):
    app = _open(monkeypatch, [_summary("@community/hooked")], tmp_path)
    monkeypatch.setattr(
        "sparkrun.cli._monitor_recipes.untrusted_hooks",
        lambda *_args, **_kwargs: {"pre_exec": ("setup.sh",)},
    )

    async def _run():
        async with app.run_test(size=(120, 50)) as pilot:
            picker = app.picker
            modal = await _open_modal(pilot, picker)
            box = modal.query_one("#param-trust", Checkbox)
            assert box.value is False

            # Unauthorized Run keeps the dialog (and the typed parameters) open.
            await pilot.click("#param-run")
            await pilot.pause(0.1)
            assert app.screen is modal
            assert "Authorize" in str(modal.query_one("#param-error", Static).content)

            box.value = True
            await pilot.pause(0.05)
            await pilot.click("#param-run")
            await _wait_result(pilot, app)

    asyncio.run(_run())
    launch = app.result
    assert isinstance(launch, RecipeLaunch)
    assert launch.options.trust is True


def test_picker_late_catalog_callback_after_dismiss_is_ignored(monkeypatch, tmp_path):
    app = _open(monkeypatch, [_summary("@community/alpha")], tmp_path)

    async def _run():
        async with app.run_test(size=(120, 32)) as pilot:
            picker = app.picker
            await _wait_loaded(pilot, picker)
            await pilot.pause(0.05)
            await pilot.press("escape")
            await pilot.pause(0.1)
            assert not picker.is_attached
            # The loader thread's queued call_from_thread still lands here after
            # the screen is gone; it must not touch unmounted widgets.
            picker._on_loaded([_summary("@community/late")], picker._catalog_request, {"vllm-ray": "vllm"})

    asyncio.run(_run())


def test_params_host_section_preselects_scheduler_placement(monkeypatch, tmp_path):
    app = _open(monkeypatch, [_summary("@community/alpha")], tmp_path)
    notes = ("2 nodes required, using 1 of 2 hosts",)
    monkeypatch.setattr("sparkrun.cli._monitor_recipes.plan", _fake_plan(("h2",), "dense-pack", notes))

    async def _run():
        async with app.run_test(size=(120, 60)) as pilot:
            modal = await _open_modal(pilot, app.picker)
            assert modal.query_one("#param-host-0", Checkbox).value is False
            assert modal.query_one("#param-host-1", Checkbox).value is True
            note = str(modal.query_one("#param-hosts-note", Static).content)
            assert "dense-pack" in note
            assert "1/2 selected" in note
            assert "2 nodes required" in note

    asyncio.run(_run())


def test_params_host_selection_narrows_the_launch(monkeypatch, tmp_path):
    app = _open(monkeypatch, [_summary("@community/alpha")], tmp_path)

    async def _run():
        async with app.run_test(size=(120, 60)) as pilot:
            modal = await _open_modal(pilot, app.picker)
            modal.query_one("#param-host-1", Checkbox).value = False
            await pilot.pause(0.05)
            assert "1/2 selected" in str(modal.query_one("#param-hosts-note", Static).content)
            await pilot.click("#param-run")
            await _wait_result(pilot, app)

    asyncio.run(_run())
    launch = app.result
    assert isinstance(launch, RecipeLaunch)
    assert launch.options.hosts == ("h1",)


def test_params_host_selection_requires_one_host(monkeypatch, tmp_path):
    app = _open(monkeypatch, [_summary("@community/alpha")], tmp_path)

    async def _run():
        async with app.run_test(size=(120, 60)) as pilot:
            modal = await _open_modal(pilot, app.picker)
            modal.query_one("#param-host-0", Checkbox).value = False
            modal.query_one("#param-host-1", Checkbox).value = False
            await pilot.pause(0.05)
            await pilot.click("#param-run")
            await pilot.pause(0.1)
            assert app.screen is modal
            assert "host" in str(modal.query_one("#param-error", Static).content)
            assert app.result == "unset"

    asyncio.run(_run())


def test_picker_search_input_has_room_for_its_text(monkeypatch, tmp_path):
    app = _open(monkeypatch, [_summary("@community/alpha")], tmp_path)

    async def _run():
        async with app.run_test(size=(120, 40)) as pilot:
            await _wait_loaded(pilot, app.picker)
            await pilot.pause(0.1)
            search = app.picker.query_one("#recipe-search", Input)
            # A one-row Input spends its only row on the border: the text and
            # cursor are clipped away, which is what made typing invisible.
            # (size/content_size exclude the border, so check the widget box.)
            assert search.region.height > 1

    asyncio.run(_run())


def test_params_columns_split_short_and_long_inputs(monkeypatch, tmp_path):
    app = _open(monkeypatch, [_summary("@community/alpha")], tmp_path)

    async def _run():
        async with app.run_test(size=(140, 50)) as pilot:
            modal = await _open_modal(pilot, app.picker)
            dialog = modal.query_one("#params-dialog")
            short = modal.query_one("#param-short")
            long = modal.query_one("#param-long")
            assert 0 < short.region.width < long.region.width
            assert abs(short.region.width / dialog.region.width - 0.2) <= 0.1
            # Content-sized: the dialog must not stretch to its 90% max-height
            # (45 rows of 50) while the Advanced section is collapsed.
            assert dialog.region.height < 40

    asyncio.run(_run())


def test_params_advanced_checkbox_reveals_recipe_and_launch_knobs(monkeypatch, tmp_path):
    app = _open(monkeypatch, [_summary("@community/alpha")], tmp_path)

    async def _run():
        async with app.run_test(size=(140, 50)) as pilot:
            modal = await _open_modal(pilot, app.picker)
            assert not modal.query_one("#param-adv-short").display
            assert not modal.query_one("#param-adv-long").display

            modal.query_one("#param-advanced-toggle", Checkbox).value = True
            await pilot.pause(0.1)
            assert modal.query_one("#param-adv-short").display
            assert modal.query_one("#param-adv-long").display

            # Every scalar recipe default owns a field, except keys that already
            # have a dedicated control (the common knobs and every launch knob).
            defaults = [widget for widget in modal.query(Input) if (widget.id or "").startswith("param-default-")]
            assert len(defaults) == len(modal._default_knobs) == 2
            assert [key for key, _value in modal._default_knobs] == ["n_gpu_layers", "attention_backend"]
            envs = [widget for widget in modal.query(Input) if (widget.id or "").startswith("param-env-")]
            assert len(envs) == len(modal._env_keys) == 1

            modal.query_one("#param-knob-pipeline_parallel", Input).value = "2"
            modal.query_one("#param-knob-image", Input).value = "override/image:tag"
            modal.query_one("#param-knob-offline", Select).value = "offline"
            modal.query_one("#param-knob-sync_tuning", Checkbox).value = False
            modal.query_one("#param-default-0", Input).value = "42"
            modal.query_one("#param-env-0", Input).value = "0"
            await pilot.click("#param-run")
            launch = await _wait_result(pilot, app)

            assert isinstance(launch, RecipeLaunch)
            assert launch.options.overrides["pipeline_parallel"] == 2
            assert launch.options.overrides["n_gpu_layers"] == 42
            assert launch.options.recipe.env["VLLM_USE_V1"] == "0"
            assert launch.options.offline is True
            assert launch.options.sync_tuning is False

    asyncio.run(_run())


def test_params_advanced_values_are_ignored_while_collapsed(monkeypatch, tmp_path):
    app = _open(monkeypatch, [_summary("@community/alpha")], tmp_path)

    async def _run():
        async with app.run_test(size=(140, 50)) as pilot:
            modal = await _open_modal(pilot, app.picker)
            # Values set programmatically never matter until Advanced is on.
            modal.query_one("#param-knob-pipeline_parallel", Input).value = "2"

            # A value typed while Advanced is revealed must not leak through
            # once the section is collapsed again.
            modal.query_one("#param-advanced-toggle", Checkbox).value = True
            await pilot.pause(0.1)
            modal.query_one("#param-advanced", Input).value = "max_num_seqs=256"
            modal.query_one("#param-knob-pipeline_parallel", Input).value = "2"
            modal.query_one("#param-advanced-toggle", Checkbox).value = False
            await pilot.pause(0.1)
            await pilot.click("#param-run")
            launch = await _wait_result(pilot, app)

            assert isinstance(launch, RecipeLaunch)
            assert "pipeline_parallel" not in launch.options.overrides
            assert "max_num_seqs" not in launch.options.overrides
            assert launch.options.scheduler is None
            assert launch.options.sync_tuning is True

    asyncio.run(_run())


def test_checks_handoff_opens_params_with_placement(monkeypatch, tmp_path):
    app = _open(monkeypatch, [_summary("@community/alpha")], tmp_path)
    notes = ("2 nodes required, using 1 of 2 hosts",)
    monkeypatch.setattr("sparkrun.cli._monitor_recipes.plan", _fake_plan(("h2",), "dense-pack", notes))

    async def _run():
        async with app.run_test(size=(120, 60)) as pilot:
            picker = app.picker
            # No interaction between picking a recipe and the parameter modal:
            # the progress modal replaces itself as soon as checks finish.
            modal = await _open_modal(pilot, picker)
            assert modal.checks == RuntimeChecks(scheduler="dense-pack", selected=("h2",), selectable=("h1", "h2"), notes=notes)
            title = str(modal.query_one("#param-hosts-title", Static).content)
            note = str(modal.query_one("#param-hosts-note", Static).content)
            assert "dense-pack" in title
            assert "2 nodes required" in note

    asyncio.run(_run())


def test_checks_modal_is_progress_only_and_cancel_returns(monkeypatch, tmp_path):
    app = _open(monkeypatch, [_summary("@community/alpha")], tmp_path)
    release = threading.Event()

    def _slow_plan(_options, **_kwargs):
        release.wait(5)
        return SimpleNamespace(host_list=("h1",), scheduler="greedy", notes=(), cluster=None)

    monkeypatch.setattr("sparkrun.cli._monitor_recipes.plan", _slow_plan)

    async def _run():
        async with app.run_test(size=(120, 40)) as pilot:
            picker = app.picker
            await _activate_first(pilot, picker)
            checks = await _wait_for_screen(pilot, picker, RuntimeChecksScreen)
            # The modal holds a title, an indeterminate progress bar, and Cancel.
            assert str(checks.query_one("#checks-title", Static).content) == "Querying scheduler..."
            assert checks.query_one("#checks-progress", ProgressBar).total is None
            assert str(checks.query_one("#checks-cancel", Button).label) == "Cancel"
            assert not checks.query("#checks-continue")
            assert not checks.query("#checks-body")
            assert not checks.query("#checks-back")
            await pilot.click("#checks-cancel")
            await pilot.pause(0.1)
            assert picker.app.screen is picker
            assert picker.app.focused is picker.query_one("#recipe-list", OptionList)
            # A plan that finishes after Cancel must not open the params modal.
            release.set()
            await pilot.pause(0.3)
            assert picker.app.screen is picker

    try:
        asyncio.run(_run())
    finally:
        release.set()


def test_preflight_requeries_before_launch(monkeypatch, tmp_path):
    app = _open(monkeypatch, [_summary("@community/alpha")], tmp_path)
    calls: list[Any] = []

    def _counting_plan(options, **_kwargs):
        calls.append(options)
        return SimpleNamespace(host_list=("h1", "h2"), scheduler="greedy", notes=(), cluster=None)

    monkeypatch.setattr("sparkrun.cli._monitor_recipes.plan", _counting_plan)
    monkeypatch.setattr(
        "sparkrun.cli._monitor_recipes.catalog_cluster_capacity",
        lambda *_args, **_kwargs: _capacity({"h1": (True, 2), "h2": (True, 2)}),
    )

    async def _run():
        async with app.run_test(size=(120, 60)) as pilot:
            await _open_modal(pilot, app.picker)
            assert len(calls) == 1
            await pilot.click("#param-run")
            launch = await _wait_result(pilot, app)
            assert isinstance(launch, RecipeLaunch)
            assert launch.options.hosts == ("h1", "h2")
            # Run re-queries the scheduler with exactly the options it launches.
            assert len(calls) == 2
            assert calls[1].hosts == ("h1", "h2")

    asyncio.run(_run())


def test_preflight_returns_to_settings_when_host_becomes_unavailable(monkeypatch, tmp_path):
    app = _open(monkeypatch, [_summary("@community/alpha")], tmp_path)
    app.picker.hosts = ("h1", "h2", "h3")
    capacity_calls: list[int] = []

    def _probe(*_args, **_kwargs):
        # Call 0 is the picker's check, call 1 the Run preflight, call 2 the
        # recovery preflight. Indexed rather than an iterator so an unexpected
        # extra call shows up in the assertions instead of a silent StopIteration.
        index = len(capacity_calls)
        capacity_calls.append(index)
        if index == 1:
            return _capacity({"h1": (True, 2), "h2": (False, None), "h3": (False, None)})
        return _capacity({"h1": (True, 2), "h2": (True, 2), "h3": (True, 2)})

    plan_calls: list[Any] = []

    def _counting_plan(options, **_kwargs):
        plan_calls.append(options)
        return SimpleNamespace(host_list=("h1", "h2"), scheduler="greedy", notes=(), cluster=None)

    monkeypatch.setattr("sparkrun.cli._monitor_recipes.plan", _counting_plan)
    monkeypatch.setattr("sparkrun.cli._monitor_recipes.catalog_cluster_capacity", _probe)

    async def _run():
        async with app.run_test(size=(120, 60)) as pilot:
            modal = await _open_modal(pilot, app.picker)
            assert modal.query_one("#param-host-0", Checkbox).value is True
            assert modal.query_one("#param-host-1", Checkbox).value is True
            assert len(capacity_calls) == 1

            await pilot.click("#param-run")
            for _ in range(200):
                if app.screen is modal and modal.checks.selectable == ("h1",):
                    break
                await pilot.pause(0.05)
            else:
                raise AssertionError(
                    "preflight did not return to settings: "
                    f"screen={type(app.screen).__name__} result={app.result!r} "
                    f"checks={modal.checks!r} probes={len(capacity_calls)} plans={len(plan_calls)}"
                )
            assert len(capacity_calls) == 2

            # Nothing launched: the settings stay open with the dead rows fixed.
            assert app.result == "unset"
            assert modal.checks.ok is True
            states = {host: (box.value, box.disabled, str(box.label)) for host, box in modal._host_boxes()}
            assert states["h1"] == (True, False, "h1")
            assert states["h2"] == (False, True, "h2 (unavailable)")
            assert states["h3"] == (False, True, "h3 (unavailable)")
            error = str(modal.query_one("#param-error", Static).content)
            assert "No longer available" in error
            assert "h2" in error

            # Recovery: a clean re-check launches the remaining host. Wait until the
            # preflight screen is fully torn down; a click dispatched while its
            # removal is still settling is swallowed before reaching the button.
            for _ in range(100):
                if not app.query(RuntimeChecksScreen):
                    break
                await pilot.pause(0.05)
            await pilot.pause(0.1)
            await pilot.click("#param-run")
            for _ in range(200):
                if app.result != "unset":
                    break
                await pilot.pause(0.05)
            else:
                raise AssertionError(
                    "recovery launch never arrived: "
                    f"screen={type(app.screen).__name__} plans={len(plan_calls)} probes={len(capacity_calls)} "
                    f"pending={modal._pending_options!r} checks={modal.checks!r} "
                    f"stack={[type(s).__name__ for s in app.screen_stack]} "
                    f"error={str(modal.query_one('#param-error', Static).content)!r}"
                )
            launch = app.result
            assert isinstance(launch, RecipeLaunch)
            assert launch.options.hosts == ("h1",)
            assert len(plan_calls) == 3
            assert len(capacity_calls) == 3

    asyncio.run(_run())


def test_preflight_plan_failure_returns_to_settings(monkeypatch, tmp_path):
    app = _open(monkeypatch, [_summary("@community/alpha")], tmp_path)
    plan_calls: list[Any] = []

    def _second_plan_fails(options, **_kwargs):
        plan_calls.append(options)
        if len(plan_calls) == 1:
            return SimpleNamespace(host_list=("h1", "h2"), scheduler="greedy", notes=(), cluster=None)
        raise InsufficientCapacity(
            "cluster has no free capacity for a solo (1-node) run",
            host_list=["h1"],
            required=1,
            rejections=(CapacityRejection("h1", 0, "occupied", "GPU has an existing allocation"),),
        )

    monkeypatch.setattr("sparkrun.cli._monitor_recipes.plan", _second_plan_fails)

    async def _run():
        async with app.run_test(size=(120, 60)) as pilot:
            modal = await _open_modal(pilot, app.picker)
            await pilot.click("#param-run")
            await _wait_until(pilot, lambda: app.screen is modal and modal.checks.ok is False)

            assert app.result == "unset"
            first = modal.query_one("#param-host-0", Checkbox)
            assert (first.disabled, first.value) == (True, False)
            assert str(first.label) == "h1 (no free slot)"
            error = str(modal.query_one("#param-error", Static).content)
            assert "No longer available" in error
            assert "h1" in error and "no free slot" in error

    asyncio.run(_run())


def test_picker_click_highlights_then_second_click_opens(monkeypatch, tmp_path):
    app = _open(monkeypatch, [_summary("@community/alpha"), _summary("@community/beta")], tmp_path)

    async def _run():
        async with app.run_test(size=(120, 60)) as pilot:
            picker = app.picker
            await _wait_loaded(pilot, picker)
            await pilot.pause(0.1)
            listed = picker.query_one("#recipe-list", OptionList)

            def row(index: int) -> tuple[int, int]:
                # Option rows live in the list's content box, not its border box.
                return (
                    listed.content_region.x - listed.region.x,
                    listed.content_region.y - listed.region.y + index,
                )

            assert listed.highlighted == 0
            # One click on a different row highlights it and nothing else.
            await pilot.click("#recipe-list", offset=row(1))
            await pilot.pause(0.1)
            assert listed.highlighted == 1
            assert picker.app.screen is picker
            assert "@community/beta" in str(picker.query_one("#recipe-meta-head", Static).content)
            # Clicking the highlighted row opens the run configuration flow.
            await pilot.click("#recipe-list", offset=row(1))
            modal = await _wait_for_screen(pilot, picker, RunParamsScreen)
            assert isinstance(modal, RunParamsScreen)
            # Cancel returns to the picker with the list focused, and enter
            # keeps activating the highlighted recipe as before.
            await pilot.click("#param-cancel")
            await pilot.pause(0.1)
            assert picker.app.screen is picker
            assert picker.app.focused is listed
            await pilot.press("enter")
            assert isinstance(await _wait_for_screen(pilot, picker, RunParamsScreen), RunParamsScreen)

    asyncio.run(_run())


def test_checks_failure_is_advisory_and_params_still_opens(monkeypatch, tmp_path):
    app = _open(monkeypatch, [_summary("@community/alpha")], tmp_path)

    def _boom(_options, **_kwargs):
        raise SparkrunError("cluster unreachable: no accelerator slots")

    monkeypatch.setattr("sparkrun.cli._monitor_recipes.plan", _boom)

    async def _run():
        async with app.run_test(size=(120, 60)) as pilot:
            picker = app.picker
            # A failed pre-check is advisory: the parameter modal still opens.
            modal = await _open_modal(pilot, picker)
            # No plan means no placement and no fit data: every host stays
            # selectable and pre-checked, like the CLI's default candidate list.
            assert modal.checks.scheduler == ""
            assert modal.checks.selected == ("h1", "h2")
            assert modal.checks.selectable == ("h1", "h2")
            assert str(modal.query_one("#param-hosts-title", Static).content) == "Hosts (unavailable)"
            note = str(modal.query_one("#param-hosts-note", Static).content)
            assert "no accelerator slots" in note
            boxes = [box for box in modal.query(Checkbox) if (box.id or "").startswith("param-host-")]
            assert all(box.value and not box.disabled for box in boxes)

    asyncio.run(_run())


def test_checks_capacity_failure_disables_and_unchecks_hosts(monkeypatch, tmp_path):
    app = _open(monkeypatch, [_summary("@community/alpha")], tmp_path)

    def _boom(_options, **_kwargs):
        # The shape api.plan raises for a solo recipe when every candidate
        # accelerator is already taken.
        raise InsufficientCapacity(
            "cluster has no free capacity for a solo (1-node) run: No single host can satisfy the requested GPU allocation",
            host_list=["h1", "h2"],
            required=1,
            rejections=(
                CapacityRejection("h1", 0, "occupied", "GPU has an existing allocation"),
                CapacityRejection("h2", 0, "exclusive_owner", "GPU has an existing allocation"),
            ),
        )

    monkeypatch.setattr("sparkrun.cli._monitor_recipes.plan", _boom)

    async def _run():
        async with app.run_test(size=(120, 60)) as pilot:
            picker = app.picker
            modal = await _open_modal(pilot, picker)
            # The scheduler rejected every candidate: nothing is placed, so
            # nothing is pre-checked, and every row is a dead end.
            assert modal.checks.selected == ()
            assert modal.checks.selectable == ()
            assert dict(modal.checks.reasons) == {"h1": "no free slot", "h2": "in use"}
            boxes = [box for box in modal.query(Checkbox) if (box.id or "").startswith("param-host-")]
            assert all(box.disabled and not box.value for box in boxes)
            assert [str(box.label) for box in boxes] == ["h1 (no free slot)", "h2 (in use)"]
            # The capacity error itself stays visible in the hosts note.
            assert "no free capacity" in str(modal.query_one("#param-hosts-note", Static).content)

            # Run retries the scheduler even when every previous row was disabled.
            await pilot.click("#param-run")
            await _wait_until(pilot, lambda: app.screen is modal and modal._pending_options is None)
            assert "no free capacity" in str(modal.query_one("#param-error", Static).content)
            assert app.result == "unset"

            # Without per-host attribution the whole candidate set is dead.
            checks = RuntimeChecksScreen(recipe=None, summary=None, cluster="lab", hosts=("h1", "h2"))
            bare = checks._blocked_checks(InsufficientCapacity("runtime requires 2 nodes, but only 1 hosts provided"))
            assert bare.selected == ()
            assert bare.selectable == ()
            assert dict(bare.reasons) == {"h1": "insufficient capacity", "h2": "insufficient capacity"}

    asyncio.run(_run())


def test_checks_unreachable_failure_disables_only_named_hosts(monkeypatch, tmp_path):
    app = _open(monkeypatch, [_summary("@community/alpha")], tmp_path)

    def _boom(_options, **_kwargs):
        raise HostsUnreachable("hosts unreachable: h2 (timeout)", hosts=["h2"])

    monkeypatch.setattr("sparkrun.cli._monitor_recipes.plan", _boom)

    async def _run():
        async with app.run_test(size=(120, 60)) as pilot:
            picker = app.picker
            modal = await _open_modal(pilot, picker)
            # A partial reachability failure disables only the hosts it names;
            # the survivors stay enabled but start unchecked (nothing placed).
            assert modal.checks.selectable == ("h1",)
            assert dict(modal.checks.reasons) == {"h2": "unavailable"}
            assert "hosts unreachable" in str(modal.query_one("#param-hosts-note", Static).content)
            first = modal.query_one("#param-host-0", Checkbox)
            second = modal.query_one("#param-host-1", Checkbox)
            assert (first.value, first.disabled) == (False, False)
            assert (second.value, second.disabled, str(second.label)) == (False, True, "h2 (unavailable)")

    asyncio.run(_run())


def test_params_host_rows_disable_hosts_that_exceed_memory(monkeypatch, tmp_path):
    app = _open(monkeypatch, [_summary("@community/alpha")], tmp_path)
    cluster = SimpleNamespace(hosts=("h1", "h2"))
    monkeypatch.setattr("sparkrun.cli._monitor_recipes.plan", _fake_plan(("h1", "h2"), "greedy", (), cluster=cluster))
    # Never touch the Hub: both the estimate and the fit check are stubbed.
    monkeypatch.setattr(Recipe, "estimate_vram", lambda self, **_kwargs: SimpleNamespace(total_per_gpu_gb=10.0))
    monkeypatch.setattr(
        "sparkrun.cli._monitor_recipes.check_fit",
        lambda _estimate, _cluster, placement=None: _fit_result({"h1": True, "h2": False}),
    )

    async def _run():
        async with app.run_test(size=(120, 60)) as pilot:
            modal = await _open_modal(pilot, app.picker)
            good = modal.query_one("#param-host-0", Checkbox)
            bad = modal.query_one("#param-host-1", Checkbox)
            assert modal.checks.selectable == ("h1",)
            assert good.disabled is False
            assert bad.disabled is True
            assert "no headroom" in str(bad.label)
            # A disabled row cannot be unchecked, so an over-capacity host the
            # plan selected must not start checked; the launch proceeds from the
            # eligible set alone.
            assert bad.value is False
            await pilot.click("#param-run")
            launch = await _wait_result(pilot, app)
            assert isinstance(launch, RecipeLaunch)
            assert launch.options.hosts == ("h1",)

    asyncio.run(_run())


def test_checks_disable_unreachable_hosts_but_leave_capacity_to_scheduler(monkeypatch, tmp_path):
    app = _open(monkeypatch, [_summary("@community/alpha")], tmp_path)
    app.picker.hosts = ("h1", "h2", "h3")
    cluster = SimpleNamespace(hosts=("h1", "h2", "h3"))
    monkeypatch.setattr("sparkrun.cli._monitor_recipes.plan", _fake_plan(("h1",), "greedy", (), cluster=cluster))
    # Memory is fine everywhere. Unused slots alone cannot decide admission.
    monkeypatch.setattr(Recipe, "estimate_vram", lambda self, **_kwargs: SimpleNamespace(total_per_gpu_gb=10.0))
    monkeypatch.setattr(
        "sparkrun.cli._monitor_recipes.check_fit",
        lambda _estimate, _cluster, placement=None: _fit_result({"h1": True, "h2": True, "h3": True}),
    )
    monkeypatch.setattr(
        "sparkrun.cli._monitor_recipes.catalog_cluster_capacity",
        lambda *_args, **_kwargs: _capacity({"h1": (True, 2), "h2": (False, None), "h3": (True, 0)}),
    )

    async def _run():
        async with app.run_test(size=(120, 60)) as pilot:
            modal = await _open_modal(pilot, app.picker)
            assert modal.checks.selectable == ("h1", "h3")
            assert dict(modal.checks.reasons) == {"h2": "unavailable"}
            states = {host: (box.value, box.disabled, str(box.label)) for host, box in modal._host_boxes()}
            # A reachable host may accept sharing or reuse with no unused slots.
            assert states["h1"] == (True, False, "h1")
            assert states["h2"] == (False, True, "h2 (unavailable)")
            assert states["h3"] == (False, False, "h3")

    asyncio.run(_run())


def test_checks_capacity_probe_failure_keeps_hosts_selectable(monkeypatch, tmp_path):
    app = _open(monkeypatch, [_summary("@community/alpha")], tmp_path)
    cluster = SimpleNamespace(hosts=("h1", "h2"))
    monkeypatch.setattr("sparkrun.cli._monitor_recipes.plan", _fake_plan(("h1",), "greedy", (), cluster=cluster))
    monkeypatch.setattr(Recipe, "estimate_vram", lambda self, **_kwargs: SimpleNamespace(total_per_gpu_gb=10.0))
    monkeypatch.setattr(
        "sparkrun.cli._monitor_recipes.check_fit",
        lambda _estimate, _cluster, placement=None: _fit_result({"h1": True, "h2": True}),
    )
    monkeypatch.setattr(
        "sparkrun.cli._monitor_recipes.catalog_cluster_capacity",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(SparkrunError("occupancy sweep failed")),
    )

    async def _run():
        async with app.run_test(size=(120, 60)) as pilot:
            modal = await _open_modal(pilot, app.picker)
            # Unknown availability is not evidence of a problem: nothing is
            # disabled, and the failure is reported as a warning.
            assert modal.checks.selectable == ("h1", "h2")
            assert modal.checks.reasons == ()
            assert any("occupancy sweep failed" in warning for warning in modal.checks.warnings)
            assert "occupancy sweep failed" in str(modal.query_one("#param-hosts-note", Static).content)

    asyncio.run(_run())


def test_params_hosts_title_names_the_scheduler(monkeypatch, tmp_path):
    app = _open(monkeypatch, [_summary("@community/alpha")], tmp_path)
    monkeypatch.setattr("sparkrun.cli._monitor_recipes.plan", _fake_plan(("h1",), "greedy"))

    async def _run():
        async with app.run_test(size=(120, 60)) as pilot:
            modal = await _open_modal(pilot, app.picker)
            assert str(modal.query_one("#param-hosts-title", Static).content) == "Hosts (greedy)"

    asyncio.run(_run())


def test_params_placeholders_drop_the_default_prefix(monkeypatch, tmp_path):
    app = _open(monkeypatch, [_summary("@community/alpha")], tmp_path)
    path = tmp_path / "common.yaml"
    path.write_text(
        'recipe_version: "2"\nmodel: test/model\nruntime: sglang\ncontainer: test/image:latest\n'
        "defaults:\n  tensor_parallel: 2\n  gpu_memory_utilization: 0.9\n  max_model_len: 4096\n  port: 8000\n"
    )
    monkeypatch.setattr("sparkrun.cli._monitor_recipes.resolve_recipe", lambda _path, **_kwargs: Recipe.load(str(path)))

    async def _run():
        async with app.run_test(size=(120, 60)) as pilot:
            modal = await _open_modal(pilot, app.picker)
            placeholders = {widget.id: widget.placeholder for widget in modal.query(Input)}
            assert placeholders["param-tp"] == "2"
            assert placeholders["param-gpu-mem"] == "0.9"
            assert placeholders["param-max-len"] == "4096"
            assert placeholders["param-port"] == "8000"
            assert not any(value.startswith("default ") for value in placeholders.values())

    asyncio.run(_run())


def test_params_placeholders_mark_missing_defaults_unset(monkeypatch, tmp_path):
    app = _open(monkeypatch, [_summary("@community/alpha")], tmp_path)

    async def _run():
        async with app.run_test(size=(120, 60)) as pilot:
            modal = await _open_modal(pilot, app.picker)
            assert modal.query_one("#param-tp", Input).placeholder == "unset"
            assert modal.query_one("#param-gpu-mem", Input).placeholder == "unset"

    asyncio.run(_run())


def test_picker_meta_orders_decision_fields_before_description(monkeypatch, tmp_path):
    app = _open(monkeypatch, [_summary("@community/alpha")], tmp_path)

    async def _run():
        async with app.run_test(size=(120, 40)) as pilot:
            picker = app.picker
            await _wait_loaded(pilot, picker)
            await pilot.pause(0.1)
            head_widget = picker.query_one("#recipe-meta-head", Static)
            stats_widget = picker.query_one("#recipe-meta-stats", Static)
            desc_widget = picker.query_one("#recipe-meta-desc", Static)
            head = head_widget.content.markup
            stats = str(stats_widget.content)
            desc = str(desc_widget.content)
            # Identity fields come first, directly under the recipe name, and
            # nothing else may push the model or runtime below them.
            assert head.splitlines() == [
                "[bold]@community/alpha[/bold]",
                "model: org/model",
                "runtime: vllm-ray",
            ]
            # Decision stats follow the identity block; provenance internals
            # (path, a separate registry line) are not decision material.
            assert "min nodes: 1" in stats
            assert "tensor parallel" in stats
            assert "gpu memory util" in stats
            assert "path:" not in stats
            assert "registry" not in stats
            # The description is the last field, so it renders below the stats.
            assert "description" in desc
            assert "model:" not in desc
            assert "min nodes" not in desc
            assert desc_widget.region.y > stats_widget.region.y > head_widget.region.y
            assert desc_widget.region.height >= 1
            # The description is supporting prose, rendered dim; the decision
            # fields above it stay at full intensity.
            assert desc_widget.styles.text_style.dim is True
            assert not head_widget.styles.text_style.dim
            assert not stats_widget.styles.text_style.dim

    asyncio.run(_run())


def test_picker_layout_rows_order_and_box_height(monkeypatch, tmp_path):
    """Boxes share one row and height; search and list own full-width rows below."""
    app = _open(monkeypatch, [_summary("@community/alpha")], tmp_path)

    async def _run():
        async with app.run_test(size=(120, 40)) as pilot:
            picker = app.picker
            await _wait_loaded(pilot, picker)
            await pilot.pause(0.1)
            filters = picker.query_one("#recipe-filters", OptionList)
            meta = picker.query_one("#recipe-meta", Vertical)
            search = picker.query_one("#recipe-search", Input)
            listed = picker.query_one("#recipe-list", OptionList)
            # The metadata box is exactly as tall as the filter box beside it.
            assert meta.region.y == filters.region.y
            assert meta.region.height == filters.region.height
            assert meta.region.x > filters.region.x
            # Search sits on its own row under both boxes, list under search.
            assert search.region.y >= filters.region.y + filters.region.height
            assert listed.region.y >= search.region.y + search.region.height
            # Both rows span the picker, not a column of it.
            assert search.region.width == listed.region.width == filters.region.width + meta.region.width
            assert listed.region.height > search.region.height

    asyncio.run(_run())


def test_params_host_checkboxes_use_multiple_columns(monkeypatch, tmp_path):
    app = _open(monkeypatch, [_summary("@community/alpha")], tmp_path)
    app.picker.hosts = ("h1", "h2", "h3", "h4")

    async def _run():
        async with app.run_test(size=(120, 60)) as pilot:
            modal = await _open_modal(pilot, app.picker)
            await pilot.pause(0.1)
            grid = modal.query_one("#param-hosts", ItemGrid)
            hosts = [box for box in modal.query(Checkbox) if (box.id or "").startswith("param-host-")]
            assert len(hosts) == 4
            # ItemGrid picks the column count from its width: several columns
            # here, one when the pane is narrow.
            assert len({box.region.x for box in hosts}) > 1
            assert grid.region.height <= 2 * max(box.region.height for box in hosts)

    asyncio.run(_run())


def _contrast_ratio(first: Color, second: Color) -> float:
    def _luminance(color: Color) -> float:
        def _channel(value: int) -> float:
            scaled = value / 255
            return scaled / 12.92 if scaled <= 0.03928 else ((scaled + 0.055) / 1.055) ** 2.4

        return 0.2126 * _channel(color.r) + 0.7152 * _channel(color.g) + 0.0722 * _channel(color.b)

    lighter, darker = sorted((_luminance(first), _luminance(second)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def test_params_dialog_background_is_neutral_and_readable(monkeypatch, tmp_path):
    app = _open(monkeypatch, [_summary("@community/alpha")], tmp_path)

    async def _run():
        async with app.run_test(size=(120, 60)) as pilot:
            modal = await _open_modal(pilot, app.picker)
            background = modal.query_one("#params-dialog").styles.background
            variables = app.get_css_variables()
            assert background == Color.parse(variables["background-lighten-2"])
            assert background != Color.parse(variables["panel"])
            ratio = _contrast_ratio(Color.parse(variables["foreground"]), background)
            assert ratio >= 7.0, f"contrast {ratio:.2f}:1"

    asyncio.run(_run())


@pytest.mark.parametrize("ensure", [False, True])
def test_scheduler_approved_shared_host_can_launch_without_unused_slots(monkeypatch, tmp_path, ensure):
    from sparkrun.core.cluster_status import ClusterStatus, GpuOccupancy, HostOccupancy
    from sparkrun.core.hardware import AcceleratorSpec, HostHardware
    from sparkrun.core.parallelism import ParallelismConfig
    from sparkrun.core.scheduler import ResourceRequest, SchedulingRequest
    from sparkrun.schedulers.dense_pack import DensePackScheduler

    app = _open(monkeypatch, [_summary("@community/shared")], tmp_path)
    hardware = HostHardware(accelerators=[AcceleratorSpec("nvidia", "gb10", memory_gb=121)])
    occupancy = HostOccupancy("h1", used_slots=1, free_slots=0, gpus=(GpuOccupancy(0, used_memory_gb=30, used_util_fraction=0.3),))
    placement = (
        DensePackScheduler()
        .schedule(
            SchedulingRequest(
                ParallelismConfig(),
                ("h1",),
                host_hardware={"h1": hardware},
                status=ClusterStatus((occupancy,)),
                resources=ResourceRequest(util_fraction=0.3, memory_gb=30),
            )
        )
        .assignment
    )
    monkeypatch.setattr("sparkrun.cli._monitor_recipes.plan", _fake_plan(placement.hosts_used, "dense-pack"))
    monkeypatch.setattr(
        "sparkrun.cli._monitor_recipes.catalog_cluster_capacity",
        lambda *_args, **_kwargs: _capacity({"h1": (True, occupancy.free_slots), "h2": (True, 0)}),
    )

    async def _run():
        async with app.run_test(size=(120, 60)) as pilot:
            modal = await _open_modal(pilot, app.picker)
            assert modal.query_one("#param-host-0", Checkbox).value
            assert not modal.query_one("#param-host-0", Checkbox).disabled
            if ensure:
                modal.query_one("#param-advanced-toggle", Checkbox).value = True
                await pilot.pause()
                modal.query_one("#param-knob-ensure", Checkbox).value = True
            await pilot.click("#param-run")
            launch = await _wait_result(pilot, app)
            # Untouched boxes keep every eligible host a candidate; the plan
            # handed to the launch is what places the workload on h1.
            assert launch.options.hosts == ("h1", "h2")
            assert tuple(launch.plan.host_list) == ("h1",)
            assert launch.options.ensure is ensure

    asyncio.run(_run())


def test_run_rechecks_all_rejected_hosts_with_edited_parameters(monkeypatch, tmp_path):
    app = _open(monkeypatch, [_summary("@community/alpha")], tmp_path)
    calls: list[Any] = []

    def _plan(options, **_kwargs):
        calls.append(options)
        if options.overrides.get("tensor_parallel") != 1:
            raise InsufficientCapacity("requested parallelism does not fit")
        return _fake_plan(("h1",))(options)

    monkeypatch.setattr("sparkrun.cli._monitor_recipes.plan", _plan)

    async def _run():
        async with app.run_test(size=(120, 60)) as pilot:
            modal = await _open_modal(pilot, app.picker)
            assert not modal.checks.selectable
            modal.query_one("#param-tp", Input).value = "1"
            await pilot.click("#param-run")
            await _wait_until(pilot, lambda: app.screen is modal and modal.checks.ok)
            assert len(calls) == 2
            assert calls[1].hosts == ("h1", "h2")
            assert modal.query_one("#param-host-0", Checkbox).value
            assert not modal.query_one("#param-host-0", Checkbox).disabled
            assert app.result == "unset"  # recovered host choices remain reviewable
            await _wait_until(
                pilot, lambda: not app.query(RuntimeChecksScreen) and not modal.query_one("#param-run", Button).has_class("-active")
            )
            await pilot.click("#param-run")
            launch = await _wait_result(pilot, app)
            assert launch.options.hosts == ("h1", "h2")
            assert tuple(launch.plan.host_list) == ("h1",)
            assert launch.options.overrides["tensor_parallel"] == 1

    asyncio.run(_run())


@pytest.mark.parametrize("width", [60, 120])
def test_recheck_restores_a_recovered_host_without_changing_parameters(monkeypatch, tmp_path, width):
    app = _open(monkeypatch, [_summary("@community/alpha")], tmp_path)
    probes = iter([_capacity({"h1": (True, 1), "h2": (False, None)}), _capacity({"h1": (True, 1), "h2": (True, 1)})])
    monkeypatch.setattr("sparkrun.cli._monitor_recipes.catalog_cluster_capacity", lambda *_a, **_kw: next(probes))

    async def _run():
        async with app.run_test(size=(width, 60)) as pilot:
            modal = await _open_modal(pilot, app.picker)
            for button in modal.query("#param-buttons Button"):
                assert button.region.right <= modal.query_one("#params-dialog").content_region.right
            assert modal.query_one("#param-host-1", Checkbox).disabled
            await pilot.click("#param-recheck")
            await _wait_until(pilot, lambda: app.screen is modal and not modal.query_one("#param-host-1", Checkbox).disabled)
            assert modal.query_one("#param-host-0", Checkbox).value
            assert not modal.query_one("#param-host-1", Checkbox).value
            assert app.result == "unset"

    asyncio.run(_run())


def test_clearing_overrides_after_failed_validation_restores_recipe(monkeypatch, tmp_path):
    app = _open(monkeypatch, [_summary("@community/alpha")], tmp_path)

    async def _run():
        async with app.run_test(size=(120, 60)) as pilot:
            modal = await _open_modal(pilot, app.picker)
            image = modal.recipe.container
            env = dict(modal.recipe.env)
            modal.query_one("#param-advanced-toggle", Checkbox).value = True
            await pilot.pause()
            modal.query_one("#param-knob-image", Input).value = "custom/wrong-image:latest"
            modal.query_one("#param-advanced", Input).value = "env.VLLM_USE_V1=0 env.EXTRA=temporary"
            for _host, box in modal._host_boxes():
                box.value = False
            await pilot.click("#param-run")
            assert "Select at least one host" in str(modal.query_one("#param-error", Static).content)
            assert modal.recipe.container == image
            assert modal.recipe.env == env

            modal.query_one("#param-knob-image", Input).value = ""
            modal.query_one("#param-advanced", Input).value = ""
            modal.query_one("#param-host-0", Checkbox).value = True
            await _wait_until(pilot, lambda: not modal.query_one("#param-run", Button).has_class("-active"))
            await pilot.click("#param-run")
            launch = await _wait_result(pilot, app)
            assert launch.options.recipe.container == image
            assert launch.options.recipe.env == env
            assert launch.options.recipe is not modal.recipe

    asyncio.run(_run())


def test_initial_planning_does_not_mutate_parameter_defaults(monkeypatch, tmp_path):
    app = _open(monkeypatch, [_summary("@community/alpha")], tmp_path)

    def _mutating_plan(options, **_kwargs):
        options.recipe.container = "planned/image"
        options.recipe.env["VLLM_USE_V1"] = "changed"
        return _fake_plan()(options)

    monkeypatch.setattr("sparkrun.cli._monitor_recipes.plan", _mutating_plan)

    async def _run():
        async with app.run_test(size=(120, 60)) as pilot:
            modal = await _open_modal(pilot, app.picker)
            assert modal.recipe.container == "test/image:latest"
            assert modal.recipe.env["VLLM_USE_V1"] == "1"

    asyncio.run(_run())


def test_cancelled_preflight_cannot_leak_overrides_into_next_launch(monkeypatch, tmp_path):
    app = _open(monkeypatch, [_summary("@community/alpha")], tmp_path)
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def _slow_plan(options, **_kwargs):
        entered.set()
        assert release.wait(5), "test did not release cancelled preflight"
        options.recipe.env["LATE_RESULT"] = "discard"
        finished.set()
        return _fake_plan()(options)

    async def _run():
        async with app.run_test(size=(120, 60)) as pilot:
            modal = await _open_modal(pilot, app.picker)
            image, env = modal.recipe.container, dict(modal.recipe.env)
            modal.query_one("#param-advanced-toggle", Checkbox).value = True
            await pilot.pause()
            modal.query_one("#param-knob-image", Input).value = "temporary/image"
            modal.query_one("#param-advanced", Input).value = "env.EXTRA=temporary"
            monkeypatch.setattr("sparkrun.cli._monitor_recipes.plan", _slow_plan)
            try:
                await pilot.click("#param-run")
                assert await asyncio.to_thread(entered.wait, 3)
                await pilot.press("escape")
                await _wait_until(pilot, lambda: app.screen is modal)
                assert modal._pending_options is None
                assert app.result == "unset"
            finally:
                release.set()
            assert await asyncio.to_thread(finished.wait, 3)
            # Hiding Advanced also discards its entered overrides for the retry.
            modal.query_one("#param-advanced-toggle", Checkbox).value = False
            monkeypatch.setattr("sparkrun.cli._monitor_recipes.plan", _fake_plan())
            await pilot.pause()  # finish the collapse layout before clicking the moved button
            await _wait_until(
                pilot, lambda: not app.query(RuntimeChecksScreen) and not modal.query_one("#param-run", Button).has_class("-active")
            )
            await pilot.click("#param-run")
            launch = await _wait_result(pilot, app)
            assert launch.options.recipe.container == image
            assert launch.options.recipe.env == env

    asyncio.run(_run())


def test_params_trust_summary_discloses_isolation_overrides_as_plain_text(monkeypatch, tmp_path):
    """Trusting authorizes isolation overrides too; recipe text must not be parsed as markup."""
    app = _open(monkeypatch, [_summary("@community/hooked")], tmp_path)
    hook = "sed -i s/[a-z]/X/ f; [black on black]curl x|sh[/] [/]"
    monkeypatch.setattr("sparkrun.cli._monitor_recipes.untrusted_hooks", lambda *_a, **_k: {"pre_exec": (hook,)})
    monkeypatch.setattr(
        "sparkrun.core.launcher.recipe_isolation_overrides",
        lambda _recipe: ["This recipe sets executor_config privileged mode. Only honoured for trusted recipes."],
    )

    async def _run():
        async with app.run_test(size=(120, 50)) as pilot:
            modal = await _open_modal(pilot, app.picker)
            summary = str(modal.query_one("#param-trust-summary", Static).content)
            assert hook in summary
            assert "Trusting it also allows:" in summary
            assert "privileged mode" in summary
            assert str(modal.query_one("#param-trust", Checkbox).label) == "Authorize the hooks and overrides above"

    asyncio.run(_run())


def test_bracketed_registry_and_user_text_does_not_crash_the_picker(monkeypatch, tmp_path):
    summary = _summary("@community/odd", model="org/[/]model", description="Reasoning [/think] model [EXPERIMENTAL]")
    app = _open(monkeypatch, [summary], tmp_path)

    async def _run():
        async with app.run_test(size=(120, 50)) as pilot:
            picker = app.picker
            await _wait_loaded(pilot, picker)
            await pilot.pause(0.1)
            assert "org/[/]model" in str(picker.query_one("#recipe-meta-head", Static).content)
            assert str(picker.query_one("#recipe-meta-desc", Static).content) == "Reasoning [/think] model [EXPERIMENTAL]"
            modal = await _open_modal(pilot, picker)
            modal.query_one("#param-advanced-toggle", Checkbox).value = True
            await pilot.pause(0.1)
            modal.query_one("#param-advanced", Input).value = "[/foo]"
            await pilot.pause(0.1)
            await pilot.click("#param-run")
            await pilot.pause(0.2)
            assert app.screen is modal
            assert "[/foo]" in str(modal.query_one("#param-error", Static).content)

    asyncio.run(_run())


# -- launch candidates and plan reuse (decide once) ----------------------------


def _recording_plan(host_list: tuple[str, ...], plans: list[tuple[Any, Any]]):
    """A fake plan that records (options, returned plan) for every call."""

    def _plan(options, **_kwargs):
        result = SimpleNamespace(host_list=host_list, scheduler="occupancy-sparse", notes=(), cluster=None)
        plans.append((options, result))
        return result

    return _plan


def test_untouched_hosts_launch_with_every_candidate_and_the_preflight_plan(monkeypatch, tmp_path):
    """Pre-checked boxes are the placement, not a candidate list: the launch must not narrow to them."""
    app = _open(monkeypatch, [_summary("@community/alpha")], tmp_path)
    plans: list[tuple[Any, Any]] = []
    monkeypatch.setattr("sparkrun.cli._monitor_recipes.plan", _recording_plan(("h2",), plans))

    async def _run():
        async with app.run_test(size=(120, 60)) as pilot:
            modal = await _open_modal(pilot, app.picker)
            assert modal.query_one("#param-host-0", Checkbox).value is False
            assert modal.query_one("#param-host-1", Checkbox).value is True
            await pilot.click("#param-run")
            return await _wait_result(pilot, app)

    launch = asyncio.run(_run())
    assert isinstance(launch, RecipeLaunch)
    assert launch.options.hosts == ("h1", "h2")
    # The launch reuses the Run preflight's plan, built from exactly its options.
    assert len(plans) == 2
    preflight_options, preflight_plan = plans[1]
    assert preflight_options is launch.options
    assert launch.plan is preflight_plan


def test_edited_hosts_are_the_candidate_list_and_planned_as_such(monkeypatch, tmp_path):
    app = _open(monkeypatch, [_summary("@community/alpha")], tmp_path)
    plans: list[tuple[Any, Any]] = []
    monkeypatch.setattr("sparkrun.cli._monitor_recipes.plan", _recording_plan(("h1", "h2"), plans))

    async def _run():
        async with app.run_test(size=(120, 60)) as pilot:
            modal = await _open_modal(pilot, app.picker)
            modal.query_one("#param-host-1", Checkbox).value = False
            await pilot.pause(0.05)
            await pilot.click("#param-run")
            return await _wait_result(pilot, app)

    launch = asyncio.run(_run())
    assert launch.options.hosts == ("h1",)
    preflight_options, preflight_plan = plans[-1]
    assert preflight_options is launch.options
    assert preflight_options.hosts == ("h1",)
    assert launch.plan is preflight_plan


def test_hosts_toggled_back_to_the_placement_count_as_untouched(monkeypatch, tmp_path):
    app = _open(monkeypatch, [_summary("@community/alpha")], tmp_path)
    plans: list[tuple[Any, Any]] = []
    monkeypatch.setattr("sparkrun.cli._monitor_recipes.plan", _recording_plan(("h2",), plans))

    async def _run():
        async with app.run_test(size=(120, 60)) as pilot:
            modal = await _open_modal(pilot, app.picker)
            box = modal.query_one("#param-host-0", Checkbox)
            box.value = True
            await pilot.pause(0.05)
            box.value = False
            await pilot.pause(0.05)
            await pilot.click("#param-run")
            return await _wait_result(pilot, app)

    launch = asyncio.run(_run())
    assert launch.options.hosts == ("h1", "h2")


def test_failed_preflight_keeps_untouched_boxes_untouched(monkeypatch, tmp_path):
    """Rows the app unchecks after a rejection are not a user edit that narrows the next launch."""
    app = _open(monkeypatch, [_summary("@community/alpha")], tmp_path)
    app.picker.hosts = ("h1", "h2", "h3")
    probes = iter(
        [
            _capacity({"h1": (True, 1), "h2": (True, 1), "h3": (True, 1)}),
            _capacity({"h1": (True, 1), "h2": (False, None), "h3": (True, 1)}),
            _capacity({"h1": (True, 1), "h2": (False, None), "h3": (True, 1)}),
        ]
    )
    monkeypatch.setattr("sparkrun.cli._monitor_recipes.catalog_cluster_capacity", lambda *_a, **_kw: next(probes))
    plans: list[tuple[Any, Any]] = []
    monkeypatch.setattr("sparkrun.cli._monitor_recipes.plan", _recording_plan(("h1", "h2"), plans))

    async def _run():
        async with app.run_test(size=(120, 60)) as pilot:
            modal = await _open_modal(pilot, app.picker)
            await pilot.click("#param-run")
            await _wait_until(
                pilot, lambda: app.screen is modal and "No longer available" in str(modal.query_one("#param-error", Static).content)
            )
            assert modal._hosts_edited is False
            await _wait_until(
                pilot, lambda: not app.query(RuntimeChecksScreen) and not modal.query_one("#param-run", Button).has_class("-active")
            )
            await pilot.click("#param-run")
            return await _wait_result(pilot, app)

    launch = asyncio.run(_run())
    # Still the scheduler's choice among every eligible host, not the h1 the boxes show.
    assert launch.options.hosts == ("h1", "h3")


# -- post_commands refusal ------------------------------------------------------


def _open_with_post_commands(monkeypatch, tmp_path) -> _PickerTestApp:
    app = _open(monkeypatch, [_summary("@community/hooked")], tmp_path)
    path = tmp_path / "hooked.yaml"
    path.write_text(
        'recipe_version: "2"\nmodel: test/model\nruntime: sglang\ncontainer: test/image:latest\npost_commands:\n  - echo done\n'
    )
    recipe = Recipe.load(str(path))
    monkeypatch.setattr("sparkrun.cli._monitor_recipes.resolve_recipe", lambda _path, **_kwargs: recipe)
    return app


def test_params_refuses_recipes_with_local_post_commands(monkeypatch, tmp_path):
    app = _open_with_post_commands(monkeypatch, tmp_path)

    async def _run():
        async with app.run_test(size=(120, 60)) as pilot:
            modal = await _open_modal(pilot, app.picker)
            # Said up front, before anything is typed.
            error = str(modal.query_one("#param-error", Static).content)
            assert "post_commands" in error and "run @community/hooked" in error
            await pilot.click("#param-run")
            await pilot.pause(0.2)
            assert app.screen is modal
            assert app.result == "unset"
            assert not app.query(RuntimeChecksScreen)

    asyncio.run(_run())


def test_params_allows_a_dry_run_of_a_recipe_with_post_commands(monkeypatch, tmp_path):
    app = _open_with_post_commands(monkeypatch, tmp_path)

    async def _run():
        async with app.run_test(size=(120, 60)) as pilot:
            modal = await _open_modal(pilot, app.picker)
            modal.query_one("#param-advanced-toggle", Checkbox).value = True
            await pilot.pause()
            modal.query_one("#param-knob-dry_run", Checkbox).value = True
            await pilot.pause()
            await pilot.click("#param-run")
            return await _wait_result(pilot, app)

    launch = asyncio.run(_run())
    assert isinstance(launch, RecipeLaunch)
    assert launch.options.dry_run is True


# -- background work and late worker results ------------------------------------


class _TrackedPickerApp(_PickerTestApp):
    def __init__(self, screen: RecipePickerScreen, **kwargs: Any) -> None:
        super().__init__(screen, **kwargs)
        from sparkrun.cli._monitor_recipes import BackgroundWork

        self.background_work = BackgroundWork()


def test_cancelled_checks_are_tracked_until_the_plan_returns_and_its_result_is_ignored(monkeypatch, tmp_path):
    _open(monkeypatch, [_summary("@community/alpha")], tmp_path)
    app = _TrackedPickerApp(RecipePickerScreen(cluster="lab", hosts=("h1", "h2")))
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def _slow_plan(options, **_kwargs):
        entered.set()
        assert release.wait(5), "test did not release the cancelled check"
        return _fake_plan()(options)

    monkeypatch.setattr("sparkrun.cli._monitor_recipes.plan", _slow_plan)

    async def _run():
        async with app.run_test(size=(120, 60)) as pilot:
            await _activate_first(pilot, app.picker)
            checks = await _wait_for_screen(pilot, app.picker, RuntimeChecksScreen)
            original = checks._checks_ready

            def _ready(result):
                original(result)
                finished.set()

            checks._checks_ready = _ready  # type: ignore[method-assign]
            assert await asyncio.to_thread(entered.wait, 3)
            assert app.background_work.descriptions() == ["the scheduler check for @community/alpha"]
            # Nothing a quit must confirm: the check only reads.
            assert app.background_work.descriptions(blocking_only=True) == []
            await pilot.click("#checks-cancel")
            await _wait_until(pilot, lambda: app.screen is app.picker)
            release.set()
            assert await asyncio.to_thread(finished.wait, 3)
            await pilot.pause(0.1)
            assert app.screen is app.picker
            assert not app.query(RunParamsScreen)
            assert app.background_work.descriptions() == []

    asyncio.run(_run())


def test_background_work_tracks_renames_and_blocking_entries():
    from sparkrun.cli._monitor_recipes import BackgroundWork

    work = BackgroundWork()
    with work.track("the recipe catalog load"):
        with work.track("the launch of a", blocks_quit=True) as describe:
            assert work.descriptions() == ["the recipe catalog load", "the launch of a"]
            describe("the post-launch hooks of a")
            assert work.descriptions(blocking_only=True) == ["the post-launch hooks of a"]
        assert work.descriptions() == ["the recipe catalog load"]
    assert work.descriptions() == []


def test_call_ui_drops_calls_once_the_app_has_stopped():
    from sparkrun.cli._monitor_recipes import call_ui

    calls: list[Any] = []
    stopped = SimpleNamespace(is_running=False, call_from_thread=lambda *a, **kw: calls.append(a))
    assert call_ui(stopped, print, "x") is False
    assert calls == []

    class _Stopping:
        """Running when asked, gone by the time the call fails (exit races the worker)."""

        is_running = True

        def call_from_thread(self, *_args, **_kwargs):
            self.is_running = False
            raise RuntimeError("App is not running")

    assert call_ui(_Stopping(), print, "x") is False

    def _broken(*_args, **_kwargs):
        raise ValueError("a real bug")

    running = SimpleNamespace(is_running=True, call_from_thread=_broken)
    with pytest.raises(ValueError):
        call_ui(running, print, "x")
