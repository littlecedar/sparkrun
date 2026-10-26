"""Local checkpoint usability without assigning HF identity or downloading."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sparkrun.core import model_distribution as md
from sparkrun.models.artifacts import ModelArtifactError
from sparkrun.models.host_io import ModelHostIO
from sparkrun.models.preparation import validate_preplaced
from sparkrun.models.runtime import verify_workload_access
from sparkrun.orchestration.executors.docker import DockerExecutor
from sparkrun.scripts.resolve_model_path import validate_local_model
from sparkrun.utils.shell import quote


def checkpoint(root):
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text('{"model_type":"fixture"}')
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": "weights.safetensors"}}))
    (root / "weights.safetensors").write_bytes(b"weights")
    return root


@pytest.fixture
def binding_scope():
    token = md._BINDINGS.set({})
    yield
    md._BINDINGS.reset(token)


def test_flat_model_remains_unverified_and_unchanged(tmp_path, caplog, binding_scope):
    root = checkpoint(tmp_path / "model with 'quotes'")
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    assert validate_preplaced(str(root), ["localhost"], model_format="hf") == ()
    assert md.prepared_models() == ()
    assert "unverified" in caplog.text
    assert {p.name: p.read_bytes() for p in root.iterdir()} == before


@pytest.mark.parametrize(
    "filename,contents,reason",
    [
        ("config.json", None, "config.json"),
        ("config.json", "{bad", "invalid.*JSON"),
        ("config.json", "[]", "must be an object"),
        ("weights.safetensors", None, "weights.safetensors"),
        ("model.safetensors.index.json", "{bad", "invalid.*JSON"),
        ("model.safetensors.index.json", '{"weight_map":{}}', "weight map"),
        ("model.safetensors.index.json", '{"weight_map":{"w":null}}', "invalid weight file"),
        ("model.safetensors.index.json", '{"weight_map":{"w":"../outside"}}', "invalid weight file"),
    ],
)
def test_invalid_local_checkpoint_reports_host_and_path(tmp_path, filename, contents, reason):
    root = checkpoint(tmp_path)
    target = root / filename
    if contents is None:
        target.unlink()
    else:
        target.write_text(contents)
    with pytest.raises(ModelArtifactError, match=reason) as error:
        validate_preplaced(str(root), ["localhost"], model_format="hf")
    assert "localhost" in str(error.value) and str(root) in str(error.value)


@pytest.mark.parametrize("kind", ["dangling", "directory"])
def test_invalid_weight_file_type(tmp_path, kind):
    root = checkpoint(tmp_path)
    weight = root / "weights.safetensors"
    weight.unlink()
    weight.symlink_to(root / "missing") if kind == "dangling" else weight.mkdir()
    with pytest.raises(ModelArtifactError, match="weights.safetensors"):
        validate_preplaced(str(root), ["localhost"], model_format="hf")


def test_readability_uses_open_not_permission_bits(tmp_path, monkeypatch):
    root = checkpoint(tmp_path)
    original = Path.open

    def deny_weights(path, *args, **kwargs):
        if path.name == "weights.safetensors":
            raise PermissionError("fixture denies access")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", deny_weights)
    with pytest.raises(ValueError, match="unreadable.*weights.safetensors"):
        validate_local_model(root, "hf")


def test_unindexed_model_requires_weights(tmp_path):
    root = checkpoint(tmp_path)
    (root / "model.safetensors.index.json").unlink()
    assert validate_preplaced(str(root), ["localhost"], model_format="hf") == ()
    (root / "weights.safetensors").unlink()
    with pytest.raises(ModelArtifactError, match="no model weight files"):
        validate_preplaced(str(root), ["localhost"], model_format="hf")


def test_native_gguf_and_split_gguf_need_no_config(tmp_path):
    single = tmp_path / "model.gguf"
    single.write_bytes(b"GGUFfixture")
    assert validate_preplaced(str(single), ["localhost"], model_format="hf") == ()
    shards = [tmp_path / f"model-{i:05}-of-00002.gguf" for i in (1, 2)]
    for path in shards:
        path.write_bytes(b"GGUFfixture")
    assert validate_preplaced(str(shards[0]), ["localhost"]) == ()
    shards[1].unlink()
    with pytest.raises(ModelArtifactError, match="00002-of-00002.gguf"):
        validate_preplaced(str(shards[0]), ["localhost"])


def test_other_runtime_directory_formats_do_not_require_hf_weights(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "rank0.engine").write_bytes(b"compiled engine")
    assert validate_preplaced(str(tmp_path), ["localhost"], model_format="opaque") == ()


def test_missing_on_one_target_is_a_failure(tmp_path, monkeypatch):
    root = checkpoint(tmp_path / "head")
    execute = ModelHostIO.execute

    def local_execute(self, host, script, **kwargs):
        return execute(self, "localhost", script, **kwargs)

    monkeypatch.setattr(ModelHostIO, "execute", local_execute)
    monkeypatch.setattr(ModelHostIO, "root", lambda self, host, path: str(root if host == "head" else tmp_path / "absent-worker"))
    with pytest.raises(ModelArtifactError, match="worker.*absent-worker"):
        validate_preplaced("/models/local", ["head", "worker"], model_format="hf")


@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("gguf", [False, True])
def test_unverified_model_is_checked_through_workload_mounts(tmp_path, monkeypatch, binding_scope, missing, gguf):
    host_root = checkpoint(tmp_path / "host")
    (host_root / "model.gguf").write_bytes(b"GGUFfixture")
    container_root = tmp_path / "container"
    shutil.copytree(host_root, container_root)
    model = host_root / "model.gguf" if gguf else host_root
    if missing:
        (container_root / ("model.gguf" if gguf else "weights.safetensors")).unlink()
    executor = DockerExecutor()

    def container_command(image, script, volumes, extra_opts):
        assert "python" not in script
        assert image == "fixture"
        assert volumes == {str(host_root): str(container_root)}
        return "printf %s " + quote(script) + " | bash -s"

    command = Mock(side_effect=container_command)
    monkeypatch.setattr(executor, "model_access_command", command)

    def verify():
        verify_workload_access(
            ["localhost"],
            {"localhost": executor},
            ["fixture"],
            {str(host_root): str(container_root)},
            {},
            local_model_path=str(model),
            local_model_format="hf",
        )

    if missing:
        with pytest.raises(ModelArtifactError, match="inaccessible.*localhost"):
            verify()
    else:
        verify()
    command.assert_called_once()
    assert md.prepared_models() == ()


@pytest.mark.parametrize("case", ["valid", "missing_config", "workload_missing", "dry_run"])
def test_launcher_checks_local_models_before_replacement(tmp_path, monkeypatch, case):
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.launcher import launch_inference
    from sparkrun.core.recipe import Recipe
    from sparkrun.runtimes.sglang import SglangRuntime

    root = checkpoint(tmp_path / "model")
    if case == "missing_config":
        (root / "config.json").unlink()
    config_path = tmp_path / "config.yaml"
    config_path.write_text("cache_dir: " + str(tmp_path / "cache"))
    config = SparkrunConfig(config_path)
    recipe = Recipe.from_dict({"model": str(root), "runtime": "sglang", "container": "fixture", "runtime_cache": False})
    runtime = SglangRuntime()
    events = []
    monkeypatch.setattr(runtime, "prepare", lambda *a, **kw: None)
    monkeypatch.setattr(runtime, "get_extra_volumes", lambda: {})
    monkeypatch.setattr(runtime, "run", lambda **kw: events.append("run") or 0)
    monkeypatch.setattr(runtime, "_collect_runtime_info", lambda *a, **kw: {})
    monkeypatch.setattr("sparkrun.core.launcher.resolve_effective_cache_dir", lambda *a, **kw: str(tmp_path))
    monkeypatch.setattr("sparkrun.core.launcher._verify_mount_sources", lambda *a, **kw: None)
    monkeypatch.setattr("sparkrun.orchestration.distribution.resolve_auto_transfer_mode", lambda *a, **kw: SimpleNamespace(mode="local"))

    def distribute(*args, **kwargs):
        assert kwargs["skip_model"] is True
        return None, {}, {}, {}

    monkeypatch.setattr("sparkrun.orchestration.distribution.distribute_from_config", distribute)
    monkeypatch.setattr("sparkrun.orchestration.primitives.try_clear_page_cache", lambda *a, **kw: None)
    monkeypatch.setattr("sparkrun.orchestration.job_metadata.save_job_metadata", lambda *a, **kw: None)
    execute = ModelHostIO.execute

    def local_execute(self, host, script, **kwargs):
        assert case != "dry_run", "dry-run must not probe model files"
        return execute(self, "localhost", script, **kwargs)

    monkeypatch.setattr(ModelHostIO, "execute", local_execute)

    def workload_command(self, image, script, volumes, extra_opts):
        events.append("access")
        assert volumes[str(root)] == str(root)
        if case == "workload_missing":
            script = script.replace(str(root), str(tmp_path / "not-mounted"))
        return "printf %s " + quote(script) + " | bash -s"

    monkeypatch.setattr(DockerExecutor, "model_access_command", workload_command)

    def launch():
        return launch_inference(
            recipe=recipe,
            runtime=runtime,
            host_list=["head", "worker"],
            overrides={},
            config=config,
            sync_tuning=False,
            dry_run=case == "dry_run",
            before_start=lambda: events.append("replace"),
            trust=True,
        )

    if case in {"missing_config", "workload_missing"}:
        with pytest.raises(ModelArtifactError, match="config.json" if case == "missing_config" else "inaccessible"):
            launch()
        assert "replace" not in events and "run" not in events
    else:
        assert launch().rc == 0
        assert events == (["run"] if case == "dry_run" else ["access", "access", "replace", "run"])


def test_leftover_bin_index_beside_safetensors_is_ignored(tmp_path):
    """A '*.safetensors' + '*.json' download keeps the .bin index without its shards."""
    root = checkpoint(tmp_path)
    (root / "pytorch_model.bin.index.json").write_text(json.dumps({"weight_map": {"w": "pytorch_model-00001-of-00001.bin"}}))
    files = validate_local_model(root, "hf")
    assert str(root / "weights.safetensors") in files
    assert not any("pytorch_model" in f for f in files)


def test_bin_index_is_validated_when_it_is_the_only_index(tmp_path):
    root = checkpoint(tmp_path)
    (root / "model.safetensors.index.json").unlink()
    (root / "pytorch_model.bin.index.json").write_text(json.dumps({"weight_map": {"w": "pytorch_model-00001-of-00001.bin"}}))
    with pytest.raises(ValueError, match="missing.*pytorch_model-00001-of-00001.bin"):
        validate_local_model(root, "hf")
    (root / "pytorch_model-00001-of-00001.bin").write_bytes(b"w")
    assert str(root / "pytorch_model-00001-of-00001.bin") in validate_local_model(root, "hf")
