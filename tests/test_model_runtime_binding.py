"""The command, env, and workload access probe consume one prepared artifact."""

import logging
import shlex
import subprocess
from unittest.mock import Mock

import pytest

from sparkrun.core import model_distribution as md
from sparkrun.core.env_templates import prepare_env_templates
from sparkrun.core.recipe import Recipe
from sparkrun.models.artifacts import CacheValidationReport, ValidatedModelBinding
from sparkrun.models.runtime import bind_runtime_models, validate_runtime_command, verify_workload_access
from sparkrun.models.observation import observation_script, parse_observations
from sparkrun.orchestration.executors.docker import DockerExecutor
from sparkrun.orchestration.executors._base import ExecutorConfig
from sparkrun.runtimes.sglang import SglangRuntime
from sparkrun.runtimes.trtllm import TrtllmRuntime
from sparkrun.runtimes.vllm_distributed import VllmDistributedRuntime
from test_model_artifacts import artifact, cache, SHA


@pytest.fixture
def binding_scope():
    token = md._BINDINGS.set({})
    yield
    md._BINDINGS.reset(token)


def bind(root, model="org/model", manifest=None):
    manifest = manifest or artifact()
    binding = ValidatedModelBinding(
        model, "localhost", str(root), manifest, CacheValidationReport("localhost", manifest.identity, "metadata", "complete")
    )
    md.record_binding(binding)
    return binding


@pytest.mark.parametrize("runtime_class", [SglangRuntime, VllmDistributedRuntime])
def test_structured_command_and_env_use_same_revision(tmp_path, monkeypatch, binding_scope, runtime_class):
    runtime = runtime_class()
    recipe = Recipe.from_dict(
        {
            "model": "org/model",
            "runtime": runtime.runtime_name,
            "container": "test",
            "env": {"MODEL": "{launch.model_path}", "REV": "{launch.model_revision}"},
        }
    )
    binding = bind(tmp_path)
    executor = DockerExecutor(ExecutorConfig(accelerator_vendor="cpu", privileged=False))
    volumes = {str(tmp_path): "/cache/huggingface"}
    overrides = {}
    bind_runtime_models(recipe, overrides, runtime, ["localhost"], {"localhost": executor}, volumes)
    command = runtime.generate_command(recipe, overrides=overrides, is_cluster=False)
    validate_runtime_command(recipe, overrides, runtime, command)
    assert overrides["_prepared_model_path"] in shlex.split(command)
    assert "--served-model-name org/model" in command
    assert SHA in command
    monkeypatch.setattr("sparkrun.core.env_templates.probe_model_path", Mock(side_effect=AssertionError("reprobe")))
    plan = prepare_env_templates(recipe, overrides, ["localhost"], "job", str(tmp_path), {})
    assert plan.models["localhost"].host_path == binding.model_path
    assert plan.render("localhost", volumes, None, executor) == {"MODEL": overrides["_prepared_model_path"], "REV": SHA}
    assert recipe.model_revision is None


def test_custom_command_cannot_load_unvalidated_revision(binding_scope, tmp_path):
    bind(tmp_path)
    recipe = Recipe.from_dict(
        {"model": "org/model", "runtime": "sglang", "container": "test", "command": "sglang serve org/model --revision main"}
    )
    runtime = SglangRuntime()
    overrides = {}
    bind_runtime_models(recipe, overrides, runtime, ["localhost"], {"localhost": DockerExecutor()}, {str(tmp_path): "/cache/huggingface"})
    with pytest.raises(ValueError, match="validated model"):
        validate_runtime_command(recipe, overrides, runtime, recipe.command)


def test_local_model_keeps_revision_from_validated_manifest(tmp_path, binding_scope):
    from sparkrun.models.preparation import validate_preplaced

    manifest = artifact()
    root = cache(tmp_path, manifest)
    (root / ".sparkrun-model-manifest.json").write_text(manifest.to_json())
    validate_preplaced(str(root), ["localhost"])
    recipe = Recipe.from_dict({"model": str(root), "runtime": "sglang", "container": "test", "env": {"REV": "{launch.model_revision}"}})
    executor = DockerExecutor()
    overrides = {}
    bind_runtime_models(recipe, overrides, SglangRuntime(), ["localhost"], {"localhost": executor}, {str(root): "/model"})
    plan = prepare_env_templates(recipe, overrides, ["localhost"], "job", str(tmp_path), {})
    assert plan.render("localhost", {str(root): "/model"}, None, executor) == {"REV": SHA}


@pytest.fixture
def prepared_command(binding_scope, tmp_path):
    bind(tmp_path)

    def prepare(command, *, runtime_class=VllmDistributedRuntime, recipe_version="2", model_revision=None):
        runtime = runtime_class()
        recipe = Recipe.from_dict(
            {
                "recipe_version": recipe_version,
                "model": "org/model",
                "model_revision": model_revision,
                "runtime": runtime.runtime_name,
                "container": "test",
                "command": command,
            }
        )
        overrides = {}
        bind_runtime_models(
            recipe, overrides, runtime, ["localhost"], {"localhost": DockerExecutor()}, {str(tmp_path): "/cache/huggingface"}
        )
        rendered = runtime.generate_command(recipe, overrides=overrides, is_cluster=False)
        return recipe, overrides, runtime, rendered

    return prepare


@pytest.mark.parametrize("recipe_version", ["1", "2"])
@pytest.mark.parametrize("model_revision", [None, SHA])
@pytest.mark.parametrize(
    "runtime_class, command",
    [
        (VllmDistributedRuntime, "vllm serve org/model"),
        (VllmDistributedRuntime, "vllm serve 'org/model'"),
        (VllmDistributedRuntime, "python -m vllm.entrypoints.openai.api_server --model org/model"),
        (SglangRuntime, "sglang serve org/model"),
        (SglangRuntime, 'python -m sglang.launch_server --model-path="org/model"'),
        (TrtllmRuntime, "trtllm-serve org/model"),
    ],
)
def test_matching_literal_model_warns_and_allows_launch(prepared_command, caplog, recipe_version, model_revision, runtime_class, command):
    args = prepared_command(command, runtime_class=runtime_class, recipe_version=recipe_version, model_revision=model_revision)
    with caplog.at_level(logging.WARNING, logger="sparkrun.models.runtime"):
        validate_runtime_command(*args)
    warnings = [record for record in caplog.records if record.name == "sparkrun.models.runtime"]
    assert len(warnings) == 1
    assert warnings[0].levelno == logging.WARNING
    assert "org/model" in warnings[0].message
    assert SHA in warnings[0].message
    assert "{model}" in warnings[0].message
    assert "allowing launch" in warnings[0].message


@pytest.mark.parametrize(
    "command",
    [
        "vllm serve {model}",
        "vllm serve org/model --revision {model_revision}",
        'vllm serve org/model --revision="{model_revision}"',
    ],
)
def test_bound_custom_command_still_passes_without_warning(prepared_command, caplog, command):
    args = prepared_command(command)
    with caplog.at_level(logging.WARNING, logger="sparkrun.models.runtime"):
        validate_runtime_command(*args)
    assert not [record for record in caplog.records if record.name == "sparkrun.models.runtime"]


@pytest.mark.parametrize(
    "revision_args",
    [
        "--revision main",
        "--revision=other-sha",
        "--revision",
        "--revision=",
        "--revision {model_revision} --revision main",
        "--revision main --revision {model_revision}",
    ],
)
def test_literal_model_with_explicit_conflicting_revision_still_fails(prepared_command, caplog, revision_args):
    args = prepared_command("vllm serve org/model " + revision_args)
    with pytest.raises(ValueError, match="validated model"):
        validate_runtime_command(*args)
    assert not [record for record in caplog.records if record.name == "sparkrun.models.runtime"]


@pytest.mark.parametrize(
    "command",
    [
        "vllm serve org/other",
        "vllm serve org/model-extra",
        "vllm serve org/other --served-model-name org/model",
        "python -m vllm.entrypoints.openai.api_server --model org/other --served-model-name=org/model",
        "vllm serve org/other --tokenizer org/model",
        "vllm serve org/other --chat-template 'vllm serve org/model'",
        "vllm serve org/other --chat-template '--model org/model'",
    ],
)
def test_other_model_or_incidental_literal_is_not_downgraded_to_warning(prepared_command, caplog, command):
    args = prepared_command(command)
    with pytest.raises(ValueError, match="validated model"):
        validate_runtime_command(*args)
    assert not [record for record in caplog.records if record.name == "sparkrun.models.runtime"]


def test_workload_visibility_detects_deleted_file(tmp_path, binding_scope):
    binding = bind(tmp_path)
    snapshot = cache(tmp_path, binding.manifest)
    executor = Mock(executor_name="local", config=ExecutorConfig())
    del executor.model_access_command
    verify_workload_access(["localhost"], {"localhost": executor}, ["test"], {}, {})
    (snapshot / "tokenizer.json").unlink()
    with pytest.raises(ValueError, match="inaccessible.*tokenizer.json"):
        verify_workload_access(["localhost"], {"localhost": executor}, ["test"], {}, {})


def test_docker_probe_uses_effective_user_and_mounts_without_gpu():
    executor = DockerExecutor(ExecutorConfig(user="1234:5678", accelerator_vendor="nvidia", privileged=False))
    executor.bind_image_references({"requested:tag": "sha256:" + "a" * 64})
    command = executor.model_access_command(
        "requested:tag", "echo probe", {"/cache": "/cache/huggingface:ro"}, ["--gpus", "all", "--entrypoint=ignored"]
    )
    tokens = shlex.split(command)
    assert tokens[tokens.index("--user") + 1] == "1234:5678"
    assert "/cache:/cache/huggingface:ro" in tokens
    assert "--pull=never" in tokens
    assert "--gpus" not in tokens and not any("nvidia.com/gpu" in t for t in tokens)
    assert tokens[tokens.index("--entrypoint") + 1] == "/bin/bash"
    assert executor.config.entrypoint is None


@pytest.mark.parametrize("entrypoint", [None, "", "/app/boot.py"])
@pytest.mark.parametrize("missing", [False, True])
def test_docker_probe_executes_observation_from_stdin(tmp_path, entrypoint, missing):
    manifest = artifact()
    snapshot = cache(tmp_path / "cache with 'quotes'", manifest)
    if missing:
        (snapshot / "tokenizer.json").unlink()
    executor = DockerExecutor(ExecutorConfig(entrypoint=entrypoint, accelerator_vendor="cpu", privileged=False))
    command = executor.model_access_command("probe:test", observation_script(manifest, str(snapshot)), {}, [])
    # Execute the generated pipeline with Docker's ENTRYPOINT + CMD semantics.
    # This needs no daemon and catches a second `bash` being mistaken for a
    # script filename, or a command wrapper consuming the observation stdin.
    docker = """
docker() {
    local entrypoint
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --entrypoint) entrypoint=$2; shift 2 ;;
            probe:test) shift; "$entrypoint" "$@"; return ;;
            *) shift ;;
        esac
    done
    return 99
}
"""
    result = subprocess.run(["bash"], input=docker + command, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    report = parse_observations("localhost", manifest, result.stdout, returncode=result.returncode)
    assert report.complete is not missing
    assert [(failure.path, failure.reason) for failure in report.failures] == ([("tokenizer.json", "missing")] if missing else [])
    assert executor.config.entrypoint == entrypoint


def test_workload_probe_failure_includes_bounded_stderr(tmp_path, monkeypatch, binding_scope):
    from sparkrun.models.host_io import ModelHostIO
    from sparkrun.transports.session import HostCommandResult

    bind(tmp_path)
    executor = DockerExecutor(ExecutorConfig(accelerator_vendor="cpu", privileged=False))
    diagnostic = "/usr/bin/bash: /usr/bin/bash: cannot execute binary file"
    monkeypatch.setattr(
        ModelHostIO,
        "execute",
        Mock(return_value=HostCommandResult("localhost", 126, b"", ("x" * 2000 + "\n" + diagnostic).encode())),
    )
    with pytest.raises(ValueError, match="rc=126") as error:
        verify_workload_access(["localhost"], {"localhost": executor}, ["probe:test"], {str(tmp_path): "/cache/huggingface"}, {})
    assert str(error.value).endswith(diagnostic)
    assert len(str(error.value).split("; stderr: ", 1)[1]) == 1500


def test_shadow_mount_refused_before_access(tmp_path, binding_scope):
    bind(tmp_path)
    executor = DockerExecutor(ExecutorConfig(volumes=["/other:/cache/huggingface/hub"]))
    with pytest.raises(ValueError, match="shadow"):
        verify_workload_access(["localhost"], {"localhost": executor}, ["image"], {str(tmp_path): "/cache/huggingface"}, {})
