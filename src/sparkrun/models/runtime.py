"""Consume prepared artifacts without resolving refs or scanning the cache again."""

from __future__ import annotations

from dataclasses import replace
import json
import logging
from pathlib import PurePosixPath
import shlex

from sparkrun.core.env_templates import _mounts, workload_model_path
from sparkrun.core.model_distribution import prepared_model, prepared_models
from sparkrun.core.validation import _model_arg_literal
from sparkrun.models.artifacts import ModelArtifactError
from sparkrun.models.observation import local_model_access_script, local_model_probe_script, observation_script, parse_observations
from sparkrun.utils.shell import quote

logger = logging.getLogger(__name__)


def bind_runtime_models(recipe, overrides: dict, runtime, hosts, executors, volumes) -> None:
    """Pin normal HF loaders; use selected files for GGUF and directory loaders.

    Recipe identity and served names remain declarative. Only operation-local
    overrides carry prepared paths and immutable revisions.
    """
    bindings = [prepared_model(recipe.model, host) for host in hosts]
    if not any(bindings):
        return
    if not all(bindings):
        raise ModelArtifactError("served model was not validated on every launch host")
    binding = bindings[0]
    assert binding is not None
    if any(b is not None and b.manifest.identity != binding.manifest.identity for b in bindings):
        raise ModelArtifactError("launch hosts have different model artifacts")
    paths = {
        workload_model_path(b.model_path, _mounts(volumes, executors[b.host]), executors[b.host].executor_name)
        for b in bindings
        if b is not None
    }
    if len(paths) != 1:
        raise ModelArtifactError("this runtime requires the same workload model path on every host")
    path = paths.pop()
    overrides["model_revision"] = binding.manifest.revision
    overrides["_prepared_model_path"] = path
    overrides["_prepared_model_revision"] = binding.manifest.revision
    # Templated commands explicitly referring to {model} use the prepared path.
    # Matching literal IDs without a revision retain legacy behavior with a
    # warning; explicit revision conflicts are still rejected below.
    if recipe.command or binding.manifest.weight_path:
        overrides["model"] = path
    if binding.manifest.weight_path:
        overrides["_gguf_model_path"] = path
    if binding.manifest.projector_path:
        projector = str(PurePosixPath(binding.snapshot_path) / binding.manifest.projector_path)
        overrides["mmproj"] = overrides["_mmproj_path"] = workload_model_path(
            projector, _mounts(volumes, executors[hosts[0]]), executors[hosts[0]].executor_name
        )
    if runtime.get_family() in {"vllm", "sglang"}:
        overrides["revision"] = binding.manifest.revision
    config = recipe.build_config_chain(overrides)

    def draft_binding(model):
        matches = [b for b in prepared_models() if b.model == model]
        if len({b.manifest.identity for b in matches}) > 1:
            raise ModelArtifactError("draft model differs between launch hosts")
        return matches[0] if matches else None

    # Draft artifacts have independent identities, never the served model SHA.
    if runtime.get_family() == "sglang":
        draft = config.get("speculative_draft_model_path") or config.get("speculative_draft_model")
        prepared = draft_binding(str(draft)) if draft else None
        if prepared:
            overrides["speculative_draft_model_revision"] = prepared.manifest.revision
    elif runtime.get_family() == "vllm":
        spec = config.get("speculative_config")
        if spec:
            values = json.loads(spec) if isinstance(spec, str) else dict(spec)
            prepared = draft_binding(str(values.get("model", "")))
            if prepared:
                values["revision"] = prepared.manifest.revision
                overrides["speculative_config"] = json.dumps(values)


def validate_runtime_command(recipe, overrides, runtime, command: str) -> None:
    """Reject conflicting bindings; warn for an unpinned matching model ID."""
    path = overrides.get("_prepared_model_path")
    if not path:
        return
    tokens = shlex.split(command)
    if path in tokens or any(token.endswith("=" + path) for token in tokens):
        return
    revision = overrides["_prepared_model_revision"]
    if recipe.model not in tokens and not any(t.endswith("=" + recipe.model) for t in tokens):
        raise ModelArtifactError("serve command does not identify the validated model")
    flags = runtime.model_revision_flags()
    revisions = []
    for i, token in enumerate(tokens):
        if token in flags:
            revisions.append(tokens[i + 1] if i + 1 < len(tokens) else "")
        elif any(token.startswith(flag + "=") for flag in flags):
            revisions.append(token.split("=", 1)[1])
    if revisions and all(value == revision for value in revisions):
        return
    # Use the same model-argument recognition as recipe validation, with
    # shell quoting already removed. An alias or tokenizer matching the model
    # must not qualify for the compatibility exception.
    if not revisions and _model_arg_literal(tokens, recipe.model):
        logger.warning(
            "Serve command uses recipe model '%s' without pinning validated revision '%s'; allowing launch for compatibility. "
            "The engine may load a different cached revision. Use {model} or a supported revision flag with {model_revision} "
            "to consume the validated model.",
            recipe.model,
            revision,
        )
        return
    raise ModelArtifactError(
        "serve command does not consume the validated model; use {model} or a supported revision flag with {model_revision}"
    )


def _access_command(executor, image, script, volumes, extra_opts):
    hook = getattr(executor, "model_access_command", None)
    if callable(hook):
        command = hook(image, script, volumes, list(extra_opts))
    elif executor.executor_name == "local":
        command = "printf %s " + quote(script) + " | bash -s"
    else:
        raise ModelArtifactError("executor does not support model workload access validation: " + executor.executor_name)
    if not isinstance(command, str) or not command:
        raise ModelArtifactError("executor returned an invalid model access command")
    return command


def verify_workload_access(
    hosts, executors, images, volumes, ssh_kwargs, *, extra_opts=(), local_model_path=None, local_model_format="auto"
) -> None:
    """Check model files through workload mounts and the effective UID.

    Probes run before eviction, without GPUs or the image entrypoint. Third-party
    executors can supply model_access_command(image, script, volumes, extra_opts).
    They must preserve the final workload's filesystem and identity semantics.
    Local models without inventories receive readability checks only.
    """
    from sparkrun.models.host_io import ModelHostIO
    from sparkrun.orchestration.executors._seccomp import split_options

    bindings = prepared_models()
    if not bindings and not local_model_path:
        return
    _, tokens, _ = split_options(None, list(extra_opts))
    if any(
        t in {"-v", "--volume", "--mount", "--tmpfs", "--volumes-from"}
        or t.startswith(("-v", "--volume=", "--mount=", "--tmpfs=", "--volumes-from="))
        for t in tokens
    ):
        raise ModelArtifactError("model access checks require structured executor_config.volumes instead of raw mount options")
    with ModelHostIO(ssh_kwargs) as io:
        for binding in bindings:
            if binding.host not in hosts:
                continue
            executor = executors[binding.host]
            mounts = _mounts(volumes, executor)
            snapshot = workload_model_path(binding.snapshot_path, mounts, executor.executor_name)
            script = observation_script(binding.manifest, snapshot)
            command = _access_command(executor, images[hosts.index(binding.host)], script, volumes, extra_opts)
            result = io.execute(binding.host, command, timeout=90)
            report = parse_observations(
                binding.host, binding.manifest, result.stdout.decode(errors="replace"), returncode=result.returncode
            )
            report = replace(report, access_context=executor.executor_name)
            if not report.complete:
                details = "; ".join(f.path + ": " + f.reason for f in report.failures[:5])
                details = details or report.detail
                stderr = result.stderr.decode(errors="replace").strip()[-1500:]
                if stderr:
                    details += "; stderr: " + stderr
                raise ModelArtifactError("model is inaccessible to workload on %s: %s" % (binding.host, details))
        if local_model_path:
            for host, image in zip(hosts, images, strict=True):
                if any(b.host == host and b.model == local_model_path for b in bindings):
                    continue  # The authoritative inventory was checked above.
                executor = executors[host]
                if executor.executor_name != "local" and not callable(getattr(executor, "model_access_command", None)):
                    logger.warning("Executor %s cannot check local model access inside the workload on %s", executor.executor_name, host)
                    continue
                # Host Python understands config/index JSON. The image only
                # needs bash to open that file selection as the serving user.
                # This remains a usability check, not an authoritative binding.
                result = io.execute(host, local_model_probe_script(local_model_path, local_model_format), timeout=60)
                if result.returncode:
                    raise ModelArtifactError(
                        "local model preflight failed on %s at %s: %s"
                        % (host, local_model_path, result.stderr.decode(errors="replace").strip()[-1500:])
                    )
                try:
                    lines = result.stdout.decode().splitlines()
                    data = json.loads(
                        next(line.removeprefix("SPARKRUN_MODEL_PATH=") for line in lines if line.startswith("SPARKRUN_MODEL_PATH="))
                    )
                    files = data["files"]
                    if not isinstance(files, list) or any(not isinstance(p, str) or not p.startswith("/") or "\x00" in p for p in files):
                        raise ValueError("invalid local model file selection")
                except (ValueError, KeyError, TypeError, StopIteration) as exc:
                    raise ModelArtifactError("invalid local model probe response from " + host) from exc
                mounts = _mounts(volumes, executor)
                path = workload_model_path(local_model_path, mounts, executor.executor_name)
                files = [workload_model_path(file, mounts, executor.executor_name) for file in files]
                script = local_model_access_script(path, files)
                command = _access_command(executor, image, script, volumes, extra_opts)
                result = io.execute(host, command, timeout=90)
                if result.returncode:
                    raise ModelArtifactError(
                        "local model is inaccessible to workload on %s at %s (rc=%s): %s"
                        % (host, path, result.returncode, result.stderr.decode(errors="replace").strip()[-1500:])
                    )
