"""Read a prepared HF snapshot on the target. Stdlib-only; no downloads."""

import json
import os
import re
import sys
from pathlib import Path


def _readable_file(path):
    if not path.is_file():
        raise ValueError("missing or non-regular model file: " + str(path))
    try:
        # Open as the executing user, rather than trusting permission bits.
        # Do not read or hash the weight payload.
        with path.open("rb"):
            pass
    except OSError as exc:
        raise ValueError("unreadable model file: %s (%s)" % (path, exc)) from exc


def _json_object(path):
    _readable_file(path)
    try:
        data = json.loads(path.read_text())
    except (ValueError, OSError) as exc:
        raise ValueError("invalid or unreadable model JSON: %s (%s)" % (path, exc)) from exc
    if not isinstance(data, dict):
        raise ValueError("model JSON must be an object: " + str(path))
    return data


def validate_local_model(path: Path, model_format: str = "auto") -> list[str]:
    """Check basic usability, without inventing an inventory or revision.

    Native HF loaders request ``hf``. ``auto`` recognizes HF directories by
    config.json; ``opaque`` leaves other runtimes' directory layouts alone.
    GGUF files, including split files, are recognized in all modes.
    """
    if not path.exists():
        raise ValueError("local model path does not exist: " + str(path))
    if not path.is_dir():
        _readable_file(path)
        files = {str(path)}
        if path.suffix.lower() == ".gguf":
            match = re.fullmatch(r"(.+)-(\d{5})-of-(\d{5})\.gguf", path.name, re.IGNORECASE)
            if match:
                total = int(match[3])
                if not 1 <= int(match[2]) <= total:
                    raise ValueError("invalid GGUF shard name: " + str(path))
                for i in range(1, total + 1):
                    shard = path.with_name(f"{match[1]}-{i:05}-of-{total:05}{path.suffix}")
                    _readable_file(shard)
                    files.add(str(shard))
        elif model_format == "hf":
            raise ValueError("expected an HF model directory or GGUF file: " + str(path))
        return sorted(files)

    config = path / "config.json"
    if model_format == "opaque" or (model_format != "hf" and not config.exists() and not config.is_symlink()):
        # Enumerating checks directory access without requiring a particular
        # layout for compiled engines or third-party runtime formats.
        with os.scandir(path):
            pass
        return []
    _json_object(config)
    files = {str(config)}
    # Choose the index the way the loaders choose weights: safetensors first.
    # A download filtered to "*.safetensors" + "*.json" keeps the
    # pytorch_model.bin index (it is JSON) without the .bin shards it names,
    # and the loaders never read it, so requiring its shards refused a usable
    # checkpoint.
    indexes = sorted(path.glob("*.safetensors.index.json")) or sorted(path.glob("*.index.json"))
    if indexes:
        for index in indexes:
            weights = _json_object(index).get("weight_map")
            files.add(str(index))
            if not isinstance(weights, dict) or not weights:
                raise ValueError("checkpoint index has no valid weight map: " + str(index))
            for name in weights.values():
                if not isinstance(name, str) or not name or "\x00" in name:
                    raise ValueError("invalid weight file in checkpoint index: " + str(index))
            for name in sorted(set(weights.values())):
                relative = Path(name)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("invalid weight file %r in checkpoint index: %s" % (name, index))
                _readable_file(path / relative)
                files.add(str(path / relative))
    else:
        weights = sorted({p for pattern in ("*.safetensors", "*.bin", "*.pt", "*.gguf") for p in path.glob(pattern)})
        if not weights:
            raise ValueError("no model weight files found in: " + str(path))
        for weight in weights:
            _readable_file(weight)
            files.add(str(weight))
    return sorted(files)


def resolve_model_path(model, revision, cache_dir, expected_revision="", local_format="auto", include_local_files=False):
    """Return an exact model path and HF commit, or an empty local revision."""
    if Path(model).is_absolute():
        path = Path(model)
        if not path.exists():
            raise ValueError("local model path does not exist: " + str(path))
        result: dict[str, str | list[str]] = {"path": str(path), "revision": ""}
        if include_local_files:
            result["files"] = validate_local_model(path, local_format)
        return result
    repo, separator, quant = model.partition(":")
    if any(part in ("", ".", "..") for part in repo.split("/")) or len(repo.split("/")) != 2:
        raise ValueError("model must be a local absolute path or org/repository")
    cache = Path(cache_dir).expanduser().absolute() / "hub" / ("models--" + repo.replace("/", "--"))
    snapshots = cache / "snapshots"
    requested = revision or "main"
    if requested.startswith("/") or ".." in requested.split("/"):
        raise ValueError("invalid model revision")
    if expected_revision:
        commit = expected_revision
    elif re.fullmatch(r"[0-9a-fA-F]{40}", requested):
        commit = requested
    else:
        ref = cache / "refs" / requested
        if not ref.is_file():
            raise ValueError("requested revision has no cached ref; pin a cached commit explicitly")
        commit = ref.read_text().strip()
    if not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
        raise ValueError("cached revision is not a concrete commit")
    snapshot = snapshots / commit
    if not snapshot.is_dir():
        raise ValueError("required checkpoint revision is not present on this node")
    if separator or "gguf" in repo.lower():
        candidates = sorted(
            p for p in snapshot.rglob("*.gguf") if "mmproj" not in p.name.lower() and (not quant or quant.lower() in p.name.lower())
        )
        if not candidates or any(not p.is_file() for p in candidates):
            raise ValueError("GGUF checkpoint files are absent or incomplete")
        path = candidates[0]
        match = re.fullmatch(r"(.+)-(\d{5})-of-(\d{5})\.gguf", path.name)
        if match:
            total = int(match[3])
            expected = [path.with_name(f"{match[1]}-{i:05}-of-{total:05}.gguf") for i in range(1, total + 1)]
            if total < 1 or candidates != expected:
                raise ValueError("GGUF checkpoint is ambiguous or has missing shards")
        elif len(candidates) != 1:
            raise ValueError("GGUF checkpoint is ambiguous; select a single quantization")
    else:
        path = snapshot
        if not (path / "config.json").is_file():
            raise ValueError("checkpoint config.json is absent or incomplete")
        indexes = sorted(path.glob("*.index.json"))
        for index in indexes:
            data = json.loads(index.read_text())
            weights = data.get("weight_map") if isinstance(data, dict) else None
            if not isinstance(weights, dict) or not weights or any(not isinstance(name, str) for name in weights.values()):
                raise ValueError("checkpoint index has no valid weight map")
            for name in set(weights.values()):
                relative = Path(name)
                if relative.is_absolute() or ".." in relative.parts or not (path / relative).is_file():
                    raise ValueError("checkpoint index references a missing or invalid weight file")
    return {"path": str(path), "revision": commit}


if __name__ == "__main__":
    try:
        print("SPARKRUN_MODEL_PATH=" + json.dumps(resolve_model_path(*json.loads(sys.argv[1]))))
    except (ValueError, OSError, KeyError) as exc:
        print("Model path resolution failed: " + str(exc), file=sys.stderr)
        sys.exit(1)
