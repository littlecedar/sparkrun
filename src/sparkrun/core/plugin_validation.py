"""Private bounded child-process validation protocol for staged plugin packages."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from sparkrun.core.plugin_manifest import PluginError, PluginMetadata

PROTOCOL_VERSION = 1


def validate_candidate(
    metadata: PluginMetadata, package: Path, config_path: Path, *, timeout: int = 60, default_managed: bool = False
) -> None:
    from sparkrun.core.application_profile import child_environment

    with tempfile.TemporaryDirectory(prefix="sparkrun-plugin-check-") as temporary:
        request = Path(temporary) / "request.json"
        result_path = Path(temporary) / "result.json"
        request.write_text(
            json.dumps(
                {
                    "protocol": PROTOCOL_VERSION,
                    "metadata": metadata.to_dict(),
                    "package": str(package.resolve()),
                    "result": str(result_path),
                    "default_managed": default_managed,
                }
            )
        )
        env = {**os.environ, **child_environment(config_path=config_path), "PYTHONDONTWRITEBYTECODE": "1", "SPARKRUN_NO_TELEMETRY": "1"}
        env["SPARKRUN_NO_DEFAULT_PLUGINS"] = "1"
        try:
            result = subprocess.run(
                [sys.executable, "-m", __name__, str(request)], env=env, capture_output=True, text=True, timeout=timeout
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise PluginError("Plugin load validation failed or timed out: %s" % exc) from exc
        if result.returncode or not result_path.exists():
            raise PluginError(
                "Plugin load validation failed: %s" % (result.stderr or result.stdout or "child exited without validation")[-4000:]
            )
        try:
            response = json.loads(result_path.read_text())
        except ValueError as exc:
            raise PluginError("Invalid plugin validation response") from exc
        if response != {"protocol": PROTOCOL_VERSION, "loaded": metadata.name}:
            raise PluginError("Invalid plugin validation response")


def main():
    from sparkrun.core.application_profile import initialize_child_application_profile
    from sparkrun.core.config import SparkrunConfig
    import sparkrun.core.managed_plugins as managed

    request = json.loads(Path(sys.argv[1]).read_text())
    if request.get("protocol") != PROTOCOL_VERSION:
        raise PluginError("Unsupported plugin validation protocol")
    initialize_child_application_profile()
    metadata = PluginMetadata.from_dict(request["metadata"])
    config = SparkrunConfig()
    # Explicit validation is authorized to execute this candidate even when the
    # final capability gate is off. No overrides are persisted.
    from sparkrun.core.application_profile import env_name
    from sparkrun.core.bootstrap import init_sparkrun
    from sparkrun.core.external_plugins import loaded_plugin_module

    if metadata.feature:
        os.environ[env_name("FEATURE_" + metadata.feature.upper().replace(".", "_").replace("-", "_"))] = "1"
    record = {"metadata": metadata.to_dict()}
    if request.get("default_managed") is True:
        record["trust"] = {"granted": True, "default_managed": True}
    managed._validation_candidate = (record, Path(request["package"]))
    init_sparkrun(config=config)
    if loaded_plugin_module(metadata.module) is None:
        raise PluginError("Candidate did not register successfully")
    Path(request["result"]).write_text(json.dumps({"protocol": PROTOCOL_VERSION, "loaded": metadata.name}))


if __name__ == "__main__":
    main()
