"""Public, early application entry points (application profile API version 1)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path
    from scitrera_app_framework import Variables
    from sparkrun.core.context import SparkrunContext

from sparkrun.core.application_profile import (
    ApplicationProfile,
    DefaultPlugin,
    UpdateSource,
    APPLICATION_PROFILE_API_VERSION,
    child_environment,
    get_application_profile,
    initialize_child_application_profile,
    select_application_profile,
)
from sparkrun.core.application_identity import ApplicationIdentity, ControllerIdentity, get_application_identity, get_controller_identity

__all__ = (
    "ApplicationProfile",
    "DefaultPlugin",
    "UpdateSource",
    "APPLICATION_PROFILE_API_VERSION",
    "ApplicationIdentity",
    "ControllerIdentity",
    "get_application_profile",
    "get_application_identity",
    "get_controller_identity",
    "child_environment",
    "initialize",
    "run_cli",
    "main",
)


def initialize(
    profile: ApplicationProfile | None = None,
    *,
    config_path: str | Path | None = None,
    variables: Variables | None = None,
) -> SparkrunContext:
    """Initialize a shared context without importing or invoking the CLI.

    Initialize once before starting application workers. The process binds to
    one profile/config path; successful calls with the same binding reuse its
    plugin registry. Variables may be injected on the first call; later calls
    may omit them or pass that same instance. A different instance is rejected.
    A plugin-bootstrap exception poisons that initialization:
    subsequent calls raise with the original cause and require a new process.
    This avoids returning a context whose plugins only partly registered.
    """
    select_application_profile(profile) if profile is not None else initialize_child_application_profile()
    from sparkrun.core.bootstrap import init_sparkrun
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.context import SparkrunContext

    config = SparkrunConfig(config_path)
    return SparkrunContext(variables=init_sparkrun(v=variables, config=config), config=config)


def run_cli(profile: ApplicationProfile | None = None, *, args=None, **kwargs):
    """Select identity before importing Click commands, including eager options."""
    profile = select_application_profile(profile) if profile is not None else initialize_child_application_profile()
    from sparkrun.cli import main

    return main(args=args, prog_name=profile.command, complete_var="_%s_COMPLETE" % profile.command.upper().replace("-", "_"), **kwargs)


def main():
    run_cli()
