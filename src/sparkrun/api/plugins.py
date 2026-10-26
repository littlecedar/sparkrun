"""Plugin management without application bootstrap (including recovery)."""

from sparkrun.core.config import SparkrunConfig
from sparkrun.core.plugin_lifecycle import change_plugin


def _config(config, sctx):
    """Explicit *config* wins, then the shared context's, then a fresh load."""
    if config is not None:
        return config
    return sctx.config if sctx is not None else SparkrunConfig()


def enable(name, *, config=None, dry_run=False, sctx=None):
    """Enable loading, retaining capability settings; run the explicit enable hook."""
    return change_plugin(_config(config, sctx), name, True, dry_run=dry_run)


def disable(name, *, config=None, dry_run=False, sctx=None):
    """Disable loading, retaining settings and allowing broken-plugin recovery."""
    return change_plugin(_config(config, sctx), name, False, dry_run=dry_run)


def sync_defaults(*, config=None, dry_run=False, sctx=None):
    """Install missing trusted defaults; dry runs use local metadata only."""
    from sparkrun.core.plugin_defaults import reconcile_default_plugins

    return reconcile_default_plugins(_config(config, sctx), dry_run=dry_run)
