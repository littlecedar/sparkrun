"""Managed plugin lifecycle and metadata-only recovery commands."""

from __future__ import annotations

from functools import wraps
import logging
import sys

import click

from sparkrun.core.plugin_manifest import PluginError
from ._common import json_option, print_json

logger = logging.getLogger(__name__)


def management_config(ctx):
    from sparkrun.core.config import SparkrunConfig
    from ._common import _bind_application_config

    _bind_application_config(ctx)
    return SparkrunConfig((ctx.find_root().obj or {}).get("config_path"))


def plugin_errors(function):
    @wraps(function)
    def call(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (ValueError, OSError) as exc:
            raise click.ClickException(str(exc)) from exc

    return call


def _manager(ctx):
    from sparkrun.core.plugin_manager import PluginManager

    return PluginManager(management_config(ctx))


def _trust(details):
    click.echo("Plugin: %s %s (%s)" % (details["name"], details["version"], details["module"]))
    click.echo("Source: %s @ %s" % (details["source"].get("fetch_url", details["source"].get("url")), details["revision"]))
    if details["repository_mismatch"]:
        click.echo("The manifest declares a different repository; trust applies only to this supplied source.")
    if details["enable_external"]:
        click.echo("This will enable core.external_plugins.")
    if details.get("enable_plugin"):
        click.echo("This will enable plugin %s; capability settings are preserved." % details["enable_plugin"])
    click.echo("Plugin code runs with your permissions. Trust covers future compatible updates during sparkrun update.")
    if not sys.stdin.isatty():
        return False
    return click.confirm("Trust this source and install its plugin?", default=False)


def _legacy_trust(details):
    click.echo("Enabling external loading will also activate: %s" % ", ".join(details["legacy_paths"]))
    return sys.stdin.isatty() and click.confirm("Trust and enable these existing plugin paths too?", default=False)


def _show_result(result):
    click.echo(
        "%s: %s%s%s"
        % (
            result["name"],
            result["status"],
            " " + result["version"] if result.get("version") else "",
            " @ " + result["revision"][:12] if result.get("revision") else "",
        )
    )
    for message in result.get("messages", []):
        click.echo(message)
    if result.get("reason"):
        click.echo("  " + result["reason"])
    if result.get("error"):
        click.echo("  " + result["error"], err=True)
    if result.get("restart_required"):
        click.echo("Restart long-running plugin consumers to apply this change.")
    if result.get("warning"):
        click.echo("Warning: " + result["warning"], err=True)
    if result["status"] in {"updated", "rolled back", "removed"} or result.get("history_pruned"):
        click.echo("Restart long-running plugin consumers; older generations may be pruned.")


@click.group("plugins", invoke_without_command=True)
@click.pass_context
def plugins(ctx):
    """Manage plugin loading, installations, and compatibility.

    With no subcommand, list enabled plugins (`plugins list --all` includes
    disabled ones). Listing covers bundled plugins, directory plugins,
    installed integrations, and managed external plugins.
    Add, update, rollback, and remove manage plugins installed with this group.
    Sync installs missing application defaults; --dry-run previews the catalog.

    Managed installations are stored beside the active configuration, separately
    from the Python environment, and survive {app_command} upgrades. The
    `setup plugins` group is an alias with the same commands.

    \b
    Examples:
      {app_command} plugins list
      {app_command} plugins add https://github.com/sparksq/sparkrun-sparkroute-plugin
      {app_command} plugins update --all
    """
    if ctx.invoked_subcommand is None:
        ctx.invoke(plugins_list)


@plugins.command("enable")
@click.argument("name")
@click.option("--dry-run", is_flag=True, help="Preview the plugin loading change without running hooks or saving.")
@click.pass_context
@plugin_errors
def plugins_enable(ctx, name, dry_run):
    """Allow plugin NAME to load, preserving its capability settings.

    Works for bundled, installed, directory and managed plugins. Enabling a
    gateway plugin does not select it; explicitly enable its gateway feature.
    For a missing default, clears its removal opt-out; run plugins sync to install.
    """
    from sparkrun.core.plugin_lifecycle import change_plugin

    _show_result(change_plugin(management_config(ctx), name, True, dry_run=dry_run))


@plugins.command("disable")
@click.argument("name")
@click.option("--dry-run", is_flag=True, help="Preview disabling without running hooks or saving.")
@click.pass_context
@plugin_errors
def plugins_disable(ctx, name, dry_run):
    """Prevent plugin NAME from loading; retain its installation and settings.

    A disabled plugin contributes no commands, operations or feature definitions
    to new processes. Existing services are not stopped. This command remains
    usable when the plugin is broken; optional cleanup hook failures are reported.
    Also prevents installation of a declared default that is not installed yet.
    """
    from sparkrun.core.plugin_lifecycle import change_plugin

    _show_result(change_plugin(management_config(ctx), name, False, dry_run=dry_run))


@plugins.command("list", short_help="Show plugin versions, state, and compatibility.")
@json_option(help="Output plugin metadata, compatibility reasons, and load-check results as JSON")
@click.option("--all", "include_all", is_flag=True, help="Include disabled plugins (default: enabled plugins only)")
@click.option(
    "--check", is_flag=True, help="Also execute trusted managed plugins in child processes to check loading (60-second timeout each)"
)
@click.pass_context
@plugin_errors
def plugins_list(ctx, output_json, include_all, check):
    """List enabled plugin sources, versions, and compatibility with this host.

    Disabled plugins are omitted unless --all is given (in --json output too).
    STATE shows the effective plugin loading policy;
    COMPATIBILITY checks declared requirements against the current installation.
    Explanations for unknown or incompatible results follow the table.

    FEATURES lists declared capability flags that can be configured separately
    within a plugin. Names do not indicate whether those features are enabled;
    use `{app_command} setup features list` to inspect their effective state.
    `none` means no separate capability flags are declared; `unavailable` means
    their declaration metadata could not be read. Disabled plugins still show
    their declarations, but their capabilities are inactive. Legacy whole-plugin
    flags are represented by STATE; use `plugins enable/disable` to control loading.

    Listing reads metadata without importing plugin code. Use --check to also
    validate loading for the listed managed plugins (with --all, disabled ones
    too). JSON reports actual loaded status separately from compatibility and
    load-check results.

    \b
    Examples:
      {app_command} plugins list
      {app_command} plugins list --all
      {app_command} plugins list --json
      {app_command} plugins list --check
    """
    from sparkrun.core.plugin_inventory import list_plugins
    from sparkrun.core.plugin_compatibility import HostSnapshot, evaluate_plugin_compatibility, metadata_for_plugin
    from sparkrun.core.plugin_manifest import PluginMetadata, module_parts
    from sparkrun.core.plugin_validation import validate_candidate
    from pathlib import Path

    config = management_config(ctx)
    host = HostSnapshot.current()
    from sparkrun.core.plugin_lifecycle import plugin_details
    from sparkrun.core.plugin_policy import plugin_state
    from sparkrun.proxy.config import ProxyConfig

    selected = ProxyConfig(config.config_path.parent / "proxy.yaml").gateway
    rows = []
    feature_labels = []
    hidden = 0
    # The one disabled-plugin fact worth showing even when the plugin is hidden:
    # the configured gateway cannot start while its provider is off.
    disabled_gateway_providers = []
    for plugin in list_plugins(config):
        if not plugin.enabled:
            if selected and _declares_gateway(plugin, selected, plugin_details):
                disabled_gateway_providers.append(plugin.name)
            if not include_all:
                hidden += 1
                continue
        row = plugin.to_dict()
        row["managed"] = plugin.managed

        features_available = True
        try:
            details = plugin_details(plugin)
        except (PluginError, OSError, ValueError) as exc:
            details = {"declarations": {}}
            row["failure"] = str(exc)
            features_available = False
        row["features"] = sorted(details["declarations"].get("features", {}))
        feature_labels.append((", ".join(row["features"]) or "none") if features_available else "unavailable")
        state = plugin_state(
            plugin.plugin_id or plugin.name,
            config,
            source=plugin.source,
            legacy_feature=plugin.feature_flag if plugin.managed and not details["declarations"] else None,
        )
        row["state_source"] = state.source
        row["blocked_reason"] = state.reason
        row["selected_gateway"] = None
        if selected and "gateway." + selected in row["features"]:
            row["selected_gateway"] = selected
        row["compatibility"] = evaluate_plugin_compatibility(metadata_for_plugin(plugin), host).to_dict()
        if check and plugin.managed and plugin.path is not None:
            try:
                from sparkrun.core.plugin_store import PluginStore

                PluginStore(config).receipt(plugin.name, plugin.management["active_generation"])
                validate_candidate(
                    PluginMetadata.from_dict(plugin.management["metadata"]),
                    plugin.path / "payload" / Path(*module_parts(plugin.module)),
                    config.config_path,
                    default_managed=plugin.source == "default-managed",
                )
                row["load_check"] = "passed"
            except PluginError as exc:
                row["load_check"] = str(exc)
        rows.append(row)
    if output_json:
        print_json(rows)
        return
    hidden_note = "%d disabled plugin(s) hidden; use --all to include them." % hidden if hidden else None
    if not rows:
        click.echo("No enabled plugins found." if hidden else "No plugins found.")
        if hidden_note:
            click.echo(hidden_note)
        _echo_disabled_gateway_providers(disabled_gateway_providers, selected, heading=True)
        return
    header = ("NAME", "VERSION", "SOURCE", "STATE", "COMPATIBILITY", "FEATURES")
    table: list[tuple[str, ...]] = [header]
    for row, features in zip(rows, feature_labels, strict=True):
        state = "disabled" if not row["enabled"] else "enabled (load failed)" if row["failure"] else "enabled"
        table.append(
            (
                row["name"],
                row["version"] or "unknown",
                row["source"],
                state,
                row["compatibility"]["status"],
                features,
            )
        )
    widths = [max(len(cell) for cell in column) for column in zip(*table, strict=True)]
    lines = ["  ".join(cell.ljust(width) for cell, width in zip(cells, widths, strict=True)).rstrip() for cells in table]
    click.echo("\n".join(lines))
    click.echo("\nFEATURES: declared capability flags; none = no separate flags. Disabled plugins' capabilities are inactive.")
    if hidden_note:
        click.echo(hidden_note)
    # A disabled plugin's diagnostics restate that it is disabled; STATE says so.
    diagnostics = [
        row
        for row in rows
        if row["enabled"]
        and (
            row["failure"]
            or row["compatibility"]["reasons"]
            or row.get("load_check")
            or row.get("selected_gateway")
            or row.get("blocked_reason")
        )
    ]
    if diagnostics:
        click.echo("\nPlugin diagnostics:")
    _echo_disabled_gateway_providers(disabled_gateway_providers, selected, heading=not diagnostics)
    for row in diagnostics:
        click.echo("  %s:" % row["name"])
        if row.get("blocked_reason"):
            click.echo("    Unavailable: " + row["blocked_reason"])
        if row.get("selected_gateway"):
            click.echo("    Selected gateway: " + row["selected_gateway"])
        if row["failure"]:
            click.echo("    Load failure: %s" % row["failure"])
        if row["compatibility"]["reasons"]:
            click.echo("    Compatibility:")
            for reason in row["compatibility"]["reasons"]:
                click.echo("      - %s" % reason)
        if row.get("load_check"):
            click.echo("    Load check: %s" % row["load_check"])


def _echo_disabled_gateway_providers(names, gateway, *, heading: bool) -> None:
    """Report disabled plugins that provide the configured gateway (it cannot start)."""
    if not names:
        return
    if heading:
        click.echo("\nPlugin diagnostics:")
    for name in names:
        click.echo("  %s:" % name)
        click.echo("    Selected gateway: %s (plugin disabled)" % gateway)


def _declares_gateway(plugin, gateway: str, plugin_details) -> bool:
    """Whether *plugin* declares the ``gateway.<gateway>`` capability flag."""
    try:
        details = plugin_details(plugin)
    except (PluginError, OSError, ValueError):
        logger.debug("Could not read declarations of disabled plugin %s", plugin.name, exc_info=True)
        return False
    return "gateway." + gateway in details["declarations"].get("features", {})


@plugins.command("sync", short_help="Install missing default plugins.")
@click.option("--dry-run", is_flag=True, help="Preview defaults using local metadata only; no network or installation")
@json_option()
@click.pass_context
@plugin_errors
def plugins_sync(ctx, dry_run, output_json):
    """Install the application's trusted defaults unless disabled or removed.

    Existing providers, sources, and pins are preserved. Use plugins update to
    update installed plugins. This command is also available as setup plugins sync.
    """
    from sparkrun.core.plugin_defaults import reconcile_default_plugins

    rows = reconcile_default_plugins(management_config(ctx), dry_run=dry_run)
    if output_json:
        print_json(rows)
    else:
        for row in rows:
            _show_result(row)
        if not rows:
            click.echo("No default plugins declared by this application.")
    if any(row["status"] == "failed" for row in rows):
        ctx.exit(1)


@plugins.command("add", short_help="Install a plugin from a Git repository.")
@click.argument("source", metavar="SOURCE")
@click.option("--ref", metavar="REF", default="main", show_default=True, help="Git branch, tag, or full commit ID to install")
@click.option("--trust", is_flag=True, help="Trust the source and future compatible updates without prompting")
@click.option("--enable", is_flag=True, help="Enable plugin loading after installation; preserve its capability settings")
@click.option("--dry-run", is_flag=True, help="Fetch and check metadata without granting trust, executing plugin code, or installing")
@click.pass_context
@plugin_errors
def plugins_add(ctx, source, ref, trust, enable, dry_run):
    """Install an external plugin from SOURCE.

    SOURCE is an HTTPS/SSH Git repository URL or local Git repository path.
    The repository must provide plugin.toml and pyproject.toml. Dependencies
    must already be installed; this command installs the plugin's source files.

    The default ref is main. Branches track new commits on update; tags and full
    commit IDs remain pinned. Compatibility and loading are checked before
    activation. Dry runs may fetch from the network but do not execute plugins.

    Installation asks for source trust and enables core.external_plugins if
    needed. Existing plugin directories may need separate trust confirmation.
    Capability settings are preserved. --enable explicitly allows plugin loading
    and runs its enable hook; it does not enable every capability or select a gateway. Use --trust for noninteractive source consent.

    \b
    Examples:
      {app_command} plugins add https://github.com/sparksq/sparkrun-sparkroute-plugin
      {app_command} plugins add https://github.com/sparksq/sparkrun-coldsnap-plugin --enable
      {app_command} plugins add ./my-plugin --ref main --dry-run
    """
    manager = _manager(ctx)
    result = manager.add(source, ref=ref, trust=trust, enable=enable, dry_run=dry_run, consent=_trust, legacy_consent=_legacy_trust)
    _show_result(result)
    if not dry_run:
        from sparkrun.core.managed_plugins import managed_enabled

        config, records = manager.store.snapshot()
        record = records[result["name"]]
        if not managed_enabled(record, config):
            click.echo(
                "Installed but loading is disabled. Use sparkrun plugins enable %s after resolving any global loading override."
                % result["name"]
            )
        feature = record["metadata"].get("feature")
        from sparkrun.core.plugin_policy import LEGACY_PLUGIN_FLAGS

        if feature and feature not in LEGACY_PLUGIN_FLAGS.values() and not config.is_feature_enabled(feature):
            click.echo("Enable with: sparkrun setup features enable %s" % feature)


@plugins.command("update", short_help="Update one or all managed plugins.")
@click.argument("selector", metavar="[PLUGIN]", required=False)
@click.option("--all", "update_all", is_flag=True, help="Update every managed plugin using its saved branch or pin; omit PLUGIN")
@click.option("--ref", metavar="REF", default=None, help="Switch one plugin to a Git branch, tag, or full commit ID (default: saved ref)")
@click.option(
    "--trust", is_flag=True, help="Accept changed source/plugin declarations and future updates without prompting; single plugin only"
)
@click.option(
    "--dry-run", is_flag=True, help="Preview updates and compatibility without executing plugin code or changing installations; may fetch"
)
@click.pass_context
@plugin_errors
def plugins_update(ctx, selector, update_all, ref, trust, dry_run):
    """Update PLUGIN, or use --all to update every managed plugin.

    PLUGIN is a managed plugin name from `plugins list` or an unambiguous saved
    source URL. Choose either PLUGIN or --all; --ref and --trust require PLUGIN.

    The saved branch or pin is reused unless --ref selects a new one. Branches
    advance to their latest commit even if the version string is unchanged.
    Tags and commits stay pinned; a moved tag is rejected.

    Each replacement must pass compatibility and loading checks. Failed updates
    leave the active plugin intact, and --all continues with the other plugins.
    Previous source trust is reused; changed declarations require new consent.
    --all also installs missing trusted application defaults, honoring opt-outs.

    Core updates also update managed plugins by default. To audit only, use:

    \b
      {app_command} update --no-update-plugins

    \b
    Examples:
      {app_command} plugins update sparkroute
      {app_command} plugins update --all --dry-run
      {app_command} plugins update sparkroute --ref refs/tags/v0.2.1
    """
    if bool(selector) == update_all or (update_all and (ref or trust)):
        raise click.UsageError("Choose one plugin or --all; --ref/--trust require a single plugin")
    manager = _manager(ctx)
    failures = []
    installed = set()
    if update_all and not dry_run:
        with manager.store.mutation():
            pass
    if update_all:
        from sparkrun.core.plugin_defaults import reconcile_default_plugins

        for row in reconcile_default_plugins(manager.config, dry_run=dry_run, manager=manager):
            if row["status"] != "present":
                _show_result(row)
            if row["status"] == "installed":
                installed.add(row["name"])
            if row["status"] == "failed":
                failures.append(row["name"])
    selectors = [selector] if selector else sorted(manager.store.snapshot()[1])
    for name in selectors:
        if name in installed:
            continue
        try:
            _show_result(manager.update(name, ref=ref, trust=trust, dry_run=dry_run, consent=_trust))
        except PluginError as exc:
            failures.append("%s: %s" % (name, exc))
            click.echo("Warning: %s" % failures[-1], err=True)
    if failures:
        raise click.ClickException("%d plugin update(s) failed" % len(failures))


@plugins.command("remove", short_help="Remove a managed plugin and its history.")
@click.argument("selector", metavar="PLUGIN")
@click.option("--yes", is_flag=True, help="Confirm removal of the plugin and all retained generations without prompting")
@click.option("--dry-run", is_flag=True, help="Show the selected plugin without deleting files or changing configuration")
@click.pass_context
@plugin_errors
def plugins_remove(ctx, selector, yes, dry_run):
    """Remove PLUGIN and all of its retained installation generations.

    PLUGIN is a managed plugin name from `plugins list` or an unambiguous saved
    source URL. Removes manager-owned plugin files and the installation record,
    preserving feature settings, runtime data, and recipe-registry settings.

    Removal works even if the plugin cannot load. It requires confirmation;
    use --yes for noninteractive removal. Restart long-running plugin consumers
    afterward.

    \b
    Examples:
      {app_command} plugins remove sparkroute --dry-run
      {app_command} plugins remove sparkroute
      {app_command} plugins remove sparkroute --yes
    """
    manager = _manager(ctx)
    if not dry_run and not yes:
        if not sys.stdin.isatty() or not click.confirm("Remove managed plugin %s and its retained generations?" % selector, default=False):
            raise click.ClickException("Removal cancelled; use --yes to confirm noninteractively")
    _show_result(manager.remove(selector, dry_run=dry_run))


@plugins.command("rollback", short_help="Restore the previous plugin generation.")
@click.argument("selector", metavar="PLUGIN")
@click.option(
    "--dry-run", is_flag=True, help="Check the previous generation's integrity and compatibility without loading or activating it"
)
@click.pass_context
@plugin_errors
def plugins_rollback(ctx, selector, dry_run):
    """Restore PLUGIN's most recently active retained generation.

    PLUGIN is a managed plugin name from `plugins list` or an unambiguous saved
    source URL. The previous generation must pass integrity, compatibility,
    and loading checks before it replaces the current one. No fetch is needed.

    Rollback requires retained history. The plugins.retain_generations setting
    defaults to 2 (current and previous); setting it to 1 disables rollback.
    The current generation becomes the previous one after rollback.

    The saved source/ref stays unchanged, so a later update may advance the
    plugin again. Use `plugins update PLUGIN --ref REF` to pin an older revision.
    Restart long-running plugin consumers after rollback.

    \b
    Examples:
      {app_command} plugins rollback sparkroute --dry-run
      {app_command} plugins rollback sparkroute
    """
    _show_result(_manager(ctx).rollback(selector, dry_run=dry_run))


@plugins.command("_reconcile-update", hidden=True)
@click.option("--protocol", type=int, required=True, help="Reconciliation protocol version; must be 1")
@click.option("--no-update-plugins", is_flag=True, help="Audit compatibility without fetching or updating managed plugins")
@click.pass_context
@plugin_errors
def reconcile_update(ctx, protocol, no_update_plugins):
    """Private fresh-host reconciliation protocol; emits JSON only."""
    if protocol != 1:
        raise click.ClickException("Unsupported plugin update protocol")
    from sparkrun.core.plugin_update import reconcile_plugins

    print_json(reconcile_plugins(management_config(ctx), update=not no_update_plugins))
