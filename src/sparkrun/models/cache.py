"""Inventory/prune transport and optional pressure policy; no origin requests."""

import json
import logging
import math
from pathlib import Path

from sparkrun.models.artifacts import ModelArtifactError
from sparkrun.models.host_io import ModelHostIO
from sparkrun.utils.data import parse_byte_size
from sparkrun.utils.shell import quote

logger = logging.getLogger(__name__)


def age_seconds(days):
    if type(days) not in (int, float) or not math.isfinite(days) or days < 0:
        raise ValueError("minimum retention age must be finite and nonnegative")
    return days * 86400


def host_cache(io, host, root, **options):
    # Read source rather than importing the helper: controllers may be Windows,
    # while fcntl and disk mutations belong exclusively to the Linux cache host.
    source = Path(__file__).with_name("_cache_agent.py").read_text()
    payload = json.dumps({"root": root, **options})
    contract = Path(__file__).with_name("_cache_contract.py").read_text()
    bootstrap = (
        "import sys, types\nm=types.ModuleType('_sparkrun_cache_contract')\nsys.modules[m.__name__]=m\nexec("
        + repr(contract)
        + ", m.__dict__)\n"
    )
    program = bootstrap + source + "\nprint(json.dumps(run(**json.loads(" + repr(payload) + "))))\n"
    result = io.execute(host, "python3 -c " + str(quote(program)), timeout=300)
    if result.returncode:
        raise ModelArtifactError("cache operation on %s failed: %s" % (host or "control", result.stderr.decode(errors="replace")[-1500:]))
    try:
        report = json.loads(result.stdout)
    except ValueError as exc:
        raise ModelArtifactError("invalid host cache report") from exc
    report["host"] = host or "control"
    if options.get("action") in {"pin", "reconcile"} and not any(row.get("action") == options["action"] for row in report["entries"]):
        raise ModelArtifactError("cache mutation was not applied: " + "; ".join(r["reason"] for r in report["entries"]))
    return report


def _mutates(options) -> bool:
    return options.get("action") in {"pin", "reconcile"} or (
        options.get("action") in {"prune", "delete"} and not options.get("dry_run", True)
    )


def resolve_cache_locations(*, cache_dir, hosts=("localhost",), ssh_kwargs=None, session=None, verify=False):
    """The cache locations *hosts* see; nonce-verified when *verify* (see ``models.cache_locations``)."""
    from sparkrun.models.cache_locations import resolve_locations, verify_locations

    with ModelHostIO(ssh_kwargs, session=session) as io:
        locations = resolve_locations(io, hosts if hosts is not None else ("localhost",), cache_dir)
        return verify_locations(io, locations) if verify else locations


def cache_operation(*, cache_dir, hosts=("localhost",), ssh_kwargs=None, session=None, locations=None, **options):
    """Run one cache-agent action once per cache location, not once per host.

    Hosts that see the same cache (one NFS export, or one machine under two
    names) form a location; the action runs on its first host and the report
    carries the location. A mutating action first confirms each location by a
    nonce round-trip; a preview stays read-only and relies on the probed
    identity alone. Pass *locations* to reuse a grouping across calls (delete
    gates its apply on a preview of exactly the caches it will change).

    Only a read-only action moves on to another host of the same cache when one
    fails: a mutating action that failed may still have changed the cache, so
    its error is reported rather than masked by a second run.
    """
    from dataclasses import replace

    from sparkrun.models.cache_reports import CacheReport

    mutating = _mutates(options)
    if locations is None:
        locations = resolve_cache_locations(cache_dir=cache_dir, hosts=hosts, ssh_kwargs=ssh_kwargs, session=session, verify=mutating)
    reports = []
    with ModelHostIO(ssh_kwargs, session=session) as io:
        for location in locations:
            for index, host in enumerate(location.hosts):
                try:
                    report = host_cache(io, host, location.paths[host], **options)
                except ModelArtifactError:
                    if mutating or index == len(location.hosts) - 1:
                        raise
                    logger.info("Cache inventory on %s failed; asking another host of the same cache", host)
                    continue
                reports.append(replace(CacheReport.from_dict(report), location=location))
                break
    return tuple(reports)


def auto_prune(io, locations, *, config, exclude_repos):
    get = getattr(config, "get", None)
    policy = get("model_cache_prune", None) if callable(get) else None
    if policy is None:
        return
    if not isinstance(policy, dict) or type(policy.get("enabled", False)) is not bool:
        raise ModelArtifactError("model_cache_prune must be a mapping with a boolean enabled setting")
    if not policy.get("enabled", False):
        return
    unknown = set(policy) - {"enabled", "min_age_days", "max_bytes", "min_free_bytes"}
    if unknown:
        raise ModelArtifactError("unknown model_cache_prune settings: " + ", ".join(sorted(unknown)))
    if policy.get("max_bytes") is None and policy.get("min_free_bytes") is None:
        raise ModelArtifactError("automatic cache pruning requires max_bytes or min_free_bytes")
    sizes = {}
    for key in ("max_bytes", "min_free_bytes"):
        if policy.get(key) is not None:
            try:
                sizes[key] = parse_byte_size(policy[key])
            except ValueError as exc:
                raise ModelArtifactError("model_cache_prune.%s: %s" % (key, exc)) from exc
    if sizes.get("max_bytes") == 0:
        # A zero budget is met only by an empty cache, so every launch would
        # evict every eligible revision: far more likely a typo than a policy.
        raise ModelArtifactError("model_cache_prune.max_bytes must be positive; to stop pruning set enabled: false")
    from sparkrun.feedback import get_feedback

    for host, root in locations:
        report = host_cache(
            io,
            host,
            root,
            action="prune",
            dry_run=False,
            min_age_seconds=age_seconds(policy.get("min_age_days", 7)),
            max_bytes=sizes.get("max_bytes"),
            min_free_bytes=sizes.get("min_free_bytes"),
            exclude_repos=exclude_repos,
        )
        if report["reclaimed_bytes"]:
            logger.info("Model cache on %s reclaimed %d bytes", host or "control", report["reclaimed_bytes"])
        if not report["target_met"]:
            logger.warning(
                "Model cache retention policy could not meet its space target on %s; protected data was retained", host or "control"
            )
        get_feedback("model-cache").progress(
            "Model cache cleanup on %s: %d bytes reclaimed" % (host or "control", report["reclaimed_bytes"]),
            host=host or "control",
            stage="prune",
            status="succeeded",
            metrics={"reclaimed_bytes": report["reclaimed_bytes"]},
        )
