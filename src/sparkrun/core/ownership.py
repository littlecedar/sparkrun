"""Deployment ownership is separate from portable recipe identity."""

from __future__ import annotations

import re

from sparkrun.core.application_profile import get_application_profile

OWNER_LABEL = "sparkrun.distribution"
CONTROLLER_LABEL = "sparkrun.controller"
_JOB_NAME = re.compile(r"^(?P<namespace>[a-z][a-z0-9-]{0,47})_[0-9a-f]{16}_[0-9a-f]{12}(?:_.+)?$")
_LEGACY_NAME = re.compile(r"^sparkrun_[a-zA-Z0-9_]+$")


def owns_resource(name: str, labels: dict | None = None) -> bool:
    """Labels win; absent labels establish ownership only for legacy Sparkrun."""
    owner = (labels or {}).get(OWNER_LABEL)
    if owner is not None:
        return owner == get_application_profile().id
    return get_application_profile().id == "sparkrun" and bool(_LEGACY_NAME.fullmatch(name))


def owns_metadata(data: dict) -> bool:
    owner = data.get("distribution")
    if owner is not None:
        return owner == get_application_profile().id
    return get_application_profile().id == "sparkrun" and (
        "sparkrun_version" in data or bool(_LEGACY_NAME.fullmatch(str(data.get("cluster_id", ""))))
    )


def resource_namespace_of(name: str) -> str | None:
    """The distribution namespace a workload id carries, or ``None`` if it is not one."""
    match = _JOB_NAME.fullmatch(name)
    return match["namespace"] if match else None


def assert_resource_namespace(name: str) -> None:
    match = _JOB_NAME.fullmatch(name)
    if match and match["namespace"] != get_application_profile().resource_namespace:
        raise ValueError("Resource %r belongs to another distribution; active owner is %r" % (name, get_application_profile().id))


def docker_owner_guard(name: str, *, allow_missing: bool = False) -> str:
    """Shell condition checking actual executor-visible ownership before mutation."""
    from sparkrun.utils.shell import quote

    assert_resource_namespace(name)
    legacy = get_application_profile().id == "sparkrun" and bool(_LEGACY_NAME.fullmatch(name))
    inspect = "docker inspect --format '{{ index .Config.Labels \"%s\" }}' %s 2>/dev/null" % (OWNER_LABEL, quote(name))
    fallback = ' || [ -z "$_owner" ] || [ "$_owner" = "<no value>" ]' if legacy else ""
    guard = '(_owner=$(%s) && { [ "$_owner" = %s ]%s; })' % (inspect, quote(get_application_profile().id), fallback)
    if allow_missing:
        return "( %s || ! docker inspect %s >/dev/null 2>&1 )" % (guard, quote(name))
    return guard
