"""Public retention DTOs. Mapping access is available to report renderers."""

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any


@dataclass(frozen=True)
class ReportMapping(Mapping):
    def __getitem__(self, key):
        if key not in {field.name for field in fields(self)}:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self):
        return (field.name for field in fields(self))

    def __len__(self):
        return len(fields(self))


@dataclass(frozen=True)
class CacheEntry(ReportMapping):
    repo: str
    revision: str
    snapshot: str
    tracked: bool
    present: bool
    last_used: float | None
    logical_bytes: int
    reclaimable_bytes: int
    pinned: bool
    claims: dict[str, Any]
    eligible: bool
    reason: str
    reason_code: str
    usage: dict[str, Any] | None = None
    action: str | None = None


@dataclass(frozen=True)
class CacheLocation(ReportMapping):
    """One model cache and the hosts that see it.

    *kind* is ``network``, ``local`` or ``unknown`` (the probe failed).
    *source* names the storage: the mount source for a network share, else the
    host. *paths* maps each host to its own path to the cache, which may differ
    per host for one share. *verified* is true when sharing was confirmed by a
    nonce round-trip (always true for a single host).
    """

    key: str
    kind: str
    source: str
    hosts: tuple[str, ...]
    paths: dict[str, str] = field(default_factory=dict)
    verified: bool = False


@dataclass(frozen=True)
class CacheReport(ReportMapping):
    host: str
    cache_root: str
    dry_run: bool
    entries: tuple[CacheEntry, ...]
    free_bytes: int
    reclaimed_bytes: int
    planned_bytes: int
    target_met: bool
    budget_scope: str
    reclaimed_bytes_kind: str
    free_bytes_after_kind: str
    free_bytes_after: int | None = None
    #: The cache this report covers and every host that sees it; the operation
    #: ran once, on ``host``. Set by every ``api.model_cache`` operation.
    location: CacheLocation | None = None

    @classmethod
    def from_dict(cls, value):
        location = value.get("location")
        if isinstance(location, dict):
            location = CacheLocation(**{**location, "hosts": tuple(location["hosts"])})
        return cls(**{**value, "entries": tuple(CacheEntry(**entry) for entry in value["entries"]), "location": location})


@dataclass(frozen=True)
class ClaimReconciliation(ReportMapping):
    """One outstanding workload claim and what ``reconcile_claims`` decided.

    *status* is ``orphaned`` (the workload was confirmed absent from every
    host in the claim), ``running``, ``foreign`` (another distribution's
    workload, which status cannot see), ``unverified`` (a claim host or the
    workload's executor was not observed, or the record is malformed), ``preparing`` (a pending operation on this control machine
    names the workload) or ``recent`` (younger than the minimum age, so it may
    belong to a launch still preparing elsewhere). Only an orphaned claim gets
    an *action*: ``would_release``, ``released``, or ``failed`` with *reason*.
    """

    repo: str
    revision: str
    claim: str
    workload: str | None
    hosts: tuple[str, ...]
    created: float | None
    status: str
    reason: str
    action: str | None = None
    location: CacheLocation | None = None
