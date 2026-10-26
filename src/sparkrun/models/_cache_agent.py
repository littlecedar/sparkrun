"""Standard-library host helper for explicit inventory, conservative pruning and deletion.

Shipped over the existing host session; no Sparkrun or Hub install is required.
Pruning never deletes lock files or unknown snapshots; with an evicted
snapshot it removes the blobs only that snapshot linked to. An explicit delete
removes blobs, likewise only those nothing else in the cache references.
Nothing outside HF_HOME is ever removed.
"""

from dataclasses import dataclass
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import time

try:
    from ._cache_contract import STATE_DIRECTORY as STATE, CACHE_REASONS, USAGE_SCHEMA_VERSION, same_origin, validate_usage
except ImportError:  # Shipped by the controller without installing Sparkrun.
    from importlib import import_module

    _contract = import_module("_sparkrun_cache_contract")
    STATE, CACHE_REASONS, validate_usage = _contract.STATE_DIRECTORY, _contract.CACHE_REASONS, _contract.validate_usage
    USAGE_SCHEMA_VERSION, same_origin = _contract.USAGE_SCHEMA_VERSION, _contract.same_origin


def traversal_error(error):
    raise error


def safe_directory(path):
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError("symlinked cache directory: " + str(parent))
        if parent.exists() and not parent.is_dir():
            raise ValueError("not a cache directory: " + str(parent))


def read_record(path, slug, revision):
    if path.is_symlink():
        raise ValueError("symlinked usage record")
    record = json.loads(path.read_text())
    return validate_usage(record, slug, revision)


def retention_reason(row, record, *, unknown, now, min_age_seconds, exclude_repos):
    """Pure retention policy; explicit deletion deliberately uses different rules."""
    if not row["present"]:
        return "absent"
    if record["pinned"]:
        return "pinned"
    if record["claims"]:
        return "workload_claim"
    if record["repo"] in exclude_repos:
        return "requested_model"
    if unknown:
        return "untracked_files"
    if now - record["last_used"] < min_age_seconds:
        return "retention_age"
    return "eligible"


def observe(root, slug, revision, *, now, min_age_seconds, exclude_repos, context=None):
    snapshot = root / "hub" / slug / "snapshots" / revision
    path = root / STATE / "usage" / slug.removeprefix("models--") / (revision + ".json")
    row = {
        "repo": slug.removeprefix("models--").replace("--", "/", 1),
        "revision": revision,
        "snapshot": str(snapshot),
        "tracked": False,
        "present": snapshot.exists(),
        "last_used": None,
        "logical_bytes": 0,
        "reclaimable_bytes": 0,
        "pinned": False,
        "claims": {},
        "eligible": False,
        "reason": CACHE_REASONS["untracked"],
        "reason_code": "untracked",
    }
    try:
        safe_directory(snapshot)
        safe_directory(path.parent)
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            return row
        if not path.exists():
            return row
        record = read_record(path, slug.removeprefix("models--"), revision)
        row.update({k: record[k] for k in ("repo", "last_used", "pinned", "claims")})
        row.update(tracked=True, usage=record)
        unknown = False
        files = set(record["files"])
        if snapshot.exists():
            evidence = scan_removal(root, slug, [revision], context=context)
            for item, info in evidence.files.items():
                if inside(item, snapshot):
                    relative = str(item.relative_to(snapshot))
                    if relative not in files or not (stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)):
                        unknown = True
            row["logical_bytes"] = evidence.logical.get(revision, 0)
            row["reclaimable_bytes"] = plan_removal(evidence).planned_bytes
        reason_code = retention_reason(row, record, unknown=unknown, now=now, min_age_seconds=min_age_seconds, exclude_repos=exclude_repos)
        row.update(reason=CACHE_REASONS[reason_code], reason_code=reason_code, eligible=reason_code == "eligible")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        row.update(reason=CACHE_REASONS["unverifiable"] + ": " + str(exc), reason_code="unverifiable", eligible=False)
    return row


def inventory(root, **policy):
    safe_directory(root)
    safe_directory(root / "hub")
    safe_directory(root / STATE / "usage")
    identities = set()
    for path in (root / "hub").glob("models--*/snapshots/*"):
        identities.add((path.parent.parent.name, path.name))
    for path in (root / STATE / "usage").glob("*/*.json"):
        identities.add(("models--" + path.parent.name, path.stem))
    return [observe(root, slug, revision, **policy) for slug, revision in sorted(identities)]


def write_record(path, record):
    import tempfile

    fd, temp = tempfile.mkstemp(prefix=".usage-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(record, stream, sort_keys=True)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def free_space(root):
    probe = root
    while not probe.exists():
        probe = probe.parent
    return shutil.disk_usage(probe).free


def cache_report(root, rows, *, dry_run, free):
    """Public report vocabulary; no filesystem reads or mutations."""
    return {
        "cache_root": str(root),
        "dry_run": dry_run,
        "entries": rows,
        "free_bytes": free,
        "reclaimed_bytes": 0,
        "planned_bytes": 0,
        "target_met": True,
        "budget_scope": "private_managed_snapshots",
        "reclaimed_bytes_kind": "allocated_bytes_estimate",
        "free_bytes_after_kind": "projected" if dry_run else "measured",
    }


def update_usage(row, path, *, action, pinned, claim, endpoint, now, dry_run):
    """Update a freshly observed record while its publication lock is held."""
    record = row["usage"]
    if endpoint is not None and not same_origin(record["endpoint"], endpoint):
        raise ValueError("cache usage endpoint mismatch")
    if action == "pin":
        record["pinned"] = pinned
        row["pinned"] = pinned
    else:
        if claim not in record["claims"]:
            raise ValueError("outstanding claim not found")
        del record["claims"][claim]
        record["last_used"] = record["last_released"] = now
    if not dry_run:
        write_record(path, record)
    row["action"] = action


def apply_prune(root, slug, row, plan, *, now, context):
    """Apply a plan made under the current publication lock and finish bookkeeping."""
    if not shutil.rmtree.avoids_symlink_attacks:
        raise ValueError("safe snapshot deletion is unavailable on this host")
    shutil.rmtree(row["snapshot"])
    retained = remove_exclusive_blobs(root, slug, plan, context=context)
    record = row["usage"]
    record["last_evicted"] = now
    path = root / STATE / "usage" / slug.removeprefix("models--") / (row["revision"] + ".json")
    write_record(path, record)
    context.refresh()
    return RemovalOutcome(plan, frozenset(retained))


def run(
    root,
    *,
    action="inventory",
    dry_run=True,
    min_age_seconds=7 * 86400,
    max_bytes=None,
    min_free_bytes=None,
    exclude_repos=(),
    repo=None,
    revision=None,
    pinned=True,
    claim=None,
    confirm_inactive=False,
    endpoint=None,
    force=False,
    allowed=None,
    now=None,
    files=None,
    adopt_last_used=None,
):
    root = Path(root)
    if not root.is_absolute() or root == Path("/"):
        raise ValueError("an absolute HF cache home is required")
    for value in (min_age_seconds, max_bytes, min_free_bytes):
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
            raise ValueError("cache limits must be finite and nonnegative")
    if action not in {"inventory", "prune", "pin", "reconcile", "delete", "scan", "adopt"}:
        raise ValueError("unknown cache action")
    if any(type(flag) is not bool for flag in (dry_run, pinned, confirm_inactive, force)):
        raise ValueError("cache mutation flags must be boolean")
    now = time.time() if now is None else now
    context = ObservationContext()
    if action == "delete":
        return delete(root, repo=repo, revision=revision, force=force, dry_run=dry_run, now=now, allowed=allowed, context=context)
    if action == "scan":
        return {"snapshots": scan(root, repo=repo, revision=revision)}
    if action == "adopt":
        return adopt(root, repo=repo, revision=revision, files=files, endpoint=endpoint, last_used=adopt_last_used, pinned=pinned, now=now)
    return retention(
        root,
        action=action,
        dry_run=dry_run,
        now=now,
        min_age_seconds=min_age_seconds,
        max_bytes=max_bytes,
        min_free_bytes=min_free_bytes,
        exclude_repos=exclude_repos,
        repo=repo,
        revision=revision,
        pinned=pinned,
        claim=claim,
        confirm_inactive=confirm_inactive,
        endpoint=endpoint,
        context=context,
    )


def retention(
    root,
    *,
    action,
    dry_run,
    now,
    min_age_seconds,
    max_bytes,
    min_free_bytes,
    exclude_repos,
    repo,
    revision,
    pinned,
    claim,
    confirm_inactive,
    endpoint,
    context,
):
    """Coordinate retention/usage changes; observations are refreshed under each publication lock."""
    policy = dict(now=now, min_age_seconds=min_age_seconds, exclude_repos=exclude_repos, context=context)
    rows = inventory(root, **policy)
    free = free_space(root)
    result = cache_report(root, rows, dry_run=dry_run, free=free)
    if action == "inventory":
        return result
    if action in {"pin", "reconcile"}:
        revision = resolve_named_revision(root, repo, revision)
        rows = [r for r in rows if r["repo"] == repo and r["revision"] == revision and r["tracked"]]
        if len(rows) != 1:
            raise ValueError("tracked model revision not found")
        if action == "reconcile" and (not confirm_inactive or not claim):
            raise ValueError("reconciliation requires a claim and explicit confirmation that its workloads stopped")
    total = sum(r["reclaimable_bytes"] for r in result["entries"])

    def satisfied():
        return (max_bytes is None or total <= max_bytes) and (min_free_bytes is None or free >= min_free_bytes)

    for row in sorted(rows, key=lambda r: (r["last_used"] or 0, r["repo"], r["revision"])):
        if action == "prune" and (not row["eligible"] or ((max_bytes is not None or min_free_bytes is not None) and satisfied())):
            continue
        slug = Path(row["snapshot"]).parent.parent.name
        lock = root / STATE / (slug.removeprefix("models--") + ".lock")
        safe_directory(lock.parent)
        # Preparation creates the lock before recording usage. Never create it
        # in a dry run, nor unlink it (which would split the locking domain).
        try:
            fd = os.open(lock, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except OSError:
            row.update(eligible=False, reason=CACHE_REASONS["publication_lock_unavailable"], reason_code="publication_lock_unavailable")
            continue
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                row.update(eligible=False, reason=CACHE_REASONS["invalid_publication_lock"], reason_code="invalid_publication_lock")
                continue
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                row.update(eligible=False, reason=CACHE_REASONS["cache_busy"], reason_code="cache_busy")
                continue
            observed_last_used = row["last_used"]
            context.refresh()
            fresh = observe(root, slug, row["revision"], **policy)
            row.update(fresh)
            if not fresh["tracked"]:
                continue
            path = root / STATE / "usage" / slug.removeprefix("models--") / (row["revision"] + ".json")
            if action != "prune":
                update_usage(row, path, action=action, pinned=pinned, claim=claim, endpoint=endpoint, now=now, dry_run=dry_run)
                continue
            if not fresh["eligible"]:
                continue
            if row["last_used"] != observed_last_used:
                row.update(eligible=False, reason=CACHE_REASONS["usage_changed_during_cleanup"], reason_code="usage_changed_during_cleanup")
                continue
            if hf_download_active(root / "hub", slug):
                # A download may be about to link one of the blobs this would take.
                row.update(eligible=False, reason=CACHE_REASONS["download_in_progress"], reason_code="download_in_progress")
                continue
            try:
                scan = scan_removal(root, slug, [row["revision"]], fresh=True, context=context)
            except OSError:
                scan = scan_removal(root, slug, [row["revision"]], blobs=())
            plan = plan_removal(scan)
            amount = row["reclaimable_bytes"] = plan.planned_bytes
            if amount == 0 and (max_bytes is not None or min_free_bytes is not None):
                # Every byte is shared with another revision (hard-linked reuse):
                # evicting it would report a removal that frees nothing.
                continue
            result["planned_bytes"] += amount
            row["action"] = "would_remove" if dry_run else "removed"
            if not dry_run:
                outcome = apply_prune(root, slug, row, plan, now=now, context=context)
                result["reclaimed_bytes"] += outcome.reclaimed_bytes
                if max_bytes is not None:
                    # Evicting one revision can make another's blobs its own:
                    # recount rather than subtract a figure taken before.
                    context.refresh()
                    total = sum(r["reclaimable_bytes"] for r in inventory(root, **policy))
                    free = shutil.disk_usage(root).free
                    continue
            total -= amount
            free = free + amount if dry_run else shutil.disk_usage(root).free
        finally:
            os.close(fd)
    result["target_met"] = satisfied()
    result["free_bytes_after"] = free
    return result


REPO_PATTERN = r"[A-Za-z0-9_][A-Za-z0-9_.-]*(/[A-Za-z0-9_][A-Za-z0-9_.-]*)?"
# Snapshot, ref and usage-record names: never ".", ".." or hidden entries.
NAME_PATTERN = r"[A-Za-z0-9_][A-Za-z0-9_.-]*"


def inside(path, parent):
    try:
        Path(path).relative_to(parent)
        return True
    except ValueError:
        return False


def tree_entries(top):
    """Every non-directory entry under *top*, never following symlinks."""
    for directory, dirs, names in os.walk(top, followlinks=False, onerror=traversal_error):
        for child in names + [d for d in dirs if (Path(directory) / d).is_symlink()]:
            yield Path(directory) / child


def read_ref(path):
    """A ref file's commit, or None for anything but a small regular file (never block on a FIFO)."""
    try:
        info = path.lstat()
    except OSError:
        return None
    if not stat.S_ISREG(info.st_mode) or info.st_size > 4096:
        return None
    value = path.read_text().strip()
    return value if re.fullmatch(NAME_PATTERN, value) else None


def resolve_named_revision(root, repo, revision):
    """A full commit as given, or the commit this cache's ``refs/<revision>`` names (e.g. ``main``)."""
    if not isinstance(repo, str) or not re.fullmatch(REPO_PATTERN, repo):
        raise ValueError("invalid repository")
    if not isinstance(revision, str) or not re.fullmatch(NAME_PATTERN, revision):
        raise ValueError("invalid revision")
    if re.fullmatch(r"[0-9a-f]{40}", revision):
        return revision
    resolved = read_ref(root / "hub" / ("models--" + repo.replace("/", "--")) / "refs" / revision)
    if resolved is None or not re.fullmatch(r"[0-9a-f]{40}", resolved):
        raise ValueError("%s has no cached ref %r here; name a full commit" % (repo, revision))
    return resolved


def blob_link(path, repo_blobs):
    """The per-model blob a snapshot entry links to (``../../blobs/<sha>``), or None."""
    if not path.is_symlink():
        return None
    hop = Path(os.path.normpath(path.parent / os.readlink(path)))
    return hop.name if hop.parent == repo_blobs else None


# huggingface_hub's cache-wide shared blob store (utils/_shared_blobs.py), layout
# version 1: per-repo blobs/<etag> symlinks into hub/blobs/<xx>/<hash>, a
# <hash>.refs manifest listing those symlinks, and a <hash>.lock that publication,
# linking and collection all take. Following the same protocol keeps sparkrun and
# huggingface_hub from racing; an unknown layout is never touched.
SHARED_MARKER = ".huggingface-shared-blobs"
SHARED_LAYOUT = "1\n"
SHARED_HASH = r"[0-9a-f]{64}"
SHARED_REPO_DIR = r"(?:models|datasets|spaces)--.+"
SHARED_LOCK_TIMEOUT = 10


def shared_store(hub):
    """hub/blobs when it is a shared store this helper understands, else None."""
    store = hub / "blobs"
    try:
        if stat.S_ISDIR(store.lstat().st_mode) and stat.S_ISREG((store / SHARED_MARKER).lstat().st_mode):
            return store if (store / SHARED_MARKER).read_text() == SHARED_LAYOUT else None
    except OSError:
        pass
    return None


def store_target(path, hub, store):
    """The store payload a per-repo ``<repo>/blobs/<etag>`` symlink names, or None."""
    try:
        relative = Path(os.path.abspath(path)).relative_to(os.path.abspath(hub))
    except ValueError:
        return None
    if len(relative.parts) != 3 or not re.fullmatch(SHARED_REPO_DIR, relative.parts[0]) or relative.parts[1] != "blobs":
        return None
    if not stat.S_ISLNK(path.lstat().st_mode):
        return None
    target = Path(os.path.abspath(path.parent / os.readlink(path)))
    try:
        inner = target.relative_to(os.path.abspath(store))
    except ValueError:
        return None
    if len(inner.parts) != 2 or not re.fullmatch(SHARED_HASH, inner.parts[1]) or inner.parts[0] != inner.parts[1][:2]:
        return None
    return store / inner.parts[0] / inner.parts[1]


def manifest_references(target, hub, store):
    """Live symlinks a payload's manifest lists; None (keep the payload) when it cannot be read."""
    manifest = target.with_name(target.name + ".refs")
    try:
        if not stat.S_ISREG(manifest.lstat().st_mode):
            return None
        lines = manifest.read_text().splitlines()
    except (OSError, UnicodeError):
        return None
    references = set()
    for line in lines:
        relative = Path(line)
        if relative.is_absolute() or ".." in relative.parts:
            continue
        candidate = hub / relative
        try:
            if store_target(candidate, hub, store) == target:
                references.add(candidate)
        except FileNotFoundError:
            continue
        except OSError:
            return None
    return references


class ObservationContext:
    """Reference observations owned by one invocation, refreshed at lock boundaries."""

    def __init__(self):
        self.references = {}

    def refresh(self):
        self.references.clear()


def repo_references(root, slug, *, fresh=False, context=None):
    """Per revision, the ``(device, inode)`` of every file its snapshot entries resolve to.

    Identity, not spelling: a sibling may link a blob relatively, absolutely,
    through a symlinked cache path, or hold it as a hard link, and every one
    of those must protect the blob. None when a snapshot is not a plain
    directory (it could reference anything) or cannot be read.
    """
    key = (str(root), slug)
    if context is not None and not fresh and key in context.references:
        return context.references[key]
    snapshots_dir = root / "hub" / slug / "snapshots"
    references = {}
    try:
        if snapshots_dir.is_dir():
            for snapshot in snapshots_dir.iterdir():
                if snapshot.is_symlink() or not snapshot.is_dir():
                    references = None
                    break
                found = set()
                for path in tree_entries(snapshot):
                    try:
                        info = path.stat()
                    except FileNotFoundError:
                        continue  # dangling: references nothing
                    if stat.S_ISREG(info.st_mode):
                        found.add((info.st_dev, info.st_ino))
                references[snapshot.name] = found
    except OSError:
        references = None
    if context is not None:
        context.references[key] = references
    return references


def exclusive_blob_paths(root, slug, revisions, *, fresh=False, context=None):
    """Blobs used only by selected revisions, by file identity rather than link spelling.

    Shared by prune and explicit revision deletion. An unreadable sibling or
    an unwritable blob directory makes every blob ineligible for collection.
    """
    repo_blobs = root / "hub" / slug / "blobs"
    references = repo_references(root, slug, fresh=fresh, context=context)
    if references is None or not repo_blobs.is_dir():
        return []
    if repo_blobs.is_symlink() or not os.access(repo_blobs, os.W_OK | os.X_OK):
        return []
    mine = set().union(*(ids for name, ids in references.items() if name in revisions))
    others = set().union(*(ids for name, ids in references.items() if name not in revisions))
    paths = []
    for path in sorted(repo_blobs.iterdir()):
        if path.name.endswith(".incomplete"):
            continue
        try:
            info = path.stat()
        except OSError:
            continue
        identity = (info.st_dev, info.st_ino)
        if identity in mine and identity not in others:
            paths.append(path)
    return paths


@dataclass(frozen=True)
class RemovalScan:
    """Read-only host evidence; an unknown payload manifest remains None."""

    snapshots: tuple
    linked: tuple
    blobs: tuple
    targets: dict
    store: object
    files: dict
    owners: dict
    logical: dict
    payload_references: dict


@dataclass(frozen=True)
class AllocationCredit:
    amount: int
    paths: frozenset
    owner: str


@dataclass(frozen=True)
class RemovalPlan:
    """Host-local estimates, never authority to replay deletions after a lock boundary."""

    scan: RemovalScan
    credits: tuple

    @property
    def planned_bytes(self):
        return sum(credit.amount for credit in self.credits)

    def row_bytes(self, revision):
        return self.scan.logical.get(revision, 0), sum(c.amount for c in self.credits if c.owner == revision)

    def reclaimed_bytes(self, retained):
        # If any name needed to free an inode survives, none of its allocation
        # was freed. This also handles snapshot/blob/shared-payload hard links.
        return sum(c.amount for c in self.credits if not c.paths.intersection(retained))


@dataclass(frozen=True)
class RemovalOutcome:
    """An applied plan may retain optional blobs or payloads without losing its bookkeeping."""

    plan: RemovalPlan
    retained: frozenset

    @property
    def reclaimed_bytes(self):
        return self.plan.reclaimed_bytes(self.retained)


def scan_removal(root, slug, selected, *, whole=False, fresh=False, context=None, blobs=None):
    """Read removal evidence without creating locks or changing the filesystem."""
    hub = root / "hub"
    repo = hub / slug
    entries = [
        repo / "snapshots" / r
        for r in selected
        if r != LEFTOVER_REVISION and ((repo / "snapshots" / r).is_dir() or (repo / "snapshots" / r).is_symlink())
    ]
    linked = tuple(p for p in entries if p.is_symlink())
    snapshots = tuple(p for p in entries if not p.is_symlink())
    files, owners, logical, row_refs = {}, {}, {}, {}
    for snapshot in snapshots:
        logical[snapshot.name], row_refs[snapshot.name] = 0, set()
        for path in tree_entries(snapshot):
            files[path], owners[path] = path.lstat(), snapshot.name
            name = blob_link(path, repo / "blobs")
            if name is not None:
                row_refs[snapshot.name].add(name)
            try:
                info = path.stat()
            except OSError:
                continue
            if stat.S_ISREG(info.st_mode):
                logical[snapshot.name] += info.st_size
    if blobs is None:
        if whole:
            blobs = sorted((repo / "blobs").iterdir()) if (repo / "blobs").is_dir() else []
        else:
            blobs = exclusive_blob_paths(root, slug, set(selected), fresh=fresh, context=context)
    store = shared_store(hub)
    targets, payload_references = {}, {}
    for path in blobs:
        files[path] = path.lstat()
        owners[path] = next((r for r, names in row_refs.items() if path.name in names), selected[0])
        if store is not None:
            try:
                target = store_target(path, hub, store)
            except OSError:
                target = None
            if target is not None:
                targets[path.name] = target
                owners.setdefault(target, owners[path])
    for target in set(targets.values()):
        payload_references[target] = manifest_references(target, hub, store)
        try:
            files[target] = target.lstat()
        except OSError:
            pass  # a missing/unreadable payload has no known allocation to credit
    return RemovalScan(snapshots, linked, tuple(blobs), targets, store, files, owners, logical, payload_references)


def plan_removal(scan):
    """Pure allocation accounting shared by inventory, prune, and explicit delete."""
    inodes = {}
    for path, info in scan.files.items():
        if path in scan.payload_references:
            references = scan.payload_references[path]
            if not references or not references <= set(scan.blobs):
                continue
        if stat.S_ISREG(info.st_mode):
            inodes.setdefault((info.st_dev, info.st_ino), []).append((path, info))
    credits = []
    for names in inodes.values():
        first, info = names[0]
        # Inconsistent observations imply a concurrent change: do not credit it.
        if all(item.st_nlink == len(names) and item.st_blocks == info.st_blocks for _, item in names):
            credits.append(AllocationCredit(info.st_blocks * 512, frozenset(path for path, _ in names), scan.owners[first]))
    return RemovalPlan(scan, tuple(credits))


def remove_exclusive_blobs(root, slug, plan, *, context=None):
    """Unlink blobs a just-evicted snapshot used alone; return retained paths.

    Each blob is held under huggingface_hub's own per-blob lock while sibling
    references are re-read, so a download linking it into another revision
    meanwhile is either seen or waited out (a busy lock skips the blob).
    Failures skip the blob rather than abort the prune: the snapshot is
    already gone, and its bookkeeping must still be written.
    """
    hub = root / "hub"
    paths, targets = plan.scan.blobs, plan.scan.targets
    held, kept = [], []
    locks = hub / ".locks" / slug
    try:
        for path in paths:
            try:
                locks.mkdir(parents=True, exist_ok=True)
                fd = os.open(locks / (path.name + ".lock"), os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o664)
            except OSError:
                kept.append(path)
                continue
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                os.close(fd)
                kept.append(path)
                continue
            held.append((path, fd))
        references = repo_references(root, slug, fresh=True, context=context)
        # {} (no snapshots left) means nothing references anything; None means unknown.
        used = None if references is None else set().union(*references.values())
        for path, _ in held:
            try:
                info = path.stat()
                if used is None or (info.st_dev, info.st_ino) in used:
                    kept.append(path)
                    continue
                path.unlink(missing_ok=True)
            except OSError:
                kept.append(path)
    finally:
        for _, fd in held:
            os.close(fd)
    kept_targets = {targets[p.name] for p in kept if p.name in targets}
    return set(kept) | sweep_planned_payloads(plan, hub, retained=kept_targets)


def sweep_planned_payloads(plan, hub, *, retained=()):
    """Collect optional payloads; record failures separately from the estimate."""
    kept = set(retained)
    for target in sorted(set(plan.scan.targets.values()) - kept):
        if not sweep_shared(target, hub, plan.scan.store):
            kept.add(target)
    return kept


def sweep_shared(target, hub, store):
    """Collect one payload under its huggingface_hub lock; report whether it was removed."""
    lock = target.with_name(target.name + ".lock")
    if os.path.lexists(str(lock) + ".soft"):
        return False  # a filesystem without flock: huggingface_hub holds the soft lock
    try:
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o666)
    except OSError:
        return False  # shared store may be owned by another user
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return False
        deadline = time.monotonic() + SHARED_LOCK_TIMEOUT
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() > deadline:
                    return False  # busy: leak it rather than wait on a download
                time.sleep(0.1)
        if manifest_references(target, hub, store) != set():
            return False
        if stat.S_ISREG(target.lstat().st_mode):
            target.unlink()
            try:
                target.with_name(target.name + ".refs").unlink()
            except OSError:
                pass  # the payload was freed even if its stale manifest remains
            return True
    except OSError:
        return False
    finally:
        os.close(fd)
    return False


def hf_download_active(hub, slug):
    """True when a huggingface_hub download holds one of this repository's file locks."""
    locks = hub / ".locks" / slug
    if locks.is_symlink() or not locks.is_dir():
        return False
    for path in locks.iterdir():
        if path.suffix != ".lock" or path.is_symlink():
            continue
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except OSError:
            continue
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        finally:
            os.close(fd)
    return False


COMMIT_PATTERN = r"[0-9a-f]{40}"


def snapshot_entries(root, slug, snapshot):
    """Every entry observe() would count, with what it resolves to.

    *kind*: ``blob`` (a link into the repo's or the cache-wide blob store, the
    Hub's layout), ``regular`` (a plain file), ``link`` (a link elsewhere),
    ``dangling`` or ``other``. Sizes follow links.
    """
    hub = Path(os.path.realpath(root / "hub"))
    stores = (hub / slug / "blobs", hub / "blobs")
    entries = []
    for path in tree_entries(snapshot):
        info = path.lstat()
        entry = {"path": str(path.relative_to(snapshot)), "size": None, "mtime": info.st_mtime}
        try:
            target = path.stat()
        except OSError:
            entry["kind"] = "dangling"
            entries.append(entry)
            continue
        if not stat.S_ISREG(target.st_mode):
            entry["kind"] = "other"
        elif stat.S_ISLNK(info.st_mode):
            real = Path(os.path.realpath(path))
            entry["kind"] = "blob" if any(inside(real, store) for store in stores) else "link"
        else:
            entry["kind"] = "regular"
        entry["size"] = target.st_size
        entries.append(entry)
    return sorted(entries, key=lambda e: e["path"])


def scan(root, *, repo=None, revision=None):
    """Untracked snapshots (of *repo*, or every model) and their entries, for adoption.

    *revision* may be a commit or a ref the repo records (``main``). Read-only.
    """
    hub = root / "hub"
    safe_directory(root)
    if not hub.is_dir():
        return []
    if repo is not None and not re.fullmatch(REPO_PATTERN, repo):
        raise ValueError("invalid repository")
    if revision is not None and not re.fullmatch(NAME_PATTERN, revision):
        raise ValueError("invalid revision")
    safe_directory(hub)
    slugs = ["models--" + repo.replace("/", "--")] if repo else sorted(p.name for p in hub.glob("models--*"))
    found = []
    for slug in slugs:
        repo_dir = hub / slug
        snapshots_dir = repo_dir / "snapshots"
        try:
            # The checks every other action makes: a symlinked cache directory
            # is refused, or the record written here would never verify.
            safe_directory(snapshots_dir)
        except ValueError:
            continue
        if not snapshots_dir.is_dir():
            continue
        name = slug.removeprefix("models--")
        wanted = revision
        if revision is not None and not re.fullmatch(COMMIT_PATTERN, revision):
            wanted = read_ref(repo_dir / "refs" / revision)
        for snapshot in sorted(snapshots_dir.iterdir()):
            if not re.fullmatch(COMMIT_PATTERN, snapshot.name) or (revision is not None and snapshot.name != wanted):
                continue
            if snapshot.is_symlink() or not snapshot.is_dir():
                continue
            record = root / STATE / "usage" / name / (snapshot.name + ".json")
            problem = None
            if record.exists() or record.is_symlink():
                try:
                    read_record(record, name, snapshot.name)
                    continue  # already tracked
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    # observe() reports this revision unverifiable; say why here
                    # rather than skip it as if it were tracked.
                    problem = "its usage record is unreadable (%s); remove %s to adopt it again" % (exc, record)
            found.append(
                {
                    "repo": name.replace("--", "/", 1),
                    "revision": snapshot.name,
                    "snapshot": str(snapshot),
                    "entries": snapshot_entries(root, slug, snapshot),
                    "problem": problem,
                }
            )
    return found


def adopt(root, *, repo, revision, files, endpoint, last_used, pinned, now):
    """Start tracking an existing snapshot: write its usage record, under the publication lock.

    The record lists exactly the files found, so nothing in the snapshot is
    "untracked files". Refuses when the snapshot changed since the caller's
    scan, or became tracked meanwhile.
    """
    if not isinstance(repo, str) or not re.fullmatch(REPO_PATTERN, repo):
        raise ValueError("invalid repository")
    if not isinstance(revision, str) or not re.fullmatch(COMMIT_PATTERN, revision):
        raise ValueError("adoption needs a full commit revision")
    if not isinstance(files, list) or not files or not all(isinstance(f, list) and len(f) == 3 for f in files):
        raise ValueError("adoption needs the snapshot's scanned entries")
    if type(last_used) not in (int, float) or not math.isfinite(last_used) or last_used < 0:
        raise ValueError("invalid adoption time")
    if not isinstance(endpoint, str) or not endpoint.startswith(("https://", "http://")):
        raise ValueError("adoption needs the model origin endpoint")
    name = repo.replace("/", "--")
    slug = "models--" + name
    snapshot = root / "hub" / slug / "snapshots" / revision
    lock = root / STATE / (name + ".lock")
    record_path = root / STATE / "usage" / name / (revision + ".json")
    for path in (root / STATE, record_path.parent, snapshot.parent):
        safe_directory(path)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o644)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("invalid publication lock")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("cache busy: another operation holds this model's lock") from None
        if record_path.exists():
            raise ValueError("revision is already tracked")
        if snapshot.is_symlink() or not snapshot.is_dir():
            raise ValueError("snapshot is no longer present")
        # Path, kind and size: the caller verified exactly these against the origin.
        found = sorted([e["path"], e["kind"], e["size"]] for e in snapshot_entries(root, slug, snapshot))
        if found != sorted(files):
            raise ValueError("snapshot changed since it was scanned; scan again")
        record = {
            "schema": USAGE_SCHEMA_VERSION,
            "repo": repo,
            "revision": revision,
            "endpoint": endpoint,
            "first_seen": now,
            "last_used": last_used,
            "last_adopted": now,
            "files": sorted(f[0] for f in files),
            "claims": {},
            "pinned": pinned,
        }
        validate_usage(record, name, revision)
        write_record(record_path, record)
    finally:
        os.close(fd)
    return observe(root, slug, revision, now=now, min_age_seconds=0, exclude_repos=())


# A whole-model delete can select a repository containing only leftover blobs.
LEFTOVER_REVISION = "-"


def select_delete(root, slug, revision):
    repo = root / "hub" / slug
    snapshots = repo / "snapshots"
    usage = root / STATE / "usage" / slug.removeprefix("models--")
    present = (
        {p.name for p in snapshots.iterdir() if re.fullmatch(NAME_PATTERN, p.name) and (p.is_dir() or p.is_symlink())}
        if snapshots.is_dir()
        else set()
    )
    recorded = {p.stem for p in usage.glob("*.json") if re.fullmatch(NAME_PATTERN, p.stem)} if usage.is_dir() else set()
    revisions = present | recorded
    if revision is None:
        return sorted(revisions) or ([LEFTOVER_REVISION] if repo.is_dir() else [])
    if revision in revisions:
        return [revision]
    resolved = read_ref(repo / "refs" / revision)
    return [resolved] if resolved is not None and resolved in revisions else []


def observe_delete(root, slug, selected, **policy):
    rows = []
    for revision in selected:
        row = observe(root, slug, "0" * 40 if revision == LEFTOVER_REVISION else revision, **policy)
        if revision == LEFTOVER_REVISION:
            row.update(revision=LEFTOVER_REVISION, snapshot=str(root / "hub" / slug), present=True)
        rows.append(row)
    return rows


def delete_selection_refusal(selected, rows, *, allowed, force):
    """Explicit-delete policy, independent of retention pins, age, and tracking."""
    if allowed is not None and set(selected) - set(allowed):
        return "selection_changed"
    if any(row["claims"] for row in rows) and not force:
        return "workload_claim"
    return None


def check_delete(root, slug, selected, rows, *, allowed, force):
    refusal = delete_selection_refusal(selected, rows, allowed=allowed, force=force)
    if refusal is None and hf_download_active(root / "hub", slug):
        refusal = "download_in_progress"
    return refusal


def refuse_delete(result, rows, code):
    for row in rows:
        row_code = "delete_blocked" if code == "workload_claim" and not row["claims"] else code
        row.update(eligible=False, reason=CACHE_REASONS[row_code], reason_code=row_code)
    result.update(entries=rows, target_met=False, free_bytes_after=result["free_bytes"])
    return result


def describe_delete_plan(result, rows, plan):
    """Render the preview/apply selection without consulting the filesystem."""
    for row in rows:
        logical, private = plan.row_bytes(row["revision"])
        row.update(
            logical_bytes=logical,
            reclaimable_bytes=private,
            eligible=True,
            reason=CACHE_REASONS["explicit_delete"],
            reason_code="explicit_delete",
            action="would_delete" if result["dry_run"] else "deleted",
        )
    result.update(entries=rows, planned_bytes=plan.planned_bytes)


def delete(root, *, repo, revision, force, dry_run, now, allowed=None, context=None):
    """Remove one revision, or every revision, of *repo* regardless of retention policy.

    Pins, untracked snapshots and retention age do not protect against an
    explicit delete. Outstanding workload claims do unless *force*; an active
    download or a held publication lock always does, in a preview too, so a
    preview that passes everywhere predicts the apply. *allowed* (the revisions
    a caller confirmed) refuses anything that appeared since. A blocked request
    deletes nothing on this host and reports ``target_met: false``.

    Shared-store payloads are collected with huggingface_hub's own lock and
    manifest, so a concurrent download linking the same content cannot race it.
    """
    if not isinstance(repo, str) or not re.fullmatch(REPO_PATTERN, repo):
        raise ValueError("invalid repository")
    if revision is not None and (not isinstance(revision, str) or not re.fullmatch(NAME_PATTERN, revision)):
        raise ValueError("invalid revision")
    if allowed is not None and (not isinstance(allowed, list) or not all(isinstance(a, str) for a in allowed)):
        raise ValueError("allowed revisions must be a list of names")
    safe_directory(root)
    hub = root / "hub"
    name = repo.replace("/", "--")
    slug = "models--" + name
    repo_dir = hub / slug
    snapshots_dir = repo_dir / "snapshots"
    repo_blobs = repo_dir / "blobs"
    usage_dir = root / STATE / "usage" / name
    for path in (hub, repo_dir, snapshots_dir, repo_blobs, repo_dir / "refs", usage_dir):
        safe_directory(path)
    whole = revision is None
    result = cache_report(root, [], dry_run=dry_run, free=free_space(root))
    context = context if context is not None else ObservationContext()
    policy = dict(now=now, min_age_seconds=0, exclude_repos=(), context=context)
    selected = select_delete(root, slug, revision)
    if not selected:
        result["free_bytes_after"] = result["free_bytes"]
        return result
    rows = observe_delete(root, slug, selected, **policy)
    refusal = check_delete(root, slug, selected, rows, allowed=allowed, force=force)
    if refusal:
        return refuse_delete(result, rows, refusal)

    lock = root / STATE / (name + ".lock")
    safe_directory(lock.parent)
    if dry_run:
        # Probe without writing: a missing lock cannot be held by anyone.
        try:
            fd = os.open(lock, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except FileNotFoundError:
            fd = None
        except OSError:
            return refuse_delete(result, rows, "invalid_publication_lock")
    else:
        lock.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(lock, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o644)
        except OSError:
            return refuse_delete(result, rows, "invalid_publication_lock")
    try:
        if fd is not None:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                return refuse_delete(result, rows, "invalid_publication_lock")
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return refuse_delete(result, rows, "cache_busy")
            # Re-read under the lock: a launch may have claimed, or a download
            # added, a revision since.
            context.refresh()
            selected = select_delete(root, slug, revision)
            rows = observe_delete(root, slug, selected, **policy)
            refusal = check_delete(root, slug, selected, rows, allowed=allowed, force=force)
            if refusal:
                return refuse_delete(result, rows, refusal)

        if not selected:
            result["free_bytes_after"] = result["free_bytes"]
            return result
        plan = plan_removal(scan_removal(root, slug, selected, whole=whole, fresh=True, context=context))
        describe_delete_plan(result, rows, plan)
        if dry_run:
            result["free_bytes_after"] = result["free_bytes"] + plan.planned_bytes
            return result

        outcome = apply_delete(root, slug, selected, plan, whole=whole)
        result["reclaimed_bytes"] = outcome.reclaimed_bytes
        result["free_bytes_after"] = free_space(root)
        return result
    finally:
        if fd is not None:
            os.close(fd)


def apply_delete(root, slug, selected, plan, *, whole):
    """Mutate only after the caller reselected and replanned under its publication lock."""
    hub = root / "hub"
    repo_dir = hub / slug
    usage_dir = root / STATE / "usage" / slug.removeprefix("models--")
    if not shutil.rmtree.avoids_symlink_attacks:
        raise ValueError("safe snapshot deletion is unavailable on this host")
    if whole:
        if repo_dir.is_dir():
            shutil.rmtree(repo_dir)
        if usage_dir.is_dir():
            shutil.rmtree(usage_dir)
    else:
        for path in plan.scan.linked:
            path.unlink()
        for snap in plan.scan.snapshots:
            shutil.rmtree(snap)
        for path in plan.scan.blobs:
            path.unlink()
        refs = repo_dir / "refs"
        if refs.is_dir():
            for ref in tree_entries(refs):
                if read_ref(ref) in selected:
                    ref.unlink()
        for r in selected:
            record = usage_dir / (r + ".json")
            if record.is_file() and not record.is_symlink():
                record.unlink()
    retained = sweep_planned_payloads(plan, hub)
    return RemovalOutcome(plan, frozenset(retained))
