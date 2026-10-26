"""Host/cache-local usage and durable workload protection, shared by all callers.

Records are separate from immutable manifests. Every read/modify/write holds
the same repository lease as preparation and pruning. Claims deliberately have
no TTL: an abandoned controller is not evidence that its workload stopped.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import json
import logging
from pathlib import PurePosixPath
import time
import uuid

from sparkrun.models.artifacts import ModelArtifactError, ModelArtifactRequest
from sparkrun.models._cache_contract import STATE_DIRECTORY, USAGE_SCHEMA_VERSION, same_origin, validate_usage

logger = logging.getLogger(__name__)


@dataclass
class UsageScope:
    token: str = field(default_factory=lambda: uuid.uuid4().hex)
    workload: str | None = None
    executor: str | None = None
    activated: bool = False
    claims: list[dict] = field(default_factory=list)
    ssh_kwargs: dict = field(default_factory=dict)


_SCOPE: ContextVar[UsageScope | None] = ContextVar("model_cache_usage", default=None)


@contextmanager
def usage_operation():
    if _SCOPE.get() is not None:
        yield
        return
    scope = UsageScope()
    token = _SCOPE.set(scope)
    try:
        yield
    finally:
        try:
            if not scope.activated:
                release_claims(scope.claims, ssh_kwargs=scope.ssh_kwargs)
        except Exception:
            logger.warning("Could not release model cache preparation claims; reconcile them before pruning", exc_info=True)
        _SCOPE.reset(token)


def begin_workload(cluster_id, ssh_kwargs):
    scope = _SCOPE.get()
    if scope is not None:
        scope.workload = cluster_id
        scope.ssh_kwargs = dict(ssh_kwargs)


def set_workload_executor(name):
    """Record the executor the workload runs under in its claims.

    ``model-cache reconcile`` then requires that executor's status to have
    been swept before it reads the workload as gone.
    """
    scope = _SCOPE.get()
    if scope is not None:
        scope.executor = name if isinstance(name, str) and name else None


def activate_workload():
    scope = _SCOPE.get()
    if scope is not None and not scope.activated:
        scope.activated = True
        from sparkrun.models.host_io import ModelHostIO

        if scope.claims:
            with ModelHostIO(scope.ssh_kwargs) as io:
                for ref in sorted(scope.claims, key=lambda r: (r["host"], r["cache_root"], r["repo"])):
                    host, root, repo, revision = (ref[k] for k in ("host", "cache_root", "repo", "revision"))
                    io.lock(host, root, repo)
                    record = read_usage(io, host, root, repo, revision)
                    if record is None or scope.token not in record["claims"]:
                        raise ModelArtifactError("model cache launch claim disappeared")
                    record["last_launched"] = record["last_used"] = max(record["last_used"], time.time())
                    io.publish(host, root, repo, usage_path(root, repo, revision), json.dumps(record, sort_keys=True))


def workload_claims():
    scope = _SCOPE.get()
    return list(scope.claims) if scope is not None else []


def usage_path(root, repo, revision):
    ModelArtifactRequest(repo, revision)
    import re

    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ModelArtifactError("cache usage requires an immutable revision")
    return str(PurePosixPath(root) / STATE_DIRECTORY / "usage" / repo.replace("/", "--") / (revision + ".json"))


def read_usage(io, host, root, repo, revision):
    text = io.read(host, usage_path(root, repo, revision))
    if text is None:
        return None
    try:
        record = json.loads(text)
        return validate_usage(record, repo.replace("/", "--"), revision)
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelArtifactError("invalid model cache usage record; refusing mutation") from exc


def record_use(io, host, root, manifest, *, event="prepared", acquired=False):
    """Called under the publication lease, after successful validation."""
    io.lock(host, root, manifest.repo)
    now = time.time()
    record = read_usage(io, host, root, manifest.repo, manifest.revision) or {
        "schema": USAGE_SCHEMA_VERSION,
        "repo": manifest.repo,
        "revision": manifest.revision,
        "endpoint": manifest.endpoint,
        "first_seen": now,
        "last_used": now,
        "files": [],
        "claims": {},
        "pinned": False,
    }
    if not same_origin(record["endpoint"], manifest.endpoint):
        if "last_adopted" in record and not any(k in record for k in ("last_prepared", "last_acquired", "last_distributed")):
            # Adoption recorded whatever origin the adopter named (or, offline,
            # none it consulted); the first preparation is the authoritative one.
            record["endpoint"] = manifest.endpoint
        else:
            raise ModelArtifactError("cache usage endpoint mismatch")
    record["files"] = sorted(set(record["files"]) | {f.path for f in manifest.files})
    record["last_" + event] = now
    record["last_used"] = max(record["last_used"], now)
    if acquired:
        record["last_acquired"] = now
    if event == "prepared":
        _claim(record, host, root, manifest.repo, manifest.revision)
    io.publish(host, root, manifest.repo, usage_path(root, manifest.repo, manifest.revision), json.dumps(record, sort_keys=True))


def _claim(record, host, root, repo, revision):
    scope = _SCOPE.get()
    if scope is not None and scope.workload and host is not None:
        claim = record["claims"].setdefault(scope.token, {"workload": scope.workload, "hosts": [], "created": time.time()})
        if scope.executor:
            claim.setdefault("executor", scope.executor)
        claim["hosts"] = sorted(set(claim["hosts"]) | {host})
        reference = {"host": host, "cache_root": root, "repo": repo, "revision": revision, "claim": scope.token}
        if reference not in scope.claims:
            scope.claims.append(reference)


def protect_unverified(model, hosts, cache_dir, *, ssh_kwargs=None, revision=None):
    """Legacy fallback cannot prove a revision: protect all tracked revisions.

    Unknown/untracked revisions are already ineligible for deletion. This does
    not turn a compatibility fallback into a validated or managed artifact.
    """
    scope = _SCOPE.get()
    if scope is None or not scope.workload:
        return
    from sparkrun.models.host_io import ModelHostIO
    from sparkrun.utils.shell import quote

    repo = ModelArtifactRequest(model).repo
    with ModelHostIO(ssh_kwargs) as io:
        for host in sorted(hosts):
            root = io.root(host, cache_dir)
            io.lock(host, root, repo)
            directory = str(PurePosixPath(root) / STATE_DIRECTORY / "usage" / repo.replace("/", "--"))
            names = io.checked(host, "for f in " + str(quote(directory)) + '/*.json; do [ ! -e "$f" ] || basename -- "$f"; done\n')
            for name in names.splitlines():
                selected_revision = name.removesuffix(".json")
                if revision is not None and revision != selected_revision:
                    continue
                record = read_usage(io, host, root, repo, selected_revision)
                if record is None:
                    continue
                _claim(record, host, root, repo, selected_revision)
                record["last_used"] = max(record["last_used"], time.time())
                io.publish(host, root, repo, usage_path(root, repo, selected_revision), json.dumps(record, sort_keys=True))


def protect_preplaced(path, host, *, ssh_kwargs=None):
    """Retain an already tracked HF snapshot used through an absolute path.

    The path may select a nested GGUF file/component. Arbitrary directories
    outside an HF snapshot never become managed or eligible for pruning.
    """
    import re

    path = PurePosixPath(path)
    for snapshot in (path, *path.parents):
        if (
            re.fullmatch(r"[0-9a-f]{40}", snapshot.name)
            and snapshot.parent.name == "snapshots"
            and snapshot.parent.parent.name.startswith("models--")
            and snapshot.parent.parent.parent.name == "hub"
        ):
            repo = snapshot.parent.parent.name.removeprefix("models--").replace("--", "/", 1)
            protect_unverified(repo, [host], str(snapshot.parents[3]), ssh_kwargs=ssh_kwargs, revision=snapshot.name)
            return


def release_claims(claims, *, hosts=None, ssh_kwargs=None, session=None):
    from sparkrun.models.host_io import ModelHostIO

    if not claims:
        return
    with ModelHostIO(ssh_kwargs, session=session) as io:
        for ref in sorted(claims, key=lambda r: (r["host"], r["cache_root"], r["repo"])):
            host, root, repo, revision = (ref[k] for k in ("host", "cache_root", "repo", "revision"))
            if hosts is not None and host not in hosts:
                continue
            io.lock(host, root, repo)
            record = read_usage(io, host, root, repo, revision)
            if record is None or ref["claim"] not in record["claims"]:
                continue
            claim = record["claims"][ref["claim"]]
            claim["hosts"] = [h for h in claim["hosts"] if h != host]
            if not claim["hosts"]:
                del record["claims"][ref["claim"]]
            record["last_released"] = time.time()
            record["last_used"] = max(record["last_used"], record["last_released"])
            io.publish(host, root, repo, usage_path(root, repo, revision), json.dumps(record, sort_keys=True))
