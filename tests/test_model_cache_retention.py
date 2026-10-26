"""Real host-agent filesystem/locking tests; no downloads or remote workloads."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("fcntl")

from sparkrun.api import model_cache
from sparkrun.models import _cache_agent as agent
from sparkrun.models.artifacts import ModelArtifactRequest, ModelArtifactError
from sparkrun.models.host_io import ModelHostIO
from sparkrun.models.preparation import prepare_model
from sparkrun.models.usage import (
    activate_workload,
    begin_workload,
    record_use,
    release_claims,
    usage_operation,
    usage_path,
    workload_claims,
)
from test_model_artifacts import artifact, cache, SHA


def cli_context(*, default_cluster=None, default_hosts=None):
    """Fake CLI context with no saved clusters unless one is given."""
    config = SimpleNamespace(ssh_user=None, ssh_key=None, ssh_options=None, default_hosts=default_hosts)
    return SimpleNamespace(config=config, cluster_manager=Mock(get_default=Mock(return_value=default_cluster)))


def tracked(root, *, revision=SHA, shared=False, last_used=1, repo="org/model"):
    manifest = replace(artifact(), revision=revision, repo=repo)
    snapshot = cache(root, manifest, shared=shared)
    with ModelHostIO() as io:
        record_use(io, "localhost", str(root), manifest)
    path = Path(usage_path(root, manifest.repo, revision))
    record = json.loads(path.read_text())
    record["last_used"] = last_used
    path.write_text(json.dumps(record))
    return manifest, snapshot, path


def test_retention_dtos_validate_on_wire_and_reject_wrong_origin(tmp_path):
    from sparkrun.api._wire import to_wire, validate
    from sparkrun.api._wire_schemas import RESULT_SCHEMAS
    from sparkrun.models.cache_reports import CacheReport

    tracked(tmp_path)
    reports = model_cache.inventory_cache(cache_dir=str(tmp_path))
    assert isinstance(reports[0], CacheReport)
    assert reports[0].entries[0].reason_code == "eligible"
    assert reports[0].budget_scope == "private_managed_snapshots"
    validate(to_wire(reports), RESULT_SCHEMAS["model_cache.list"])
    request = ModelArtifactRequest("org/model", SHA, endpoint="https://other.example")
    with pytest.raises(ModelArtifactError, match="endpoint mismatch"):
        model_cache.pin_revision(request, cache_dir=str(tmp_path))
    assert not model_cache.inventory_cache(cache_dir=str(tmp_path))[0].entries[0].pinned


def test_metadata_age_pruning_preserves_claim_recovery_handles(tmp_path, monkeypatch):
    from sparkrun.orchestration.job_metadata import prune_job_metadata

    job = SimpleNamespace(workload_id="workload", metadata={"model_cache_claims": [{"claim": "keep"}]})
    monkeypatch.setattr("sparkrun.api._jobs.list_jobs", lambda **kw: [job])
    assert prune_job_metadata(cache_dir=str(tmp_path), max_age_days=1, keep_per_intent=0) == []


def test_finalization_retains_metadata_on_partial_teardown_and_release_error(monkeypatch):
    from sparkrun.orchestration import workload_cleanup as cleanup

    claims = [{"host": "a"}, {"host": "b"}]
    monkeypatch.setattr(
        "sparkrun.orchestration.job_metadata.load_job_metadata", lambda *a, **k: {"hosts": ["a", "b"], "model_cache_claims": claims}
    )
    released, removed = [], []
    monkeypatch.setattr("sparkrun.models.usage.release_claims", lambda claims, **kw: released.append(kw["hosts"]))
    monkeypatch.setattr("sparkrun.orchestration.job_metadata.remove_job_metadata", lambda *a, **k: removed.append(True))
    assert cleanup.finalize_workload("workload", hosts=["a"]) is False
    assert released == [{"a"}] and not removed
    monkeypatch.setattr("sparkrun.models.usage.release_claims", Mock(side_effect=ModelArtifactError("unreachable")))
    with pytest.raises(ModelArtifactError):
        cleanup.finalize_workload("workload", hosts=["a", "b"])
    assert not removed


def test_inventory_and_preview_are_read_only(tmp_path):
    _, snapshot, path = tracked(tmp_path)
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()}
    listing = model_cache.inventory_cache(cache_dir=str(tmp_path))[0]
    assert listing["entries"][0]["eligible"]
    assert listing["entries"][0]["last_used"] == 1
    report = model_cache.prune_cache(cache_dir=str(tmp_path), min_age_days=0)[0]
    assert report["dry_run"] and report["reclaimed_bytes"] == 0
    assert report["entries"][0]["action"] == "would_remove"
    assert report["planned_bytes"] > 0
    assert snapshot.exists() and path.exists()
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()}


def test_lru_budget_and_retained_inventory(tmp_path):
    _, old, old_record = tracked(tmp_path, last_used=1)
    _, recent, _ = tracked(tmp_path, revision="b" * 40, last_used=2)
    size = agent.run(str(tmp_path))["entries"][0]["reclaimable_bytes"]
    result = model_cache.prune_cache(cache_dir=str(tmp_path), min_age_days=0, max_bytes=size, dry_run=False)[0]
    assert result["target_met"]
    assert result["reclaimed_bytes"] == size
    assert not old.exists() and recent.exists()
    assert "last_evicted" in json.loads(old_record.read_text())
    assert Path(tmp_path / ".sparkrun-model-artifacts/org--model.lock").exists()


def test_unknown_snapshots_and_unknown_files_are_protected(tmp_path):
    _, snapshot, _ = tracked(tmp_path)
    (snapshot / "operator-notes").write_text("keep")
    unknown = cache(tmp_path, replace(artifact(), revision="c" * 40))
    report = model_cache.prune_cache(cache_dir=str(tmp_path), min_age_days=0, dry_run=False)[0]
    assert {e["reason"] for e in report["entries"]} == {"untracked", "untracked files"}
    assert snapshot.exists() and unknown.exists()


def test_shared_blobs_and_other_revision_survive(tmp_path):
    _, snapshot, _ = tracked(tmp_path, shared=True)
    blobs = [p.resolve() for p in snapshot.iterdir()]
    untracked = cache(tmp_path, replace(artifact(), revision="b" * 40))
    report = model_cache.prune_cache(cache_dir=str(tmp_path), min_age_days=0, dry_run=False)[0]
    assert not snapshot.exists() and untracked.exists()
    assert all(p.exists() for p in blobs)
    assert report["reclaimed_bytes"] == 0


def test_hardlinks_are_not_counted_as_freed_bytes(tmp_path):
    import os

    _, snapshot, _ = tracked(tmp_path)
    keep = tmp_path / "keep"
    keep.mkdir()
    for p in snapshot.iterdir():
        os.link(p, keep / p.name)
    report = model_cache.prune_cache(cache_dir=str(tmp_path), min_age_days=0, dry_run=False)[0]
    assert report["reclaimed_bytes"] == 0
    assert all(p.read_bytes() for p in keep.iterdir())


def test_pin_and_age_prevent_deletion_and_unmet_target_is_reported(tmp_path):
    _, snapshot, _ = tracked(tmp_path)
    request = ModelArtifactRequest("org/model", SHA)
    model_cache.pin_revision(request, cache_dir=str(tmp_path))
    report = model_cache.prune_cache(cache_dir=str(tmp_path), max_bytes=0, min_age_days=0, dry_run=False)[0]
    assert not report["target_met"] and report["entries"][0]["reason"] == "pinned"
    model_cache.pin_revision(request, cache_dir=str(tmp_path), pinned=False)
    report = model_cache.prune_cache(cache_dir=str(tmp_path), min_age_days=10**6, dry_run=False)[0]
    assert report["entries"][0]["reason"] == "retention age" and snapshot.exists()


def test_publication_lock_protects_transfer_and_warm_reader(tmp_path):
    _, snapshot, _ = tracked(tmp_path)
    with ModelHostIO() as io:
        io.lock("localhost", str(tmp_path), "org/model")
        report = model_cache.prune_cache(cache_dir=str(tmp_path), min_age_days=0, dry_run=False)[0]
        assert report["entries"][0]["reason"] == "cache busy"
    assert snapshot.exists()


@pytest.mark.parametrize("activated", [False, True])
def test_launch_claim_lifetime_survives_controller_scope(tmp_path, activated):
    manifest, snapshot, _ = tracked(tmp_path)
    with pytest.raises(RuntimeError), usage_operation():
        begin_workload("test-workload", {})
        prepare_model("org/model", ["localhost"], cache_dir=str(tmp_path), manifest=manifest)
        claims = workload_claims()
        assert claims
        if activated:
            activate_workload()
        raise RuntimeError("interrupted launch")
    result = model_cache.prune_cache(cache_dir=str(tmp_path), min_age_days=0, dry_run=False)[0]
    assert snapshot.exists() == activated
    if activated:
        assert result["entries"][0]["reason"] == "workload claim"
        assert "last_launched" in result["entries"][0]["usage"]
        release_claims(claims)
        report = model_cache.inventory_cache(cache_dir=str(tmp_path))[0]
        assert report["entries"][0]["claims"] == {}
        assert report["entries"][0]["reason"] == "retention age"


def test_shared_cache_claim_is_only_released_after_all_hosts_stop(tmp_path, monkeypatch):
    manifest, _, path = tracked(tmp_path)

    class SharedHosts(ModelHostIO):
        def execute(self, host, script, **kwargs):
            return super().execute(None, script, **kwargs)

        def lock(self, host, root, repo):
            return super().lock(None, root, repo)

    monkeypatch.setattr("sparkrun.models.host_io.ModelHostIO", SharedHosts)
    with usage_operation():
        begin_workload("shared-workload", {})
        with SharedHosts() as io:
            record_use(io, "host-a", str(tmp_path), manifest)
            record_use(io, "host-b", str(tmp_path), manifest)
        claims = workload_claims()
        activate_workload()
    release_claims(claims, hosts=["host-a"])
    record = json.loads(path.read_text())
    assert next(iter(record["claims"].values()))["hosts"] == ["host-b"]
    release_claims(claims, hosts=["host-b"])
    assert not json.loads(path.read_text())["claims"]


def test_reconciliation_requires_explicit_acknowledgement(tmp_path):
    manifest, _, _ = tracked(tmp_path)
    with usage_operation():
        begin_workload("interrupted-workload", {})
        with ModelHostIO() as io:
            record_use(io, "localhost", str(tmp_path), manifest)
        claim = workload_claims()[0]["claim"]
        activate_workload()
    request = ModelArtifactRequest("org/model", SHA)
    with pytest.raises(ModelArtifactError, match="explicit confirmation"):
        model_cache.reconcile_claim(request, cache_dir=str(tmp_path), claim=claim)
    model_cache.reconcile_claim(request, cache_dir=str(tmp_path), claim=claim, confirm_inactive=True)
    assert model_cache.inventory_cache(cache_dir=str(tmp_path))[0]["entries"][0]["claims"] == {}


@pytest.mark.parametrize("attack", ["snapshot_symlink", "usage_symlink", "invalid_record", "lock_symlink"])
def test_unverifiable_paths_fail_closed(tmp_path, attack):
    _, snapshot, path = tracked(tmp_path)
    other = tmp_path / "outside"
    other.mkdir()
    (other / "keep").write_text("safe")
    if attack == "snapshot_symlink":
        import shutil

        shutil.rmtree(snapshot)
        snapshot.symlink_to(other, target_is_directory=True)
    elif attack == "usage_symlink":
        path.unlink()
        path.symlink_to(other / "keep")
    elif attack == "invalid_record":
        path.write_text("{}")
    else:
        lock = tmp_path / ".sparkrun-model-artifacts/org--model.lock"
        lock.unlink()
        lock.symlink_to(other / "keep")
    report = model_cache.prune_cache(cache_dir=str(tmp_path), min_age_days=0, dry_run=False)[0]
    assert not any(e.get("action") == "removed" for e in report["entries"])
    assert (other / "keep").read_text() == "safe"


def test_auto_prune_is_opt_in_and_excludes_the_requested_repository(tmp_path):
    old_manifest, old, _ = tracked(tmp_path, repo="other/old")
    manifest, current, _ = tracked(tmp_path)
    prepare_model("org/model", ["localhost"], cache_dir=str(tmp_path), manifest=manifest)
    assert old.exists()
    config = {"model_cache_prune": {"enabled": True, "max_bytes": 1, "min_age_days": 0}}
    prepare_model("org/model", ["localhost"], cache_dir=str(tmp_path), manifest=manifest, config=config)
    assert current.exists() and not old.exists()


def test_cli_preview_default_and_daemon_wire_report(tmp_path, monkeypatch):
    from click.testing import CliRunner
    from sparkrun.cli import _model_cache as cli
    from sparkrun.api._wire import validate
    from sparkrun.api._wire_schemas import RESULT_SCHEMAS
    from sparkrun.daemon.special_ops import execute

    _, snapshot, _ = tracked(tmp_path)
    context = cli_context()
    monkeypatch.setattr(cli, "_get_context", lambda ctx: context)
    result = CliRunner().invoke(cli.model_cache, ["prune", "--cache-dir", str(tmp_path), "--min-age-days", "0", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)[0]["dry_run"] and snapshot.exists()
    report = execute("model_cache.list", {"cache_dir": str(tmp_path)}, context)
    validate(report, RESULT_SCHEMAS["model_cache.list"])
    prune = Mock(return_value=[])
    monkeypatch.setattr(model_cache, "prune_cache", prune)
    execute("model_cache.prune", {"cache_dir": str(tmp_path)}, context)
    assert "dry_run" not in prune.call_args.kwargs  # API default remains preview.


def test_empty_inventory_does_not_create_a_cache(tmp_path):
    missing = tmp_path / "not-created"
    assert model_cache.inventory_cache(cache_dir=str(missing))[0].entries == ()
    assert not missing.exists()


def test_legacy_and_absolute_path_launches_protect_tracked_revisions(tmp_path):
    from sparkrun.models.usage import protect_unverified
    from sparkrun.models.preparation import validate_preplaced

    _, snapshot, _ = tracked(tmp_path)
    for prepare in (
        lambda: protect_unverified("org/model", ["localhost"], str(tmp_path)),
        lambda: validate_preplaced(str(snapshot), ["localhost"]),
    ):
        with usage_operation():
            begin_workload("legacy", {})
            prepare()
            assert workload_claims()
            activate_workload()
        assert model_cache.prune_cache(cache_dir=str(tmp_path), min_age_days=0)[0]["entries"][0]["reason"] == "workload claim"


def test_claims_for_distinct_launches_do_not_release_each_other(tmp_path):
    manifest, _, path = tracked(tmp_path)
    claims = []
    for workload in ("one", "two"):
        with usage_operation():
            begin_workload(workload, {})
            with ModelHostIO() as io:
                record_use(io, "localhost", str(tmp_path), manifest)
            claims.append(workload_claims())
            activate_workload()
    release_claims(claims[0])
    remaining = json.loads(path.read_text())["claims"]
    assert len(remaining) == 1 and next(iter(remaining.values()))["workload"] == "two"


def test_recent_use_between_scan_and_lock_is_preserved(tmp_path, monkeypatch):
    _, snapshot, path = tracked(tmp_path)
    original = agent.inventory

    def changed(root, **policy):
        rows = original(root, **policy)
        record = json.loads(path.read_text())
        record["last_used"] = 100
        path.write_text(json.dumps(record))
        return rows

    monkeypatch.setattr(agent, "inventory", changed)
    result = agent.run(str(tmp_path), action="prune", min_age_seconds=0, dry_run=False)
    assert result["entries"][0]["reason"] == "usage changed during cleanup"
    assert snapshot.exists()


def test_daemon_prune_and_reconcile_require_admin_scope():
    from sparkrun.daemon.ops import register_core_ops, check_request
    from sparkrun.core.daemon_registry import get_daemon_op

    register_core_ops()
    assert get_daemon_op("model_cache.list").scope == "read"
    assert get_daemon_op("model_cache.prune").scope == "admin"
    op = get_daemon_op("model_cache.reconcile")
    assert op.scope == "admin"
    with pytest.raises(ValueError):
        check_request(op, {"cache_dir": "/hf", "model": "org/model", "revision": SHA, "claim": "x", "confirm_inactive": False})


@pytest.mark.parametrize("days", [-1, float("inf"), float("nan"), True, "7"])
def test_invalid_retention_cannot_mutate_cache(tmp_path, days):
    with pytest.raises(ValueError):
        model_cache.prune_cache(cache_dir=str(tmp_path), min_age_days=days, dry_run=False)


@pytest.mark.parametrize(
    "command,function,extra",
    [
        ("pin", "pin_revision", []),
        ("reconcile", "reconcile_claim", ["--claim", "interrupted", "--confirm-inactive"]),
    ],
)
@pytest.mark.parametrize("via_env", [False, True])
def test_cli_retention_preserves_custom_endpoint(monkeypatch, tmp_path, command, function, extra, via_env):
    from click.testing import CliRunner
    from sparkrun.cli import _model_cache as cli

    context = cli_context()
    monkeypatch.setattr(cli, "_get_context", lambda ctx: context)
    mutation = Mock(return_value=())
    monkeypatch.setattr(model_cache, function, mutation)
    endpoint = "https://mirror.example"
    monkeypatch.setenv("HF_ENDPOINT", endpoint if via_env else "https://unused.example")
    args = [command, "org/model", "--revision", SHA, "--cache-dir", str(tmp_path), *extra]
    if not via_env:
        args.extend(["--endpoint", endpoint])
    result = CliRunner().invoke(cli.model_cache, args)
    assert result.exit_code == 0, result.output
    assert mutation.call_args.args[0].endpoint == endpoint


GONE = "sparkrun_" + "a" * 16 + "_" + "b" * 12
LIVE = "sparkrun_" + "c" * 16 + "_" + "d" * 12


def _claimed(root, workload, *, created=1.0):
    """A tracked revision holding one activated claim of *workload* on localhost."""
    manifest, _, path = tracked(root)
    with usage_operation():
        begin_workload(workload, {})
        with ModelHostIO() as io:
            record_use(io, "localhost", str(root), manifest)
        token = workload_claims()[0]["claim"]
        activate_workload()
    record = json.loads(path.read_text())
    record["claims"][token]["created"] = created
    path.write_text(json.dumps(record))
    return token, path


def _sweep(monkeypatch, *, running=(), errors=None, pending=(), jobs=(), swept=("docker",), gated=None):
    coverage = tuple(SimpleNamespace(target=SimpleNamespace(executor=name), status_scope="host") for name in swept)
    monkeypatch.setattr("sparkrun.orchestration.executors._base._GATED_OFF_EXECUTORS", dict(gated or {}))
    snapshot = SimpleNamespace(running_cluster_ids=lambda: tuple(running), observation_errors=dict(errors or {}), coverage=coverage)
    records = [SimpleNamespace(workload_id=w, metadata={"executor": e}) for w, e in jobs]
    monkeypatch.setattr("sparkrun.api._jobs.list_jobs", lambda **kw: records)
    sweep = Mock(return_value=snapshot)
    monkeypatch.setattr("sparkrun.api._status.status", sweep)
    monkeypatch.setattr("sparkrun.core.pending_ops.list_pending_ops", lambda: [{"job_cluster_id": w} for w in pending])
    return sweep


def test_reconcile_all_previews_then_releases_an_orphaned_claim(tmp_path, monkeypatch):
    token, path = _claimed(tmp_path, GONE)
    sweep = _sweep(monkeypatch)

    (preview,) = model_cache.reconcile_claims(cache_dir=str(tmp_path), cluster="lab")
    assert (preview.claim, preview.workload, preview.status, preview.action) == (token, GONE, "orphaned", "would_release")
    assert token in json.loads(path.read_text())["claims"]
    assert sweep.call_args.kwargs["cluster"] == "lab"

    (applied,) = model_cache.reconcile_claims(cache_dir=str(tmp_path), dry_run=False)
    assert applied.action == "released"
    assert json.loads(path.read_text())["claims"] == {}


@pytest.mark.parametrize(
    "sweep,created,status",
    [
        ({"running": [LIVE]}, 1.0, "running"),
        ({"errors": {"localhost": "unreachable"}}, 1.0, "unverified"),
        ({"pending": [LIVE]}, 1.0, "preparing"),
        ({}, None, "recent"),
        ({"jobs": [(LIVE, "local")]}, 1.0, "unverified"),
        ({"jobs": [(GONE, "local")]}, 1.0, "unverified"),
        ({"jobs": [(LIVE, "docker")]}, 1.0, "orphaned"),
    ],
)
def test_reconcile_all_keeps_claims_it_cannot_prove_orphaned(tmp_path, monkeypatch, sweep, created, status):
    import time

    token, path = _claimed(tmp_path, LIVE, created=time.time() if created is None else created)
    _sweep(monkeypatch, **sweep)
    (result,) = model_cache.reconcile_claims(cache_dir=str(tmp_path), dry_run=False)
    if status == "orphaned":
        assert result.action == "released"
        return
    assert (result.status, result.action) == (status, None)
    assert token in json.loads(path.read_text())["claims"]


@pytest.mark.parametrize(
    "workload,patch,status",
    [
        ("othername_" + "a" * 16 + "_" + "b" * 12, {}, "foreign"),
        ("not-a-workload-id", {}, "foreign"),
        (GONE, {"created": "yesterday"}, "unverified"),
        (GONE, {"hosts": "localhost"}, "unverified"),
    ],
)
def test_reconcile_all_keeps_foreign_and_malformed_claims(tmp_path, monkeypatch, workload, patch, status):
    token, path = _claimed(tmp_path, workload)
    record = json.loads(path.read_text())
    record["claims"][token].update(patch)
    path.write_text(json.dumps(record))
    _sweep(monkeypatch)
    (result,) = model_cache.reconcile_claims(cache_dir=str(tmp_path), dry_run=False)
    assert (result.status, result.action) == (status, None)
    assert token in json.loads(path.read_text())["claims"]


@pytest.mark.parametrize(
    "gated,status",
    [
        ({"local": ("executor.local", "host")}, "unverified"),
        # Another substrate's executor cannot hold a workload on these hosts.
        ({"k8s": ("executor.k8s", "k8s")}, "orphaned"),
    ],
)
def test_reconcile_all_keeps_claims_an_executor_turned_off_here_could_hold(tmp_path, monkeypatch, gated, status):
    # Another controller may have launched it under that executor, leaving no
    # job record here, so the docker sweep finding nothing proves nothing.
    token, path = _claimed(tmp_path, GONE)
    _sweep(monkeypatch, gated=gated)
    (result,) = model_cache.reconcile_claims(cache_dir=str(tmp_path), dry_run=False)
    assert result.status == status
    if status == "unverified":
        assert "enable executor.local" in result.reason
        assert token in json.loads(path.read_text())["claims"]


def _record_executor(path, token, executor):
    record = json.loads(path.read_text())
    record["claims"][token]["executor"] = executor
    path.write_text(json.dumps(record))


def test_reconcile_all_requires_the_claims_recorded_executor_in_the_sweep(tmp_path, monkeypatch):
    token, path = _claimed(tmp_path, GONE)
    _record_executor(path, token, "local")
    _sweep(monkeypatch)
    (result,) = model_cache.reconcile_claims(cache_dir=str(tmp_path), dry_run=False)
    assert (result.status, result.action) == ("unverified", None)
    assert "local" in result.reason
    assert token in json.loads(path.read_text())["claims"]


def test_reconcile_all_releases_a_recorded_swept_executor_despite_gated_peers(tmp_path, monkeypatch):
    # The claim says where its workload ran, so an unrelated gated executor
    # no longer has to block it.
    token, path = _claimed(tmp_path, GONE)
    _record_executor(path, token, "docker")
    _sweep(monkeypatch, gated={"local": ("executor.local", "host")})
    (result,) = model_cache.reconcile_claims(cache_dir=str(tmp_path), dry_run=False)
    assert (result.status, result.action) == ("orphaned", "released")


@pytest.mark.parametrize(("jobs", "status"), [([(GONE, "docker")], "orphaned"), ([(GONE, None)], "unverified"), ([], "unverified")])
def test_reconcile_all_lets_a_job_record_place_a_legacy_claim(tmp_path, monkeypatch, jobs, status):
    # Claims written before they recorded an executor would otherwise all stay
    # unverified on stable, where executor.local is gated off by default.
    token, path = _claimed(tmp_path, GONE)
    _sweep(monkeypatch, jobs=jobs, gated={"local": ("executor.local", "host")})
    (result,) = model_cache.reconcile_claims(cache_dir=str(tmp_path), dry_run=False)
    assert result.status == status
    assert (token in json.loads(path.read_text())["claims"]) is (status != "orphaned")


def test_launch_claims_record_their_executor(tmp_path):
    from sparkrun.models.usage import set_workload_executor

    manifest, _, path = tracked(tmp_path)
    with usage_operation():
        begin_workload(GONE, {})
        set_workload_executor("local")
        with ModelHostIO() as io:
            record_use(io, "localhost", str(tmp_path), manifest)
        token = workload_claims()[0]["claim"]
        activate_workload()
    assert json.loads(path.read_text())["claims"][token]["executor"] == "local"


def test_gated_off_executor_registration_is_recorded(monkeypatch):
    from sparkrun.orchestration.executors import _base

    class Gated(_base.Executor):
        executor_name = "gated-test"
        required_feature_flag = "executor.gated-test"

    monkeypatch.setattr(_base, "_GATED_OFF_EXECUTORS", {})
    monkeypatch.setattr(Gated, "is_multi_extension", lambda self, v: False)
    Gated.on_registration(Gated.__new__(Gated), None)
    assert _base.gated_off_executors() == {"gated-test": ("executor.gated-test", "host")}
    monkeypatch.setattr(Gated, "is_multi_extension", lambda self, v: True)
    Gated.on_registration(Gated.__new__(Gated), None)
    assert _base.gated_off_executors() == {}


def test_reconcile_all_skips_the_status_sweep_without_claims(tmp_path, monkeypatch):
    tracked(tmp_path)
    sweep = _sweep(monkeypatch)
    assert model_cache.reconcile_claims(cache_dir=str(tmp_path)) == ()
    sweep.assert_not_called()


def test_cli_reconcile_all_forwards_cluster_scope_and_age(monkeypatch, tmp_path):
    from click.testing import CliRunner
    from sparkrun.cli import _model_cache as cli

    monkeypatch.setattr(cli, "_get_context", lambda ctx: cli_context())
    monkeypatch.setattr(cli, "_resolve_targets", lambda sctx, **kw: (["h1"], "/hf", "Cluster: lab — 1 host(s)", "lab"))
    reconcile = Mock(return_value=())
    monkeypatch.setattr(model_cache, "reconcile_claims", reconcile)
    result = CliRunner().invoke(cli.model_cache, ["reconcile", "org/model", "--apply", "--min-age-minutes", "5"])
    assert result.exit_code == 0, result.output
    assert "No outstanding workload claims." in result.output
    kwargs = reconcile.call_args.kwargs
    assert (kwargs["cluster"], kwargs["repo"], kwargs["dry_run"], kwargs["min_age_seconds"], kwargs["hosts"]) == (
        "lab",
        "org/model",
        False,
        300,
        ("h1",),
    )


@pytest.mark.parametrize(
    "args,message",
    [
        ([], "Name a MODEL to reconcile, or pass --all."),
        (["--all", "org/model"], "Name a MODEL to reconcile, or pass --all."),
        (["org/model", "--revision", SHA], "apply only with --claim"),
        (["--all", "--endpoint", "https://mirror.example"], "apply only with --claim"),
        (["--all", "--claim", "x"], "--claim releases one claim"),
        (["org/model", "--revision", SHA, "--claim", "x", "--confirm-inactive", "--apply"], "--claim releases one claim"),
        (["org/model", "--claim", "x", "--confirm-inactive"], "--claim needs MODEL and --revision."),
    ],
)
def test_cli_reconcile_rejects_mixed_modes(args, message):
    from click.testing import CliRunner
    from sparkrun.cli import _model_cache as cli

    result = CliRunner().invoke(cli.model_cache, ["reconcile", *args])
    assert result.exit_code != 0
    assert message in result.output


def test_pin_and_reconcile_accept_a_cached_ref(tmp_path):
    tracked(tmp_path)
    refs = tmp_path / "hub" / "models--org--model" / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    (refs / "main").write_text(SHA)
    (report,) = model_cache.pin_revision(ModelArtifactRequest("org/model", "main"), cache_dir=str(tmp_path))
    assert [(e.revision, e.pinned) for e in report.entries if e.action] == [(SHA, True)]

    with pytest.raises(ModelArtifactError, match="no cached ref 'dev'"):
        model_cache.pin_revision(ModelArtifactRequest("org/model", "dev"), cache_dir=str(tmp_path))
    with pytest.raises(ModelArtifactError, match="ref name"):
        model_cache.pin_revision(ModelArtifactRequest("org/model", "../main"), cache_dir=str(tmp_path))

    token, _ = _claimed(tmp_path, GONE)
    model_cache.reconcile_claim(ModelArtifactRequest("org/model", "main"), cache_dir=str(tmp_path), claim=token, confirm_inactive=True)
    assert model_cache.inventory_cache(cache_dir=str(tmp_path))[0].entries[0].claims == {}


@pytest.mark.parametrize(
    "value,expected",
    [("500G", 500 * 2**30), ("500gib", 500 * 2**30), ("500GB", 500 * 10**9), ("1.5T", 3 * 2**39), ("1024", 1024), (0, 0), (1e3, 1000)],
)
def test_byte_sizes_parse_case_insensitively(value, expected):
    from sparkrun.utils.data import parse_byte_size

    assert parse_byte_size(value) == expected


@pytest.mark.parametrize("value", ["1.5", "12x", "GiB", "-1G", -1, True, 1.5])
def test_byte_sizes_reject_ambiguous_or_fractional_input(value):
    from sparkrun.utils.data import parse_byte_size

    with pytest.raises(ValueError):
        parse_byte_size(value)


def test_cli_prune_takes_sizes_and_echoes_the_target(monkeypatch, tmp_path):
    from click.testing import CliRunner
    from sparkrun.cli import _model_cache as cli

    monkeypatch.setattr(cli, "_get_context", lambda ctx: cli_context())
    prune = Mock(return_value=())
    monkeypatch.setattr(model_cache, "prune_cache", prune)
    result = CliRunner().invoke(
        cli.model_cache, ["prune", "--cache-dir", str(tmp_path), "--max-bytes", "500G", "--min-free-bytes", "100GB"]
    )
    assert result.exit_code == 0, result.output
    assert (prune.call_args.kwargs["max_bytes"], prune.call_args.kwargs["min_free_bytes"]) == (500 * 2**30, 100 * 10**9)
    assert "Target: managed snapshots <= 500.0 GiB; free space >= 93.1 GiB" in result.output
    bad = CliRunner().invoke(cli.model_cache, ["prune", "--cache-dir", str(tmp_path), "--max-bytes", "lots"])
    assert bad.exit_code == 2 and "is not a byte size" in bad.output


def test_auto_prune_config_accepts_sizes(monkeypatch):
    from sparkrun.models import cache

    calls = []
    monkeypatch.setattr(cache, "host_cache", lambda io, host, root, **kw: calls.append(kw) or {"reclaimed_bytes": 0, "target_met": True})
    policy = {"model_cache_prune": {"enabled": True, "max_bytes": "1T"}}
    cache.auto_prune(
        None, [("h1", "/hf")], config=SimpleNamespace(get=lambda key, default=None: policy.get(key, default)), exclude_repos=()
    )
    assert calls[0]["max_bytes"] == 2**40 and calls[0]["min_free_bytes"] is None


@pytest.mark.parametrize("value", [0, "0", "0G"])
def test_auto_prune_config_rejects_a_zero_budget(monkeypatch, value):
    from sparkrun.models import cache

    host_cache = Mock()
    monkeypatch.setattr(cache, "host_cache", host_cache)
    policy = {"model_cache_prune": {"enabled": True, "max_bytes": value}}
    with pytest.raises(ModelArtifactError, match="max_bytes must be positive"):
        cache.auto_prune(
            None, [("h1", "/hf")], config=SimpleNamespace(get=lambda key, default=None: policy.get(key, default)), exclude_repos=()
        )
    host_cache.assert_not_called()


@pytest.mark.parametrize("action,force", [("inventory", False), ("delete", True)])
def test_claimed_revisions_point_at_reconcile(tmp_path, action, force):
    from sparkrun.utils.cli_formatters import format_model_cache_reports

    _claimed(tmp_path, GONE)
    text = format_model_cache_reports(model_cache.inventory_cache(cache_dir=str(tmp_path)), action=action)
    assert "model-cache reconcile org/model` (a preview; add --apply)" in text
    assert ("--force" in text) == force
