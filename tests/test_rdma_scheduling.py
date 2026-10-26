"""Host reservations, concurrent progress, and deterministic RDMA reports."""

from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from itertools import combinations
from threading import Barrier, Event, Lock
from unittest import mock

import pytest

from sparkrun.api.setup._rdma import STATUS_OK, PairTestResult, _run_pairs
from sparkrun.orchestration.rdma import RdmaPair, pair_test_rounds


def _pairs(*keys):
    return [RdmaPair(*key) for key in keys]


@pytest.mark.parametrize("size", range(2, 13))
def test_complete_switch_schedule_covers_every_pair_in_minimal_rounds(size):
    hosts = ["h%d" % n for n in range(size)]
    pairs = _pairs(*combinations(hosts, 2))
    rounds = pair_test_rounds(pairs)
    assert len(rounds) == (size - 1 if size % 2 == 0 else size)
    flattened = [pair for batch in rounds for pair in batch]
    assert len(flattened) == len(pairs)
    assert {p.key for p in flattened} == {p.key for p in pairs}
    for batch in rounds:
        members = [host for pair in batch for host in pair.key]
        assert len(members) == len(set(members))


def test_four_hosts_starts_with_the_users_two_independent_pairs():
    pairs = _pairs(*combinations((".13", ".14", ".16", ".17"), 2))
    assert {p.key for p in pair_test_rounds(pairs)[0]} == {(".13", ".14"), (".16", ".17")}


def test_sparse_graph_preserves_only_actual_candidates():
    pairs = _pairs(("a", "b"), ("b", "c"), ("c", "d"), ("a", "d"))
    rounds = pair_test_rounds(pairs)
    assert len(rounds) == 2
    assert {p.key for batch in rounds for p in batch} == {p.key for p in pairs}
    assert pair_test_rounds([]) == []


def test_four_host_work_really_overlaps_without_shared_hosts_and_preserves_result_order():
    pairs = _pairs(*combinations(("a", "b", "c", "d"), 2))
    barrier = Barrier(2, timeout=3)
    lock = Lock()
    active = set()
    peak = 0
    seen = []

    def run(pair):
        nonlocal peak
        with lock:
            assert active.isdisjoint(pair.key), "a host was assigned two simultaneous tests"
            active.update(pair.key)
            peak = max(peak, len(active) // 2)
            seen.append(pair.key)
        # Both disjoint pairs must actually start before either can finish.
        barrier.wait()
        with lock:
            active.difference_update(pair.key)
        return PairTestResult(pair, status=STATUS_OK)

    results = _run_pairs(pairs, run, 2)
    assert peak == 2
    assert not active
    assert len(seen) == len(set(seen)) == 6
    assert [r.pair for r in results] == pairs


def test_freed_hosts_start_new_work_while_an_unrelated_pair_is_still_running():
    pairs = _pairs(("a", "b"), ("c", "d"), ("c", "e"))
    later_started = Event()
    finished = []

    def run(pair):
        if pair.key == ("a", "b"):
            assert later_started.wait(3), "scheduler waited for an unrelated slow pair"
        if pair.key == ("c", "e"):
            assert ("c", "d") in finished
            later_started.set()
        finished.append(pair.key)
        return PairTestResult(pair, status=STATUS_OK)

    results = _run_pairs(pairs, run, 2)
    assert later_started.is_set()
    assert [r.pair for r in results] == pairs


def test_pair_limit_one_is_serial_in_original_order():
    pairs = _pairs(*combinations(("a", "b", "c", "d"), 2))
    seen = []

    def run(pair):
        seen.append(pair)
        return PairTestResult(pair, status=STATUS_OK)

    with mock.patch("sparkrun.api.setup._rdma.ThreadPoolExecutor") as pool:
        assert [r.pair for r in _run_pairs(pairs, run, 1)] == pairs
    assert seen == pairs
    pool.assert_not_called()


def test_shared_host_pairs_never_overlap_even_with_a_large_limit():
    pairs = _pairs(("a", "b"), ("a", "c"), ("a", "d"))
    gate = Lock()
    seen = []

    def run(pair):
        assert gate.acquire(blocking=False), "overlapping tests on host a"
        try:
            seen.append(pair)
            return PairTestResult(pair, status=STATUS_OK)
        finally:
            gate.release()

    assert [r.pair for r in _run_pairs(pairs, run, 3)] == pairs
    assert len(seen) == len(pairs)


def test_scheduler_preserves_the_callers_context_in_workers():
    value = ContextVar("rdma-test-context", default="absent")
    token = value.set("caller")
    try:

        def run(pair):
            assert value.get() == "caller"
            value.set("worker")
            return PairTestResult(pair, status=STATUS_OK)

        _run_pairs(_pairs(("a", "b"), ("c", "d")), run, 2)
        assert value.get() == "caller"
    finally:
        value.reset(token)


def test_worker_error_drains_active_work_before_allowing_container_cleanup():
    pairs = _pairs(("a", "b"), ("c", "d"))
    peer_started, failure_raised, release_peer, peer_finished = Event(), Event(), Event(), Event()

    def run(pair):
        if pair.key == ("a", "b"):
            assert peer_started.wait(3)
            failure_raised.set()
            raise RuntimeError("pair failed")
        peer_started.set()
        assert release_peer.wait(3)
        peer_finished.set()
        return PairTestResult(pair, status=STATUS_OK)

    with ThreadPoolExecutor(max_workers=1) as caller:
        job = caller.submit(_run_pairs, pairs, run, 2)
        try:
            assert failure_raised.wait(3)
            assert not job.done(), "active peer would outlive runner cleanup"
        finally:
            release_peer.set()
        with pytest.raises(RuntimeError, match="pair failed"):
            job.result(timeout=3)
    assert peer_finished.is_set()


@pytest.mark.parametrize("size", range(2, 9))
def test_quick_selection_is_a_spanning_chain_on_each_subnet(size):
    from sparkrun.orchestration.rdma import derive_link_pairs, select_quick_pairs
    from test_rdma_coverage import _switch

    # Non-alphabetical caller order must be retained.
    hosts = ["h%d" % i for i in reversed(range(size))]
    dets = _switch(hosts)
    all_pairs = derive_link_pairs(dets, hosts)
    selected = select_quick_pairs(all_pairs)
    assert [p.key for p in selected] == list(zip(hosts[:-1], hosts[1:], strict=True))
    for subnet in ("10.0.0.0/24", "10.1.0.0/24"):
        links = [link for p in selected for link in p.links if link.subnet == subnet]
        assert len(links) == size - 1
        assert {h for link in links for h in (link.host_a, link.host_b)} == set(hosts)
    assert select_quick_pairs([]) == []


def test_quick_selection_handles_different_members_on_each_subnet():
    from sparkrun.orchestration.rdma import derive_link_pairs, select_quick_pairs
    from test_rdma_coverage import _switch

    dets = _switch()
    dets["h2"].interfaces.pop()
    selected = select_quick_pairs(derive_link_pairs(dets, list(dets)))
    actual = {(link.host_a, link.host_b, link.subnet) for pair in selected for link in pair.links}
    assert actual == {
        ("h1", "h2", "10.0.0.0/24"),
        ("h2", "h3", "10.0.0.0/24"),
        ("h3", "h4", "10.0.0.0/24"),
        ("h1", "h3", "10.1.0.0/24"),
        ("h3", "h4", "10.1.0.0/24"),
    }


def test_pair_completion_is_counted_as_pairs_finish(caplog):
    caplog.set_level(25, logger="sparkrun.progress")
    pairs = _pairs(("a", "b"), ("c", "d"), ("a", "c"))

    def run(pair):
        return PairTestResult(pair=pair, status="ok")

    _run_pairs(pairs, run, 2)
    counted = [m for m in caplog.messages if m.startswith("[")]
    assert len(counted) == 3
    assert counted[-1].startswith("[3/3]")
    assert all(m.endswith(": OK") for m in counted)
