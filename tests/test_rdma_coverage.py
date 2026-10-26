"""Coverage and direction regressions for the RDMA test planner and runner."""

from dataclasses import replace
from itertools import combinations
from unittest import mock

import pytest

from sparkrun.api.setup._rdma import (
    STATUS_FAIL,
    STATUS_OK,
    STATUS_SKIP,
    STATUS_WARN,
    RdmaTestError,
    rdma_test,
)
from sparkrun.core.cluster_manager import ClusterDefinition
from sparkrun.orchestration.rdma import RdmaDevice, RdmaHostFacts, derive_link_pairs
from test_rdma import REAL_BW_OUTPUT, REAL_LAT_OUTPUT, _Sctx, _det, _iface


def _switch(hosts=("h1", "h2", "h3", "h4"), rails=2):
    return {
        h: _det(h, [_iface("e%d" % r, "10.%d.0.%d" % (r, i + 1), "10.%d.0.0/24" % r, "hca%d" % r) for r in range(rails)])
        for i, h in enumerate(hosts)
    }


def _facts(detections):
    return {
        h: RdmaHostFacts(
            host=h,
            complete=True,
            has_perftest=True,
            devices=[RdmaDevice(name=i.hca, state="4: ACTIVE", rate="100 Gb/sec") for i in det.interfaces],
        )
        for h, det in detections.items()
    }


def _cluster(dets, topology, **kw):
    return ClusterDefinition(name="test", hosts=list(dets), topology=topology, **kw)


def _success(server, server_cmds, client, client_cmds, *args, **kwargs):
    return {i: (REAL_LAT_OUTPUT if cmd.startswith("ib_write_lat") else REAL_BW_OUTPUT, 0) for i, cmd in enumerate(client_cmds)}


def _run(dets, *, facts=None, cluster=None, round_fn=_success, suite="perftest", **options):
    with (
        mock.patch("sparkrun.api.setup._rdma._run_probe", return_value=facts if facts is not None else _facts(dets)),
        mock.patch("sparkrun.orchestration.networking.detect_cx7_for_hosts", return_value=dets),
        mock.patch("sparkrun.api.setup._rdma._run_round", side_effect=round_fn) as rounds,
        mock.patch("sparkrun.api.setup._rdma._prepare_container") as prepare,
        mock.patch("sparkrun.api.setup._rdma._run_nccl") as nccl,
        mock.patch("sparkrun.api.setup._rdma._stop_containers"),
    ):
        report = rdma_test(_Sctx(), list(dets), {}, cluster=cluster, suite=suite, **options)
    return report, rounds, prepare, nccl


def _details(report):
    return "; ".join(issue.detail for issue in report.coverage.issues)


def test_full_four_hosts_two_switch_subnets_complete_bidirectional_coverage(caplog):
    dets = _switch()
    caplog.set_level(25, logger="sparkrun.progress")
    report, rounds, prepare, _ = _run(dets, cluster=_cluster(dets, "switch"), full=True)
    assert report.full
    assert not report.has_failure
    assert not any("quick sample" in msg or "Use --full" in msg for msg in caplog.messages)
    assert report.coverage.status == STATUS_OK
    assert report.coverage.pair_count == report.coverage.expected_pair_count == 6
    assert report.coverage.path_count == report.coverage.expected_path_count == 12
    assert report.coverage.components == (tuple(dets),)
    assert {p.pair.key for p in report.pairs} == set(combinations(dets, 2))
    for pair in report.pairs:
        assert len(pair.links) == 4
        assert len(pair.aggregates) == 2
        assert {(r.link.host_a, r.link.host_b) for r in pair.links} == {pair.pair.key, pair.pair.key[::-1]}
        assert {(r.host_a, r.host_b) for r in pair.aggregates} == {pair.pair.key, pair.pair.key[::-1]}
    assert report.ok_count == 36  # 24 directed path checks + 12 aggregates; no pair rollups
    assert rounds.call_count == 60  # latency + bandwidth per direction, then aggregates
    prepare.assert_not_called()


def test_each_command_binds_its_own_endpoint_in_both_directions():
    dets = _switch(("h1", "h2"))
    _, rounds, _, _ = _run(dets)
    for call in rounds.call_args_list:
        server, servers, client, clients, _ = call.args
        for commands, host in ((servers, server), (clients, client)):
            for cmd in commands:
                iface = next(i for i in dets[host].interfaces if "-d %s " % i.hca in cmd)
                assert "--bind_source_ip %s " % iface.ip in cmd


@pytest.mark.parametrize("kind", ["isolated", "disconnected", "missing_rail", "incomplete_probe", "failed_probe", "missing_probe"])
def test_coverage_failure_stops_before_any_transfers_or_containers(kind):
    dets = _switch()
    facts = _facts(dets)
    cluster = _cluster(dets, "switch") if kind == "missing_rail" else None
    if kind == "isolated":
        dets["h4"].interfaces = []
    elif kind == "disconnected":
        dets.update(_switch(("h3", "h4")))
        for host in ("h3", "h4"):
            dets[host].interfaces = [
                replace(i, ip=i.ip.replace("10.", "11."), subnet=i.subnet.replace("10.", "11.")) for i in dets[host].interfaces
            ]
    elif kind == "missing_rail":
        dets["h4"].interfaces.pop()
    elif kind == "incomplete_probe":
        facts["h4"].complete = False
    elif kind == "failed_probe":
        facts["h4"].error = "SSH timeout"
        facts["h4"].has_perftest = False
    else:
        facts.pop("h4")
    report, rounds, prepare, nccl = _run(dets, facts=facts, cluster=cluster, suite="all")
    assert report.has_failure
    assert report.coverage.status == STATUS_FAIL
    assert report.ok_count == report.fail_count == 0  # coverage is separate from measurements
    assert report.pairs == ()
    rounds.assert_not_called()
    prepare.assert_not_called()
    nccl.assert_not_called()
    if kind == "missing_rail":
        assert "10.1.0.0/24: switch subnet is missing hosts: h4" in _details(report)
        assert report.coverage.path_count == 9
        assert report.coverage.expected_path_count == 12
    if kind in ("isolated", "disconnected"):
        assert len(report.coverage.components) == 2


def test_unknown_topology_only_claims_observed_subnet_coverage():
    dets = _switch()
    dets["h4"].interfaces.pop()
    report, _, _, _ = _run(dets)
    assert not report.has_failure
    assert report.coverage.topology is None
    assert report.coverage.expected_path_count is None
    assert report.coverage.path_count == 9


def _ring():
    dets = {h: _det(h, []) for h in ("h1", "h2", "h3", "h4")}
    for n, (a, b) in enumerate((("h1", "h2"), ("h2", "h3"), ("h3", "h4"), ("h4", "h1"))):
        for end, host in enumerate((a, b), 1):
            dets[host].interfaces.append(_iface("e%d" % n, "10.%d.0.%d" % (n, end), "10.%d.0.0/24" % n, "hca%d" % n))
    return dets


def test_four_host_ring_tests_neighbors_without_inventing_diagonal_paths():
    dets = _ring()
    report, rounds, _, _ = _run(dets, cluster=_cluster(dets, "ring"))
    assert not report.has_failure
    assert {p.pair.key for p in report.pairs} == {("h1", "h2"), ("h2", "h3"), ("h3", "h4"), ("h1", "h4")}
    assert report.ok_count == 8
    assert rounds.call_count == 16
    assert all(not p.aggregates for p in report.pairs)


def test_open_ring_fails_even_though_every_host_still_has_a_neighbor():
    dets = _ring()
    for host in ("h1", "h4"):
        dets[host].interfaces = [i for i in dets[host].interfaces if i.hca != "hca3"]
    report, rounds, _, _ = _run(dets, cluster=_cluster(dets, "ring"))
    assert report.has_failure
    assert report.coverage.uncovered_hosts == ()
    assert "ring requires two distinct neighbors" in _details(report)
    rounds.assert_not_called()


def test_ring_subset_does_not_require_closing_the_whole_ring():
    dets = _ring()
    cluster = _cluster(dets, "ring")
    report, _, _, _ = _run({h: dets[h] for h in ("h1", "h2")}, cluster=cluster)
    assert not report.has_failure
    assert report.coverage.status == STATUS_WARN  # addressed paths to unselected peers remain untested
    assert report.coverage.pair_count == 1


def test_fabric_selection_excludes_other_addressed_networks():
    dets = _switch()
    for det in dets.values():
        det.interfaces.append(_iface("other", "12.0.0.1", "12.0.0.0/24", "missing_hca"))
    report, _, _, _ = _run(dets, cluster=_cluster(dets, "switch", fabric_interfaces=["e*"]))
    assert not report.has_failure
    assert report.coverage.path_count == 12
    assert set(report.coverage.subnets) == {"10.0.0.0/24", "10.1.0.0/24"}


def test_no_interfaces_matching_selection_is_a_coverage_failure():
    dets = _switch()
    report, _, _, _ = _run(dets, cluster=_cluster(dets, "switch", fabric_interfaces=["missing*"]))
    assert report.has_failure
    assert "no configured RDMA interfaces matching fabric_interfaces" in _details(report)


def test_duplicate_subnet_interfaces_cannot_choose_an_arbitrary_endpoint():
    dets = _switch()
    dets["h1"].interfaces.append(_iface("duplicate", "10.0.0.99", "10.0.0.0/24", "hca_other"))
    with pytest.raises(ValueError, match="multiple interfaces"):
        derive_link_pairs(dets, list(dets))
    report, rounds, _, _ = _run(dets)
    assert report.has_failure
    assert "multiple interfaces" in _details(report)
    rounds.assert_not_called()


def test_duplicate_addresses_cannot_test_the_wrong_host():
    dets = _switch()
    dets["h2"].interfaces[0].ip = dets["h1"].interfaces[0].ip
    report, _, _, _ = _run(dets)
    assert report.has_failure
    assert "Duplicate RDMA address" in _details(report)


def test_unused_unaddressed_ports_are_not_required_but_configured_inactive_devices_fail():
    dets = _switch(("h1", "h2"))
    facts = _facts(dets)
    dets["h1"].interfaces.append(_iface("unused", "", "", "down"))
    facts["h1"].devices.append(RdmaDevice(name="down", state="DOWN"))
    report, _, _, _ = _run(dets, facts=facts)
    assert not report.has_failure
    facts["h1"].devices[0].state = "DOWN"
    report, _, _, _ = _run(dets, facts=facts)
    assert report.has_failure
    assert "configured RDMA device is missing or inactive" in _details(report)


def test_reverse_failure_is_visible_even_when_forward_results_pass():
    dets = _switch(("h1", "h2"), rails=1)

    def round_fn(server, servers, client, clients, *args, **kwargs):
        runs = _success(server, servers, client, clients)
        return {i: (out, 1 if client == "h2" else rc) for i, (out, rc) in runs.items()}

    report, _, _, _ = _run(dets, round_fn=round_fn)
    pair = report.pairs[0]
    assert [r.status for r in pair.links] == [STATUS_OK, STATUS_FAIL]
    assert pair.links[1].bandwidth is not None  # printed table does not override exit status
    assert "rc=1" in pair.links[1].detail
    assert pair.status == STATUS_FAIL
    assert report.has_failure
    assert report.fail_count == report.ok_count == 1


@pytest.mark.parametrize("kind", ["partial", "nonzero", "no_results", "slow"])
def test_aggregate_gets_its_own_verdict_and_changes_the_pair_rollup(kind):
    dets = _switch(("h1", "h2"))

    def round_fn(server, servers, client, clients, *args, **kwargs):
        runs = _success(server, servers, client, clients)
        if len(clients) == 1:
            return runs
        if kind == "partial":
            runs.pop(1)
        elif kind == "nonzero":
            runs[1] = (REAL_BW_OUTPUT, 124)
        elif kind == "no_results":
            return {}
        else:
            return {i: (REAL_BW_OUTPUT.replace("111.71", "10.00"), 0) for i in runs}
        return runs

    report, _, _, _ = _run(dets, round_fn=round_fn)
    pair = report.pairs[0]
    expected_status = STATUS_WARN if kind == "slow" else STATUS_FAIL
    assert all(r.status == STATUS_OK for r in pair.links)
    assert all(a.status == expected_status for a in pair.aggregates)
    assert pair.status == expected_status
    assert report.ok_count == 4
    if kind == "slow":
        assert report.warn_count == 2
        assert not report.has_failure
    else:
        assert report.fail_count == 2
        assert report.has_failure
        assert all(a.bandwidth_gbps is None for a in pair.aggregates)


def test_aggregate_expectation_uses_the_slower_endpoint_in_both_directions():
    dets = _switch(("h1", "h2"))
    facts = _facts(dets)
    for dev in facts["h1"].devices:
        dev.rate = "200 Gb/sec"
    report, _, _, _ = _run(dets, facts=facts)
    assert [a.expected_gbps for a in report.pairs[0].aggregates] == [200, 200]


def test_dry_run_does_not_claim_live_discovery_or_missing_hosts():
    with (
        mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel") as probe,
        mock.patch("sparkrun.orchestration.ssh.run_remote_script") as run,
    ):
        report = rdma_test(_Sctx(), ["h1", "h2"], {}, dry_run=True)
    assert report.coverage.status == STATUS_SKIP
    assert not report.coverage.discovered
    assert report.coverage.uncovered_hosts == ()
    assert not report.has_failure
    assert all(call.kwargs["dry_run"] for call in probe.call_args_list)
    run.assert_not_called()


@pytest.mark.parametrize(
    "hosts,kwargs", [(["h1"], {}), (["h1", "h1"], {}), (["h1", "h2"], {"duration": 0}), (["h1", "h2"], {"queue_pairs": 0})]
)
def test_invalid_run_arguments_fail_before_discovery(hosts, kwargs):
    with mock.patch("sparkrun.api.setup._rdma._run_probe") as probe:
        with pytest.raises(RdmaTestError):
            rdma_test(_Sctx(), hosts, {}, **kwargs)
    probe.assert_not_called()


def test_transport_exit_failure_invalidates_success_frames():
    from sparkrun.api.setup._rdma import _run_commands
    from sparkrun.orchestration.ssh import RemoteResult

    # A completed subcommand can print a frame before the outer session fails.
    with (
        mock.patch("sparkrun.orchestration.ssh.run_remote_script", return_value=RemoteResult("h1", 255, "framed", "connection lost")),
        mock.patch("sparkrun.api.setup._rdma.parse_framed_runs", return_value={0: (REAL_BW_OUTPUT, 0)}),
    ):
        runs = _run_commands("h1", ["ib_write_bw"], {}, container=None, timeout=10)
    assert runs[0][1] == 255
    assert "connection lost" in runs[0][0]


@pytest.mark.parametrize("server_result", [{0: ("server error", 1)}, {}, RuntimeError("lost server")])
def test_server_failure_cannot_be_hidden_by_a_printed_client_measurement(server_result):
    from sparkrun.api.setup._rdma import _run_round

    def commands(host, *args, **kwargs):
        if host == "server":
            if isinstance(server_result, Exception):
                raise server_result
            return server_result
        return {0: (REAL_BW_OUTPUT, 0)}

    with mock.patch("sparkrun.api.setup._rdma._run_commands", side_effect=commands):
        runs = _run_round("server", ["server command"], "client", ["client command"], {}, container=None, timeout=10)
    assert runs[0][1] != 0
    assert "server failed" in runs[0][0]


def test_successful_round_preserves_client_measurement():
    from sparkrun.api.setup._rdma import _run_round

    with mock.patch("sparkrun.api.setup._rdma._run_commands", return_value={0: (REAL_BW_OUTPUT, 0)}):
        assert _run_round("s", ["s"], "c", ["c"], {}, container=None, timeout=10) == {0: (REAL_BW_OUTPUT, 0)}


def test_json_and_text_expose_both_directions_and_aggregate_verdicts(capsys):
    from sparkrun.cli._setup._rdma import _render, _report_to_dict

    report, _, _, _ = _run(_switch(("h1", "h2")))
    data = _report_to_dict(report)
    assert data["coverage"]["path_count"] == 2
    assert data["ok"] == 6
    assert data["has_failure"] is False
    pair = data["pairs"][0]
    assert {(link["host_a"], link["host_b"]) for link in pair["links"]} == {("h1", "h2"), ("h2", "h1")}
    assert len(pair["aggregates"]) == 2
    assert all(a["status"] == STATUS_OK for a in pair["aggregates"])
    _render(report)
    text = capsys.readouterr().out
    assert "h1:hca0 -> h2:hca0" in text
    assert "h2:hca0 -> h1:hca0" in text
    assert "[OK]   h1 -> h2 aggregate" in text
    assert "[OK]   h2 -> h1 aggregate" in text


def test_nccl_nonzero_exit_cannot_pass_with_a_printed_bandwidth():
    from sparkrun.api.setup._rdma import _run_nccl
    from sparkrun.orchestration.ssh import RemoteResult
    from test_rdma import REAL_NCCL_OUTPUT

    dets = _switch(("h1", "h2"))
    with (
        mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", return_value=[RemoteResult(h, 0, "", "") for h in dets]),
        mock.patch("sparkrun.api.setup._rdma._run_commands", return_value={0: (REAL_NCCL_OUTPUT, 1)}),
        mock.patch("sparkrun.orchestration.primitives.try_clear_page_cache"),
    ):
        result = _run_nccl(
            list(dets), derive_link_pairs(dets, list(dets)), _facts(dets), dets, {}, msg_size="16G", binary="all_gather_perf", timeout=60
        )
    assert result.status == STATUS_FAIL
    assert "rc=1" in result.detail


@pytest.mark.parametrize("host_count,requested,expected", [(4, None, 2), (10, None, 4), (12, 6, 6), (4, 1, 1), (4, 99, 2)])
def test_concurrency_defaults_to_four_and_is_bounded_by_host_capacity(host_count, requested, expected):
    from sparkrun.cli._setup._rdma import _report_to_dict

    kwargs = {} if requested is None else {"parallel_pairs": requested}
    with mock.patch("sparkrun.orchestration.networking.detect_cx7_for_hosts", return_value={}):
        report = rdma_test(_Sctx(), ["h%d" % i for i in range(host_count)], {}, dry_run=True, **kwargs)
    assert report.parallel_pairs == expected
    assert _report_to_dict(report)["parallel_pairs"] == expected


def test_invalid_concurrency_fails_before_remote_discovery():
    with mock.patch("sparkrun.api.setup._rdma._run_probe") as probe:
        with pytest.raises(RdmaTestError, match="parallel_pairs must be positive"):
            rdma_test(_Sctx(), ["h1", "h2"], {}, parallel_pairs=0)
    probe.assert_not_called()


def test_default_samples_a_chain_and_discloses_complete_inventory_counts(capsys, caplog):
    from sparkrun.cli._setup._rdma import _render, _report_to_dict

    caplog.set_level(25, logger="sparkrun.progress")
    dets = _switch()
    report, rounds, _, _ = _run(dets, cluster=_cluster(dets, "switch"))
    assert not report.has_failure
    assert not report.full
    assert report.coverage.pair_count == 6
    assert report.coverage.path_count == 12
    assert report.selected_pair_count == 3
    assert report.selected_path_count == 6
    assert [p.pair.key for p in report.pairs] == [("h1", "h2"), ("h2", "h3"), ("h3", "h4")]
    assert all(len(p.links) == 4 and len(p.aggregates) == 2 for p in report.pairs)
    assert report.ok_count == 18
    assert rounds.call_count == 30
    data = _report_to_dict(report)
    assert data["full"] is False
    assert data["selected_pair_count"] == 3
    assert data["selected_path_count"] == 6
    assert data["coverage"]["path_count"] == 12
    _render(report)
    text = capsys.readouterr().out
    assert "Quick sample (default): selected 3/6 host pairs and 6/12 paths" in text
    assert "skipped pairs remain unverified" in text
    assert "Use --full to test all candidate link paths." in text
    planned = next(i for i, msg in enumerate(caplog.messages) if msg.startswith("Planned 6 of 12"))
    assert caplog.messages[planned + 1] == "Use --full to test all candidate link paths."


def test_quick_sample_still_runs_disjoint_end_pairs_concurrently():
    from threading import Barrier

    barrier = Barrier(2, timeout=3)

    def round_fn(server, servers, client, clients, *args, **kwargs):
        if {server, client} in ({"h1", "h2"}, {"h3", "h4"}):
            # Every latency/bandwidth/aggregate round must overlap its peer.
            barrier.wait()
        return _success(server, servers, client, clients)

    report, _, _, _ = _run(_switch(), round_fn=round_fn)
    assert report.ok_count == 18
    assert not report.has_failure


@pytest.mark.parametrize("kind", ["missing_switch_rail", "isolated_host", "failed_probe"])
def test_quick_mode_cannot_hide_inventory_failures(kind):
    dets = _switch()
    facts = _facts(dets)
    cluster = _cluster(dets, "switch")
    if kind == "missing_switch_rail":
        dets["h4"].interfaces.pop()
    elif kind == "isolated_host":
        dets["h4"].interfaces.clear()
    else:
        facts["h4"].error = "SSH timeout"
    report, rounds, prepare, _ = _run(dets, facts=facts, cluster=cluster)
    assert not report.full
    assert report.has_failure
    assert report.pairs == ()
    rounds.assert_not_called()
    prepare.assert_not_called()


def test_quick_mode_leaves_unique_ring_paths_intact():
    dets = _ring()
    report, _, _, _ = _run(dets, cluster=_cluster(dets, "ring"))
    assert not report.has_failure
    assert report.selected_pair_count == report.coverage.pair_count == 4
    assert report.selected_path_count == report.coverage.path_count == 4
    assert report.ok_count == 8


def test_quick_all_suite_still_passes_the_complete_fabric_to_nccl():
    dets = _switch()
    report, _, _, nccl = _run(dets, suite="all")
    assert report.selected_pair_count == 3
    assert report.nccl is not None
    assert nccl.call_args.args[0] == list(dets)
    assert len(nccl.call_args.args[1]) == 6


def test_default_nccl_suite_uses_all_hosts_without_sampling_guidance(caplog):
    caplog.set_level(25, logger="sparkrun.progress")
    dets = _switch()
    report, rounds, _, nccl = _run(dets, suite="nccl")
    assert report.selected_pair_count == 6
    assert report.selected_path_count == 12
    assert nccl.call_args.args[0] == list(dets)
    assert len(nccl.call_args.args[1]) == 6
    rounds.assert_not_called()
    assert not any("quick sample" in msg or "Use --full" in msg for msg in caplog.messages)


# ---------------------------------------------------------------------------
# NCCL collective: MPI pinning, memory, diagnosis and progress
# ---------------------------------------------------------------------------


def _mgmt(host, ip, subnets, iface="enP7s7"):
    det = _det(host, [], mgmt_iface=iface)
    det.mgmt_ip = ip
    det.used_subnets = set(subnets)
    return det


def test_mpi_pin_prefers_the_shared_management_subnet_over_docker_bridges():
    from sparkrun.api.setup._rdma import _mpi_tcp_if_include

    # The control node carries docker bridge networks; peers cannot route to
    # them, which is what left MPI_Init blocked until the timeout.
    dets = {
        "a": _mgmt("a", "10.24.11.11", {"10.24.11.0/24", "172.18.0.0/16", "172.19.0.0/16", "10.0.0.0/8"}),
        "b": _mgmt("b", "10.24.11.13", {"10.24.11.0/24"}),
    }
    assert _mpi_tcp_if_include(["a", "b"], dets) == "10.24.11.0/24"


def test_mpi_pin_falls_back_to_a_shared_interface_name_then_to_none():
    from sparkrun.api.setup._rdma import _mpi_tcp_if_include

    routed = {"a": _mgmt("a", "10.1.0.5", {"10.1.0.0/24"}), "b": _mgmt("b", "10.2.0.5", {"10.2.0.0/24"})}
    assert _mpi_tcp_if_include(["a", "b"], routed) == "enP7s7"

    # Different subnets and different names: a list would make Open MPI warn
    # (CIDR) or abort (name) on the hosts lacking an entry, so stay unpinned.
    mixed = {"a": _mgmt("a", "10.1.0.5", {"10.1.0.0/24"}, "eth0"), "b": _mgmt("b", "10.2.0.5", {"10.2.0.0/24"}, "eno1")}
    assert _mpi_tcp_if_include(["a", "b"], mixed) is None
    assert _mpi_tcp_if_include(["a", "b"], {}) is None


def test_nccl_command_pins_both_open_mpi_tcp_channels():
    from sparkrun.orchestration.rdma import build_nccl_test_cmd

    cmd = build_nccl_test_cmd(["h1", "h2"], rsh_agent="/tmp/agent", tcp_if_include="10.24.11.0/24")
    assert "--mca oob_tcp_if_include 10.24.11.0/24" in cmd
    assert "--mca btl_tcp_if_include 10.24.11.0/24" in cmd
    # Options precede the binary, or mpirun hands them to all_gather_perf.
    assert cmd.index("btl_tcp_if_include") < cmd.index("all_gather_perf")
    assert "if_include" not in build_nccl_test_cmd(["h1", "h2"], rsh_agent="/tmp/agent")


def _nccl_run(dets, output, rc):
    from sparkrun.api.setup._rdma import _run_nccl
    from sparkrun.orchestration.ssh import RemoteResult

    with (
        mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", return_value=[RemoteResult(h, 0, "", "") for h in dets]),
        mock.patch("sparkrun.api.setup._rdma._run_commands", return_value={0: (output, rc)}) as commands,
        mock.patch("sparkrun.orchestration.primitives.try_clear_page_cache") as clear,
    ):
        result = _run_nccl(
            list(dets), derive_link_pairs(dets, list(dets)), _facts(dets), dets, {}, msg_size="16G", binary="all_gather_perf", timeout=600
        )
    return result, commands, clear


def test_nccl_drops_page_cache_and_pins_mpi_before_the_collective():
    from test_rdma import REAL_NCCL_OUTPUT

    dets = _switch(("h1", "h2"))
    for i, det in enumerate(dets.values()):
        det.mgmt_ip = "10.24.11.%d" % (i + 1)
        det.used_subnets = {"10.24.11.0/24"}
    result, commands, clear = _nccl_run(dets, REAL_NCCL_OUTPUT, 0)
    assert result.status == "ok"
    clear.assert_called_once()
    assert sorted(clear.call_args.args[0]) == ["h1", "h2"]
    sent = commands.call_args.args[1][0]
    assert "btl_tcp_if_include 10.24.11.0/24" in sent
    assert "HWLOC_COMPONENTS=-gl" in sent


@pytest.mark.parametrize(
    "output,rc,expected",
    [
        ("", 124, "timed out after 600s"),
        ("spark-2918: Test CUDA failure common.cu:1569 'out of memory'\n", 2, "ran out of GPU memory"),
        ("mpirun noticed something\n", 1, "collective failed (rc=1)"),
    ],
)
def test_nccl_failure_detail_names_the_cause(output, rc, expected):
    result, _, _ = _nccl_run(_switch(("h1", "h2")), output, rc)
    assert result.status == STATUS_FAIL
    assert expected in result.detail


def test_heartbeat_reports_elapsed_time_until_the_step_returns(caplog):
    import threading

    from sparkrun.api.setup._rdma import _with_heartbeat

    caplog.set_level(25, logger="sparkrun.progress")
    release = threading.Event()

    def slow():
        release.wait(5)
        return "done"

    timer = threading.Timer(0.35, release.set)
    timer.start()
    try:
        assert _with_heartbeat("NCCL all_gather_perf", slow, budget_s=600, interval_s=0.1) == "done"
    finally:
        timer.cancel()
    beats = [m for m in caplog.messages if "still running" in m]
    assert beats, caplog.messages
    assert all("NCCL all_gather_perf" in m and "gives up at 600s" in m for m in beats)


def test_heartbeat_propagates_the_step_exception():
    from sparkrun.api.setup._rdma import _with_heartbeat

    def boom():
        raise RdmaTestError("nope")

    with pytest.raises(RdmaTestError, match="nope"):
        _with_heartbeat("x", boom, interval_s=0.05)


# ---------------------------------------------------------------------------
# Test image distribution goes through the launch's distribution
# ---------------------------------------------------------------------------


class _Config:
    """Just enough SparkrunConfig for distribution scoping."""

    def __init__(self, ssh_user="configuser"):
        self.ssh_user = ssh_user
        self.ssh_key = None
        self.ssh_options = []
        self.hf_cache_dir = "/cache/hf"

    def for_cluster(self, cluster):
        if cluster is not None and cluster.user:
            from copy import copy

            scoped = copy(self)
            scoped.ssh_user = cluster.user
            return scoped
        return self


def _prepare(cluster, ssh_kwargs, *, offline=False, side_effect=None):
    from sparkrun.api.setup._rdma import _prepare_container

    with (
        mock.patch("sparkrun.orchestration.distribution.distribute_resources", side_effect=side_effect) as dist,
        mock.patch("sparkrun.api.setup._rdma._start_containers", return_value=([], {})),
    ):
        _prepare_container(
            ["h1", "h2"], "img:1", ssh_kwargs, need_gpu=True, mount_ssh=True, config=_Config(), cluster=cluster, offline=offline
        )
    return dist


def test_image_uses_the_clusters_transfer_mode_and_connection():
    cluster = ClusterDefinition(name="c", hosts=["h1", "h2"], user="clusteruser", transfer_mode="delegated", topology="switch")
    dist = _prepare(cluster, {"ssh_user": "clusteruser"}, offline=True)
    args, kwargs = dist.call_args
    assert args[0] == "img:1" and args[1] == "" and args[2] == ["h1", "h2"]
    assert args[4].ssh_user == "clusteruser"
    assert kwargs["transfer_mode"] == "delegated"
    assert kwargs["topology"] == "switch"
    assert kwargs["offline"] is True


def test_image_distribution_connects_as_an_explicit_user_override():
    cluster = ClusterDefinition(name="c", hosts=["h1", "h2"], user="clusteruser")
    dist = _prepare(cluster, {"ssh_user": "override"})
    assert dist.call_args.args[4].ssh_user == "override"
    assert dist.call_args.kwargs["transfer_mode"] == "auto"


def test_image_distribution_failure_is_an_rdma_test_error():
    from sparkrun.orchestration.distribution import DistributionError

    with pytest.raises(RdmaTestError, match="Could not distribute img:1.*h2"):
        _prepare(None, {"ssh_user": "configuser"}, side_effect=DistributionError("Image distribution failed on: h2"))


def test_offline_comes_from_the_cluster_unless_the_caller_chooses():
    cluster = ClusterDefinition(name="c", hosts=["h1", "h2"], offline=True)
    with (
        mock.patch("sparkrun.api.setup._rdma._run_probe", return_value=_facts(_switch(("h1", "h2")))),
        mock.patch("sparkrun.orchestration.networking.detect_cx7_for_hosts", return_value=_switch(("h1", "h2"))),
        mock.patch("sparkrun.api.setup._rdma._prepare_container", side_effect=RdmaTestError("stop here")) as prep,
    ):
        for offline, expected in ((None, True), (False, False)):
            with pytest.raises(RdmaTestError, match="stop here"):
                rdma_test(_Sctx(), ["h1", "h2"], {}, suite="perftest", force_container=True, cluster=cluster, offline=offline)
            assert prep.call_args.kwargs["offline"] is expected
