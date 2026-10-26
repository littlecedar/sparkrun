"""Tests for cli/_monitor_logs.py: pure log-view models, scope helpers, and monitor TUI widgets."""

from __future__ import annotations

from collections import deque
from datetime import datetime
import re
import threading
import time
from typing import Any, Iterator, Sequence

import pytest
from rich.text import Text
from textual.app import App, ComposeResult
from textual.color import Color
from textual.geometry import Size
from textual.widgets import Button, Input, OptionList, Static

from sparkrun.api._errors import SparkrunError
from sparkrun.cli._monitor_logs import (
    FollowState,
    LogBuffer,
    LogPane,
    LogScope,
    LogView,
    Marker,
    START_OF_LOG_MARKER,
    STREAM_ENDED_MARKER,
    LOADING_MARKER,
    LOG_RESTARTED_MARKER,
    ScopeSummary,
    ViewState,
    WorkloadList,
    DumpLogScreen,
    StopWorkloadScreen,
    canonical_scope,
    default_dump_path,
    distinct_scopes,
    entry_plain_text,
    format_prefix,
    render_text,
    target_args,
    workload_rows,
    WorkloadRow,
    workload_scope_summary,
    _trim_page_drift,
)
from sparkrun.cli._monitor_metrics import RuntimeMetricsScraper
from sparkrun.core.cluster_status import ContainerDetail, RunningWorkload
from sparkrun.core.monitoring import HostActivity, MonitorFrame
from sparkrun.orchestration.logs import LogLine, LogSource


# ---------------------------------------------------------------------------
# Scope Helpers & Target Args & Dump Path
# ---------------------------------------------------------------------------


def test_workload_scope_summary():
    # 1. Solo workload: 1 container on h1
    w1 = RunningWorkload(
        cluster_id="cid1",
        containers=(ContainerDetail(name="c1", role="solo", status="Up", image="img"),),
    )
    frame1 = MonitorFrame(hosts=(HostActivity(host="h1", workloads=(w1,)),))
    s1 = workload_scope_summary(frame1, "cid1", "h1")
    assert s1 == ScopeSummary(node_containers=1, cluster_containers=1, cluster_hosts=1)

    # 2. Two containers of cid1 on h1, none elsewhere
    w2 = RunningWorkload(
        cluster_id="cid1",
        containers=(
            ContainerDetail(name="c1", role="head", status="Up", image="img"),
            ContainerDetail(name="c2", role="worker", status="Up", image="img"),
        ),
    )
    frame2 = MonitorFrame(hosts=(HostActivity(host="h1", workloads=(w2,)), HostActivity(host="h2")))
    s2 = workload_scope_summary(frame2, "cid1", "h1")
    assert s2 == ScopeSummary(node_containers=2, cluster_containers=2, cluster_hosts=1)

    # 3. One container on h1, one container on h2
    w3_h1 = RunningWorkload(
        cluster_id="cid1",
        containers=(ContainerDetail(name="c1", role="head", status="Up", image="img"),),
    )
    w3_h2 = RunningWorkload(
        cluster_id="cid1",
        containers=(ContainerDetail(name="c2", role="worker", status="Up", image="img"),),
    )
    frame3 = MonitorFrame(hosts=(HostActivity(host="h1", workloads=(w3_h1,)), HostActivity(host="h2", workloads=(w3_h2,))))
    s3 = workload_scope_summary(frame3, "cid1", "h1")
    assert s3 == ScopeSummary(node_containers=1, cluster_containers=2, cluster_hosts=2)

    # 4. Two on h1, one on h2
    w4_h1 = RunningWorkload(
        cluster_id="cid1",
        containers=(
            ContainerDetail(name="c1", role="head", status="Up", image="img"),
            ContainerDetail(name="c2", role="worker", status="Up", image="img"),
        ),
    )
    w4_h2 = RunningWorkload(
        cluster_id="cid1",
        containers=(ContainerDetail(name="c3", role="worker", status="Up", image="img"),),
    )
    frame4 = MonitorFrame(hosts=(HostActivity(host="h1", workloads=(w4_h1,)), HostActivity(host="h2", workloads=(w4_h2,))))
    s4 = workload_scope_summary(frame4, "cid1", "h1")
    assert s4 == ScopeSummary(node_containers=2, cluster_containers=3, cluster_hosts=2)

    # Workload with no container details defaults to 1 container
    w_empty = RunningWorkload(cluster_id="cid1", containers=())
    frame_empty = MonitorFrame(hosts=(HostActivity(host="h1", workloads=(w_empty,)),))
    s_empty = workload_scope_summary(frame_empty, "cid1", "h1")
    assert s_empty == ScopeSummary(node_containers=1, cluster_containers=1, cluster_hosts=1)

    # Empty frame
    assert workload_scope_summary(None, "cid1", "h1") == ScopeSummary(0, 0, 0)

    # 5. Ray unranked worker case (role="worker") and user@host matching
    w_ray = RunningWorkload(
        cluster_id="cid_ray",
        containers=(
            ContainerDetail(name="ray_head", role="head", status="Up", image="ray:vllm"),
            ContainerDetail(name="ray_worker_1", role="worker", status="Up", image="ray:vllm"),
        ),
    )
    w_ray_remote = RunningWorkload(
        cluster_id="cid_ray",
        containers=(ContainerDetail(name="ray_worker_2", role="worker", status="Up", image="ray:vllm"),),
    )
    frame_ray = MonitorFrame(
        hosts=(
            HostActivity(host="ubuntu@h1", workloads=(w_ray,)),
            HostActivity(host="ubuntu@h2", workloads=(w_ray_remote,)),
        )
    )
    # Target "h1" matches "ubuntu@h1"
    s_ray_h1 = workload_scope_summary(frame_ray, "cid_ray", "h1")
    assert s_ray_h1 == ScopeSummary(node_containers=2, cluster_containers=3, cluster_hosts=2)

    # Target "h2" matches "ubuntu@h2"
    s_ray_h2 = workload_scope_summary(frame_ray, "cid_ray", "h2")
    assert s_ray_h2 == ScopeSummary(node_containers=1, cluster_containers=3, cluster_hosts=2)

    # Target "h3" (mismatch) has 0 node containers
    s_ray_h3 = workload_scope_summary(frame_ray, "cid_ray", "h3")
    assert s_ray_h3 == ScopeSummary(node_containers=0, cluster_containers=3, cluster_hosts=2)


def test_distinct_scopes():
    # Solo workload: [1]
    assert distinct_scopes(ScopeSummary(node_containers=1, cluster_containers=1, cluster_hosts=1)) == [LogScope.CONTAINER]

    # Two containers on one host, none elsewhere: [1, 2]
    assert distinct_scopes(ScopeSummary(node_containers=2, cluster_containers=2, cluster_hosts=1)) == [
        LogScope.CONTAINER,
        LogScope.NODE,
    ]

    # One container on each of two hosts: [1, 3]
    assert distinct_scopes(ScopeSummary(node_containers=1, cluster_containers=2, cluster_hosts=2)) == [
        LogScope.CONTAINER,
        LogScope.CLUSTER,
    ]

    # Two on this host plus one elsewhere: [1, 2, 3]
    assert distinct_scopes(ScopeSummary(node_containers=2, cluster_containers=3, cluster_hosts=2)) == [
        LogScope.CONTAINER,
        LogScope.NODE,
        LogScope.CLUSTER,
    ]


def test_canonical_scope():
    # When solo: all scopes map to 1
    solo = ScopeSummary(node_containers=1, cluster_containers=1, cluster_hosts=1)
    assert canonical_scope(LogScope.CONTAINER, solo) == LogScope.CONTAINER
    assert canonical_scope(LogScope.NODE, solo) == LogScope.CONTAINER
    assert canonical_scope(LogScope.CLUSTER, solo) == LogScope.CONTAINER

    # When 2 on host, none elsewhere: 3 maps to 2, 2 stays 2, 1 stays 1
    two_node = ScopeSummary(node_containers=2, cluster_containers=2, cluster_hosts=1)
    assert canonical_scope(LogScope.CONTAINER, two_node) == LogScope.CONTAINER
    assert canonical_scope(LogScope.NODE, two_node) == LogScope.NODE
    assert canonical_scope(LogScope.CLUSTER, two_node) == LogScope.NODE

    # When 1 on host, 1 elsewhere: 2 maps to 1, 3 stays 3
    one_each = ScopeSummary(node_containers=1, cluster_containers=2, cluster_hosts=2)
    assert canonical_scope(LogScope.CONTAINER, one_each) == LogScope.CONTAINER
    assert canonical_scope(LogScope.NODE, one_each) == LogScope.CONTAINER
    assert canonical_scope(LogScope.CLUSTER, one_each) == LogScope.CLUSTER

    # When 2 on host, 1 elsewhere: all 3 distinct
    all_three = ScopeSummary(node_containers=2, cluster_containers=3, cluster_hosts=2)
    assert canonical_scope(LogScope.CONTAINER, all_three) == LogScope.CONTAINER
    assert canonical_scope(LogScope.NODE, all_three) == LogScope.NODE
    assert canonical_scope(LogScope.CLUSTER, all_three) == LogScope.CLUSTER


def test_target_args():
    assert target_args(LogScope.CONTAINER, "h1", "c1") == {"host": "h1", "container": "c1", "scope": "all"}
    assert target_args(LogScope.NODE, "h1", "c1") == {"host": "h1", "scope": "all"}
    assert target_args(LogScope.CLUSTER, "h1", "c1") == {"host": None, "scope": "all"}

    with pytest.raises(ValueError):
        target_args(99, "h1", "c1")


def test_default_dump_path():
    fixed_time = datetime(2026, 10, 11, 14, 30, 45)
    ts = "20261011-143045"

    p1 = default_dump_path("my-recipe", LogScope.CONTAINER, "spark-01", "my-cont", now=fixed_time)
    assert p1 == f"./my-recipe-spark-01-my-cont-{ts}.log"

    p2 = default_dump_path("my-recipe", LogScope.NODE, "spark-01", "my-cont", now=fixed_time)
    assert p2 == f"./my-recipe-spark-01-{ts}.log"

    p3 = default_dump_path("my-recipe", LogScope.CLUSTER, "spark-01", "my-cont", now=fixed_time)
    assert p3 == f"./my-recipe-all-{ts}.log"

    # Qualified registry recipe label and user@host: sanitized to safe filename without subdirectories
    p_qual = default_dump_path("@official/qwen3", LogScope.CONTAINER, "user@h1", "c1", now=fixed_time)
    assert p_qual == f"./official-qwen3-user-h1-c1-{ts}.log"
    # Ensure no subdirectories in filename
    assert "/" not in p_qual.removeprefix("./")


# ---------------------------------------------------------------------------
# Formatting: format_prefix & render_text
# ---------------------------------------------------------------------------


def test_format_prefix():
    line = LogLine(host="h1", container="c1", text="msg", role="worker", rank=1)

    # Multi host -> [host/role/rank]
    assert format_prefix(line, multi_host=True, multi_source=True) == "[h1/worker/1]"

    # Single host, multi source -> [role/rank]
    assert format_prefix(line, multi_host=False, multi_source=True) == "[worker/1]"

    # Neither -> empty
    assert format_prefix(line, multi_host=False, multi_source=False) == ""

    # Rank is None
    line_no_rank = LogLine(host="h1", container="c1", text="msg", role="solo", rank=None)
    assert format_prefix(line_no_rank, multi_host=True, multi_source=True) == "[h1/solo]"
    assert format_prefix(line_no_rank, multi_host=False, multi_source=True) == "[solo]"


def test_render_text_color_and_strip():
    ansi_msg = "\x1b[32mSuccess\x1b[0m: details"

    # Color on: Rich parsed ANSI
    t_color = render_text(ansi_msg, color=True)
    assert isinstance(t_color, Text)
    assert "Success" in t_color.plain
    assert len(t_color.spans) > 0  # ANSI styling present

    # Color off: stripped plain text
    t_plain = render_text(ansi_msg, color=False)
    assert isinstance(t_plain, Text)
    assert t_plain.plain == "Success: details"
    assert len(t_plain.spans) == 0  # ANSI stripped


def test_entry_plain_text_no_ansi_passthrough():
    """No-ANSI text passes through unchanged; ANSI/control-char text still takes from_ansi."""
    line = LogLine(host="h1", container="c1", role="head", rank=0, text="Avg prefill throughput: 150.0 tokens/s")
    assert entry_plain_text(line) == line.text
    assert entry_plain_text(line) == line.text  # memo returns the same value

    ansi = LogLine(host="h1", container="c1", role="head", rank=0, text="\x1b[32mok\x1b[0m: 12 tokens/s")
    assert entry_plain_text(ansi) == "ok: 12 tokens/s"

    # Control rewrites must match Text.from_ansi exactly (guard does not take the
    # fast path): CR, and the splitlines breaks NEL / U+2028 / U+2029.
    for sep in ("\r", "\x85", "\u2028", "\u2029"):
        line = LogLine(host="h1", container="c1", role="head", rank=0, text=f"a{sep}b")
        assert entry_plain_text(line) == Text.from_ansi(f"a{sep}b").plain


def test_render_text_search_highlights():
    line = LogLine(host="h1", container="c1", text="error in worker node", role="worker", rank=0)

    # Regex highlight appears in both color and plain modes
    t_col = render_text(line, color=True, pattern="error")
    assert any(s.style == "reverse" for s in t_col.spans)

    t_plain = render_text(line, color=False, pattern="error")
    assert any(s.style == "reverse" for s in t_plain.spans)

    # Regex pattern matching worker: prefix must NOT be highlighted, only body
    t_pref = render_text(line, color=True, pattern="worker", multi_host=False, multi_source=True)
    assert t_pref.plain == "[worker/0] error in worker node"
    reverse_spans = [s for s in t_pref.spans if s.style == "reverse"]
    assert len(reverse_spans) == 1
    assert reverse_spans[0].start == 20
    assert reverse_spans[0].end == 26
    assert t_pref.plain[reverse_spans[0].start : reverse_spans[0].end] == "worker"

    # Search on ANSI text: highlight matches parsed text
    ansi_line = LogLine(host="h1", container="c1", text="\x1b[32mINFO\x1b[0m 10-11 start", role="worker", rank=0)
    t_ansi = render_text(ansi_line, color=True, pattern="INFO 10")
    rev_ansi = [s for s in t_ansi.spans if s.style == "reverse"]
    assert len(rev_ansi) == 1
    assert t_ansi.plain[rev_ansi[0].start : rev_ansi[0].end] == "INFO 10"


@pytest.mark.parametrize(
    ("text", "token", "style"),
    [
        ("INFO 10-11 08:12:34 [core] init", "INFO", "cyan"),  # vLLM
        ("(EngineCore pid=9527) INFO 10-11 08:12:34 [core.py:130] Worker ready", "INFO", "cyan"),  # vLLM engine core
        ("[2026-10-11 08:12:34 TP0] INFO backend ready", "INFO", "cyan"),  # SGLang
        ("0.00.035.064 I llama_model_load: vocab only - skipping tensors", "I", "cyan"),  # llama.cpp
        ("0.00.000.000 E llama_model_load_from_file_impl: failed to load model", "E", "red"),  # llama.cpp
        ("0.12.345.678 D llama_server: verbose", "D", "dim"),  # llama.cpp debug
        ("[10/11/2026-08:12:34 UTC] [E] rank 1 exited", "[E]", "red"),  # TRT-LLM (span covers the bracket token)
        ("E1011 08:12:34.123  9527 trt_ffi.cc:42] bad tensor", "E", "red"),  # glog
        ("dgx01:28971 [0] NCCL INFO Ring ALGO", "INFO", "cyan"),  # NCCL
        ("[2026-10-11T08:12:34Z WARN atlas::serve] oom", "WARN", "yellow"),  # Atlas env_logger
        ("2026-10-11T08:12:34.123Z  INFO atlas::serve: GET /v1/chat", "INFO", "cyan"),  # Atlas tracing
        ("INFO:engine.core:Started engine", "INFO", "cyan"),  # Python logging
    ],
)
def test_render_text_colorizes_runtime_levels(text, token, style):
    line = LogLine(host="h", container="c", text=text, role="worker", rank=0)
    t = render_text(line, color=True)
    # Slice-aware: the dim baseline and rank colors share styles with level spans.
    assert any(s.style == style and t.plain[s.start : s.end] == token for s in t.spans), (text, t.spans)
    assert render_text(line, color=False).spans == []


def test_render_text_colorizes_tensorfold_lines():
    """TensorFold shapes from ashhart/TensorFold zig/src/server/log.zig + serve.zig."""
    prefix = "[tensorfold] "
    cases = [
        # (message after the prefix, spans that must be present, styles that must not appear)
        (
            "serving nemotron at http://127.0.0.1:8080/v1 (sampling: greedy; drafts: on; context: 262144; loaded in 1.2s)",
            [("http://127.0.0.1:8080/v1", "underline blue"), ("1.2s", "yellow"), ("serving", "green")],
            ("red",),
        ),
        ("done req-8f3a2c1e prompt=100 tokens=200 tok/s=100.0 ttft=0.10s", [("done", "green"), ("req-8f3a2c1e", "magenta")], ("red",)),
        ("warning: non-loopback server has no API key; requests are open", [("warning:", "yellow")], ("red", "green")),
        ("refused cmpl-1: model not found", [("refused", "red")], ("green",)),
        ("ended chatcmpl-1 reason=client-left prompt=12 tokens=3 after=0.50s", [("chatcmpl-1", "magenta")], ("red", "green")),
        ("ended chatcmpl-1 reason=error (OutOfMemory) prompt=12 tokens=0 after=1.25s", [("reason=error", "red")], ("green",)),
        ('10.0.0.5 "GET /v1/models HTTP/1.1" 200 -', [("200", "green")], ("red",)),
        ('10.0.0.5 "POST /v1/chat/completions HTTP/1.1" 404 - key=cli-1', [("404", "yellow")], ("red", "green")),
        ('10.0.0.5 "POST /v1/chat/completions HTTP/1.1" 503 -', [("503", "red")], ("green",)),
    ]
    for msg, required, forbidden in cases:
        line = LogLine(host="h", container="c", text=prefix + msg, role="worker", rank=0)
        t = render_text(line, color=True)
        got = [(t.plain[s.start : s.end], s.style) for s in t.spans]
        assert any(s.style == "dim" and s.start == 0 and s.end >= len(prefix) for s in t.spans)  # prefix is dimmed
        for pair in required:
            assert pair in got, (msg, pair, got)
        for style in forbidden:
            assert not any(s.style == style for s in t.spans), (msg, style, got)
        assert render_text(line, color=False).spans == []


def test_render_text_colorizes_tensorfold_fatal_line():
    fatal = "tensorfold: cannot listen on 127.0.0.1:8080 (AddressInUse)"
    line = LogLine(host="h", container="c", text=fatal, role="worker", rank=0)
    t = render_text(line, color=True)
    assert any(s.style == "red" and s.start == 0 and s.end == len(fatal) for s in t.spans)
    assert render_text(line, color=False).spans == []


def test_render_text_colorizes_traceback_whole_line():
    text = "Traceback (most recent call last):"
    line = LogLine(host="h", container="c", text=text, role="worker", rank=0)
    t = render_text(line, color=True)
    assert any(s.style == "bold red" and s.start == 0 and s.end == len(text) for s in t.spans)


def test_render_text_dims_plain_prose():
    line = LogLine(host="h", container="c", text="serving requests on 0.0.0.0", role="worker", rank=0)
    t = render_text(line, color=True)
    assert t.spans  # the baseline is dimmed, not left plain
    assert all(s.style == "dim" for s in t.spans)


def test_render_text_dim_baseline_spares_level_and_ansi():
    line = LogLine(host="h", container="c", text="INFO 10-11 worker ready", role="worker", rank=0)
    t = render_text(line, color=True)
    got = {(t.plain[s.start : s.end], s.style) for s in t.spans}
    assert ("INFO", "cyan") in got  # the level token keeps its color...
    assert (" 10-11 worker ready", "dim") in got  # ...and only the remainder dims

    ansi = LogLine(host="h", container="c", text="\x1b[32mSuccess\x1b[0m: details", role="worker", rank=0)
    t_ansi = render_text(ansi, color=True)
    assert any(s.style == "reverse" or str(s.style) == "color(2)" for s in t_ansi.spans)
    assert not any(s.start == 0 and s.end == 7 and s.style == "dim" for s in t_ansi.spans)
    assert any(t_ansi.plain[s.start : s.end] == ": details" and s.style == "dim" for s in t_ansi.spans)


def test_render_text_search_highlight_is_never_dimmed():
    line = LogLine(host="h", container="c", text="INFO a trailing match tail", role="worker", rank=0)
    t = render_text(line, color=True, pattern="match")
    hl = [s for s in t.spans if s.style in ("reverse", "bold reverse underline")]
    assert len(hl) == 1
    assert t.plain[hl[0].start : hl[0].end] == "match"
    assert not any(s.style == "dim" and s.start < hl[0].end and s.end > hl[0].start for s in t.spans)


def test_rendered_segments_keep_data_bright_under_dim_baseline():
    """Effective styles, not span lists: the dim baseline must never wash a highlight."""
    import io

    from rich.console import Console

    console = Console(file=io.StringIO(), width=100, force_terminal=True, color_system="standard")
    line = LogLine(host="h", container="c", text="INFO worker failed to bind", role="worker", rank=0)
    rendered = render_text(line, color=True, pattern="worker")
    segments = [seg for seg in console.render(rendered, console.options) if seg.text.strip()]

    def style_of(predicate):
        return next(seg.style for seg in segments if predicate(seg.text))

    assert style_of(lambda t: t == "INFO").dim is not True  # level keeps its color
    highlight = style_of(lambda t: t == "worker")
    assert highlight.reverse is True and highlight.dim is not True
    assert style_of(lambda t: "failed" in t).dim is not True  # severity keeps red
    assert style_of(lambda t: "bind" in t).dim is True  # ...and the baseline really is dim


def test_render_text_with_prefix():
    line = LogLine(host="h1", container="c1", text="hello", role="worker", rank=0)
    t = render_text(line, multi_host=True, multi_source=True)
    assert t.plain.startswith("[h1/worker/0] ")
    assert t.plain.endswith("hello")


def test_render_text_renders_marker_text_not_repr():
    """A Marker renders its text; its dataclass repr must never reach the view."""
    marker = Marker("— start of log —")
    assert render_text(marker, color=False).plain == "— start of log —"
    assert render_text(marker, color=True).plain == "— start of log —"


def test_log_view_renders_marker_text():
    import asyncio

    async def _run():
        buf = LogBuffer()
        buf.append(Marker("— start of log —"))
        view = LogView(buf)

        class MarkerApp(App):
            def compose(self) -> ComposeResult:
                yield view

        async with MarkerApp().run_test():
            text = "".join(seg.text for seg in view.render_line(0))
            assert "— start of log —" in text
            assert "Marker(" not in text

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# workload_rows
# ---------------------------------------------------------------------------


def test_workload_rows(monkeypatch):
    # Empty
    assert workload_rows(None)[0].key == "empty"
    assert workload_rows(HostActivity(host="h1"))[0].key == "empty"

    # Running workload with container details
    wl = RunningWorkload(
        cluster_id="sparkrun_abc",
        recipe_name="qwen3-1.7b-vllm",
        runtime_name="vllm",
        ranks_on_host=2,
        containers=(
            ContainerDetail(name="sparkrun_abc_head", role="head", status="Up 5m", image="vllm:latest"),
            ContainerDetail(name="sparkrun_abc_worker_0", role="worker", status="Up 5m", image="vllm:latest"),
        ),
    )
    act = HostActivity(host="h1", workloads=(wl,))
    rows = workload_rows(act)

    assert len(rows) == 3
    # Header row
    assert rows[0].is_header is True
    assert rows[0].key == "header:sparkrun_abc"
    assert "sparkrun_abc" in rows[0].text
    assert "×2" in rows[0].text
    assert "recipe=" in rows[0].text

    # Container rows
    assert rows[1].is_header is False
    assert rows[1].key == "sparkrun_abc:sparkrun_abc_head"
    assert rows[1].container == "sparkrun_abc_head"
    assert rows[1].role == "head"

    assert rows[2].is_header is False
    assert rows[2].key == "sparkrun_abc:sparkrun_abc_worker_0"
    assert rows[2].container == "sparkrun_abc_worker_0"
    assert rows[2].role == "worker"

    # Real job metadata keys: recipe, runtime, model, tensor_parallel
    def fake_load_job_metadata(cluster_id, cache_dir=None):
        return {
            "recipe": "@official/qwen3-30b-vllm",
            "runtime": "vllm-distributed",
            "model": "Qwen/Qwen3-30B",
            "tensor_parallel": 4,
        }

    monkeypatch.setattr("sparkrun.cli._monitor_logs.load_job_metadata", fake_load_job_metadata)

    wl_meta = RunningWorkload(
        cluster_id="sparkrun_meta",
        ranks_on_host=4,
        containers=(ContainerDetail(name="sparkrun_meta_0", role="node_0", status="Up", image="vllm"),),
    )
    act_meta = HostActivity(host="h1", workloads=(wl_meta,))
    rows_meta = workload_rows(act_meta)
    assert len(rows_meta) == 2
    header_text = rows_meta[0].text
    assert "recipe=[bold]@official/qwen3-30b-vllm[/bold]" in header_text
    assert "model=Qwen/Qwen3-30B" in header_text
    assert "runtime=vllm-distributed" in header_text
    assert "tp=4" in header_text

    # Solo workload with no containers: only header row produced, NO fake container row
    wl_no_containers = RunningWorkload(
        cluster_id="sparkrun_no_cont",
        ranks_on_host=1,
        containers=(),
    )
    act_no_cont = HostActivity(host="h1", workloads=(wl_no_containers,))
    rows_no_cont = workload_rows(act_no_cont)
    assert len(rows_no_cont) == 1
    assert rows_no_cont[0].is_header is True
    assert rows_no_cont[0].cluster_id == "sparkrun_no_cont"
    assert rows_no_cont[0].key == "header:sparkrun_no_cont"


# ---------------------------------------------------------------------------
# LogBuffer: Paging Offsets, Interleaving, Eviction, Search
# ---------------------------------------------------------------------------


def test_log_buffer_append_and_prepend_skip_offsets():
    buf = LogBuffer()
    src1 = LogSource(host="h1", container="c1", role="solo", rank=0)
    src2 = LogSource(host="h2", container="c2", role="solo", rank=1)

    # Initial append: 3 lines for src1, 2 for src2
    buf.append(LogLine(host="h1", container="c1", text="line 1", role="solo", rank=0))
    buf.append(LogLine(host="h1", container="c1", text="line 2", role="solo", rank=0))
    buf.append(LogLine(host="h2", container="c2", text="h2 line 1", role="solo", rank=1))
    buf.append(LogLine(host="h1", container="c1", text="line 3", role="solo", rank=0))
    buf.append(LogLine(host="h2", container="c2", text="h2 line 2", role="solo", rank=1))

    assert buf.get_skip(src1) == 3
    assert buf.get_skip(src2) == 2

    # Prepend page of 2 lines for src1
    page1 = [
        LogLine(host="h1", container="c1", text="old line -2", role="solo", rank=0),
        LogLine(host="h1", container="c1", text="old line -1", role="solo", rank=0),
    ]
    buf.prepend_page(src1, page1)
    assert buf.get_skip(src1) == 5
    assert buf.get_skip(src2) == 2

    # Markers do not increment skip
    buf.append(Marker("marker"))
    assert buf.get_skip(src1) == 5
    assert buf.get_skip(src2) == 2

    # Interleaved live line append for src1
    buf.append(LogLine(host="h1", container="c1", text="line 4", role="solo", rank=0))
    assert buf.get_skip(src1) == 6
    assert buf.get_skip(src2) == 2

    # Prepend page with separator and exhausted=True
    page_old = [
        LogLine(host="h1", container="c1", text="first ever line 0", role="solo", rank=0),
    ]
    buf.prepend_page(src1, page_old, separator=START_OF_LOG_MARKER, exhausted=True)
    assert buf[0] == START_OF_LOG_MARKER
    assert buf[1].text == "first ever line 0"
    assert buf.is_exhausted(src1) is True
    assert buf.get_skip(src1) == 7  # 6 + 1 (LogLine only, not marker)


def test_log_buffer_eviction_top_and_bottom():
    buf = LogBuffer()
    src1 = LogSource(host="h1", container="c1", role="solo", rank=0)
    src2 = LogSource(host="h2", container="c2", role="solo", rank=1)

    # Interleaved lines: 6 from src1, 4 from src2 (total 10)
    # 0: src1, 1: src1, 2: src2, 3: src1, 4: src2, 5: src1, 6: src2, 7: src1, 8: src2, 9: src1
    for i in range(10):
        if i in (2, 4, 6, 8):
            buf.append(LogLine(host="h2", container="c2", text=f"src2 line {i}", role="solo", rank=1))
        else:
            buf.append(LogLine(host="h1", container="c1", text=f"src1 line {i}", role="solo", rank=0))

    assert len(buf) == 10
    assert buf.get_skip(src1) == 6
    assert buf.get_skip(src2) == 4

    # Eviction with (top, bottom) viewport tuple:
    # 1. Viewport (6, 8) -> user near bottom (dist_to_top=6, dist_to_bottom=9-8=1)
    # Evict to cap 8: excess 2 lines dropped from top (lines 0 and 1, both src1)
    evicted = buf.evict(cap=8, viewport=(6, 8))
    assert evicted == 2
    assert len(buf) == 8
    assert buf.get_skip(src1) == 4  # 6 - 2 = 4
    assert buf.get_skip(src2) == 4  # unchanged, no src2 lines were dropped
    assert buf[0].container == "c2"  # line 2 was src2

    # 2. Viewport (1, 3) -> user near top (dist_to_top=1, dist_to_bottom=7-3=4)
    # Add 4 lines (making 12)
    for i in range(10, 14):
        buf.append(LogLine(host="h1", container="c1", text=f"src1 line {i}", role="solo", rank=0))
    assert len(buf) == 12

    # Evict to cap 8 with viewport (1, 3): excess 4 lines dropped from bottom (newest live lines)
    evicted_bot = buf.evict(cap=8, viewport=(1, 3))
    assert evicted_bot == 4
    assert len(buf) == 8
    assert buf.evicted_live == 4

    # 3. Append after bottom eviction:
    # While evicted_live > 0, appended live lines are counted and skip is incremented,
    # but they are NOT added to entries (preventing an untracked gap in the buffer)
    old_len = len(buf)
    skip_src1_before = buf.get_skip(src1)
    buf.append(LogLine(host="h1", container="c1", text="live after bottom evict", role="solo", rank=0))
    assert len(buf) == old_len
    assert buf.evicted_live == 5
    assert buf.get_skip(src1) == skip_src1_before + 1

    # 4. Exhausted reset on top eviction:
    # Mark src exhausted, then prepend marker and lines, then evict from top
    buf_ex = LogBuffer()
    src = LogSource(host="h1", container="c1", role="solo", rank=0)
    for i in range(6):
        buf_ex.append(LogLine(host="h1", container="c1", text=f"l{i}", role="solo", rank=0))
    buf_ex.prepend_page(
        src,
        [
            LogLine(host="h1", container="c1", text="p1", role="solo", rank=0),
            LogLine(host="h1", container="c1", text="p2", role="solo", rank=0),
        ],
        separator=Marker("marker"),
        exhausted=True,
    )
    assert buf_ex.is_exhausted(src) is True
    assert buf_ex.get_skip(src) == 8  # 6 + 2
    # Evict to cap 6 near bottom (viewport=8) -> drops marker and p1, p2 (3 items)
    ev = buf_ex.evict(cap=6, viewport=8)
    assert ev == 3
    assert len(buf_ex) == 6
    # Skip drops from 8 to 6 (marker doesn't decrement skip, only 2 LogLines)
    assert buf_ex.get_skip(src) == 6
    # Exhausted flag is reset to False because start of log was evicted
    assert buf_ex.is_exhausted(src) is False


def _ll(text, container="c1"):
    return LogLine(host="h1", container=container, text=text, role="solo", rank=0)


def test_log_buffer_bottom_eviction_retains_terminal_markers_and_reload_tail():
    buf = LogBuffer()
    for i in range(5):
        buf.append(_ll(f"line {i}"))
    buf.append(LOADING_MARKER)
    assert buf.stream_ended is False

    # Bottom eviction drops the transient loading marker and line 4.
    assert buf.evict(cap=4, viewport=0) == 2
    assert buf.evicted_live == 1
    assert buf.pending_markers == []  # transient marker not preserved

    buf.append(STREAM_ENDED_MARKER)
    assert buf.evicted_live == 1
    assert buf.pending_markers == [STREAM_ENDED_MARKER]
    assert buf.stream_ended is True
    buf.append(_ll("live after"))
    assert buf.evicted_live == 2

    buf.search("line")
    carried = buf.reload_tail([_ll("t3"), _ll("t4 line"), _ll("n1 line")])
    assert carried == [STREAM_ENDED_MARKER]
    assert buf.entries[:3] == [_ll("t3"), _ll("t4 line"), _ll("n1 line")]
    assert buf.entries[-1] == STREAM_ENDED_MARKER
    assert len(buf) == 4
    assert buf.evicted_live == 0 and buf.evicted_history == 0
    assert buf.pending_markers == []
    assert buf.get_skip(("h1", "c1")) == 3
    assert buf._base_offset == 0
    assert buf.is_exhausted(("h1", "c1")) is False
    assert buf.search_matches == [1, 2]
    assert buf.stream_ended is True


def test_log_buffer_pending_markers_chronological_and_not_deduped():
    buf = LogBuffer()
    for i in range(4):
        buf.append(_ll(f"l{i}"))
    buf.append(LOG_RESTARTED_MARKER)
    buf.append(_ll("l4"))
    buf.append(STREAM_ENDED_MARKER)
    # entries: l0..l3, R, l4, S -> drop S first, then R
    buf.evict(cap=6, viewport=0)
    assert buf.pending_markers == [STREAM_ENDED_MARKER]
    buf.evict(cap=4, viewport=0)
    assert buf.pending_markers == [LOG_RESTARTED_MARKER, STREAM_ENDED_MARKER]

    # Genuine repeated restarts are both kept
    buf.append(LOG_RESTARTED_MARKER)
    buf.append(LOG_RESTARTED_MARKER)
    assert buf.pending_markers == [LOG_RESTARTED_MARKER, STREAM_ENDED_MARKER, LOG_RESTARTED_MARKER, LOG_RESTARTED_MARKER]


def test_log_buffer_eviction_amortized_performance_and_invariants():
    buf = LogBuffer()
    assert isinstance(buf._entries, deque)
    assert isinstance(buf._search_matches, deque)

    # Populate 50,000 lines
    for i in range(50000):
        text = f"target line {i}" if i % 10 == 0 else f"other line {i}"
        buf.append(LogLine(host="h1", container="c1", text=text, role="solo", rank=0))

    assert len(buf) == 50000
    # Search for target -> 5,000 matches
    matches = buf.search("target")
    assert len(matches) == 5000
    assert matches[0] == 0
    assert matches[1] == 10

    # Defensive return: mutating result doesn't corrupt buffer
    matches.append(999999)
    assert 999999 not in buf.search_matches
    defensive_matches = buf.search_matches
    defensive_matches.append(888888)
    assert 888888 not in buf.search_matches
    assert buf.search_pattern is not None
    assert buf.search_pattern.pattern == "target"

    # Append 1,000 lines at cap=50000 with viewport at bottom
    t0 = time.perf_counter()
    for i in range(1000):
        buf.append(LogLine(host="h1", container="c1", text=f"live target {50000 + i}", role="solo", rank=0))
        buf.evict(cap=50000, viewport=50000)
    elapsed = time.perf_counter() - t0

    # Invariants check:
    assert len(buf) == 50000
    assert buf._base_offset == 1000
    assert buf.evicted_history == 1000

    # Matches check: the first 100 matches (which were at index < 1000) got evicted from the top,
    # and 1,000 live target matches were appended at the bottom.
    current_matches = buf.search_matches
    assert len(current_matches) == 5000 - 100 + 1000
    # Verify that the relative match coordinates in current_matches correctly match the lines in buf
    for m_idx in (0, len(current_matches) // 2, -1):
        line_idx = current_matches[m_idx]
        assert "target" in buf[line_idx].text

    # Performance check: 1000 append+evict operations at 50k cap must be O(1) amortized (< 0.2s)
    assert elapsed < 0.2


def test_log_buffer_search():
    buf = LogBuffer()
    buf.append(LogLine(host="h1", container="c1", text="first apple", role="solo", rank=0))
    buf.append(LogLine(host="h1", container="c1", text="second banana", role="solo", rank=0))
    buf.append(LogLine(host="h1", container="c1", text="third apple pie", role="solo", rank=0))

    matches = buf.search("apple")
    assert matches == [0, 2]
    assert buf.search_matches == [0, 2]

    # Append new line matching search pattern -> updates search index
    buf.append(LogLine(host="h1", container="c1", text="fourth apple crumble", role="solo", rank=0))
    assert buf.search_matches == [0, 2, 3]

    # Prepend page shifts search indices
    page = [LogLine(host="h1", container="c1", text="prepended apple", role="solo", rank=0)]
    buf.prepend_page(LogSource("h1", "c1", "solo", 0), page)
    assert buf.search_matches == [0, 1, 3, 4]

    # Search on ANSI text: matches ANSI-stripped plain text, not escape codes
    buf_ansi = LogBuffer()
    buf_ansi.append(LogLine(host="h1", container="c1", text="\x1b[32mINFO\x1b[0m 10-11 start", role="solo", rank=0))
    buf_ansi.append(LogLine(host="h1", container="c1", text="plain 32m line", role="solo", rank=0))
    assert buf_ansi.search("INFO 10") == [0]
    # "32m" only matches the literal text in line 1, not the ANSI escape code in line 0
    assert buf_ansi.search("32m") == [1]

    # Markers do not count as search matches
    buf_ansi.append(Marker("— start of log —"))
    assert buf_ansi.search("start of log") == []

    # Invalid regex raises re.error and preserves previous search pattern and matches
    buf_ansi.search("INFO 10")
    assert buf_ansi.search_matches == [0]
    with pytest.raises(re.error):
        buf_ansi.search("(")
    assert buf_ansi.search_matches == [0]


# ---------------------------------------------------------------------------
# ViewState: Scroll and Follow State Machine
# ---------------------------------------------------------------------------


def test_view_state_transitions():
    vs = ViewState()
    assert vs.is_following is True
    assert vs.state == FollowState.FOLLOWING
    assert vs.pending_lines == 0

    # New lines while following do not accumulate pending
    vs.on_new_line(5)
    assert vs.pending_lines == 0

    # Scroll away from bottom -> SCROLLED
    vs.on_scroll(at_bottom=False)
    assert vs.is_scrolled is True
    assert vs.state == FollowState.SCROLLED

    # New lines while scrolled accumulate
    vs.on_new_line(3)
    assert vs.pending_lines == 3
    assert "+3" in vs.status_label

    # Space while scrolled -> jumps to end and resumes following
    jumped = vs.on_space(at_bottom=False)
    assert jumped is True
    assert vs.is_following is True
    assert vs.state == FollowState.FOLLOWING
    assert vs.pending_lines == 0

    # Space while at bottom -> toggles pause
    jumped = vs.on_space(at_bottom=True)
    assert jumped is False
    assert vs.is_paused is True
    assert vs.state == FollowState.PAUSED

    # Lines while paused accumulate
    vs.on_new_line(2)
    assert vs.pending_lines == 2

    # Space again while paused -> resumes following and clears pending
    jumped = vs.on_space(at_bottom=True)
    assert jumped is False
    assert vs.is_following is True
    assert vs.state == FollowState.FOLLOWING
    assert vs.pending_lines == 0


def test_view_state_space_toggle_reliable():
    """Space toggle reliably switches state regardless of at_bottom parameter."""
    # 1. In FOLLOWING:
    # Space always transitions to PAUSED, even if at_bottom is False or None
    for at_bottom in (True, False, None):
        vs = ViewState()
        assert vs.state == FollowState.FOLLOWING
        jumped = vs.on_space(at_bottom=at_bottom)
        assert jumped is False
        assert vs.state == FollowState.PAUSED
        assert vs.is_paused is True

    # 2. In PAUSED:
    # Space always transitions to FOLLOWING, even if at_bottom is False or None
    for at_bottom in (True, False, None):
        vs = ViewState()
        vs.state = FollowState.PAUSED
        vs.pending_lines = 42
        jumped = vs.on_space(at_bottom=at_bottom)
        assert jumped is False
        assert vs.state == FollowState.FOLLOWING
        assert vs.is_following is True
        assert vs.pending_lines == 0

    # 3. In SCROLLED:
    # Space always transitions to FOLLOWING and returns jumped=True
    for at_bottom in (True, False, None):
        vs = ViewState()
        vs.state = FollowState.SCROLLED
        vs.pending_lines = 15
        jumped = vs.on_space(at_bottom=at_bottom)
        assert jumped is True
        assert vs.state == FollowState.FOLLOWING
        assert vs.is_following is True
        assert vs.pending_lines == 0


def test_view_state_user_scroll_up_and_resilient_bottom():
    """User scroll up explicitly decouples following; bottom resume only works from scrolled."""
    vs = ViewState()
    # From FOLLOWING -> on_user_scroll_up transitions to SCROLLED
    assert vs.state == FollowState.FOLLOWING
    vs.on_user_scroll_up()
    assert vs.state == FollowState.SCROLLED

    # Subsequent user scrolls stay SCROLLED
    vs.on_user_scroll_up()
    assert vs.state == FollowState.SCROLLED

    # When reaching bottom, SCROLLED transitions back to FOLLOWING
    vs.pending_lines = 5
    vs.on_scroll(at_bottom=True)
    assert vs.state == FollowState.FOLLOWING
    assert vs.pending_lines == 0

    # In PAUSED, on_user_scroll_up does NOT leave PAUSED
    vs.state = FollowState.PAUSED
    vs.on_user_scroll_up()
    assert vs.state == FollowState.PAUSED

    # In PAUSED, on_scroll(True) does NOT resume following (pause stays paused)
    vs.on_scroll(at_bottom=True)
    assert vs.state == FollowState.PAUSED

    # In PAUSED, on_scroll(False) stays PAUSED
    vs.on_scroll(at_bottom=False)
    assert vs.state == FollowState.PAUSED


def test_log_view_scroll_up_decouples_synchronously():
    """LogView scroll actions immediately decouple ViewState without waiting for messages."""
    import asyncio

    buf = LogBuffer()
    for i in range(100):
        buf.append(LogLine(host="h1", container="c1", role="head", rank=0, text=f"line {i}"))

    vs = ViewState()
    view = LogView(buf, view_state=vs)

    class ViewApp(App):
        def compose(self) -> ComposeResult:
            yield view

    async def _run():
        app = ViewApp()
        async with app.run_test(size=(80, 20)) as pilot:
            view.scroll_to_end()
            await pilot.pause(0.05)
            assert vs.state == FollowState.FOLLOWING

            # action_scroll_up synchronously sets state to SCROLLED
            view.action_scroll_up()
            assert vs.state == FollowState.SCROLLED

            # Resume following
            vs.state = FollowState.FOLLOWING
            # action_page_up synchronously sets state to SCROLLED
            view.action_page_up()
            assert vs.state == FollowState.SCROLLED

            # Resume following
            vs.state = FollowState.FOLLOWING
            # action_scroll_home synchronously sets state to SCROLLED
            view.action_scroll_home()
            assert vs.state == FollowState.SCROLLED

    asyncio.run(_run())


def test_log_pane_key_scroll_and_pause_decoupling(monkeypatch):
    """Up arrow immediately decouples from follow stream; space reliably pauses and resumes."""
    import asyncio
    import time

    lines = [LogLine(host="h1", container="c1", text=f"stream line {i}") for i in range(50)]

    class StreamingTarget(FakeLogTarget):
        def follow(self, tail: int | None = 200, cancel: threading.Event | None = None):
            yield from super().follow(tail=tail, cancel=cancel)
            while cancel and not cancel.is_set():
                time.sleep(0.01)

    target = StreamingTarget(lines=lines)
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: target)

    async def _run():
        pane = LogPane(host="h1", cluster_id="cid1", container="c1", scope=LogScope.CONTAINER)
        app = _LogPaneTestApp(pane)
        async with app.run_test(size=(80, 20)) as pilot:
            await pilot.pause(0.1)
            assert pane.view_state.state == FollowState.FOLLOWING

            # Press 'up' arrow -> immediately decouples to SCROLLED
            await pilot.press("up")
            assert pane.view_state.state == FollowState.SCROLLED
            scrolled_y = pane.log_view.scroll_y

            # Deliver a new live line while scrolled
            pane._on_line(LogLine(host="h1", container="c1", text="incoming live line"), gen=pane._generation)
            # View must NOT have snapped back to bottom
            assert pane.view_state.state == FollowState.SCROLLED
            assert pane.log_view.scroll_y == scrolled_y
            assert pane.view_state.pending_lines == 1

            # Press 'space' -> jumps to end and resumes following
            await pilot.press("space")
            assert pane.view_state.state == FollowState.FOLLOWING
            assert pane.view_state.pending_lines == 0

            # Press 'space' while following -> unconditionally pauses
            await pilot.press("space")
            assert pane.view_state.state == FollowState.PAUSED

            # Deliver another line while paused
            pane._on_line(LogLine(host="h1", container="c1", text="line while paused"), gen=pane._generation)
            assert pane.view_state.state == FollowState.PAUSED
            assert pane.view_state.pending_lines == 1

            # Press 'space' while paused -> resumes following
            await pilot.press("space")
            assert pane.view_state.state == FollowState.FOLLOWING
            assert pane.view_state.pending_lines == 0

    asyncio.run(_run())


def test_log_pane_burst_lines_then_synchronous_up_race(monkeypatch):
    """Lines delivered followed immediately by synchronous scroll_up must not be snapped back."""
    import asyncio
    import time

    lines = [LogLine(host="h1", container="c1", text=f"line {i}") for i in range(50)]

    class StreamingTarget(FakeLogTarget):
        def follow(self, tail: int | None = 200, cancel: threading.Event | None = None):
            yield from super().follow(tail=tail, cancel=cancel)
            while cancel and not cancel.is_set():
                time.sleep(0.01)

    target = StreamingTarget(lines=lines)
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: target)

    async def _run():
        pane = LogPane(host="h1", cluster_id="cid1", container="c1", scope=LogScope.CONTAINER)
        app = _LogPaneTestApp(pane)
        async with app.run_test(size=(80, 20)) as pilot:
            await pilot.pause(0.1)
            assert pane.view_state.state == FollowState.FOLLOWING

            # Deliver burst of 5 lines
            for i in range(5):
                pane._on_line(LogLine(host="h1", container="c1", text=f"burst {i}"), gen=pane._generation)

            # Immediately trigger upward scroll action without awaiting idle/refresh
            pane.log_view.action_scroll_up()
            assert pane.view_state.state == FollowState.SCROLLED
            target_y = pane.log_view.scroll_target_y

            # After refresh/pause, scroll position and SCROLLED state must be preserved
            await pilot.pause(0.2)
            assert pane.view_state.state == FollowState.SCROLLED
            assert pane.log_view.scroll_y == target_y
            assert pane.log_view.scroll_y < pane.log_view.max_scroll_y
            assert not pane.log_view.is_at_bottom

            # Incoming line while scrolled must not snap to bottom
            pane._on_line(LogLine(host="h1", container="c1", text="live after scroll"), gen=pane._generation)
            assert pane.view_state.state == FollowState.SCROLLED
            assert pane.log_view.scroll_y == target_y
            assert pane.view_state.pending_lines == 1

    asyncio.run(_run())


def test_log_pane_mouse_wheel_and_bottom_resume(monkeypatch):
    """Mouse wheel up decouples follow; wheel down to bottom resumes; paused stays paused."""
    import asyncio
    from textual import events

    lines = [LogLine(host="h1", container="c1", text=f"line {i}") for i in range(100)]
    target = FakeLogTarget(lines=lines)
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: target)

    async def _run():
        pane = LogPane(host="h1", cluster_id="cid1", container="c1", scope=LogScope.CONTAINER)
        app = _LogPaneTestApp(pane)
        async with app.run_test(size=(80, 20)) as pilot:
            await pilot.pause(0.1)
            assert pane.view_state.state == FollowState.FOLLOWING

            # Wheel up decouples to SCROLLED
            event_up = events.MouseScrollUp(pane.log_view, 10, 10, 0, 0, 0, False, False, False, screen_x=10, screen_y=10)
            pane.log_view.post_message(event_up)
            await pilot.pause(0.05)
            assert pane.view_state.state == FollowState.SCROLLED
            assert not pane.log_view.is_at_bottom

            # Wheel down to bottom resumes FOLLOWING
            for _ in range(50):
                event_down = events.MouseScrollDown(pane.log_view, 10, 10, 0, 0, 0, False, False, False, screen_x=10, screen_y=10)
                pane.log_view.post_message(event_down)
            await pilot.pause(0.05)
            assert pane.log_view.is_at_bottom is True
            assert pane.view_state.state == FollowState.FOLLOWING

            # Pause with space
            await pilot.press("space")
            assert pane.view_state.state == FollowState.PAUSED

            # Wheel up while paused stays PAUSED
            pane.log_view.post_message(event_up)
            await pilot.pause(0.05)
            assert pane.view_state.state == FollowState.PAUSED

            # Wheel down to bottom while paused stays PAUSED
            for _ in range(50):
                event_down = events.MouseScrollDown(pane.log_view, 10, 10, 0, 0, 0, False, False, False, screen_x=10, screen_y=10)
                pane.log_view.post_message(event_down)
            await pilot.pause(0.05)
            assert pane.log_view.is_at_bottom is True
            assert pane.view_state.state == FollowState.PAUSED

    asyncio.run(_run())


def test_log_view_no_op_up_when_no_scrollable_space(monkeypatch):
    """When content fits on screen, scroll up actions do not transition to SCROLLED."""
    import asyncio
    from textual import events

    lines = [LogLine(host="h1", container="c1", text=f"line {i}") for i in range(2)]
    target = FakeLogTarget(lines=lines)
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: target)

    async def _run():
        pane = LogPane(host="h1", cluster_id="cid1", container="c1", scope=LogScope.CONTAINER)
        app = _LogPaneTestApp(pane)
        async with app.run_test(size=(80, 20)) as pilot:
            await pilot.pause(0.1)
            assert pane.log_view.max_scroll_y <= 0
            assert pane.view_state.state == FollowState.FOLLOWING

            # Action scroll up does not decouple
            pane.log_view.action_scroll_up()
            assert pane.view_state.state == FollowState.FOLLOWING

            # Action page up does not decouple
            pane.log_view.action_page_up()
            assert pane.view_state.state == FollowState.FOLLOWING

            # Action scroll home does not decouple
            pane.log_view.action_scroll_home()
            assert pane.view_state.state == FollowState.FOLLOWING

            # Mouse wheel up does not decouple
            event_up = events.MouseScrollUp(pane.log_view, 10, 10, 0, 0, 0, False, False, False, screen_x=10, screen_y=10)
            pane.log_view.post_message(event_up)
            await pilot.pause(0.05)
            assert pane.view_state.state == FollowState.FOLLOWING

    asyncio.run(_run())


def test_log_pane_end_key_jumps_bottom_and_resumes_following(monkeypatch):
    """End key jumps to bottom: resumes FOLLOWING from SCROLLED, preserves PAUSED."""
    import asyncio

    lines = [LogLine(host="h1", container="c1", text=f"line {i}") for i in range(100)]
    target = FakeLogTarget(lines=lines)
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: target)

    async def _run():
        pane = LogPane(host="h1", cluster_id="cid1", container="c1", scope=LogScope.CONTAINER)
        app = _LogPaneTestApp(pane)
        async with app.run_test(size=(80, 20)) as pilot:
            await pilot.pause(0.1)
            assert pane.view_state.state == FollowState.FOLLOWING

            # Scroll up into history
            await pilot.press("pageup")
            await pilot.pause(0.05)
            assert pane.view_state.state == FollowState.SCROLLED
            assert pane.log_view.scroll_y < pane.log_view.max_scroll_y

            # Press 'end': reaches bottom and resumes FOLLOWING
            await pilot.press("end")
            await pilot.pause(0.1)
            assert pane.log_view.is_at_bottom is True
            assert pane.log_view.scroll_y == pane.log_view.max_scroll_y
            assert pane.view_state.state == FollowState.FOLLOWING

            # Pause with space
            await pilot.press("space")
            assert pane.view_state.state == FollowState.PAUSED

            # Scroll up while paused
            await pilot.press("pageup")
            await pilot.pause(0.05)
            assert pane.view_state.state == FollowState.PAUSED
            assert pane.log_view.scroll_y < pane.log_view.max_scroll_y

            # Press 'end' while paused: reaches bottom but PAUSED remains PAUSED
            await pilot.press("end")
            await pilot.pause(0.1)
            assert pane.log_view.is_at_bottom is True
            assert pane.log_view.scroll_y == pane.log_view.max_scroll_y
            assert pane.view_state.state == FollowState.PAUSED

    asyncio.run(_run())


@pytest.mark.parametrize("rate", [50, 200, 600])
def test_log_pane_live_stream_end_key_resumes_following_under_load(monkeypatch, rate):
    """Under sustained live streaming (50-600/s), pageup decouples and End reliably resumes FOLLOWING at bottom.

    While paused, End jumps to bottom but strictly preserves PAUSED without following new lines.
    """
    import asyncio
    import time

    class StreamingTarget(FakeLogTarget):
        def __init__(self, stream_rate: int):
            lines = [LogLine(host="h1", container="c1", text=f"initial {i}") for i in range(200)]
            super().__init__(lines=lines)
            self.stream_rate = stream_rate

        def follow(self, tail: int | None = 200, cancel: threading.Event | None = None):
            yield from super().follow(tail=tail, cancel=cancel)
            delay = 1.0 / self.stream_rate
            i = 0
            while cancel and not cancel.is_set():
                i += 1
                yield LogLine(host="h1", container="c1", text=f"live {i}")
                time.sleep(delay)

    target = StreamingTarget(stream_rate=rate)
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: target)

    async def _run():
        pane = LogPane(host="h1", cluster_id="cid1", container="c1", scope=LogScope.CONTAINER)
        app = _LogPaneTestApp(pane)
        async with app.run_test(size=(80, 20)):
            await asyncio.sleep(0.05)
            assert pane.view_state.state == FollowState.FOLLOWING

            # 1. User scrolls up away from live tail
            app.simulate_key("pageup")
            await asyncio.sleep(0.1)
            assert pane.view_state.state == FollowState.SCROLLED
            scrolled_y = pane.log_view.scroll_y
            assert scrolled_y < pane.log_view.max_scroll_y

            # 2. Press End while scrolled: must immediately transition to FOLLOWING and jump to bottom
            app.simulate_key("end")
            await asyncio.sleep(0.15)
            assert pane.view_state.state == FollowState.FOLLOWING
            assert pane.log_view.scroll_y == pane.log_view.max_scroll_y
            assert pane.view_state.pending_lines == 0

            # 3. Subsequent live lines must keep view anchored at the live bottom
            await asyncio.sleep(0.15)
            assert pane.view_state.state == FollowState.FOLLOWING
            assert pane.log_view.scroll_y == pane.log_view.max_scroll_y

            # 4. Pause streaming with space
            app.simulate_key("space")
            await asyncio.sleep(0.05)
            assert pane.view_state.state == FollowState.PAUSED

            # 5. Scroll up into history while paused
            app.simulate_key("pageup")
            await asyncio.sleep(0.05)
            assert pane.view_state.state == FollowState.PAUSED
            scrolled_paused_y = pane.log_view.scroll_y
            assert scrolled_paused_y < pane.log_view.max_scroll_y

            # 6. Press End while paused: jumps to bottom but remains PAUSED
            app.simulate_key("end")
            for _ in range(50):
                if pane.log_view.scroll_y > scrolled_paused_y:
                    break
                await asyncio.sleep(0.002)
            assert pane.view_state.state == FollowState.PAUSED
            paused_landed_y = pane.log_view.scroll_y
            assert paused_landed_y > scrolled_paused_y

            # 7. As new lines arrive while paused, view stays at paused position (no auto-follow)
            await asyncio.sleep(0.15)
            assert pane.view_state.state == FollowState.PAUSED
            assert pane.log_view.scroll_y == paused_landed_y
            assert pane.view_state.pending_lines > 0

    asyncio.run(_run())


def test_log_pane_cluster_scope_stream_responsive_under_load(monkeypatch):
    """CLUSTER scope under sustained multi-host streaming keeps keys responsive and follow intact."""
    import asyncio
    import time

    sources = [
        LogSource(host="h1", container="c0", role="node_0", rank=0),
        LogSource(host="h1", container="c1", role="node_1", rank=1),
        LogSource(host="h2", container="c2", role="node_2", rank=2),
        LogSource(host="h2", container="c3", role="node_3", rank=3),
    ]

    class ClusterStreamingTarget(FakeLogTarget):
        def __init__(self, stream_rate: int):
            lines = [
                LogLine(host=s.host, container=s.container, role=s.role, rank=s.rank, text=f"initial {s.container} {i}")
                for i in range(200)
                for s in sources
            ]
            super().__init__(sources=sources, lines=lines, runtime="vllm-distributed")
            self.stream_rate = stream_rate

        def follow(self, tail: int | None = 200, cancel: threading.Event | None = None):
            yield from super().follow(tail=tail, cancel=cancel)
            delay = 1.0 / self.stream_rate
            i = 0
            while cancel and not cancel.is_set():
                i += 1
                s = sources[i % len(sources)]
                yield LogLine(host=s.host, container=s.container, role=s.role, rank=s.rank, text=f"live {i}")
                time.sleep(delay)

    target = ClusterStreamingTarget(stream_rate=600)
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: target)

    async def _run():
        pane = LogPane(
            host="h1",
            cluster_id="cid1",
            container="c0",
            scope=LogScope.CLUSTER,
            summary=ScopeSummary(node_containers=2, cluster_containers=4, cluster_hosts=2),
        )
        app = _LogPaneTestApp(pane)
        async with app.run_test(size=(80, 20)):
            await asyncio.sleep(0.05)
            assert pane.view_state.state == FollowState.FOLLOWING
            assert pane.log_view.multi_source and pane.log_view.multi_host

            # Space pauses under load, measured with a bounded poll
            t0 = time.time()
            app.simulate_key("space")
            paused = False
            for _ in range(150):
                if pane.view_state.state == FollowState.PAUSED:
                    paused = True
                    break
                await asyncio.sleep(0.002)
            assert paused, "Space press must register under cluster-scope load"
            latency = (time.time() - t0) * 1000
            assert latency < 300, f"Pause latency was {latency:.1f}ms, expected < 300ms"

            # Space resumes following
            app.simulate_key("space")
            resumed = False
            for _ in range(150):
                if pane.view_state.state == FollowState.FOLLOWING:
                    resumed = True
                    break
                await asyncio.sleep(0.002)
            assert resumed

            # View stays anchored at the live bottom while lines keep arriving
            await asyncio.sleep(0.15)
            assert pane.view_state.state == FollowState.FOLLOWING
            assert pane.log_view.scroll_y == pane.log_view.max_scroll_y

            # The cluster scope streamed from every source
            seen = {(e.host, e.container) for e in pane.buffer._entries if isinstance(e, LogLine)}
            assert seen == {(s.host, s.container) for s in sources}

    asyncio.run(_run())


def test_log_pane_home_pageup_stay_scrolled_after_animation(monkeypatch):
    """home/pageup from the bottom stay SCROLLED after animation; incoming lines do not snap."""
    import asyncio

    lines = [LogLine(host="h1", container="c1", text=f"line {i}") for i in range(200)]

    class StreamingTarget(FakeLogTarget):
        def follow(self, tail: int | None = 200, cancel: threading.Event | None = None):
            yield from super().follow(tail=tail, cancel=cancel)
            while cancel and not cancel.is_set():
                time.sleep(0.01)

    target = StreamingTarget(lines=lines)
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: target)

    async def _assert_holds_scroll(pane: LogPane) -> None:
        assert pane.view_state.state == FollowState.SCROLLED
        scrolled_y = pane.log_view.scroll_y
        assert scrolled_y < pane.log_view.max_scroll_y
        pane._on_line(LogLine(host="h1", container="c1", text="live after upward scroll"), gen=pane._generation)
        assert pane.view_state.state == FollowState.SCROLLED
        assert pane.log_view.scroll_y == scrolled_y
        assert pane.log_view.scroll_y < pane.log_view.max_scroll_y

    async def _run():
        pane = LogPane(host="h1", cluster_id="cid1", container="c1", scope=LogScope.CONTAINER)
        app = _LogPaneTestApp(pane)
        async with app.run_test(size=(80, 20)) as pilot:
            await pilot.pause(0.1)
            assert pane.view_state.state == FollowState.FOLLOWING
            assert pane.log_view.is_at_bottom is True

            # home from the bottom: animation must not flip back to FOLLOWING
            await pilot.press("home")
            await pilot.wait_for_scheduled_animations()
            await _assert_holds_scroll(pane)

            # Resume, then pageup from the bottom
            await pilot.press("end")
            await pilot.wait_for_scheduled_animations()
            assert pane.view_state.state == FollowState.FOLLOWING
            assert pane.log_view.is_at_bottom is True

            await pilot.press("pageup")
            await pilot.wait_for_scheduled_animations()
            await _assert_holds_scroll(pane)

            # pageup immediately after an End-resume must also stay SCROLLED
            await pilot.press("end")
            await pilot.wait_for_scheduled_animations()
            assert pane.view_state.state == FollowState.FOLLOWING
            await pilot.press("pageup")
            await pilot.wait_for_scheduled_animations()
            await _assert_holds_scroll(pane)

            # home while PAUSED stays PAUSED after the animation settles
            await pilot.press("end")
            await pilot.wait_for_scheduled_animations()
            await pilot.press("space")
            assert pane.view_state.state == FollowState.PAUSED
            await pilot.press("home")
            await pilot.wait_for_scheduled_animations()
            assert pane.view_state.state == FollowState.PAUSED
            assert pane.log_view.scroll_y < pane.log_view.max_scroll_y
            paused_y = pane.log_view.scroll_y
            pane._on_line(LogLine(host="h1", container="c1", text="live while paused after home"), gen=pane._generation)
            assert pane.view_state.state == FollowState.PAUSED
            assert pane.log_view.scroll_y == paused_y

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# OptionList: bar visible from init
# ---------------------------------------------------------------------------


class _OptionListApp(App):
    def __init__(self, *options, **kwargs):
        super().__init__(**kwargs)
        self.options = options
        self.selected_index = None

    def compose(self) -> ComposeResult:
        yield OptionList(*self.options, id="test-list")

    def on_option_list_option_selected(self, event):
        self.selected_index = event.option_index


def test_option_list_highlight_bar_visible_from_init():
    async def _run():
        app = _OptionListApp("Row 0", "Row 1", "Row 2")
        async with app.run_test() as pilot:
            ol = app.query_one(OptionList)
            assert ol.highlighted == 0

            # No keypress: the highlighted row already renders with a distinct bgcolor
            styles0 = [seg.style for seg in ol.render_line(0) if seg.text.strip()]
            styles1 = [seg.style for seg in ol.render_line(1) if seg.text.strip()]
            assert styles0[0].bgcolor != styles1[0].bgcolor

            # First down moves the selection: nothing is swallowed to reveal it
            await pilot.press("down")
            assert ol.highlighted == 1

    import asyncio

    asyncio.run(_run())


def test_option_list_first_enter_selects_default_row():
    async def _run():
        app = _OptionListApp("Default 0", "Row 1", "Row 2")
        async with app.run_test() as pilot:
            ol = app.query_one(OptionList)
            assert ol.highlighted == 0
            styles0 = [seg.style for seg in ol.render_line(0) if seg.text.strip()]
            styles1 = [seg.style for seg in ol.render_line(1) if seg.text.strip()]
            assert styles0[0].bgcolor != styles1[0].bgcolor

            # Enter selects the default row with no prior reveal press
            await pilot.press("enter")
            assert app.selected_index == 0

    import asyncio

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Fake LogTarget & Test Apps
# ---------------------------------------------------------------------------


class FakeLogTarget:
    def __init__(
        self,
        sources: Sequence[LogSource] | None = None,
        lines: Sequence[LogLine] | None = None,
        history: dict[tuple[str, str], list[LogLine]] | None = None,
        error_on_follow: Exception | None = None,
        error_on_page: Exception | None = None,
        error_on_dump: Exception | None = None,
        runtime: str | None = None,
    ) -> None:
        self.sources = tuple(sources or [LogSource(host="h1", container="c1", role="head", rank=0)])
        self._lines = list(lines or [])
        self._history = history or {}
        self.runtime: str | None = runtime
        self.error_on_follow = error_on_follow
        self.error_on_page = error_on_page
        self.error_on_dump = error_on_dump
        self.follow_cancel: threading.Event | None = None
        self.page_calls: list[dict[str, Any]] = []
        self.dump_calls: list[dict[str, Any]] = []

    def follow(self, tail: int | None = 200, cancel: threading.Event | None = None) -> Iterator[LogLine]:
        self.follow_cancel = cancel
        if self.error_on_follow:
            raise self.error_on_follow
        for line in self._lines:
            if cancel and cancel.is_set():
                break
            yield line

    def page(self, source: LogSource, *, skip: int, count: int = 500) -> list[LogLine]:
        self.page_calls.append({"source": source, "skip": skip, "count": count})
        if self.error_on_page:
            raise self.error_on_page
        key = (source.host, source.container)
        hist = self._history.get(key, [])
        if skip >= len(hist):
            return []
        start = max(0, len(hist) - skip - count)
        end = len(hist) - skip
        return hist[start:end]

    def read(self, *, follow: bool = False, tail: int | None = 200, cancel: threading.Event | None = None) -> Iterator[LogLine]:
        if cancel and cancel.is_set():
            return
        yield from self._lines[-tail:] if tail else self._lines

    def dump(self, path: str, *, overwrite: bool = False, cancel: threading.Event | None = None) -> int:
        import os

        self.dump_calls.append({"path": path, "overwrite": overwrite})
        if self.error_on_dump:
            raise self.error_on_dump
        if os.path.exists(path) and not overwrite:
            raise SparkrunError(f"Target file {path!r} already exists")
        partial = f"{path}.partial"
        with open(partial, "w", encoding="utf-8") as f:
            for line in self._lines:
                if cancel and cancel.is_set():
                    try:
                        os.unlink(partial)
                    except OSError:
                        pass
                    return 0
                f.write(line.text + "\n")
        os.replace(partial, path)
        return len(self._lines)


class _LogPaneTestApp(App):
    def __init__(self, pane: LogPane, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.pane = pane
        self.closed = False
        self.dump_request = None
        # Stands in for the app-level background tracker the pane reads from.
        self.metrics_scraper = RuntimeMetricsScraper()

    def metrics_scraper_for(self, cluster_id: str) -> RuntimeMetricsScraper:
        return self.metrics_scraper

    def compose(self) -> ComposeResult:
        yield self.pane

    def on_log_pane_closed(self, event: LogPane.Closed) -> None:
        self.closed = True

    def on_log_pane_request_dump(self, event: LogPane.RequestDump) -> None:
        self.dump_request = event


class _WorkloadListTestApp(App):
    def __init__(self, wl: WorkloadList, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.wl = wl
        self.opened_log = None
        self.dump_request = None
        self.stop_request = None
        self.backed_to_hosts = False

    def compose(self) -> ComposeResult:
        yield self.wl

    def on_workload_list_open_log(self, event: WorkloadList.OpenLog) -> None:
        self.opened_log = (event.host, event.cluster_id, event.container)

    def on_workload_list_request_dump(self, event: WorkloadList.RequestDump) -> None:
        self.dump_request = (event.host, event.cluster_id, event.container, event.scope)

    def on_workload_list_request_stop(self, event: WorkloadList.RequestStop) -> None:
        self.stop_request = (event.host, event.cluster_id, event.container)

    def on_workload_list_back_to_hosts(self, event: WorkloadList.BackToHosts) -> None:
        self.backed_to_hosts = True


# ---------------------------------------------------------------------------
# LogView & LogPane Pilot Tests
# ---------------------------------------------------------------------------


def test_log_view_widget():
    import asyncio

    async def _run():
        buf = LogBuffer()
        for i in range(5):
            buf.append(LogLine(host="h1", container="c1", role="head", rank=0, text=f"\x1b[32mline {i}\x1b[0m"))

        view = LogView(buf)

        class ViewApp(App):
            def compose(self) -> ComposeResult:
                yield view

        app = ViewApp()
        async with app.run_test():
            assert view.virtual_size.height == 5
            assert view.is_at_bottom is True

            # Line rendering with color=True preserves ANSI styling
            strip0_color = view.render_line(0)
            assert any(seg.style.color and seg.style.color.number == 2 for seg in strip0_color)

            # Toggle color=False strips ANSI
            view.color = False
            view.refresh()
            strip0_nocolor = view.render_line(0)
            assert not any(seg.style.color and seg.style.color.number == 2 for seg in strip0_nocolor)

            # Search highlighting
            view.search_pattern = "line 0"
            view.highlight = True
            view.refresh()
            strip0_hl = view.render_line(0)
            rev_segs = [seg for seg in strip0_hl if seg.style.reverse]
            assert len(rev_segs) > 0

            view.highlight = False
            view.refresh()
            strip0_nohl = view.render_line(0)
            rev_segs_off = [seg for seg in strip0_nohl if seg.style.reverse]
            assert len(rev_segs_off) == 0

            # Toggle wrap
            view.wrap = True
            view.refresh_buffer()
            assert view.wrap is True
            assert len(view._wrap_map) >= 5

            # Incremental append
            new_line = LogLine(host="h1", container="c1", role="head", rank=0, text="line 5 appended")
            buf.append(new_line)
            view.on_entry_appended(new_line)
            assert len(view._wrap_map) >= 6

            # Scroll to line
            view.scroll_to_line(2)
            assert view.scroll_offset[1] <= len(view._wrap_map)

    asyncio.run(_run())


def test_log_pane_follow_and_controls(monkeypatch):
    import asyncio

    lines = [LogLine(host="h1", container="c1", role="head", rank=0, text=f"step {i}: status ok") for i in range(10)]
    target = FakeLogTarget(lines=lines)
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: target)

    async def _run():
        pane = LogPane(
            host="h1",
            cluster_id="cid123456789",
            container="c1",
            scope=LogScope.CONTAINER,
            summary=ScopeSummary(node_containers=1, cluster_containers=1, cluster_hosts=1),
        )
        app = _LogPaneTestApp(pane)
        async with app.run_test() as pilot:
            # Wait for follow worker to finish initial lines
            await pilot.pause(0.1)

            # 10 lines + stream ended marker
            assert len(pane.buffer) == 11
            assert pane.buffer.stream_ended is True

            # Title
            title = pane.query_one("#log-title", Static).content
            assert "cid12345" in str(title)
            assert "c1" in str(title)

            # Status bar
            status = str(pane.query_one("#log-status", Static).content)
            assert "following" in status
            assert "1 container c1" in status
            assert "color on" in status
            assert "highlight on" in status
            assert "wrap on" in status  # the pane opens wrapped

            # Color toggle 'c'
            await pilot.press("c")
            assert pane.log_view.color is False
            status = str(pane.query_one("#log-status", Static).content)
            assert "color off" in status

            # Highlight toggle 'h'
            await pilot.press("h")
            assert pane.log_view.highlight is False
            status = str(pane.query_one("#log-status", Static).content)
            assert "highlight off" in status

            # Wrap toggle 'l'
            await pilot.press("l")
            assert pane.log_view.wrap is False
            status = str(pane.query_one("#log-status", Static).content)
            assert "wrap off" in status

            # Search with '/'
            await pilot.press("slash")
            search_inp = pane.query_one("#log-search")
            assert search_inp.display is True
            assert search_inp.has_focus is True

            # Type pattern and press enter
            search_inp.value = "step 5"
            await pilot.press("enter")
            assert search_inp.display is False
            assert pane.buffer.search_pattern is not None
            assert len(pane.buffer.search_matches) == 1
            status = str(pane.query_one("#log-status", Static).content)
            assert "1 matches in loaded lines" in status

            # Jump with 'n' and 'b'
            await pilot.press("n")
            await pilot.press("b")

            # Invalid regex handling
            await pilot.press("slash")
            search_inp.value = "[unclosed("
            await pilot.press("enter")
            assert search_inp.display is True
            assert "Invalid regex:" in search_inp.value
            # Previous search matches preserved
            assert len(pane.buffer.search_matches) == 1

            # Escape closes search input and clears
            await pilot.press("escape")
            assert search_inp.display is False

            # Space toggles pause
            await pilot.press("space")
            assert pane.view_state.is_paused is True
            assert "paused" in str(pane.query_one("#log-status", Static).content)

            # Space again resumes following
            await pilot.press("space")
            assert pane.view_state.is_following is True
            assert "following" in str(pane.query_one("#log-status", Static).content)

            # 'd' opens dump modal
            await pilot.press("d")
            assert app.dump_request is not None
            assert app.dump_request.container == "c1"
            assert app.dump_request.scope == LogScope.CONTAINER

            # 'escape' closes log pane and sets cancel event
            assert not pane._cancel.is_set()
            await pilot.press("escape")
            assert app.closed is True
            assert pane._cancel.is_set()

    asyncio.run(_run())


def test_log_pane_scope_switching(monkeypatch):
    import asyncio

    opened_targets: list[dict[str, Any]] = []

    def fake_open(*args, **kwargs):
        t = FakeLogTarget(lines=[LogLine(host="h1", container="c1", role="head", rank=0, text="hello")])
        opened_targets.append(kwargs)
        return t

    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", fake_open)

    async def _run():
        summary = ScopeSummary(node_containers=2, cluster_containers=3, cluster_hosts=2)
        pane = LogPane(
            host="h1",
            cluster_id="cid1",
            container="c1",
            scope=LogScope.CONTAINER,
            summary=summary,
        )
        app = _LogPaneTestApp(pane)
        async with app.run_test() as pilot:
            await pilot.pause(0.1)
            assert len(opened_targets) == 1
            assert opened_targets[0]["container"] == "c1"

            # 1 -> 2: non-equivalent scope switch
            old_cancel = pane._cancel
            await pilot.press("2")
            await pilot.pause(0.1)
            assert old_cancel.is_set()  # previous follow worker cancelled
            assert pane.scope == LogScope.NODE
            assert len(opened_targets) == 2
            assert "container" not in opened_targets[1]
            assert opened_targets[1]["host"] == "h1"

            # Pressing '2' again (current scope): rule 0a least astonishment, no reload
            await pilot.press("2")
            await pilot.pause(0.05)
            assert len(opened_targets) == 2

            # Now test equivalent scope switch:
            pane.scope_summary = ScopeSummary(node_containers=1, cluster_containers=1, cluster_hosts=1)
            pane.scope = LogScope.CONTAINER
            # 1 -> 2 when node_containers == 1 is equivalent
            await pilot.press("2")
            await pilot.pause(0.05)
            # Scope updated in UI, but no new target opened
            assert pane.scope == LogScope.NODE
            assert len(opened_targets) == 2
            assert "2 node h1" in str(pane.query_one("#log-status", Static).content)

    asyncio.run(_run())


def test_log_pane_paging_and_windowing(monkeypatch):
    import asyncio

    src1 = LogSource(host="h1", container="c1", role="head", rank=0)
    src2 = LogSource(host="h1", container="c2", role="worker", rank=1)
    hist = {
        ("h1", "c1"): [LogLine(host="h1", container="c1", role="head", rank=0, text=f"c1 history {i}") for i in range(600)],
        ("h1", "c2"): [LogLine(host="h1", container="c2", role="worker", rank=1, text=f"c2 history {i}") for i in range(600)],
    }
    tail_lines = [
        LogLine(host="h1", container="c1", role="head", rank=0, text="c1 tail 0"),
        LogLine(host="h1", container="c2", role="worker", rank=1, text="c2 tail 0"),
    ]
    target = FakeLogTarget(sources=[src1, src2], lines=tail_lines, history=hist)
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: target)

    async def _run():
        pane = LogPane(
            host="h1",
            cluster_id="cid1",
            container="c1",
            scope=LogScope.NODE,
            summary=ScopeSummary(node_containers=2, cluster_containers=2, cluster_hosts=1),
        )
        app = _LogPaneTestApp(pane)
        async with app.run_test() as pilot:
            await pilot.pause(0.1)
            initial_count = len(pane.buffer)

            # Trigger page fetch via NearTop
            pane._trigger_page_fetch()
            await pilot.pause(0.1)

            # Page calls were made in reverse rank order (src2 then src1)
            assert len(target.page_calls) == 2
            assert target.page_calls[0]["source"].rank == 1
            assert target.page_calls[1]["source"].rank == 0

            # Buffer has prepended lines and separators
            assert len(pane.buffer) > initial_count

            # Exhaustion test: fetch remaining lines
            pane._trigger_page_fetch()
            await pilot.pause(0.1)

            # Both sources now returned < 500 lines so they are exhausted
            assert pane.buffer.is_exhausted(src1)
            assert pane.buffer.is_exhausted(src2)
            # START_OF_LOG_MARKER prepended
            assert START_OF_LOG_MARKER in pane.buffer._entries

            # Triggering near top again when exhausted does not make further page calls
            calls_before = len(target.page_calls)
            pane._trigger_page_fetch()
            await pilot.pause(0.05)
            assert len(target.page_calls) == calls_before

    asyncio.run(_run())


def test_log_pane_error_handling(monkeypatch):
    import asyncio

    target_err = FakeLogTarget(error_on_follow=SparkrunError("Host connection refused"))
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: target_err)

    async def _run():
        pane = LogPane(
            host="h1",
            cluster_id="cid1",
            container="c1",
            scope=LogScope.CONTAINER,
            summary=ScopeSummary(node_containers=1, cluster_containers=1, cluster_hosts=1),
        )
        app = _LogPaneTestApp(pane)
        async with app.run_test() as pilot:
            await pilot.pause(0.1)
            # Red error marker in buffer
            assert any("Host connection refused" in str(e) for e in pane.buffer._entries)

    asyncio.run(_run())


def test_log_pane_space_tail_reload(monkeypatch):
    import asyncio

    target = FakeLogTarget(lines=[LogLine(host="h1", container="c1", role="head", rank=0, text=f"line {i}") for i in range(20)])
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: target)

    async def _run():
        pane = LogPane(
            host="h1",
            cluster_id="cid1",
            container="c1",
            scope=LogScope.CONTAINER,
            summary=ScopeSummary(node_containers=1, cluster_containers=1, cluster_hosts=1),
        )
        app = _LogPaneTestApp(pane)
        async with app.run_test() as pilot:
            await pilot.pause(0.1)
            # Simulate bottom eviction
            pane.buffer.evicted_live = 5
            # Space triggers reload_tail
            await pilot.press("space")
            await pilot.pause(0.1)
            assert pane.buffer.evicted_live == 0
            assert pane.view_state.is_following is True

    asyncio.run(_run())


def test_workload_list_pilot():
    import asyncio

    rows = [
        WorkloadRow(key="header:cid1", text="cid1", is_header=True, cluster_id="cid1"),
        WorkloadRow(key="cid1:c1", text="  c1", is_header=False, cluster_id="cid1", container="c1"),
        WorkloadRow(key="cid1:c2", text="  c2", is_header=False, cluster_id="cid1", container="c2"),
    ]
    wl = WorkloadList(host="h1", rows=rows)
    app = _WorkloadListTestApp(wl)

    async def _run():
        async with app.run_test() as pilot:
            assert wl.highlighted == 1  # first container row highlighted by default

            # Bar visible at init: highlighted row differs from unhighlighted row 2
            styles1 = [seg.style for seg in wl.render_line(1) if seg.text.strip()]
            styles2 = [seg.style for seg in wl.render_line(2) if seg.text.strip()]
            assert styles1[0].bgcolor != styles2[0].bgcolor

            # First down moves to row 2 (second container); no reveal press consumed
            await pilot.press("down")
            assert wl.highlighted == 2

            # Enter selects container
            await pilot.press("enter")
            assert app.opened_log == ("h1", "cid1", "c2")

            # 'd' requests dump
            await pilot.press("d")
            assert app.dump_request == ("h1", "cid1", "c2", LogScope.CONTAINER)

            # 'escape' requests back to hosts
            await pilot.press("escape")
            assert app.backed_to_hosts is True

            # set_rows preserves highlighted key
            new_rows = [
                WorkloadRow(key="header:cid1", text="cid1", is_header=True, cluster_id="cid1"),
                WorkloadRow(key="cid1:c2", text="  c2 (updated)", is_header=False, cluster_id="cid1", container="c2"),
            ]
            wl.set_rows(new_rows, host="h1")
            assert wl.highlighted == 1
            assert wl._rows[1].key == "cid1:c2"

            # When container disappears, highlight moves to nearest selectable row
            wl.highlighted = 1
            disappeared_rows = [
                WorkloadRow(key="header:cid1", text="cid1", is_header=True, cluster_id="cid1"),
                WorkloadRow(key="cid1:c0", text="  c0", is_header=False, cluster_id="cid1", container="c0"),
                WorkloadRow(key="cid1:c3", text="  c3", is_header=False, cluster_id="cid1", container="c3"),
            ]
            wl.set_rows(disappeared_rows, host="h1")
            # Old index was 1, nearest selectable is index 1 (c0)
            assert wl.highlighted == 1
            assert wl._rows[1].key == "cid1:c0"

    asyncio.run(_run())


def test_workload_list_k_requests_stop():
    import asyncio

    rows = [
        WorkloadRow(key="header:cid1", text="cid1", is_header=True, cluster_id="cid1"),
        WorkloadRow(key="cid1:c1", text="  c1", is_header=False, cluster_id="cid1", container="c1"),
        WorkloadRow(key="cid1:c2", text="  c2", is_header=False, cluster_id="cid1", container="c2"),
    ]
    wl = WorkloadList(host="h1", rows=rows)
    app = _WorkloadListTestApp(wl)

    async def _run():
        async with app.run_test() as pilot:
            assert wl.highlighted == 1  # first container row highlighted by default
            await pilot.press("down")
            await pilot.press("k")
            assert app.stop_request == ("h1", "cid1", "c2")

    asyncio.run(_run())


def test_workload_list_k_ignores_header_only_rows():
    import asyncio

    rows = [WorkloadRow(key="empty", text="No workloads", is_header=True)]
    wl = WorkloadList(host="h1", rows=rows)
    app = _WorkloadListTestApp(wl)

    async def _run():
        async with app.run_test() as pilot:
            await pilot.press("k")
            assert app.stop_request is None

    asyncio.run(_run())


def test_workload_list_mouse_click_selects():
    import asyncio

    rows = [
        WorkloadRow(key="header:cid1", text="cid1", is_header=True, cluster_id="cid1"),
        WorkloadRow(key="cid1:c1", text="  c1", is_header=False, cluster_id="cid1", container="c1"),
        WorkloadRow(key="cid1:c2", text="  c2", is_header=False, cluster_id="cid1", container="c2"),
    ]
    wl = WorkloadList(host="h1", rows=rows)
    app = _WorkloadListTestApp(wl)

    async def _run():
        async with app.run_test() as pilot:
            # Option rows start one row below and stay right of the list's padding;
            # derive the click point from the widget's own gutter so the click lands
            # on the option's text (a click in the padding carries no option meta).
            gutter = wl.styles.gutter
            click_x = gutter.left + 1

            # Click on the second container row (option index 2) opens its logs
            await pilot.click(wl, offset=(click_x, gutter.top + 2))
            await pilot.pause(0.05)
            assert app.opened_log == ("h1", "cid1", "c2")
            assert wl.highlighted == 2

            # Clicking the disabled header row is a no-op
            app.opened_log = None
            await pilot.click(wl, offset=(click_x, gutter.top))
            await pilot.pause(0.05)
            assert app.opened_log is None

    asyncio.run(_run())


def test_log_pane_inflight_page_scope_switch(monkeypatch):
    """Scope switch mid-fetch resets _loading_page, cancels page worker, and paging still works."""
    import asyncio

    page_started = threading.Event()
    page_proceed = threading.Event()

    class BlockingPageTarget(FakeLogTarget):
        def page(self, source, *, skip, count=500):
            page_started.set()
            page_proceed.wait(timeout=2.0)
            return super().page(source, skip=skip, count=count)

    src1 = LogSource(host="h1", container="c1", role="head", rank=0)
    hist1 = {("h1", "c1"): [LogLine(host="h1", container="c1", text=f"h1 {i}") for i in range(100)]}
    hist2 = {("h1", "c1"): [LogLine(host="h1", container="c1", text=f"cluster {i}") for i in range(100)]}

    target1 = BlockingPageTarget(sources=[src1], lines=[LogLine(host="h1", container="c1", text="tail 1")], history=hist1)
    target2 = FakeLogTarget(sources=[src1], lines=[LogLine(host="h1", container="c1", text="tail 2")], history=hist2)

    opened = []

    def fake_open(*a, **kw):
        if kw.get("scope") == "head" or kw.get("container") == "c1":
            opened.append("target1")
            return target1
        opened.append("target2")
        return target2

    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", fake_open)

    async def _run():
        pane = LogPane(
            host="h1",
            cluster_id="cid1",
            container="c1",
            scope=LogScope.CONTAINER,
            summary=ScopeSummary(node_containers=1, cluster_containers=2, cluster_hosts=2),
        )
        app = _LogPaneTestApp(pane)
        async with app.run_test(size=(80, 20)) as pilot:
            await pilot.pause(0.1)
            assert pane.target is target1

            # Set an active search pattern
            pane.buffer.search("tail")
            assert pane.buffer.search_pattern is not None

            # Start a page fetch that blocks
            pane._trigger_page_fetch()
            await pilot.pause(0.05)
            assert pane._loading_page is True
            assert page_started.is_set()

            # Switch scope while page is in flight
            await pilot.press("3")
            await pilot.pause(0.1)

            # Unblock the old page worker
            page_proceed.set()
            await pilot.pause(0.1)

            # _loading_page reset, new scope opened, search pattern carried over
            assert pane._loading_page is False
            assert pane.scope == LogScope.CLUSTER
            assert pane.target is target2
            assert pane.buffer.search_pattern == re.compile("tail")

            # Subsequent page fetch works in new scope
            calls_before = len(target2.page_calls)
            pane._trigger_page_fetch()
            await pilot.pause(0.1)
            assert len(target2.page_calls) > calls_before
            assert pane._loading_page is False

    asyncio.run(_run())


def test_log_pane_wrapped_search_navigation_and_input(monkeypatch):
    """Wrapped rows do not cause search jumps to skip match entries."""
    import asyncio

    # Create lines where each line is 100 characters long
    # On an 80-column terminal, each line wraps to 2 virtual rows
    lines = []
    for i in range(40):
        marker = "MATCH" if i in (2, 6, 12, 18, 25) else "NOPE"
        lines.append(LogLine(host="h1", container="c1", text=f"entry {i} {marker} " + ("x" * 80)))

    target = FakeLogTarget(lines=lines)
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: target)

    async def _run():
        pane = LogPane(host="h1", cluster_id="cid1", container="c1", scope=LogScope.CONTAINER)
        app = _LogPaneTestApp(pane)
        async with app.run_test(size=(80, 20)) as pilot:
            await pilot.pause(0.1)
            assert pane.log_view.wrap is True  # wrapped by default; search runs in wrap mode

            # Open search input and type pattern
            await pilot.press("slash")
            search_inp = pane.query_one("#log-search", Input)
            assert search_inp.display is True
            assert search_inp.has_focus is True

            # Verify typing characters (including priority key characters like 'n', 'c')
            await pilot.press("M", "A", "T", "C", "H")
            assert search_inp.value == "MATCH"
            await pilot.press("enter")

            assert search_inp.display is False
            # Scroll to top to test forward navigation
            pane.log_view.scroll_to(y=0, animate=False)
            pane._set_current_match(None)
            await pilot.pause(0.1)

            expected_matches = list(pane.buffer.search_matches)

            # Press 'n' sequentially: must visit each match in order
            visited_entries = []
            for _ in range(len(expected_matches)):
                await pilot.press("n")
                curr_vy = int(pane.log_view.scroll_offset[1])
                curr_entry = pane.log_view.virtual_row_to_entry(curr_vy)
                visited_entries.append(curr_entry)

            assert visited_entries == expected_matches

            # Press 'b' sequentially: must visit in reverse
            rev_visited = []
            for _ in range(len(expected_matches)):
                await pilot.press("b")
                curr_vy = int(pane.log_view.scroll_offset[1])
                curr_entry = pane.log_view.virtual_row_to_entry(curr_vy)
                rev_visited.append(curr_entry)

            assert rev_visited == [
                expected_matches[-2],
                expected_matches[-3],
                expected_matches[-4],
                expected_matches[0],
                expected_matches[-1],
            ]

    asyncio.run(_run())


def test_log_pane_search_slash_inserts_into_open_input(monkeypatch):
    """A '/' typed into the open search input is inserted, not treated as 'open search'.

    Regression: the LogPane 'slash' binding was priority=True, so it fired before the
    focused Input could consume the key and action_open_search reset the value mid-typing.
    """
    import asyncio

    lines = [LogLine(host="h1", container="c1", role="head", rank=0, text=f"step {i}: status ok") for i in range(10)]
    target = FakeLogTarget(lines=lines)
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: target)

    async def _run():
        pane = LogPane(host="h1", cluster_id="cid1", container="c1", scope=LogScope.CONTAINER)
        app = _LogPaneTestApp(pane)
        async with app.run_test() as pilot:
            await pilot.pause(0.1)
            search_inp = pane.query_one("#log-search", Input)

            # '/' opens the search bar while the log view has focus
            await pilot.press("slash")
            assert search_inp.display is True
            assert search_inp.has_focus is True

            # Typing a pattern containing '/' must insert the character, not reset
            await pilot.press("f", "o", "o", "slash")
            await pilot.pause(0.05)
            assert search_inp.value == "foo/"
            assert search_inp.display is True
            assert search_inp.has_focus is True

            # The pattern still submits as a regex
            await pilot.press("enter")
            await pilot.pause(0.05)
            assert search_inp.display is False
            assert pane.buffer.search_pattern is not None
            assert pane.buffer.search_pattern.pattern == "foo/"

    asyncio.run(_run())


def test_log_pane_wrap_eviction_no_phantom_rows(monkeypatch):
    """Buffer eviction in wrap mode leaves no phantom rows in _wrap_map."""
    import asyncio
    import time

    lines = [LogLine(host="h1", container="c1", text=f"line {i} " + ("longtext " * 12)) for i in range(30)]

    class StreamingTarget(FakeLogTarget):
        def follow(self, tail: int | None = 200, cancel: threading.Event | None = None):
            yield from super().follow(tail=tail, cancel=cancel)
            while cancel and not cancel.is_set():
                time.sleep(0.01)

    target = StreamingTarget(lines=lines)
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: target)

    async def _run():
        pane = LogPane(host="h1", cluster_id="cid1", container="c1", scope=LogScope.CONTAINER)
        pane.CAP = 10  # Cap at 10 lines
        app = _LogPaneTestApp(pane)
        async with app.run_test(size=(80, 20)) as pilot:
            # Enable wrapping
            pane.log_view.wrap = True
            await pilot.pause(0.15)

            # Buffer has exactly 10 entries
            assert len(pane.buffer) == 10
            # Virtual size and live wrap map match the 10 entries exactly
            origin = pane.log_view._wrap_origin
            base = pane.log_view._wrap_base
            live_wrap = pane.log_view._wrap_map[origin:]
            assert len(live_wrap) == pane.log_view.virtual_size.height
            # Every live entry index in _wrap_map must be in range 0..9
            entry_indices = {e_idx - base for e_idx, _ in live_wrap}
            assert entry_indices == set(range(10))

            # Render each line: must succeed with no IndexError
            for y in range(len(live_wrap)):
                strip = pane.log_view.render_line(y)
                assert strip is not None

    asyncio.run(_run())


def test_log_pane_stale_generation_callbacks_ignored(monkeypatch):
    """Callbacks tagged with a stale generation are discarded."""
    import asyncio

    target = FakeLogTarget(lines=[LogLine(host="h1", container="c1", text="initial")])
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: target)

    async def _run():
        pane = LogPane(host="h1", cluster_id="cid1", container="c1", scope=LogScope.CONTAINER)
        app = _LogPaneTestApp(pane)
        async with app.run_test() as pilot:
            await pilot.pause(0.1)
            initial_count = len(pane.buffer)
            current_gen = pane._generation

            # Stale line callback
            stale_line = LogLine(host="h1", container="c1", text="stale line")
            pane._on_line(stale_line, gen=current_gen - 1)
            assert len(pane.buffer) == initial_count

            # Stale target ready callback
            bogus_target = FakeLogTarget(lines=[])
            pane._on_target_ready(bogus_target, gen=current_gen - 1)
            assert pane.target is not bogus_target

            # Stale page success callback
            pane._on_page_success([], bogus_target, gen=current_gen - 1)
            assert len(pane.buffer) == initial_count

    asyncio.run(_run())


def test_log_pane_wrapped_page_anchoring(monkeypatch):
    """When wrapped lines are prepended, scroll_y is anchored by the virtual row delta."""
    import asyncio

    # Tail has 20 lines (each 1 row)
    tail = [LogLine(host="h1", container="c1", text=f"tail {i}") for i in range(20)]
    # History has 100 lines that wrap to 3 virtual rows each (width 40, length 100)
    hist_lines = [LogLine(host="h1", container="c1", text=f"page {i} " + ("y" * 90)) for i in range(100)]
    target = FakeLogTarget(
        sources=[LogSource(host="h1", container="c1", role="head", rank=0)],
        lines=tail,
        history={("h1", "c1"): hist_lines},
    )
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: target)

    async def _run():
        pane = LogPane(host="h1", cluster_id="cid1", container="c1", scope=LogScope.CONTAINER)
        app = _LogPaneTestApp(pane)
        async with app.run_test(size=(40, 20)) as pilot:
            await pilot.pause(0.1)
            pane.log_view.wrap = True
            pane.log_view.refresh_buffer()

            # Scroll to virtual row 2
            pane.log_view.scroll_to(y=2, animate=False)
            assert pane.log_view.scroll_y == 2

            # Fetch page
            pane._trigger_page_fetch()
            await pilot.pause(0.15)

            # History lines prepended; verify scroll_y was shifted to anchor
            assert pane.log_view.scroll_y > 2
            # Verify the entry at the viewport top is still the same line ("tail 2")
            # 80 older lines + START_OF_LOG_MARKER were prepended, so original entry 2 is now at index 83
            anchor_entry = pane.log_view.virtual_row_to_entry(int(pane.log_view.scroll_y))
            assert anchor_entry == 83
            assert pane.buffer[anchor_entry].text == "tail 2"

    asyncio.run(_run())


def test_log_pane_opens_with_wrapping_enabled(monkeypatch):
    """A fresh pane wraps without any keypress: long engine lines stay readable."""
    import asyncio

    long_line = "INFO 10-17 08:12:34 [loggers.py:331] Engine 000: Avg prompt throughput: 123.4 tokens/s" + (" tail-marker" * 12)
    target = FakeLogTarget(lines=[LogLine(host="h1", container="c1", text=long_line)])
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: target)

    async def _run():
        pane = LogPane(host="h1", cluster_id="cid1", container="c1", scope=LogScope.CONTAINER)
        app = _LogPaneTestApp(pane)
        async with app.run_test(size=(80, 20)) as pilot:
            await pilot.pause(0.15)
            view = pane.log_view
            assert view.wrap is True  # default, no 'l' pressed
            assert "wrap on" in str(pane.query_one("#log-status", Static).content)

            # One long line occupies several virtual rows, and the overflow is
            # rendered rather than clipped off the right edge.
            assert view.virtual_size.height > len(pane.buffer)
            rows = ["".join(seg.text for seg in view.render_line(y)) for y in range(view.virtual_size.height)]
            assert any("tail-marker" in row for row in rows)

    asyncio.run(_run())


def test_log_view_rebuilds_wrap_map_once_per_settled_width(monkeypatch):
    """A width drag costs one rebuild; a height-only resize costs none."""
    import asyncio
    from textual import events as tevents
    from textual.geometry import Size as TSize

    target = FakeLogTarget(lines=[LogLine(host="h1", container="c1", text="x" * 300)])
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: target)

    async def _run():
        pane = LogPane(host="h1", cluster_id="cid1", container="c1", scope=LogScope.CONTAINER)
        app = _LogPaneTestApp(pane)
        async with app.run_test(size=(80, 20)) as pilot:
            await pilot.pause(0.15)
            view = pane.log_view
            # The mount-time width change armed a debounce of its own; let it land
            # before counting, or it is measured as a rebuild this test did not cause.
            for _ in range(50):
                if view._wrap_resize_timer is None:
                    break
                await asyncio.sleep(0.01)
            calls: list[int] = []
            real = view.refresh_buffer

            def counted():
                calls.append(1)
                real()

            monkeypatch.setattr(view, "refresh_buffer", counted)
            width, height = view.size.width, view.size.height
            view._wrap_for_width = width  # the map already matches this width

            for size in (TSize(width, height), TSize(width, height + 3)):
                view.on_resize(tevents.Resize(size, TSize(width, height)))
            await pilot.pause(0.25)
            assert calls == []  # same width, and height alone cannot change wrapping

            for w in (90, 96, 100, 104):  # a drag: many events, one settled width
                view.on_resize(tevents.Resize(TSize(w, height), TSize(width, height)))
            assert calls == []
            await pilot.pause(0.3)
            assert len(calls) == 1

    asyncio.run(_run())


def test_log_pane_pause_and_stream_end(monkeypatch):
    """Pause accumulates pending lines; stream end marker appended."""
    import asyncio

    class StreamingTarget(FakeLogTarget):
        def follow(self, tail=200, cancel=None):
            yield LogLine(host="h1", container="c1", text="line 0")
            yield LogLine(host="h1", container="c1", text="line 1")

    target = StreamingTarget()
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: target)

    async def _run():
        pane = LogPane(host="h1", cluster_id="cid1", container="c1", scope=LogScope.CONTAINER)
        app = _LogPaneTestApp(pane)
        async with app.run_test() as pilot:
            await pilot.pause(0.1)
            assert pane.view_state.is_following is True

            # Pause via space
            await pilot.press("space")
            assert pane.view_state.is_paused is True

            # Deliver new line while paused
            pane._on_line(LogLine(host="h1", container="c1", text="line 2"), gen=pane._generation)
            assert pane.view_state.pending_lines == 1
            assert "paused (+1)" in str(pane.query_one("#log-status", Static).content)

            # Stream ended while paused
            pane._on_stream_ended(gen=pane._generation)
            assert STREAM_ENDED_MARKER in pane.buffer._entries

            # Unpause via space -> resumes following
            await pilot.press("space")
            assert pane.view_state.is_following is True
            assert pane.view_state.pending_lines == 0

    asyncio.run(_run())


def test_log_pane_search_last_screen_and_following_wraparound(monkeypatch):
    """Search matches in the last screenful can be traversed with 'n'/'b' and wrap around."""
    import asyncio

    # 40 lines; matches at lines 6, 36, 38, 39
    lines = []
    for i in range(40):
        marker = "TARGET" if i in (6, 36, 38, 39) else "other"
        lines.append(LogLine(host="h1", container="c1", text=f"line {i} {marker}"))

    target = FakeLogTarget(lines=lines)
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: target)

    async def _run():
        pane = LogPane(host="h1", cluster_id="cid1", container="c1", scope=LogScope.CONTAINER)
        app = _LogPaneTestApp(pane)
        async with app.run_test(size=(80, 20)) as pilot:
            for _ in range(40):
                if len([e for e in pane.buffer._entries if isinstance(e, LogLine)]) >= 40:
                    break
                await pilot.pause(0.05)
            await pilot.pause(0.05)
            # Exhaust the empty fake history so the wrap-to-top 'n' cannot start a
            # page fetch (and shift search indices) mid-assertion.
            pane._trigger_page_fetch()
            for _ in range(40):
                if START_OF_LOG_MARKER in pane.buffer._entries:
                    break
                await pilot.pause(0.05)
            assert START_OF_LOG_MARKER in pane.buffer._entries

            # In default following state, user is at the bottom (lines ~20..39)
            assert pane.view_state.is_following is True
            assert pane.log_view.is_at_bottom is True

            # Open search and submit
            await pilot.press("slash")
            search_inp = pane.query_one("#log-search", Input)
            search_inp.value = "TARGET"
            await pilot.press("enter")
            await pilot.pause(0.05)

            assert len(pane.buffer.search_matches) == 4
            # Index-relative assertions: the START_OF_LOG_MARKER from history
            # exhaustion shifts absolute buffer indices, match order does not.
            matches = list(pane.buffer.search_matches)
            # While at bottom, the first match in/after the viewport was selected
            assert pane.log_view.current_match_entry in matches[1:]

            # Press 'n' to step to the next match
            await pilot.press("n")
            assert pane.log_view.current_match_entry == matches[2]
            status = str(pane.query_one("#log-status", Static).content)
            assert "match 3/4" in status

            # Press 'n' to step to the last match
            await pilot.press("n")
            assert pane.log_view.current_match_entry == matches[3]
            status = str(pane.query_one("#log-status", Static).content)
            assert "match 4/4" in status

            # Press 'n': wraps around back to the first match at the top!
            await pilot.press("n")
            assert pane.log_view.current_match_entry == matches[0]
            status = str(pane.query_one("#log-status", Static).content)
            assert "match 1/4" in status

            # Press 'b': wraps backwards to the last match!
            await pilot.press("b")
            assert pane.log_view.current_match_entry == matches[-1]
            assert pane.log_view.is_at_bottom is True
            status = str(pane.query_one("#log-status", Static).content)
            assert "match 4/4" in status

            # Press 'b': steps to the second-to-last match
            await pilot.press("b")
            assert pane.log_view.current_match_entry == matches[-2]
            status = str(pane.query_one("#log-status", Static).content)
            assert "match 3/4" in status

    asyncio.run(_run())


def test_log_pane_paging_enforces_cap_from_bottom(monkeypatch):
    """Paging older lines enforces the CAP by dropping newest lines from bottom."""
    import asyncio

    tail = [LogLine(host="h1", container="c1", text=f"tail {i}") for i in range(10)]
    hist = {("h1", "c1"): [LogLine(host="h1", container="c1", text=f"hist {i}") for i in range(100)]}
    target = FakeLogTarget(
        sources=[LogSource(host="h1", container="c1", role="head", rank=0)],
        lines=tail,
        history=hist,
    )
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: target)

    async def _run():
        pane = LogPane(host="h1", cluster_id="cid1", container="c1", scope=LogScope.CONTAINER)
        pane.CAP = 10
        app = _LogPaneTestApp(pane)
        async with app.run_test(size=(80, 20)) as pilot:
            await pilot.pause(0.1)
            # Buffer initially has 10 lines + stream ended marker
            assert len([e for e in pane.buffer._entries if isinstance(e, LogLine)]) == 10
            assert pane.buffer.evicted_live == 0

            # Trigger first page fetch (fetches 500 lines or up to history length 100)
            pane._trigger_page_fetch()
            await pilot.pause(0.15)

            # Cap must be strictly respected!
            assert len(pane.buffer) <= 10
            # Bottom lines must have been evicted into evicted_live
            assert pane.buffer.evicted_live > 0

            # Trigger another page fetch (if any remains)
            pane._trigger_page_fetch()
            await pilot.pause(0.15)
            assert len(pane.buffer) <= 10

            # Pressing space when evicted_live > 0 reloads tail
            await pilot.press("space")
            await pilot.pause(0.15)
            assert pane.buffer.evicted_live == 0
            assert pane.view_state.is_following is True

    asyncio.run(_run())


def test_trim_page_drift_uses_snapshot_counts_not_text():
    """Seam trim is proven by skip growth only; repeated text is not overlap."""
    heartbeats = [LogLine(host="h1", container="c1", text="heartbeat") for _ in range(10)]
    # No live growth: keep the whole page, including identical lines.
    assert _trim_page_drift(heartbeats, loaded_skip=15, snap_skip=15) == heartbeats
    assert _trim_page_drift(heartbeats, loaded_skip=15, snap_skip=20) == heartbeats

    unique = [LogLine(host="h1", container="c1", text=f"line {i}") for i in range(10)]
    # 3 live lines arrived: drop the last 3 regardless of text.
    trimmed = _trim_page_drift(unique, loaded_skip=53, snap_skip=50)
    assert [e.text for e in trimmed] == [f"line {i}" for i in range(7)]

    # Growth larger than the page drops everything (still no text matching).
    assert _trim_page_drift(unique, loaded_skip=80, snap_skip=50) == []


def test_log_pane_paging_repeated_lines_no_drift_keeps_history(monkeypatch):
    """Identical lines at the seam with no growth are kept, not treated as overlap."""
    import asyncio

    class UnifiedTarget(FakeLogTarget):
        def __init__(self, log_list: list[LogLine], **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.full_log = log_list

        def page(self, source: LogSource, *, skip: int, count: int = 500) -> list[LogLine]:
            self.page_calls.append({"source": source, "skip": skip, "count": count})
            total = len(self.full_log)
            start = max(0, total - skip - count)
            end = max(0, total - skip)
            return list(self.full_log[start:end])

    # 15 unique + 10 identical heartbeats + 10 unique = 35. Tail is last 15:
    # 5 heartbeats + 10 unique. Page (skip=15) is 15 unique + 5 heartbeats.
    full = [LogLine(host="h1", container="c1", text=f"unique {i}") for i in range(15)]
    full.extend(LogLine(host="h1", container="c1", text="heartbeat") for _ in range(10))
    full.extend(LogLine(host="h1", container="c1", text=f"tail {i}") for i in range(10))
    target = UnifiedTarget(
        log_list=full,
        sources=[LogSource(host="h1", container="c1", role="head", rank=0)],
        lines=full[-15:],
    )
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: target)

    async def _run():
        pane = LogPane(host="h1", cluster_id="cid1", container="c1", scope=LogScope.CONTAINER)
        app = _LogPaneTestApp(pane)
        async with app.run_test(size=(80, 20)) as pilot:
            for _ in range(40):
                n_lines = len([e for e in pane.buffer._entries if isinstance(e, LogLine)])
                if n_lines >= 15:
                    break
                await pilot.pause(0.05)
            assert len([e for e in pane.buffer._entries if isinstance(e, LogLine)]) == 15

            pane._trigger_page_fetch()
            await pilot.pause(0.15)

            texts = [e.text for e in pane.buffer._entries if isinstance(e, LogLine)]
            assert texts.count("heartbeat") == 10
            assert len(texts) == 35
            assert texts[:15] == [f"unique {i}" for i in range(15)]
            assert texts[-10:] == [f"tail {i}" for i in range(10)]

    asyncio.run(_run())


def test_log_pane_paging_live_growth_trims_by_skip_count(monkeypatch):
    """Live lines during an in-flight page trim only the proven growth count."""
    import asyncio

    class BlockingGrowTarget(FakeLogTarget):
        def __init__(self, log_list: list[LogLine], **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.full_log = log_list
            self.page_started = threading.Event()
            self.page_continue = threading.Event()

        def page(self, source: LogSource, *, skip: int, count: int = 500) -> list[LogLine]:
            self.page_calls.append({"source": source, "skip": skip, "count": count})
            total = len(self.full_log)
            start = max(0, total - skip - count)
            end = max(0, total - skip)
            return list(self.full_log[start:end])

    original = [LogLine(host="h1", container="c1", text=f"log line {i}") for i in range(100)]
    target = BlockingGrowTarget(
        log_list=list(original),
        sources=[LogSource(host="h1", container="c1", role="head", rank=0)],
        lines=original[50:],
    )
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: target)

    async def _run():
        pane = LogPane(host="h1", cluster_id="cid1", container="c1", scope=LogScope.CONTAINER)
        app = _LogPaneTestApp(pane)
        # Block the worker's count capture (2nd call: the 1st is the trigger
        # snapshot) so growth lands between the snapshot and the remote read.
        real_count = pane._get_delivered_count
        calls = {"n": 0}

        def gated_count(source):
            calls["n"] += 1
            if calls["n"] == 2:
                target.page_started.set()
                target.page_continue.wait(timeout=2)
            return real_count(source)

        monkeypatch.setattr(pane, "_get_delivered_count", gated_count)
        async with app.run_test(size=(80, 20)) as pilot:
            for _ in range(50):
                n_lines = len([e for e in pane.buffer._entries if isinstance(e, LogLine)])
                if n_lines >= 50:
                    break
                await pilot.pause(0.05)
            assert len([e for e in pane.buffer._entries if isinstance(e, LogLine)]) == 50

            pane._trigger_page_fetch()
            for _ in range(40):
                if target.page_started.is_set():
                    break
                await pilot.pause(0.05)
            assert target.page_started.is_set()

            growth = [LogLine(host="h1", container="c1", text=f"growth {i}") for i in range(7)]
            target.full_log.extend(growth)
            for line in growth:
                pane._on_line(line, gen=pane._generation)

            target.page_continue.set()
            for _ in range(40):
                if not pane._loading_page:
                    break
                await pilot.pause(0.05)

            texts = [e.text for e in pane.buffer._entries if isinstance(e, LogLine)]
            # 7 live lines proved the overlap; unique history 0..99 plus growth 0..6.
            assert texts.count("log line 50") == 1
            assert "log line 0" in texts
            assert "log line 99" in texts
            assert texts[-7:] == [f"growth {i}" for i in range(7)]
            assert len(texts) == 107

    asyncio.run(_run())


def test_log_pane_paging_post_read_growth_keeps_all_history_and_skip(monkeypatch):
    """Lines delivered after remote read completes do not cause data loss; skip equals buffer span."""
    import asyncio

    class PostReadGrowTarget(FakeLogTarget):
        def __init__(self, log_list: list[LogLine], **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.full_log = log_list
            self.read_done = threading.Event()
            self.release = threading.Event()

        def page(self, source: LogSource, *, skip: int, count: int = 500) -> list[LogLine]:
            self.page_calls.append({"source": source, "skip": skip, "count": count})
            total = len(self.full_log)
            start = max(0, total - skip - count)
            end = max(0, total - skip)
            result = list(self.full_log[start:end])
            self.read_done.set()
            self.release.wait(timeout=5)
            return result

    original = [LogLine(host="h1", container="c1", text=f"log line {i}") for i in range(100)]
    source = LogSource(host="h1", container="c1", role="head", rank=0)
    target = PostReadGrowTarget(
        log_list=list(original),
        sources=[source],
        lines=original[50:],
    )
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: target)

    async def _run():
        pane = LogPane(host="h1", cluster_id="cid1", container="c1", scope=LogScope.CONTAINER)
        app = _LogPaneTestApp(pane)
        async with app.run_test(size=(80, 20)) as pilot:
            for _ in range(50):
                n_lines = len([e for e in pane.buffer._entries if isinstance(e, LogLine)])
                if n_lines >= 50 and pane.target is not None:
                    break
                await pilot.pause(0.05)
            assert len([e for e in pane.buffer._entries if isinstance(e, LogLine)]) == 50
            assert pane.buffer.get_skip(source) == 50

            pane._trigger_page_fetch()
            for _ in range(40):
                if target.read_done.is_set():
                    break
                await pilot.pause(0.05)
            assert target.read_done.is_set()

            # 7 lines are written and delivered *after* the read completed:
            growth = [LogLine(host="h1", container="c1", text=f"live line {i}") for i in range(100, 107)]
            target.full_log.extend(growth)
            for line in growth:
                pane._on_line(line, gen=pane._generation)

            target.release.set()
            for _ in range(40):
                if not pane._loading_page:
                    break
                await pilot.pause(0.05)

            texts = [e.text for e in pane.buffer._entries if isinstance(e, LogLine)]
            # All 107 lines must be present: original 0..99 and live 100..106
            assert len(texts) == 107
            assert texts == [f"log line {i}" for i in range(100)] + [f"live line {i}" for i in range(100, 107)]
            # Skip must equal the buffer span
            assert pane.buffer.get_skip(source) == 107
            assert pane.buffer.get_skip(source) == len(texts)

    asyncio.run(_run())


def test_log_pane_paging_multi_source_drift_independence(monkeypatch):
    """In multi-source scopes, growth during earlier source reads does not drop lines for later sources."""
    import asyncio

    s0 = LogSource(host="h1", container="c0", role="head", rank=0)
    s1 = LogSource(host="h1", container="c1", role="worker", rank=1)

    class MultiSourceTarget(FakeLogTarget):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.logs = {
                (s0.host, s0.container): [LogLine(host="h1", container="c0", text=f"s0 line {i}") for i in range(100)],
                (s1.host, s1.container): [LogLine(host="h1", container="c1", text=f"s1 line {i}") for i in range(100)],
            }
            self.s1_read_done = threading.Event()
            self.s1_release = threading.Event()

        def page(self, source: LogSource, *, skip: int, count: int = 500) -> list[LogLine]:
            self.page_calls.append({"source": source, "skip": skip, "count": count})
            key = (source.host, source.container)
            full = self.logs[key]
            total = len(full)
            start = max(0, total - skip - count)
            end = max(0, total - skip)
            result = list(full[start:end])
            if source.container == "c1":
                self.s1_read_done.set()
                self.s1_release.wait(timeout=5)
            return result

    initial_lines = []
    # 50 lines for each source
    for i in range(50, 100):
        initial_lines.append(LogLine(host="h1", container="c0", text=f"s0 line {i}"))
        initial_lines.append(LogLine(host="h1", container="c1", text=f"s1 line {i}"))

    target = MultiSourceTarget(
        sources=[s0, s1],
        lines=initial_lines,
    )
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: target)

    async def _run():
        pane = LogPane(host="h1", cluster_id="cid1", container="c0", scope=LogScope.NODE)
        app = _LogPaneTestApp(pane)
        async with app.run_test(size=(80, 20)) as pilot:
            for _ in range(50):
                n_lines = len([e for e in pane.buffer._entries if isinstance(e, LogLine)])
                if n_lines >= 100 and pane.target is not None:
                    break
                await pilot.pause(0.05)
            assert pane.buffer.get_skip(s0) == 50
            assert pane.buffer.get_skip(s1) == 50

            pane._trigger_page_fetch()
            for _ in range(40):
                if target.s1_read_done.is_set():
                    break
                await pilot.pause(0.05)
            assert target.s1_read_done.is_set()

            # s1 has read lines 0..49.
            # Now simulate live lines arriving for both s0 and s1:
            growth_s0 = [LogLine(host="h1", container="c0", text=f"s0 live {i}") for i in range(100, 107)]
            growth_s1 = [LogLine(host="h1", container="c1", text=f"s1 live {i}") for i in range(100, 107)]
            target.logs[(s0.host, s0.container)].extend(growth_s0)
            target.logs[(s1.host, s1.container)].extend(growth_s1)
            for line in growth_s0 + growth_s1:
                pane._on_line(line, gen=pane._generation)

            target.s1_release.set()
            for _ in range(40):
                if not pane._loading_page:
                    break
                await pilot.pause(0.05)

            s0_lines = [e.text for e in pane.buffer._entries if isinstance(e, LogLine) and e.container == "c0"]
            s1_lines = [e.text for e in pane.buffer._entries if isinstance(e, LogLine) and e.container == "c1"]

            assert len(s0_lines) == 107
            assert len(s1_lines) == 107
            assert s0_lines == [f"s0 line {i}" for i in range(100)] + [f"s0 live {i}" for i in range(100, 107)]
            assert s1_lines == [f"s1 line {i}" for i in range(100)] + [f"s1 live {i}" for i in range(100, 107)]

            assert pane.buffer.get_skip(s0) == 107
            assert pane.buffer.get_skip(s1) == 107

    asyncio.run(_run())


def test_log_pane_no_restart_marker_from_mixed_clocks(monkeypatch):
    """File-mode pages with ISO text / mismatched timestamps do not insert a restart marker."""
    import asyncio

    class TimestampPageTarget(FakeLogTarget):
        def page(self, source: LogSource, *, skip: int, count: int = 500) -> list[LogLine]:
            # Real file-mode page reads have timestamp=None; ISO in the text is remote UTC.
            return [
                LogLine(
                    host="h1",
                    container="c1",
                    text="2026-10-11T08:00:00.000 older line",
                    timestamp=None,
                )
            ]

    tail = [
        LogLine(
            host="h1",
            container="c1",
            text="followed line",
            timestamp=2000.0,  # control-machine arrival time
        )
    ]
    target = TimestampPageTarget(
        sources=[LogSource(host="h1", container="c1", role="head", rank=0)],
        lines=tail,
    )
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: target)

    async def _run():
        pane = LogPane(host="h1", cluster_id="cid1", container="c1", scope=LogScope.CONTAINER)
        app = _LogPaneTestApp(pane)
        async with app.run_test(size=(80, 20)) as pilot:
            await pilot.pause(0.1)
            pane._trigger_page_fetch()
            await pilot.pause(0.15)
            assert LOG_RESTARTED_MARKER not in pane.buffer._entries
            texts = [e.text for e in pane.buffer._entries if isinstance(e, LogLine)]
            assert "2026-10-11T08:00:00.000 older line" in texts
            assert "followed line" in texts

    asyncio.run(_run())


def test_log_pane_search_cursor_survives_page_prepend(monkeypatch):
    """The current match stays on the same line after an older page is prepended."""
    import asyncio

    class UnifiedTarget(FakeLogTarget):
        def __init__(self, log_list: list[LogLine], **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.full_log = log_list

        def page(self, source: LogSource, *, skip: int, count: int = 500) -> list[LogLine]:
            self.page_calls.append({"source": source, "skip": skip, "count": count})
            total = len(self.full_log)
            start = max(0, total - skip - count)
            end = max(0, total - skip)
            return list(self.full_log[start:end])

    # 800 lines so history is larger than one 500-line page. Matches every 50.
    full = [LogLine(host="h1", container="c1", text=f"{'err' if i % 50 == 0 else 'ok'} {i}") for i in range(800)]
    target = UnifiedTarget(
        log_list=full,
        sources=[LogSource(host="h1", container="c1", role="head", rank=0)],
        lines=full[-200:],
    )
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: target)

    async def _run():
        pane = LogPane(host="h1", cluster_id="cid1", container="c1", scope=LogScope.CONTAINER)
        app = _LogPaneTestApp(pane)
        async with app.run_test(size=(80, 20)) as pilot:
            for _ in range(50):
                n_lines = len([e for e in pane.buffer._entries if isinstance(e, LogLine)])
                if n_lines >= 200:
                    break
                await pilot.pause(0.05)
            assert len([e for e in pane.buffer._entries if isinstance(e, LogLine)]) >= 200

            pane.buffer.search("err")
            pane.log_view.search_pattern = pane.buffer.search_pattern
            pane._set_current_match(len(pane.buffer.search_matches) - 1)
            before_idx = pane._current_match_idx
            before_entry = pane.log_view.current_match_entry
            before_text = pane.buffer[before_entry].text
            assert before_text.startswith("err ")
            before_count = len(pane.buffer.search_matches)

            pane._trigger_page_fetch()
            for _ in range(40):
                if not pane._loading_page:
                    break
                await pilot.pause(0.05)

            after_entry = pane.log_view.current_match_entry
            assert after_entry is not None
            assert pane.buffer[after_entry].text == before_text
            after_count = len(pane.buffer.search_matches)
            assert after_count > before_count
            assert pane._current_match_idx == before_idx + (after_count - before_count)
            status = str(pane.query_one("#log-status", Static).content)
            assert f"match {pane._current_match_idx + 1}/{after_count}" in status

    asyncio.run(_run())


def test_log_pane_search_cursor_survives_or_clears_on_top_eviction(monkeypatch):
    """Top eviction shifts the current match, or clears it when that line is gone."""
    import asyncio

    lines = [LogLine(host="h1", container="c1", text=f"{'err' if i in (5, 15) else 'ok'} {i}") for i in range(20)]
    target = FakeLogTarget(lines=lines)
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: target)

    async def _run():
        pane = LogPane(host="h1", cluster_id="cid1", container="c1", scope=LogScope.CONTAINER)
        pane.CAP = 20
        app = _LogPaneTestApp(pane)
        async with app.run_test(size=(80, 20)) as pilot:
            await pilot.pause(0.1)
            pane.buffer.search("err")
            pane.log_view.search_pattern = pane.buffer.search_pattern

            # Cursor on err 5, then live lines at the cap evict it from the top.
            pane._set_current_match(0)
            assert pane.buffer[pane.log_view.current_match_entry].text == "err 5"
            pane.log_view.scroll_to_end()
            for i in range(20, 40):
                pane.log_view.scroll_to_end()
                pane._on_line(LogLine(host="h1", container="c1", text=f"ok {i}"), gen=pane._generation)
            assert pane.log_view.current_match_entry is None
            assert pane._current_match_idx is None
            assert pane._current_match_abs is None
            status = str(pane.query_one("#log-status", Static).content)
            assert "(match " not in status
            assert "err 5" not in [e.text for e in pane.buffer._entries if isinstance(e, LogLine)]

    asyncio.run(_run())

    async def _run_kept():
        pane = LogPane(host="h1", cluster_id="cid1", container="c1", scope=LogScope.CONTAINER)
        pane.CAP = 20
        app = _LogPaneTestApp(pane)
        async with app.run_test(size=(80, 20)) as pilot:
            await pilot.pause(0.1)
            pane.buffer.search("err")
            pane.log_view.search_pattern = pane.buffer.search_pattern
            pane._set_current_match(1)
            assert pane.buffer[pane.log_view.current_match_entry].text == "err 15"
            pane.log_view.scroll_to_end()
            for i in range(20, 28):
                pane.log_view.scroll_to_end()
                pane._on_line(LogLine(host="h1", container="c1", text=f"ok {i}"), gen=pane._generation)
            after_entry = pane.log_view.current_match_entry
            assert after_entry is not None
            assert pane.buffer[after_entry].text == "err 15"
            assert pane._current_match_idx == 0
            status = str(pane.query_one("#log-status", Static).content)
            assert "match 1/" in status

    asyncio.run(_run_kept())


# ---------------------------------------------------------------------------
# DumpLogScreen Pilot Tests
# ---------------------------------------------------------------------------


class _DumpModalTestApp(App):
    def __init__(self, screen: DumpLogScreen, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.modal = screen
        self.result = None

    def on_mount(self) -> None:
        self.push_screen(self.modal, callback=self._on_result)

    def _on_result(self, res: Any) -> None:
        self.result = res


class _StopModalTestApp(App):
    def __init__(self, screen: StopWorkloadScreen, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.modal = screen
        self.result = None

    def on_mount(self) -> None:
        self.push_screen(self.modal, callback=self._on_result)

    def _on_result(self, res: Any) -> None:
        self.result = res


def test_stop_modal_arrow_keys_move_between_buttons():
    import asyncio

    async def _run():
        app = _StopModalTestApp(StopWorkloadScreen(host="h1", cluster_id="cid1", container="c1"))
        async with app.run_test() as pilot:
            await pilot.pause(0.05)
            cancel = app.modal.query_one("#stop-cancel")
            confirm = app.modal.query_one("#stop-confirm")
            assert app.focused is cancel

            # The y/n shortcuts are gone: the button row is the whole interface.
            await pilot.press("y")
            await pilot.pause(0.05)
            assert app.result is None
            assert isinstance(app.screen, StopWorkloadScreen)

            await pilot.press("right")
            await pilot.pause(0.05)
            assert app.focused is confirm
            await pilot.press("enter")
            await pilot.pause(0.05)
            assert app.result is True

        # Left/up walk back, so the row is navigable in both directions.
        app = _StopModalTestApp(StopWorkloadScreen(host="h1", cluster_id="cid1", container="c1"))
        async with app.run_test() as pilot:
            await pilot.pause(0.05)
            await pilot.press("right")
            await pilot.pause(0.05)
            assert app.focused is app.modal.query_one("#stop-confirm")
            await pilot.press("left")
            await pilot.pause(0.05)
            assert app.focused is app.modal.query_one("#stop-cancel")

    asyncio.run(_run())


def test_stop_modal_buttons_confirm_and_cancel():
    import asyncio

    async def _run():
        # Cancel is focused on open, so a stray enter cancels rather than stops.
        app = _StopModalTestApp(StopWorkloadScreen(host="h1", cluster_id="cid1", container="c1"))
        async with app.run_test() as pilot:
            await pilot.pause(0.05)
            assert app.focused is app.modal.query_one("#stop-cancel")
            await pilot.press("enter")
            await pilot.pause(0.05)
            assert app.result is False

        # Mouse paths: the buttons dismiss with the same verdict as the keys.
        app = _StopModalTestApp(StopWorkloadScreen(host="h1", cluster_id="cid1", container="c1"))
        async with app.run_test() as pilot:
            await pilot.pause(0.05)
            await pilot.click("#stop-cancel")
            await pilot.pause(0.05)
            assert app.result is False

        app = _StopModalTestApp(StopWorkloadScreen(host="h1", cluster_id="cid1", container="c1"))
        async with app.run_test() as pilot:
            await pilot.pause(0.05)
            await pilot.click("#stop-confirm")
            await pilot.pause(0.05)
            assert app.result is True

        # Keyboard activation of the Stop button.
        app = _StopModalTestApp(StopWorkloadScreen(host="h1", cluster_id="cid1", container="c1"))
        async with app.run_test() as pilot:
            await pilot.pause(0.05)
            app.modal.query_one("#stop-confirm").focus()
            await pilot.press("enter")
            await pilot.pause(0.05)
            assert app.result is True

    asyncio.run(_run())


def _relative_luminance(color: Color) -> float:
    def channel(value: int) -> float:
        c = value / 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * channel(color.r) + 0.7152 * channel(color.g) + 0.0722 * channel(color.b)


def _contrast_ratio(a: Color, b: Color) -> float:
    hi, lo = sorted((_relative_luminance(a), _relative_luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def test_stop_modal_cancel_button_contrasts_with_the_dialog():
    import asyncio

    async def _run():
        app = _StopModalTestApp(StopWorkloadScreen(host="h1", cluster_id="cid1", container="c1"))
        async with app.run_test() as pilot:
            await pilot.pause(0.05)
            dialog = app.modal.query_one("#stop-dialog")
            cancel = app.modal.query_one("#stop-cancel")
            background = dialog.styles.background
            # The dialog sits on a neutral elevated shade: readable against the
            # theme foreground, and distinct from the buttons' own $surface fill.
            assert background == Color.parse(app.theme_variables["background-lighten-2"])
            assert background != Color.parse(app.theme_variables["panel"])
            assert background != cancel.styles.background
            assert _contrast_ratio(background, Color.parse(app.theme_variables["foreground"])) >= 7.0

    asyncio.run(_run())


def test_stop_modal_buttons_are_centered_in_the_dialog():
    import asyncio

    async def _run():
        for size in ((120, 40), (80, 24)):
            app = _StopModalTestApp(StopWorkloadScreen(host="h1", cluster_id="cid1", container="c1"))
            async with app.run_test(size=size) as pilot:
                await pilot.pause(0.05)
                dialog = app.modal.query_one("#stop-dialog")
                row = app.modal.query_one("#stop-buttons")
                cancel = app.modal.query_one("#stop-cancel")
                confirm = app.modal.query_one("#stop-confirm")

                for button, label in ((cancel, "Cancel"), (confirm, "Stop")):
                    assert dialog.region.contains_region(button.region), size
                    # Wide enough for the whole label, no ellipsis.
                    assert button.size.width >= len(label), size
                assert cancel.region.right <= confirm.region.x, size

                left_gap = cancel.region.x - row.region.x
                right_gap = row.region.right - confirm.region.right
                assert abs(left_gap - right_gap) <= 2, (size, left_gap, right_gap)

    asyncio.run(_run())


def test_stop_modal_cancel_keys():
    import asyncio

    async def _run():
        for key in ("escape", "q"):
            app = _StopModalTestApp(StopWorkloadScreen(host="h1", cluster_id="cid1", container="c1"))
            async with app.run_test() as pilot:
                await pilot.pause(0.05)
                await pilot.press(key)
                await pilot.pause(0.05)
                assert app.result is False
                assert not isinstance(app.screen, StopWorkloadScreen)

    asyncio.run(_run())


def test_dump_modal_single_distinct_scope(tmp_path, monkeypatch):
    import asyncio
    import os

    summary = ScopeSummary(node_containers=1, cluster_containers=1, cluster_hosts=1)
    target = FakeLogTarget(lines=[LogLine(host="h1", container="c1", text="line 1")])
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: target)

    out_file = str(tmp_path / "dump.log")

    async def _run():
        screen = DumpLogScreen(
            host="h1",
            cluster_id="cid1",
            container="c1",
            summary=summary,
        )
        app = _DumpModalTestApp(screen)
        async with app.run_test() as pilot:
            await pilot.pause(0.05)
            # Scope list is omitted when only one scope
            assert len(screen.query("#dump-scope-list")) == 0
            assert screen.query_one("#dump-path-input", Input).has_focus

            # Enter dumps to target file
            screen.query_one("#dump-path-input", Input).value = out_file
            await pilot.press("enter")
            await pilot.pause(0.1)

            assert os.path.exists(out_file)
            assert app.result == out_file
            assert len(target.dump_calls) == 1
            assert target.dump_calls[0]["overwrite"] is False

    asyncio.run(_run())


def test_dump_modal_distinct_scopes_navigation_and_selection(tmp_path, monkeypatch):
    import asyncio

    summary = ScopeSummary(node_containers=2, cluster_containers=3, cluster_hosts=2)
    target = FakeLogTarget(lines=[LogLine(host="h1", container="c1", text="line 1")])
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: target)

    async def _run():
        screen = DumpLogScreen(
            host="h1",
            cluster_id="cid1",
            container="c1",
            default_scope=LogScope.CONTAINER,
            summary=summary,
        )
        app = _DumpModalTestApp(screen)
        async with app.run_test() as pilot:
            await pilot.pause(0.05)
            scope_list = screen.query_one("#dump-scope-list", OptionList)
            assert scope_list.has_focus
            # Bar visible on the default row (0) with no keypress
            styles0 = [seg.style for seg in scope_list.render_line(0) if seg.text.strip()]
            styles1 = [seg.style for seg in scope_list.render_line(1) if seg.text.strip()]
            assert styles0[0].bgcolor != styles1[0].bgcolor
            assert scope_list.highlighted == 0

            # First down moves it to row 1 (Node): no reveal press consumed
            await pilot.press("down")
            assert scope_list.highlighted == 1

            # Enter selects Node (scope 2) and focuses path input
            await pilot.press("enter")
            assert screen.chosen_scope == LogScope.NODE
            path_input = screen.query_one("#dump-path-input", Input)
            assert path_input.has_focus
            assert "-h1-" in path_input.value

    asyncio.run(_run())


def test_dump_modal_scope_equivalence_drops_redundant_scope():
    import asyncio

    # 2 containers on h1, none on other hosts -> cluster scope (3) is equivalent to node (2)
    summary = ScopeSummary(node_containers=2, cluster_containers=2, cluster_hosts=1)
    assert distinct_scopes(summary) == [LogScope.CONTAINER, LogScope.NODE]

    async def _run():
        # Opened in scope 3 (Cluster), canonical is scope 2 (Node)
        screen = DumpLogScreen(
            host="h1",
            cluster_id="cid1",
            container="c1",
            default_scope=LogScope.CLUSTER,
            summary=summary,
        )
        app = _DumpModalTestApp(screen)
        async with app.run_test() as pilot:
            await pilot.pause(0.05)
            scope_list = screen.query_one("#dump-scope-list", OptionList)
            # Scope 3 is omitted, 2 options remain
            assert len(scope_list._options) == 2
            # Default row is 1 (Node)
            assert scope_list.highlighted == 1

            # Single enter without arrow selects default row 1 (Node)
            await pilot.press("enter")
            assert screen.chosen_scope == LogScope.NODE
            assert screen.query_one("#dump-path-input", Input).has_focus

    asyncio.run(_run())


def test_dump_modal_path_regeneration_unless_edited():
    import asyncio

    summary = ScopeSummary(node_containers=2, cluster_containers=3, cluster_hosts=2)

    async def _run():
        screen = DumpLogScreen(
            host="h1",
            cluster_id="cid1",
            container="c1",
            summary=summary,
        )
        app = _DumpModalTestApp(screen)
        async with app.run_test() as pilot:
            await pilot.pause(0.05)
            path_input = screen.query_one("#dump-path-input", Input)
            init_val = path_input.value
            assert "-c1-" in init_val

            # Select Node (scope 2) -> path regenerates
            await pilot.press("down")  # move to 1 (Node)
            await pilot.press("enter")
            assert path_input.value != init_val
            assert "-h1-" in path_input.value and "-c1-" not in path_input.value

            # User edits path
            path_input.value = "my_custom_dump.log"
            await pilot.pause(0.05)
            assert screen._path_edited is True

            # Escape in input goes back to scope list
            await pilot.press("escape")
            scope_list = screen.query_one("#dump-scope-list", OptionList)
            assert scope_list.has_focus

            # Select Cluster (scope 3)
            await pilot.press("down")  # move to 2 (Cluster)
            await pilot.press("enter")
            # Custom path is preserved, not overwritten by default
            assert path_input.value == "my_custom_dump.log"

    asyncio.run(_run())


def test_dump_modal_overwrite_list_cancel_and_overwrite(tmp_path, monkeypatch):
    import asyncio

    exist_file = tmp_path / "exist.log"
    exist_file.write_text("old content\n")

    target = FakeLogTarget(lines=[LogLine(host="h1", container="c1", text="new line")])
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: target)

    summary = ScopeSummary(node_containers=1, cluster_containers=1, cluster_hosts=1)

    # 1. Single enter cancels
    async def _run_cancel():
        screen = DumpLogScreen(
            host="h1",
            cluster_id="cid1",
            container="c1",
            summary=summary,
        )
        app = _DumpModalTestApp(screen)
        async with app.run_test() as pilot:
            await pilot.pause(0.05)
            screen.query_one("#dump-path-input", Input).value = str(exist_file)
            await pilot.press("enter")
            await pilot.pause(0.05)

            overwrite_list = screen.query_one("#dump-overwrite-list", OptionList)
            assert overwrite_list.display is True
            # Bar visible on the default Cancel row (1) with no keypress
            styles1 = [seg.style for seg in overwrite_list.render_line(1) if seg.text.strip()]
            styles0 = [seg.style for seg in overwrite_list.render_line(0) if seg.text.strip()]
            assert styles1[0].bgcolor != styles0[0].bgcolor
            assert overwrite_list.highlighted == 1  # Cancel is default row
            assert overwrite_list.has_focus

            # Editing path after overwrite prompt hides overwrite list
            path_input = screen.query_one("#dump-path-input", Input)
            path_input.value = str(exist_file) + "_changed"
            await pilot.pause(0.05)
            assert overwrite_list.display is False
            assert screen._pending_dump_path is None

            # Reset back to existing file to test cancel
            path_input.value = str(exist_file)
            await pilot.press("enter")
            await pilot.pause(0.05)
            assert overwrite_list.display is True
            assert overwrite_list.has_focus

            # Single enter cancels
            await pilot.press("enter")
            await pilot.pause(0.05)
            assert app.result is None
            assert exist_file.read_text() == "old content\n"

    asyncio.run(_run_cancel())

    # 2. up + enter overwrites
    async def _run_overwrite():
        screen = DumpLogScreen(
            host="h1",
            cluster_id="cid1",
            container="c1",
            summary=summary,
        )
        app = _DumpModalTestApp(screen)
        async with app.run_test() as pilot:
            await pilot.pause(0.05)
            screen.query_one("#dump-path-input", Input).value = str(exist_file)
            await pilot.press("enter")
            await pilot.pause(0.05)

            overwrite_list = screen.query_one("#dump-overwrite-list", OptionList)
            assert overwrite_list.display is True

            # up moves from Cancel (1) to Overwrite (0): no reveal press consumed
            await pilot.press("up")
            assert overwrite_list.highlighted == 0

            # enter confirms overwrite
            await pilot.press("enter")
            await pilot.pause(0.1)

            assert app.result == str(exist_file)
            assert exist_file.read_text() == "new line\n"
            assert target.dump_calls[-1]["overwrite"] is True

    asyncio.run(_run_overwrite())


def test_dump_modal_directory_error_display():
    import asyncio

    summary = ScopeSummary(node_containers=1, cluster_containers=1, cluster_hosts=1)

    async def _run():
        screen = DumpLogScreen(
            host="h1",
            cluster_id="cid1",
            container="c1",
            summary=summary,
        )
        app = _DumpModalTestApp(screen)
        async with app.run_test() as pilot:
            await pilot.pause(0.05)
            screen.query_one("#dump-path-input", Input).value = "/nonexistent_dir_12345/foo.log"
            await pilot.press("enter")
            await pilot.pause(0.05)

            err = str(screen.query_one("#dump-error", Static).content)
            assert "Directory '/nonexistent_dir_12345' does not exist" in err
            assert app.result is None

    asyncio.run(_run())


def test_dump_modal_ok_reenabled_after_worker_error(tmp_path, monkeypatch):
    import asyncio

    out_file = tmp_path / "recovered.log"
    target = FakeLogTarget(lines=[LogLine(host="h1", container="c1", text="after error")])
    calls = {"count": 0}

    def _open(*_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise SparkrunError("transport went away")
        return target

    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", _open)
    summary = ScopeSummary(node_containers=1, cluster_containers=1, cluster_hosts=1)

    async def _run():
        screen = DumpLogScreen(host="h1", cluster_id="cid1", container="c1", summary=summary)
        app = _DumpModalTestApp(screen)
        async with app.run_test() as pilot:
            await pilot.pause(0.05)
            ok = screen.query_one("#dump-ok", Button)
            screen.query_one("#dump-path-input", Input).value = str(out_file)
            await pilot.click("#dump-ok")
            for _ in range(40):
                if not screen._dumping and "transport went away" in str(screen.query_one("#dump-error", Static).content):
                    break
                await pilot.pause(0.05)
            assert "transport went away" in str(screen.query_one("#dump-error", Static).content)
            # A failed dump must not leave OK dead: the corrected path starts
            # through the same button. The short pause lets the re-enable settle
            # before the click is dispatched.
            assert ok.disabled is False
            await pilot.pause(0.15)
            await pilot.click("#dump-ok")
            for _ in range(40):
                if app.result is not None:
                    break
                await pilot.pause(0.05)
            assert app.result == str(out_file)
            assert "after error" in out_file.read_text()

    asyncio.run(_run())


def test_dump_modal_buttons_submit_and_cancel(tmp_path, monkeypatch):
    import asyncio

    out_file = tmp_path / "button-dump.log"
    target = FakeLogTarget(lines=[LogLine(host="h1", container="c1", text="button line")])
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: target)
    summary = ScopeSummary(node_containers=1, cluster_containers=1, cluster_hosts=1)

    async def _run():
        screen = DumpLogScreen(host="h1", cluster_id="cid1", container="c1", summary=summary)
        app = _DumpModalTestApp(screen)
        async with app.run_test() as pilot:
            await pilot.pause(0.05)
            assert str(screen.query_one("#dump-cancel", Button).label) == "Cancel"
            assert str(screen.query_one("#dump-ok", Button).label) == "OK"
            screen.query_one("#dump-path-input", Input).value = str(out_file)
            await pilot.click("#dump-ok")
            for _ in range(40):
                if app.result is not None:
                    break
                await pilot.pause(0.05)
            assert app.result == str(out_file)
            assert "button line" in out_file.read_text()

    asyncio.run(_run())

    async def _run_cancel():
        screen = DumpLogScreen(host="h1", cluster_id="cid1", container="c1", summary=summary)
        app = _DumpModalTestApp(screen)
        async with app.run_test() as pilot:
            await pilot.pause(0.05)
            await pilot.click("#dump-cancel")
            await pilot.pause(0.05)
            assert app.result is None
            assert not isinstance(app.screen, DumpLogScreen)

    asyncio.run(_run_cancel())


def test_dump_modal_escape_during_dump_cancels_immediately(tmp_path, monkeypatch):
    import asyncio
    import time

    dump_calls = []

    class _BlockingTarget:
        def dump(self, path, overwrite=False, cancel=None):
            dump_calls.append((path, cancel))
            while cancel and not cancel.is_set():
                time.sleep(0.02)
            return 0

    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: _BlockingTarget())

    # Multi-container summary so that scope list exists
    summary = ScopeSummary(node_containers=2, cluster_containers=2, cluster_hosts=1)

    async def _run():
        screen = DumpLogScreen(
            host="h1",
            cluster_id="cid1",
            container="c1",
            summary=summary,
        )
        app = _DumpModalTestApp(screen)
        async with app.run_test() as pilot:
            await pilot.pause(0.05)
            # Advance from scope list to path input
            await pilot.press("enter")
            await pilot.pause(0.05)
            inp = screen.query_one("#dump-path-input", Input)
            assert inp.has_focus

            out_file = tmp_path / "out.log"
            inp.value = str(out_file)
            await pilot.pause(0.05)

            # Start dump
            await pilot.press("enter")
            await pilot.pause(0.1)
            assert screen._dumping is True
            assert len(dump_calls) == 1

            # Escape during active dump cancels immediately and dismisses modal
            await pilot.press("escape")
            await pilot.pause(0.1)
            assert screen._cancel.is_set()
            assert app.result is None
            assert not isinstance(app.screen, DumpLogScreen)

    asyncio.run(_run())


def test_dump_modal_late_success_after_escape_is_ignored(tmp_path, monkeypatch):
    import asyncio
    import threading

    class _Target:
        def dump(self, path, overwrite=False, cancel=None):
            return 1

    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: _Target())
    summary = ScopeSummary(node_containers=2, cluster_containers=2, cluster_hosts=1)

    async def _run():
        screen = DumpLogScreen(host="h1", cluster_id="cid1", container="c1", summary=summary)
        app = _DumpModalTestApp(screen)
        async with app.run_test() as pilot:
            await pilot.pause(0.05)
            await pilot.press("enter")
            await pilot.pause(0.05)
            path_input = screen.query_one("#dump-path-input", Input)
            path_input.value = str(tmp_path / "out.log")

            queued = threading.Event()
            success_callbacks = []
            notifications = []
            original_call_from_thread = app.call_from_thread

            def defer_success(callback, *args, **kwargs):
                if getattr(callback, "__name__", "") == "_on_success":
                    success_callbacks.append((callback, args, kwargs))
                    queued.set()
                else:
                    original_call_from_thread(callback, *args, **kwargs)

            monkeypatch.setattr(app, "call_from_thread", defer_success)
            monkeypatch.setattr(app, "notify", lambda message, *a, **kw: notifications.append(message))

            await pilot.press("enter")
            for _ in range(50):
                if queued.is_set():
                    break
                await pilot.pause(0.02)
            assert queued.is_set()
            assert len(success_callbacks) == 1

            await pilot.press("escape")
            await pilot.pause(0.05)
            assert screen._cancel.is_set()
            assert app.result is None
            assert not isinstance(app.screen, DumpLogScreen)

            callback, args, kwargs = success_callbacks[0]
            callback(*args, **kwargs)
            assert notifications == []
            assert app.result is None

    asyncio.run(_run())


def test_dump_modal_blocks_second_submit_and_scope_change_while_dumping(tmp_path, monkeypatch):
    import asyncio
    import time

    dump_calls = []

    class _BlockingTarget:
        def dump(self, path, overwrite=False, cancel=None):
            dump_calls.append((path, cancel))
            while cancel and not cancel.is_set():
                time.sleep(0.02)
            return 0

    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: _BlockingTarget())

    summary = ScopeSummary(node_containers=2, cluster_containers=2, cluster_hosts=1)

    async def _run():
        screen = DumpLogScreen(
            host="h1",
            cluster_id="cid1",
            container="c1",
            summary=summary,
        )
        app = _DumpModalTestApp(screen)
        async with app.run_test() as pilot:
            await pilot.pause(0.05)
            # Advance from scope list to path input
            await pilot.press("enter")
            await pilot.pause(0.05)
            inp = screen.query_one("#dump-path-input", Input)

            out_file = tmp_path / "out.log"
            inp.value = str(out_file)
            await pilot.pause(0.05)

            # Start dump
            await pilot.press("enter")
            await pilot.pause(0.1)
            assert screen._dumping is True
            assert len(dump_calls) == 1
            assert inp.disabled is True
            scope_list = screen.query_one("#dump-scope-list", OptionList)
            assert scope_list.disabled is True

            # Second submit while dumping is blocked
            await pilot.press("enter")
            screen.on_input_submitted(Input.Submitted(inp, inp.value))
            await pilot.pause(0.05)
            assert len(dump_calls) == 1

            # Clean up dump
            screen._cancel.set()
            await pilot.pause(0.05)

    asyncio.run(_run())


def test_log_pane_space_pauses_during_high_speed_stream(monkeypatch):
    """Pressing space reliably pauses high-speed streaming on the first press and resumes on second."""
    import asyncio
    import time

    class HighSpeedTarget(FakeLogTarget):
        def follow(self, tail: int | None = 200, cancel: threading.Event | None = None):
            i = 0
            while cancel and not cancel.is_set() and i < 2000:
                i += 1
                yield LogLine(host="h1", container="c1", text=f"stream {i}")
                time.sleep(0.001)

    target = HighSpeedTarget()
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: target)

    async def _run():
        pane = LogPane(host="h1", cluster_id="cid1", container="c1", scope=LogScope.CONTAINER)
        app = _LogPaneTestApp(pane)
        async with app.run_test(size=(80, 20)):
            await asyncio.sleep(0.05)
            assert pane.view_state.state == FollowState.FOLLOWING

            # Press space via simulate_key while streaming at 500+/s
            t0 = time.time()
            app.simulate_key("space")
            paused = False
            for _ in range(100):
                if pane.view_state.state == FollowState.PAUSED:
                    paused = True
                    break
                await asyncio.sleep(0.002)
            assert paused, "Space key must immediately pause follow on first press"
            latency = (time.time() - t0) * 1000
            assert latency < 300, f"Space pause latency was {latency:.1f}ms, expected < 300ms"

            # Deliver more lines while paused: pending lines must accumulate
            await asyncio.sleep(0.05)
            assert pane.view_state.state == FollowState.PAUSED
            assert pane.view_state.pending_lines > 0

            # Press space again: resumes following, resets pending count, scrolls to end
            app.simulate_key("space")
            resumed = False
            for _ in range(50):
                if pane.view_state.state == FollowState.FOLLOWING:
                    resumed = True
                    break
                await asyncio.sleep(0.002)
            assert resumed, "Second space key must resume follow"
            assert pane.view_state.pending_lines == 0

    asyncio.run(_run())


def test_log_pane_scroll_up_decouples_from_live_stream(monkeypatch):
    """Pressing Up arrow immediately decouples from follow and incoming lines don't snap back."""
    import asyncio
    import time

    class HighSpeedTarget(FakeLogTarget):
        def follow(self, tail: int | None = 200, cancel: threading.Event | None = None):
            i = 0
            while cancel and not cancel.is_set() and i < 2000:
                i += 1
                yield LogLine(host="h1", container="c1", text=f"stream {i}")
                time.sleep(0.001)

    target = HighSpeedTarget()
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: target)

    async def _run():
        pane = LogPane(host="h1", cluster_id="cid1", container="c1", scope=LogScope.CONTAINER)
        app = _LogPaneTestApp(pane)
        async with app.run_test(size=(80, 20)):
            # Wait until buffer has several pages (> 60 lines) so up doesn't start near the top
            for _ in range(100):
                if len(pane.buffer) >= 60 and pane.log_view.max_scroll_y > 30:
                    break
                await asyncio.sleep(0.005)
            assert pane.view_state.state == FollowState.FOLLOWING
            assert pane.log_view.max_scroll_y > 30

            # Send 'up' key press
            app.simulate_key("up")
            scrolled = False
            for _ in range(50):
                if pane.view_state.state == FollowState.SCROLLED:
                    scrolled = True
                    break
                await asyncio.sleep(0.002)
            assert scrolled, "Up key must immediately decouple to SCROLLED state"
            saved_scroll_y = pane.log_view.scroll_y

            # Allow more lines to stream in
            await asyncio.sleep(0.05)

            # View stays scrolled, position didn't snap to bottom
            assert pane.view_state.state == FollowState.SCROLLED
            assert pane.log_view.scroll_y <= saved_scroll_y + 1
            assert pane.log_view.scroll_y < pane.log_view.max_scroll_y

            # Pressing space returns to FOLLOWING and jumps to bottom
            app.simulate_key("space")
            resumed = False
            for _ in range(50):
                if pane.view_state.state == FollowState.FOLLOWING:
                    resumed = True
                    break
                await asyncio.sleep(0.002)
            assert resumed
            assert pane.view_state.pending_lines == 0
            # The stream is still appending: follow re-anchors on its next paint,
            # so give it one instead of sampling at an arbitrary instant.
            for _ in range(50):
                if pane.log_view.is_at_bottom:
                    break
                await asyncio.sleep(0.002)
            assert pane.log_view.is_at_bottom

    asyncio.run(_run())


def test_log_pane_hotkeys_responsive_during_streaming(monkeypatch):
    """Hotkeys (c, 1, escape) respond immediately during live log streaming."""
    import asyncio
    import time

    class StreamingTarget(FakeLogTarget):
        def follow(self, tail: int | None = 200, cancel: threading.Event | None = None):
            i = 0
            while cancel and not cancel.is_set() and i < 2000:
                i += 1
                yield LogLine(host="h1", container="c1", text=f"stream {i}")
                time.sleep(0.001)

    target = StreamingTarget()
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: target)

    async def _run():
        pane = LogPane(host="h1", cluster_id="cid1", container="c1", scope=LogScope.CONTAINER)
        app = _LogPaneTestApp(pane)
        async with app.run_test(size=(80, 20)):
            await asyncio.sleep(0.05)

            # 1. Hotkey 'c' (toggle color)
            init_color = pane.log_view.color
            app.simulate_key("c")
            color_toggled = False
            for _ in range(50):
                if pane.log_view.color != init_color:
                    color_toggled = True
                    break
                await asyncio.sleep(0.002)
            assert color_toggled, "Hotkey 'c' must immediately toggle color"

            # 2. Hotkey '1' (switch scope to CONTAINER)
            pane.scope = LogScope.CLUSTER
            app.simulate_key("1")
            scope_switched = False
            for _ in range(50):
                if pane.scope == LogScope.CONTAINER:
                    scope_switched = True
                    break
                await asyncio.sleep(0.002)
            assert scope_switched, "Hotkey '1' must switch scope to CONTAINER"

            # 3. Hotkey 'slash' then 'escape' (open and cancel search)
            search_input = pane.query_one("#log-search", Input)
            assert not search_input.display
            app.simulate_key("slash")
            search_opened = False
            for _ in range(50):
                if search_input.display and search_input.has_focus:
                    search_opened = True
                    break
                await asyncio.sleep(0.002)
            assert search_opened, "Hotkey '/' must open and focus search input"

            # Press escape while search is open -> closes search, returns focus to log_view
            app.simulate_key("escape")
            search_closed = False
            for _ in range(50):
                if not search_input.display and pane.log_view.has_focus:
                    search_closed = True
                    break
                await asyncio.sleep(0.002)
            assert search_closed, "Hotkey 'escape' while search is open must close search input"

            # Press escape while search is closed -> closes pane
            assert not app.closed
            app.simulate_key("escape")
            pane_closed = False
            for _ in range(50):
                if app.closed:
                    pane_closed = True
                    break
                await asyncio.sleep(0.002)
            assert pane_closed, "Hotkey 'escape' while search is closed must close pane"

    asyncio.run(_run())


def test_log_pane_unthrottled_stream_bounded_deque_and_latency(monkeypatch):
    """Under unthrottled streaming, incoming queue remains bounded and key presses register."""
    import asyncio
    import time

    class UnthrottledTarget(FakeLogTarget):
        def follow(self, tail: int | None = 200, cancel: threading.Event | None = None):
            i = 0
            while cancel and not cancel.is_set() and i < 15000:
                i += 1
                yield LogLine(host="h1", container="c1", text=f"unthrottled line {i}")

    target = UnthrottledTarget()
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: target)

    async def _run():
        pane = LogPane(host="h1", cluster_id="cid1", container="c1", scope=LogScope.CONTAINER)
        app = _LogPaneTestApp(pane)
        async with app.run_test(size=(80, 20)):
            await asyncio.sleep(0.03)

            # Ingress queue must be strictly bounded <= MAX_INCOMING_LINES
            with pane._incoming_lock:
                assert len(pane._incoming_lines) <= pane.MAX_INCOMING_LINES

            # Press space to pause under extreme load
            t0 = time.time()
            app.simulate_key("space")
            paused = False
            for _ in range(100):
                if pane.view_state.state == FollowState.PAUSED:
                    paused = True
                    break
                await asyncio.sleep(0.002)
            assert paused, "Space press must register even during unthrottled burst"
            latency = (time.time() - t0) * 1000
            assert latency < 300, f"Pause latency was {latency:.1f}ms, expected < 300ms"

            # Press space to resume
            app.simulate_key("space")
            resumed = False
            for _ in range(100):
                if pane.view_state.state == FollowState.FOLLOWING:
                    resumed = True
                    break
                await asyncio.sleep(0.002)
            assert resumed

            # Press up during unthrottled burst
            app.simulate_key("up")
            scrolled = False
            for _ in range(100):
                if pane.view_state.state == FollowState.SCROLLED:
                    scrolled = True
                    break
                await asyncio.sleep(0.002)
            assert scrolled, "Up press must immediately decouple to SCROLLED during unthrottled stream"

            # Press escape while search is closed to close pane
            app.simulate_key("escape")
            closed = False
            for _ in range(100):
                if app.closed:
                    closed = True
                    break
                await asyncio.sleep(0.002)
            assert closed, "Escape press must close pane during unthrottled burst"

            # Queue remains bounded
            with pane._incoming_lock:
                assert len(pane._incoming_lines) <= pane.MAX_INCOMING_LINES

    asyncio.run(_run())


def test_log_pane_cluster_scope_unthrottled_burst_bounded(monkeypatch):
    """CLUSTER scope under unthrottled multi-source streaming: queue bounded, keys register."""
    import asyncio
    import time

    sources = [LogSource(host=f"h{1 + i // 4}", container=f"c{i}", role=f"node_{i % 4}", rank=i % 4) for i in range(8)]

    class ClusterUnthrottledTarget(FakeLogTarget):
        def __init__(self):
            super().__init__(sources=sources, runtime="vllm-distributed")

        def follow(self, tail: int | None = 200, cancel: threading.Event | None = None):
            i = 0
            while cancel and not cancel.is_set():
                i += 1
                s = sources[i % len(sources)]
                yield LogLine(host=s.host, container=s.container, role=s.role, rank=s.rank, text=f"unthrottled line {i}")

    target = ClusterUnthrottledTarget()
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: target)

    async def _run():
        pane = LogPane(
            host="h1",
            cluster_id="cid1",
            container="c0",
            scope=LogScope.CLUSTER,
            summary=ScopeSummary(node_containers=4, cluster_containers=8, cluster_hosts=2),
        )
        app = _LogPaneTestApp(pane)
        async with app.run_test(size=(80, 20)):
            await asyncio.sleep(0.05)

            # Ingress queue must be strictly bounded <= MAX_INCOMING_LINES
            with pane._incoming_lock:
                assert len(pane._incoming_lines) <= pane.MAX_INCOMING_LINES

            t0 = time.time()
            app.simulate_key("space")
            paused = False
            for _ in range(150):
                if pane.view_state.state == FollowState.PAUSED:
                    paused = True
                    break
                await asyncio.sleep(0.002)
            assert paused, "Space press must register during unthrottled cluster burst"
            latency = (time.time() - t0) * 1000
            assert latency < 300, f"Pause latency was {latency:.1f}ms, expected < 300ms"

            app.simulate_key("space")
            resumed = False
            for _ in range(150):
                if pane.view_state.state == FollowState.FOLLOWING:
                    resumed = True
                    break
                await asyncio.sleep(0.002)
            assert resumed

            app.simulate_key("up")
            scrolled = False
            for _ in range(150):
                if pane.view_state.state == FollowState.SCROLLED:
                    scrolled = True
                    break
                await asyncio.sleep(0.002)
            assert scrolled, "Up press must decouple during unthrottled cluster burst"

            app.simulate_key("escape")
            closed = False
            for _ in range(150):
                if app.closed:
                    closed = True
                    break
                await asyncio.sleep(0.002)
            assert closed, "Escape must close pane during unthrottled cluster burst"

            # Queue remains bounded
            with pane._incoming_lock:
                assert len(pane._incoming_lines) <= pane.MAX_INCOMING_LINES

    asyncio.run(_run())


def test_log_pane_stream_paint_gate(monkeypatch):
    """Streaming repaints are rate-limited: immediate after idle, one paint per gate window."""
    import asyncio

    target = FakeLogTarget(lines=[])
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: target)

    async def _run():
        pane = LogPane(host="h1", cluster_id="cid1", container="c1", scope=LogScope.CONTAINER)
        app = _LogPaneTestApp(pane)
        async with app.run_test(size=(80, 20)):
            await asyncio.sleep(0.05)
            pane.buffer._entries.clear()
            gen = pane._generation
            view = pane.log_view
            # The wrap map tracks entries; a raw clear() on the buffer (test shortcut)
            # leaves it stale, and heights below are compared against len(buffer).
            view.refresh_buffer()

            # 1) Arrival after an idle gap paints immediately (gate clock backdated).
            view._stream_paint_at = time.monotonic() - 1.0
            pane._on_line(LogLine(host="h1", container="c1", text="first"), gen=gen, from_follow=True)
            assert view._stream_paint_armed is False
            stamped = view._stream_paint_at
            assert view.virtual_size.height == len(pane.buffer)

            # 2) Arrivals inside the window are deferred: gate armed, no repaint yet.
            batch = [LogLine(host="h1", container="c1", text=f"burst {i}") for i in range(100)]
            pane._on_lines(batch, gen=gen, from_follow=True)
            assert view._stream_paint_armed is True
            assert view._stream_paint_at == stamped
            assert view.virtual_size.height == len(pane.buffer) - 100

            # 3) Window elapses: exactly one deferred paint accounts for every line.
            await asyncio.sleep(0.15)
            assert view._stream_paint_armed is False
            assert view._stream_paint_at > stamped
            assert view.virtual_size.height == len(pane.buffer)

    asyncio.run(_run())


def test_log_pane_stream_paint_drops_when_writer_backlogged(monkeypatch):
    """A saturated writer queue drops the paint (current state only is lost) and retries."""
    import asyncio

    target = FakeLogTarget(lines=[])
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: target)

    async def _run():
        pane = LogPane(host="h1", cluster_id="cid1", container="c1", scope=LogScope.CONTAINER)
        app = _LogPaneTestApp(pane)
        async with app.run_test(size=(80, 20)):
            await asyncio.sleep(0.05)
            gen = pane._generation
            view = pane.log_view

            # Baseline paint works (headless writer depth is 0).
            view._stream_paint_at = 0.0
            pane._on_line(LogLine(host="h1", container="c1", text="x"), gen=gen, from_follow=True)
            assert view.virtual_size.height == len(pane.buffer)
            assert view._stream_paint_armed is False

            # Saturated queue: the paint is dropped, a retry is armed, geometry stays stale.
            monkeypatch.setattr(type(view), "_writer_queue_depth", lambda self: 30)
            view._stream_paint_at = 0.0
            pane._on_line(LogLine(host="h1", container="c1", text="y"), gen=gen, from_follow=True)
            assert view._stream_paint_armed is True
            assert view.virtual_size.height == len(pane.buffer) - 1

            # Queue drains: the armed retry paints the current state.
            monkeypatch.setattr(type(view), "_writer_queue_depth", lambda self: 0)
            await asyncio.sleep(0.15)
            assert view._stream_paint_armed is False
            assert view.virtual_size.height == len(pane.buffer)

    asyncio.run(_run())


def test_log_pane_on_lines_batch_eviction_wrap_correctness(monkeypatch):
    """Batch eviction with bottom eviction > batch size clamps slices and preserves wrap rows."""
    import asyncio

    target = FakeLogTarget(lines=[])
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: target)

    async def _run():
        pane = LogPane(host="h1", cluster_id="cid1", container="c1", scope=LogScope.CONTAINER)
        app = _LogPaneTestApp(pane)
        async with app.run_test(size=(80, 20)):
            await asyncio.sleep(0.05)
            pane.buffer._entries.clear()
            pane.CAP = 30
            pane.log_view.wrap = True

            # Pre-fill buffer with 30 lines
            for i in range(30):
                pane.buffer.append(LogLine(host="h1", container="c1", text=f"existing line {i}"))
            pane.log_view.refresh_buffer()
            assert len(pane.buffer) == 30
            assert pane.log_view._wrap_row_count() == 30

            # Simulate LOADING_MARKER prepended during older page fetch
            pane.buffer.prepend_page([LOADING_MARKER])
            pane.log_view.refresh_buffer()
            assert len(pane.buffer) == 31

            # User scrolled near top (viewport near 0 forces bottom eviction)
            pane.view_state.state = FollowState.SCROLLED
            pane.log_view.scroll_to(y=0, animate=False)

            # Deliver a batch of 3 lines: len=34, cap=30 -> 4 lines evicted from bottom
            batch = [LogLine(host="h1", container="c1", text=f"batch line {j}") for j in range(3)]
            pane._on_lines(batch, gen=pane._generation, from_follow=True)

            # Buffer strictly clamped to CAP (30)
            assert len(pane.buffer) == 30
            # Wrap rows must match buffer length without duplicate phantom rows
            assert pane.log_view._wrap_row_count() == 30
            assert pane.view_state.pending_lines == 3

    asyncio.run(_run())


def test_log_pane_stream_ended_and_error_drain_leftover_queue(monkeypatch):
    """Stream end and error drain any remaining queued lines before appending their markers."""
    import asyncio

    target = FakeLogTarget(lines=[])
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: target)

    async def _run():
        pane = LogPane(host="h1", cluster_id="cid1", container="c1", scope=LogScope.CONTAINER)
        app = _LogPaneTestApp(pane)
        async with app.run_test(size=(80, 20)):
            await asyncio.sleep(0.05)
            pane.buffer._entries.clear()
            gen = pane._generation

            # Enqueue 3 lines in _incoming_lines
            lines = [LogLine(host="h1", container="c1", text=f"queued {i}") for i in range(3)]
            with pane._incoming_lock:
                for line in lines:
                    pane._incoming_lines.append((line, gen))

            pane._on_stream_ended(gen)

            # Remaining lines must have been delivered to buffer before STREAM_ENDED_MARKER
            assert len(pane.buffer) == 4
            assert [e.text for e in list(pane.buffer._entries)[:3]] == ["queued 0", "queued 1", "queued 2"]
            assert pane.buffer._entries[3] == STREAM_ENDED_MARKER

            # Test for _on_error
            pane.buffer._entries.clear()
            gen2 = pane._generation
            with pane._incoming_lock:
                pane._incoming_lines.append((LogLine(host="h1", container="c1", text="err queue line"), gen2))

            pane._on_error("Connection failure", gen2)
            assert len(pane.buffer) == 2
            assert pane.buffer._entries[0].text == "err queue line"
            assert "Connection failure" in pane.buffer._entries[1].text

    asyncio.run(_run())


def test_log_pane_stale_generation_flush_is_noop(monkeypatch):
    """A flush event from an old generation does not clear or drop lines from the new generation."""
    import asyncio

    target = FakeLogTarget(lines=[])
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: target)

    async def _run():
        pane = LogPane(host="h1", cluster_id="cid1", container="c1", scope=LogScope.CONTAINER)
        app = _LogPaneTestApp(pane)
        async with app.run_test():
            await asyncio.sleep(0.05)
            pane.buffer._entries.clear()
            old_gen = pane._generation
            new_gen = old_gen + 1
            pane._generation = new_gen

            # Enqueue lines for the new generation
            with pane._incoming_lock:
                pane._incoming_lines.append((LogLine(host="h1", container="c1", text="new gen line 1"), new_gen))
                pane._incoming_lines.append((LogLine(host="h1", container="c1", text="new gen line 2"), new_gen))
                pane._flush_scheduled = True

            # Stale flush for old_gen arriving
            await pane._flush_incoming(old_gen)

            # Lines for new_gen must NOT have been cleared
            with pane._incoming_lock:
                assert len(pane._incoming_lines) == 2
                assert pane._flush_scheduled is True

            # Flushing with new_gen delivers the lines
            await pane._flush_incoming(new_gen)
            with pane._incoming_lock:
                assert len(pane._incoming_lines) == 0

            assert len(pane.buffer) == 2
            assert pane.buffer._entries[0].text == "new gen line 1"
            assert pane.buffer._entries[1].text == "new gen line 2"

    asyncio.run(_run())


def test_log_pane_short_log_windowing_up_and_home_stay_scrolled(monkeypatch):
    """Up/home on short logs (< 2 pages) and 300-line logs decouple and stay scrolled."""
    import asyncio

    async def _check(line_count: int, key: str):
        target = FakeLogTarget(
            lines=[LogLine(host="h1", container="c1", stream="stdout", timestamp=None, text=f"line {i}") for i in range(line_count)],
        )
        monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: target)
        pane = LogPane(host="h1", cluster_id="cid1", container="c1", scope=LogScope.CONTAINER)
        app = _LogPaneTestApp(pane)
        async with app.run_test(size=(80, 20)) as pilot:
            # Wait past the stream-paint gate (100 ms) so the press lands on a painted
            # view: the scroll-position guarantee is asserted against painted geometry.
            await asyncio.sleep(0.15)
            assert pane.view_state.state == FollowState.FOLLOWING
            await pilot.press(key)
            await asyncio.sleep(0.3)
            assert pane.view_state.state == FollowState.SCROLLED
            assert pane.log_view.scroll_y < pane.log_view.max_scroll_y

    async def _run():
        await _check(30, "up")
        await _check(30, "home")
        await _check(300, "home")

    asyncio.run(_run())


def test_log_pane_windowing_page_success_and_failure_stay_scrolled(monkeypatch):
    """Page fetch success and failure stay SCROLLED and keep anchor."""
    import asyncio

    class PagingTarget(FakeLogTarget):
        def __init__(self, fail: bool = False, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.fail = fail

        def page(self, source: LogSource, *, skip: int, count: int = 500) -> list[LogLine]:
            if self.fail:
                raise RuntimeError("simulated page failure")
            return [LogLine(host="h1", container="c1", text=f"older line {i}") for i in range(10)]

    async def _test_success():
        target = PagingTarget(
            fail=False,
            lines=[LogLine(host="h1", container="c1", text=f"line {i}") for i in range(30)],
        )
        monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: target)
        pane = LogPane(host="h1", cluster_id="cid1", container="c1", scope=LogScope.CONTAINER)
        app = _LogPaneTestApp(pane)
        async with app.run_test(size=(80, 20)) as pilot:
            await asyncio.sleep(0.05)
            await pilot.press("home")
            await asyncio.sleep(0.3)
            assert pane.view_state.state == FollowState.SCROLLED
            assert not pane._loading_page
            assert any("older line" in e.text for e in pane.buffer._entries if isinstance(e, LogLine))
            assert pane.view_state.state == FollowState.SCROLLED

    async def _test_failure():
        target = PagingTarget(
            fail=True,
            lines=[LogLine(host="h1", container="c1", text=f"line {i}") for i in range(30)],
        )
        monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: target)
        pane = LogPane(host="h1", cluster_id="cid1", container="c1", scope=LogScope.CONTAINER)
        app = _LogPaneTestApp(pane)
        async with app.run_test(size=(80, 20)) as pilot:
            await asyncio.sleep(0.05)
            await pilot.press("home")
            await asyncio.sleep(0.3)
            assert pane.view_state.state == FollowState.SCROLLED
            assert pane._page_failed
            assert any("Page fetch failed" in e.text for e in pane.buffer._entries if isinstance(e, Marker))
            assert pane.view_state.state == FollowState.SCROLLED

    async def _run():
        await _test_success()
        await _test_failure()

    asyncio.run(_run())


def test_log_pane_reload_tail_keeps_every_queued_line_including_duplicates(monkeypatch):
    """Reload tail never inspects line text: every queued live line is kept, duplicates beat drops."""
    import asyncio

    target = FakeLogTarget(lines=[])
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: target)

    async def _run():
        pane = LogPane(host="h1", cluster_id="cid1", container="c1", scope=LogScope.CONTAINER)
        app = _LogPaneTestApp(pane)
        async with app.run_test(size=(80, 20)):
            await asyncio.sleep(0.05)
            pane.buffer._entries.clear()
            gen = pane._generation

            # Real targets stamp follow lines at receipt and repeated lines are common:
            # identical text / no timestamp must not be mistaken for snapshot overlap.
            tail_lines = [LogLine(host="h1", container="c1", text="repeat", timestamp=None) for _ in range(3)]
            queued = [LogLine(host="h1", container="c1", text="repeat", timestamp=None) for _ in range(2)]
            queued.append(LogLine(host="h1", container="c1", text="newer", timestamp=None))
            with pane._incoming_lock:
                pane._flush_scheduled = False
                for line in queued:
                    pane._incoming_lines.append((line, gen))

            pane._on_reload_tail_success(tail_lines, gen)
            # exactly one flush is requested, and the flag reflects it
            assert pane._flush_scheduled is True

            for _ in range(100):
                if len(pane.buffer) == 6:
                    break
                await asyncio.sleep(0.005)

            texts = [e.text for e in pane.buffer._entries if isinstance(e, LogLine)]
            assert texts == ["repeat"] * 5 + ["newer"]
            assert pane._flush_scheduled is False

    asyncio.run(_run())


def test_log_pane_reload_tail_does_not_reschedule_when_flush_in_flight(monkeypatch):
    """A flush already in flight (flag set) is reused; reload does not post a redundant one."""
    import asyncio

    target = FakeLogTarget(lines=[])
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: target)

    async def _run():
        pane = LogPane(host="h1", cluster_id="cid1", container="c1", scope=LogScope.CONTAINER)
        app = _LogPaneTestApp(pane)
        async with app.run_test(size=(80, 20)):
            await asyncio.sleep(0.05)
            gen = pane._generation
            posted: list[int] = []
            monkeypatch.setattr(pane, "_schedule_flush", lambda g: posted.append(g))
            with pane._incoming_lock:
                pane._flush_scheduled = True
                pane._incoming_lines.append((LogLine(host="h1", container="c1", text="q"), gen))
            pane._on_reload_tail_success([LogLine(host="h1", container="c1", text="t")], gen)
            assert posted == []
            assert pane._flush_scheduled is True

    asyncio.run(_run())


def test_log_view_layout_clamp_does_not_trigger_page_fetch(monkeypatch):
    """Only user-initiated upward scrolls fetch older pages; layout/programmatic clamps near the top do not."""
    import asyncio

    target = FakeLogTarget(
        lines=[LogLine(host="h1", container="c1", stream="stdout", timestamp=None, text=f"line {i}") for i in range(30)],
    )
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: target)

    async def _run():
        pane = LogPane(host="h1", cluster_id="cid1", container="c1", scope=LogScope.CONTAINER)
        app = _LogPaneTestApp(pane)
        async with app.run_test(size=(80, 20)) as pilot:
            await asyncio.sleep(0.2)
            fetches: list[int] = []
            orig = pane._trigger_page_fetch
            monkeypatch.setattr(pane, "_trigger_page_fetch", lambda: (fetches.append(1), orig())[1])
            lv = pane.log_view
            assert lv.max_scroll_y > 3

            # Programmatic scroll near the top (what a layout clamp looks like to the watcher)
            lv.scroll_to(y=3, animate=False, immediate=True)
            await pilot.pause()
            lv.scroll_to(y=1, animate=False, immediate=True)
            await pilot.pause()
            # Shrinking the virtual height clamps scroll_y down without any user action
            lv.virtual_size = Size(lv.virtual_size.width, lv.scrollable_content_region.height + 1)
            await pilot.pause()
            await asyncio.sleep(0.1)
            assert fetches == []
            assert not any(e == START_OF_LOG_MARKER or e == LOADING_MARKER for e in pane.buffer._entries)

            # A genuine user scroll up near the top does fetch
            lv.virtual_size = Size(lv.virtual_size.width, 40)
            lv.scroll_to(y=2, animate=False, immediate=True)
            app.simulate_key("up")
            await asyncio.sleep(0.2)
            assert fetches

    asyncio.run(_run())


def test_log_view_compensation_after_immediate_user_scroll_keeps_target(monkeypatch):
    """A user scroll issued right before a prepend compensation is shifted, not cancelled or replayed from stale coordinates."""
    import asyncio

    target = FakeLogTarget(
        lines=[LogLine(host="h1", container="c1", stream="stdout", timestamp=None, text=f"line {i}") for i in range(120)],
    )
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: target)

    async def _run():
        pane = LogPane(host="h1", cluster_id="cid1", container="c1", scope=LogScope.CONTAINER)
        app = _LogPaneTestApp(pane)
        async with app.run_test(size=(80, 20)) as pilot:
            await asyncio.sleep(0.2)
            lv = pane.log_view
            pane.view_state.state = FollowState.SCROLLED
            lv.scroll_to(y=50, animate=False, immediate=True)
            await pilot.pause()

            # up (-1 row, animating) immediately followed by 4 prepended rows
            lv.action_scroll_up()
            assert lv.scroll_target_y == 49
            lv.compensate_scroll(4)
            await pilot.wait_for_scheduled_animations()
            assert lv.scroll_y == 53
            assert pane.view_state.state == FollowState.SCROLLED

            # page up then 3 evicted rows
            lv.action_page_up()
            expected = 53 - lv.scrollable_content_region.height - 3
            lv.compensate_scroll(-3)
            await pilot.wait_for_scheduled_animations()
            assert lv.scroll_y == expected
            assert pane.view_state.state == FollowState.SCROLLED

            # home keeps targeting the very top after a compensating prepend
            lv.action_scroll_home()
            lv.compensate_scroll(5)
            await pilot.wait_for_scheduled_animations()
            assert lv.scroll_y == 0
            assert pane.view_state.state == FollowState.SCROLLED

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Log pane metrics footer
# ---------------------------------------------------------------------------

# One vLLM v1 engine stat line as the container emits it (vllm/v1/metrics/loggers.py).
_VLLM_STAT_LINE = (
    "INFO 10-17 08:12:34 [loggers.py:331] Engine 000: Avg prompt throughput: 123.4 tokens/s, "
    "Avg generation throughput: 45.6 tokens/s, Running: 3 reqs, Waiting: 1 reqs, "
    "GPU KV cache usage: 12.3%, Prefix cache hit rate: 45.6%"
)


def test_log_pane_metrics_footer_renders_tracker_sample_once_target_ready(monkeypatch):
    """The footer shows what the background tracker already collected, as soon as the target resolves."""
    import asyncio

    from sparkrun.cli._monitor_metrics import _SPARK_GLYPHS

    target = FakeLogTarget(lines=[])
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: target)

    async def _run():
        pane = LogPane(host="h1", cluster_id="cid1", container="c1", scope=LogScope.CONTAINER)
        app = _LogPaneTestApp(pane)
        app.metrics_scraper.observe(("h1", "c1"), _VLLM_STAT_LINE)
        app.metrics_scraper.commit()

        async with app.run_test(size=(100, 20)):
            await asyncio.sleep(0.05)
            footer = pane.query_one("#log-metrics", Static)

            # No stream line was delivered: the row comes from the tracker alone.
            assert footer.display is True
            content = str(footer.content)
            assert "tg" in content  # 5-column label field
            assert any(glyph in content for glyph in _SPARK_GLYPHS)
            assert "45.6" in content

            # ... and it sits above the status/settings bar, which stays the
            # pane's bottom row whether or not the metrics row is showing.
            status = pane.query_one("#log-status", Static)
            assert footer.region.y < status.region.y
            assert status.region.bottom == pane.region.bottom

    asyncio.run(_run())


def test_log_pane_footer_shows_the_window_average(monkeypatch):
    """The number is the newest window's mean: several prints in one poll collapse to one point."""
    import asyncio

    target = FakeLogTarget(lines=[])
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: target)

    async def _run():
        pane = LogPane(host="h1", cluster_id="cid1", container="c1", scope=LogScope.CONTAINER)
        app = _LogPaneTestApp(pane)
        # Both prints belong to one tracker window, so its single ring point is their mean.
        app.metrics_scraper.observe(("h1", "c1"), "Avg generation throughput: 40.0 tokens/s")
        app.metrics_scraper.observe(("h1", "c1"), "Avg generation throughput: 50.0 tokens/s")
        app.metrics_scraper.commit()

        async with app.run_test(size=(100, 20)):
            await asyncio.sleep(0.05)
            content = str(pane.query_one("#log-metrics", Static).content)

            assert "45.0" in content
            assert "40.0" not in content
            assert "50.0" not in content

    asyncio.run(_run())


def test_log_pane_blank_rows_frame_the_log_view(monkeypatch):
    """A blank row under the title and one above the sparklines, with no orphan row when they hide."""
    import asyncio

    target = FakeLogTarget(lines=[])
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: target)

    async def _run():
        pane = LogPane(host="h1", cluster_id="cid1", container="c1", scope=LogScope.CONTAINER)
        app = _LogPaneTestApp(pane)
        app.metrics_scraper.observe(("h1", "c1"), _VLLM_STAT_LINE)
        app.metrics_scraper.commit()

        async with app.run_test(size=(100, 20)):
            await asyncio.sleep(0.25)
            title = pane.query_one("#log-title", Static)
            view = pane.log_view
            footer = pane.query_one("#log-metrics", Static)
            status = pane.query_one("#log-status", Static)

            # Wrapping on: no horizontal scrollbar, so both blank rows are spent.
            assert view.show_horizontal_scrollbar is False
            assert view.region.y == title.region.bottom + 1
            assert footer.region.y == view.region.bottom + 1
            assert status.region.y == footer.region.bottom

            # A hidden metrics row takes its own spacer with it; the title's stays.
            footer.display = False
            await asyncio.sleep(0.05)
            assert view.region.y == title.region.bottom + 1
            assert status.region.y == view.region.bottom

    asyncio.run(_run())


def test_log_pane_blank_rows_yield_to_the_horizontal_scrollbar(monkeypatch):
    """Unwrapped wide content trades the spacers for the bar, and a follower keeps the tail."""
    import asyncio

    lines = [LogLine(host="h1", container="c1", text=f"line {i} " + "x" * 180) for i in range(60)]
    target = FakeLogTarget(lines=lines)
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: target)

    async def _run():
        pane = LogPane(host="h1", cluster_id="cid1", container="c1", scope=LogScope.CONTAINER)
        pane.log_view.wrap = False
        app = _LogPaneTestApp(pane)
        app.metrics_scraper.observe(("h1", "c1"), _VLLM_STAT_LINE)
        app.metrics_scraper.commit()

        async with app.run_test(size=(60, 20)):
            # The pane acts on the settled verdict, one settle window after mount.
            await asyncio.sleep(0.3)
            title = pane.query_one("#log-title", Static)
            view = pane.log_view
            footer = pane.query_one("#log-metrics", Static)

            # Unwrapped content wider than the view: the bar is up and takes the
            # sparkline spacer, but the row below the title stays.
            assert view.show_horizontal_scrollbar is True
            assert "spark-gap" not in pane.classes
            assert view.region.y == title.region.bottom + 1
            assert footer.region.y == view.region.bottom

            # Wrapping back on retires the bar, and the viewport shrinking under a
            # following view must not leave it short of the tail: the stream is
            # idle here, so nothing else re-anchors it.
            pane.log_view.wrap = True
            await asyncio.sleep(0.3)
            assert view.show_horizontal_scrollbar is False
            assert "spark-gap" in pane.classes
            assert view.region.y == title.region.bottom + 1
            assert footer.region.y == view.region.bottom + 1
            assert pane.view_state.is_following is True
            assert view.is_at_bottom is True

    asyncio.run(_run())


def test_log_pane_metrics_footer_stream_lines_do_not_feed_metrics(monkeypatch):
    """The pane's own stream never feeds metrics: the app tracker owns them."""
    import asyncio

    target = FakeLogTarget(lines=[])
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: target)

    async def _run():
        pane = LogPane(host="h1", cluster_id="cid1", container="c1", scope=LogScope.CONTAINER)
        app = _LogPaneTestApp(pane)
        async with app.run_test(size=(100, 20)):
            await asyncio.sleep(0.05)
            footer = pane.query_one("#log-metrics", Static)
            line = LogLine(host="h1", container="c1", text=_VLLM_STAT_LINE)

            # Neither prepended history nor follow traffic warms the strip.
            pane.buffer.prepend_page([line])
            assert footer.display is False

            pane._on_lines([line], gen=pane._generation, from_follow=False)
            pane._on_lines([line], gen=pane._generation, from_follow=True)
            await asyncio.sleep(0.05)
            assert footer.display is False
            assert app.metrics_scraper.latest("gen") is None

    asyncio.run(_run())


def test_log_pane_metrics_footer_scope_switch_hides_footer_and_keeps_tracker_data(monkeypatch):
    """A scope switch hides the row until the new target resolves; the tracker's data survives."""
    import asyncio

    from sparkrun.core.log_source import LogSource

    target = FakeLogTarget(lines=[])
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: target)

    async def _run():
        pane = LogPane(host="h1", cluster_id="cid1", container="c1", scope=LogScope.CONTAINER)
        app = _LogPaneTestApp(pane)
        app.metrics_scraper.observe(("h1", "c1"), _VLLM_STAT_LINE)
        app.metrics_scraper.commit()

        async with app.run_test(size=(100, 20)):
            await asyncio.sleep(0.05)
            footer = pane.query_one("#log-metrics", Static)
            assert footer.display is True

            pane._start_follow_worker(LogScope.CLUSTER)
            assert footer.display is False
            # Scope is a pane view concern: the tracked history is not discarded.
            assert app.metrics_scraper.latest("gen") == 45.6
            assert app.metrics_scraper.history("gen") == [45.6]

            cluster_target = FakeLogTarget(
                lines=[],
                sources=[
                    LogSource(host="h1", container="c1", role="head", rank=0),
                    LogSource(host="h2", container="c2", role="worker", rank=1),
                ],
            )
            pane._on_target_ready(cluster_target, pane._generation)
            assert footer.display is True

    asyncio.run(_run())


def test_log_pane_metrics_footer_filters_to_target_sources(monkeypatch):
    """The row aggregates only the sources in the pane's current scope."""
    import asyncio

    from sparkrun.core.log_source import LogSource

    target = FakeLogTarget(lines=[])
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: target)

    async def _run():
        pane = LogPane(host="h1", cluster_id="cid1", container="c1", scope=LogScope.CONTAINER)
        app = _LogPaneTestApp(pane)
        app.metrics_scraper.observe(("h1", "c1"), "Avg generation throughput: 100.0 tokens/s")
        app.metrics_scraper.observe(("h2", "c2"), "Avg generation throughput: 50.0 tokens/s")
        app.metrics_scraper.commit()

        async with app.run_test(size=(100, 20)):
            await asyncio.sleep(0.05)
            footer = pane.query_one("#log-metrics", Static)

            # Container scope: one source, so one rank's value — not the cluster sum.
            assert "100.0" in str(footer.content)
            assert "150.0" not in str(footer.content)

            pane._on_target_ready(
                FakeLogTarget(
                    lines=[],
                    sources=[
                        LogSource(host="h1", container="c1", role="head", rank=0),
                        LogSource(host="h2", container="c2", role="worker", rank=1),
                    ],
                ),
                pane._generation,
            )
            assert "150.0" in str(footer.content)

    asyncio.run(_run())
