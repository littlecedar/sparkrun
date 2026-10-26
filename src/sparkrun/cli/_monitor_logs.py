"""Pure log-view models, scope helpers, and monitor TUI widgets for log streaming.

Implements LogBuffer with per-source paging offsets
and eviction, ViewState stream state machine, and log dump/prefix formatting.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, IntEnum
import asyncio
import logging
import os
import re
import threading
import time
from typing import TYPE_CHECKING, Any, Callable, Iterator, Sequence, cast
import bisect

from rich.segment import Segment
from rich.text import Text
from textual import events, work
from textual.message import Message
from textual.app import ComposeResult
from textual.binding import Binding
from textual.actions import SkipAction
from textual.containers import Horizontal, Vertical
from textual.geometry import Size
from textual.screen import ModalScreen
from textual.scrollbar import ScrollTo, ScrollUp
from textual.scroll_view import ScrollView
from textual.strip import Strip
from textual.widgets import Button, Input, OptionList, Static
from textual.widgets.option_list import Option

from sparkrun.api import LogTarget, SparkrunError, open_log_target
from sparkrun.cli._monitor_colorize import colorize_log_spans, unclaimed_ranges
from sparkrun.cli._monitor_metrics import _spark_palette, render_metrics_footer
from sparkrun.core.log_source import SCOPE_ALL, LogLine, LogSource, source_sort_key
from sparkrun.core.monitoring import HostActivity, MonitorFrame
from sparkrun.orchestration.job_metadata import load_job_metadata

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from rich.console import Console, ConsoleOptions

    from sparkrun.cli._monitor_metrics import RuntimeMetricsScraper


class LogScope(IntEnum):
    CONTAINER = 1
    NODE = 2
    CLUSTER = 3


@dataclass(frozen=True)
class ScopeSummary:
    node_containers: int
    cluster_containers: int
    cluster_hosts: int


def _hosts_match(h1: str, h2: str) -> bool:
    if h1 == h2:
        return True
    return h1.split("@")[-1].lower() == h2.split("@")[-1].lower()


def workload_scope_summary(frame: MonitorFrame, cluster_id: str, host: str) -> ScopeSummary:
    node_containers = 0
    cluster_containers = 0
    cluster_hosts = 0

    if frame and frame.hosts:
        for act in frame.hosts:
            host_count = 0
            if act.workloads:
                for w in act.workloads:
                    if w.cluster_id == cluster_id:
                        host_count += len(w.containers) if w.containers else 1
            if host_count > 0:
                cluster_hosts += 1
                cluster_containers += host_count
                if _hosts_match(act.host, host):
                    node_containers += host_count

    return ScopeSummary(
        node_containers=node_containers,
        cluster_containers=cluster_containers,
        cluster_hosts=cluster_hosts,
    )


def distinct_scopes(summary: ScopeSummary) -> list[LogScope]:
    scopes = [LogScope.CONTAINER]
    if summary.node_containers > 1:
        scopes.append(LogScope.NODE)
    if summary.cluster_containers > summary.node_containers:
        scopes.append(LogScope.CLUSTER)
    return scopes


def canonical_scope(scope: LogScope | int, summary: ScopeSummary) -> LogScope:
    scope = LogScope(scope)
    if scope == LogScope.CONTAINER:
        return LogScope.CONTAINER
    if scope == LogScope.NODE:
        if summary.node_containers <= 1:
            return LogScope.CONTAINER
        return LogScope.NODE
    if scope == LogScope.CLUSTER:
        if summary.cluster_containers <= summary.node_containers:
            return canonical_scope(LogScope.NODE, summary)
        return LogScope.CLUSTER
    return scope


def target_args(scope: LogScope | int, host: str, container: str) -> dict[str, Any]:
    scope = LogScope(scope)
    if scope == LogScope.CONTAINER:
        return {"host": host, "container": container, "scope": SCOPE_ALL}
    if scope == LogScope.NODE:
        return {"host": host, "scope": SCOPE_ALL}
    if scope == LogScope.CLUSTER:
        return {"host": None, "scope": SCOPE_ALL}
    raise ValueError(f"Unknown scope: {scope!r}")


def _sanitize_slug(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(text))
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-._")
    return cleaned or "log"


def default_dump_path(
    label: str,
    scope: LogScope | int,
    host: str,
    container: str,
    now: datetime | None = None,
) -> str:
    scope = LogScope(scope)
    if now is None:
        now = datetime.now()
    elif isinstance(now, (int, float)):
        now = datetime.fromtimestamp(now)
    ts = now.strftime("%Y%m%d-%H%M%S")

    safe_label = _sanitize_slug(label)
    safe_host = _sanitize_slug(host)
    safe_container = _sanitize_slug(container)

    if scope == LogScope.CONTAINER:
        return f"./{safe_label}-{safe_host}-{safe_container}-{ts}.log"
    elif scope == LogScope.NODE:
        return f"./{safe_label}-{safe_host}-{ts}.log"
    elif scope == LogScope.CLUSTER:
        return f"./{safe_label}-all-{ts}.log"
    raise ValueError(f"Unknown scope: {scope!r}")


def format_prefix(line: LogLine, multi_host: bool = False, multi_source: bool = False) -> str:
    if not multi_source and not multi_host:
        return ""
    parts = []
    if multi_host:
        parts.append(line.host)
    parts.append(line.role)
    if line.rank is not None:
        parts.append(str(line.rank))
    return f"[{'/'.join(parts)}]"


# Text.from_ansi() rewrites these characters as well as ESC sequences: bell and
# backspace erase, VT/FF/CR and FS-GS-RS become newlines, and NEL / U+2028 /
# U+2029 are str.splitlines() breaks it rejoins as \n. A line matching this
# pattern must take the full from_ansi path to stay equivalent.
_FROM_ANSI_NEEDED_RE = re.compile(r"[\x07\x08\x0b\x0c\r\x1b\x1c-\x1e\x85\u2028\u2029]")


def entry_plain_text(entry: LogLine | Marker | str) -> str:
    """Return the plain text of *entry* with ANSI escape sequences stripped, cached per entry."""
    if isinstance(entry, LogLine):
        plain = getattr(entry, "_plain_text", None)
        if plain is None:
            text = entry.text
            # The fast path is the norm for sparkrun's log readers; Text.from_ansi
            # costs ~27µs/line there vs ~1µs for the passthrough.
            plain = Text.from_ansi(text).plain if _FROM_ANSI_NEEDED_RE.search(text) else text
            try:
                object.__setattr__(entry, "_plain_text", plain)
            except Exception:
                pass
        return plain
    elif isinstance(entry, Marker):
        return entry.text
    return str(entry)


def render_text(
    line: LogLine | Marker | str,
    *,
    color: bool = True,
    pattern: str | re.Pattern | None = None,
    multi_host: bool = False,
    multi_source: bool = False,
    prefix: str | None = None,
    is_current_match: bool = False,
    runtime: str | None = None,
) -> Text:
    """Render one log entry to ``Text``.

    ``color=True`` keeps native ANSI styling, stacks the colorizer's spans on top
    (``_monitor_colorize``), and dims the unclaimed text so the colored data
    carries the signal; ``color=False`` renders plain text. ``runtime`` names the
    workload's runtime when known, which selects its colorizer class outright.
    """
    if isinstance(line, LogLine):
        raw_text = line.text
        if prefix is None:
            prefix = format_prefix(line, multi_host=multi_host, multi_source=multi_source)
    else:
        raw_text = entry_plain_text(line)
        if prefix is None and (multi_host or multi_source):
            prefix = ""

    if color:
        # from_ansi re-parses every rendered line; skip it for the plain lines
        # sparkrun's log readers mostly emit (see _FROM_ANSI_NEEDED_RE).
        # Text(raw_text).plain == raw_text, which is all the colorizer and dim
        # logic key on.
        body = Text.from_ansi(raw_text) if _FROM_ANSI_NEEDED_RE.search(raw_text) else Text(raw_text)
        for start, end, style in colorize_log_spans(body.plain, runtime=runtime):
            body.stylize(style, start, end)
    else:
        body = Text(entry_plain_text(line) if isinstance(line, LogLine) else Text.from_ansi(raw_text).plain)

    if pattern:
        try:
            regex = re.compile(pattern) if isinstance(pattern, str) else pattern
            hl_style = "bold reverse underline" if is_current_match else "reverse"
            body.highlight_regex(regex, style=hl_style)
        except re.error:
            pass

    if color:
        # Dim the baseline last: every span so far (native ANSI, colorizer, search
        # highlight) is claimed, so dimming only the gaps can never wash one out.
        for start, end in unclaimed_ranges(len(body.plain), ((span.start, span.end) for span in body.spans)):
            body.stylize("dim", start, end)

    if prefix:
        p_str = prefix.strip()
        prefix_text = Text(f"{p_str} ")
        full_text = prefix_text + body
    else:
        full_text = body

    return full_text


@dataclass(frozen=True)
class Marker:
    text: str


LOADING_MARKER = Marker("— loading older lines —")
START_OF_LOG_MARKER = Marker("— start of log —")
STREAM_ENDED_MARKER = Marker("— log stream ended —")
LOG_RESTARTED_MARKER = Marker("— log restarted —")
TERMINAL_MARKERS = (STREAM_ENDED_MARKER, LOG_RESTARTED_MARKER)


@dataclass(frozen=True)
class WorkloadRow:
    key: str
    text: str
    is_header: bool = False
    cluster_id: str = ""
    container: str = ""
    role: str = ""
    status: str = ""
    ranks: int = 1

    @property
    def container_name(self) -> str:
        return self.container

    @property
    def ranks_on_host(self) -> int:
        return self.ranks


def workload_rows(activity: HostActivity | None, cache_dir: str | None = None) -> list[WorkloadRow]:
    from sparkrun.core.application_profile import get_application_profile

    if activity is None or not activity.workloads:
        return [WorkloadRow(key="empty", text="[dim]No %s workloads running[/dim]" % get_application_profile().command, is_header=True)]

    rows: list[WorkloadRow] = []
    for w in activity.workloads:
        meta = load_job_metadata(w.cluster_id, cache_dir=cache_dir) or {}
        recipe = w.recipe_name or meta.get("recipe", "")
        model = meta.get("model", "")
        runtime = w.runtime_name or meta.get("runtime", "")
        tp = meta.get("tensor_parallel", "")

        rank_tag = f" [dim]×{w.ranks_on_host}[/dim]" if w.ranks_on_host > 1 else ""
        meta_parts: list[str] = []
        if recipe:
            meta_parts.append(f"recipe=[bold]{recipe}[/bold]")
        if model:
            meta_parts.append(f"model={model}")
        if runtime:
            meta_parts.append(f"runtime={runtime}")
        if tp:
            meta_parts.append(f"tp={tp}")

        header_lines = [f"[bold]{w.cluster_id}[/bold]{rank_tag}"]
        if meta_parts:
            header_lines.append(f"  [dim]{', '.join(meta_parts)}[/dim]")
        header_text = "\n".join(header_lines)

        rows.append(
            WorkloadRow(
                key=f"header:{w.cluster_id}",
                text=header_text,
                is_header=True,
                cluster_id=w.cluster_id,
                ranks=w.ranks_on_host,
            )
        )

        if w.containers:
            for c in w.containers:
                role_tag = f" [dim]({c.role})[/dim]" if c.role and c.role != "?" else ""
                status_tag = f" [dim]— {c.status}[/dim]" if c.status else ""
                c_text = f"  {c.name}{role_tag}{status_tag}"
                rows.append(
                    WorkloadRow(
                        key=f"{w.cluster_id}:{c.name}",
                        text=c_text,
                        is_header=False,
                        cluster_id=w.cluster_id,
                        container=c.name,
                        role=c.role,
                        status=c.status,
                        ranks=w.ranks_on_host,
                    )
                )

    return rows


class LogBuffer:
    """Bounded buffer of LogLine and Marker items with per-source paging offsets."""

    def __init__(self, entries: Sequence[LogLine | Marker] | None = None) -> None:
        self._entries: deque[LogLine | Marker] = deque(entries or [])
        self._base_offset: int = 0
        self._skips: dict[tuple[str, str], int] = {}
        self._exhausted: dict[tuple[str, str], bool] = {}
        self._search_pattern: re.Pattern | None = None
        self._search_matches: deque[int] = deque()
        self.evicted_live: int = 0
        self.evicted_history: int = 0
        self.pending_markers: list[Marker] = []

        for entry in self._entries:
            if isinstance(entry, LogLine):
                key = (entry.host, entry.container)
                self._skips[key] = self._skips.get(key, 0) + 1

    @property
    def entries(self) -> list[LogLine | Marker]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, index: int) -> LogLine | Marker:
        return self._entries[index]

    def __iter__(self) -> Iterator[LogLine | Marker]:
        return iter(self._entries)

    def get_skip(self, source: LogSource | LogLine | tuple[str, str]) -> int:
        key = self._source_key(source)
        return self._skips.get(key, 0)

    def is_exhausted(self, source: LogSource | LogLine | tuple[str, str]) -> bool:
        key = self._source_key(source)
        return self._exhausted.get(key, False)

    def mark_exhausted(self, source: LogSource | LogLine | tuple[str, str]) -> None:
        key = self._source_key(source)
        self._exhausted[key] = True

    def reload_tail(self, lines: Sequence[LogLine]) -> list[Marker]:
        """Replace the whole buffer with a freshly read tail.

        Resets offsets, exhaustion, eviction counters, and the search index
        (re-running the active search), then re-appends the pending terminal
        markers in chronological order after the fresh lines. Returns them.
        """
        carried = list(self.pending_markers)
        self._entries.clear()
        self._base_offset = 0
        self._skips.clear()
        self._exhausted.clear()
        self._search_matches.clear()
        self.evicted_live = 0
        self.evicted_history = 0
        self.pending_markers = []
        for entry in lines:
            self.append(entry)
        for marker in carried:
            self.append(marker)
        if self._search_pattern is not None:
            self.search(self._search_pattern)
        return carried

    @property
    def stream_ended(self) -> bool:
        return STREAM_ENDED_MARKER in self.pending_markers or any(e == STREAM_ENDED_MARKER for e in self._entries)

    def append(self, entry: LogLine | Marker) -> bool:
        if self.evicted_live > 0:
            if isinstance(entry, LogLine):
                self.evicted_live += 1
                key = self._source_key(entry)
                self._skips[key] = self._skips.get(key, 0) + 1
            elif entry in TERMINAL_MARKERS:
                self.pending_markers.append(entry)
            return False

        if isinstance(entry, LogLine):
            key = self._source_key(entry)
            self._skips[key] = self._skips.get(key, 0) + 1
        abs_idx = self._base_offset + len(self._entries)
        self._entries.append(entry)
        if self._search_pattern is not None:
            if self._matches_pattern(entry, self._search_pattern):
                self._search_matches.append(abs_idx)
        return True

    def append_batch(self, entries: Sequence[LogLine | Marker]) -> list[LogLine | Marker]:
        """Append multiple entries, returning the entries that were successfully appended."""
        appended: list[LogLine | Marker] = []
        for entry in entries:
            if self.append(entry):
                appended.append(entry)
        return appended

    def prepend_page(
        self,
        source_or_lines: Any,
        lines: Sequence[LogLine | Marker] | None = None,
        *,
        exhausted: bool = False,
        separator: str | Marker | None = None,
    ) -> None:
        if lines is None:
            page_lines = list(source_or_lines)
            source = None
        else:
            page_lines = list(lines)
            source = source_or_lines

        if source is not None:
            key = self._source_key(source)
            line_count = sum(1 for item in page_lines if isinstance(item, LogLine))
            self._skips[key] = self._skips.get(key, 0) + line_count
            if exhausted:
                self._exhausted[key] = True
        else:
            for item in page_lines:
                if isinstance(item, LogLine):
                    key = (item.host, item.container)
                    self._skips[key] = self._skips.get(key, 0) + 1

        to_prepend: list[LogLine | Marker] = []
        if separator is not None:
            sep_marker = Marker(separator) if isinstance(separator, str) else separator
            to_prepend.append(sep_marker)
        to_prepend.extend(page_lines)

        self._base_offset -= len(to_prepend)
        self._entries.extendleft(reversed(to_prepend))

        if self._search_pattern is not None:
            new_matches = []
            for idx, entry in enumerate(to_prepend):
                if self._matches_pattern(entry, self._search_pattern):
                    new_matches.append(self._base_offset + idx)
            self._search_matches.extendleft(reversed(new_matches))

    def evict(
        self,
        cap: int,
        viewport: int | tuple[int, int] = 0,
        drop_from: str | None = None,
    ) -> int:
        excess = len(self._entries) - cap
        if excess <= 0:
            return 0

        if drop_from == "top":
            from_top = True
        elif drop_from == "bottom":
            from_top = False
        else:
            if isinstance(viewport, (tuple, list)):
                v_top, v_bottom = viewport[0], viewport[1]
            else:
                v_top = v_bottom = viewport

            dist_to_top = v_top
            dist_to_bottom = (len(self._entries) - 1) - v_bottom
            from_top = dist_to_top >= dist_to_bottom

        if from_top:
            # Drop from top (oldest history)
            for _ in range(excess):
                item = self._entries.popleft()
                if isinstance(item, LogLine):
                    key = (item.host, item.container)
                    if key in self._skips:
                        self._skips[key] = max(0, self._skips[key] - 1)
                    self._exhausted[key] = False

            self._base_offset += excess
            self.evicted_history += excess

            while self._search_matches and self._search_matches[0] < self._base_offset:
                self._search_matches.popleft()
        else:
            # Drop from bottom (newest live lines)
            dropped = []
            for _ in range(excess):
                dropped.append(self._entries.pop())
            live_dropped = 0
            dropped_terminal: list[Marker] = []
            for item in reversed(dropped):
                if isinstance(item, LogLine):
                    live_dropped += 1
                elif item in TERMINAL_MARKERS:
                    dropped_terminal.append(item)
            # Dropped entries precede everything already pending.
            self.pending_markers[:0] = dropped_terminal
            self.evicted_live += live_dropped

            max_abs_idx = self._base_offset + len(self._entries) - 1
            while self._search_matches and self._search_matches[-1] > max_abs_idx:
                self._search_matches.pop()

        return excess

    def search(self, pattern: str | re.Pattern | None) -> list[int]:
        if not pattern:
            self._search_pattern = None
            self._search_matches.clear()
            return []

        # Validate / compile regex first. An invalid regex must NOT wipe the previous search.
        regex = re.compile(pattern) if isinstance(pattern, str) else pattern

        self._search_pattern = regex
        self._search_matches.clear()
        for idx, entry in enumerate(self._entries):
            if self._matches_pattern(entry, regex):
                self._search_matches.append(self._base_offset + idx)
        return [abs_idx - self._base_offset for abs_idx in self._search_matches]

    @property
    def search_matches(self) -> list[int]:
        return [abs_idx - self._base_offset for abs_idx in self._search_matches]

    @property
    def search_pattern(self) -> re.Pattern | None:
        return self._search_pattern

    @staticmethod
    def _source_key(source: Any) -> tuple[str, str]:
        if isinstance(source, tuple) and len(source) == 2:
            return (str(source[0]), str(source[1]))
        host = getattr(source, "host", None)
        container = getattr(source, "container", None)
        if host is not None and container is not None:
            return (str(host), str(container))
        return (str(source), "")

    @staticmethod
    def _matches_pattern(entry: LogLine | Marker, regex: re.Pattern | str) -> bool:
        if not isinstance(entry, LogLine):
            return False
        if isinstance(regex, str):
            try:
                regex = re.compile(regex)
            except re.error:
                return False
        return bool(regex.search(entry_plain_text(entry)))


class FollowState(str, Enum):
    FOLLOWING = "following"
    PAUSED = "paused"
    SCROLLED = "scrolled"


class ViewState:
    """Scroll and follow state machine."""

    def __init__(self, state: FollowState = FollowState.FOLLOWING, pending_lines: int = 0) -> None:
        self.state: FollowState = state
        self.pending_lines: int = pending_lines

    @property
    def is_following(self) -> bool:
        return self.state == FollowState.FOLLOWING

    @property
    def is_paused(self) -> bool:
        return self.state == FollowState.PAUSED

    @property
    def is_scrolled(self) -> bool:
        return self.state == FollowState.SCROLLED

    @property
    def status_label(self) -> str:
        if self.pending_lines > 0:
            return f"{self.state.value} (+{self.pending_lines})"
        return self.state.value

    def on_new_line(self, count: int = 1) -> None:
        if self.state in (FollowState.PAUSED, FollowState.SCROLLED):
            self.pending_lines += count

    def on_user_scroll_up(self) -> None:
        """User explicitly initiated upward scroll."""
        if self.state == FollowState.FOLLOWING:
            self.state = FollowState.SCROLLED

    def on_scroll(self, at_bottom: bool) -> None:
        if at_bottom:
            if self.state == FollowState.SCROLLED:
                self.state = FollowState.FOLLOWING
                self.pending_lines = 0
        else:
            if self.state == FollowState.FOLLOWING:
                self.state = FollowState.SCROLLED

    def on_space(self, at_bottom: bool | None = None) -> bool:
        """Process space bar press.

        Returns True if action jumps to end, False otherwise.
        """
        if self.state == FollowState.SCROLLED:
            self.state = FollowState.FOLLOWING
            self.pending_lines = 0
            return True
        elif self.state == FollowState.FOLLOWING:
            self.state = FollowState.PAUSED
            return False
        elif self.state == FollowState.PAUSED:
            self.state = FollowState.FOLLOWING
            self.pending_lines = 0
            return False
        return False


class WorkloadList(OptionList):
    """Workload container selection list."""

    class OpenLog(Message):
        """Posted when a container is chosen to open logs."""

        def __init__(self, host: str, cluster_id: str, container: str) -> None:
            super().__init__()
            self.host = host
            self.cluster_id = cluster_id
            self.container = container

    class RequestDump(Message):
        """Posted when dump modal is requested."""

        def __init__(self, host: str, cluster_id: str, container: str, scope: LogScope = LogScope.CONTAINER) -> None:
            super().__init__()
            self.host = host
            self.cluster_id = cluster_id
            self.container = container
            self.scope = scope

    class RequestStop(Message):
        """Posted when stopping the highlighted workload is requested."""

        def __init__(self, host: str, cluster_id: str, container: str) -> None:
            super().__init__()
            self.host = host
            self.cluster_id = cluster_id
            self.container = container

    class BackToHosts(Message):
        """Posted when user requests back navigation."""

    BINDINGS = [
        Binding("enter", "select_container", "Logs", priority=True),
        Binding("d", "dump_container", "Dump", priority=True),
        Binding("k", "stop_workload", "Stop", priority=True),
        Binding("escape", "back_to_hosts", "Back", priority=True),
        Binding("q", "back_to_hosts", "Back", show=False, priority=True),
    ]

    def __init__(self, host: str = "", rows: Sequence[WorkloadRow] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.host: str = host
        self._rows: list[WorkloadRow] = []
        if rows:
            self.set_rows(rows, host=host)

    def set_rows(self, rows: Sequence[WorkloadRow], host: str = "") -> None:
        new_rows = list(rows)
        if host:
            self.host = host
        if [r.key for r in self._rows] == [r.key for r in new_rows] and [r.text for r in self._rows] == [r.text for r in new_rows]:
            return

        old_key = None
        old_idx = self.highlighted
        if self._rows and self.highlighted is not None and 0 <= self.highlighted < len(self._rows):
            old_key = self._rows[self.highlighted].key

        self._rows = new_rows
        self.clear_options()

        first_selectable = None
        target_idx = None
        selectable_indices: list[int] = []

        for idx, r in enumerate(self._rows):
            is_disabled = r.is_header or not r.container
            opt = Option(prompt=r.text, id=r.key, disabled=is_disabled)
            self.add_option(opt)
            if not is_disabled:
                selectable_indices.append(idx)
                if first_selectable is None:
                    first_selectable = idx
                if old_key is not None and r.key == old_key:
                    target_idx = idx

        if target_idx is not None:
            self.highlighted = target_idx
        elif selectable_indices:
            ref = old_idx if old_idx is not None else 0
            self.highlighted = min(selectable_indices, key=lambda i: abs(i - ref))
        else:
            self.highlighted = None

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """A mouse click selects the workload the same way Enter does."""
        event.stop()
        self.action_select_container()

    def action_select_container(self) -> None:
        if self.highlighted is None or not (0 <= self.highlighted < len(self._rows)):
            return
        row = self._rows[self.highlighted]
        if row.is_header or not row.container:
            return
        self.post_message(self.OpenLog(self.host, row.cluster_id, row.container))

    def action_dump_container(self) -> None:
        if self.highlighted is None or not (0 <= self.highlighted < len(self._rows)):
            return
        row = self._rows[self.highlighted]
        if row.is_header or not row.container:
            return
        self.post_message(self.RequestDump(self.host, row.cluster_id, row.container, LogScope.CONTAINER))

    def action_stop_workload(self) -> None:
        if self.highlighted is None or not (0 <= self.highlighted < len(self._rows)):
            return
        row = self._rows[self.highlighted]
        if row.is_header or not row.container:
            return
        self.post_message(self.RequestStop(self.host, row.cluster_id, row.container))

    def action_back_to_hosts(self) -> None:
        self.post_message(self.BackToHosts())


def _wrapped_row_count(console: Console, opts: ConsoleOptions | None, plain: str) -> int:
    """Rows *plain* occupies when wrapped at *opts*' width.

    Measured from the plain characters, not the colorized renderable: wrapping
    depends on the text alone, so building spans and Rich styles first is pure
    overhead. The two agree exactly (verified over 4000 real SGLang lines at
    width 78: 11452 rows either way).

    An unmounted view has no console to wrap against and the line counts as one row,
    which is what the map holds until the first real layout.
    """
    if opts is None:
        return 1
    return max(1, len(list(Segment.split_lines(console.render(Text(plain), opts)))))


class LogView(ScrollView):
    """Line API virtual log widget rendering lines from LogBuffer."""

    DEFAULT_CSS = """
    LogView {
        width: 1fr;
        height: 1fr;
        background: $background;
        color: $foreground;
        overflow-y: scroll;
        overflow-x: auto;
    }
    """

    class NearTop(Message):
        """Posted when scroll position comes within one page of top of loaded buffer."""

    # Max repaint rate while streaming (seconds). The gate is a rate limiter,
    # not a deferrer: the first paint after an idle gap fires immediately.
    STREAM_PAINT_INTERVAL: float = 0.1

    # Textual's WriterThread drains a bounded Queue(30) to the terminal; when it is
    # this full the terminal is slower than we emit and a blocking put() would soon
    # stall the whole event loop. Drop paints at that depth and retry with headroom.
    WRITER_QUEUE_SATURATED: int = 24

    # A width change re-measures every buffered line, so a resize drag rebuilds
    # once the width settles instead of once per event.
    WRAP_RESIZE_DEBOUNCE_S: float = 0.15

    def __init__(
        self,
        buffer: LogBuffer,
        *,
        view_state: ViewState | None = None,
        color: bool = True,
        highlight: bool = True,
        wrap: bool = False,
        search_pattern: re.Pattern | str | None = None,
        multi_host: bool = False,
        multi_source: bool = False,
        runtime: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.can_focus = True
        self.buffer = buffer
        self.view_state: ViewState | None = view_state
        self._programmatic_scroll: bool = False
        self._user_scroll_until: float = 0.0
        self.color: bool = color
        self.highlight: bool = highlight
        self._wrap: bool = wrap
        self.search_pattern: re.Pattern | str | None = search_pattern
        self.multi_host: bool = multi_host
        self.multi_source: bool = multi_source
        # The workload's runtime name when known, which selects its colorizer class;
        # None falls back to content sniffing (out-of-tree engines).
        self.runtime: str | None = runtime
        self.current_match_entry: int | None = None
        self._stream_paint_at: float = 0.0
        self._stream_paint_timer: asyncio.TimerHandle | None = None
        self._stream_paint_armed: bool = False

        self._wrap_map: list[tuple[int, int]] = []
        self._wrap_base: int = 0
        self._wrap_origin: int = 0
        # Widget width the current wrap map was measured for; -1 means none yet.
        self._wrap_for_width: int = -1
        self._wrap_resize_timer: asyncio.TimerHandle | None = None
        self._max_width: int = 0
        if wrap:
            self.refresh_buffer()

    @property
    def wrap(self) -> bool:
        return self._wrap

    @wrap.setter
    def wrap(self, val: bool) -> None:
        if self._wrap != val:
            self._wrap = val
            self.refresh_buffer()

    @property
    def is_at_bottom(self) -> bool:
        if self.virtual_size.height <= self.scrollable_content_region.height:
            return True
        return self.scroll_y >= self.max_scroll_y - 0.5

    def watch_show_horizontal_scrollbar(self, shown: bool) -> None:
        # Textual's own verdict on the bar (overflow-x: auto vs. virtual width),
        # relayed to the pane, which trades the sparkline spacer for the bar's row.
        # The pane settles the verdict before acting on it: it flips on
        # intermediate layout passes.
        parent = self.parent
        schedule = getattr(parent, "_schedule_gap_update", None)
        if callable(schedule):
            schedule()

    def _rendered_plain_line(self, entry: LogLine | Marker) -> str:
        plain = entry_plain_text(entry) if isinstance(entry, LogLine) else entry.text
        if isinstance(entry, LogLine):
            prefix = format_prefix(entry, multi_host=self.multi_host, multi_source=self.multi_source)
            if prefix:
                return f"{prefix} {plain}"
        return plain

    def _wrap_row_count(self) -> int:
        return len(self._wrap_map) - self._wrap_origin

    def _wrap_row(self, virtual_y: int) -> tuple[int, int]:
        stored, sub = self._wrap_map[self._wrap_origin + virtual_y]
        return stored - self._wrap_base, sub

    def _compact_wrap_map(self) -> None:
        if self._wrap_origin <= 0:
            return
        if self._wrap_origin >= len(self._wrap_map):
            self._wrap_map.clear()
            self._wrap_origin = 0
            return
        if self._wrap_origin >= 1024:
            del self._wrap_map[: self._wrap_origin]
            self._wrap_origin = 0

    def entry_to_virtual_row(self, entry_idx: int) -> int:
        if not self.wrap:
            return max(0, min(entry_idx, len(self.buffer)))
        n = self._wrap_row_count()
        if entry_idx <= 0 or n <= 0:
            return 0
        threshold = entry_idx + self._wrap_base
        vy = bisect.bisect_left(self._wrap_map, threshold, lo=self._wrap_origin, key=lambda x: x[0])
        return min(vy - self._wrap_origin, n)

    def virtual_row_to_entry(self, virtual_row: int) -> int:
        if not self.wrap:
            return max(0, min(virtual_row, len(self.buffer)))
        n = self._wrap_row_count()
        if n <= 0:
            return 0
        if virtual_row <= 0:
            return self._wrap_map[self._wrap_origin][0] - self._wrap_base
        if virtual_row >= n:
            return self._wrap_map[-1][0] - self._wrap_base
        return self._wrap_map[self._wrap_origin + virtual_row][0] - self._wrap_base

    def scroll_to_end(self, animate: bool = False, immediate: bool = True) -> None:
        if self.view_state is not None and not self.view_state.is_following:
            return
        self._programmatic_scroll = True
        try:
            self.scroll_end(animate=animate, x_axis=False, immediate=immediate)
        finally:
            self._programmatic_scroll = False

    def _arm_stream_paint(self) -> None:
        now = time.monotonic()
        elapsed = now - self._stream_paint_at
        if elapsed >= self.STREAM_PAINT_INTERVAL:
            self._stream_paint()
            return
        if self._stream_paint_armed:
            return
        self._stream_paint_armed = True
        # A loop-level call_later, not Widget.set_timer: textual delivers set_timer
        # callbacks through the widget's own message pump (call_next), which an idle
        # LogView pump may never drain while the pane pump is saturated with flush
        # messages — the trailing paint would never run.
        self._stream_paint_timer = asyncio.get_running_loop().call_later(self.STREAM_PAINT_INTERVAL - elapsed, self._on_stream_paint_timer)

    def _on_stream_paint_timer(self) -> None:
        self._stream_paint_armed = False
        self._stream_paint_timer = None
        if self.is_mounted:
            self._stream_paint()

    def _stream_paint(self, force: bool = False) -> None:
        if not force and self._writer_queue_depth() >= self.WRITER_QUEUE_SATURATED:
            # Backpressure: painting now only deepens the writer queue until its
            # blocking put() stalls the loop for seconds. Drop this paint — the next
            # one renders current state, nothing is lost but staleness — and retry
            # once the queue has headroom.
            if not self._stream_paint_armed:
                self._stream_paint_armed = True
                if self._stream_paint_timer is not None:
                    self._stream_paint_timer.cancel()
                self._stream_paint_timer = asyncio.get_running_loop().call_later(0.05, self._on_stream_paint_timer)
            return
        self._stream_paint_at = time.monotonic()
        self._stream_paint_armed = False
        if self._stream_paint_timer is not None:
            self._stream_paint_timer.cancel()
            self._stream_paint_timer = None
        if not self.is_mounted:
            return
        self.virtual_size = self._virtual_size_now()
        self.refresh()
        self.scroll_to_end()  # no-op unless following
        parent = self.parent
        if parent is not None:
            update_status = getattr(parent, "_update_status", None)
            if callable(update_status):
                update_status()  # status rides the same gate

    def _writer_queue_depth(self) -> int:
        if not self.is_mounted:
            return 0
        writer = getattr(getattr(self.app._driver, "_writer_thread", None), "_queue", None)
        return writer.qsize() if writer is not None else 0

    def _virtual_size_now(self) -> Size:
        content_width = self.scrollable_content_region.width or 80
        if self.wrap:
            return Size(content_width, self._wrap_row_count())
        return Size(max(content_width, self._max_width), len(self.buffer))

    def scroll_to_line(self, line_idx: int, animate: bool = False) -> None:
        vy = self.entry_to_virtual_row(line_idx)
        self.scroll_to(y=vy, animate=animate)

    @property
    def _scroll_animating(self) -> bool:
        return self.is_mounted and self.app.animator.is_being_animated(self, "scroll_y")

    def _effective_target_y(self) -> float:
        """Where the vertical scroll is heading.

        ``scroll_target_y`` goes stale when layout clamps ``scroll_y`` (it is only
        validated on assignment), so it is honoured only while an animation runs.
        """
        if self._scroll_animating:
            return min(self.scroll_target_y, self.max_scroll_y)
        return self.scroll_y

    def compensate_scroll(self, delta_vrows: int) -> None:
        """Keep the viewport anchored when rows are prepended to (or evicted from) the top.

        Uses only public scroll APIs and runs immediately. A user scroll that is
        still animating is restarted from the shifted position toward the shifted
        target, so a key press is neither cancelled nor snapped to stale coordinates.
        """
        if delta_vrows == 0:
            return
        cur = self.scroll_y
        in_flight = self._scroll_animating
        target = self._effective_target_y()
        new_y = max(0.0, cur + delta_vrows)
        self._programmatic_scroll = True
        try:
            self.scroll_to(y=new_y, animate=False, immediate=True)
        finally:
            self._programmatic_scroll = False
        if not in_flight:
            return
        # An upward gesture toward the very top (``home``) keeps targeting the top.
        new_target = 0.0 if target <= 0 else max(0.0, target + delta_vrows)
        new_target = min(new_target, self.max_scroll_y)
        if new_target != new_y:
            distance = abs(new_target - new_y)
            self.scroll_to(
                y=new_target,
                animate=True,
                duration=max(0.1, min(1.0, distance / 50.0)),
                immediate=True,
            )

    def on_entries_evicted(self, top_evicted: int = 0, bot_evicted: int = 0) -> None:
        following = self.view_state is not None and self.view_state.is_following
        if self.wrap:
            if top_evicted > 0:
                threshold = self._wrap_base + top_evicted
                idx = bisect.bisect_left(self._wrap_map, threshold, lo=self._wrap_origin, key=lambda x: x[0])
                evicted_vrows = idx - self._wrap_origin
                self._wrap_origin = idx
                self._wrap_base += top_evicted
                self._compact_wrap_map()
                if not following:
                    self.compensate_scroll(-evicted_vrows)
            if bot_evicted > 0:
                limit = len(self.buffer) + self._wrap_base
                while self._wrap_map and self._wrap_map[-1][0] >= limit:
                    self._wrap_map.pop()
            self._arm_stream_paint()
        else:
            # While following, the pending paint re-pins to the bottom anyway; compensating
            # per batch would only churn scroll state (and repaint frames) for nothing.
            if top_evicted > 0 and not following:
                self.compensate_scroll(-top_evicted)
            self._arm_stream_paint()

    def on_entry_appended(self, entry: LogLine | Marker) -> None:
        self.on_entries_appended([entry])

    def on_entries_appended(self, entries: Sequence[LogLine | Marker]) -> None:
        if not entries:
            return
        total_len = len(self.buffer)
        start_idx = total_len - len(entries)
        content_width = self.scrollable_content_region.width or 80
        if self.wrap:
            console = self.app.console if self.is_mounted else None
            opts = console.options.update_width(content_width) if console else None
            for i, entry in enumerate(entries):
                idx = start_idx + i
                stored = idx + self._wrap_base
                full_plain = self._rendered_plain_line(entry)
                if len(full_plain) <= content_width or not console:
                    self._wrap_map.append((stored, 0))
                else:
                    count = _wrapped_row_count(console, opts, full_plain)
                    for sub in range(count):
                        self._wrap_map.append((stored, sub))
            self._arm_stream_paint()
        else:
            for entry in entries:
                plain_len = len(self._rendered_plain_line(entry))
                if plain_len > self._max_width:
                    self._max_width = plain_len
            self._arm_stream_paint()

    def refresh_buffer(self) -> None:
        self._stream_paint_at = time.monotonic()
        if self._wrap_resize_timer is not None:
            self._wrap_resize_timer.cancel()
            self._wrap_resize_timer = None
        content_width = self.scrollable_content_region.width or 80
        if self.wrap:
            # A full rebuild is O(buffer) on the UI thread — measured 1.8 s at
            # the 50k-line cap with ~170-char engine lines, 0.1 s at 2k. Width changes
            # only (guarded and debounced in on_resize), so this is one pause per resize
            # of a very old pane; chunk the build across ticks if that ever reads as a
            # freeze.
            wrap_map: list[tuple[int, int]] = []
            console = self.app.console if self.is_mounted else None
            opts = console.options.update_width(content_width) if console else None

            for idx, entry in enumerate(self.buffer._entries):
                full_plain = self._rendered_plain_line(entry)
                if len(full_plain) <= content_width or not console:
                    wrap_map.append((idx, 0))
                else:
                    count = _wrapped_row_count(console, opts, full_plain)
                    for sub in range(count):
                        wrap_map.append((idx, sub))
            self._wrap_map = wrap_map
            self._wrap_base = 0
            self._wrap_origin = 0
            self.virtual_size = Size(content_width, len(wrap_map))
            self._wrap_for_width = self.size.width
        else:
            max_w = 0
            for entry in self.buffer._entries:
                plain_len = len(self._rendered_plain_line(entry))
                if plain_len > max_w:
                    max_w = plain_len
            self._max_width = max_w
            w = max(content_width, max_w)
            self.virtual_size = Size(w, len(self.buffer))
        self.refresh()

    def on_resize(self, event: events.Resize) -> None:
        # Only a width change can alter wrapping, and only once it settles: a drag
        # fires many Resize events, and each rebuild re-measures the whole buffer.
        if not self.wrap or event.size.width == self._wrap_for_width:
            return
        if self._wrap_resize_timer is not None:
            self._wrap_resize_timer.cancel()
        # Loop-level, like the stream paint timer: the widget's own pump may be
        # busy while the pane drains flush messages, and this must still fire.
        self._wrap_resize_timer = asyncio.get_running_loop().call_later(self.WRAP_RESIZE_DEBOUNCE_S, self._on_wrap_resize)

    def _on_wrap_resize(self) -> None:
        self._wrap_resize_timer = None
        if self.is_mounted:
            self.refresh_buffer()

    def on_unmount(self) -> None:
        if self._wrap_resize_timer is not None:
            self._wrap_resize_timer.cancel()
            self._wrap_resize_timer = None

    def on_mount(self) -> None:
        self.refresh_buffer()

    USER_SCROLL_WINDOW_S = 2.0

    def _notify_user_scroll_up(self) -> None:
        # A user gesture is a user-driven paint: flush any pending gated paint so the
        # scrollability decision and the scroll itself use fresh geometry.
        if self.virtual_size != self._virtual_size_now():
            self._stream_paint(force=True)
        if self.max_scroll_y <= 0:
            return
        self._user_scroll_until = time.monotonic() + self.USER_SCROLL_WINDOW_S
        if self.view_state is not None:
            self.view_state.on_user_scroll_up()
        parent = self.parent
        if parent is not None:
            update_status = getattr(parent, "_update_status", None)
            if callable(update_status):
                update_status()

    @property
    def _user_scrolling(self) -> bool:
        """True while a user-initiated (key / wheel / scrollbar) upward scroll is in progress."""
        return self.is_vertical_scrollbar_grabbed or time.monotonic() < self._user_scroll_until

    def _should_resume_follow_from_scroll(self, old_val: float, new_val: float) -> bool:
        """True only for user-initiated downward movement that lands at the bottom.

        Animated home/pageup leave the bottom in tiny steps that still satisfy
        ``is_at_bottom`` (0.5-row tolerance). Those frames must not resume follow.
        """
        if self._programmatic_scroll or new_val <= old_val or not self.is_at_bottom:
            return False
        # Upward animations set a target away from the bottom immediately.
        return self.scroll_target_y >= self.max_scroll_y - 0.5

    def watch_scroll_y(self, old_val: float, new_val: float) -> None:
        super().watch_scroll_y(old_val, new_val)
        page_threshold = max(1, self.scrollable_content_region.height)
        # NearTop is only for user-initiated upward movement. Layout clamps (resize,
        # scrollbar toggles, search bar) also lower scroll_y but must not fetch pages.
        if not self._programmatic_scroll and new_val < old_val and new_val <= page_threshold and self._user_scrolling:
            self.post_message(self.NearTop())
        if new_val == self.scroll_target_y:
            self._user_scroll_until = 0.0
        if self.is_vertical_scrollbar_grabbed and new_val < old_val and not self.is_at_bottom:
            self._notify_user_scroll_up()
        if self._should_resume_follow_from_scroll(old_val, new_val):
            if self.view_state is not None and self.view_state.state == FollowState.SCROLLED:
                self.view_state.on_scroll(True)
                parent = self.parent
                if parent is not None:
                    update_status = getattr(parent, "_update_status", None)
                    if callable(update_status):
                        update_status()

    def action_scroll_up(self) -> None:
        if not self.allow_vertical_scroll:
            raise SkipAction()
        self._notify_user_scroll_up()
        page_threshold = max(1, self.scrollable_content_region.height)
        if self.scroll_y <= page_threshold:
            self.post_message(self.NearTop())
        # immediate: a deferred scroll would be computed from pre-compensation
        # coordinates and could run after a page prepend shifted the content.
        self.scroll_to(y=self._effective_target_y() - 1, immediate=True)

    def action_page_up(self) -> None:
        if not self.allow_vertical_scroll:
            raise SkipAction()
        self._notify_user_scroll_up()
        page_threshold = max(1, self.scrollable_content_region.height)
        if self.scroll_y <= page_threshold:
            self.post_message(self.NearTop())
        self.scroll_to(y=self._effective_target_y() - self.scrollable_content_region.height, immediate=True)

    def action_scroll_home(self) -> None:
        if not (self.allow_vertical_scroll or self.allow_horizontal_scroll):
            raise SkipAction()
        self._notify_user_scroll_up()
        self.post_message(self.NearTop())
        self.scroll_home(x_axis=self.scroll_y == 0, immediate=True)

    def action_scroll_end(self) -> None:
        if not (self.allow_vertical_scroll or self.allow_horizontal_scroll):
            raise SkipAction()
        self._user_scroll_until = 0.0
        x_axis = not self.allow_vertical_scroll or self.is_vertical_scroll_end or self.scroll_y >= self.max_scroll_y
        if self.view_state is not None and self.view_state.state == FollowState.SCROLLED:
            self.view_state.on_scroll(True)
            parent = self.parent
            if parent is not None:
                update_status = getattr(parent, "_update_status", None)
                if callable(update_status):
                    update_status()
        self._programmatic_scroll = True
        try:
            self.scroll_end(animate=False, x_axis=x_axis, immediate=True)
        finally:
            self._programmatic_scroll = False

    def _on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        if self.allow_vertical_scroll and not (event.ctrl or event.shift):
            self._notify_user_scroll_up()
        super()._on_mouse_scroll_up(event)

    def _on_scroll_up(self, event: ScrollUp) -> None:
        self._notify_user_scroll_up()
        super()._on_scroll_up(event)

    def _on_scroll_to(self, message: ScrollTo) -> None:
        if message.y is not None and message.y < self.scroll_y:
            self._notify_user_scroll_up()
        super()._on_scroll_to(message)

    def render_line(self, y: int) -> Strip:
        scroll_x, scroll_y = self.scroll_offset
        virtual_y = int(scroll_y + y)
        width = self.scrollable_content_region.width or 80

        if not self.wrap:
            if virtual_y < 0 or virtual_y >= len(self.buffer):
                return Strip.blank(width, self.rich_style)
            entry = self.buffer._entries[virtual_y]
            is_curr = self.current_match_entry is not None and self.current_match_entry == virtual_y
            text = render_text(
                entry,
                color=self.color,
                pattern=(self.search_pattern if self.highlight else None),
                multi_host=self.multi_host,
                multi_source=self.multi_source,
                runtime=self.runtime,
                is_current_match=is_curr,
            )
            console = self.app.console
            opts = console.options.update(overflow="ignore", no_wrap=True).update_width(max(width, self._max_width))
            segs = list(console.render(text, opts))
            split = list(Segment.split_lines(segs))
            strip = Strip.from_lines(split)[0] if split else Strip.blank(width, self.rich_style)
            return strip.crop_extend(scroll_x, scroll_x + width, self.rich_style).apply_style(self.rich_style)
        else:
            if virtual_y < 0 or virtual_y >= self._wrap_row_count():
                return Strip.blank(width, self.rich_style)
            entry_idx, sub_idx = self._wrap_row(virtual_y)
            if entry_idx < 0 or entry_idx >= len(self.buffer):
                return Strip.blank(width, self.rich_style)
            entry = self.buffer._entries[entry_idx]
            is_curr = self.current_match_entry is not None and self.current_match_entry == entry_idx
            text = render_text(
                entry,
                color=self.color,
                pattern=(self.search_pattern if self.highlight else None),
                multi_host=self.multi_host,
                multi_source=self.multi_source,
                runtime=self.runtime,
                is_current_match=is_curr,
            )
            console = self.app.console
            opts = console.options.update_width(width)
            segs = list(console.render(text, opts))
            split = list(Segment.split_lines(segs))
            if sub_idx < len(split):
                strip = Strip.from_lines([split[sub_idx]])[0]
            else:
                strip = Strip.blank(width, self.rich_style)
            return strip.crop_extend(0, width, self.rich_style).apply_style(self.rich_style)


def _trim_page_drift(
    lines: list[LogLine],
    *,
    count_before_read: int | None = None,
    snap_count: int | None = None,
    loaded_skip: int | None = None,
    snap_skip: int | None = None,
) -> list[LogLine]:
    """Drop page-tail lines accounted for by live growth up to remote read time.

    Overlap is proven only by delivered counts captured at or before remote read
    time compared to snapshot counts. Line text is never inspected: coincidental
    repeats at the seam are real history. If nothing arrived before the read,
    the page is kept intact — duplicates beat dropped lines.
    """
    if count_before_read is not None and snap_count is not None:
        growth = max(0, int(count_before_read) - int(snap_count))
    elif loaded_skip is not None and snap_skip is not None:
        growth = max(0, int(loaded_skip) - int(snap_skip))
    else:
        growth = 0
    if not growth or not lines:
        return lines
    return lines[: -min(growth, len(lines))]


class LogPane(Vertical):
    """Log viewer pane filling the node body below node stats."""

    CAP: int = 50_000

    DEFAULT_CSS = """
    LogPane {
        width: 1fr;
        height: 1fr;
        layout: vertical;
        background: $background;
    }
    LogPane > #log-title {
        height: 1;
        width: 1fr;
        background: $surface;
        color: $text;
        text-style: bold;
        padding: 0 1;
    }
    LogPane > LogView {
        height: 1fr;
        width: 1fr;
        /* A blank row always separates the view from the title above it. */
        margin-top: 1;
    }
    LogPane > #log-search {
        height: 1;
        width: 1fr;
        display: none;
        border: none;
        padding: 0 1;
        background: $surface;
    }
    LogPane > #log-status {
        height: 1;
        width: 1fr;
        background: $surface;
        color: $text-muted;
        padding: 0 1;
    }
    LogPane > #log-metrics {
        height: 1;
        width: 1fr;
        display: none;
        background: $surface;
        color: $text-muted;
        padding: 0 1;
    }
    /* The blank row above the sparkline row is dropped while the horizontal
       scrollbar is up, and only then (toggled by LogPane._apply_gap_update). */
    LogPane.spark-gap > #log-metrics {
        margin-top: 1;
    }
    """

    class Closed(Message):
        """Posted when the log pane is closed."""

    class RequestDump(Message):
        """Posted when dump modal is requested."""

        def __init__(self, host: str, cluster_id: str, container: str, scope: LogScope) -> None:
            super().__init__()
            self.host = host
            self.cluster_id = cluster_id
            self.container = container
            self.scope = scope

    class _FlushIncoming(Message, bubble=False):
        """Internal message to drain incoming lines batch on the main thread."""

        def __init__(self, gen: int) -> None:
            super().__init__()
            self.gen = gen

    BINDINGS = [
        Binding("space", "toggle_pause_or_end", "Pause/Follow", priority=True),
        Binding("1", "scope_container", "Container", priority=True),
        Binding("2", "scope_node", "Node", priority=True),
        Binding("3", "scope_cluster", "Cluster", priority=True),
        Binding("l", "toggle_wrap", "Wrap", priority=True),
        Binding("c", "toggle_color", "Color", priority=True),
        Binding("h", "toggle_highlight", "Highlight", priority=True),
        Binding("slash", "open_search", "Search", key_display="/"),
        Binding("n", "search_next", "Next match", priority=True),
        Binding("b", "search_prev", "Prev match", priority=True),
        Binding("d", "dump_logs", "Dump", priority=True),
        Binding("escape", "close_pane", "Back", priority=True),
        Binding("q", "close_pane", "Back", show=False, priority=True),
    ]

    def __init__(
        self,
        host: str,
        cluster_id: str,
        container: str,
        scope: LogScope | int = LogScope.CONTAINER,
        *,
        frame: MonitorFrame | None = None,
        summary: ScopeSummary | None = None,
        cluster: str | None = None,
        cache_dir: str | None = None,
        sctx: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.host: str = host
        self.cluster_id: str = cluster_id
        self.container: str = container
        self.scope: LogScope = LogScope(scope)
        self.cluster: str | None = cluster
        self.cache_dir: str | None = cache_dir
        self.sctx: Any = sctx

        self.CAP: int = self.CAP
        self.buffer: LogBuffer = LogBuffer()
        self.view_state: ViewState = ViewState()
        self.scope_summary: ScopeSummary | None = summary or (workload_scope_summary(frame, cluster_id, host) if frame else None)

        self._cancel: threading.Event = threading.Event()
        self._generation: int = 0
        self.target: LogTarget | None = None
        self._loading_page: bool = False
        self._page_failed: bool = False
        self._current_match_idx: int | None = None
        self._current_match_abs: int | None = None
        self._delivered_lock: threading.Lock = threading.Lock()
        self._delivered_counts: dict[tuple[str, str], int] = {}
        self.MAX_INCOMING_LINES: int = 500
        self._incoming_lock: threading.Lock = threading.Lock()
        self._incoming_cv: threading.Condition = threading.Condition(self._incoming_lock)
        self._incoming_lines: deque[tuple[LogLine, int]] = deque()
        self._flush_scheduled: bool = False
        self._gap_update_timer: asyncio.TimerHandle | None = None
        # The sparkline spacer starts on: no scrollbar can be up before the first
        # layout, and the settled verdict corrects this if one turns out to be.
        self.add_class("spark-gap")
        # Wrapping is the pane's default view: engine logs put the metric-bearing
        # fields at the end of long lines, so an unwrapped tail hides them off-screen.
        self.log_view: LogView = LogView(self.buffer, view_state=self.view_state, wrap=True)

    def _source_key(self, source: LogSource | LogLine | tuple[str, str]) -> tuple[str, str]:
        if isinstance(source, tuple):
            return source
        return (source.host, source.container)

    def _get_delivered_count(self, source: LogSource | LogLine | tuple[str, str]) -> int:
        key = self._source_key(source)
        with self._delivered_lock:
            return self._delivered_counts.get(key, 0)

    def _inc_delivered_count(self, source: LogSource | LogLine | tuple[str, str]) -> None:
        key = self._source_key(source)
        with self._delivered_lock:
            self._delivered_counts[key] = self._delivered_counts.get(key, 0) + 1

    def compose(self) -> ComposeResult:
        yield Static(id="log-title")
        yield self.log_view
        yield Input(id="log-search", placeholder="Regex search (enter to submit, esc to cancel)")
        # The metrics row sits directly above the status/settings bar: the bar is
        # the pane's fixed bottom edge (state, scope, wrap/color/highlight, match
        # count), so a row that comes and goes belongs above it, not under it.
        yield Static(id="log-metrics", markup=True)
        yield Static(id="log-status")

    def on_mount(self) -> None:
        self._update_title()
        self._update_status()
        self.log_view.focus()
        self._start_follow_worker(self.scope)

    def on_resize(self, event: events.Resize) -> None:
        # The pane's width is 0 until the first layout, and the footer has to
        # re-fit after a resize; both land here.
        self._refresh_metrics_footer()

    def _schedule_gap_update(self) -> None:
        """Re-read the bar verdict once the layout settles.

        The verdict flips on intermediate layout passes — the first pass compares
        a view width the vertical scrollbar has not been subtracted from yet and
        reports a bar the next pass withdraws. Reacting to each pass would resize
        the log view twice for nothing, so this waits out the settle like the
        wrap-map rebuild does.
        """
        if self._gap_update_timer is not None:
            self._gap_update_timer.cancel()
        self._gap_update_timer = asyncio.get_running_loop().call_later(LogView.WRAP_RESIZE_DEBOUNCE_S, self._apply_gap_update)

    def _apply_gap_update(self) -> None:
        """Spend a blank row above the sparkline row.

        Dropped while the horizontal scrollbar is up: with wrapping off the view
        is already spending its width on the bar, and that row is the space
        someone watching long lines wants back. The row below the title is not
        part of this trade — it is always there (see the LogView rule). Textual's
        own verdict on the bar drives this
        (``LogView.watch_show_horizontal_scrollbar``), so the decision never
        drifts from what is actually displayed.
        """
        self._gap_update_timer = None
        if not self.is_mounted:
            return
        had_gap = "spark-gap" in self.classes
        self.set_class(not self.log_view.show_horizontal_scrollbar, "spark-gap")
        if ("spark-gap" in self.classes) != had_gap:
            # The viewport just changed height under the view: a follower must not
            # be left short of the tail with the stream idle, and the re-anchor has
            # to land after the layout this class change schedules.
            self.call_after_refresh(self.log_view.scroll_to_end)

    def _update_title(self) -> None:
        meta = load_job_metadata(self.cluster_id, cache_dir=self.cache_dir) or {}
        recipe = meta.get("recipe", "")
        label = recipe or self.cluster_id
        short_id = self.cluster_id[:8] if len(self.cluster_id) > 8 else self.cluster_id
        title = f"[bold]{label}[/bold] [dim]({short_id})[/dim] — [bold]{self.container}[/bold]"
        title_widget = self.query_one("#log-title", Static)
        title_widget.update(title)

    def _update_status(self) -> None:
        state_str = self.view_state.status_label
        summary = self.scope_summary
        if self.scope == LogScope.CONTAINER:
            scope_str = f"1 container {self.container}"
        elif self.scope == LogScope.NODE:
            k = summary.node_containers if summary else 1
            scope_str = f"2 node {self.host} ({k} containers)"
        elif self.scope == LogScope.CLUSTER:
            n = summary.cluster_containers if summary else 1
            m = summary.cluster_hosts if summary else 1
            scope_str = f"3 cluster ({n} containers on {m} hosts)"
        else:
            scope_str = f"scope {self.scope}"

        color_str = f"color {'on' if self.log_view.color else 'off'}"
        hl_str = f"highlight {'on' if self.log_view.highlight else 'off'}"
        wrap_str = f"wrap {'on' if self.log_view.wrap else 'off'}"

        parts = [state_str, scope_str, wrap_str, color_str, hl_str]
        if self.buffer.search_pattern is not None:
            count = len(self.buffer.search_matches)
            if count > 0 and self._current_match_idx is not None and 0 <= self._current_match_idx < count:
                parts.append(f"{count} matches in loaded lines (match {self._current_match_idx + 1}/{count})")
            else:
                parts.append(f"{count} matches in loaded lines")

        status_text = " | ".join(parts)
        status_widget = self.query_one("#log-status", Static)
        status_widget.update(status_text)

    def _refresh_metrics_footer(self) -> None:
        """Render the app tracker's latest sample for this pane's scope; hide without one."""
        widget = self.query_one("#log-metrics", Static)
        getter = cast("Callable[[str], RuntimeMetricsScraper] | None", getattr(self.app, "metrics_scraper_for", None))
        scraper = getter(self.cluster_id) if getter is not None else None
        if scraper is None:
            widget.display = False
            return
        # Sparkline colors follow the active theme; a theme that lacks the
        # variables yields an empty palette (sparks uncolored), not a crash.
        palette: tuple[str, ...] = ()
        try:
            palette = _spark_palette(self.app.get_css_variables(), dark=self.app.current_theme.dark)
        except Exception:  # noqa: BLE001 - cosmetic only; never break the stream path
            palette = ()
        sources = None if self.target is None else frozenset((s.host, s.container) for s in self.target.sources)
        text = render_metrics_footer(scraper, palette, width=max(0, self.size.width - 2), sources=sources)
        if text:
            widget.update(text)
            widget.display = True
        else:
            widget.display = False

    def _is_match_visible(self, match_entry: int) -> bool:
        scroll_y = int(self.log_view.scroll_offset[1])
        height = max(1, self.log_view.scrollable_content_region.height)
        top_entry = self.log_view.virtual_row_to_entry(scroll_y)
        bot_entry = self.log_view.virtual_row_to_entry(scroll_y + height - 1)
        return top_entry <= match_entry <= bot_entry

    def _set_current_match(self, idx: int | None) -> None:
        matches = self.buffer.search_matches
        if idx is None or not matches or not (0 <= idx < len(matches)):
            self._current_match_idx = None
            self._current_match_abs = None
            self.log_view.current_match_entry = None
            return
        self._current_match_idx = idx
        self._current_match_abs = self.buffer._base_offset + matches[idx]
        self.log_view.current_match_entry = matches[idx]

    def _rederive_match_cursor(self) -> None:
        abs_idx = self._current_match_abs
        if abs_idx is None or self.buffer.search_pattern is None:
            self._current_match_idx = None
            self.log_view.current_match_entry = None
            return
        rel = abs_idx - self.buffer._base_offset
        if rel < 0 or rel >= len(self.buffer):
            self._set_current_match(None)
            return
        matches = self.buffer.search_matches
        try:
            idx = matches.index(rel)
        except ValueError:
            self._set_current_match(None)
            return
        self._current_match_idx = idx
        self.log_view.current_match_entry = rel

    def update_frame(self, frame: MonitorFrame | None) -> None:
        if frame is not None:
            self.scope_summary = workload_scope_summary(frame, self.cluster_id, self.host)
            self._update_status()

    def _start_follow_worker(self, target_scope: LogScope) -> None:
        self._cancel = threading.Event()
        with self._incoming_lock:
            self._generation += 1
            self._incoming_lines.clear()
            self._flush_scheduled = False
            self._incoming_cv.notify_all()
        with self._delivered_lock:
            self._delivered_counts.clear()
        self.query_one("#log-metrics", Static).display = False
        self._follow_worker(target_scope, self._cancel, self._generation)

    @work(thread=True, exclusive=True, group="follow")
    def _follow_worker(self, target_scope: LogScope, cancel: threading.Event, gen: int) -> None:
        try:
            args = target_args(target_scope, self.host, self.container)
            target = open_log_target(
                self.cluster_id,
                cluster=self.cluster,
                cache_dir=self.cache_dir,
                sctx=self.sctx,
                **args,
            )
            if cancel.is_set():
                return
            self.app.call_from_thread(self._on_target_ready, target, gen)

            for line in target.follow(tail=200, cancel=cancel):
                if cancel.is_set() or gen != self._generation:
                    break
                need_schedule = False
                with self._incoming_lock:
                    while len(self._incoming_lines) >= self.MAX_INCOMING_LINES:
                        if cancel.is_set() or gen != self._generation:
                            break
                        self._incoming_cv.wait(timeout=0.02)
                    if cancel.is_set() or gen != self._generation:
                        break
                    self._incoming_lines.append((line, gen))
                    if not self._flush_scheduled:
                        self._flush_scheduled = True
                        need_schedule = True

                if need_schedule:
                    self._schedule_flush(gen)

            if not cancel.is_set() and gen == self._generation:
                self.app.call_from_thread(self._on_stream_ended, gen)
        except SparkrunError as e:
            if not cancel.is_set() and gen == self._generation:
                self.app.call_from_thread(self._on_error, str(e), gen)
        except Exception as e:
            if not cancel.is_set() and gen == self._generation:
                logger.exception("Unexpected error in follow worker")
                self.app.call_from_thread(self._on_error, str(e), gen)

    def _schedule_flush(self, gen: int) -> None:
        if not self.is_mounted or gen != self._generation or self._cancel.is_set():
            return
        self.post_message(self._FlushIncoming(gen))

    async def on_log_pane__flush_incoming(self, event: _FlushIncoming) -> None:
        await self._flush_incoming(event.gen)

    async def _flush_incoming(self, gen: int) -> None:
        if not self.is_mounted or self._cancel.is_set() or gen != self._generation:
            return

        batch_count = 0
        start_time = time.monotonic()
        while self.is_mounted and not self._cancel.is_set() and gen == self._generation:
            with self._incoming_lock:
                batch: list[LogLine] = []
                while self._incoming_lines and len(batch) < 50:
                    item_line, item_gen = self._incoming_lines.popleft()
                    if item_gen == gen:
                        batch.append(item_line)
                has_more = bool(self._incoming_lines)
                if not has_more:
                    self._flush_scheduled = False
                self._incoming_cv.notify_all()

            if batch:
                self._on_lines(batch, gen, from_follow=True)

            if not has_more:
                break

            batch_count += 1
            if batch_count >= 10 or (time.monotonic() - start_time) >= 0.016:
                with self._incoming_lock:
                    if self._incoming_lines:
                        self._flush_scheduled = True
                        self.post_message(self._FlushIncoming(gen))
                    else:
                        self._flush_scheduled = False
                break

            await asyncio.sleep(0)

    def _on_target_ready(self, target: LogTarget, gen: int) -> None:
        if not self.is_mounted or gen != self._generation or self._cancel.is_set():
            return
        self.target = target
        multi_host = len({s.host for s in target.sources}) > 1
        multi_source = len(target.sources) > 1
        self.log_view.multi_host = multi_host
        self.log_view.multi_source = multi_source
        self.log_view.runtime = target.runtime or None
        # History the background tracker already collected renders at once,
        # filtered to this scope's sources.
        self._refresh_metrics_footer()

    def _on_line(self, line: LogLine, gen: int, from_follow: bool = False) -> None:
        self._on_lines([line], gen, from_follow=from_follow)

    def _on_lines(self, lines: Sequence[LogLine], gen: int, from_follow: bool = False) -> None:
        if not self.is_mounted or gen != self._generation or self._cancel.is_set() or not lines:
            return
        for line in lines:
            self._inc_delivered_count(line)
        top_entry = self.log_view.virtual_row_to_entry(int(self.log_view.scroll_offset[1]))
        bottom_entry = self.log_view.virtual_row_to_entry(
            int(self.log_view.scroll_offset[1]) + self.log_view.scrollable_content_region.height
        )
        n_before = len(self.buffer)
        hist_before = self.buffer.evicted_history
        live_before = self.buffer.evicted_live
        appended_entries = self.buffer.append_batch(lines)
        if not appended_entries:
            self.view_state.on_new_line(len(lines))
            if from_follow:
                self.log_view._arm_stream_paint()
            else:
                self._update_status()
            return

        if self.view_state.is_following:
            self.buffer.evict(cap=self.CAP, drop_from="top")
        else:
            self.buffer.evict(cap=self.CAP, viewport=(top_entry, bottom_entry))
        top_evicted = self.buffer.evicted_history - hist_before
        bot_evicted = self.buffer.evicted_live - live_before

        if top_evicted > 0 or bot_evicted > 0:
            self.log_view.on_entries_evicted(top_evicted=top_evicted, bot_evicted=bot_evicted)
            self._rederive_match_cursor()

        self.view_state.on_new_line(len(lines))
        new_top_evicted = max(0, top_evicted - n_before)
        start = min(len(appended_entries), new_top_evicted)
        end = max(start, len(appended_entries) - bot_evicted)
        surviving = appended_entries[start:end]
        if surviving:
            self.log_view.on_entries_appended(surviving)
        if not from_follow and self.view_state.is_following:
            self.log_view.scroll_to_end()
        if not from_follow:
            self._update_status()

    def _on_stream_ended(self, gen: int) -> None:
        if not self.is_mounted or gen != self._generation or self._cancel.is_set():
            return
        with self._incoming_lock:
            remaining: list[LogLine] = []
            while self._incoming_lines:
                item_line, item_gen = self._incoming_lines.popleft()
                if item_gen == gen:
                    remaining.append(item_line)
            self._flush_scheduled = False
            self._incoming_cv.notify_all()
        if remaining:
            self._on_lines(remaining, gen, from_follow=True)
        if not self.buffer.stream_ended:
            self.buffer.append(STREAM_ENDED_MARKER)
            self.log_view.on_entry_appended(STREAM_ENDED_MARKER)
            if self.view_state.is_following:
                self.log_view.scroll_to_end()
            self._update_status()

    def _on_error(self, err_msg: str, gen: int) -> None:
        if not self.is_mounted or gen != self._generation or self._cancel.is_set():
            return
        with self._incoming_lock:
            remaining: list[LogLine] = []
            while self._incoming_lines:
                item_line, item_gen = self._incoming_lines.popleft()
                if item_gen == gen:
                    remaining.append(item_line)
            self._flush_scheduled = False
            self._incoming_cv.notify_all()
        if remaining:
            self._on_lines(remaining, gen, from_follow=True)
        err_marker = Marker(f"[red]{err_msg}[/red]")
        self.buffer.append(err_marker)
        self.log_view.on_entry_appended(err_marker)
        if self.view_state.is_following:
            self.log_view.scroll_to_end()
        self._update_status()

    def on_log_view_near_top(self, event: LogView.NearTop) -> None:
        self._trigger_page_fetch()

    def _trigger_page_fetch(self) -> None:
        if self._loading_page or self.target is None:
            return
        sorted_sources = sorted(self.target.sources, key=source_sort_key)
        rev_sources = list(reversed(sorted_sources))
        source_skips: list[tuple[LogSource, int, int]] = []
        for s in rev_sources:
            if not self.buffer.is_exhausted(s):
                source_skips.append((s, self.buffer.get_skip(s), self._get_delivered_count(s)))

        if not source_skips:
            if not any(e == START_OF_LOG_MARKER for e in self.buffer._entries):
                self.buffer.prepend_page([START_OF_LOG_MARKER])
                self.log_view.refresh_buffer()
                self._rederive_match_cursor()
                marker_rows = self.log_view.entry_to_virtual_row(1)
                self.log_view.compensate_scroll(marker_rows)
            return

        self._loading_page = True
        self._page_failed = False
        self.buffer.prepend_page([LOADING_MARKER])
        self.log_view.refresh_buffer()
        self._rederive_match_cursor()
        marker_rows = self.log_view.entry_to_virtual_row(1)
        self.log_view.compensate_scroll(marker_rows)

        gen = self._generation
        self._page_worker(self.target, source_skips, self._cancel, gen)

    @work(thread=True, exclusive=False, group="page")
    def _page_worker(
        self,
        target: LogTarget,
        source_skips: list[tuple[LogSource, int, int]],
        cancel: threading.Event,
        gen: int,
    ) -> None:
        fetched_pages: list[tuple[LogSource, list[LogLine], bool]] = []
        try:
            for source, skip, snap_count in source_skips:
                if cancel.is_set():
                    return
                count_before_read = self._get_delivered_count(source)
                lines = target.page(source, skip=skip, count=500)
                exhausted = len(lines) < 500
                trimmed_lines = _trim_page_drift(
                    lines,
                    count_before_read=count_before_read,
                    snap_count=snap_count,
                )
                fetched_pages.append((source, trimmed_lines, exhausted))

            if not cancel.is_set():
                self.app.call_from_thread(self._on_page_success, fetched_pages, target, gen)
        except Exception as e:
            if not cancel.is_set():
                logger.warning("Error fetching older log page: %s", e)
                self.app.call_from_thread(self._on_page_failure, str(e), gen)

    def _on_page_success(
        self,
        fetched_pages: list[tuple[LogSource, list[LogLine], bool]],
        target: LogTarget,
        gen: int,
    ) -> None:
        if not self.is_mounted or gen != self._generation or self._cancel.is_set():
            return
        removed_loading = False
        marker_rows = 0
        if self.buffer._entries and self.buffer._entries[0] == LOADING_MARKER:
            marker_rows = self.log_view.entry_to_virtual_row(1)
            self.buffer._entries.popleft()
            self.buffer._base_offset += 1
            removed_loading = True

        old_entries_count = len(self.buffer._entries)

        multi_source = len(target.sources) > 1
        multi_host = len({s.host for s in target.sources}) > 1

        for source, lines, exhausted in fetched_pages:
            sep = None
            if multi_source and lines:
                role = source.role or "?"
                rank_str = f"/{source.rank}" if source.rank is not None else ""
                if multi_host:
                    sep = f"── {source.host}/{role}{rank_str} ──"
                else:
                    sep = f"── {role}{rank_str} ──"
            self.buffer.prepend_page(source, lines, exhausted=exhausted, separator=sep)

        all_exhausted = all(self.buffer.is_exhausted(s) for s in target.sources)
        if all_exhausted and not any(e == START_OF_LOG_MARKER for e in self.buffer._entries):
            self.buffer.prepend_page([START_OF_LOG_MARKER])

        new_entries_count = len(self.buffer._entries)
        prepended_entries = new_entries_count - old_entries_count

        # Enforce 50k cap on back-scroll paging:
        # Excess lines are dropped from bottom (newest live lines), keeping anchor intact.
        self.buffer.evict(cap=self.CAP, drop_from="bottom")

        self.log_view.refresh_buffer()
        prepended_vrows = self.log_view.entry_to_virtual_row(prepended_entries)
        net_delta = prepended_vrows - (marker_rows if removed_loading else 0)
        self.log_view.compensate_scroll(net_delta)

        self._rederive_match_cursor()

        self._loading_page = False
        self._update_status()

    def _on_page_failure(self, err_msg: str, gen: int) -> None:
        if not self.is_mounted or gen != self._generation or self._cancel.is_set():
            return
        removed_loading = False
        marker_rows = 0
        if self.buffer._entries and self.buffer._entries[0] == LOADING_MARKER:
            marker_rows = self.log_view.entry_to_virtual_row(1)
            self.buffer._entries.popleft()
            self.buffer._base_offset += 1
            removed_loading = True

        err_marker = Marker(f"[red]Page fetch failed: {err_msg}[/red]")
        self.buffer.prepend_page([err_marker])
        self.log_view.refresh_buffer()
        err_vrows = self.log_view.entry_to_virtual_row(1)
        net_delta = err_vrows - (marker_rows if removed_loading else 0)
        self.log_view.compensate_scroll(net_delta)

        self._loading_page = False
        self._page_failed = True
        self._rederive_match_cursor()
        self._update_status()

    def _reload_tail(self) -> None:
        if self.target is None:
            return
        gen = self._generation
        self._reload_tail_worker(self.target, self._cancel, gen)

    @work(thread=True, exclusive=False)
    def _reload_tail_worker(self, target: LogTarget, cancel: threading.Event, gen: int) -> None:
        try:
            lines = list(target.read(follow=False, tail=200, cancel=cancel))
            if not cancel.is_set():
                self.app.call_from_thread(self._on_reload_tail_success, lines, gen)
        except Exception as e:
            logger.warning("Failed to reload tail: %s", e)
            if not cancel.is_set():
                self.app.call_from_thread(self._on_error, f"Failed to reload tail: {e}", gen)

    def _on_reload_tail_success(self, lines: list[LogLine], gen: int) -> None:
        if not self.is_mounted or gen != self._generation or self._cancel.is_set():
            return
        self.buffer.reload_tail(lines)
        with self._delivered_lock:
            self._delivered_counts.clear()
            for line in lines:
                key = (line.host, line.container)
                self._delivered_counts[key] = self._delivered_counts.get(key, 0) + 1
        # Queued live lines are kept as-is. Line text is never inspected here:
        # a line may appear twice (once in the tail snapshot, once queued), but
        # duplicates beat dropped lines. A flush already in flight (flag set)
        # will pick them up; otherwise schedule exactly one.
        with self._incoming_lock:
            need_schedule = bool(self._incoming_lines) and not self._flush_scheduled
            if need_schedule:
                self._flush_scheduled = True
            self._incoming_cv.notify_all()
        if need_schedule:
            self._schedule_flush(gen)
        self.view_state.state = FollowState.FOLLOWING
        self.view_state.pending_lines = 0
        self._set_current_match(None)
        self.log_view.refresh_buffer()
        self.log_view.scroll_to_end()
        self._update_status()

    def action_toggle_pause_or_end(self) -> None:
        if self.query_one("#log-search", Input).has_focus:
            return
        if self.buffer.evicted_live > 0:
            self._reload_tail()
            return
        at_bottom = self.log_view.is_at_bottom
        jumped = self.view_state.on_space(at_bottom=at_bottom)
        if jumped or self.view_state.is_following:
            self.log_view.scroll_to_end()
        self._update_status()

    def action_scope_container(self) -> None:
        if self.query_one("#log-search", Input).has_focus:
            return
        self._switch_scope(LogScope.CONTAINER)

    def action_scope_node(self) -> None:
        if self.query_one("#log-search", Input).has_focus:
            return
        self._switch_scope(LogScope.NODE)

    def action_scope_cluster(self) -> None:
        if self.query_one("#log-search", Input).has_focus:
            return
        self._switch_scope(LogScope.CLUSTER)

    def _switch_scope(self, new_scope: LogScope) -> None:
        if new_scope == self.scope:
            return
        summary = self.scope_summary
        if summary is not None:
            old_canon = canonical_scope(self.scope, summary)
            new_canon = canonical_scope(new_scope, summary)
            if old_canon == new_canon:
                self.scope = new_scope
                self._update_status()
                return

        self.scope = new_scope
        self._cancel.set()
        with self._incoming_lock:
            self._generation += 1
            self._incoming_lines.clear()
            self._flush_scheduled = False
            self._incoming_cv.notify_all()
        with self._delivered_lock:
            self._delivered_counts.clear()
        self.workers.cancel_group(self, "page")
        self.workers.cancel_group(self, "follow")
        self._loading_page = False
        self._page_failed = False
        self.target = None
        self.log_view.multi_host = False
        self.log_view.multi_source = False
        self.log_view.runtime = None

        old_pattern = self.buffer.search_pattern
        self.buffer = LogBuffer()
        self._set_current_match(None)
        if old_pattern is not None:
            matches = self.buffer.search(old_pattern)
            self.log_view.search_pattern = old_pattern
            if matches:
                self._set_current_match(0)
        else:
            self.log_view.search_pattern = None

        self.view_state = ViewState()
        self.log_view.view_state = self.view_state
        self.log_view.buffer = self.buffer
        self.log_view.refresh_buffer()
        self._update_status()
        self._start_follow_worker(self.scope)

    def action_toggle_color(self) -> None:
        if self.query_one("#log-search", Input).has_focus:
            return
        self.log_view.color = not self.log_view.color
        self.log_view.refresh()
        self._update_status()

    def action_toggle_highlight(self) -> None:
        if self.query_one("#log-search", Input).has_focus:
            return
        self.log_view.highlight = not self.log_view.highlight
        self.log_view.refresh()
        self._update_status()

    def action_toggle_wrap(self) -> None:
        if self.query_one("#log-search", Input).has_focus:
            return
        self.log_view.wrap = not self.log_view.wrap
        self.log_view.refresh_buffer()
        self._update_status()

    def action_open_search(self) -> None:
        search_input = self.query_one("#log-search", Input)
        search_input.display = True
        search_input.value = ""
        search_input.focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "log-search":
            val = event.value.strip()
            if not val:
                self.buffer.search(None)
                self.log_view.search_pattern = None
                self._set_current_match(None)
                event.input.display = False
                self.log_view.focus()
                self.log_view.refresh()
                self._update_status()
                return

            try:
                matches = self.buffer.search(val)
            except re.error as e:
                event.input.value = f"Invalid regex: {e}"
                return

            self.log_view.search_pattern = self.buffer.search_pattern
            event.input.display = False
            self.log_view.focus()

            if matches:
                current_entry = self.log_view.virtual_row_to_entry(int(self.log_view.scroll_offset[1]))
                candidates = [i for i, m in enumerate(matches) if m >= current_entry]
                self._set_current_match(candidates[0] if candidates else 0)
                idx = self._current_match_idx
                assert idx is not None
                self.log_view.scroll_to_line(matches[idx])
            else:
                self._set_current_match(None)

            self.log_view.refresh()
            self._update_status()

    def action_search_next(self) -> None:
        if self.query_one("#log-search", Input).has_focus:
            return
        matches = self.buffer.search_matches
        if not matches:
            return
        current_entry = self.log_view.virtual_row_to_entry(int(self.log_view.scroll_offset[1]))
        if (
            self._current_match_idx is None
            or not (0 <= self._current_match_idx < len(matches))
            or not self._is_match_visible(matches[self._current_match_idx])
        ):
            candidates = [i for i, m in enumerate(matches) if m > current_entry]
            self._set_current_match(candidates[0] if candidates else 0)
        else:
            self._set_current_match((self._current_match_idx + 1) % len(matches))
        idx = self._current_match_idx
        assert idx is not None
        self.log_view.scroll_to_line(matches[idx])
        self.log_view.refresh()
        self._update_status()

    def action_search_prev(self) -> None:
        if self.query_one("#log-search", Input).has_focus:
            return
        matches = self.buffer.search_matches
        if not matches:
            return
        current_entry = self.log_view.virtual_row_to_entry(int(self.log_view.scroll_offset[1]))
        if (
            self._current_match_idx is None
            or not (0 <= self._current_match_idx < len(matches))
            or not self._is_match_visible(matches[self._current_match_idx])
        ):
            candidates = [i for i, m in enumerate(matches) if m < current_entry]
            self._set_current_match(candidates[-1] if candidates else len(matches) - 1)
        else:
            self._set_current_match((self._current_match_idx - 1) % len(matches))
        idx = self._current_match_idx
        assert idx is not None
        self.log_view.scroll_to_line(matches[idx])
        self.log_view.refresh()
        self._update_status()

    def action_dump_logs(self) -> None:
        if self.query_one("#log-search", Input).has_focus:
            return
        self.post_message(self.RequestDump(self.host, self.cluster_id, self.container, self.scope))

    def action_close_pane(self) -> None:
        search_input = self.query_one("#log-search", Input)
        if search_input.display:
            self.buffer.search(None)
            self.log_view.search_pattern = None
            self._set_current_match(None)
            search_input.display = False
            self.log_view.focus()
            self.log_view.refresh()
            self._update_status()
            return
        self.close()
        self.post_message(self.Closed())

    def close(self) -> None:
        self._cancel.set()
        with self._incoming_lock:
            self._generation += 1
            self._incoming_lines.clear()
            self._flush_scheduled = False
            self._incoming_cv.notify_all()
        self.workers.cancel_group(self, "page")
        self.workers.cancel_group(self, "follow")

    def on_unmount(self) -> None:
        if self._gap_update_timer is not None:
            self._gap_update_timer.cancel()
            self._gap_update_timer = None
        self.close()


def format_scope_option(scope: LogScope | int, host: str, container: str, summary: ScopeSummary) -> str:
    scope = LogScope(scope)
    if scope == LogScope.CONTAINER:
        return f"1  This container ({container})"
    elif scope == LogScope.NODE:
        return f"2  This node ({host}, {summary.node_containers} containers)"
    elif scope == LogScope.CLUSTER:
        return f"3  Whole cluster ({summary.cluster_containers} containers on {summary.cluster_hosts} hosts)"
    raise ValueError(f"Unknown scope: {scope!r}")


class DumpLogScreen(ModalScreen[Any]):
    """Modal screen for dumping workload logs to a local file."""

    DEFAULT_CSS = """
    DumpLogScreen {
        align: center middle;
        background: rgba(0, 0, 0, 0.6);
    }
    #dump-dialog {
        width: 72;
        height: auto;
        max-height: 85%;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    #dump-title {
        text-style: bold;
        margin-bottom: 1;
    }
    #dump-scope-list {
        height: auto;
        max-height: 5;
        margin-bottom: 1;
        border: round $primary;
    }
    #dump-path-label {
        margin-top: 1;
        color: $text;
    }
    #dump-path-input {
        margin-bottom: 1;
    }
    #dump-overwrite-list {
        display: none;
        height: auto;
        max-height: 4;
        margin-top: 1;
        border: round $warning;
    }
    #dump-error {
        height: auto;
        color: $error;
    }
    #dump-buttons {
        dock: bottom;
        height: auto;
        margin-top: 1;
    }
    #dump-buttons Button {
        margin-right: 2;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", priority=True),
    ]

    def __init__(
        self,
        host: str,
        cluster_id: str,
        container: str,
        default_scope: LogScope | int = LogScope.CONTAINER,
        *,
        frame: MonitorFrame | None = None,
        summary: ScopeSummary | None = None,
        cluster: str | None = None,
        cache_dir: str | None = None,
        sctx: Any = None,
        recipe_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.host: str = host
        self.cluster_id: str = cluster_id
        self.container: str = container
        self.cluster: str | None = cluster
        self.cache_dir: str | None = cache_dir
        self.sctx: Any = sctx

        if recipe_name:
            self.label: str = recipe_name
        else:
            meta = load_job_metadata(cluster_id, cache_dir=cache_dir) or {}
            self.label = meta.get("recipe") or cluster_id

        self.summary: ScopeSummary = summary or (
            workload_scope_summary(frame, cluster_id, host)
            if frame
            else ScopeSummary(node_containers=1, cluster_containers=1, cluster_hosts=1)
        )
        self._scopes: list[LogScope] = distinct_scopes(self.summary)
        self._has_scope_list: bool = len(self._scopes) > 1

        canon_default = canonical_scope(default_scope, self.summary)
        if canon_default in self._scopes:
            self._default_scope_idx: int = self._scopes.index(canon_default)
            self.chosen_scope: LogScope = canon_default
        else:
            self._default_scope_idx = 0
            self.chosen_scope = self._scopes[0]

        self._scope_options: list[tuple[LogScope, str]] = [
            (s, format_scope_option(s, self.host, self.container, self.summary)) for s in self._scopes
        ]

        self._path_edited: bool = False
        self._initial_path: str = default_dump_path(self.label, self.chosen_scope, self.host, self.container)
        self._last_default_path: str = self._initial_path
        self._pending_dump_path: str | None = None
        self._cancel: threading.Event = threading.Event()
        self._dumping: bool = False

    def compose(self) -> ComposeResult:
        with Vertical(id="dump-dialog"):
            yield Static(f"Dump logs: {self.label}", id="dump-title")
            if self._has_scope_list:
                yield OptionList(
                    *[Option(prompt=text, id=str(int(s))) for s, text in self._scope_options],
                    id="dump-scope-list",
                )
            yield Static("File path:", id="dump-path-label")
            yield Input(value=self._initial_path, id="dump-path-input")
            yield OptionList(
                Option("Overwrite", id="overwrite"),
                Option("Cancel", id="cancel"),
                id="dump-overwrite-list",
            )
            yield Static(id="dump-error")
            with Horizontal(id="dump-buttons"):
                yield Button("Cancel", id="dump-cancel")
                yield Button("OK", id="dump-ok", variant="primary")

    def on_mount(self) -> None:
        if self._has_scope_list:
            scope_list = self.query_one("#dump-scope-list", OptionList)
            scope_list.highlighted = self._default_scope_idx
            scope_list.focus()
        else:
            self.query_one("#dump-path-input", Input).focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        if self._dumping:
            return
        if event.input.id == "dump-path-input":
            if event.value != self._last_default_path:
                self._path_edited = True
            if self._pending_dump_path is not None and event.value != self._pending_dump_path:
                self._pending_dump_path = None
                try:
                    overwrite_list = self.query_one("#dump-overwrite-list", OptionList)
                    overwrite_list.display = False
                    self._set_error("")
                except Exception:
                    pass

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if self._dumping:
            return
        if event.option_list.id == "dump-scope-list":
            idx = event.option_index
            if 0 <= idx < len(self._scopes):
                self.chosen_scope = self._scopes[idx]
                if not self._path_edited:
                    new_path = default_dump_path(self.label, self.chosen_scope, self.host, self.container)
                    self._last_default_path = new_path
                    self.query_one("#dump-path-input", Input).value = new_path
            self.query_one("#dump-path-input", Input).focus()
        elif event.option_list.id == "dump-overwrite-list":
            if event.option_id == "overwrite" or event.option_index == 0:
                if self._pending_dump_path:
                    self._start_dump(self._pending_dump_path, overwrite=True)
            else:
                self.dismiss(None)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "dump-path-input":
            self._submit_path(event.value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "dump-cancel":
            self.action_cancel()
        elif event.button.id == "dump-ok":
            self._submit_path(self.query_one("#dump-path-input", Input).value)

    def _submit_path(self, value: str) -> None:
        """Validate the path and start the dump (or ask before overwriting)."""
        if self._dumping:
            return
        path_str = value.strip()
        if not path_str:
            self._set_error("File path cannot be empty")
            return
        resolved = os.path.abspath(os.path.expanduser(path_str))
        parent = os.path.dirname(resolved)
        if not os.path.isdir(parent):
            self._set_error(f"Directory {parent!r} does not exist")
            return
        if not os.access(parent, os.W_OK):
            self._set_error(f"Directory {parent!r} is not writable")
            return
        if self._pending_dump_path == resolved:
            # A second OK (or Enter) on the overwrite prompt confirms it.
            self._start_dump(resolved, overwrite=True)
            return
        if os.path.exists(resolved):
            self._prompt_overwrite(resolved)
            return
        self._start_dump(resolved, overwrite=False)

    def _prompt_overwrite(self, resolved_path: str) -> None:
        self._pending_dump_path = resolved_path
        self._set_error("[yellow]File exists. Confirm overwrite:[/yellow]")
        overwrite_list = self.query_one("#dump-overwrite-list", OptionList)
        overwrite_list.display = True
        overwrite_list.highlighted = 1  # Cancel is default
        overwrite_list.focus()

    def _set_error(self, message: str) -> None:
        try:
            self.query_one("#dump-error", Static).update(message)
        except Exception:
            pass

    def action_cancel(self) -> None:
        if self._dumping:
            self._cancel.set()
            self.workers.cancel_group(self, "dump")
            self.dismiss(None)
            return
        focused = self.focused
        if focused and focused.id == "dump-path-input" and self._has_scope_list:
            self.query_one("#dump-scope-list", OptionList).focus()
            return
        self._cancel.set()
        self.workers.cancel_group(self, "dump")
        self.dismiss(None)

    def on_unmount(self) -> None:
        self._cancel.set()
        self.workers.cancel_group(self, "dump")

    def _start_dump(self, path: str, overwrite: bool) -> None:
        if self._dumping:
            return
        self._dumping = True
        try:
            self.query_one("#dump-path-input", Input).disabled = True
            self.query_one("#dump-ok", Button).disabled = True
            if self._has_scope_list:
                self.query_one("#dump-scope-list", OptionList).disabled = True
            overwrite_list = self.query_one("#dump-overwrite-list", OptionList)
            if overwrite_list.display:
                overwrite_list.disabled = True
        except Exception:
            pass
        self._set_error("[cyan]Dumping logs…[/cyan]")
        scope = self.chosen_scope
        self._dump_worker(path, overwrite, scope)

    @work(thread=True, exclusive=True, group="dump")
    def _dump_worker(self, path: str, overwrite: bool, scope: LogScope) -> None:
        try:
            t_args = target_args(scope, self.host, self.container)
            target = open_log_target(
                self.cluster_id,
                cluster=self.cluster,
                cache_dir=self.cache_dir,
                sctx=self.sctx,
                **t_args,
            )
            count = target.dump(path, overwrite=overwrite, cancel=self._cancel)
            if self._cancel.is_set():
                return

            if scope == LogScope.CONTAINER:
                scope_desc = f"container {self.container}"
            elif scope == LogScope.NODE:
                scope_desc = f"node {self.host}"
            else:
                scope_desc = "cluster"

            def _on_success() -> None:
                if self._cancel.is_set():
                    return
                self.app.notify(f"Dumped {count} line(s) ({scope_desc}) to {path}")
                self.dismiss(path)

            self.app.call_from_thread(_on_success)
        except Exception as e:
            if self._cancel.is_set():
                return

            def _on_err(err_msg: str) -> None:
                self._dumping = False
                try:
                    path_input = self.query_one("#dump-path-input", Input)
                    path_input.disabled = False
                    self.query_one("#dump-ok", Button).disabled = False
                    if self._has_scope_list:
                        self.query_one("#dump-scope-list", OptionList).disabled = False
                    overwrite_list = self.query_one("#dump-overwrite-list", OptionList)
                    if overwrite_list.display:
                        overwrite_list.disabled = False
                    path_input.focus()
                except Exception:
                    pass
                self._set_error(f"[red]{err_msg}[/red]")

            self.app.call_from_thread(_on_err, str(e))


class StopWorkloadScreen(ModalScreen[bool]):
    """Modal screen confirming a workload stop; dismisses True to confirm."""

    DEFAULT_CSS = """
    StopWorkloadScreen {
        align: center middle;
        background: rgba(0, 0, 0, 0.6);
    }
    # A neutral elevated shade, not $panel: $panel carries the theme's accent
    # tint and dims the body text. The buttons keep their own $surface fill,
    # which stays visible against the neutral dialog.
    #stop-dialog {
        width: 60;
        height: auto;
        border: thick $warning;
        background: $background-lighten-2;
        padding: 1 2;
    }
    #stop-title {
        text-style: bold;
        margin-bottom: 1;
    }
    #stop-buttons {
        height: auto;
        margin-top: 1;
        align: center middle;
    }
    #stop-buttons Button {
        margin: 0 1;
    }
    """

    BINDINGS = [
        # The buttons are the whole interface: left/right (and up/down) move
        # focus between them and enter/space activates the focused one. Escape
        # and q still cancel without leaving the button row.
        Binding("left", "app.focus_previous", "Previous", show=False, priority=True),
        Binding("up", "app.focus_previous", "Previous", show=False, priority=True),
        Binding("right", "app.focus_next", "Next", show=False, priority=True),
        Binding("down", "app.focus_next", "Next", show=False, priority=True),
        Binding("escape", "cancel", "Cancel", priority=True),
        Binding("q", "cancel", "Cancel", show=False, priority=True),
    ]

    def __init__(self, host: str, cluster_id: str, container: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.host = host
        self.cluster_id = cluster_id
        self.container = container

    def compose(self) -> ComposeResult:
        with Vertical(id="stop-dialog"):
            yield Static(f"Stop workload [bold]{self.cluster_id}[/bold]?", id="stop-title")
            yield Static(f"Every container of the workload will be stopped (highlighted: {self.container} on {self.host}).")
            with Horizontal(id="stop-buttons"):
                yield Button("Cancel", id="stop-cancel")
                yield Button("Stop", id="stop-confirm", variant="error")

    def on_mount(self) -> None:
        self.query_one("#stop-cancel", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "stop-confirm":
            self.action_confirm()
        elif event.button.id == "stop-cancel":
            self.action_cancel()

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)
