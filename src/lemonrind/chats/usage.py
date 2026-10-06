"""Usage statistics: turning saved request numbers into totals, a time series and tables.

Every model request a chat made is saved with its token counts, speed and time to first token (see ``RequestStats``).
This module takes those rows and answers questions such as "how much did I use the assistant this month?", "which
model is quicker on this computer?" and "which tool fails most often?".

It is plain functions and data classes with no database and no widgets, so every rule can be tested with a handful of
made-up rows. The repository supplies the rows (``ChatRepository.usage_requests`` and ``usage_tool_uses``) and the
usage screen (``webui/usage_dialog.py``) draws the report.

Python ideas used here:

* ``dataclass(frozen=True, slots=True)`` for small immutable records.
* ``collections.defaultdict``: a dictionary that makes a missing entry (an empty list, a zero) the first time it is
  used, which turns "group these rows by key" into a three-line loop.
* Time zones as a parameter: the caller passes a ``tzinfo`` and ``datetime.astimezone`` converts, so the code never
  asks the computer what time zone it is, and a test can use any zone.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, tzinfo

UNKNOWN_MODEL = "Unknown (before tracking)"
ALL_MODELS = "All models"
TOP_CHATS = 5

DAY_LIMIT = 45  # a period up to this many days is charted day by day
WEEK_LIMIT = 400  # up to this many, week by week; longer periods are charted month by month


@dataclass(frozen=True, slots=True)
class UsageRequest:
    """One model request: when it finished, in which chat, with which model, and the numbers it produced."""

    at: datetime
    session_id: str
    title: str
    model: str | None  # None for requests saved before the model name was recorded
    input_tokens: int
    output_tokens: int
    tokens_per_second: float
    time_to_first_token: float
    scheduled: bool  # the chat was made by a scheduled job


@dataclass(frozen=True, slots=True)
class ToolUse:
    """One tool call and whether it failed."""

    at: datetime
    name: str
    failed: bool
    model: str | None  # the model that asked for the call
    scheduled: bool


@dataclass(frozen=True, slots=True)
class Totals:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    tokens_per_second: float = 0.0  # average, weighted by how many tokens each request produced
    time_to_first_token: float = 0.0  # average, in seconds


@dataclass(frozen=True, slots=True)
class Bucket:
    """One bar of the chart: a day, a week or a month."""

    start: date
    label: str
    input_tokens: int
    output_tokens: int
    tokens_per_second: float


@dataclass(frozen=True, slots=True)
class ModelRow:
    model: str
    totals: Totals
    slowest_start: float  # the longest wait for a first token, in seconds


@dataclass(frozen=True, slots=True)
class ToolRow:
    name: str
    calls: int
    failed: int

    @property
    def failure_rate(self) -> float:
        return self.failed / self.calls if self.calls else 0.0


@dataclass(frozen=True, slots=True)
class ChatRow:
    session_id: str
    title: str
    tokens: int


@dataclass(frozen=True, slots=True)
class UsageReport:
    totals: Totals
    bucket_kind: str  # "day", "week" or "month"
    buckets: list[Bucket]
    models: list[ModelRow]
    tools: list[ToolRow]
    chats: list[ChatRow]
    available_models: list[str] = field(
        default_factory=list
    )  # for the filter, whatever the filter is set to


def model_name(model: str | None) -> str:
    return model if model else UNKNOWN_MODEL


def totals_of(requests: list[UsageRequest]) -> Totals:
    """Add up a group of requests. Speed is weighted by output tokens: a long answer counts for more than a short one."""
    if not requests:
        return Totals()
    output = sum(r.output_tokens for r in requests)
    timed = [r for r in requests if r.tokens_per_second > 0 and r.output_tokens > 0]
    weight = sum(r.output_tokens for r in timed)
    speed = sum(r.tokens_per_second * r.output_tokens for r in timed) / weight if weight else 0.0
    waits = [r.time_to_first_token for r in requests if r.time_to_first_token > 0]
    return Totals(
        requests=len(requests),
        input_tokens=sum(r.input_tokens for r in requests),
        output_tokens=output,
        tokens_per_second=speed,
        time_to_first_token=sum(waits) / len(waits) if waits else 0.0,
    )


def bucket_kind_for(first: date, last: date) -> str:
    """Charting by day, week or month, whichever keeps the chart readable for a span of this length."""
    days = (last - first).days + 1
    if days <= DAY_LIMIT:
        return "day"
    return "week" if days <= WEEK_LIMIT else "month"


def _bucket_start(day: date, kind: str) -> date:
    if kind == "week":
        return day - timedelta(days=day.weekday())  # the Monday of that week
    if kind == "month":
        return day.replace(day=1)
    return day


def _next_bucket(start: date, kind: str) -> date:
    if kind == "week":
        return start + timedelta(days=7)
    if kind == "month":
        return date(start.year + start.month // 12, start.month % 12 + 1, 1)
    return start + timedelta(days=1)


def _label(start: date, kind: str) -> str:
    if kind == "month":
        return f"{start:%b %Y}"
    prefix = "w/c " if kind == "week" else ""
    return f"{prefix}{start.day} {start:%b}"


def build_report(
    requests: list[UsageRequest],
    tool_uses: list[ToolUse],
    *,
    since: datetime | None,
    until: datetime,
    tz: tzinfo,
    model: str | None = None,
    include_scheduled: bool = True,
) -> UsageReport:
    """Everything the usage screen shows, for requests between ``since`` (``None`` = from the start) and ``until``.

    ``model`` limits the report to one model (``UNKNOWN_MODEL`` picks the requests saved without one); ``None`` means
    all of them. ``include_scheduled=False`` leaves out chats that scheduled jobs made. Days are the computer's own
    (``tz``), not UTC, so a late-evening reply counts for the day you had it.
    """

    def in_period(at: datetime) -> bool:
        return (since is None or at >= since) and at <= until

    in_scope = [
        r for r in requests if in_period(r.at) and (include_scheduled or not r.scheduled)
    ]  # the model filter is applied after, so the filter's own list can offer every model in this period
    available = sorted({model_name(r.model) for r in in_scope}, key=str.lower)
    selected = [r for r in in_scope if model is None or model_name(r.model) == model]

    tools_selected = [
        t
        for t in tool_uses
        if in_period(t.at)
        and (include_scheduled or not t.scheduled)
        and (model is None or model_name(t.model) == model)
    ]

    kind = _kind_for(selected, since, until, tz)
    return UsageReport(
        totals=totals_of(selected),
        bucket_kind=kind,
        buckets=_buckets(selected, since, until, tz, kind),
        models=_model_rows(selected),
        tools=_tool_rows(tools_selected),
        chats=_chat_rows(selected),
        available_models=available,
    )


def _kind_for(
    selected: list[UsageRequest], since: datetime | None, until: datetime, tz: tzinfo
) -> str:
    first, last = _span(selected, since, until, tz)
    return bucket_kind_for(first, last)


def _span(
    selected: list[UsageRequest], since: datetime | None, until: datetime, tz: tzinfo
) -> tuple[date, date]:
    last = until.astimezone(tz).date()
    if since is not None:
        return since.astimezone(tz).date(), last
    if selected:
        return min(r.at for r in selected).astimezone(tz).date(), last
    return last, last


def _buckets(
    selected: list[UsageRequest], since: datetime | None, until: datetime, tz: tzinfo, kind: str
) -> list[Bucket]:
    first, last = _span(selected, since, until, tz)
    grouped: dict[date, list[UsageRequest]] = defaultdict(list)
    for request in selected:
        grouped[_bucket_start(request.at.astimezone(tz).date(), kind)].append(request)

    buckets: list[Bucket] = []
    start = _bucket_start(first, kind)
    while start <= last:  # every period appears, even one with no use, so the chart has no gaps
        totals = totals_of(grouped.get(start, []))
        buckets.append(
            Bucket(
                start,
                _label(start, kind),
                totals.input_tokens,
                totals.output_tokens,
                totals.tokens_per_second,
            )
        )
        start = _next_bucket(start, kind)
    return buckets


def _model_rows(selected: list[UsageRequest]) -> list[ModelRow]:
    grouped: dict[str, list[UsageRequest]] = defaultdict(list)
    for request in selected:
        grouped[model_name(request.model)].append(request)
    rows = [
        ModelRow(name, totals_of(group), max(r.time_to_first_token for r in group))
        for name, group in grouped.items()
    ]
    return sorted(rows, key=lambda row: (-row.totals.requests, row.model.lower()))


def _tool_rows(tool_uses: list[ToolUse]) -> list[ToolRow]:
    calls: dict[str, int] = defaultdict(int)
    failed: dict[str, int] = defaultdict(int)
    for use in tool_uses:
        calls[use.name] += 1
        failed[use.name] += int(use.failed)
    rows = [ToolRow(name, calls[name], failed[name]) for name in calls]
    return sorted(rows, key=lambda row: (-row.calls, row.name.lower()))


def _chat_rows(selected: list[UsageRequest]) -> list[ChatRow]:
    tokens: dict[str, int] = defaultdict(int)
    titles: dict[str, str] = {}
    for request in selected:
        tokens[request.session_id] += request.input_tokens + request.output_tokens
        titles[request.session_id] = request.title
    rows = [ChatRow(sid, titles[sid], total) for sid, total in tokens.items() if total > 0]
    return sorted(rows, key=lambda row: -row.tokens)[:TOP_CHATS]


def compact(number: float) -> str:
    """A short form of a count: ``950``, ``1.2 k``, ``3.84 M``."""
    value = abs(number)
    if value < 1000:
        return f"{number:,.0f}"
    if value < 1_000_000:
        return f"{number / 1000:.1f} k"
    return f"{number / 1_000_000:.2f} M"
