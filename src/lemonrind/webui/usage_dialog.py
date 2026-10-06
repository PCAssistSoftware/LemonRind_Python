"""The usage screen: tokens and speed over time, per model and per tool, from the requests this app has saved.

Opened by "View usage" at the bottom of the right-hand panel. The numbers come from ``chats/usage.py`` (plain
functions, tested on their own); this file only chooses the period, asks for the report and draws it.

Python / NiceGUI ideas used here:

* ``@ui.refreshable`` again: changing the period, the model or the scheduled-jobs switch just calls ``refresh()`` and
  the whole report is drawn afresh from the new numbers.
* ``ui.echart``: a chart described by a plain dictionary of options (the ECharts library's own format). No chart
  code to write, only data.
* ``ui.table`` columns with a ``:format`` entry: a keyword starting with a colon is passed to the browser as
  JavaScript, so the cell can show ``1,234,567`` while the column still sorts by the real number.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, time, timedelta, tzinfo

from nicegui import ui

from lemonrind.chats.usage import (
    ALL_MODELS,
    UNKNOWN_MODEL,
    ToolUse,
    UsageReport,
    UsageRequest,
    build_report,
    compact,
)
from lemonrind.webui.context import AppContext
from lemonrind.webui.inspectors import make_dialog

PERIODS: dict[str, int | None] = {"7 days": 7, "30 days": 30, "12 months": 365, "All time": None}
DEFAULT_PERIOD = "30 days"

INPUT_COLOUR = "#1976d2"
OUTPUT_COLOUR = "#26a69a"
SPEED_COLOUR = "#f4a300"

# Colours that read on both the light and the dark theme: the chart's own defaults are black or white lines.
AXIS_TEXT = "#8a93a2"
GRID_LINE = "rgba(127, 127, 127, 0.25)"
VALUE_AXIS = {
    "type": "value",
    "splitLine": {"lineStyle": {"color": GRID_LINE}},
    "axisLabel": {"color": AXIS_TEXT},
}


def _category_axis(labels: list[str]) -> dict:
    return {"type": "category", "data": labels, "axisLabel": {"color": AXIS_TEXT}}


def period_start(days: int | None, now: datetime, tz: tzinfo) -> datetime | None:
    """The first moment of the period: midnight, ``days`` calendar days back counting today as one (``None``: no start)."""
    if days is None:
        return None
    first_day = now.astimezone(tz).date() - timedelta(days=days - 1)
    return datetime.combine(first_day, time.min, tzinfo=tz)


def _tile(value: str, label: str) -> None:
    with ui.column().classes("gap-0"):
        ui.label(value).classes("text-h6 text-weight-bold text-primary")
        ui.label(label).classes("text-caption lr-muted")


def _table(columns: list[dict], rows: list[dict], row_key: str, marker: str) -> None:
    ui.table(columns=columns, rows=rows, row_key=row_key).props("dense flat").classes(
        "w-full"
    ).mark(marker)


def _column(name: str, label: str, *, text: bool = False, js_format: str | None = None) -> dict:
    column: dict = {
        "name": name,
        "label": label,
        "field": name,
        "sortable": True,
        "align": "left" if text else "right",
    }
    if js_format:
        column[":format"] = (
            js_format  # runs in the browser, so the cell shows 1,234 but still sorts as a number
        )
    return column


THOUSANDS = "val => val.toLocaleString()"
SPEED = "val => val ? val.toFixed(0) : '-'"
SECONDS = "val => val ? val.toFixed(1) + ' s' : '-'"
PERCENT = "val => (val * 100).toFixed(0) + '%'"


def _token_chart(report: UsageReport) -> None:
    labels = [bucket.label for bucket in report.buckets]
    ui.echart(
        {
            "tooltip": {"trigger": "axis"},
            "legend": {
                "data": ["Tokens in", "Tokens out"],
                "top": 0,
                "textStyle": {"color": AXIS_TEXT},
            },
            "grid": {"left": 56, "right": 16, "top": 32, "bottom": 28},
            "xAxis": _category_axis(labels),
            "yAxis": VALUE_AXIS,
            "series": [
                {
                    "name": "Tokens in",
                    "type": "bar",
                    "stack": "tokens",
                    "color": INPUT_COLOUR,
                    "data": [bucket.input_tokens for bucket in report.buckets],
                },
                {
                    "name": "Tokens out",
                    "type": "bar",
                    "stack": "tokens",
                    "color": OUTPUT_COLOUR,
                    "data": [bucket.output_tokens for bucket in report.buckets],
                },
            ],
        }
    ).classes("w-full h-56 shrink-0").mark("usage-chart-tokens")


def _speed_chart(report: UsageReport) -> None:
    ui.echart(
        {
            "tooltip": {"trigger": "axis"},
            "legend": {"data": ["Tokens per second"], "top": 0, "textStyle": {"color": AXIS_TEXT}},
            "grid": {"left": 56, "right": 16, "top": 32, "bottom": 28},
            "xAxis": _category_axis([bucket.label for bucket in report.buckets]),
            "yAxis": VALUE_AXIS,
            "series": [
                {
                    "name": "Tokens per second",
                    "type": "line",
                    "color": SPEED_COLOUR,
                    "connectNulls": True,  # a period with no requests has no speed: the line bridges the gap
                    "data": [
                        round(bucket.tokens_per_second, 1) if bucket.tokens_per_second else None
                        for bucket in report.buckets
                    ],
                }
            ],
        }
    ).classes("w-full h-40 shrink-0").mark("usage-chart-speed")


class UsageScreen:
    """The state of the open screen: the saved rows, and what the three controls are set to."""

    def __init__(
        self,
        requests: list[UsageRequest],
        tool_uses: list[ToolUse],
        on_open_chat: Callable[[str], None],
        close: Callable[[], None],
    ) -> None:
        self._requests = requests
        self._tool_uses = tool_uses
        self._on_open_chat = on_open_chat
        self._close = close
        self.period = DEFAULT_PERIOD
        self.model = ALL_MODELS
        self.include_scheduled = True
        self.model_select: ui.select | None = None

    def build(self) -> None:
        with ui.row().classes("w-full items-center gap-4"):
            ui.toggle(list(PERIODS), value=self.period, on_change=self._period_changed).props(
                "dense no-caps"
            ).mark("usage-period")
            self.model_select = (
                ui.select(
                    [ALL_MODELS], value=ALL_MODELS, label="Model", on_change=self._model_changed
                )
                .props("dense outlined")
                .classes("w-64")
                .mark("usage-model")
            )
            ui.switch(
                "Include scheduled jobs",
                value=True,
                on_change=self._scheduled_changed,
            ).mark("usage-scheduled")
        # The dialog card is a flex column with a maximum height. A flex child may shrink to fit, which squeezed the
        # charts (they only have a height class) down to nothing in a short window. ``shrink-0`` on the report's own
        # column makes it keep its natural height, and the card scrolls instead.
        with ui.column().classes("w-full gap-3 shrink-0"):
            self.show()

    def _period_changed(self, event) -> None:
        self.period = event.value
        self.show.refresh()

    def _model_changed(self, event) -> None:
        chosen = event.value or ALL_MODELS
        if chosen == self.model:  # set_options during a refresh reports the same value again
            return
        self.model = chosen
        self.show.refresh()

    def _scheduled_changed(self, event) -> None:
        self.include_scheduled = bool(event.value)
        self.show.refresh()

    def _report(self, model: str | None) -> UsageReport:
        now = datetime.now(UTC)
        tz = now.astimezone().tzinfo or UTC
        return build_report(
            self._requests,
            self._tool_uses,
            since=period_start(PERIODS[self.period], now, tz),
            until=now,
            tz=tz,
            model=model,
            include_scheduled=self.include_scheduled,
        )

    @ui.refreshable
    def show(self) -> None:
        report = self._report(None if self.model == ALL_MODELS else self.model)
        if self.model != ALL_MODELS and self.model not in report.available_models:
            self.model = (
                ALL_MODELS  # the chosen model has no requests in this period: go back to all
            )
            report = self._report(None)
        if self.model_select is not None:
            self.model_select.set_options([ALL_MODELS, *report.available_models], value=self.model)

        totals = report.totals
        if totals.requests == 0:
            ui.label("No requests in this period.").classes("lr-muted q-pa-md").mark("usage-empty")
            return

        with ui.row().classes("w-full gap-8"):
            _tile(f"{totals.requests:,}", "Requests")
            _tile(compact(totals.input_tokens), "Tokens in")
            _tile(compact(totals.output_tokens), "Tokens out")
            _tile(
                f"{totals.tokens_per_second:.0f} tok/s" if totals.tokens_per_second else "-",
                "Average speed",
            )
            _tile(
                f"{totals.time_to_first_token:.1f} s" if totals.time_to_first_token else "-",
                "Average first token",
            )

        ui.label(f"Tokens per {report.bucket_kind}").classes("text-weight-medium q-mt-sm")
        _token_chart(report)
        _speed_chart(report)

        ui.label("By model").classes("text-weight-medium q-mt-sm")
        _table(
            [
                _column("model", "Model", text=True),
                _column("requests", "Requests", js_format=THOUSANDS),
                _column("input", "Tokens in", js_format=THOUSANDS),
                _column("output", "Tokens out", js_format=THOUSANDS),
                _column("speed", "Avg tok/s", js_format=SPEED),
                _column("first", "Avg first token", js_format=SECONDS),
                _column("slowest", "Slowest start", js_format=SECONDS),
            ],
            [
                {
                    "model": row.model,
                    "requests": row.totals.requests,
                    "input": row.totals.input_tokens,
                    "output": row.totals.output_tokens,
                    "speed": row.totals.tokens_per_second,
                    "first": row.totals.time_to_first_token,
                    "slowest": row.slowest_start,
                }
                for row in report.models
            ],
            "model",
            "usage-models",
        )

        ui.label("Tools").classes("text-weight-medium q-mt-sm")
        if report.tools:
            _table(
                [
                    _column("name", "Tool", text=True),
                    _column("calls", "Calls", js_format=THOUSANDS),
                    _column("failed", "Failed", js_format=THOUSANDS),
                    _column("rate", "Failure rate", js_format=PERCENT),
                ],
                [
                    {
                        "name": tool.name,
                        "calls": tool.calls,
                        "failed": tool.failed,
                        "rate": tool.failure_rate,
                    }
                    for tool in report.tools
                ],
                "name",
                "usage-tools",
            )
        else:
            ui.label("No tool calls in this period.").classes("text-caption lr-muted")

        if report.chats:
            ui.label("Busiest chats").classes("text-weight-medium q-mt-sm")
            with ui.column().classes("gap-0 items-start"):
                for chat in report.chats:
                    ui.button(
                        f"{chat.title}  ({compact(chat.tokens)} tokens)",
                        on_click=lambda sid=chat.session_id: self._open(sid),
                    ).props("flat dense no-caps").mark("usage-chat")

        notes = [
            "Tokens in is what the model had to process for each request; the part of the conversation it had "
            "already stored is not counted again. A reply that uses tools takes several requests.",
            "Speed is averaged by how long each answer was, so a long answer counts for more than a short one.",
        ]
        if UNKNOWN_MODEL in report.available_models:
            notes.append(
                "The model's name has been saved since this screen was added; earlier requests are listed as "
                f"'{UNKNOWN_MODEL}'. They still count everywhere else."
            )
        for note in notes:
            ui.label(note).classes("text-caption lr-muted")

    def _open(self, session_id: str) -> None:
        self._close()
        self._on_open_chat(session_id)


def show_usage(context: AppContext, on_open_chat: Callable[[str], None]) -> None:
    """Open the usage screen. The saved rows are read once, when it opens."""
    dialog, card = make_dialog(
        "Usage",
        "Tokens and speed from the requests this app has saved (chats and scheduled jobs).",
        wide=True,
    )
    screen = UsageScreen(
        context.repo.usage_requests(),
        context.repo.usage_tool_uses(),
        on_open_chat,
        dialog.close,
    )
    with card:
        screen.build()
    dialog.open()
