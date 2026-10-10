"""The statistics panel in the right drawer: the last reply and the running totals for the chat."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from nicegui import ui

from lemonrind.lemonade.events import RequestStats
from lemonrind.modules.base import Module
from lemonrind.webui.bubbles import ToolChip


@dataclass(slots=True)
class SessionTotals:
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, stats: RequestStats | None) -> None:
        if stats is not None:
            self.input_tokens += stats.input_tokens
            self.output_tokens += stats.output_tokens


def _tile(value: str, label: str, *, color: str = "primary") -> None:
    with ui.column().classes("gap-0"):
        ui.label(value).classes(f"text-h6 text-weight-bold text-{color}")
        ui.label(label).classes("text-caption lr-muted")


CHIP_COLOURS = {"ok": "positive", "failed": "negative", "pending": "warning", "abandoned": "grey"}
CHIP_TITLES = {
    "ok": "Succeeded",
    "failed": "Failed",
    "pending": "Running",
    "abandoned": "Did not finish",
}


@dataclass(slots=True)
class ChipGroup:
    """Every call of one tool that ended the same way, shown as one badge ("web_search x12")."""

    name: str
    state: str
    count: int
    details: list[str]  # the distinct error texts of failed calls, in the order they happened


def group_chips(chips: Sequence[ToolChip]) -> list[ChipGroup]:
    """Fold a run of tool calls into one badge per (tool, outcome), tools in the order they were first used.

    A long scheduled run makes dozens of calls to the same few tools; a row per call is unreadable. Outcomes stay
    separate, so one failed call among twelve is still a red badge of its own and not hidden in a green total.
    """
    groups: dict[tuple[str, str], ChipGroup] = {}
    for chip in chips:
        group = groups.setdefault((chip.name, chip.state), ChipGroup(chip.name, chip.state, 0, []))
        group.count += 1
        if chip.detail and chip.detail not in group.details:
            group.details.append(chip.detail)
    first_use = {name: i for i, name in enumerate(dict.fromkeys(c.name for c in chips))}
    return sorted(
        groups.values(), key=lambda g: first_use[g.name]
    )  # stable: outcomes stay in the order met


TIP_ERROR_CHARS = 140  # one error's share of a tooltip: errors can run to thousands of characters
TIP_ERRORS_SHOWN = 3


def short_error(text: str) -> str:
    """An error text boiled down for a tooltip: one line, no web-address query strings, cut to a readable length.

    Errors from web tools often repeat a long signed address several times; the query part (everything after the
    ``?``) is what makes them unreadable, and it is never the useful part. The full text stays in the tool's panel
    in the chat.
    """
    plain = re.sub(r"\?[^\s'\"]+", "?...", " ".join(text.split()))
    if len(plain) > TIP_ERROR_CHARS:
        plain = plain[: TIP_ERROR_CHARS - 3].rstrip() + "..."
    return plain


def _chip_tooltip(group: ChipGroup) -> str:
    """What you see on hover: short versions of the errors for failures (up to three kinds), else how calls ended."""
    title = CHIP_TITLES.get(group.state, "")
    if group.details:
        kinds = list(dict.fromkeys(short_error(d) for d in group.details))
        lines = kinds[:TIP_ERRORS_SHOWN]
        if len(kinds) > TIP_ERRORS_SHOWN:
            lines.append(f"(and {len(kinds) - TIP_ERRORS_SHOWN} more kinds of error)")
        lines.append("The full text is in the tool's panel in the chat.")
        return "\n".join(lines)
    return f"{title} ({group.count} calls)" if group.count > 1 else title


class StatsPanel:
    """The right-hand panel: two tabs, Stats (numbers and tool calls) and Modules (what is on and what is off).

    It shows what it is given and never talks to the model itself.
    """

    def __init__(
        self,
        modules: Callable[[], Sequence[Module]] = lambda: (),
        on_tab_change: Callable[[str], None] | None = None,
    ) -> None:
        self._modules = modules
        self._on_tab_change = on_tab_change
        self._last: RequestStats | None = None
        self._totals = SessionTotals()
        self._chips: list[ToolChip] = []
        self.tab = "stats"

    def build(self) -> None:
        # Two boxes side by side with a line between them; the open one is filled. (A "segmented control".)
        self._tab_buttons: dict[str, ui.button] = {}
        with ui.row().classes("lr-segments w-full no-wrap q-mb-sm"):
            for key, label in (("stats", "Stats"), ("modules", "Modules")):
                self._tab_buttons[key] = (
                    ui.button(label, on_click=lambda k=key: self.select(k))
                    .props("flat dense no-caps")
                    .classes("lr-segment")
                    .mark(f"tab-{key}")
                )
        self._style_tabs()
        self.render()

    def _style_tabs(self) -> None:
        """The open tab is a filled button with white text; the other is a plain one.

        Done with Quasar's own button properties (``unelevated``, ``color``, ``text-color``) rather than CSS, because
        Quasar's colour rules win over a stylesheet.
        """
        for key, button in self._tab_buttons.items():
            if key == self.tab:
                button.classes(add="lr-segment-active")
                button.props(
                    remove="flat text-color", add="unelevated color=primary text-color=white"
                )
            else:
                button.classes(remove="lr-segment-active")
                button.props(remove="unelevated text-color", add="flat color=primary")

    def select(self, tab: str) -> None:
        self.tab = tab
        self._style_tabs()
        self.render.refresh()
        if self._on_tab_change is not None:
            self._on_tab_change(tab)

    def update(self, last: RequestStats | None, totals: SessionTotals) -> None:
        self._last = last
        self._totals = totals
        self.render.refresh()

    def set_tool_calls(self, chips: list[ToolChip]) -> None:
        """Show the tool calls of the latest reply (empty list: none)."""
        self._chips = chips
        self.render.refresh()

    @ui.refreshable
    def render(self) -> None:
        if self.tab == "modules":
            self._render_modules()
        else:
            self._render_stats()

    def _render_stats(self) -> None:
        last = self._last
        with ui.card().props("flat bordered").classes("w-full"):
            ui.label("Last reply").classes("text-weight-medium")
            with ui.grid(columns=2).classes("w-full"):
                _tile(f"{last.input_tokens:,}" if last else "0", "Input tokens")
                _tile(f"{last.output_tokens:,}" if last else "0", "Output tokens")
                _tile(f"{last.tokens_per_second:.1f}" if last else "0.0", "Tokens / sec")
                _tile(f"{last.time_to_first_token:.2f}s" if last else "0.00s", "First token")
        with ui.card().props("flat bordered").classes("w-full"):
            ui.label("This chat").classes("text-weight-medium")
            with ui.grid(columns=2).classes("w-full"):
                _tile(f"{self._totals.input_tokens:,}", "Total input", color="positive")
                _tile(f"{self._totals.output_tokens:,}", "Total output", color="positive")
        with ui.card().props("flat bordered").classes("w-full"):
            ui.label("Tool calls this turn").classes("text-weight-medium")
            if not self._chips:
                ui.label("No tool calls yet.").classes("text-caption lr-muted").mark(
                    "tool-chips-empty"
                )
            with ui.row().classes("gap-1"):
                for group in group_chips(self._chips):
                    colour = CHIP_COLOURS.get(group.state, "grey")
                    with (
                        ui.badge(color=colour).props("outline").classes("q-pa-xs").mark("tool-chip")
                    ):
                        ui.icon("circle", size="8px").props(f"color={colour}")
                        ui.label(group.name)
                        if group.count > 1:
                            ui.label(f"x {group.count}").classes("text-weight-bold q-ml-xs")
                    # The real error is what you see on hover, not just "failed".
                    ui.tooltip(_chip_tooltip(group)).style(
                        "max-width: 340px; white-space: pre-line"
                    )

    def _render_modules(self) -> None:
        for module in self._modules():
            with ui.card().props("flat bordered").classes("w-full").mark("module-card"):
                with ui.row().classes("w-full items-center no-wrap justify-between"):
                    ui.label(module.name).classes("text-weight-medium")
                    ui.badge("Enabled" if module.enabled else "Disabled").props(
                        f"color={'positive' if module.enabled else 'grey'}"
                    )
                ui.label(module.description).classes("text-caption lr-muted")
