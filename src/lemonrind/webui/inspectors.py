"""The read-only "view" dialogs at the bottom of the right-hand panel.

Each answers one "what is the model actually being given?" question, which is the first thing to check when an answer
looks wrong:

* **System prompt**: the persona, behaviour, style, pinned memories and summary sent before every request.
* **Turn context**: a live preview of what would be added to the message you are typing (the date and time, recalled memories,
  matching passages from the attached knowledge base). That text is added to one message only, so it is not part of the system prompt.
* **Tools sent**: the tool descriptions the model is offered with each request. Every tool costs tokens, which is a good reason
  to switch off modules you are not using.
* **Logs**: the tail of the app's log file, where quietly handled problems are written.

They only show things; none of them changes anything.

Python ideas used here:

* Reading the *end* of a file without reading all of it: ``seek`` to near the end, read the rest, drop the partial first line.
* ``json.dumps(..., indent=2)`` to show a tool's parameter schema.
"""

from __future__ import annotations

import asyncio
import html
import json
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from nicegui import ui

from lemonrind.chats import Conversation
from lemonrind.lemonade.logstream import LEVELS, LogEntry, filter_entries, stream_logs
from lemonrind.logging_setup import clear_logs, log_path
from lemonrind.webui.context import AppContext
from lemonrind.webui.dialogs import ask_confirm

TAIL_BYTES = 256_000  # enough for a few thousand lines; the log file is capped at about 1 MB anyway
DEFAULT_LINES = 300


def read_tail(path: Path, lines: int = DEFAULT_LINES) -> str:
    """The last ``lines`` lines of a text file (empty if it does not exist)."""
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - TAIL_BYTES))
            data = handle.read()
            started_midway = size > TAIL_BYTES
    except OSError:
        return ""
    text = data.decode("utf-8", errors="replace")
    all_lines = text.splitlines()
    if started_midway and all_lines:
        all_lines = all_lines[1:]  # the first line may have been cut in half
    return "\n".join(deque(all_lines, maxlen=lines))


def filter_problems(text: str) -> str:
    """Only the lines at WARNING level or above (and their continuation lines, such as a traceback)."""
    kept: list[str] = []
    keeping = False
    for line in text.splitlines():
        parts = line.split(" ", 3)
        if (
            len(parts) >= 3 and parts[2].isalpha() and parts[2].isupper()
        ):  # "date time LEVEL name: message"
            keeping = parts[2] in {"WARNING", "ERROR", "CRITICAL"}
        if keeping:
            kept.append(line)
    return "\n".join(kept)


def make_dialog(title: str, subtitle: str, *, wide: bool = False):
    """A dialog with a title and a caption on top, a scrolling body, and a Close button that is always in view.

    Returns the dialog and the body to fill (use ``with body:``). The card is a column that cannot grow past the
    window; only the body scrolls, so a long list never pushes the Close button off the screen. The dialog
    removes itself from the page when it closes, so opening a screen repeatedly does not pile up hidden ones.
    """
    dialog = ui.dialog()
    width = "w-[64rem]" if wide else "w-[48rem]"
    with (
        dialog,
        ui.card().classes(f"{width} max-w-full max-h-[90vh] no-wrap gap-0 q-pa-none"),
    ):
        with ui.column().classes("w-full gap-0 q-px-md q-pt-md q-pb-sm shrink-0"):
            ui.label(title).classes("text-h6")
            if subtitle:
                ui.label(subtitle).classes("text-caption lr-muted")
        ui.separator()
        body = ui.column().classes("w-full grow overflow-auto q-pa-md gap-2").style("min-height: 0")
        ui.separator()
        with ui.row().classes("w-full justify-end q-pa-sm shrink-0"):
            ui.button("Close", on_click=dialog.close).props("flat").mark("dialog-close")
    dialog.on("hide", dialog.delete)
    return dialog, body


def _tokens(text: str) -> int:
    return max(1, len(text) // 4)  # a rough rule of thumb: about four characters to a token


async def show_system_prompt(conversation: Conversation) -> None:
    text = await conversation.current_system_prompt()
    dialog, card = make_dialog(
        "System prompt",
        "What the model is told at the start of every request in this chat "
        f"(about {_tokens(text):,} tokens).",
    )
    with card:
        ui.label(text).classes("lr-tool-text").style("max-height: 60vh")
    dialog.open()


async def show_turn_context(conversation: Conversation, typing: str = "") -> None:
    """A preview of what would be added to a message if it were sent right now."""
    preview = await conversation.preview_turn_context(typing)
    dialog, card = make_dialog(
        "Turn context (preview)",
        "The text added to a message, beside what you type: the current date and time, and anything the modules add "
        "(recalled memories, matching passages from the attached knowledge base). It belongs to that one message only "
        "and is never saved in the chat.",
    )
    with card:
        if typing.strip():
            ui.label("Worked out for the message you are typing now.").classes(
                "text-caption lr-muted"
            )
        else:
            ui.label(
                "Type a message first to see which memories and knowledge-base passages it would bring in."
            ).classes("text-caption lr-muted")
        if preview:
            ui.label(f"About {_tokens(preview):,} tokens.").classes("text-caption lr-muted")
            ui.label(preview).classes("lr-tool-text").style("max-height: 40vh").mark("turn-preview")
        else:
            ui.label("Nothing would be added.").classes("lr-muted")
    dialog.open()


def show_tools_sent(conversation: Conversation) -> None:
    schemas = conversation.tools.schemas() if conversation.tools is not None else []
    size = _tokens(json.dumps(schemas))
    dialog, card = make_dialog(
        "Tools sent",
        "The tools the model is offered with each request, from the modules that are switched on"
        + (f" ({len(schemas)} tools, about {size:,} tokens)." if schemas else "."),
    )
    with card:
        if not schemas:
            ui.label(
                "No tools are being sent (every module that offers tools is switched off)."
            ).classes("lr-muted")
        for schema in schemas:
            function = schema.get("function", {})
            with ui.expansion(function.get("name", "?")).classes("w-full").props("dense"):
                ui.label(function.get("description", "")).classes("text-caption")
                ui.label(json.dumps(function.get("parameters", {}), indent=2)).classes(
                    "lr-tool-text"
                )
    dialog.open()


def _app_log_panel(data_dir: Path) -> None:
    """The end of the app's own log file, with a refresh button and a warnings-only switch."""
    path = log_path(data_dir)
    ui.label(f"The end of {path}. Problems the app handles quietly are written here.").classes(
        "text-caption lr-muted"
    )

    @ui.refreshable
    def body() -> None:
        text = read_tail(path)
        if only_problems.value:
            text = filter_problems(text)
        ui.label(text or "(nothing logged yet)").classes("lr-tool-text").style(
            "max-height: 50vh"
        ).mark("log-text")

    async def clear() -> None:
        if await ask_confirm(
            "Clear the log",
            "Delete everything in the app's log file and its older copies? Lemonade's own log is not affected.",
            ok_text="Clear",
        ):
            clear_logs(data_dir)
            body.refresh()

    only_problems = ui.switch("Warnings and errors only", on_change=body.refresh)
    body()
    with ui.row().classes("gap-2"):
        ui.button("Refresh", icon="refresh", on_click=body.refresh).props("flat").mark(
            "log-refresh"
        )
        ui.button("Clear log file", icon="delete_sweep", on_click=clear).props("flat").mark(
            "log-clear-file"
        )


SHOWN_LINES = 400  # the screen shows at most this many (filtered) lines, so a huge log stays fast
BUFFER_LINES = 5000  # how many entries are kept while the screen is open


def render_log_html(entries: Sequence[LogEntry]) -> str:
    """The entries as an HTML table made of rows and columns (Time, Level, Source, Message), all text escaped.

    The columns are laid out by CSS (``lr-log-*`` in styles.py); a thin date row appears wherever the day changes, so the
    time column can stay short.
    """
    rows = [
        '<div class="lr-log-head"><span>Time</span><span>Level</span><span>Source</span><span>Message</span></div>'
    ]
    day = None
    for entry in entries[-SHOWN_LINES:]:
        if entry.day and entry.day != day:
            day = entry.day
            rows.append(f'<div class="lr-log-day">{html.escape(day)}</div>')
        level = entry.severity.strip().lower() or "info"
        rows.append(
            f'<div class="lr-log-row lr-log-row-{html.escape(level)}">'
            f'<span class="lr-log-time">{html.escape(entry.clock)}</span>'
            f'<span class="lr-log-lvl lr-log-lvl-{html.escape(level)}">{html.escape(entry.severity.upper())}</span>'
            f'<span class="lr-log-tag" title="{html.escape(entry.tag)}">{html.escape(entry.tag)}</span>'
            f'<span class="lr-log-msg">{html.escape(entry.message)}</span>'
            "</div>"
        )
    return "".join(rows)


@dataclass(slots=True)
class _LogState:
    dirty: bool = True  # something changed since the screen was last drawn
    task: asyncio.Task | None = None  # the background task listening to Lemonade


def _lemonade_log_panel(context: AppContext, dialog: ui.dialog) -> None:
    """Lemonade's own live server log, streamed over its WebSocket, with level and text filters, pause and copy."""
    entries: deque[LogEntry] = deque(maxlen=BUFFER_LINES)
    state = _LogState()

    status = (
        ui.label("Connecting to Lemonade...")
        .classes("text-caption lr-muted")
        .mark("lemonade-log-status")
    )
    with ui.row().classes("w-full items-center gap-2 no-wrap"):
        level = (
            ui.select(list(LEVELS), value="All", label="Show")
            .props("dense outlined")
            .classes("w-48")
        )
        search = (
            ui.input(placeholder="Filter text")
            .props("dense outlined clearable")
            .classes("flex-grow")
        )
        paused = ui.switch("Pause")
    box = (
        ui.html("", sanitize=False)
        .classes("lr-log")
        .style("max-height: 55vh; min-height: 12rem; overflow: auto")
        .mark("lemonade-log")
    )
    box_id = f"c{box.id}"

    def shown() -> list[LogEntry]:
        return filter_entries(entries, level=level.value, text=search.value or "")

    def redraw() -> None:
        box.set_content(render_log_html(shown()) or "(nothing to show)")
        ui.run_javascript(
            f"const e=getElementById({box_id!r}); if (e) e.scrollTop = e.scrollHeight"
        )
        state.dirty = False

    def on_snapshot(batch: list[LogEntry]) -> None:
        entries.extend(batch)
        state.dirty = True

    def on_entry(entry: LogEntry) -> None:
        entries.append(entry)
        state.dirty = True

    async def run() -> None:
        try:
            health = await context.client.health()
            if not health.websocket_port:
                status.text = "Lemonade did not report a log port, so its log cannot be streamed."
                return
            status.text = "Streaming Lemonade's log"
            lemonade = context.settings.lemonade
            await stream_logs(
                lemonade.base_url, health.websocket_port, lemonade.api_key, on_snapshot, on_entry
            )
            status.text = "The log connection closed."
        except asyncio.CancelledError:
            raise
        except (
            Exception
        ) as error:  # a refused connection, a dropped network: tell the user, never crash the page
            status.text = f"Disconnected: {type(error).__name__}: {error}"

    def start() -> None:
        old = state.task
        if old is not None and not old.done():
            old.cancel()
        entries.clear()
        state.dirty = True
        status.text = "Connecting to Lemonade..."
        state.task = asyncio.create_task(run())

    async def copy() -> None:
        text = "\n".join(entry.render() for entry in shown())
        ui.clipboard.write(text)
        ui.notify(f"Copied {len(text.splitlines()):,} lines", timeout=1500)

    def clear() -> None:
        entries.clear()
        state.dirty = True

    with ui.row().classes("gap-2"):
        ui.button("Copy", icon="content_copy", on_click=copy).props("flat dense").mark("log-copy")
        ui.button("Clear", icon="delete_sweep", on_click=clear).props("flat dense")
        ui.button("Reconnect", icon="refresh", on_click=start).props("flat dense")

    def tick() -> None:
        if state.dirty and not paused.value:
            redraw()

    def mark_dirty(_: object = None) -> None:
        state.dirty = True

    level.on_value_change(mark_dirty)
    search.on_value_change(mark_dirty)
    ui.timer(0.7, tick)
    dialog.on(
        "hide", lambda: state.task and state.task.cancel()
    )  # stop listening when the screen closes
    start()


def show_logs(context: AppContext) -> None:
    """The logs screen: Lemonade's live server log on one tab, this app's own log file on the other."""
    dialog, card = make_dialog("Logs", "", wide=True)
    with card:
        with ui.tabs().props("dense align=left") as tabs:
            lemonade_tab = ui.tab("Lemonade server")
            app_tab = ui.tab("This app")
        with ui.tab_panels(tabs, value=lemonade_tab).classes("w-full").props("animated=false"):
            with ui.tab_panel(lemonade_tab).classes("q-pa-none gap-2"):
                _lemonade_log_panel(context, dialog)
            with ui.tab_panel(app_tab).classes("q-pa-none gap-2"):
                _app_log_panel(context.settings_file.parent)
    dialog.open()
