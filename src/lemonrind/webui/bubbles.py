"""Chat message widgets: your messages, and the assistant's (which also stream in live).

Python / NiceGUI ideas used here:

* **Building widgets inside a ``with`` block.** In NiceGUI, widgets created inside ``with some_container:``
  become children of that container. The "current container" is tracked for you, like a stack.
* **Keeping a reference** to a widget (``self._active``, the text being written) lets you change it later
  (``self._active.set_content(...)``). NiceGUI sends only the change to the browser.
* **Throttling.** A model produces tokens faster than a page should repaint, so the streaming bubble
  repaints at most about ten times a second.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime

from nicegui import ui

from lemonrind.attachment_markers import parse_file_attachment, parse_image_marker
from lemonrind.attachments import Attachment, attached_url
from lemonrind.chats import (
    CompactionFinished,
    CompactionStarted,
    ContextUsed,
    Reply,
    StoredMessage,
    ToolFinished,
    ToolStarted,
)
from lemonrind.lemonade import PromptProgress, ReasoningDelta, TextDelta
from lemonrind.lemonade.events import RequestStats, ToolCall

MARKDOWN_EXTRAS = ["fenced-code-blocks", "tables", "break-on-newline"]
REPAINT_INTERVAL = 0.1  # seconds: the fastest a streaming bubble repaints
ARGUMENT_PREVIEW_CHARS = 70
# A picture the assistant drew: the Markdown the Images module puts in a reply (the folder is served at /generated).
GENERATED_IMAGE = re.compile(r"!\[[^\]]*\]\((/generated/[^)\s]+)\)")


def generated_images(text: str) -> list[str]:
    """The addresses of the pictures drawn in a reply, in order, without repeats."""
    return list(dict.fromkeys(GENERATED_IMAGE.findall(text)))


def format_time(moment: datetime) -> str:
    """``14:05`` in the computer's own time zone."""
    return moment.astimezone().strftime("%H:%M")


def format_date(moment: datetime) -> str:
    """``Friday, 2 October 2026`` (used for the dividers between days)."""
    local = moment.astimezone()
    return f"{local:%A}, {local.day} {local:%B %Y}"


def pretty_arguments(arguments: str) -> str:
    """The model's tool arguments indented for reading; the raw text if it is not valid JSON."""
    try:
        return json.dumps(json.loads(arguments), indent=2, ensure_ascii=False)
    except json.JSONDecodeError:
        return arguments


def preview_arguments(arguments: str) -> str:
    """The arguments squeezed onto one short line for the panel header."""
    try:
        data = json.loads(arguments)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict):
        shown = ", ".join(f"{key}: {value!r}" for key, value in data.items())
    else:
        shown = " ".join(arguments.split())
    if len(shown) > ARGUMENT_PREVIEW_CHARS:
        shown = shown[: ARGUMENT_PREVIEW_CHARS - 3] + "..."
    return shown


def add_date_divider(moment: datetime) -> None:
    with ui.row().classes("w-full justify-center"):
        ui.label(format_date(moment)).classes("text-caption lr-muted")


@dataclass(frozen=True, slots=True)
class UserView:
    """What to show for one of your messages: the typed text, plus a picture or a document if one was attached."""

    text: str
    image_url: str | None = None  # a picture that is still on disk
    image_missing: bool = False  # a picture was attached but its stored copy is gone
    file_name: str | None = None
    file_text: str = ""


def user_view(content: str) -> UserView:
    """Read a stored user message's attachment marker (if any) back into something to show."""
    path, typed = parse_image_marker(content)
    if path is not None:
        if path.is_file():
            return UserView(typed, image_url=attached_url(path))
        return UserView(typed, image_missing=True)
    if (parts := parse_file_attachment(content)) is not None:
        return UserView(parts.typed, file_name=parts.name, file_text=parts.text)
    return UserView(content)


def view_for(text: str, attachment: Attachment | None) -> UserView:
    """What to show for a message you are sending now, from what is attached to it."""
    if attachment is None:
        return UserView(text)
    if attachment.is_image and attachment.path is not None:
        return UserView(text, image_url=attached_url(attachment.path))
    return UserView(text, file_name=attachment.name, file_text=attachment.text)


def add_user_message(text: str, created_at: datetime, view: UserView | None = None) -> None:
    """Your message: a blue bubble on the right, with its picture or document above the text when there is one."""
    view = view or UserView(text)
    with ui.chat_message(name="You", stamp=format_time(created_at), sent=True).classes("w-full"):
        with ui.column().classes("gap-1"):
            if view.image_url:
                ui.image(view.image_url).classes("rounded-borders").style("max-width: 16rem")
            elif view.image_missing:
                ui.label("(the attached picture is no longer on disk)").classes("text-caption")
            if view.file_name:
                with (
                    ui.expansion(f"Attached: {view.file_name}", icon="attach_file")
                    .props("dense")
                    .classes("text-caption")
                ):
                    ui.label(
                        view.file_text[:3000] + ("..." if len(view.file_text) > 3000 else "")
                    ).classes("lr-tool-text")
            if view.text:
                ui.label(view.text).classes("whitespace-pre-wrap")


def add_summary_marker(summary: str) -> None:
    """A line in the chat showing where old messages were summarised, with the summary behind it."""
    with (
        ui.expansion("Earlier messages above were summarised to save space (show summary)")
        .props("dense icon=compress")
        .classes("w-full text-caption lr-muted")
    ):
        ui.label(summary).classes("lr-tool-text")


@dataclass(frozen=True, slots=True)
class ToolChip:
    """One tool call as the stats panel shows it: a name and whether it succeeded, failed, or is still going."""

    name: str
    state: str  # "pending", "ok", "failed" or "abandoned"
    detail: str = ""  # the real error text for a failed call, shown when you hover


class ToolPanel:
    """One tool call, shown as a collapsed panel: what was called, then the arguments and the result.

    It starts in a "running" state (a spinner) and ``finish`` switches it to done or failed, which is
    why a panel is created when the call *starts*, not when it ends.
    """

    def __init__(self, call: ToolCall) -> None:
        self.call = call
        self.is_running = True
        self.state = "pending"  # "pending" while it runs, then "ok", "failed" or "abandoned"
        self.detail = ""  # the error text when it failed
        with ui.expansion().props("dense").classes("w-full lr-tool") as self.root:
            with self.root.add_slot("header"):
                with ui.row().classes("items-center no-wrap gap-2"):
                    self._spinner = ui.spinner(size="xs")
                    self._icon = ui.icon("build", size="xs")
                    self._icon.set_visibility(False)
                    self._title = ui.label(f"Using {call.name}...").classes("text-caption")
                    ui.label(preview_arguments(call.arguments)).classes(
                        "text-caption lr-muted ellipsis"
                    )
            ui.label("Arguments").classes("text-caption lr-muted")
            ui.label(pretty_arguments(call.arguments)).classes("lr-tool-text")
            ui.label("Result").classes("text-caption lr-muted q-mt-sm")
            self._result = ui.label("(waiting)").classes("lr-tool-text")

    def finish(self, content: str, *, is_error: bool, seconds: float | None = None) -> None:
        self.is_running = False
        self.state = "failed" if is_error else "ok"
        self.detail = content if is_error else ""
        self._spinner.set_visibility(False)
        self._icon.name = "error_outline" if is_error else "check_circle"
        self._icon.props(f"color={'negative' if is_error else 'positive'}")
        self._icon.set_visibility(True)
        took = f" ({seconds:.1f}s)" if seconds is not None else ""
        self._title.text = f"{self.call.name} {'failed' if is_error else 'done'}{took}"
        self._result.text = content or "(empty)"

    def abandon(self) -> None:
        """The tool never reported back (Stop, or an error): stop the spinner."""
        self.is_running = False
        self.state = "abandoned"
        self._spinner.set_visibility(False)
        self._icon.set_visibility(True)
        self._title.text = f"{self.call.name} did not finish"
        self._result.text = "(no result)"


class AssistantBubble:
    """The assistant's message: thinking panel, tool panels, the answer as Markdown, stats, copy button.

    One class serves both a reply that is streaming in right now (``on_event``) and a reply loaded from the
    database (``show_stored``, called once per stored message; a reply that used tools is several stored
    messages, which all go into the one bubble).
    """

    def __init__(self, *, thinking_open: bool = False) -> None:
        self._reasoning = ""
        self._text = ""
        self._thinking_started: float | None = None
        self._thinking_seconds = 0.0  # total over all requests of the reply (tools mean several)
        self._last_paint = 0.0
        # The reply is a flow of text blocks and tool panels in the order they happened. ``_active`` is the text block
        # being written now (``None`` right after a tool panel, until more text arrives); ``_active_start`` is where
        # it begins inside ``_text``, which holds all the text of the reply (for Copy and for finding pictures).
        self._active: ui.markdown | None = None
        self._active_start = 0
        self._panels: dict[str, ToolPanel] = {}
        self._image_urls: list[str] = []  # the pictures that have Save / Copy buttons so far
        self._repaint_pending = False

        with ui.chat_message(name="Assistant", sent=False).classes("w-full") as self.root:
            with self.root.add_slot("avatar"):
                ui.icon("smart_toy", color="primary", size="2.4rem").classes(
                    "q-message-avatar q-message-avatar--received"
                )

            # Quasar makes every direct child of a chat message its own bubble, so everything goes in ONE column.
            with ui.column().classes("gap-1 w-full"):
                self._recalled_box = ui.column().classes("w-full gap-0")
                self._progress = ui.label().classes("text-caption lr-warn")
                self._progress.set_visibility(False)

                with (
                    ui.expansion(value=thinking_open)
                    .props("dense")
                    .classes("w-full") as self._thinking
                ):
                    with self._thinking.add_slot("header"):
                        with ui.row().classes("items-center no-wrap gap-2"):
                            self._thinking_spinner = ui.spinner(size="xs")
                            self._thinking_label = ui.label("Thinking...").classes(
                                "text-caption lr-muted"
                            )
                    self._thinking_text = ui.label().classes("lr-thinking-text")
                self._thinking.set_visibility(False)

                self._flow = ui.column().classes(
                    "w-full gap-1"
                )  # text and tool panels, in time order
                self._images_box = ui.column().classes(
                    "w-full gap-1"
                )  # Save / Copy buttons for drawn pictures
                self._note = ui.label().classes("text-caption lr-warn")
                self._note.set_visibility(False)

                with ui.row().classes("items-center gap-1 w-full") as self._footer:
                    self._stats_label = ui.label().classes("text-caption lr-muted")
                    ui.space()
                    ui.button(icon="content_copy", on_click=self._copy).props(
                        "flat dense round size=sm"
                    ).tooltip("Copy the answer")
                self._footer.set_visibility(False)

    # --- streaming ---------------------------------------------------------------------------------------

    def on_event(self, event) -> None:
        """Called for every event while the reply streams in."""
        match event:
            case PromptProgress():
                text = event.describe()
                self._progress.text = text or ""
                self._progress.set_visibility(bool(text))
            case ReasoningDelta(text=piece):
                if self._thinking_started is None:  # thinking (again) begins
                    self._thinking_started = time.monotonic()
                    self._thinking_spinner.set_visibility(True)
                    self._thinking_label.text = "Thinking..."
                self._reasoning += piece
                self._thinking.set_visibility(True)
            case TextDelta(text=piece):
                self._progress.set_visibility(False)
                self._end_thinking()
                self._add_text(piece)
            case ContextUsed(text=background):
                self._show_recalled(background)
            case CompactionStarted():
                self._progress.text = "The chat is nearly full: summarising the oldest messages..."
                self._progress.set_visibility(True)
            case CompactionFinished(messages_summarised=count):
                self._progress.set_visibility(False)
                ui.notify(f"Summarised {count} earlier messages to make room.", timeout=3000)
            case ToolStarted(call=call):
                self._progress.set_visibility(False)
                self._end_thinking()
                self._add_panel(call)
                ui.run_javascript("lrScrollDown(false)")
            case ToolFinished(call=call, result=result, seconds=seconds):
                if panel := self._panels.get(call.id):
                    panel.finish(result.content, is_error=result.is_error, seconds=seconds)
        now = time.monotonic()
        if now - self._last_paint >= REPAINT_INTERVAL:
            self._paint()
            ui.run_javascript("lrScrollDown(false)")

    def finish(self, reply: Reply) -> None:
        """The reply is complete: paint the final state, show the stats."""
        self._end_thinking()
        self._progress.set_visibility(False)
        self._paint()
        self._show_stats(reply.stats)
        if not reply.text:
            hint = (
                "The model used up its token limit while thinking, so there is no answer."
                if reply.finish_reason == "length"
                else "The reply was empty."
            )
            self._note_text(hint + " It was not saved.")
        ui.run_javascript("lrScrollDown(false)")

    def repaint_soon(self, delay: float = 0.3) -> None:
        """Make sure the text that has arrived gets drawn within ``delay`` seconds, even if no more events come.

        The streaming draw is throttled, so the last piece of a burst of events could otherwise wait for the next
        one, which for a job busy with a long tool call may be a minute away. Call inside the page's layout.
        """
        if self._repaint_pending:
            return
        self._repaint_pending = True

        def flush() -> None:
            self._repaint_pending = False
            self._paint()

        ui.timer(delay, flush, once=True)

    def repaint(self) -> None:
        """Draw what has arrived so far now (the streaming draw is throttled, so a batch of replayed events needs this)."""
        self._paint()

    def show_progress(self, text: str) -> None:
        """A line of progress under the (empty) reply, for work that has no text to stream, such as drawing."""
        self._progress.text = text
        self._progress.set_visibility(True)

    def show_text(self, text: str) -> None:
        """Fill the reply with finished text (a drawn picture) instead of streaming it."""
        self._progress.set_visibility(False)
        self._end_text()
        self._add_text(text)
        self._paint()
        self._show_stats(None)
        ui.run_javascript("lrScrollDown(false)")

    def mark_stopped(self) -> None:
        self._wrap_up()
        self._note_text(
            "Stopped." if self._text else "Stopped before any answer arrived. Nothing was saved."
        )

    def show_error(self, message: str) -> None:
        self._wrap_up()
        self._note_text(message)

    def _wrap_up(self) -> None:
        self._end_thinking()
        self._progress.set_visibility(False)
        for panel in self._panels.values():
            if panel.is_running:
                panel.abandon()
        self._paint()

    # --- stored replies ------------------------------------------------------------------------------------

    def show_stored(self, message: StoredMessage) -> None:
        """Add one stored assistant message to this bubble (call again for the next one of the same reply)."""
        self._end_text()  # each stored message starts a new block
        if message.content:
            self._add_text(message.content)
        if message.reasoning:
            self._reasoning += ("\n\n" if self._reasoning else "") + message.reasoning
            self._thinking.set_visibility(True)
            self._thinking_spinner.set_visibility(False)
            self._thinking_label.text = f"Thought ({len(self._reasoning.split()):,} words)"
        self._paint()  # the text of this message, before the panels of the tools it called
        for call in message.tool_calls:
            self._add_panel(call)
        self._show_stats(message.stats)

    def tool_chips(self) -> list[ToolChip]:
        """The tool calls of this reply and how each ended, for the stats panel."""
        return [ToolChip(p.call.name, p.state, p.detail) for p in self._panels.values()]

    def show_stored_tool_result(self, message: StoredMessage) -> None:
        """Fill in a tool panel from the stored tool message that answered it."""
        if panel := self._panels.get(message.tool_call_id or ""):
            panel.finish(message.content, is_error=message.tool_failed)

    # --- internals ---------------------------------------------------------------------------------------

    def _show_recalled(self, text: str) -> None:
        """A small panel at the top of the reply: what the modules (memory) told the model about you."""
        with self._recalled_box:
            with (
                ui.expansion("Background added to your message")
                .props("dense icon=psychology")
                .classes("w-full text-caption lr-muted")
            ):
                ui.label(text).classes("lr-tool-text")

    def _add_text(self, piece: str) -> None:
        """Add text to the reply: to the block being written, or to a new block below the last tool panel."""
        if self._active is None:
            if self._text:
                self._text += "\n\n"  # (in the Copy text, separate blocks are separate paragraphs)
            self._active_start = len(self._text)
            with self._flow:
                self._active = ui.markdown(extras=MARKDOWN_EXTRAS).classes("lr-markdown")
        self._text += piece

    def _end_text(self) -> None:
        """The block being written is complete (a tool panel comes next): draw it for the last time."""
        if self._active is None:
            return
        shown = self._text[self._active_start :]
        if shown.strip():
            self._active.set_content(shown)
        else:
            self._active.delete()  # a block of nothing but blank lines would only leave a gap
        self._active = None

    def _add_panel(self, call: ToolCall) -> None:
        self._end_text()
        with self._flow:
            self._panels[call.id] = ToolPanel(call)

    def _paint(self) -> None:
        self._last_paint = time.monotonic()
        if self._active is not None:
            self._active.set_content(self._text[self._active_start :])
        self._thinking_text.text = self._reasoning
        self._update_image_buttons()

    def _update_image_buttons(self) -> None:
        """A Save and a Copy button under each drawn picture (rebuilt only when the set of pictures changes)."""
        urls = generated_images(self._text)
        if urls == self._image_urls:
            return
        self._image_urls = urls
        self._images_box.clear()
        with self._images_box:
            for number, url in enumerate(urls, start=1):
                label = f"Picture {number}: " if len(urls) > 1 else ""
                with ui.row().classes("items-center gap-1"):
                    if label:
                        ui.label(label).classes("text-caption lr-muted")
                    ui.button(
                        "Save image", icon="download", on_click=lambda u=url: ui.download(u)
                    ).props("flat dense no-caps size=sm").mark("save-image")
                    ui.button(
                        "Copy image", icon="content_copy", on_click=lambda u=url: copy_image(u)
                    ).props("flat dense no-caps size=sm").mark("copy-image")

    def _end_thinking(self) -> None:
        if self._thinking_started is not None:
            self._thinking_seconds += time.monotonic() - self._thinking_started
            self._thinking_started = None
            words = len(self._reasoning.split())
            self._thinking_label.text = (
                f"Thought for {self._thinking_seconds:.1f}s ({words:,} words)"
            )
            self._thinking_spinner.set_visibility(False)

    def _show_stats(self, stats: RequestStats | None) -> None:
        if stats is not None:
            self._stats_label.text = stats.summary()
        self._footer.set_visibility(bool(self._text))

    def _note_text(self, text: str) -> None:
        self._note.text = text
        self._note.set_visibility(True)

    async def _copy(self) -> None:
        ui.clipboard.write(self._text)
        ui.notify("Copied", type="positive", timeout=1000)


async def copy_image(url: str) -> None:
    """Put a drawn picture on the clipboard. The browser only allows this on a focused page that is https or localhost."""
    copied = await ui.run_javascript(
        "(async () => { try {"
        f" const blob = await (await fetch({json.dumps(url)})).blob();"
        " await navigator.clipboard.write([new ClipboardItem({[blob.type]: blob})]);"
        " return true; } catch (e) { return false; } })()"
    )
    if copied:
        ui.notify("Picture copied", type="positive", timeout=1500)
    else:
        ui.notify(
            "The browser did not allow copying the picture. Use Save image instead.", type="warning"
        )
