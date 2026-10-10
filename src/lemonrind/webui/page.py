"""The chat page: header, chat list, messages, stats panel and the message box.

How NiceGUI works, in short:

* You describe the page in Python by creating widgets (``ui.label``, ``ui.button``...). NiceGUI turns that
  into a web page, and keeps a live connection to the browser. When Python changes a widget
  (``label.text = "..."``) the browser updates by itself; when the user clicks, your Python handler runs.
* Every browser tab gets its **own page**: ``index`` runs once per tab, so each tab has its own
  ``ChatPage`` object, its own conversation and its own state. (Blazor Server works the same way.)
* Handlers may be ``async``. While one handler is waiting (for the model, say) the app keeps serving
  every other tab and button.

Python ideas used here: a class that owns a page's state; ``asyncio.create_task`` and task cancellation (the
Stop button); ``try / except / else / finally``.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import time
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal

from nicegui import Client, ui

from lemonrind.attachments import (
    ACCEPT,
    Attachment,
    AttachmentError,
    attached_url,
    attachments_dir,
    prepare,
)
from lemonrind.chats import (
    CONTINUE_PROMPT,
    ChatSession,
    Conversation,
    Reply,
    StoredMessage,
    ToolFinished,
    ToolStarted,
    build_system_prompt,
    format_duration,
)
from lemonrind.lemonade.client import LemonadeError
from lemonrind.lemonade.events import RequestStats
from lemonrind.lemonade.models import ModelInfo
from lemonrind.lemonade.selection import (
    ModelSelectionError,
    default_is_missing,
    pick_chat_model,
)
from lemonrind.modules import ImagesModule, KnowledgeModule, McpModule, SchedulerModule
from lemonrind.modules.knowledge import OPTION_KEY as KNOWLEDGE_OPTION
from lemonrind.modules.tool import ToolError
from lemonrind.webui.bubbles import (
    AssistantBubble,
    add_date_divider,
    add_summary_marker,
    add_user_message,
    user_view,
    view_for,
)
from lemonrind.webui.context import AppContext, get_context
from lemonrind.webui.dialogs import ask_confirm, ask_image_size, ask_tool_approval
from lemonrind.webui.inspectors import (
    show_logs,
    show_system_prompt,
    show_tools_sent,
    show_turn_context,
)
from lemonrind.webui.model_ui import (
    GroupedSelect,
    add_group_styling,
    fill_model_select,
    is_header,
    show_context_breakdown,
    show_model_details,
)
from lemonrind.webui.settings_dialog import open_settings
from lemonrind.webui.settings_sections import ModelChoices
from lemonrind.webui.sidebar import ChatList
from lemonrind.webui.stats import SessionTotals, StatsPanel
from lemonrind.webui.styles import CSS, JS
from lemonrind.webui.usage_dialog import show_usage

HEALTH_POLL_SECONDS = 20

# theme setting -> (what NiceGUI's dark mode wants, icon for the toolbar button, next theme in the cycle)
THEMES: dict[str, tuple[bool | None, str, str]] = {
    "system": (None, "brightness_auto", "light"),
    "light": (False, "light_mode", "dark"),
    "dark": (True, "dark_mode", "system"),
}


class ChatPage:
    def __init__(self, context: AppContext, client: Client) -> None:
        self.context = context
        self.client = client
        self.conversation = Conversation(
            client=context.client,
            repo=context.repo,
            system_prompt=build_system_prompt(context.settings),
            model="",
            tools=context.modules,
            context=context.modules,
            max_rounds=context.settings.modules.max_tool_rounds,
            compaction_percent=context.settings.assistant.compaction_percent,
            compaction_keep_turns=context.settings.assistant.compaction_keep_turns,
        )
        self.conversation.time_awareness = True  # the model is always told the real date and time
        self.stats_panel = StatsPanel(
            modules=lambda: context.modules.modules if context.modules else (),
            on_tab_change=self._stats_tab_changed,
        )
        self.totals = SessionTotals()
        self._last_stats: RequestStats | None = None
        self._context_tokens: int | None = None
        self._context_window: int | None = None
        self._chat_models: list[str] = []
        self._models: list[
            ModelInfo
        ] = []  # every downloaded model, for the picker and the details popup
        self._model_choices = ModelChoices()  # for the drop-downs in Settings
        self._context_windows: dict[str, int] = {}
        self._healthy = False
        self._busy = False
        self._picker_followed = (
            False  # the picker is showing a watched scheduled run's model (read-only)
        )
        self._model_told: set[str] = (
            set()
        )  # model problems already shown on this page: each is said once
        self._task: asyncio.Task | None = None
        self._continuable: AssistantBubble | None = (
            None  # the newest reply that has a Continue button
        )
        self._last_day: date | None = None
        self._ignore_model_change = False
        # An image model picked in the model list: what you type is then drawn, not chatted about. The chat model
        # stays as it was, so picking a chat model again carries on where the chat left off.
        self._image_model: str | None = None
        self._ignore_kb_change = False
        self._pending: Attachment | None = (
            None  # a picture or document waiting to go with the next message
        )
        self._allowed_tools: set[str] = set()  # tools the user allowed for the rest of this chat
        scheduler = self._module(SchedulerModule)
        self._runs_seen = scheduler.runs_finished if scheduler else 0
        self._starts_seen = scheduler.runs_started if scheduler else 0
        self._stop_following: Callable[[], None] | None = (
            None  # set while watching a running job's chat
        )
        self.conversation.approve = self._approve_tool

    # --- building the page -------------------------------------------------------------------------------

    async def build(self) -> None:
        ui.add_css(CSS)
        ui.add_head_html(JS)
        ui.colors(primary="#2F6BFF")
        self.dark = ui.dark_mode(THEMES[self.context.settings.ui.theme][0])

        self._build_header()
        self._build_drawers()
        self.messages = ui.column().classes("lr-column gap-3 q-pa-md")
        with self.messages:
            self._empty_hint()
        self._build_footer()
        ui.timer(HEALTH_POLL_SECONDS, self._poll_health)
        ui.timer(5, self._check_scheduled_runs)
        ui.timer(1.5, self._update_job_note)

        await (
            self.client.connected()
        )  # wait until the browser is really there before doing slow work
        await self._startup()

    def _build_header(self) -> None:
        with ui.header().props("bordered").classes("items-center gap-2 q-px-sm"):
            ui.button(icon="menu", on_click=lambda: self.left.toggle()).props("flat dense round")
            self.model_select = (
                GroupedSelect({}, on_change=self._model_selected)
                .props("dense outlined options-dense behavior=menu")
                .classes("w-64 gt-xs")
                .mark("model-select")
            )
            add_group_styling(self.model_select)
            self.model_info_button = ui.button(icon="info_outline", on_click=self._show_model_info)
            self.model_info_button.props("flat dense round").classes("gt-xs").mark("model-info")
            self.model_info_button.tooltip("Model details")
            with ui.row().classes("items-center no-wrap gap-1") as self.loading_row:
                ui.spinner(size="sm")
                self.loading_label = ui.label().classes("text-caption")
            # One line that is cut off with "..." if it is long (the whole text is in the tooltip), and left out on a
            # narrow window, where it would push the header's buttons onto a second row.
            self.job_note = ui.label().classes("text-caption lr-muted gt-sm lr-one-line")
            self.job_note.style("max-width: 28rem").mark("job-running-note")
            self.job_note_tip = self.job_note.tooltip("")
            self.job_note.set_visibility(False)
            self.loading_row.set_visibility(False)
            self.title = (
                ui.label("Lemon Rind")
                .classes("text-subtitle1 text-weight-medium ellipsis flex-grow text-center q-mx-sm")
                .style(
                    "min-width: 0"
                )  # lets a long title shrink to an ellipsis instead of pushing the buttons out
            )

            self.health_button = ui.button(on_click=self._poll_health).props("flat dense no-caps")
            with self.health_button:
                self.health_dot = ui.icon("circle", size="xs")
                self.health_text = ui.label().classes("gt-sm q-ml-xs")
            self._show_health(False)

            self.theme_button = ui.button(on_click=self._cycle_theme).props("flat dense round")
            self.theme_button.tooltip("Light / dark / follow the system").mark("theme")
            self._apply_theme_icon()
            ui.button(icon="settings", on_click=lambda: self._open_settings()).props(
                "flat dense round"
            ).tooltip("Settings").mark("settings")
            ui.button(icon="insights", on_click=lambda: self.right.toggle()).props(
                "flat dense round"
            ).tooltip("Stats / Modules")

    def _build_drawers(self) -> None:
        # value=None means "open only when the window is wide enough"; on a phone both start closed and the
        # header buttons open them over the page. (value=True would force them open, overlapping each other.)
        ui_settings = self.context.settings.ui
        self.left = ui.left_drawer(value=None, bordered=True).props(
            f"width={ui_settings.left_panel_width} breakpoint=900"
        )
        with self.left:
            self.chat_list = ChatList(
                self.context,
                current_id=lambda: (
                    self.conversation.session.id if self.conversation.session else None
                ),
                on_new=self._new_chat,
                on_open=self._open_chat,
                on_changed=self._chat_changed,
                on_deleted=self._chat_deleted,
            )
            self.chat_list.build()
        self.right = ui.right_drawer(value=None, bordered=True).props(
            f"width={ui_settings.right_panel_width} breakpoint=1200"
        )
        ui.on(
            "lr_resize", self._panel_resized, throttle=0.05
        )  # sent by the drag handles (see styles.py)
        with self.right:
            # The numbers at the top, the "what is the model given?" buttons pinned to the bottom of the panel.
            with (
                ui.column()
                # self-start/shrink-0: be as tall as the content, not squeezed to the drawer's height, so the sticky
                # Stats / Modules switch at the top stays in view for the whole length of a long list
                .classes("w-full no-wrap gap-0 self-start shrink-0")
                .style("min-height: calc(100vh - 15rem)")
            ):
                self.stats_panel.build()
                ui.space()
                ui.separator()
                with ui.column().classes("w-full gap-0 q-pt-xs items-start") as self.view_buttons:
                    for label, icon, handler, marker in (
                        ("View system prompt", "article", self._view_prompt, "view-prompt"),
                        (
                            "View turn context",
                            "playlist_add",
                            self._view_turn_context,
                            "view-turn-context",
                        ),
                        ("View tools sent", "construction", self._view_tools, "view-tools"),
                        ("View logs", "receipt_long", self._view_logs, "view-logs"),
                        ("View usage", "bar_chart", self._view_usage, "view-usage"),
                    ):
                        ui.button(label, icon=icon, on_click=handler).props(
                            "flat dense no-caps"
                        ).mark(marker)
                with ui.column().classes("w-full gap-0 q-pt-xs items-start") as self.manage_buttons:
                    ui.button(
                        "Manage in Settings",
                        icon="settings",
                        on_click=lambda: self._open_settings("modules"),
                    ).props("flat dense no-caps").mark("manage-modules")
                self.manage_buttons.set_visibility(False)

    def _panel_resized(self, event) -> None:
        """A drag handle moved: set that panel's width (kept within sensible limits) and remember it when the drag ends."""
        side, width = event.args["side"], int(event.args["width"])
        ui_settings = self.context.settings.ui
        if side == "left":
            ui_settings.left_panel_width = width = max(200, min(560, width))
            self.left.props(f"width={width}")
        else:
            ui_settings.right_panel_width = width = max(220, min(520, width))
            self.right.props(f"width={width}")
        if event.args.get("done"):
            self.context.save_settings()

    def _stats_tab_changed(self, tab: str) -> None:
        """The Modules tab swaps the "view" buttons for a way into Settings."""
        self.view_buttons.set_visibility(tab == "stats")
        self.manage_buttons.set_visibility(tab == "modules")

    def _build_footer(self) -> None:
        # One centred column holds everything (the footer itself lays its children out in a row).
        with (
            ui.footer().classes("items-end justify-center"),
            ui.column().classes("lr-column no-wrap gap-0"),
        ):
            # What is attached to the next message: a small picture or a file chip, with a way to remove it.
            self.attach_row = ui.row().classes("w-full items-center no-wrap gap-2 q-px-md q-pt-xs")
            self.attach_row.set_visibility(False)
            self.attach_note = ui.label().classes("w-full text-caption text-negative q-px-md")
            self.attach_note.set_visibility(False)
            with ui.row().classes("w-full items-end no-wrap q-px-sm q-pt-xs"):
                self.upload = (
                    ui.upload(
                        on_multi_upload=self._on_upload,
                        auto_upload=True,
                        multiple=False,
                        max_file_size=25_000_000,
                        on_rejected=lambda: self._show_attach_error(
                            "That file is too large (the limit is 25 MB)."
                        ),
                    )
                    .props(f"accept={ACCEPT}")
                    .classes("hidden")
                )
                self.attach_button = ui.button(
                    icon="attach_file", on_click=lambda: self.upload.run_method("pickFiles")
                )
                self.attach_button.props("flat round dense").tooltip(
                    "Attach a picture or a document"
                ).mark("attach")
                # Which knowledge base answers this chat (as in the other editions); hidden while that module is off.
                self.kb_select = (
                    ui.select({"": "No knowledge base"}, value="", on_change=self._kb_selected)
                    .props("dense outlined options-dense")
                    .classes("w-48 gt-xs")
                    .mark("kb-select")
                )
                self.kb_select.tooltip("Knowledge base for this chat")
                self.kb_select.set_visibility(False)
                self.input = (
                    ui.textarea(placeholder="Type a message...")
                    .props("outlined dense autogrow input-style='max-height: 12rem'")
                    .classes("flex-grow")
                    .mark("message-input")
                )
                # Enter sends; Shift+Enter (and the other modifier keys) still insert a new line.
                self.input.on("keydown.enter.exact.prevent", self._send_clicked)
                self.send_button = ui.button(icon="arrow_upward", on_click=self._send_clicked)
                self.send_button.props("round color=primary").mark("send")
                self.send_button.tooltip("Send (Enter)")
                # How full the model's memory is: a ring beside the Send button with the percentage inside. Hover
                # for the exact numbers; click for the full breakdown.
                self.context_ring = ui.circular_progress(
                    value=0, min=0, max=100, size="2.6rem", show_value=False
                )
                self.context_ring.props("thickness=0.18 track-color=grey-4 instant-feedback")
                self.context_ring.classes("cursor-pointer self-center")
                self.context_ring.on("click", self._show_context_breakdown)
                self.context_ring.mark("context-meter")
                with self.context_ring:
                    self.context_percent = ui.label("").classes("lr-ring-text")
                    self.context_tip = ui.tooltip("").mark("context-tip")
                self.context_ring.set_visibility(False)

    def _empty_hint(self, *, running: bool = False) -> None:
        self.hint = ui.column().classes("w-full items-center q-pa-xl gap-1")
        with self.hint:
            if running:
                ui.spinner(size="xl")
                ui.label("This scheduled job is still running").classes("text-subtitle1")
                ui.label(
                    "Its answer appears here when it finishes. You can carry on with other chats meanwhile."
                ).classes("text-caption lr-muted")
                return
            ui.icon("chat_bubble_outline", size="xl").classes("lr-muted")
            ui.label("Start a conversation").classes("text-subtitle1")
            ui.label("Your chat is saved when the first reply arrives.").classes(
                "text-caption lr-muted"
            )

    # --- start-up: model, health, the latest chat ------------------------------------------------------------

    async def _startup(self) -> None:
        self.context.repo.prune_empty_folders()  # folders left empty by an earlier version or a crash
        self.chat_list.render.refresh()
        await self._refresh_lemonade(announce_problems=True)
        # (a scheduled run that is still going is skipped: it is empty until it finishes)
        if latest := next(
            (s for s in self.context.repo.list_sessions() if "running" not in s.tags), None
        ):
            self._open_chat(latest)
        self._refresh_kb_select()  # also when there is no chat to open yet

    async def _refresh_lemonade(self, *, announce_problems: bool) -> None:
        """Ask Lemonade who it is, pick the model, and load it if needed."""
        client = self.context.client
        try:
            health = await client.health()
            models = await client.list_models()
        except LemonadeError as error:
            if self._gone():
                return
            self._show_health(False)
            if announce_problems:
                ui.notify(str(error), type="negative", multi_line=True)
            return
        if (
            self._gone()
        ):  # the tab was closed or reloaded while Lemonade was answering: nothing left to update
            return
        self._show_health(health.is_ok)

        self._models = list(models)
        self._chat_models = [m.id for m in models if m.category == "chat"]
        self._model_choices = ModelChoices(
            chat=self._chat_models,
            embedding=[m.id for m in models if m.category == "embedding"],
            image=[m.id for m in models if m.category == "image"],
        )
        self._context_windows = {m.id: m.context_window for m in models if m.context_window}
        if self.conversation.model:  # already chosen on an earlier check: just keep the list fresh
            self._set_model_options(self._picked())
            return

        default = self.context.settings.lemonade.chat_model
        try:
            model = pick_chat_model("", health, models, default=default)
        except ModelSelectionError as error:
            # This check runs again at every health poll while no model is chosen, so say it once, not every time.
            self._tell_once(str(error), type="negative")
            self._set_model_options("")
            return
        if default_is_missing(default, models):
            self._tell_once(
                f"Your default model '{default}' is no longer on this Lemonade, so {model} is being used. Pick "
                "another from the model list, or change the default in Settings > Lemonade.",
                type="warning",
            )
        self.conversation.model = model
        self._set_model_options(model)
        if model not in {m.model_name for m in health.all_models_loaded}:
            await self._load_model(model)
        self._update_context_label()

    def _tell_once(self, message: str, *, type: Literal["negative", "warning"]) -> None:
        """Show a notice unless this page has already shown exactly this one."""
        if message in self._model_told:
            return
        self._model_told.add(message)
        ui.notify(message, type=type, multi_line=True, timeout=15000)

    def _gone(self) -> bool:
        """Has this page been closed? Updating a widget of a closed page makes NiceGUI log an error."""
        return self.model_select.is_deleted or self.client.id not in Client.instances

    def _set_model_options(self, current: str) -> None:
        if self._gone():
            return
        # set_options(..., value=) would fire on_change; the flag tells _model_selected to ignore it.
        self._ignore_model_change = True
        fill_model_select(self.model_select, self._models, current)
        self._ignore_model_change = False

    async def _load_model(self, name: str) -> None:
        """Load a model with an honest stopwatch: Lemonade reports no load progress, so we show elapsed seconds."""
        started = time.monotonic()
        self.loading_row.set_visibility(True)
        load = asyncio.create_task(self.context.client.load_model(name))
        try:
            while not load.done():
                self.loading_label.text = f"Loading {name}... {int(time.monotonic() - started)}s"
                await asyncio.wait({load}, timeout=1)
            await load  # re-raises LemonadeError if the load failed
        except LemonadeError as error:
            ui.notify(str(error), type="negative", multi_line=True)
        finally:
            self.loading_row.set_visibility(False)

    def _running_jobs(self) -> list:
        """The scheduled runs going now (empty when the Scheduler module is off or idle)."""
        module = self._module(SchedulerModule)
        return module.live.running() if module is not None else []

    def _update_job_note(self) -> None:
        """Beside the model picker: which scheduled job is running, and with which model."""
        if self._gone():
            return
        runs = self._running_jobs()
        if not runs:
            self.job_note.set_visibility(False)
            return
        text = "Job running: " + ", ".join(
            f"{run.name or 'scheduled job'} ({run.model or 'choosing a model'})" for run in runs
        )
        self.job_note.text = text
        self.job_note_tip.text = text
        self.job_note.set_visibility(True)

    async def _ok_to_switch_while_jobs_run(self, name: str) -> bool:
        """Loading another model can push a running job's model out of Lemonade's memory, so ask first.

        Picking the model the job itself is using needs no question. Nothing is blocked: Lemonade may have room for
        both, which this app cannot know.
        """
        others = [run for run in self._running_jobs() if run.model != name]
        if not others:
            return True
        using = "; ".join(
            f"{run.name or 'A scheduled job'} is using {run.model or 'a model'}" for run in others
        )
        return await ask_confirm(
            "A scheduled job is running",
            f"{using}. Loading {name} may stop it. Switch anyway?",
            ok_text="Switch anyway",
        )

    def _show_followed_model(self, model: str) -> None:
        """While you watch a running job, the picker shows the job's model and cannot be changed."""
        self._picker_followed = True
        self._set_model_options(model or self._picked())
        self.model_select.disable()

    def _end_followed_model(self) -> None:
        """Back to your own model (safe to call when nothing was being followed)."""
        if not self._picker_followed:
            return
        self._picker_followed = False
        self._set_model_options(self._picked())
        if not self._busy:
            self.model_select.enable()

    def _picked(self) -> str:
        """What the model list shows: the image model if one is picked, otherwise the chat model."""
        return self._image_model or self.conversation.model

    def _is_image_model(self, name: str) -> bool:
        return any(m.id == name and m.category == "image" for m in self._models)

    def _show_drawing_mode(self) -> None:
        self.input.props(
            f"placeholder='{'Describe the picture to draw...' if self._image_model else 'Type a message...'}'"
        )

    async def _model_selected(self, event) -> None:
        name = event.value
        if self._ignore_model_change or not name or name == self._picked():
            return
        # While a running job is being watched the picker is read-only and shows the job's model. The change that puts
        # that model there reaches this handler a moment later, after ``_ignore_model_change`` has been switched off
        # again (NiceGUI runs an async handler as a separate task), so it is recognised by the flag and by being
        # out of date: the box no longer shows the value this event carries.
        if self._picker_followed or name != self.model_select.value:
            return
        if is_header(name):  # a group heading is not a model
            self._set_model_options(self._picked())
            return
        if self._busy:
            ui.notify("Wait for the current reply to finish first.", type="warning")
            self._set_model_options(self._picked())
            return
        if not self._is_image_model(name) and not await self._ok_to_switch_while_jobs_run(name):
            self._set_model_options(
                self._picked()
            )  # the person said no: the picker goes back to what it was
            return
        if self._is_image_model(name):
            images = self._module(ImagesModule)
            if images is None or not images.enabled:
                ui.notify(
                    "Switch the Images module on in Settings > Modules to draw pictures.",
                    type="warning",
                )
                self._set_model_options(self._picked())
                return
            self._image_model = name
            self._show_drawing_mode()  # the message box now says "Describe the picture to draw..."
            return
        if self._image_model is not None:
            self._image_model = None
            self._show_drawing_mode()
        self.conversation.model = name
        self.context.settings.lemonade.chat_model = name  # remember the choice for next time
        self.context.save_settings()
        await self._load_model(name)
        self._update_context_label()

    # --- health ----------------------------------------------------------------------------------------------

    async def _poll_health(self) -> None:
        if self._busy:
            return
        await self._refresh_lemonade(announce_problems=False)

    def _show_health(self, healthy: bool) -> None:
        self._healthy = healthy
        self.health_dot.props(f"color={'positive' if healthy else 'negative'}")
        self.health_text.text = "Lemonade healthy" if healthy else "Lemonade unreachable"

    # --- chats: open, new, changed, deleted --------------------------------------------------------------------

    def _new_chat(self) -> None:
        if self._busy:
            ui.notify("Wait for the current reply to finish first.", type="warning")
            return
        if not self.conversation.is_saved:
            # Nothing to reset: this already is a new chat. Say so, or the button looks as if it did nothing.
            ui.notify("This is already a new chat.", timeout=1500)
            self.input.run_method("focus")
            return
        self._unfollow()
        self._allowed_tools.clear()
        self.conversation.new()
        self._show_messages([])
        self._reset_stats(None)
        self.chat_list.render.refresh()
        self._update_title()

    def _open_chat(self, session: ChatSession) -> None:
        if self._busy:
            ui.notify("Wait for the current reply to finish first.", type="warning")
            return
        self._unfollow()
        self._allowed_tools.clear()
        messages = self.conversation.open(session)
        self._show_messages(messages, running="running" in session.tags)
        if "running" in session.tags:
            self._follow_live(session)
        self._reset_stats(messages)
        self.chat_list.render.refresh()
        self._update_title()
        ui.run_javascript("lrScrollDown(true)")

    def _chat_changed(self) -> None:
        self.conversation.refresh_session()
        self._update_title()

    def _chat_deleted(self, session_id: str) -> None:
        if self.conversation.session and self.conversation.session.id == session_id:
            self._new_chat()

    def _update_title(self) -> None:
        session = self.conversation.session
        self.title.text = session.title if session else "New chat"
        self._refresh_kb_select()

    def _show_messages(self, messages: list[StoredMessage], *, running: bool = False) -> None:
        """Replace everything in the message area with the given stored messages.

        ``running`` is for a scheduled job's chat while the job is still working (it has no messages yet).
        """
        self.messages.clear()
        self._last_day = None
        thinking_open = self.context.settings.ui.thinking_open_by_default
        with self.messages:
            if not any(m.role in ("user", "assistant") for m in messages):
                self._empty_hint(running=running)
            # A reply that used tools is stored as several messages (assistant, tool results, assistant...).
            # They all belong to one bubble, which ends when the next user message begins.
            bubble: AssistantBubble | None = None
            for message in messages:
                if message.role == "user":
                    bubble = None
                    self._divider_if_new_day(message.created_at)
                    view = user_view(message.content)
                    add_user_message(view.text, message.created_at, view)
                elif message.role == "assistant":
                    if bubble is None:
                        self._divider_if_new_day(message.created_at)
                        bubble = AssistantBubble(thinking_open=thinking_open)
                    bubble.show_stored(message)
                elif message.role == "tool" and bubble is not None:
                    bubble.show_stored_tool_result(message)
                summary_ends_here = (
                    self.conversation.session is not None
                    and message.id == self.conversation.session.summary_upto
                )
                if (
                    summary_ends_here
                ):  # everything above this line is only sent to the model as a summary
                    add_summary_marker(self.conversation.session.summary)  # type: ignore[union-attr]
                    bubble = None
        self.stats_panel.set_tool_calls(bubble.tool_chips() if bubble else [])

    def _divider_if_new_day(self, moment) -> None:
        day = moment.astimezone().date()
        if day != self._last_day:
            add_date_divider(moment)
            self._last_day = day

    # --- stats -------------------------------------------------------------------------------------------------

    def _reset_stats(self, messages: list[StoredMessage] | None) -> None:
        """Recompute the totals and the context size from a chat's stored messages."""
        self.totals = SessionTotals()
        last: RequestStats | None = None
        for message in messages or []:
            if message.stats is not None:
                self.totals.add(message.stats)
                last = message.stats
        self._set_last_stats(last)

    def _set_last_stats(self, stats: RequestStats | None) -> None:
        self._last_stats = stats
        self._context_tokens = (stats.prompt_tokens + stats.output_tokens) if stats else None
        self.stats_panel.update(stats, self.totals)
        self._update_context_label()

    def _update_context_label(self) -> None:
        window = self._context_windows.get(self.conversation.model)
        self.conversation.context_window = (
            window  # lets the conversation summarise old messages in time
        )
        shown = self._context_tokens is not None and bool(window)
        self.context_ring.set_visibility(shown)
        if not shown or window is None or self._context_tokens is None:
            self.context_tip.text = ""
            return
        fraction = min(1.0, self._context_tokens / window)
        percent = round(100 * fraction)
        self.context_tip.text = (
            f"Context used: {self._context_tokens:,} / {window:,} tokens ({percent}%). "
            "Click for the breakdown."
        )
        self.context_percent.text = f"{percent}%"
        self.context_ring.value = percent
        self.context_ring.props(
            f"color={'negative' if fraction > 0.9 else 'warning' if fraction > 0.75 else 'primary'}"
        )

    async def _show_model_info(self) -> None:
        await show_model_details(self.context.client, self._models, self._picked())

    async def _show_context_breakdown(self) -> None:
        await show_context_breakdown(
            self.context.client,
            self.conversation,
            window=self._context_windows.get(self.conversation.model),
            tokens_now=self._context_tokens,
            typing=self.input.value or "",
        )

    # --- attachments -----------------------------------------------------------------------------------------

    def _images_folder(self):
        return attachments_dir(self.context.settings_file.parent)

    async def _on_upload(self, event) -> None:
        """A file was chosen: save it, then read it (a picture is shrunk, a document turned into text)."""
        self._show_attach_error("")
        upload = event.files[0]
        folder = Path(tempfile.mkdtemp(prefix="lemonrind-attach-"))
        try:
            path = (
                folder / Path(upload.name).name
            )  # .name drops any directory part of the file name
            await upload.save(path)
            try:
                self._pending = await asyncio.to_thread(prepare, path, self._images_folder())
            except AttachmentError as error:
                self._pending = None
                self._show_attach_error(str(error))
        finally:
            shutil.rmtree(folder, ignore_errors=True)
            self.upload.reset()
        self._render_pending()

    def _render_pending(self) -> None:
        """Draw the little preview of what is attached (or hide it), and re-check whether Send is allowed."""
        self.attach_row.clear()
        pending = self._pending
        self.attach_row.set_visibility(pending is not None)
        if pending is not None:
            with self.attach_row:
                if pending.is_image and pending.path is not None:
                    ui.image(attached_url(pending.path)).classes("rounded-borders").style(
                        "width: 44px; height: 44px; object-fit: cover"
                    )
                else:
                    ui.icon("attach_file").classes("lr-muted")
                ui.label(pending.name).classes("text-caption ellipsis").style("max-width: 18rem")
                ui.button(icon="close", on_click=self._remove_pending).props(
                    "flat round dense size=sm"
                ).tooltip("Remove the attachment").mark("attach-remove")
        self._update_attach_state()

    def _remove_pending(self) -> None:
        self._pending = None
        self._show_attach_error("")
        self._render_pending()

    def _restore_attachment(self, attachment: Attachment | None) -> None:
        """A message that was not saved goes back into the box, and so does what was attached to it."""
        if attachment is not None:
            self._pending = attachment
            self._render_pending()

    def _show_attach_error(self, message: str) -> None:
        self.attach_note.text = message
        self.attach_note.set_visibility(bool(message))

    def _vision_ok(self) -> bool:
        model = next((m for m in self._models if m.id == self.conversation.model), None)
        return bool(model and model.supports_vision)

    def _attachment_blocked(self) -> bool:
        """A picture is attached but the chosen model cannot see it."""
        return self._pending is not None and self._pending.is_image and not self._vision_ok()

    @staticmethod
    def _vision_warning() -> str:
        return "The selected model doesn't support image input - pick a vision-capable model or remove the attached picture."

    def _update_attach_state(self) -> None:
        """Warn and hold Send back while a picture is attached and the model cannot see it."""
        blocked = self._attachment_blocked()
        if blocked:
            self._show_attach_error(self._vision_warning())
        elif self.attach_note.text == self._vision_warning():
            self._show_attach_error("")
        self.send_button.set_enabled(
            self._busy or not blocked
        )  # the button is also Stop while a reply streams

    # --- sending -------------------------------------------------------------------------------------------------

    async def _send_clicked(self) -> None:
        """The send button doubles as the Stop button while a reply is streaming."""
        if self._busy:
            if self._task is not None:
                self._task.cancel()
            return
        text = (self.input.value or "").strip()
        if not text:
            return
        if self._image_model is not None:
            await self._draw(text)
            return
        if not self.conversation.model:
            ui.notify("No chat model is selected yet.", type="warning")
            return
        if self._attachment_blocked():
            ui.notify(self._vision_warning(), type="warning")
            return
        attachment, self._pending = self._pending, None
        self._render_pending()
        self.input.value = ""
        await self._send(text, attachment)

    async def _continue_reply(self, bubble: AssistantBubble) -> None:
        """The Continue button on a reply that stopped at the output limit: ask the model to carry on."""
        if self._busy:
            return
        bubble.hide_continue()
        await self._send(CONTINUE_PROMPT)

    async def _send(self, text: str, attachment: Attachment | None = None) -> None:
        self._set_busy(True)
        if self._continuable is not None and not self._continuable.root.is_deleted:
            self._continuable.hide_continue()  # a newer message makes the old button stale
        self._continuable = None
        thinking_open = self.context.settings.ui.thinking_open_by_default
        with self.messages:
            if not self.hint.is_deleted:
                self.hint.delete()
            now = datetime.now(UTC)
            self._divider_if_new_day(now)
            add_user_message(text, now, view_for(text, attachment))
            bubble = AssistantBubble(thinking_open=thinking_open)
        ui.run_javascript("lrScrollDown(true)")

        async def run() -> Reply:
            # NiceGUI remembers "which container am I building inside" per task, and a new task starts
            # with none. Entering the message area here lets the bubble's callbacks use ui.* calls.
            def on_event(event) -> None:
                bubble.on_event(event)
                if isinstance(event, ToolStarted | ToolFinished):
                    self.stats_panel.set_tool_calls(bubble.tool_chips())

            with self.messages:
                return await self.conversation.send(text, on_event, attachment=attachment)

        self._task = asyncio.create_task(run())
        try:
            reply = await self._task
        except asyncio.CancelledError:
            if self._task.cancelled():  # the Stop button: keep the part that arrived
                bubble.mark_stopped()
                if not self.conversation.is_saved or not self._exchange_saved(text):
                    self.input.value = (
                        text  # nothing was kept: put the message back so it can be re-sent
                    )
                    self._restore_attachment(attachment)
            else:
                raise  # the whole page is being torn down: let that happen
        except LemonadeError as error:
            bubble.show_error(str(error))
            self.input.value = text  # not saved: put the message back so it can be re-sent
            self._restore_attachment(attachment)
        except Exception as error:  # a bug: show it in the chat instead of leaving the page stuck
            bubble.show_error(f"Something went wrong: {error}")
            raise
        else:
            bubble.finish(reply, lambda: self._continue_reply(bubble))
            if reply.text and reply.finish_reason == "length":
                self._continuable = bubble
            for stats in reply.round_stats or (
                reply.stats,
            ):  # one entry per request (tools: several)
                self.totals.add(stats)
            self._set_last_stats(reply.stats)
        finally:
            self._task = None
            self._set_busy(False)
            self.conversation.refresh_session()
            self._update_title()
            self.chat_list.render.refresh()
            self.input.run_method("focus")

    async def _draw(self, prompt: str) -> None:
        """Draw a picture from ``prompt`` with the image model picked in the list, and save it as an exchange."""
        images = self._module(ImagesModule)
        model = self._image_model
        if images is None or not images.enabled or model is None:
            ui.notify(
                "Switch the Images module on in Settings > Modules to draw pictures.",
                type="warning",
            )
            return
        if self._pending is not None:
            ui.notify(
                "A picture or file cannot go with a drawing request. Remove it first.",
                type="warning",
            )
            return
        config = self.context.settings.modules.images
        size = await ask_image_size(prompt, model, config.width, config.height, config.max_side)
        if size is None:
            return  # cancelled: the description stays in the box
        self.input.value = ""
        self._set_busy(True)
        with self.messages:
            if not self.hint.is_deleted:
                self.hint.delete()
            now = datetime.now(UTC)
            self._divider_if_new_day(now)
            add_user_message(prompt, now)
            bubble = AssistantBubble()
        ui.run_javascript("lrScrollDown(true)")

        # Drawing sends no tokens back, so there is nothing to stream: show an honest stopwatch instead.
        self._task = asyncio.create_task(
            images.draw(prompt, model=model, width=size[0], height=size[1])
        )
        started = time.monotonic()
        try:
            while not self._task.done():
                bubble.show_progress(f"Drawing with {model}... {int(time.monotonic() - started)}s")
                await asyncio.wait({self._task}, timeout=1)
            drawn = self._task.result()
        except asyncio.CancelledError:
            if self._task.cancelled():  # the Stop button
                bubble.mark_stopped()
                self.input.value = prompt
            else:
                raise  # the whole page is being torn down: let that happen
        except (ToolError, LemonadeError) as error:
            bubble.show_error(str(error))
            self.input.value = prompt
        else:
            bubble.show_text(drawn.markdown)
            self.conversation.record_drawing(prompt, drawn.markdown)
        finally:
            self._task = None
            self._set_busy(False)
            self._update_title()
            self.chat_list.render.refresh()
            self.input.run_method("focus")

    def _exchange_saved(self, text: str) -> bool:
        """Was the message just sent written to the current chat? (A stopped reply is only kept if some answer arrived.)"""
        history = self.conversation.history
        return len(history) >= 2 and history[-2].get("role") == "user"

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        if busy:
            self.send_button.props("icon=stop color=negative")
            self.send_button.tooltip("Stop")
            self.model_select.disable()
        else:
            self.send_button.props("icon=arrow_upward color=primary")
            self.send_button.tooltip("Send (Enter)")
            self.model_select.enable()
        self._update_attach_state()

    # --- settings and theme -----------------------------------------------------------------------------------

    async def _open_settings(self, section: str = "lemonade") -> None:
        outcome = await open_settings(
            self.context, self._model_choices, self.conversation, initial=section
        )
        for note in outcome.notes:
            ui.notify(note, type="info", multi_line=True, timeout=8000)
        self._refresh_kb_select()  # the knowledge base attached to this chat may have changed
        if outcome.open_chat:
            self._open_chat_by_id(outcome.open_chat)
        if outcome.saved:
            self.conversation.set_system_prompt(build_system_prompt(self.context.settings))
            self.conversation.max_rounds = self.context.settings.modules.max_tool_rounds
            assistant = self.context.settings.assistant
            self.conversation.compaction_percent = assistant.compaction_percent
            self.conversation.compaction_keep_turns = assistant.compaction_keep_turns
            if self.context.modules is not None:
                await self.context.modules.reconcile()  # start or stop modules switched on or off
            self._apply_theme()
            ui.notify("Settings saved", type="positive")

    async def _view_prompt(self) -> None:
        await show_system_prompt(self.conversation)

    async def _view_turn_context(self) -> None:
        await show_turn_context(self.conversation, self.input.value or "")

    def _view_tools(self) -> None:
        show_tools_sent(self.conversation)

    def _view_logs(self) -> None:
        show_logs(self.context)

    def _view_usage(self) -> None:
        show_usage(self.context, self._open_chat_by_id)

    def _open_chat_by_id(self, session_id: str) -> None:
        if session := self.context.repo.get_session(session_id):
            self._open_chat(session)

    def _unfollow(self) -> None:
        if self._stop_following is not None:
            self._stop_following()
            self._stop_following = None
        self._end_followed_model()

    def _follow_live(self, session: ChatSession) -> None:
        """Show a running scheduled job as it works: catch up on what has happened, then follow new events."""
        module = self._module(SchedulerModule)
        live = module.live.get(session.id) if module is not None else None
        if live is None:
            return  # nothing to watch (the run just ended, or the app was restarted): the hint stays
        with self.messages:
            if not self.hint.is_deleted:
                self.hint.delete()
            add_user_message(live.prompt, datetime.now(UTC))
            bubble = AssistantBubble(
                thinking_open=self.context.settings.ui.thinking_open_by_default
            )
            for event in list(live.events):
                bubble.on_event(event)  # type: ignore[arg-type]
            bubble.repaint()
        ui.run_javascript("lrScrollDown(true)")
        self._show_followed_model(live.model)

        def gone() -> bool:
            return self.messages.is_deleted or self.client.id not in Client.instances

        def on_event(event: object) -> None:
            if gone():
                self._unfollow()
                return
            if live.model and self._picker_followed and self.model_select.value != live.model:
                self._show_followed_model(
                    live.model
                )  # the run has chosen its model since you opened it
            # the job runs in another task, which has no page context of its own
            with self.messages:
                bubble.on_event(event)  # type: ignore[arg-type]
                bubble.repaint_soon()

        def on_finished() -> None:
            if not gone():
                self._end_followed_model()
                asyncio.get_running_loop().create_task(self._after_live_run())

        self._stop_following = live.subscribe(on_event, on_finished)

    async def _after_live_run(self) -> None:
        """The run being watched has ended: after a moment (for the module to record it), show the saved chat."""
        await asyncio.sleep(0.6)
        if self.messages.is_deleted:
            return
        with self.messages:
            self._check_scheduled_runs()

    def _check_scheduled_runs(self) -> None:
        """Runs started by the scheduler happen in the background: when one finishes, show its chat in the list."""
        module = self._module(SchedulerModule)
        if module is None:
            return
        started = module.runs_started
        if (
            started != self._starts_seen
        ):  # a run began: its chat is already in the list, tagged "running"
            self._starts_seen = started
            self.chat_list.render.refresh()
        if module.runs_finished == self._runs_seen:
            return
        self._runs_seen = module.runs_finished
        self.chat_list.render.refresh()
        if module.last_outcome is not None:
            name, outcome = module.last_outcome
            viewing = self.conversation.session
            if (
                not self._busy
                and viewing is not None
                and viewing.id == outcome.session_id
                and (finished := self.context.repo.get_session(viewing.id)) is not None
            ):
                self._open_chat(
                    finished
                )  # you were looking at the empty chat: show what it holds now
            took = (
                f" (took {format_duration(outcome.seconds)})" if outcome.seconds is not None else ""
            )
            ui.notify(
                f"Scheduled job '{name}': {outcome.status}{took}",
                type="positive" if outcome.status == "ok" else "warning",
            )

    def _module(self, kind):
        """The registered module of this class, or ``None``."""
        modules = self.context.modules
        if modules is None:
            return None
        return next((m for m in modules.modules if isinstance(m, kind)), None)

    async def _approve_tool(self, call) -> bool:
        """Ask before a tool that needs permission runs. "Allow for this chat" lasts until the chat changes."""
        if call.name in self._allowed_tools:
            return True
        mcp = self._module(McpModule)
        server = mcp.server_name_for(call.name) if mcp else None
        answer = await ask_tool_approval(call.name, server, self.conversation.describe_call(call))
        if answer == "chat":
            self._allowed_tools.add(call.name)
        return answer in ("once", "chat")

    def _refresh_kb_select(self) -> None:
        """Fill the footer's knowledge-base drop-down and select the one attached to this chat."""
        module = self._module(KnowledgeModule)
        shown = module is not None and module.enabled
        self.kb_select.set_visibility(shown)
        if not shown or module is None:
            return
        options = {"": "No knowledge base", **{kb.id: kb.name for kb in module.repo.list_kbs()}}
        attached = self.conversation.options.get(KNOWLEDGE_OPTION, "")
        self._ignore_kb_change = (
            True  # set_options(value=) fires on_change, which would save what we just read
        )
        self.kb_select.set_options(options, value=attached if attached in options else "")
        self._ignore_kb_change = False

    def _kb_selected(self, event) -> None:
        if self._ignore_kb_change:
            return
        self.conversation.set_option(KNOWLEDGE_OPTION, event.value or None)

    def _cycle_theme(self) -> None:
        settings = self.context.settings.ui
        settings.theme = THEMES[settings.theme][2]  # type: ignore[assignment]  # always one of the three names
        self.context.save_settings()
        self._apply_theme()

    def _apply_theme(self) -> None:
        value, _, _ = THEMES[self.context.settings.ui.theme]
        if value is None:
            self.dark.auto()
        elif value:
            self.dark.enable()
        else:
            self.dark.disable()
        self._apply_theme_icon()

    def _apply_theme_icon(self) -> None:
        self.theme_button.props(f"icon={THEMES[self.context.settings.ui.theme][1]}")


async def index(client: Client) -> None:
    """The page served at ``/``: one new ``ChatPage`` for every browser tab."""
    await ChatPage(get_context(), client).build()


def register_pages() -> None:
    """Attach the page to the app. A function, so tests can register it again after NiceGUI resets."""
    ui.page("/")(index)
