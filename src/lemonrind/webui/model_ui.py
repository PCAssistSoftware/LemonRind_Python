"""The model picker and the two popups about models and context.

* **The picker** lists every downloaded model in groups (Chat, Image, Embedding, Other) with a header and a divider
  between groups, read live from Lemonade. Quasar's ``q-select`` has no groups of its own, so the headers are ordinary
  options marked ``disable`` (they cannot be chosen) and drawn differently by an ``option`` slot.
* **Model details** (the info button beside the picker): what the catalog says about a model (capabilities, largest
  context window, size on disk, backend, where it comes from), what Lemonade is running right now if it is loaded (device,
  the context size actually in use, the real launch arguments and command line), and a table of every model on the server.
* **Context breakdown** (click the context ring beside the Send button): how full the model's memory is, split into the
  system prompt, the tools, and the conversation, counted by Lemonade's own tokenizer (an exact count, not characters
  divided by four), with the point where the chat would be summarised marked on the bar.

Python / NiceGUI ideas used here:

* Reaching into a widget's ``_props`` to add a per-option flag the public API does not offer, then ``update()``.
* ``dataclass`` for a result that the screen only displays, with the arithmetic in a plain function that tests can call.
* ``asyncio.gather`` to run several independent requests at once.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass

from nicegui import ui

from lemonrind.chats import Conversation
from lemonrind.lemonade.client import LemonadeClient, LemonadeError
from lemonrind.lemonade.models import Health, LoadedModel, ModelInfo
from lemonrind.webui.inspectors import make_dialog

GROUPS = (
    ("chat", "Chat models"),
    ("image", "Image models"),
    ("embedding", "Embedding models"),
    ("other", "Other models"),
)
HEADER_PREFIX = "\x00group:"  # option keys that are group headers, not models


def group_models(models: Sequence[ModelInfo]) -> list[tuple[str, str, list[ModelInfo]]]:
    """``(key, header, models)`` for each group that has any model, in Chat, Image, Embedding, Other order."""
    grouped = []
    for key, header in GROUPS:
        members = sorted((m for m in models if m.category == key), key=lambda m: m.id.lower())
        if members:
            grouped.append((key, header, members))
    return grouped


class GroupedSelect(ui.select):
    """A ``ui.select`` whose group-header options cannot be chosen.

    NiceGUI rebuilds the option list from scratch every time ``update()`` runs, which would throw away a flag added from
    outside. Overriding the method that builds the list makes the flags part of every rebuild.
    """

    def _update_options(self) -> None:
        super()._update_options()
        for option, key in zip(self._props["options"], self._values, strict=False):
            if isinstance(key, str) and key.startswith(HEADER_PREFIX):
                option["disable"] = True  # Quasar will not let a disabled option be picked
                option["header"] = True  # our option slot draws it as a heading


def fill_model_select(select: ui.select, models: Sequence[ModelInfo], current: str) -> None:
    """Show ``models`` in groups (a heading, then its models) with ``current`` selected."""
    options: dict[str, str] = {}
    for key, header, members in group_models(models):
        options[HEADER_PREFIX + key] = header
        options.update({m.id: m.id for m in members})
    if current and current not in options:
        options[current] = current  # a model Lemonade no longer lists, kept so the box is not blank
    select.set_options(options, value=current or None)


def add_group_styling(select: ui.select) -> None:
    """Draw header options as small bold captions with a line above, and models as ordinary rows."""
    select.add_slot(
        "option",
        """
        <q-item-label v-if="props.opt.header" header class="lr-model-group">{{ props.opt.label }}</q-item-label>
        <q-item v-else v-bind="props.itemProps">
            <q-item-section><q-item-label>{{ props.opt.label }}</q-item-label></q-item-section>
        </q-item>
        """,
    )


def is_header(value: str | None) -> bool:
    return bool(value) and str(value).startswith(HEADER_PREFIX)


# --- model details -------------------------------------------------------------------------------------------------


def _row(label: str, value: str, *, mono: bool = False) -> None:
    ui.label(label).classes("text-caption lr-muted")
    ui.label(value).classes("lr-tool-text" if mono else "")


def _loaded(health: Health | None, name: str) -> LoadedModel | None:
    if health is None:
        return None
    return next((m for m in health.all_models_loaded if m.model_name.lower() == name.lower()), None)


async def show_model_details(
    client: LemonadeClient, models: Sequence[ModelInfo], name: str
) -> None:
    """Open the details popup for ``name``. The live part asks Lemonade at the moment of opening."""
    catalog = next((m for m in models if m.id == name), None)
    try:
        health: Health | None = await client.health()
    except LemonadeError:
        health = None
    live = _loaded(health, name)

    dialog, card = make_dialog(name or "Model details", "")
    with card:
        if catalog is not None:
            with ui.row().classes("gap-1"):
                ui.badge(catalog.category).props("outline")
                for label in catalog.labels:
                    if label.lower() != catalog.category:  # "chat" is already shown as the group
                        ui.badge(label).props("outline color=grey")
            with ui.grid(columns="11rem 1fr").classes("w-full gap-x-3 gap-y-1"):
                _row(
                    "Largest context window",
                    f"{catalog.context_window:,} tokens" if catalog.context_window else "(unknown)",
                )
                _row(
                    "Size on disk",
                    f"{catalog.size:.2f} GB" if catalog.size is not None else "(unknown)",
                )
                _row("Backend", catalog.recipe or "(unknown)")
                _row("Comes from", catalog.source or "(unknown)")
                if catalog.checkpoint:
                    _row("Checkpoint", catalog.checkpoint, mono=True)
                if catalog.recipe_options:
                    _row("Default options", json.dumps(catalog.recipe_options), mono=True)
        else:
            ui.label("No catalog details are available for this model.").classes("lr-muted")

        ui.separator()
        if health is None:
            ui.label("Could not ask Lemonade whether it is loaded.").classes("lr-muted")
        elif live is None:
            ui.label("Not currently loaded.").classes("lr-muted")
        else:
            ui.label("Currently loaded (live)").classes("text-weight-medium")
            with ui.grid(columns="11rem 1fr").classes("w-full gap-x-3 gap-y-1"):
                _row("Device", live.device or "(unknown)")
                _row("Status", f"{live.status or 'ready'}{' (busy)' if live.is_busy else ''}")
                options = live.recipe_options
                if options and options.ctx_size:
                    _row("Context size in use", f"{options.ctx_size:,} tokens")
                if live.max_context_window:
                    _row("Largest it supports", f"{live.max_context_window:,} tokens")
                if options and options.llamacpp_args:
                    _row("Launch arguments", options.llamacpp_args, mono=True)
            if live.launch_command:
                with ui.expansion("Full launch command").classes("w-full").props("dense"):
                    ui.label(" ".join(live.launch_command)).classes("lr-tool-text")

        with (
            ui.expansion(f"All models on this server ({len(models)})")
            .classes("w-full")
            .props("dense")
        ):
            loaded_names = (
                {m.model_name.lower() for m in health.all_models_loaded} if health else set()
            )
            for _key, header, members in group_models(models):
                ui.label(header).classes("text-caption text-weight-medium q-mt-sm")
                for model in members:
                    with ui.row().classes("w-full no-wrap items-center gap-2"):
                        ui.icon("circle", size="xs").props(
                            f"color={'positive' if model.id.lower() in loaded_names else 'grey'}"
                        )
                        ui.label(model.id).classes("flex-grow ellipsis")
                        size = f"{model.size:.1f} GB" if model.size is not None else ""
                        ui.label(size).classes("text-caption lr-muted")
    dialog.open()


# --- context breakdown ---------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ContextBreakdown:
    now: int  # tokens in use
    maximum: int  # the model's window (0 if unknown)
    system_prompt: int
    tools: int
    history: int
    threshold: int  # where summarising of old messages would start (0 if compaction is off)
    pending: int  # what the message being typed would add
    exact: bool  # counted by Lemonade's tokenizer (False = estimated from text length)
    summarised: bool  # older messages of this chat have already been replaced by a summary

    @property
    def free(self) -> int:
        return max(0, self.maximum - self.system_prompt - self.tools - self.history)

    @property
    def percent(self) -> float:
        return 100 * self.now / self.maximum if self.maximum else 0.0

    @property
    def would_overflow(self) -> bool:
        return bool(self.maximum) and self.now + self.pending >= self.maximum


def _message_text(message: dict) -> str:
    text = message.get("content") or ""
    if not isinstance(text, str):  # a message with image parts: count only its text parts
        text = " ".join(part.get("text", "") for part in text if isinstance(part, dict))
    calls = message.get("tool_calls")
    return text + (json.dumps(calls) if calls else "")


async def compute_breakdown(
    client: LemonadeClient,
    conversation: Conversation,
    *,
    window: int | None,
    tokens_now: int | None,
    typing: str = "",
) -> ContextBreakdown:
    system_text = await conversation.current_system_prompt()
    tools = conversation.tools.schemas() if conversation.tools is not None else []
    tools_text = json.dumps(tools) if tools else ""
    history_text = "\n".join(_message_text(m) for m in conversation.history[1:])
    pending_text = ""
    if typing.strip():
        pending_text = f"{await conversation.preview_turn_context(typing)}\n\n{typing}"

    texts = [system_text, tools_text, history_text, pending_text]
    try:
        counts = await asyncio.gather(*(client.tokenize(text) for text in texts))
        exact = True
    except (
        LemonadeError
    ):  # the tokenizer is unreachable: fall back to the usual rough rule of four characters a token
        counts = [max(1, len(text) // 4) if text else 0 for text in texts]
        exact = False
    system, tools_tokens, history, pending = counts
    maximum = window or 0
    percent = conversation.compaction_percent
    return ContextBreakdown(
        now=tokens_now if tokens_now is not None else system + tools_tokens + history,
        maximum=maximum,
        system_prompt=system,
        tools=tools_tokens,
        history=history,
        threshold=round(maximum * percent / 100) if maximum and percent else 0,
        pending=pending,
        exact=exact,
        summarised=bool(conversation.session and conversation.session.summary),
    )


SEGMENT_COLOURS = {"system": "#2F6BFF", "tools": "#E0A030", "history": "#3AA76D"}


def _segment(width: float, colour: str, title: str) -> None:
    ui.element("div").style(f"width: {width:.2f}%; background: {colour}; height: 100%").tooltip(
        title
    )


async def show_context_breakdown(
    client: LemonadeClient,
    conversation: Conversation,
    *,
    window: int | None,
    tokens_now: int | None,
    typing: str = "",
) -> None:
    dialog, card = make_dialog("Context usage breakdown", "")
    with card:
        loading = ui.label("Counting tokens...").classes("lr-muted")
    dialog.open()

    info = await compute_breakdown(
        client, conversation, window=window, tokens_now=tokens_now, typing=typing
    )
    loading.delete()
    with card:
        total = (
            f"{info.now:,} / {info.maximum:,} tokens ({round(info.percent)}% full)"
            if info.maximum
            else f"{info.now:,} tokens"
        )
        ui.label(total).classes("text-weight-medium").mark("breakdown-total")
        if not info.exact:
            ui.label("Approximate: Lemonade's own tokenizer could not be reached.").classes(
                "text-caption lr-warn"
            )
        if info.maximum:
            with (
                ui.element("div")
                .classes("w-full")
                .style(
                    "position: relative; height: 14px; background: rgba(127,127,127,0.25); border-radius: 7px; overflow: hidden; display: flex"
                )
            ):
                for key, count, title in (
                    ("system", info.system_prompt, "System prompt"),
                    ("tools", info.tools, "Tools"),
                    ("history", info.history, "Conversation"),
                ):
                    _segment(
                        100 * count / info.maximum,
                        SEGMENT_COLOURS[key],
                        f"{title}: {count:,} tokens",
                    )
                if info.threshold:
                    ui.element("div").style(
                        f"position: absolute; left: {100 * info.threshold / info.maximum:.2f}%; top: 0; width: 2px; height: 100%; background: #d33"
                    ).tooltip("Older messages are summarised from here")
        with ui.grid(columns="1fr auto").classes("w-full gap-x-3 gap-y-1"):
            for key, label, count in (
                ("system", "System prompt", info.system_prompt),
                ("tools", "Tools", info.tools),
                ("history", "Conversation (summary, pinned facts, messages)", info.history),
            ):
                with ui.row().classes("items-center no-wrap gap-2"):
                    ui.element("div").style(
                        f"width: 10px; height: 10px; border-radius: 2px; background: {SEGMENT_COLOURS[key]}"
                    )
                    ui.label(label)
                ui.label(f"{count:,} tokens").classes("lr-muted")
            if info.maximum:
                ui.label("Free")
                ui.label(f"{info.free:,} tokens").classes("lr-muted")
            if info.threshold:
                ui.label("Summarising starts at")
                ui.label(f"{info.threshold:,} tokens").classes("lr-muted")
            if info.summarised:
                ui.label("This chat has been summarised")
                ui.label("older messages are condensed").classes("lr-muted")
        if typing.strip():
            note = f"If you send what you are typing now: +{info.pending:,} tokens"
            if info.would_overflow:
                note += " - this would exceed the model's context window"
            ui.label(note).classes("lr-warn" if info.would_overflow else "text-caption")
