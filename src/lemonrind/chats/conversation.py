"""A conversation: the logic every front end shares for "send a message, run any tools, save the reply".

The terminal chat and the web UI both need exactly the same rules:

* the model is sent the system prompt plus the whole conversation so far;
* if the model asks for tools, they are run and their results sent back, over and over, until the model
  gives a final answer (the **tool loop**), up to a limit;
* modules may add to what the model is told: facts for the system prompt, background for one message;
* when the chat nears the model's context limit, the oldest messages are summarised (compaction);
* a chat is only written to the database once it has a real reply (no empty chats);
* a failed or empty reply leaves no trace;
* if the user presses Stop, the part of the answer that already arrived is kept.

Putting those rules here, once, means a front end only has to *show* events as they arrive.

Python ideas used here:

* ``typing.Protocol`` - describes "anything with a ``stream_chat`` method", without inheritance. The real
  ``LemonadeClient`` and the fakes in the tests both fit it. It is Python's answer to a C# interface, but
  structural: no class has to declare that it implements it. ``ToolRunner`` does the same for the module
  registry, so this file does not import any module.
* A *callback*: ``send`` takes an ``on_event`` function and calls it for every event. The front end decides
  what to do (print, update a web page).
* ``asyncio.create_task`` for background work (learning facts after a reply) that must not delay the
  user: the task is kept in a set so Python does not discard it while it is still running.
* ``except asyncio.CancelledError`` - how Python tells a task "you were cancelled". The Stop button works by
  cancelling the task that is running ``send``; catching the error lets us save the partial answer, and
  re-raising it afterwards is required so the cancellation completes properly.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from lemonrind.attachment_markers import parse_image_marker
from lemonrind.attachments import (
    Attachment,
    has_image,
    image_part,
    stored_text,
    text_of,
    wire_content,
)
from lemonrind.chats.compaction import SUMMARY_HEADER, split_for_compaction, summarise
from lemonrind.chats.models import ChatSession, Role, StoredMessage
from lemonrind.chats.repository import ChatRepository
from lemonrind.lemonade.events import (
    ChatEvent,
    Finished,
    PromptProgress,
    ReasoningDelta,
    RequestStats,
    TextDelta,
    ToolCall,
    ToolCallsRequested,
)
from lemonrind.modules.base import ChatContext
from lemonrind.modules.tool import ToolResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ToolStarted:
    """The conversation is about to run a tool."""

    call: ToolCall


@dataclass(frozen=True, slots=True)
class ToolFinished:
    call: ToolCall
    result: ToolResult
    seconds: float


@dataclass(frozen=True, slots=True)
class CompactionStarted:
    """The chat is nearly full, so its oldest messages are being summarised (this takes a moment)."""


@dataclass(frozen=True, slots=True)
class CompactionFinished:
    messages_summarised: int


@dataclass(frozen=True, slots=True)
class ContextUsed:
    """Modules added background text (for example recalled memories) to this message."""

    text: str


# Everything ``send`` reports: the model's events, plus these about tools and added context.
type ConversationEvent = (
    ChatEvent | ToolStarted | ToolFinished | ContextUsed | CompactionStarted | CompactionFinished
)
type EventHandler = Callable[[ConversationEvent], None]
# Asked before a tool that needs permission runs; True lets it run. The front end shows the question.
type Approver = Callable[[ToolCall], Awaitable[bool]]


class ChatStreamer(Protocol):
    """The one thing a conversation needs from the model client."""

    def stream_chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        model: str,
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[ChatEvent]: ...


class ToolRunner(Protocol):
    """What a conversation needs to offer and run tools (the module registry fits)."""

    def schemas(self) -> list[dict[str, Any]]: ...

    def requires_approval(self, call: ToolCall) -> bool: ...

    async def run(self, call: ToolCall) -> ToolResult: ...


class ContextProvider(Protocol):
    """What a conversation needs to let modules add context (the module registry fits)."""

    async def stable_context(self) -> str: ...

    async def turn_context(self, user_text: str, *, chat: ChatContext) -> str: ...

    async def after_turn(self, user_text: str, reply_text: str, *, model: str) -> None: ...


@dataclass(frozen=True, slots=True)
class Reply:
    """Everything one reply produced."""

    text: str  # the visible answer (empty if the model only thought, or hit the token limit)
    reasoning: str  # the model's thinking in the final request, "" when there was none
    stats: RequestStats | None  # the final request's numbers
    finish_reason: str | None  # "stop", "length", or "stopped" when the user pressed Stop
    # One entry per request made for this message (more than one when tools were used), for totals.
    round_stats: tuple[RequestStats, ...] = ()


@dataclass(slots=True)
class _Pending:
    """A message of the current turn, waiting to be saved once the turn succeeds."""

    role: Role
    content: str
    reasoning: str | None = None
    stats: RequestStats | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    tool_failed: bool = False


def wire_tool_call(call: ToolCall) -> dict[str, Any]:
    """A tool call in the shape the chat API wants inside an assistant message."""
    try:
        json.loads(call.arguments)
        arguments = call.arguments
    except json.JSONDecodeError:
        arguments = "{}"  # models sometimes write broken JSON; the server insists on valid JSON when we send it back
    return {
        "id": call.id,
        "type": "function",
        "function": {"name": call.name, "arguments": arguments},
    }


def wire_message(message: StoredMessage) -> dict[str, Any]:
    """A stored message as the dictionary sent to the model. (Saved *thinking* is never sent back.)"""
    if message.role == "assistant" and message.tool_calls:
        return {
            "role": "assistant",
            "content": message.content,
            "tool_calls": [wire_tool_call(call) for call in message.tool_calls],
        }
    if message.role == "tool":
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id or "",
            "content": message.content,
        }
    if message.role == "user":
        path, typed = parse_image_marker(message.content)
        if path is not None:
            if path.is_file():
                return {
                    "role": "user",
                    "content": [{"type": "text", "text": typed}, image_part(path)],
                }
            return {
                "role": "user",
                "content": typed,
            }  # the picture is gone: keep the question, drop the marker
    return {"role": message.role, "content": message.content}


def time_awareness_text(now: datetime | None = None) -> str:
    """One line telling the model the real date and time, because a model otherwise guesses from its training data."""
    moment = (now or datetime.now()).astimezone()
    return f"Current date/time: {moment:%A, %d %B %Y %H:%M} ({moment.tzname()})"


class Conversation:
    """The current chat: its saved record (once there is one) and the messages sent to the model."""

    def __init__(
        self,
        *,
        client: ChatStreamer,
        repo: ChatRepository,
        system_prompt: str,
        model: str,
        tools: ToolRunner | None = None,
        context: ContextProvider | None = None,
        max_rounds: int = 10,
        compaction_percent: int = 0,
        compaction_keep_turns: int = 3,
        context_window: int | None = None,
    ) -> None:
        self._client = client
        self._repo = repo
        self.model = model
        self.tools = tools
        self.context = context
        # Without an approver, tools that need permission are refused rather than run unasked.
        self.approve: Approver | None = None
        # An upper limit on one reply's length (None = the server's own default). Unattended runs set one so a
        # model stuck in a loop cannot generate until the job times out.
        self.max_output_tokens: int | None = None
        # When on, every message is sent with the current date and time added (volatile text goes in the message, not the
        # system prompt, so the server's cache of the prompt stays valid). The front ends switch it on.
        self.time_awareness = False
        self.max_rounds = max_rounds  # requests allowed for one message; the last one gets no tools
        self.compaction_percent = compaction_percent  # 0 = never compact
        self.compaction_keep_turns = compaction_keep_turns
        self.context_window = context_window  # the model's limit in tokens; set by the front end
        self._context_tokens: int | None = None  # how full the last request left it
        self._summary = ""  # summary of the oldest messages ("" until the chat has been compacted)
        self.options: dict[
            str, str
        ] = {}  # per-chat choices that modules read (e.g. knowledge base)
        self._base_prompt = system_prompt
        self._stable = ""  # what modules always add to the system prompt
        self._system_message = {"role": "system", "content": system_prompt}
        self._background: set[asyncio.Task] = set()  # work still running after a reply was shown

        self.session: ChatSession | None = None  # None = a draft that has not been saved yet
        self.history: list[dict[str, Any]] = [self._system_message]  # what is sent to the model

    # --- which chat is current -------------------------------------------------------------------------

    @property
    def is_saved(self) -> bool:
        return self.session is not None

    def _with_time(self, modules_added: str) -> str:
        """The text added to a message: the time line (if switched on) followed by whatever modules added."""
        parts = [time_awareness_text()] if self.time_awareness else []
        if modules_added:
            parts.append(modules_added)
        return "\n\n".join(parts)

    async def preview_turn_context(self, text: str) -> str:
        """What would be added to a message with this ``text`` if it were sent right now (for the "Turn context" screen).

        Memories and knowledge are searched *for the text*, so with nothing typed there is nothing to search with and only
        the time line is previewed.
        """
        modules_added = ""
        if text.strip():
            modules_added = await self._ask_context(
                "turn_context", text, chat=ChatContext(options=dict(self.options))
            )
        return self._with_time(modules_added)

    async def current_system_prompt(self) -> str:
        """The full system prompt the model will be sent next (so a screen can show it): persona, memory, summary."""
        await self._refresh_system_prompt()
        return self._system_message["content"]

    def set_system_prompt(self, prompt: str) -> None:
        """Change the system prompt, for this and the following messages."""
        self._base_prompt = prompt
        self._apply_system_text()

    def _apply_system_text(self) -> None:
        """Build the system prompt: base prompt, what modules always add, and the summary of old messages."""
        parts = [self._base_prompt]
        if self._stable:
            parts.append(self._stable)
        if self._summary:
            parts.append(SUMMARY_HEADER + self._summary)
        self._system_message = {"role": "system", "content": "\n\n".join(parts)}
        if self.history:
            self.history[0] = self._system_message

    async def _refresh_system_prompt(self) -> None:
        """Re-read what the modules always want the model to know (pinned memories can change anytime)."""
        self._stable = await self._ask_context("stable_context")
        self._apply_system_text()

    async def _ask_context(self, method: str, *args: str, **kwargs: object) -> str:
        """Ask the context provider; a failure means "nothing to add", never a failed chat."""
        if self.context is None:
            return ""
        try:
            return await getattr(self.context, method)(*args, **kwargs)
        except Exception:
            logger.warning("Context provider failed in %s", method, exc_info=True)
            return ""

    def new(self) -> None:
        """Start a fresh, unsaved chat."""
        self.session = None
        self._summary = ""
        self._context_tokens = None
        self.options = {}
        self.history = [self._system_message]
        self._apply_system_text()

    def set_option(self, key: str, value: str | None) -> None:
        """Set a per-chat option for modules (``None`` removes it). Saved with the chat once it exists."""
        if value is None:
            self.options.pop(key, None)
        else:
            self.options[key] = value
        if self.session is not None:
            self._repo.set_option(self.session.id, key, value)
            self.session = self._repo.get_session(self.session.id)

    def open(self, session: ChatSession) -> list[StoredMessage]:
        """Make ``session`` current, rebuilding the history from its stored messages.

        Returns the stored messages so the front end can show them.
        """
        messages = self._repo.list_messages(session.id)
        self.session = session
        self._summary = session.summary
        self.options = dict(session.options)
        # Messages the summary already covers are not sent to the model any more.
        self.history = [self._system_message] + [
            wire_message(m) for m in messages if m.role != "system" and m.id > session.summary_upto
        ]
        self._apply_system_text()
        # How full the context was after the last reply (replies before the summary no longer count).
        self._context_tokens = next(
            (
                m.stats.prompt_tokens + m.stats.output_tokens
                for m in reversed(messages)
                if m.stats is not None and m.id > session.summary_upto
            ),
            None,
        )
        return messages

    def refresh_session(self) -> None:
        """Re-read the saved chat (after a rename, folder or tag change made elsewhere)."""
        if self.session is not None:
            self.session = self._repo.get_session(self.session.id)

    # --- sending ---------------------------------------------------------------------------------------------

    async def send(
        self,
        text: str,
        on_event: EventHandler | None = None,
        *,
        attachment: Attachment | None = None,
    ) -> Reply:
        """Send ``text`` (with an optional picture or document), call ``on_event`` for each event, and save the exchange.

        A picture is sent to the model as an image part (only the newest picture in a chat is ever sent); a document's text
        goes inline. The saved message carries a marker so the chat can be rebuilt later (see ``attachments.py``).

        If the model asks for tools they are run (``ToolStarted`` / ``ToolFinished`` events) and the
        conversation continues until the model answers. Raises ``LemonadeError`` if a request fails
        (nothing is saved, and the same goes for any other error). If the task running this is cancelled
        (a Stop button), the part of the answer received so far is saved and ``CancelledError`` is
        raised again.
        """
        await self._refresh_system_prompt()
        await self._compact_if_needed(on_event)
        modules_added = await self._ask_context(
            "turn_context", text, chat=ChatContext(options=dict(self.options))
        )
        background = self._with_time(modules_added)
        if modules_added and on_event is not None:
            on_event(
                ContextUsed(modules_added)
            )  # only what modules added: the time line would be noise on every turn

        turn_start = len(self.history)  # on failure, everything after this point is thrown away
        self.history.append({"role": "user", "content": wire_content(text, attachment)})
        pending = [_Pending("user", stored_text(text, attachment))]
        round_stats: list[RequestStats] = []
        reasoning: list[str] = []
        answer: list[str] = []
        streaming = False  # True only while a reply is arriving (not while a tool runs)
        reply: Reply | None = None

        try:
            for round_number in range(1, max(1, self.max_rounds) + 1):
                reasoning, answer = [], []
                finished: Finished | None = None
                calls: tuple[ToolCall, ...] = ()
                # On the last allowed round tools are withheld, so the model has to answer in words.
                offered = self._tool_schemas() if round_number < self.max_rounds else []

                streaming = True
                limits = {"max_tokens": self.max_output_tokens} if self.max_output_tokens else {}
                async for event in self._client.stream_chat(
                    self._request_messages(turn_start, text, background),
                    self.model,
                    tools=offered or None,
                    **limits,
                ):
                    match event:
                        case ReasoningDelta(text=piece):
                            reasoning.append(piece)
                        case TextDelta(text=piece):
                            answer.append(piece)
                        case ToolCallsRequested(calls=requested):
                            calls = requested
                        case Finished():
                            finished = event
                        case PromptProgress():
                            pass  # only for the front end to display
                    if on_event is not None:
                        on_event(event)
                streaming = False

                round_text = "".join(answer).strip()
                round_thinking = "".join(reasoning).strip()
                stats = finished.stats if finished else None
                if stats is not None:
                    round_stats.append(stats)

                if not calls or self.tools is None or round_number >= self.max_rounds:
                    reply = Reply(
                        text=round_text,
                        reasoning=round_thinking,
                        stats=stats,
                        finish_reason=finished.finish_reason if finished else None,
                        round_stats=tuple(round_stats),
                    )
                    break

                # The model asked for tools: remember what it said, run each tool, and go round again.
                self.history.append(
                    {
                        "role": "assistant",
                        "content": round_text,
                        "tool_calls": [wire_tool_call(call) for call in calls],
                    }
                )
                pending.append(
                    _Pending(
                        "assistant",
                        round_text,
                        reasoning=round_thinking or None,
                        stats=stats,
                        tool_calls=calls,
                    )
                )
                for call in calls:
                    result = await self._run_tool(call, on_event)
                    self.history.append(
                        {"role": "tool", "tool_call_id": call.id, "content": result.content}
                    )
                    pending.append(
                        _Pending(
                            "tool",
                            result.content,
                            tool_call_id=call.id,
                            tool_failed=result.is_error,
                        )
                    )
        except asyncio.CancelledError:
            partial = "".join(answer).strip() if streaming else ""
            if partial:
                self._finish_turn(pending, partial, "".join(reasoning).strip() or None, None)
            else:
                del self.history[turn_start:]
            raise
        except Exception:
            # LemonadeError, or a bug in the front end's callback: either way the exchange did not
            # complete, so the question is not part of the conversation and nothing is saved.
            del self.history[turn_start:]
            raise

        assert reply is not None  # the loop always ends by building the reply (max_rounds >= 1)
        if reply.stats is not None:
            self._context_tokens = reply.stats.prompt_tokens + reply.stats.output_tokens
        if reply.text:
            self._finish_turn(pending, reply.text, reply.reasoning or None, reply.stats)
            self._learn_in_background(text, reply.text)
        else:
            del self.history[turn_start:]
        return reply

    def record_drawing(self, prompt: str, markdown: str) -> None:
        """Save a picture drawn straight from an image model as an exchange: your description, then the picture.

        (When the assistant draws through the ``generate_image`` tool the exchange is saved by ``send`` as usual.)
        The chat is tagged ``image``, as such chats are everywhere else.
        """
        self.history.append({"role": "user", "content": prompt})
        self._finish_turn([_Pending("user", prompt)], markdown, None, None)
        assert self.session is not None  # _finish_turn creates the chat if there was none yet
        self._repo.add_tag(self.session.id, "image")
        self.session = self._repo.get_session(self.session.id)

    async def _compact_if_needed(self, on_event: EventHandler | None) -> None:
        """Summarise the oldest messages when the chat has filled enough of the context window.

        Runs at the start of a send, before anything is added to the history, so a failure here (the model
        could not summarise) just means "try again next time". The chat carries on uncompacted.
        """
        limit = self.context_window
        if not (self.compaction_percent and limit and self._context_tokens and self.session):
            return
        if self._context_tokens < limit * self.compaction_percent / 100:
            return

        session = self.session
        stored = [
            m
            for m in self._repo.list_messages(session.id)
            if m.role != "system" and m.id > session.summary_upto
        ]
        cut = split_for_compaction(stored, self.compaction_keep_turns)
        if cut is None:
            return
        folded, kept = stored[:cut], stored[cut:]

        if on_event is not None:
            on_event(CompactionStarted())
        try:
            summary = await summarise(self._client, self.model, self._summary, folded)
        except Exception:  # a failed summary must not stop the chat (CancelledError still passes)
            logger.warning("Could not summarise the conversation", exc_info=True)
            summary = ""
        if not summary:
            return

        self._repo.set_summary(session.id, summary, folded[-1].id)
        self.session = self._repo.get_session(session.id)
        self._summary = summary
        self.history = [self._system_message] + [wire_message(m) for m in kept]
        self._apply_system_text()
        self._context_tokens = None  # unknown until the next reply reports it
        if on_event is not None:
            on_event(CompactionFinished(len(folded)))

    def _request_messages(
        self, turn_start: int, text: str, background: str
    ) -> list[dict[str, Any]]:
        """The history as sent to the model: the same, except the current message may carry background text, and
        only the newest picture keeps its image.

        The background is added to a *copy*, so the stored history (and the saved chat) holds exactly what
        you typed, and earlier messages stay identical from request to request, which lets the server reuse
        its cache of them. (Dropping older pictures is the one thing that changes an earlier message, and it
        happens once, the first time a newer picture arrives.)
        """
        messages = list(self.history)
        if background:
            current = messages[turn_start]
            prefix = (
                f"(Background for the assistant, not part of the user's message:\n{background})\n\n"
            )
            content = current["content"]
            if isinstance(
                content, list
            ):  # a message with a picture: the background goes in front of its text part
                content = [
                    {**part, "text": prefix + part["text"]}
                    if index == 0 and part.get("type") == "text"
                    else part
                    for index, part in enumerate(content)
                ]
            else:
                content = prefix + content
            messages[turn_start] = {**current, "content": content}
        newest = max((i for i, m in enumerate(messages) if has_image(m)), default=-1)
        return [
            {**m, "content": text_of(m)} if has_image(m) and i != newest else m
            for i, m in enumerate(messages)
        ]

    def _auto_tags(self, pending: list[_Pending]) -> list[str]:
        """Tags that say, at a glance in the chat list, how a chat was used: ``image``, ``coder`` or ``mcp``.

        (Chats made by scheduled jobs are tagged ``scheduled`` by the job runner.)
        """
        names = {call.name for message in pending for call in message.tool_calls}
        tags = []
        if "generate_image" in names:
            tags.append("image")
        if any(name.startswith("code_") for name in names):
            tags.append("coder")
        module_for_tool = getattr(self.tools, "module_for_tool", None)
        if module_for_tool is not None:
            for name in sorted(names):
                module = module_for_tool(name)
                if module is not None and getattr(module, "config_key", "") == "mcp":
                    tags.append("mcp")
                    break
        return tags

    def _learn_in_background(self, user_text: str, reply_text: str) -> None:
        """Let modules learn from the finished turn without making the user wait."""
        if self.context is None:
            return

        async def run() -> None:
            try:
                await self.context.after_turn(user_text, reply_text, model=self.model)  # type: ignore[union-attr]
            except Exception:
                logger.warning("Learning from the turn failed", exc_info=True)

        task = asyncio.create_task(run())
        self._background.add(task)  # a task nobody references can be garbage-collected mid-run
        task.add_done_callback(self._background.discard)

    async def drain(self, timeout: float = 60.0) -> None:
        """Wait (up to ``timeout`` seconds) for background learning to finish, e.g. before the app exits."""
        if self._background:
            await asyncio.wait(set(self._background), timeout=timeout)

    def _tool_schemas(self) -> list[dict[str, Any]]:
        return self.tools.schemas() if self.tools is not None else []

    async def _run_tool(self, call: ToolCall, on_event: EventHandler | None) -> ToolResult:
        if on_event is not None:
            on_event(ToolStarted(call))
        started = time.monotonic()
        assert self.tools is not None
        if self.tools.requires_approval(call) and not await self._is_allowed(call):
            result = ToolResult(
                "The user did not allow this tool to run. Do not try it again; tell the user what you "
                "wanted to do and ask how they would like to proceed.",
                True,
            )
        else:
            result = await self.tools.run(call)
        if on_event is not None:
            on_event(ToolFinished(call, result, time.monotonic() - started))
        return result

    def describe_call(self, call: ToolCall) -> str:
        """What to show in a permission question about ``call``: the tool's own preview when it has one."""
        describe = getattr(self.tools, "describe", None)
        return describe(call) if describe is not None else call.arguments

    async def _is_allowed(self, call: ToolCall) -> bool:
        """Ask the front end whether a tool that needs permission may run. No approver means no."""
        if self.approve is None:
            return False
        try:
            return bool(await self.approve(call))
        except Exception:  # a broken question is a refusal, never a silent yes
            logger.warning("Tool approval failed", exc_info=True)
            return False

    def _finish_turn(
        self,
        pending: list[_Pending],
        text: str,
        reasoning: str | None,
        stats: RequestStats | None,
    ) -> None:
        """Add the final assistant message to the turn and write the whole turn to the database."""
        pending.append(_Pending("assistant", text, reasoning=reasoning, stats=stats))
        self.history.append({"role": "assistant", "content": text})
        if self.session is None:
            self.session = self._repo.create_session()
            for key, value in self.options.items():  # choices made before the chat existed
                self._repo.set_option(self.session.id, key, value)
        for message in pending:
            self._repo.add_message(
                self.session.id,
                message.role,
                message.content,
                reasoning=message.reasoning,
                stats=message.stats,
                tool_calls=message.tool_calls,
                tool_call_id=message.tool_call_id,
                tool_failed=message.tool_failed,
                model=self.model if message.stats is not None else None,  # for the usage screen
            )
        for tag in self._auto_tags(pending):
            self._repo.add_tag(self.session.id, tag)
        self.session = self._repo.get_session(
            self.session.id
        )  # picks up the automatic title and tags
