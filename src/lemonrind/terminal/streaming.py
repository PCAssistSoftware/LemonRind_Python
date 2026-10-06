"""Showing one streamed reply on screen.

The conversation logic (what is sent, what is saved) lives in ``lemonrind.chats.Conversation``; this module
only *displays* the events it reports while a reply streams in.

Python ideas used here:

* ``match`` / ``case`` - pattern matching on the events.
* ``rich.live.Live`` - repaints one region of the terminal in place.
* A nested function that changes variables of the enclosing function, declared with ``nonlocal``.
"""

from __future__ import annotations

import asyncio
import time

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from lemonrind.attachments import Attachment
from lemonrind.chats import (
    CompactionFinished,
    CompactionStarted,
    ContextUsed,
    Conversation,
    ConversationEvent,
    Reply,
    ToolFinished,
    ToolStarted,
)
from lemonrind.lemonade import PromptProgress, ReasoningDelta, TextDelta, ToolCall
from lemonrind.lemonade.client import LemonadeError
from lemonrind.lemonade.events import RequestStats
from lemonrind.modules import ToolResult

# While the model is thinking, only the last few lines are shown so the screen does not jump around.
THINKING_TAIL_LINES = 5


async def stream_reply(
    conversation: Conversation, text: str, console: Console, *, attachment: Attachment | None = None
) -> Reply | None:
    """Send ``text`` and show the reply as it streams in. Returns ``None`` if the request failed."""
    reasoning: list[str] = []
    answer: list[str] = []
    progress_text: str | None = None
    tool_lines: dict[str, Text] = {}  # tool call id -> its status line, in the order called
    break_before_text = False
    notes: list[Text] = []  # what modules added to this message (recalled memories)
    thinking_started: float | None = None
    thinking_seconds: float | None = None

    def current_view() -> RenderableType:
        return render_reply(
            "".join(reasoning),
            "".join(answer),
            progress_text,
            thinking_seconds,
            notes + list(tool_lines.values()),
        )

    def on_event(event: ConversationEvent) -> None:
        # ``nonlocal`` lets this inner function assign to the variables defined just above.
        nonlocal progress_text, thinking_started, thinking_seconds, break_before_text
        match event:
            case PromptProgress():
                progress_text = event.describe()
            case ReasoningDelta(text=piece):
                thinking_started = thinking_started or time.monotonic()
                reasoning.append(piece)
            case CompactionStarted():
                notes.append(
                    Text(
                        "The chat is nearly full: summarising the oldest messages...",
                        style="dim yellow",
                    )
                )
            case CompactionFinished(messages_summarised=count):
                notes.append(
                    Text(f"Summarised {count} earlier messages to make room.", style="dim yellow")
                )
            case ContextUsed(text=background):
                notes.append(  # the full text is in the web app (View turn context); here a short note is enough
                    Text(
                        f"(background added to your message: {len(background.split())} words)",
                        style="dim cyan",
                    )
                )
            case ToolStarted(call=call):
                tool_lines[call.id] = describe_tool(call.name, call.arguments, None)
                break_before_text = bool(answer)
            case ToolFinished(call=call, result=result, seconds=seconds):
                tool_lines[call.id] = describe_tool(call.name, call.arguments, (result, seconds))
            case TextDelta(text=piece):
                progress_text = None
                if thinking_started is not None and thinking_seconds is None:
                    thinking_seconds = time.monotonic() - thinking_started
                if break_before_text:  # text before and after a tool call are separate paragraphs
                    answer.append("\n\n")
                    break_before_text = False
                answer.append(piece)
        live.update(current_view(), refresh=False)

    console.print("[bold cyan]Assistant:[/bold cyan]")
    # Live redraws one region of the terminal in place; refresh=False above means "just store the new
    # content", and Live's own timer repaints about 12 times a second.
    with Live(
        current_view(), console=console, refresh_per_second=12, vertical_overflow="visible"
    ) as live:

        async def approve(call: ToolCall) -> bool:
            """Ask whether a tool that needs permission may run (the live region is paused for the question)."""
            nonlocal progress_text, thinking_started, thinking_seconds
            live.stop()  # a question and a live-updating region cannot share the screen
            try:
                console.print(
                    Text.assemble(
                        ("\nThe assistant wants to run ", "bold yellow"), (call.name, "bold")
                    )
                )
                console.print(Text(conversation.describe_call(call)[:2000], style="dim"))
                answered = await asyncio.to_thread(console.input, "Allow this? [y/N] ")
            finally:
                # Everything shown so far stays on screen; carry on in a fresh region with what comes next.
                reasoning.clear()
                answer.clear()
                notes.clear()
                tool_lines.clear()
                progress_text, thinking_started, thinking_seconds = None, None, None
                live.start()
            return answered.strip().lower() in {"y", "yes"}

        previous_approver, conversation.approve = conversation.approve, approve
        try:
            reply = await conversation.send(text, on_event, attachment=attachment)
        except LemonadeError as error:
            live.update(Text(""), refresh=True)
            console.print(f"[red]{error}[/red]")
            return None
        finally:
            conversation.approve = previous_approver
        if thinking_started is not None and thinking_seconds is None:  # it only ever thought
            thinking_seconds = time.monotonic() - thinking_started
            live.update(current_view(), refresh=False)

    if not console.is_terminal:
        console.print()  # Live only ends its last line itself on a real terminal

    if reply.stats is not None:
        console.print(describe_stats(reply.stats))
    if not reply.text:
        hint = (
            "hit the token limit while thinking" if reply.finish_reason == "length" else "was empty"
        )
        console.print(f"[yellow]The reply {hint}, so it was not saved.[/yellow]")
    console.print()
    return reply


def render_reply(
    reasoning: str,
    answer: str,
    progress_text: str | None,
    thinking_seconds: float | None,
    tool_lines: list[Text] | None = None,
) -> RenderableType:
    """Build what the live region shows right now."""
    parts: list[RenderableType] = []
    if progress_text:
        parts.append(Text(progress_text, style="yellow"))
    if reasoning:
        if thinking_seconds is None:
            # Still thinking: show the tail of the thinking in a dim box.
            tail = "\n".join(reasoning.splitlines()[-THINKING_TAIL_LINES:])
            parts.append(
                Panel(Text(tail, style="dim italic"), title="Thinking...", border_style="grey50")
            )
        else:
            words = len(reasoning.split())
            parts.append(
                Text(f"Thought for {thinking_seconds:.1f}s ({words:,} words)", style="dim italic")
            )
    parts.extend(tool_lines or [])
    if answer:
        parts.append(Markdown(answer))
    return Group(*parts) if parts else Text("")


def describe_tool(name: str, arguments: str, outcome: tuple[ToolResult, float] | None) -> Text:
    """One status line for a tool call: ``tool web_search {"query": "..."}`` then how it went."""
    shown = " ".join(arguments.split())
    if len(shown) > 80:
        shown = shown[:77] + "..."
    line = Text.assemble(("tool ", "magenta"), (name, "bold magenta"), (f" {shown}", "dim"))
    if outcome is None:
        line.append(" ...", style="dim")
    else:
        result, seconds = outcome
        if result.is_error:
            line.append(f"  failed: {result.content[:100]}", style="red")
        else:
            line.append(
                f"  done in {seconds:.1f}s ({len(result.content):,} characters)", style="green"
            )
    return line


def describe_stats(stats: RequestStats) -> Text:
    return Text(stats.summary(), style="dim")
