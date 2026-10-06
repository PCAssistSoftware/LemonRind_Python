"""Conversation compaction: when a chat outgrows the model's memory, summarise its oldest part.

A model can only read so much at once, its *context window* (about 41,000 tokens for the model used in
development). Every request sends the whole conversation, so a long chat eventually would not fit. The usual
answer, and the one the .NET editions use, is a **running summary**: when the conversation fills a set share of
the window, the oldest messages are condensed into a short summary, and from then on the model is sent

    system prompt + summary of the old part + the recent messages word for word

Nothing is deleted. The old messages stay in the database and on screen; they are just no longer sent.

The decisions that matter, all in this file as plain functions so they can be tested without a model:

* **Where to cut.** Always at the start of one of *your* messages, never in the middle of an exchange: a
  reply that used tools is several stored messages (the assistant's tool call, the tool result, the answer) and
  splitting them would leave a tool result without the call it answers. The last few exchanges are kept
  word for word.
* **What to summarise.** The previous summary plus the messages being folded in, so the summary is "running":
  it keeps absorbing older material instead of growing without limit.
* **Tool results are shortened** in the transcript: a web page or file the model read is long and the
  summary only needs to know it happened.

Python ideas used here:

* Slicing with a computed index (``messages[:cut]``, ``messages[cut:]``).
* ``enumerate`` and list comprehensions to find positions.
* Reusing the existing streaming client for a one-off request, instead of adding a second way to call it.
"""

from __future__ import annotations

from collections.abc import Sequence

from lemonrind.chats.models import StoredMessage
from lemonrind.lemonade.events import TextDelta

SYSTEM_PROMPT = (
    "Summarize the following conversation excerpt concisely, in plain prose. Preserve concrete facts, "
    "decisions, names, numbers, file names, code snippets worth keeping, and anything the assistant would "
    "need to continue the conversation naturally. If an earlier summary is included, fold it into the new "
    "one. Omit small talk and filler. Output only the summary text, with no preamble or commentary about the "
    "summary itself."
)

# How the summary is presented to the model inside the system prompt.
SUMMARY_HEADER = (
    "Summary of the earlier part of this conversation (those messages are no longer included):\n"
)

TOOL_RESULT_CHARS = 600  # a tool's result is cut to this many characters in the transcript


def split_for_compaction(messages: Sequence[StoredMessage], keep_turns: int) -> int | None:
    """Where to cut: messages before the returned index get summarised, the rest are kept.

    ``messages`` are the ones still sent to the model. A *turn* starts at each user message. The cut is
    placed at the start of the ``keep_turns``-th most recent turn, so that many recent turns stay intact.
    Returns ``None`` when there is nothing worth folding (too few turns).
    """
    turn_starts = [index for index, message in enumerate(messages) if message.role == "user"]
    if len(turn_starts) <= keep_turns:
        return None
    cut = turn_starts[-keep_turns]
    return cut if cut > 0 else None


def build_transcript(previous_summary: str, messages: Sequence[StoredMessage]) -> str:
    """The text the summarising request reads: the earlier summary (if any) and the messages to fold in."""
    tool_names: dict[str, str] = {}  # tool call id -> tool name, to label results
    lines: list[str] = []
    if previous_summary:
        lines.append(f"Earlier summary: {previous_summary}\n")
    for message in messages:
        if message.role == "user":
            lines.append(f"User: {message.content}")
        elif message.role == "assistant":
            if message.content:
                lines.append(f"Assistant: {message.content}")
            for call in message.tool_calls:
                tool_names[call.id] = call.name
                lines.append(f"Assistant used the tool {call.name} with {call.arguments}")
        elif message.role == "tool":
            name = tool_names.get(message.tool_call_id or "", "tool")
            shown = message.content
            if len(shown) > TOOL_RESULT_CHARS:
                shown = shown[:TOOL_RESULT_CHARS] + " [...]"
            lines.append(f"Result of {name}: {shown}")
    return "\n".join(lines)


async def summarise(
    client, model: str, previous_summary: str, messages: Sequence[StoredMessage]
) -> str:
    """Ask the model for a summary. Returns "" if it produced none.

    ``client`` is anything with ``stream_chat`` (the same thing a conversation uses); only the visible text of
    the reply is kept, not the model's thinking.
    """
    request = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_transcript(previous_summary, messages)},
    ]
    parts: list[str] = []
    async for event in client.stream_chat(request, model):
        if isinstance(event, TextDelta):
            parts.append(event.text)
    return "".join(parts).strip()
