"""The shapes the chat storage hands back: plain, immutable data classes.

Dataclass or pydantic model? A rule of thumb used throughout this project:

* **pydantic** where data comes from *outside* and cannot be trusted: JSON from a file or from Lemonade.
  Pydantic validates it and gives a clear error.
* **dataclass** for data we build ourselves from values we already trust, such as a row we just read
  from our own database. Simpler and lighter, no validation cost.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from lemonrind.lemonade.events import RequestStats, ToolCall

# The four roles the OpenAI message format has. ``Literal`` lets type checkers catch a typo like
# "assistent"; the database also enforces it with a CHECK constraint.
type Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One chat found by a search, and (when a message matched) a short piece of that message to show.

    ``snippet`` is the best matching message's text, shortened around the match, with each matched word wrapped in
    ``SNIPPET_START`` and ``SNIPPET_END`` (see ``parse_snippet``). ``None`` when the chat matched only through its
    title, folder or a tag.
    """

    session: ChatSession
    snippet: str | None = None


@dataclass(frozen=True, slots=True)
class Folder:
    id: str
    name: str
    collapsed: bool = False


@dataclass(frozen=True, slots=True)
class ChatSession:
    """One conversation (without its messages; load those with ``list_messages``)."""

    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    folder_id: str | None = None
    folder_name: str | None = None
    tags: tuple[str, ...] = ()  # a tuple, not a list: immutable, like the rest of the dataclass
    summary: str = ""  # summary of the oldest messages, once the chat has been compacted
    summary_upto: int = 0  # id of the last message the summary covers (0 = none)
    # Small per-chat choices that modules read (for example which knowledge base this chat uses).
    options: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StoredMessage:
    id: int
    session_id: str
    role: Role
    content: str
    created_at: datetime
    reasoning: str | None = None  # the model's thinking, kept for display but never sent back to it
    stats: RequestStats | None = None  # token counts and speed, for replies
    tool_calls: tuple[ToolCall, ...] = ()  # on an assistant message: the tools it asked for
    tool_call_id: str | None = None  # on a tool message: which call it answers
    tool_failed: bool = False  # on a tool message: the call failed (the content is then the error)
    model: str | None = (
        None  # on a reply with statistics: the model that answered (None before this was kept)
    )
