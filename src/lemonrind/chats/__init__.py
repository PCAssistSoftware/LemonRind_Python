"""Saved conversations: storage, search and the small helpers around them."""

from lemonrind.chats.conversation import (
    ChatStreamer,
    CompactionFinished,
    CompactionStarted,
    ContextProvider,
    ContextUsed,
    Conversation,
    ConversationEvent,
    Reply,
    ToolFinished,
    ToolRunner,
    ToolStarted,
)
from lemonrind.chats.export import export_markdown, safe_filename
from lemonrind.chats.models import ChatSession, Folder, Role, StoredMessage
from lemonrind.chats.persona import build_system_prompt
from lemonrind.chats.repository import ChatNotFoundError, ChatRepository
from lemonrind.chats.text import DEFAULT_TITLE, auto_title, build_fts_query, relative_time

__all__ = [
    "DEFAULT_TITLE",
    "ChatNotFoundError",
    "ChatRepository",
    "ChatSession",
    "ChatStreamer",
    "CompactionFinished",
    "CompactionStarted",
    "ContextProvider",
    "ContextUsed",
    "Conversation",
    "ConversationEvent",
    "Folder",
    "Reply",
    "Role",
    "StoredMessage",
    "ToolFinished",
    "ToolRunner",
    "ToolStarted",
    "auto_title",
    "build_fts_query",
    "build_system_prompt",
    "export_markdown",
    "relative_time",
    "safe_filename",
]
