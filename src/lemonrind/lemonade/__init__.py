"""Everything that talks to Lemonade Server: the HTTP client and the typed events it produces."""

from lemonrind.lemonade.client import LemonadeClient
from lemonrind.lemonade.events import (
    ChatEvent,
    Finished,
    PromptProgress,
    ReasoningDelta,
    TextDelta,
    ToolCall,
    ToolCallsRequested,
)

__all__ = [
    "ChatEvent",
    "Finished",
    "LemonadeClient",
    "PromptProgress",
    "ReasoningDelta",
    "TextDelta",
    "ToolCall",
    "ToolCallsRequested",
]
