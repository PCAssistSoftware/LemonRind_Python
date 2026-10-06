"""The contract every capability module follows.

A *module* is one self-contained feature: web search, the file system, later memory, knowledge bases, MCP
servers, the scheduler. Each one

* has a name, a stable key for the settings file, and a one-line description (shown in Settings),
* can be switched on or off on its own, and a disabled module is never asked for tools or started,
* contributes zero or more tools the model may call,
* may add text to what the model is told: always (``stable_context``) or just for one message
  (``turn_context``), and may act after a reply is finished (``after_turn``),
* may do work when it is switched on (``on_startup``) and clean up when it is switched off (``on_shutdown``).

Nothing outside a module needs to know what is inside it. The chat code only sees "some tools"; the
settings screen only sees "some modules with names and switches". That is what makes adding a feature a
one-file job. This is the same design as the ``IAssistantModule`` interface in the .NET editions.

Python ideas used here:

* ``abc.ABC`` and ``@abstractmethod`` - an *abstract base class*: a class that cannot be created itself and
  forces every subclass to implement the abstract methods. (Compare ``typing.Protocol`` in
  ``chats/conversation.py``: a Protocol is "anything shaped like this" with no inheritance, an ABC is
  "inherit from me", which suits a family of classes that share some default behaviour, as modules do.)
* **Class attributes** (``name = "Web search"``): data that belongs to the class itself, so it can be read
  without making an instance.
* A default implementation (``on_startup`` does nothing) so a simple module only writes what it needs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass

from lemonrind.config import Settings
from lemonrind.modules.tool import Tool


@dataclass(frozen=True, slots=True)
class ChatContext:
    """What a module may know about the chat a message belongs to.

    Modules are shared by every browser tab, so they cannot keep per-chat state themselves. Anything that
    belongs to one chat (such as the knowledge base attached to it) arrives here instead, as ``options``.
    """

    options: Mapping[str, str]


class Module(ABC):
    name: str  # shown in Settings, e.g. "Web search"
    config_key: str  # the key in settings.json's modules.enabled, e.g. "web_search"
    description: str  # one line shown under the name
    enabled_by_default: bool = True

    def __init__(self, settings: Settings) -> None:
        # The *same* Settings object the rest of the app uses, so a change made in the Settings dialog
        # is seen here straight away without anyone telling the module.
        self.settings = settings

    @property
    def enabled(self) -> bool:
        return self.settings.modules.enabled.get(self.config_key, self.enabled_by_default)

    # (Empty on purpose: a default so simple modules need not write these. The linter's B027 rule
    # warns about empty methods in an abstract class; here it is a deliberate choice.)
    async def on_startup(self) -> None:  # noqa: B027
        """Runs when the module becomes enabled (at start-up, or when it is switched on)."""

    async def on_shutdown(self) -> None:  # noqa: B027
        """Runs when the module is switched off, or when the app stops."""

    async def stable_context(self) -> str:
        """Text added to the system prompt. Keep it stable: it is sent with every request, and a prompt that
        changes every turn forces the server to re-read the whole conversation instead of using its cache."""
        return ""

    async def turn_context(self, user_text: str, *, chat: ChatContext) -> str:
        """Text added to just this one user message (for example "possibly relevant things you know")."""
        return ""

    async def after_turn(self, user_text: str, reply_text: str, *, model: str) -> None:  # noqa: B027
        """Runs in the background after a reply is complete (for example to learn facts from it)."""

    @abstractmethod
    def get_tools(self) -> list[Tool]:
        """The tools this module offers. Only asked while the module is enabled."""
