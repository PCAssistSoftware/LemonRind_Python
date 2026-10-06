"""The Memory module: the assistant remembers lasting facts about you, across chats.

Three things happen, each through one of the module hooks:

1. **Pinned facts** ("The user's name is Darren.") are core facts that go into the system prompt of every
   conversation (``stable_context``).
2. **Other facts** are found by *meaning*. For each message you send, the module embeds it, compares it
   with every stored fact's embedding, and adds the best matches to just that message
   (``turn_context``), but only above a similarity threshold, so a store full of unrelated facts adds nothing.
3. **Learning**: after a reply, a separate model request picks out new lasting facts from what you said
   and stores them (``after_turn``). A new fact that is nearly identical to one already stored is dropped.

The model can also be asked directly: the ``remember`` and ``recall`` tools.

Memory is "best effort" throughout. If the embedding model or Lemonade is unavailable, a lookup returns
nothing and learning is skipped; the chat itself is never affected.

Python ideas used here:

* ``typing.Protocol`` for the client (so tests pass a tiny fake), as in ``chats/conversation.py``.
* ``numpy`` similarity search through ``vectors.rank``.
* ``logging`` for failures that must not interrupt the user.
"""

from __future__ import annotations

import logging
from typing import Protocol

from lemonrind.config import Settings
from lemonrind.embeddings import Embedder
from lemonrind.lemonade.client import LemonadeError
from lemonrind.lemonade.models import ModelInfo
from lemonrind.modules.base import ChatContext, Module
from lemonrind.modules.memory import extraction
from lemonrind.modules.memory.repository import Memory, MemoryRepository
from lemonrind.modules.tool import Tool, ToolError, tool_from_function
from lemonrind.vectors import rank

logger = logging.getLogger(__name__)


class MemoryClient(Protocol):
    """What the module needs from the Lemonade client."""

    async def list_models(self, *, downloaded_only: bool = True) -> list[ModelInfo]: ...

    async def embed(self, texts: list[str], model: str) -> list[list[float]]: ...

    async def complete_json(
        self, messages: list[dict[str, str]], model: str, *, schema_name: str, schema: dict
    ) -> str: ...


class MemoryModule(Module):
    name = "Memory"
    config_key = "memory"
    description = "Remember lasting facts about you across chats."

    def __init__(
        self,
        settings: Settings,
        repo: MemoryRepository,
        client: MemoryClient,
        embedder: Embedder | None = None,
    ) -> None:
        super().__init__(settings)
        self.repo = repo
        self._client = client
        self._embedder = embedder or Embedder(settings, client)

    # --- embeddings ----------------------------------------------------------------------------------

    async def _embed(self, text: str) -> list[float] | None:
        """The embedding of ``text``, or ``None`` when no embedding model is available (or the call fails)."""
        return await self._embedder.embed_one(text)

    async def embed_missing(self) -> int:
        """Give an embedding to memories saved while no embedding model was available. Returns how many."""
        done = 0
        for memory in self.repo.without_embeddings():
            if (vector := await self._embed(memory.content)) is not None:
                self.repo.set_embedding(memory.id, vector)
                done += 1
        return done

    async def on_startup(self) -> None:
        await self.embed_missing()

    # --- the hooks ---------------------------------------------------------------------------------------

    async def stable_context(self) -> str:
        pinned = self.repo.list_pinned()
        if not pinned:
            return ""
        return "Known facts about the user:\n" + "\n".join(f"- {m.content}" for m in pinned)

    async def turn_context(self, user_text: str, *, chat: ChatContext) -> str:
        config = self.settings.modules.memory
        found = await self.search(
            user_text, limit=config.max_relevant, min_similarity=config.min_similarity
        )
        if not found:
            return ""
        return "Possibly relevant things you know about the user:\n" + "\n".join(
            f"- {memory.content}" for memory, _ in found
        )

    async def after_turn(self, user_text: str, reply_text: str, *, model: str) -> None:
        if not self.settings.modules.memory.extract_facts or not user_text.strip():
            return
        try:
            answer = await self._client.complete_json(
                extraction.build_messages(user_text, reply_text),
                model,
                schema_name=extraction.SCHEMA_NAME,
                schema=extraction.SCHEMA,
            )
        except LemonadeError:
            logger.warning("Fact extraction failed", exc_info=True)
            return
        for fact in extraction.parse_facts(answer):
            await self.save_fact(fact.content, pinned=fact.pinned)

    # --- operations (used by the tools and by the screens) -----------------------------------------------

    async def search(
        self, text: str, *, limit: int, min_similarity: float = 0.0
    ) -> list[tuple[Memory, float]]:
        """Stored memories most similar in meaning to ``text`` (pinned ones excluded: they are always given)."""
        query = await self._embed(text)
        if query is None:
            return []
        stored = [
            (memory, vector)
            for memory, vector in self.repo.with_embeddings()
            if not memory.pinned
            and len(vector) == len(query)  # a different embedding model gives another size
        ]
        ranked = rank(query, [vector for _, vector in stored], limit=limit)
        return [(stored[i][0], score) for i, score in ranked if score >= min_similarity]

    async def save_fact(self, content: str, *, pinned: bool) -> Memory | None:
        """Store a fact unless an almost identical one exists. Returns the new memory, or ``None`` if skipped."""
        content = content.strip()
        if not content:
            return None
        vector = await self._embed(content)
        if vector is not None:
            duplicate_at = self.settings.modules.memory.duplicate_similarity
            stored = [v for _, v in self.repo.with_embeddings() if len(v) == len(vector)]
            if any(score >= duplicate_at for _, score in rank(vector, stored, limit=1)):
                return None
        return self.repo.add(content, pinned=pinned, embedding=vector)

    async def edit_fact(self, memory_id: str, content: str) -> None:
        """Change a memory's text and recompute its embedding so searches match what it now says."""
        content = content.strip()
        if content:
            self.repo.set_content(memory_id, content, await self._embed(content))

    # --- tools --------------------------------------------------------------------------------------------

    def get_tools(self) -> list[Tool]:
        return [tool_from_function(self.remember), tool_from_function(self.recall)]

    async def remember(self, fact: str, always_include: bool = False) -> str:
        """Remember a fact about the user when they explicitly ask you to remember something. (Facts they merely mention are picked up automatically afterwards, so do not call this for those.)

        Args:
            fact: One short, self-contained sentence about the user, e.g. "The user prefers metric units."
            always_include: True only for core facts (name, role, home town) worth having in every conversation.
        """
        if not fact.strip():
            raise ToolError("Nothing to remember: the fact was empty.")
        saved = await self.save_fact(fact, pinned=always_include)
        return "Remembered." if saved else "I already remember something very similar."

    async def recall(self, query: str) -> str:
        """Search what you remember about the user.

        Args:
            query: What to look for, e.g. "favourite food" or "where the user lives".
        """
        found = await self.search(query, limit=5)
        pinned = self.repo.list_pinned()
        lines = [f"- {m.content} (always known)" for m in pinned]
        lines += [f"- {m.content} (match {score:.2f})" for m, score in found if score >= 0.3]
        return "\n".join(lines) or "Nothing is remembered yet."
