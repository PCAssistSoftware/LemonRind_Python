"""Turning text into embedding vectors, shared by every feature that searches by meaning.

Memory (step 5) and knowledge bases (step 7) both need the same thing: pick an embedding model that Lemonade
has, and call it. Keeping that in one place means one rule for choosing the model and one place that knows
what to do when there is none.

Python ideas used here:

* A custom exception (``EmbeddingUnavailableError``) for "this cannot work right now", so callers choose
  between *failing loudly* (ingesting a document must say why it failed) and *carrying on quietly*
  (a memory lookup must never stop a chat).
* Batching: ``embed_many`` sends several texts in one request, far fewer round trips than one per text.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Protocol

from lemonrind.config import Settings
from lemonrind.lemonade.client import LemonadeError
from lemonrind.lemonade.models import ModelInfo
from lemonrind.lemonade.selection import pick_embedding_model

logger = logging.getLogger(__name__)

BATCH_SIZE = 32  # texts per request: big enough to be efficient, small enough not to time out


class EmbeddingUnavailableError(LemonadeError):
    """There is no embedding model to use (or it could not be reached). The message is fit to show."""


class EmbeddingClient(Protocol):
    """What an ``Embedder`` needs from the Lemonade client."""

    async def list_models(self, *, downloaded_only: bool = True) -> list[ModelInfo]: ...

    async def embed(self, texts: list[str], model: str) -> list[list[float]]: ...


class Embedder:
    def __init__(self, settings: Settings, client: EmbeddingClient) -> None:
        self._settings = settings
        self._client = client
        self._model: str | None = None  # looked up on first use, then remembered

    def reset(self) -> None:
        """Forget the chosen model (call after the set of models may have changed)."""
        self._model = None

    async def model_name(self) -> str:
        """The embedding model in use (looked up on first use). Raises ``EmbeddingUnavailableError`` if there is none."""
        return await self._model_name()

    async def _model_name(self) -> str:
        if self._model is None:
            models = await self._client.list_models()
            self._model = pick_embedding_model(self._settings.lemonade.embedding_model, models)
        if self._model is None:
            raise EmbeddingUnavailableError(
                "Lemonade has no embedding model downloaded (for example Qwen3-Embedding-0.6B-GGUF)."
            )
        return self._model

    async def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed every text, in batches. Raises ``LemonadeError`` if that cannot be done."""
        model = await self._model_name()
        vectors: list[list[float]] = []
        for start in range(0, len(texts), BATCH_SIZE):
            vectors += await self._client.embed(list(texts[start : start + BATCH_SIZE]), model)
        return vectors

    async def embed_one(self, text: str) -> list[float] | None:
        """One embedding, or ``None`` if unavailable. For best-effort callers such as memory lookups."""
        try:
            return (await self.embed_many([text]))[0]
        except LemonadeError:
            logger.warning("Could not embed text", exc_info=True)
            return None
