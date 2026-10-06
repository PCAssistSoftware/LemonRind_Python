"""Ingestion: turning a file, folder, web page or pasted text into searchable chunks.

The pipeline is the same for every kind of source: **get the text, chunk it, embed the chunks, save them**.
Only the first step differs. ``_ingest`` is that shared pipeline; the ``add_*`` methods each supply their own
way of getting the text.

The source is registered first, in the ``ingesting`` state, so a screen that lists sources shows it at once,
and it ends as ``ready`` or ``failed`` with a readable reason. Nothing is raised to the caller for ordinary
failures (an unreadable file, no embedding model, a dead web address): those are recorded on the source,
which is what the person looking at the screen needs to see.

A **folder is one source**, however many files it holds, so it appears as one entry and is deleted in one
action. Each file in it is read in its own ``try``, so one corrupt PDF does not sink the other forty-nine.

Python ideas used here:

* ``asyncio.to_thread`` for the blocking work of reading and parsing files (PDF parsing can take seconds and
  would freeze every other tab if run on the main thread).
* Passing a *function* as an argument (``get_text``) to share one pipeline between four sources.
* ``Path.rglob`` for walking a folder tree.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path

import httpx2 as httpx

from lemonrind.config import Settings
from lemonrind.embeddings import Embedder
from lemonrind.lemonade.client import LemonadeError
from lemonrind.modules.knowledge.chunker import chunk_text
from lemonrind.modules.knowledge.extract import ExtractionError, extract_text, is_supported
from lemonrind.modules.knowledge.repository import (
    KnowledgeRepository,
    ModelMismatchError,
    Source,
    SourceType,
    mismatch_message,
)
from lemonrind.modules.knowledge.webpage import WebPageError, fetch_page_text

logger = logging.getLogger(__name__)

MAX_FOLDER_FILES = 1000  # a sane outer bound against pointing at a huge folder by accident


class KnowledgeIngestor:
    def __init__(
        self,
        settings: Settings,
        repo: KnowledgeRepository,
        embedder: Embedder,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings
        self._repo = repo
        self._embedder = embedder
        self._transport = transport  # only tests pass one

    # --- the four kinds of source ---------------------------------------------------------------------

    async def add_file(self, kb_id: str, path: Path) -> Source:
        return await self._ingest(
            kb_id, "file", str(path), path.name, lambda: asyncio.to_thread(extract_text, path)
        )

    async def add_website(self, kb_id: str, url: str) -> Source:
        async def get_text() -> str:
            title, text = await fetch_page_text(url, transport=self._transport)
            return f"{title}\n\n{text}" if title else text

        return await self._ingest(kb_id, "website", url.strip(), url.strip(), get_text)

    async def add_text(self, kb_id: str, name: str, text: str) -> Source:
        async def get_text() -> str:
            return text

        return await self._ingest(kb_id, "text", None, name.strip() or "Pasted text", get_text)

    async def add_folder(self, kb_id: str, folder: Path) -> Source:
        source = self._repo.create_source(kb_id, "folder", str(folder), folder.name or str(folder))
        try:
            files = sorted(p for p in folder.rglob("*") if p.is_file() and is_supported(p))
            if not files:
                raise ExtractionError("No supported files were found in this folder.")
            if len(files) > MAX_FOLDER_FILES:
                raise ExtractionError(
                    f"The folder has {len(files):,} supported files; the limit is {MAX_FOLDER_FILES:,}."
                )
            total = skipped = 0
            for file in files:
                try:
                    text = await asyncio.to_thread(extract_text, file)
                except ExtractionError:
                    skipped += 1  # unreadable: skip it, keep the rest
                    continue
                label = file.relative_to(folder).as_posix()
                total += await self._save_chunks(kb_id, source.id, text, total, label)
            if total == 0:
                raise ExtractionError("None of the files in this folder could be read.")
            self._repo.mark_ready(source.id, total)
        except (ExtractionError, LemonadeError, ModelMismatchError) as error:
            self._repo.mark_failed(source.id, str(error))
        except Exception as error:  # a bug or an unexpected failure: still leave an honest record
            logger.exception("Folder ingestion failed")
            self._repo.mark_failed(source.id, f"{type(error).__name__}: {error}")
        return self._repo.get_source(source.id) or source

    # --- the shared pipeline ---------------------------------------------------------------------------

    async def _ingest(
        self,
        kb_id: str,
        source_type: SourceType,
        reference: str | None,
        display_name: str,
        get_text: Callable[[], Awaitable[str]],
    ) -> Source:
        source = self._repo.create_source(kb_id, source_type, reference, display_name)
        try:
            text = await get_text()
            count = await self._save_chunks(kb_id, source.id, text, 0)
            if count == 0:
                raise ExtractionError("The source had no readable text.")
            self._repo.mark_ready(source.id, count)
        except (ExtractionError, WebPageError, LemonadeError, ModelMismatchError) as error:
            self._repo.mark_failed(source.id, str(error))
        except Exception as error:
            logger.exception("Ingestion failed")
            self._repo.mark_failed(source.id, f"{type(error).__name__}: {error}")
        return self._repo.get_source(source.id) or source

    async def _save_chunks(
        self, kb_id: str, source_id: str, text: str, first_index: int, label: str = ""
    ) -> int:
        """Chunk, embed (in batches) and store ``text``. Returns how many chunks were saved."""
        config = self._settings.modules.knowledge
        chunks = chunk_text(text, config.chunk_words, config.chunk_overlap)
        if not chunks:
            return 0
        model = await self._embedder.model_name()
        kb = self._repo.get_kb(kb_id)
        if kb is not None and kb.embedding_model and kb.embedding_model != model:
            # Refuse before spending time embedding: mixing models in one knowledge base would ruin its search.
            raise ModelMismatchError(mismatch_message(kb, model))
        vectors = await self._embedder.embed_many(chunks)
        self._repo.add_chunks(kb_id, source_id, first_index, chunks, vectors, label, model=model)
        return len(chunks)
