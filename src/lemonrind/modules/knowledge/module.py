"""The Knowledge bases module: let a chat draw on your own documents.

You build **knowledge bases** (collections of files, folders, web pages and pasted text), attach one to a chat,
and for each message you send, the module finds the few chunks of your documents that best match it and
adds them to that message. This pattern is called *retrieval-augmented generation* (RAG): the model does not
"know" your documents, it is shown the relevant excerpts at the moment it needs them.

It is the same mechanism as memory (embed the message, rank stored vectors by similarity, keep the best
ones above a threshold), with two differences:

* the thing searched is chunks of documents, chosen per chat (``chat.options["knowledge_base"]``), where
  memory searches everything you have ever told it;
* the module **always** tells the model what sources the knowledge base contains. Similarity search cannot
  answer "what files are in here?", because that question is not close in meaning to any chunk's content.

The module offers no tools: the excerpts simply appear in the model's input. (Do not confuse a knowledge
base with the File system module: files there can be read by tools, a knowledge base cannot, and the added
text says so, because a model will otherwise try to browse for it.)

Python ideas used here:

* Reuse: ``vectors.rank`` and ``Embedder`` are exactly what memory uses.
* Keyword-only ``chat:`` parameter in the hook, so the module receives per-chat state without keeping any.
"""

from __future__ import annotations

from lemonrind.config import Settings
from lemonrind.embeddings import Embedder, EmbeddingUnavailableError
from lemonrind.modules.base import ChatContext, Module
from lemonrind.modules.knowledge.ingestion import KnowledgeIngestor
from lemonrind.modules.knowledge.repository import (
    Chunk,
    KnowledgeBase,
    KnowledgeRepository,
    mismatch_message,
)
from lemonrind.modules.tool import Tool
from lemonrind.vectors import rank

MAX_FILES_LISTED = 20  # how many file names of a folder source are listed to the model

OPTION_KEY = "knowledge_base"  # the per-chat option that holds the attached knowledge base's id

NONE_ATTACHED_NOTE = (
    "No knowledge base is attached to this chat. If asked what is in the knowledge base, say that none is "
    "attached, rather than looking in the file system: a knowledge base is not a folder, and file tools "
    "cannot see one."
)


class KnowledgeModule(Module):
    name = "Knowledge bases"
    config_key = "knowledge"
    description = "Attach a knowledge base (your files, folders, web pages, notes) to a chat."

    def __init__(
        self,
        settings: Settings,
        repo: KnowledgeRepository,
        embedder: Embedder,
        ingestor: KnowledgeIngestor,
    ) -> None:
        super().__init__(settings)
        self.repo = repo
        self.ingestor = ingestor
        self._embedder = embedder
        self.reembedding: set[str] = (
            set()
        )  # knowledge bases being re-embedded now (screens show it)

    async def on_startup(self) -> None:
        self.repo.fail_interrupted()  # a source left "ingesting" was cut off by the app closing

    def get_tools(self) -> list[Tool]:
        return []

    # --- the hook ----------------------------------------------------------------------------------------

    async def turn_context(self, user_text: str, *, chat: ChatContext) -> str:
        kb_id = chat.options.get(OPTION_KEY)
        kb = self.repo.get_kb(kb_id) if kb_id else None
        if kb is None:
            # Said every time the module is on, not only when the question mentions it (as in the other editions):
            # silence would leave the model to guess what "the knowledge base" means, and it could guess the file
            # system's workspace. The cost is a couple of dozen tokens in the per-message text.
            return NONE_ATTACHED_NOTE

        problem = await self.compatibility(kb)
        if problem:
            return (
                f"A knowledge base called '{kb.name}' is attached to this chat, but it cannot be searched "
                f"right now: {problem}"
            )
        ready = [s for s in self.repo.list_sources(kb.id) if s.status == "ready"]
        if not ready:
            return (
                f"A knowledge base called '{kb.name}' is attached to this chat, but it has no sources "
                "ready yet, so it has nothing to offer."
            )
        sections = [
            f"This chat has a knowledge base called '{kb.name}' attached. It is separate from any file "
            "system access you have: it is not a folder you can browse, and file tools cannot see it. "
            "It contains these sources:\n"
            + "\n".join(self._describe(source) for source in ready)
            + "\nTreat this list as authoritative for what the knowledge base contains."
        ]

        config = self.settings.modules.knowledge
        found = await self.search(
            kb.id, user_text, limit=config.max_chunks, min_similarity=config.min_similarity
        )
        if found:
            excerpts = "\n---\n".join(
                f"[Source: {chunk.source_name}]\n{chunk.content}" for chunk, _ in found
            )
            sections.append(
                "Relevant excerpts from the knowledge base (use them when they answer the question, and "
                "say which source you used):\n" + excerpts
            )
        return "\n\n".join(sections)

    async def current_model(self) -> str | None:
        """The embedding model in use now, or ``None`` if there is none (or Lemonade cannot be reached)."""
        try:
            return await self._embedder.model_name()
        except EmbeddingUnavailableError:
            return None

    async def compatibility(self, kb: KnowledgeBase) -> str:
        """Why this knowledge base cannot be searched with the current embedding model, or ``""`` if it can.

        A knowledge base whose model is not known (built before it was recorded) is allowed: search then compares
        the size of the vectors, which catches the usual mismatch.
        """
        if kb.id in self.reembedding:
            return "it is being re-embedded at the moment, which takes a while."
        current = await self.current_model()
        if current and kb.embedding_model and kb.embedding_model != current:
            return mismatch_message(kb, current)
        return ""

    async def reembed(self, kb_id: str) -> int:
        """Embed every chunk again with the current embedding model. Returns how many chunks were done.

        The chunk text is stored in the knowledge base file, so the original documents are not needed. The new
        vectors are all computed first and swapped in together, so a failure part-way leaves it as it was.
        """
        if kb_id in self.reembedding:
            return 0
        self.reembedding.add(kb_id)
        try:
            contents = self.repo.chunk_contents(kb_id)
            model = await self._embedder.model_name()
            vectors = await self._embedder.embed_many([text for _, text in contents])
            self.repo.replace_embeddings(
                kb_id,
                [
                    (chunk_id, vector)
                    for (chunk_id, _), vector in zip(contents, vectors, strict=True)
                ],
                model,
            )
            return len(contents)
        finally:
            self.reembedding.discard(kb_id)

    def _describe(self, source) -> str:
        """One line of the source list; a folder also lists the files inside it."""
        files = self.repo.file_labels(source.id) if source.source_type == "folder" else []
        if not files:
            return f"- {source.display_name}"
        shown = ", ".join(files[:MAX_FILES_LISTED])
        more = f" and {len(files) - MAX_FILES_LISTED} more" if len(files) > MAX_FILES_LISTED else ""
        return f"- {source.display_name} (folder containing: {shown}{more})"

    async def search(
        self, kb_id: str, text: str, *, limit: int, min_similarity: float = 0.0
    ) -> list[tuple[Chunk, float]]:
        """The chunks of one knowledge base closest in meaning to ``text``, best first."""
        kb = self.repo.get_kb(kb_id)
        if kb is None or await self.compatibility(kb):
            return []
        query = await self._embedder.embed_one(text)
        if query is None:
            return []
        stored = [
            (chunk, vector)
            for chunk, vector in self.repo.searchable_chunks(kb_id)
            if len(vector) == len(query)  # vectors from another embedding model cannot be compared
        ]
        ranked = rank(query, [vector for _, vector in stored], limit=limit)
        return [(stored[i][0], score) for i, score in ranked if score >= min_similarity]
