"""Storing knowledge bases: one SQLite file each, in the ``knowledge`` folder of the data folder.

Three levels, as in the .NET editions:

* a **knowledge base** is a named collection ("Work docs");
* it has **sources**: a file, a folder (one source, however many files are inside), a web page, or pasted text;
* each source is split into **chunks**, each stored with its embedding.

**Why a file per knowledge base.** Building one can take a long time (reading, splitting and embedding every
document), and the result is worth keeping and reusing: on another computer, or by someone else. So each knowledge
base is a single self-contained ``.kb`` file (an ordinary SQLite database) holding its name, description, sources
and chunks. To use one elsewhere, copy the file in with "Add knowledge base from file" (``import_file``); to share
one, ``export_copy`` writes a consistent copy. Nothing about a knowledge base lives in the main database except
which one a chat has attached (by its id, which stays the same everywhere).

**The embedding model travels with it.** An embedding only means something to the model that made it, so the file
records which model (and how many dimensions). The app refuses to mix models in one knowledge base and offers to
*re-embed* it: the chunk text is stored, so that needs the embedding model but not the original documents.

A source has a *status*: ``ingesting`` while it is being read and embedded, ``ready`` when searchable, or ``failed``
(with the reason). Only chunks of ready sources are searched, so a half-finished ingestion is never visible to a chat.

Python ideas used here:

* A *facade*: ``KnowledgeRepository`` keeps the same methods as when everything was in one database, and works out
  which file each call belongs to, so the module, the ingestor and the screens did not need rewriting.
* ``sqlite3``'s backup API (``Connection.backup``) to copy a live database file consistently.
* Opening a file read-only (``?mode=ro`` in a URI) to *look* before touching it.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np

from lemonrind.storage import Database
from lemonrind.vectors import from_blob, to_blob

logger = logging.getLogger(__name__)

type SourceType = Literal["file", "folder", "website", "text"]
type SourceStatus = Literal["ingesting", "ready", "failed"]

EXTENSION = ".kb"
FOLDER_NAME = "knowledge"  # inside the data folder

# The schema of one knowledge base file. Like the main database it is a list: add an entry to change it, never edit one.
KB_MIGRATIONS: list[str] = [
    """
    CREATE TABLE meta (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );

    CREATE TABLE sources (
        id           TEXT PRIMARY KEY,
        source_type  TEXT NOT NULL CHECK (source_type IN ('file', 'folder', 'website', 'text')),
        reference    TEXT,
        display_name TEXT NOT NULL,
        status       TEXT NOT NULL CHECK (status IN ('ingesting', 'ready', 'failed')),
        error        TEXT,
        chunk_count  INTEGER NOT NULL DEFAULT 0,
        created_at   TEXT NOT NULL
    );

    CREATE TABLE chunks (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        source_id   TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
        chunk_index INTEGER NOT NULL,
        label       TEXT NOT NULL DEFAULT '',  -- which file inside a folder source this came from
        content     TEXT NOT NULL,
        embedding   BLOB NOT NULL
    );
    CREATE INDEX ix_chunks_source ON chunks(source_id);
    """,
]


class DuplicateNameError(ValueError):
    """A knowledge base with that name already exists."""


class KnowledgeFileError(ValueError):
    """A file cannot be used as a knowledge base (not one, made by a newer version, or already added). Fit to show."""


class ModelMismatchError(Exception):
    """The embedding model in use differs from the one the knowledge base was built with. Fit to show."""


@dataclass(frozen=True, slots=True)
class KnowledgeBase:
    id: str
    name: str
    created_at: datetime
    description: str = ""
    embedding_model: str = (
        ""  # "" = not known (nothing embedded yet, or built before this was recorded)
    )
    embedding_dim: int = 0  # 0 = not known
    path: Path | None = None  # the .kb file


@dataclass(frozen=True, slots=True)
class Source:
    id: str
    kb_id: str
    source_type: SourceType
    reference: str | None  # the file or folder path, or the web address
    display_name: str
    status: SourceStatus
    error: str | None
    chunk_count: int
    created_at: datetime


@dataclass(frozen=True, slots=True)
class Chunk:
    content: str
    source_name: (
        str  # which source it came from, shown to the model so it can say where an answer is from
    )


@dataclass(frozen=True, slots=True)
class ImportPlan:
    """What adding a knowledge base file will do: its id, its name (changed if taken) and where the copy goes."""

    id: str
    original_name: str
    name: str
    destination: Path


@dataclass(frozen=True, slots=True)
class KnowledgeStats:
    chunks: int
    size_bytes: int


def now_utc() -> datetime:
    return datetime.now(UTC)


def mismatch_message(kb: KnowledgeBase, current_model: str) -> str:
    return (
        f"The knowledge base '{kb.name}' was built with the embedding model '{kb.embedding_model}', but "
        f"'{current_model}' is in use now. Their numbers cannot be compared. Re-embed the knowledge base "
        "(Settings > Knowledge bases) to use it with the current model."
    )


def file_name_for(kb_id: str, name: str) -> str:
    """``work-docs-1a2b3c4d.kb``: readable, and unique because of the start of the id."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "knowledge-base"
    return f"{slug}-{kb_id[:8]}{EXTENSION}"


_SOURCE_SELECT = "SELECT id, source_type, reference, display_name, status, error, chunk_count, created_at FROM sources"


def _peek(path: Path) -> dict[str, str]:
    """Read a file's ``meta`` table without changing anything, or explain why it is not a usable knowledge base."""
    try:
        conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    except sqlite3.Error as error:
        raise KnowledgeFileError(f"'{path.name}' could not be opened ({error}).") from error
    try:
        conn.row_factory = sqlite3.Row
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        try:
            rows = conn.execute("SELECT key, value FROM meta").fetchall()
            conn.execute("SELECT 1 FROM sources LIMIT 1")
            conn.execute("SELECT 1 FROM chunks LIMIT 1")
        except sqlite3.Error as error:
            raise KnowledgeFileError(f"'{path.name}' is not a knowledge base file.") from error
        if version > len(KB_MIGRATIONS):
            raise KnowledgeFileError(
                f"'{path.name}' was made by a newer version of this app (its format is {version}, this one "
                f"understands up to {len(KB_MIGRATIONS)}). Update the app to use it."
            )
        meta = {row["key"]: row["value"] for row in rows}
    except sqlite3.DatabaseError as error:  # not a database at all
        raise KnowledgeFileError(
            f"'{path.name}' is not a knowledge base file ({error})."
        ) from error
    finally:
        conn.close()
    if not meta.get("id") or not meta.get("name"):
        raise KnowledgeFileError(
            f"'{path.name}' is not a knowledge base file (it has no id or name)."
        )
    return meta


def copy_database(source: Path, destination: Path) -> None:
    """Copy a SQLite database file consistently with SQLite's own backup, even if the app is using it.

    It opens its own connections, so it can run on a worker thread (a big knowledge base can take a while) without
    touching the app's. ``destination`` is replaced.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    origin = sqlite3.connect(f"{source.resolve().as_uri()}?mode=ro", uri=True, timeout=30)
    try:
        copy = sqlite3.connect(destination)
        try:
            origin.backup(copy)
        finally:
            copy.close()
    finally:
        origin.close()


class _KbFile:
    """One open knowledge base file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        # wal=False keeps it a single file with no companions, so it can be copied as it is
        self.db = Database(path, KB_MIGRATIONS, wal=False)

    def meta(self) -> dict[str, str]:
        return {
            row["key"]: row["value"] for row in self.db.conn.execute("SELECT key, value FROM meta")
        }

    def set_meta(self, **values: str) -> None:
        with self.db.transaction() as conn:
            conn.executemany(
                "INSERT INTO meta (key, value) VALUES (?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                list(values.items()),
            )

    def knowledge_base(self) -> KnowledgeBase:
        meta = self.meta()
        return KnowledgeBase(
            id=meta["id"],
            name=meta["name"],
            created_at=datetime.fromisoformat(meta.get("created_at", "1970-01-01T00:00:00+00:00")),
            description=meta.get("description", ""),
            embedding_model=meta.get("embedding_model", ""),
            embedding_dim=int(meta.get("embedding_dim", "0") or 0),
            path=self.path,
        )

    def close(self) -> None:
        self.db.close()


def _source_from_row(kb_id: str, row: sqlite3.Row) -> Source:
    return Source(
        id=row["id"],
        kb_id=kb_id,
        source_type=row["source_type"],
        reference=row["reference"],
        display_name=row["display_name"],
        status=row["status"],
        error=row["error"],
        chunk_count=row["chunk_count"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


class KnowledgeRepository:
    def __init__(self, folder: Path, clock: Callable[[], datetime] = now_utc) -> None:
        self._folder = folder
        self._clock = clock
        self._files: dict[str, _KbFile] = {}  # by knowledge base id
        self._sync()

    def close(self) -> None:
        for file in self._files.values():
            file.close()
        self._files.clear()

    # --- finding the files -----------------------------------------------------------------------------

    def _sync(self) -> None:
        """Open knowledge base files that have appeared in the folder, and forget ones that have gone."""
        self._folder.mkdir(parents=True, exist_ok=True)
        for kb_id, file in list(self._files.items()):
            if not file.path.exists():
                file.close()
                del self._files[kb_id]
        known = {file.path.resolve() for file in self._files.values()}
        for path in sorted(self._folder.glob(f"*{EXTENSION}")):
            if path.resolve() in known:
                continue
            try:
                meta = _peek(path)
                if meta["id"] in self._files:
                    logger.warning(
                        "Skipped %s: another file already holds knowledge base %s",
                        path.name,
                        meta["id"],
                    )
                    continue
                self._files[meta["id"]] = _KbFile(path)
            except (KnowledgeFileError, sqlite3.Error) as error:
                logger.warning("Skipped %s in the knowledge folder: %s", path.name, error)

    def _file(self, kb_id: str) -> _KbFile:
        file = self._files.get(kb_id)
        if file is None:
            self._sync()
            file = self._files.get(kb_id)
        if file is None:
            raise KeyError(kb_id)
        return file

    def _file_of_source(self, source_id: str) -> _KbFile | None:
        for file in self._files.values():
            if file.db.conn.execute("SELECT 1 FROM sources WHERE id = ?", (source_id,)).fetchone():
                return file
        return None

    # --- knowledge bases -------------------------------------------------------------------------------

    def create_kb(self, name: str, description: str = "") -> KnowledgeBase:
        name = name.strip()
        if not name:
            raise ValueError("A knowledge base needs a name.")
        if self.find_kb(name) is not None:
            raise DuplicateNameError(f"A knowledge base called '{name}' already exists.")
        kb_id = str(uuid.uuid4())
        file = _KbFile(self._folder / file_name_for(kb_id, name))
        file.set_meta(
            id=kb_id,
            name=name,
            description=description.strip(),
            created_at=self._clock().isoformat(),
            embedding_model="",
            embedding_dim="0",
        )
        self._files[kb_id] = file
        return file.knowledge_base()

    def get_kb(self, kb_id: str) -> KnowledgeBase | None:
        try:
            return self._file(kb_id).knowledge_base()
        except KeyError:
            return None

    def find_kb(self, name: str) -> KnowledgeBase | None:
        wanted = name.strip().casefold()
        return next((kb for kb in self.list_kbs() if kb.name.casefold() == wanted), None)

    def list_kbs(self) -> list[KnowledgeBase]:
        self._sync()
        return sorted(
            (file.knowledge_base() for file in self._files.values()),
            key=lambda kb: kb.name.casefold(),
        )

    def delete_kb(self, kb_id: str) -> None:
        file = self._files.pop(kb_id, None)
        if file is None:
            return
        file.close()
        file.path.unlink(missing_ok=True)

    def stats(self, kb_id: str) -> KnowledgeStats:
        file = self._file(kb_id)
        chunks = file.db.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return KnowledgeStats(chunks, file.path.stat().st_size if file.path.exists() else 0)

    # --- moving a knowledge base between computers ------------------------------------------------------

    def export_copy(self, kb_id: str, destination: Path) -> Path:
        """Write a consistent, self-contained copy of a knowledge base to ``destination`` (overwriting it)."""
        copy_database(self._file(kb_id).path, destination)
        return destination

    def plan_import(self, source: Path) -> ImportPlan:
        """Check a ``.kb`` file and decide where its copy will go and what it will be called (nothing is changed yet).

        Raises ``KnowledgeFileError`` if it is not a usable knowledge base or this one is already here. If another
        knowledge base already has its name, the copy is called "Name (2)" rather than refused.
        """
        meta = _peek(source)
        self._sync()
        if meta["id"] in self._files:
            raise KnowledgeFileError(
                f"The knowledge base '{self._files[meta['id']].knowledge_base().name}' is already added."
            )
        name = meta["name"]
        if self.find_kb(name) is not None:
            number = 2
            while self.find_kb(f"{name} ({number})") is not None:
                number += 1
            name = f"{name} ({number})"
        return ImportPlan(
            meta["id"], meta["name"], name, self._folder / file_name_for(meta["id"], name)
        )

    def finish_import(self, plan: ImportPlan) -> KnowledgeBase:
        """Open the copy made by ``copy_database(source, plan.destination)`` and add it to the list."""
        file = _KbFile(plan.destination)  # also brings an older file's schema up to date
        if plan.name != plan.original_name:
            file.set_meta(name=plan.name)
        self._files[plan.id] = file
        return file.knowledge_base()

    def import_file(self, source: Path) -> KnowledgeBase:
        """Add a knowledge base from a ``.kb`` file by copying it into the knowledge folder (see ``plan_import``).

        A screen that must not freeze while a big file is copied does the three steps itself: ``plan_import``, then
        ``copy_database`` on a worker thread, then ``finish_import``.
        """
        plan = self.plan_import(source)
        copy_database(source, plan.destination)
        return self.finish_import(plan)

    def migrate_from_main_database(self, db: Database) -> int:
        """Move knowledge bases that earlier versions kept in the main database into files. Returns how many.

        Each is written completely before it is removed from the main database, and the id is kept, so a chat that
        had it attached still has. The embedding model is not known for these (it was not recorded then).
        """
        try:
            rows = db.conn.execute(
                "SELECT id, name, description, created_at FROM knowledge_bases"
            ).fetchall()
        except sqlite3.Error:
            return 0
        moved = 0
        for row in rows:
            if row["id"] not in self._files:
                path = self._folder / file_name_for(row["id"], row["name"])
                partial = path.with_suffix(".partial")
                partial.unlink(missing_ok=True)
                file = _KbFile(partial)
                chunks = db.conn.execute(
                    "SELECT source_id, chunk_index, label, content, embedding FROM knowledge_chunks WHERE kb_id = ? ORDER BY id",
                    (row["id"],),
                ).fetchall()
                with file.db.transaction() as conn:
                    conn.executemany(
                        "INSERT INTO sources (id, source_type, reference, display_name, status, error, chunk_count, created_at)"
                        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        [
                            tuple(source)
                            for source in db.conn.execute(
                                "SELECT id, source_type, reference, display_name, status, error, chunk_count, created_at"
                                " FROM knowledge_sources WHERE kb_id = ?",
                                (row["id"],),
                            )
                        ],
                    )
                    conn.executemany(
                        "INSERT INTO chunks (source_id, chunk_index, label, content, embedding) VALUES (?, ?, ?, ?, ?)",
                        [tuple(chunk) for chunk in chunks],
                    )
                dimension = len(chunks[0]["embedding"]) // 4 if chunks else 0  # 4 bytes per number
                file.set_meta(
                    id=row["id"],
                    name=row["name"],
                    description=row["description"] or "",
                    created_at=row["created_at"],
                    embedding_model="",
                    embedding_dim=str(dimension),
                )
                file.close()
                partial.replace(path)
            with db.transaction() as conn:
                conn.execute(
                    "DELETE FROM knowledge_bases WHERE id = ?", (row["id"],)
                )  # cascades to its sources
            moved += 1
        if moved:
            self._sync()
        return moved

    # --- sources ---------------------------------------------------------------------------------------

    def create_source(
        self, kb_id: str, source_type: SourceType, reference: str | None, display_name: str
    ) -> Source:
        """Register a source in the ``ingesting`` state; it becomes searchable when ``mark_ready`` is called."""
        file = self._file(kb_id)
        source_id = str(uuid.uuid4())
        with file.db.transaction() as conn:
            conn.execute(
                "INSERT INTO sources (id, source_type, reference, display_name, status, created_at)"
                " VALUES (?, ?, ?, ?, 'ingesting', ?)",
                (source_id, source_type, reference, display_name, self._clock().isoformat()),
            )
        source = self.get_source(source_id)
        assert source is not None
        return source

    def get_source(self, source_id: str) -> Source | None:
        file = self._file_of_source(source_id)
        if file is None:
            return None
        row = file.db.conn.execute(f"{_SOURCE_SELECT} WHERE id = ?", (source_id,)).fetchone()
        return _source_from_row(file.meta()["id"], row) if row else None

    def list_sources(self, kb_id: str) -> list[Source]:
        try:
            file = self._file(kb_id)
        except KeyError:
            return []
        rows = file.db.conn.execute(f"{_SOURCE_SELECT} ORDER BY created_at, rowid")
        return [_source_from_row(kb_id, row) for row in rows]

    def mark_ready(self, source_id: str, chunk_count: int) -> None:
        if file := self._file_of_source(source_id):
            with file.db.transaction() as conn:
                conn.execute(
                    "UPDATE sources SET status = 'ready', error = NULL, chunk_count = ? WHERE id = ?",
                    (chunk_count, source_id),
                )

    def mark_failed(self, source_id: str, error: str) -> None:
        """Record why a source failed and throw away any chunks it had already saved."""
        if file := self._file_of_source(source_id):
            with file.db.transaction() as conn:
                conn.execute("DELETE FROM chunks WHERE source_id = ?", (source_id,))
                conn.execute(
                    "UPDATE sources SET status = 'failed', error = ?, chunk_count = 0 WHERE id = ?",
                    (error, source_id),
                )

    def fail_interrupted(self) -> int:
        """Sources still ``ingesting`` when the app starts were cut off (it was closed or crashed)."""
        self._sync()
        total = 0
        for file in self._files.values():
            with file.db.transaction() as conn:
                conn.execute(
                    "DELETE FROM chunks WHERE source_id IN (SELECT id FROM sources WHERE status = 'ingesting')"
                )
                cursor = conn.execute(
                    "UPDATE sources SET status = 'failed', error = 'Interrupted: the app closed"
                    " before this finished.' WHERE status = 'ingesting'"
                )
                total += cursor.rowcount
        return total

    def delete_source(self, source_id: str) -> None:
        if file := self._file_of_source(source_id):
            with file.db.transaction() as conn:
                conn.execute("DELETE FROM sources WHERE id = ?", (source_id,))

    # --- chunks ----------------------------------------------------------------------------------------

    def add_chunks(
        self,
        kb_id: str,
        source_id: str,
        first_index: int,
        chunks: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        label: str = "",
        *,
        model: str = "",
    ) -> None:
        """Save chunks with their embeddings. ``model`` names the embedding model that made them: the first time it
        is stored with the knowledge base, after that it must match (``ModelMismatchError`` otherwise)."""
        file = self._file(kb_id)
        meta = file.meta()
        known = meta.get("embedding_model", "")
        if model and known and model != known:
            raise ModelMismatchError(mismatch_message(file.knowledge_base(), model))
        with file.db.transaction() as conn:
            conn.executemany(
                "INSERT INTO chunks (source_id, chunk_index, label, content, embedding) VALUES (?, ?, ?, ?, ?)",
                [
                    (source_id, first_index + offset, label, content, to_blob(vector))
                    for offset, (content, vector) in enumerate(zip(chunks, embeddings, strict=True))
                ],
            )
            if embeddings:
                conn.executemany(
                    "INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    [
                        ("embedding_model", model or known),
                        ("embedding_dim", str(len(embeddings[0]))),
                    ],
                )

    def chunk_contents(self, kb_id: str) -> list[tuple[int, str]]:
        """Every chunk's id and text (for re-embedding)."""
        rows = self._file(kb_id).db.conn.execute("SELECT id, content FROM chunks ORDER BY id")
        return [(row["id"], row["content"]) for row in rows]

    def replace_embeddings(
        self, kb_id: str, vectors: Sequence[tuple[int, Sequence[float]]], model: str
    ) -> None:
        """Give chunks new embeddings (all in one transaction) and record the model that made them."""
        file = self._file(kb_id)
        with file.db.transaction() as conn:
            conn.executemany(
                "UPDATE chunks SET embedding = ? WHERE id = ?",
                [(to_blob(vector), chunk_id) for chunk_id, vector in vectors],
            )
            dimension = len(vectors[0][1]) if vectors else 0
            conn.executemany(
                "INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                [("embedding_model", model), ("embedding_dim", str(dimension))],
            )

    def file_labels(self, source_id: str) -> list[str]:
        """The files a folder source was made from (empty for other kinds of source)."""
        file = self._file_of_source(source_id)
        if file is None:
            return []
        rows = file.db.conn.execute(
            "SELECT DISTINCT label FROM chunks WHERE source_id = ? AND label != '' ORDER BY label",
            (source_id,),
        )
        return [row["label"] for row in rows]

    def searchable_chunks(self, kb_id: str) -> list[tuple[Chunk, np.ndarray]]:
        """Every chunk of the knowledge base's *ready* sources, with its embedding."""
        try:
            file = self._file(kb_id)
        except KeyError:
            return []
        rows = file.db.conn.execute(
            "SELECT c.content, c.embedding, COALESCE(NULLIF(c.label, ''), s.display_name) AS name"
            " FROM chunks c JOIN sources s ON s.id = c.source_id"
            " WHERE s.status = 'ready' ORDER BY c.id"
        )
        return [(Chunk(row["content"], row["name"]), from_blob(row["embedding"])) for row in rows]
