"""Storing memories (lasting facts about the user) in SQLite.

Each memory is a short sentence, a ``pinned`` flag, and (when an embedding model was available) the
embedding vector of its text. Pinned memories are always given to the model; the others are found by
meaning when a message looks related (see ``module.py``).

Python ideas used here:

* The same repository pattern as ``chats/repository.py``: plain methods around SQL, parameters only.
* ``numpy`` arrays read back from ``BLOB`` columns (``vectors.from_blob``).
"""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

from lemonrind.storage import Database
from lemonrind.vectors import from_blob, to_blob


@dataclass(frozen=True, slots=True)
class Memory:
    id: str
    content: str
    pinned: bool
    created_at: datetime
    has_embedding: bool  # False when no embedding model was available yet when it was saved


def now_utc() -> datetime:
    return datetime.now(UTC)


_SELECT = (
    "SELECT id, content, pinned, created_at, embedding IS NOT NULL AS has_embedding FROM memories"
)


def _memory_from_row(row: sqlite3.Row) -> Memory:
    return Memory(
        id=row["id"],
        content=row["content"],
        pinned=bool(row["pinned"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        has_embedding=bool(row["has_embedding"]),
    )


class MemoryRepository:
    def __init__(self, db: Database, clock: Callable[[], datetime] = now_utc) -> None:
        self._db = db
        self._clock = clock

    def add(
        self, content: str, *, pinned: bool, embedding: Sequence[float] | None = None
    ) -> Memory:
        memory_id = str(uuid.uuid4())
        now = self._clock().isoformat()
        blob = to_blob(embedding) if embedding is not None else None
        with self._db.transaction() as conn:
            conn.execute(
                "INSERT INTO memories (id, content, pinned, embedding, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (memory_id, content.strip(), int(pinned), blob, now, now),
            )
        memory = self.get(memory_id)
        assert memory is not None
        return memory

    def get(self, memory_id: str) -> Memory | None:
        row = self._db.conn.execute(f"{_SELECT} WHERE id = ?", (memory_id,)).fetchone()
        return _memory_from_row(row) if row else None

    def list_all(self) -> list[Memory]:
        """Pinned memories first, then the rest, newest first within each group."""
        rows = self._db.conn.execute(f"{_SELECT} ORDER BY pinned DESC, created_at DESC, rowid DESC")
        return [_memory_from_row(row) for row in rows]

    def list_pinned(self) -> list[Memory]:
        rows = self._db.conn.execute(f"{_SELECT} WHERE pinned = 1 ORDER BY created_at, rowid")
        return [_memory_from_row(row) for row in rows]

    def with_embeddings(self) -> list[tuple[Memory, np.ndarray]]:
        """Every memory that has an embedding, with it (for similarity search)."""
        rows = self._db.conn.execute(
            "SELECT id, content, pinned, created_at, 1 AS has_embedding, embedding"
            " FROM memories WHERE embedding IS NOT NULL"
        )
        return [(_memory_from_row(row), from_blob(row["embedding"])) for row in rows]

    def without_embeddings(self) -> list[Memory]:
        rows = self._db.conn.execute(f"{_SELECT} WHERE embedding IS NULL")
        return [_memory_from_row(row) for row in rows]

    def set_content(self, memory_id: str, content: str, embedding: Sequence[float] | None) -> None:
        """Change a memory's text. Its old embedding described the old text, so it is replaced
        (or cleared, when there is no new one to give)."""
        blob = to_blob(embedding) if embedding is not None else None
        self._update(memory_id, "content = ?, embedding = ?", (content.strip(), blob))

    def set_embedding(self, memory_id: str, embedding: Sequence[float]) -> None:
        self._update(memory_id, "embedding = ?", (to_blob(embedding),))

    def set_pinned(self, memory_id: str, pinned: bool) -> None:
        self._update(memory_id, "pinned = ?", (int(pinned),))

    def delete(self, memory_id: str) -> None:
        with self._db.transaction() as conn:
            conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))

    def _update(self, memory_id: str, assignment: str, params: tuple) -> None:
        # ``assignment`` is one of the fixed pieces of SQL below; the values travel as parameters.
        if assignment not in ("content = ?, embedding = ?", "embedding = ?", "pinned = ?"):
            raise ValueError(f"Not an allowed update: {assignment!r}")
        with self._db.transaction() as conn:
            conn.execute(
                f"UPDATE memories SET {assignment}, updated_at = ? WHERE id = ?",  # nosec B608
                (*params, self._clock().isoformat(), memory_id),
            )
