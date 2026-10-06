"""One SQLite database file, opened once and shared by the repositories.

Python ideas used here:

* The standard library's ``sqlite3`` module - no installation, no server, just a file.
* ``sqlite3.Row`` - result rows you can read by column name (``row["title"]``) instead of by position.
* ``@contextmanager`` - turns a function with a single ``yield`` into something usable in a ``with`` block.
* ``with connection:`` - SQLite's built-in "transaction block": everything inside is committed together
  when the block ends normally, or undone if an exception is raised.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from lemonrind.storage.migrations import apply_migrations

# SQLite's special name for a throwaway database that lives only in RAM (used by tests).
IN_MEMORY = ":memory:"


class Database:
    """Owns the connection. Create it once at start-up and call ``close()`` (or use ``with``) at the end."""

    def __init__(
        self,
        path: Path | str = IN_MEMORY,
        migrations: Sequence[str] | None = None,
        *,
        wal: bool = True,
    ) -> None:
        """Open (and create or bring up to date) a database file.

        ``migrations`` is the schema to apply: the main database's by default, or another list for a database
        with its own schema (a knowledge base file). ``wal=False`` keeps the file a single self-contained file
        (no ``-wal`` and ``-shm`` companions), which is what a file you intend to copy needs.
        """
        in_memory = str(path) == IN_MEMORY
        if not in_memory:
            Path(path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        # SQLite does not enforce foreign keys unless asked, and the setting is per connection.
        self._conn.execute("PRAGMA foreign_keys = ON")
        if not in_memory and wal:
            # Write-ahead logging: readers do not block the writer, and a crash cannot corrupt the file.
            self._conn.execute("PRAGMA journal_mode = WAL")
        self.schema_version = apply_migrations(self._conn, migrations)

    @property
    def conn(self) -> sqlite3.Connection:
        """The raw connection, for read queries. Use ``transaction()`` for anything that writes."""
        return self._conn

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Group several statements so they succeed or fail together::

        with db.transaction() as conn:
            conn.execute("INSERT ...")
            conn.execute("UPDATE ...")   # if this raises, the INSERT above is undone too
        """
        # Used as a context manager the connection commits on success and rolls back on an exception;
        # it does not close the connection.
        with self._conn:
            yield self._conn

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
