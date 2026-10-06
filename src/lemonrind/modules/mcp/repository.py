"""Storing the MCP servers you have configured.

A server is either a **program to start** (``stdio``: the app launches ``command`` with ``args`` and talks to it
through its standard input and output) or a **web address** (``http``: a server that is already running
somewhere, reached with the MCP "streamable HTTP" protocol). Either way it may need secrets: an API key in an
environment variable for a started program, a token in a header for a web server.

**Those secrets are stored as plain text in the database**, like the .NET editions do. The data folder is
private to you, but treat it that way: do not share ``lemonrind.db``, and use keys that can be revoked.

Python ideas used here:

* ``json.dumps`` / ``json.loads`` for list and dictionary columns.
* A frozen dataclass for a row, and tuples (not lists) for its immutable fields.
"""

from __future__ import annotations

import builtins
import json
import sqlite3
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

from lemonrind.storage import Database

type Transport = Literal["stdio", "http"]


class DuplicateServerError(ValueError):
    """A server with that name already exists."""


@dataclass(frozen=True, slots=True)
class ServerDefinition:
    """What it takes to describe a server (no id yet): what the importer produces and the repository stores."""

    name: str
    transport: Transport
    command: str = ""
    args: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    url: str = ""
    headers: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class McpServerConfig:
    id: str
    name: str
    transport: Transport
    command: str
    args: tuple[str, ...]
    env: Mapping[str, str]
    url: str
    headers: Mapping[str, str]
    enabled: bool
    require_approval: bool  # ask before each run of this server's tools
    created_at: datetime

    @property
    def summary(self) -> str:
        """One line saying how the server is reached, with no secrets in it."""
        if self.transport == "http":
            return self.url
        return " ".join([self.command, *self.args])


def now_utc() -> datetime:
    return datetime.now(UTC)


_SELECT = (
    "SELECT id, name, transport, command, args_json, env_json, url, headers_json, enabled,"
    " require_approval, created_at FROM mcp_servers"
)


def _from_row(row: sqlite3.Row) -> McpServerConfig:
    return McpServerConfig(
        id=row["id"],
        name=row["name"],
        transport=row["transport"],
        command=row["command"],
        args=tuple(json.loads(row["args_json"])),
        env=json.loads(row["env_json"]),
        url=row["url"],
        headers=json.loads(row["headers_json"]),
        enabled=bool(row["enabled"]),
        require_approval=bool(row["require_approval"]),
        created_at=datetime.fromisoformat(row["created_at"]),
    )


class McpServerRepository:
    def __init__(self, db: Database, clock: Callable[[], datetime] = now_utc) -> None:
        self._db = db
        self._clock = clock

    def add(
        self, definition: ServerDefinition, *, require_approval: bool = True
    ) -> McpServerConfig:
        """Store a new server, enabled. New servers ask before running tools unless told otherwise."""
        server_id = str(uuid.uuid4())
        try:
            with self._db.transaction() as conn:
                conn.execute(
                    "INSERT INTO mcp_servers (id, name, transport, command, args_json, env_json, url,"
                    " headers_json, enabled, require_approval, created_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)",
                    (
                        server_id,
                        definition.name.strip(),
                        definition.transport,
                        definition.command,
                        json.dumps(list(definition.args)),
                        json.dumps(dict(definition.env)),
                        definition.url,
                        json.dumps(dict(definition.headers)),
                        int(require_approval),
                        self._clock().isoformat(),
                    ),
                )
        except sqlite3.IntegrityError as error:  # the UNIQUE constraint on name
            raise DuplicateServerError(
                f"A server called '{definition.name}' already exists."
            ) from error
        config = self.get(server_id)
        assert config is not None
        return config

    def get(self, server_id: str) -> McpServerConfig | None:
        row = self._db.conn.execute(f"{_SELECT} WHERE id = ?", (server_id,)).fetchone()
        return _from_row(row) if row else None

    def find(self, name: str) -> McpServerConfig | None:
        row = self._db.conn.execute(f"{_SELECT} WHERE name = ?", (name.strip(),)).fetchone()
        return _from_row(row) if row else None

    def list(self) -> builtins.list[McpServerConfig]:
        rows = self._db.conn.execute(f"{_SELECT} ORDER BY created_at, rowid")
        return [_from_row(row) for row in rows]

    def set_enabled(self, server_id: str, enabled: bool) -> None:
        self._update(server_id, "enabled", int(enabled))

    def set_require_approval(self, server_id: str, require: bool) -> None:
        self._update(server_id, "require_approval", int(require))

    def delete(self, server_id: str) -> None:
        with self._db.transaction() as conn:
            conn.execute("DELETE FROM mcp_servers WHERE id = ?", (server_id,))

    def _update(self, server_id: str, column: str, value: int) -> None:
        # ``column`` is one of the two fixed names below, never user input; the check makes that a rule.
        if column not in ("enabled", "require_approval"):
            raise ValueError(f"Not an updatable column: {column!r}")
        with self._db.transaction() as conn:
            conn.execute(
                f"UPDATE mcp_servers SET {column} = ? WHERE id = ?",  # nosec B608
                (value, server_id),
            )


__all__ = [
    "DuplicateServerError",
    "McpServerConfig",
    "McpServerRepository",
    "ServerDefinition",
]
