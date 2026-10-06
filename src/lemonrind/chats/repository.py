"""Saving and finding chats: sessions, messages, folders, tags and search.

Python ideas used here:

* SQL with **parameters** (``?`` and ``:name``) - values are passed separately from the SQL text, never
  pasted in. This is what stops SQL injection, exactly as with ``SqlParameter`` in ADO.NET.
* A **clock function** passed to the constructor, so tests can control "now" (injecting a *function*
  instead of an interface).
* Building a result list with a **list comprehension**.
* ``dataclasses.asdict`` and ``json`` to store a small structured value in one text column.

The repository is *synchronous*: SQLite calls on a local file take microseconds, so calling them straight
from async code is fine. (Anything slow, such as scanning thousands of embeddings later, will be moved
to a worker thread with ``asyncio.to_thread``.)
"""

from __future__ import annotations

import dataclasses
import json
import sqlite3
import uuid
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any

from lemonrind.attachment_markers import parse_image_marker
from lemonrind.chats.models import ChatSession, Folder, Role, SearchHit, StoredMessage
from lemonrind.chats.text import DEFAULT_TITLE, auto_title, build_fts_query, snippet_from_text
from lemonrind.chats.usage import ToolUse, UsageRequest
from lemonrind.lemonade.events import RequestStats, ToolCall
from lemonrind.storage import Database

# A sub-query that lists a chat's tag names as one text value, separated by an unlikely character
# (the ASCII "unit separator"), so tag names may contain commas and spaces.
_TAG_SEPARATOR = "\x1f"

# The only column assignments ``_update_session`` will run.
_SESSION_ASSIGNMENTS = frozenset({"title = ?", "folder_id = ?", "summary = ?, summary_upto = ?"})

# (char(31) is the unit separator, the same character as _TAG_SEPARATOR: the SQL is written out in full so that no text
# is ever pasted into it.)
_SESSION_SELECT = """
    SELECT s.id, s.title, s.created_at, s.updated_at, s.folder_id, f.name AS folder_name,
           s.summary, s.summary_upto, s.options_json,
           (SELECT group_concat(t.name, char(31))
              FROM session_tags st JOIN tags t ON t.id = st.tag_id
             WHERE st.session_id = s.id) AS tag_names
      FROM sessions s
      LEFT JOIN folders f ON f.id = s.folder_id
"""

# Escape character for LIKE patterns, so a typed "%" or "_" is matched literally.
_LIKE_ESCAPE = "!"


class ChatNotFoundError(LookupError):
    """Raised when an operation names a chat that does not exist."""


def now_utc() -> datetime:
    return datetime.now(UTC)


def _like_pattern(text: str) -> str:
    for char in (_LIKE_ESCAPE, "%", "_"):
        text = text.replace(char, _LIKE_ESCAPE + char)
    return f"%{text}%"


class ChatRepository:
    def __init__(self, db: Database, clock: Callable[[], datetime] = now_utc) -> None:
        self._db = db
        self._clock = clock

    # --- chats ---------------------------------------------------------------------------------

    def create_session(
        self, title: str = DEFAULT_TITLE, folder_id: str | None = None
    ) -> ChatSession:
        session_id = str(uuid.uuid4())
        now = self._clock().isoformat()
        with self._db.transaction() as conn:
            conn.execute(
                "INSERT INTO sessions (id, title, folder_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, title, folder_id, now, now),
            )
        session = self.get_session(session_id)
        assert session is not None  # we inserted it a moment ago
        return session

    def get_session(self, session_id: str) -> ChatSession | None:
        row = self._db.conn.execute(f"{_SESSION_SELECT} WHERE s.id = ?", (session_id,)).fetchone()
        return _session_from_row(row) if row else None

    def list_sessions(self, limit: int | None = None) -> list[ChatSession]:
        """All chats, most recently active first."""
        sql = f"{_SESSION_SELECT} ORDER BY s.updated_at DESC, s.rowid DESC"
        params: tuple[Any, ...] = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (limit,)
        return [_session_from_row(row) for row in self._db.conn.execute(sql, params)]

    def rename_session(self, session_id: str, title: str) -> None:
        self._update_session(session_id, "title = ?", (title.strip() or DEFAULT_TITLE,))

    def move_to_folder(self, session_id: str, folder_id: str | None) -> None:
        """Put a chat in a folder, or pass ``None`` to take it out of its folder.

        A folder the chat leaves is removed if that was its last chat.
        """
        left = self._folder_of(session_id)
        self._update_session(session_id, "folder_id = ?", (folder_id,))
        if left is not None and left != folder_id:
            self._remove_folder_if_empty(left)

    def set_summary(self, session_id: str, summary: str, upto_message_id: int) -> None:
        """Record that messages up to ``upto_message_id`` are now covered by ``summary``."""
        self._update_session(
            session_id, "summary = ?, summary_upto = ?", (summary, upto_message_id)
        )

    def set_option(self, session_id: str, key: str, value: str | None) -> None:
        """Set one per-chat option (or remove it with ``None``)."""
        with self._db.transaction() as conn:
            row = conn.execute(
                "SELECT options_json FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if row is None:
                raise ChatNotFoundError(session_id)
            options = json.loads(row["options_json"])
            if value is None:
                options.pop(key, None)
            else:
                options[key] = value
            conn.execute(
                "UPDATE sessions SET options_json = ? WHERE id = ?",
                (json.dumps(options), session_id),
            )

    def delete_session(self, session_id: str) -> None:
        # Messages and tag links go too (ON DELETE CASCADE), and a trigger removes the search entries.
        left = self._folder_of(session_id)
        with self._db.transaction() as conn:
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        if left is not None:
            self._remove_folder_if_empty(left)

    def _folder_of(self, session_id: str) -> str | None:
        row = self._db.conn.execute(
            "SELECT folder_id FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return row["folder_id"] if row else None

    def _remove_folder_if_empty(self, folder_id: str) -> None:
        """A folder only exists to hold chats: when its last chat leaves (moved out or deleted), it goes too."""
        with self._db.transaction() as conn:
            conn.execute(
                "DELETE FROM folders WHERE id = ? AND NOT EXISTS "
                "(SELECT 1 FROM sessions WHERE folder_id = ?)",
                (folder_id, folder_id),
            )

    def prune_empty_folders(self) -> None:
        """Remove every folder that holds no chats (tidies up ones left by an earlier version)."""
        with self._db.transaction() as conn:
            conn.execute(
                "DELETE FROM folders WHERE id NOT IN "
                "(SELECT DISTINCT folder_id FROM sessions WHERE folder_id IS NOT NULL)"
            )

    def _update_session(self, session_id: str, assignment: str, params: tuple[Any, ...]) -> None:
        # ``assignment`` is one of the fixed pieces of SQL in ``_SESSION_ASSIGNMENTS`` (never user input), so building
        # the statement with an f-string is safe; the values still travel as parameters. The check makes that a rule.
        if assignment not in _SESSION_ASSIGNMENTS:
            raise ValueError(f"Not an allowed update: {assignment!r}")
        with self._db.transaction() as conn:
            cursor = conn.execute(
                f"UPDATE sessions SET {assignment} WHERE id = ?",  # nosec B608
                (*params, session_id),
            )
            if cursor.rowcount == 0:
                raise ChatNotFoundError(session_id)

    # --- messages --------------------------------------------------------------------------------

    def add_message(
        self,
        session_id: str,
        role: Role,
        content: str,
        *,
        reasoning: str | None = None,
        stats: RequestStats | None = None,
        tool_calls: Sequence[ToolCall] = (),
        tool_call_id: str | None = None,
        tool_failed: bool = False,
        model: str | None = None,
    ) -> StoredMessage:
        """Append a message to a chat and mark the chat as just-used.

        If this is the chat's first *user* message and the chat still has the default title, the title
        becomes a short version of the message. Doing that here means every front end gets it for free.
        """
        now = self._clock().isoformat()
        stats_json = json.dumps(dataclasses.asdict(stats)) if stats else None
        tool_calls_json = (
            json.dumps([dataclasses.asdict(call) for call in tool_calls]) if tool_calls else None
        )
        try:
            with self._db.transaction() as conn:
                cursor = conn.execute(
                    "INSERT INTO messages (session_id, role, content, reasoning, stats_json,"
                    " tool_calls_json, tool_call_id, tool_failed, model, created_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        session_id,
                        role,
                        content,
                        reasoning,
                        stats_json,
                        tool_calls_json,
                        tool_call_id,
                        int(tool_failed),
                        model,
                        now,
                    ),
                )
                message_id = cursor.lastrowid
                conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))
                if role == "user":
                    row = conn.execute(
                        "SELECT title, (SELECT count(*) FROM messages WHERE session_id = :id AND role = 'user')"
                        " AS user_messages FROM sessions WHERE id = :id",
                        {"id": session_id},
                    ).fetchone()
                    if row["title"] == DEFAULT_TITLE and row["user_messages"] == 1:
                        conn.execute(
                            "UPDATE sessions SET title = ? WHERE id = ?",
                            (auto_title(content), session_id),
                        )
        except sqlite3.IntegrityError as error:  # the foreign key to sessions failed: no such chat
            raise ChatNotFoundError(session_id) from error
        return StoredMessage(
            id=message_id or 0,
            session_id=session_id,
            role=role,
            content=content,
            created_at=datetime.fromisoformat(now),
            reasoning=reasoning,
            stats=stats,
            tool_calls=tuple(tool_calls),
            tool_call_id=tool_call_id,
            tool_failed=tool_failed,
            model=model,
        )

    def list_messages(self, session_id: str) -> list[StoredMessage]:
        rows = self._db.conn.execute(
            "SELECT id, session_id, role, content, reasoning, stats_json, tool_calls_json,"
            " tool_call_id, tool_failed, model, created_at FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        )
        return [_message_from_row(row) for row in rows]

    # --- folders -----------------------------------------------------------------------------------

    def get_or_create_folder(self, name: str) -> Folder:
        """Find a folder by name (ignoring case) or create it."""
        name = name.strip()
        row = self._db.conn.execute(
            "SELECT id, name, collapsed FROM folders WHERE name = ?", (name,)
        ).fetchone()
        if row:
            return _folder_from_row(row)
        folder_id = str(uuid.uuid4())
        with self._db.transaction() as conn:
            conn.execute("INSERT INTO folders (id, name) VALUES (?, ?)", (folder_id, name))
        return Folder(id=folder_id, name=name)

    def list_folders(self) -> list[Folder]:
        rows = self._db.conn.execute("SELECT id, name, collapsed FROM folders ORDER BY name")
        return [_folder_from_row(row) for row in rows]

    def delete_folder(self, folder_id: str) -> None:
        """Delete a folder; its chats are kept and become unfiled."""
        with self._db.transaction() as conn:
            conn.execute("DELETE FROM folders WHERE id = ?", (folder_id,))

    def rename_folder(self, folder_id: str, name: str) -> None:
        """Give a folder a new name. Raises ``ValueError`` for a blank name or one another folder already has."""
        name = name.strip()
        if not name:
            raise ValueError("A folder needs a name.")
        try:
            with self._db.transaction() as conn:
                conn.execute("UPDATE folders SET name = ? WHERE id = ?", (name, folder_id))
        except sqlite3.IntegrityError as error:  # the UNIQUE constraint on the name
            raise ValueError(f"A folder called '{name}' already exists.") from error

    def set_folder_collapsed(self, folder_id: str, collapsed: bool) -> None:
        with self._db.transaction() as conn:
            conn.execute(
                "UPDATE folders SET collapsed = ? WHERE id = ?", (int(collapsed), folder_id)
            )

    # --- tags --------------------------------------------------------------------------------------

    def add_tag(self, session_id: str, name: str) -> None:
        """Tag a chat, creating the tag if it is new. Tagging twice is harmless."""
        name = name.strip()
        try:
            with self._db.transaction() as conn:
                conn.execute("INSERT OR IGNORE INTO tags (name) VALUES (?)", (name,))
                tag_id = conn.execute("SELECT id FROM tags WHERE name = ?", (name,)).fetchone()[
                    "id"
                ]
                conn.execute(
                    "INSERT OR IGNORE INTO session_tags (session_id, tag_id) VALUES (?, ?)",
                    (session_id, tag_id),
                )
        except sqlite3.IntegrityError as error:
            raise ChatNotFoundError(session_id) from error

    def remove_tag(self, session_id: str, name: str) -> None:
        with self._db.transaction() as conn:
            conn.execute(
                "DELETE FROM session_tags WHERE session_id = ? AND tag_id = (SELECT id FROM tags WHERE name = ?)",
                (session_id, name.strip()),
            )

    def list_tags(self) -> list[str]:
        return [row["name"] for row in self._db.conn.execute("SELECT name FROM tags ORDER BY name")]

    # --- usage statistics --------------------------------------------------------------------------
    # The rows for the usage screen. They are only read here; the arithmetic is in ``chats/usage.py``.

    def usage_requests(self, since: datetime | None = None) -> list[UsageRequest]:
        """Every saved model request (a reply carrying statistics) finished at or after ``since``, oldest first."""
        rows = self._db.conn.execute(
            "SELECT m.created_at, m.session_id, s.title, m.model, m.stats_json,"
            " EXISTS (SELECT 1 FROM session_tags st JOIN tags t ON t.id = st.tag_id"
            "          WHERE st.session_id = s.id AND t.name = 'scheduled') AS scheduled"
            " FROM messages m JOIN sessions s ON s.id = m.session_id"
            " WHERE m.stats_json IS NOT NULL AND (:since IS NULL OR m.created_at >= :since)"
            " ORDER BY m.created_at",
            {"since": since.astimezone(UTC).isoformat() if since else None},
        )
        requests = []
        for row in rows:
            stats = RequestStats(**json.loads(row["stats_json"]))
            requests.append(
                UsageRequest(
                    at=datetime.fromisoformat(row["created_at"]),
                    session_id=row["session_id"],
                    title=row["title"],
                    model=row["model"],
                    input_tokens=stats.input_tokens,
                    output_tokens=stats.output_tokens,
                    tokens_per_second=stats.tokens_per_second,
                    time_to_first_token=stats.time_to_first_token,
                    scheduled=bool(row["scheduled"]),
                )
            )
        return requests

    def usage_tool_uses(self, since: datetime | None = None) -> list[ToolUse]:
        """Every tool call whose result was saved at or after ``since``, with its name and whether it failed.

        A tool result names the call it answers by id, and the call itself (with the tool's name, and the model that
        asked) is on the assistant message just before. Ids can repeat from one turn to the next, so the lookup
        is kept per chat and the most recent assistant message that used the id wins.
        """
        rows = self._db.conn.execute(
            "SELECT m.session_id, m.role, m.created_at, m.model, m.tool_calls_json, m.tool_call_id,"
            " m.tool_failed,"
            " EXISTS (SELECT 1 FROM session_tags st JOIN tags t ON t.id = st.tag_id"
            "          WHERE st.session_id = m.session_id AND t.name = 'scheduled') AS scheduled"
            " FROM messages m"
            " WHERE m.role = 'tool' OR (m.role = 'assistant' AND m.tool_calls_json IS NOT NULL)"
            " ORDER BY m.session_id, m.id"
        )
        floor = since.astimezone(UTC).isoformat() if since else ""
        uses: list[ToolUse] = []
        asked: dict[
            str, tuple[str, str | None]
        ] = {}  # call id -> (tool name, model), for the current chat
        current = None
        for row in rows:
            if row["session_id"] != current:
                current, asked = row["session_id"], {}
            if row["role"] == "assistant":
                for call in json.loads(row["tool_calls_json"]):
                    asked[call["id"]] = (call["name"], row["model"])
            elif (found := asked.get(row["tool_call_id"])) is not None and row[
                "created_at"
            ] >= floor:
                uses.append(
                    ToolUse(
                        at=datetime.fromisoformat(row["created_at"]),
                        name=found[0],
                        failed=bool(row["tool_failed"]),
                        model=found[1],
                        scheduled=bool(row["scheduled"]),
                    )
                )
        return uses

    # --- search ------------------------------------------------------------------------------------

    def search(self, query: str) -> list[ChatSession]:
        """Find chats matching ``query`` in their title, folder name, a tag, or any message text.

        Titles, folders and tags are matched as plain substrings (``LIKE``). Message text uses the
        full-text index, matching word *beginnings*: "hik" finds "hiking", but "king" does not.
        An empty query returns every chat. ``search_hits`` is the same search, plus a piece of the matching message.
        """
        return [hit.session for hit in self.search_hits(query)]

    def search_hits(self, query: str) -> list[SearchHit]:
        """Like ``search``, but each chat comes with a ``snippet`` of its best matching message (if a message matched).

        The snippet is made by SQLite's own ``snippet()`` function: a few words around the match, with each matched
        word wrapped in two marker characters that ``text.parse_snippet`` understands. The best message is the one
        the full-text index ranks highest for the query.
        """
        text = query.strip()
        if not text:
            return [SearchHit(session) for session in self.list_sessions()]

        conditions = [
            "s.title LIKE :like ESCAPE '!'",
            "f.name LIKE :like ESCAPE '!'",
            "EXISTS (SELECT 1 FROM session_tags st JOIN tags t ON t.id = st.tag_id"
            " WHERE st.session_id = s.id AND t.name LIKE :like ESCAPE '!')",
        ]
        params: dict[str, Any] = {"like": _like_pattern(text)}
        if (fts_query := build_fts_query(text)) is not None:
            conditions.append(
                "s.id IN (SELECT m.session_id FROM messages m WHERE m.role != 'tool'"
                " AND m.id IN (SELECT rowid FROM messages_fts WHERE messages_fts MATCH :fts))"
            )
            params["fts"] = fts_query

        sql = f"{_SESSION_SELECT} WHERE {' OR '.join(conditions)} ORDER BY s.updated_at DESC, s.rowid DESC"
        sessions = [_session_from_row(row) for row in self._db.conn.execute(sql, params)]
        snippets = self._best_snippets(params["fts"], text) if "fts" in params else {}
        return [SearchHit(session, snippets.get(session.id)) for session in sessions]

    def _best_snippets(self, fts_query: str, typed: str) -> dict[str, str]:
        """For each chat with a matching message, the snippet of its best-ranked matching message.

        A message with a picture attached is stored with a marker holding the picture's file path in front of what
        was typed. That path is searchable, but it is not something to show, so for those messages the snippet is
        cut from the typed words alone, and a message that matched only through its path offers no snippet.
        """
        best: dict[str, str] = {}
        rows = self._db.conn.execute(
            "SELECT m.session_id AS session_id, m.content AS content,"
            " snippet(messages_fts, 0, char(2), char(3), '...', 16) AS snippet"
            " FROM messages_fts JOIN messages m ON m.id = messages_fts.rowid"
            " WHERE messages_fts MATCH :fts AND m.role != 'tool'"
            " ORDER BY rank",
            {"fts": fts_query},
        )
        for row in rows:
            if row["session_id"] in best:
                continue  # rows come best first: keep the first usable one per chat
            picture, words = parse_image_marker(row["content"])
            if picture is None:
                best[row["session_id"]] = row["snippet"]
            elif (own := snippet_from_text(words, typed)) is not None:
                best[row["session_id"]] = own
        return best


# --- turning database rows into data classes -----------------------------------------------------------


def _session_from_row(row: sqlite3.Row) -> ChatSession:
    tag_names = row["tag_names"]
    tags = tuple(sorted(tag_names.split(_TAG_SEPARATOR), key=str.lower)) if tag_names else ()
    return ChatSession(
        id=row["id"],
        title=row["title"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        folder_id=row["folder_id"],
        folder_name=row["folder_name"],
        tags=tags,
        summary=row["summary"],
        summary_upto=row["summary_upto"],
        options=json.loads(row["options_json"]),
    )


def _message_from_row(row: sqlite3.Row) -> StoredMessage:
    stats = RequestStats(**json.loads(row["stats_json"])) if row["stats_json"] else None
    return StoredMessage(
        id=row["id"],
        session_id=row["session_id"],
        role=row["role"],
        content=row["content"],
        created_at=datetime.fromisoformat(row["created_at"]),
        reasoning=row["reasoning"],
        stats=stats,
        tool_calls=tuple(ToolCall(**item) for item in json.loads(row["tool_calls_json"] or "[]")),
        tool_call_id=row["tool_call_id"],
        tool_failed=bool(row["tool_failed"]),
        model=row["model"],
    )


def _folder_from_row(row: sqlite3.Row) -> Folder:
    return Folder(id=row["id"], name=row["name"], collapsed=bool(row["collapsed"]))
