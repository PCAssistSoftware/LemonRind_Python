"""The database schema, as a numbered list of migrations.

A *migration* is a piece of SQL that moves the database from one schema version to the next. SQLite has a
spare integer in the file header, ``PRAGMA user_version``, which we use to remember how many migrations
have already been applied. When the app opens the database it runs every migration after that number, in
order, and then records the new number. A brand new file runs all of them; an existing file runs only the
new ones. To change the schema in a later step, add a *new* entry to ``MIGRATIONS``. Never edit an old one,
because databases that already ran it would not see the change.

(This is the hand-rolled version of what Entity Framework migrations or Alembic do for you.)

Schema notes:

* Ids of chats are random UUID strings (``TEXT``); messages use an auto-incrementing integer, which also
  gives them a stable order.
* Timestamps are ISO 8601 text in UTC, e.g. ``2026-10-02T17:30:12.123456+00:00``. Text sorts correctly
  when all values use the same format and time zone.
* ``ON DELETE CASCADE`` makes deleting a chat delete its messages and tag links automatically;
  ``ON DELETE SET NULL`` makes deleting a folder move its chats back to "unfiled".
* ``messages_fts`` is a full-text index over message text (SQLite's FTS5). It is an "external content"
  table: it stores only the search index and reads the text from ``messages``. The three triggers keep
  the index in step whenever a message is inserted, deleted or edited.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence

MIGRATIONS: list[str] = [
    # 1: chats, messages, folders, tags and search
    """
    CREATE TABLE folders (
        id        TEXT PRIMARY KEY,
        name      TEXT NOT NULL UNIQUE COLLATE NOCASE,
        collapsed INTEGER NOT NULL DEFAULT 0
    );

    CREATE TABLE sessions (
        id         TEXT PRIMARY KEY,
        title      TEXT NOT NULL,
        folder_id  TEXT REFERENCES folders(id) ON DELETE SET NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    CREATE INDEX ix_sessions_updated ON sessions(updated_at DESC);

    CREATE TABLE tags (
        id   INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE COLLATE NOCASE
    );

    CREATE TABLE session_tags (
        session_id TEXT    NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        tag_id     INTEGER NOT NULL REFERENCES tags(id)     ON DELETE CASCADE,
        PRIMARY KEY (session_id, tag_id)
    );

    CREATE TABLE messages (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        role       TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant', 'tool')),
        content    TEXT NOT NULL,
        reasoning  TEXT,
        stats_json TEXT,
        created_at TEXT NOT NULL
    );
    CREATE INDEX ix_messages_session ON messages(session_id, id);

    CREATE VIRTUAL TABLE messages_fts USING fts5(
        content,
        content = 'messages',
        content_rowid = 'id',
        tokenize = 'unicode61 remove_diacritics 2'
    );

    CREATE TRIGGER messages_ai AFTER INSERT ON messages BEGIN
        INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
    END;

    CREATE TRIGGER messages_ad AFTER DELETE ON messages BEGIN
        INSERT INTO messages_fts(messages_fts, rowid, content) VALUES ('delete', old.id, old.content);
    END;

    CREATE TRIGGER messages_au AFTER UPDATE OF content ON messages BEGIN
        INSERT INTO messages_fts(messages_fts, rowid, content) VALUES ('delete', old.id, old.content);
        INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
    END;
    """,
    # 2: tool use. An assistant message may carry the tool calls it made (as JSON), and a message with
    # role 'tool' holds a tool's result and says which call it answers.
    """
    ALTER TABLE messages ADD COLUMN tool_calls_json TEXT;
    ALTER TABLE messages ADD COLUMN tool_call_id TEXT;
    """,
    # 3: long-term memory. A pinned fact is always given to the model; the others are found by meaning,
    # using the embedding (a vector stored as raw float32 bytes, see vectors.py).
    """
    CREATE TABLE memories (
        id         TEXT PRIMARY KEY,
        content    TEXT NOT NULL,
        pinned     INTEGER NOT NULL DEFAULT 0,
        embedding  BLOB,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    """,
    # 4: conversation compaction. When a chat nears the model's context limit, its oldest messages are
    # summarised. The summary is stored with the chat, and summary_upto is the id of the last message it
    # covers: those messages stay in the database (you still see them) but are no longer sent to the model.
    """
    ALTER TABLE sessions ADD COLUMN summary TEXT NOT NULL DEFAULT '';
    ALTER TABLE sessions ADD COLUMN summary_upto INTEGER NOT NULL DEFAULT 0;
    """,
    # 5: knowledge bases (search over your own documents) and per-chat module options.
    # A knowledge base has sources (a file, a folder, a web page, pasted text); each source is split into
    # chunks, and each chunk stores its embedding. options_json holds small per-chat choices that modules
    # read, such as which knowledge base this chat uses.
    """
    CREATE TABLE knowledge_bases (
        id         TEXT PRIMARY KEY,
        name       TEXT NOT NULL UNIQUE COLLATE NOCASE,
        created_at TEXT NOT NULL
    );

    CREATE TABLE knowledge_sources (
        id           TEXT PRIMARY KEY,
        kb_id        TEXT NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
        source_type  TEXT NOT NULL CHECK (source_type IN ('file', 'folder', 'website', 'text')),
        reference    TEXT,
        display_name TEXT NOT NULL,
        status       TEXT NOT NULL CHECK (status IN ('ingesting', 'ready', 'failed')),
        error        TEXT,
        chunk_count  INTEGER NOT NULL DEFAULT 0,
        created_at   TEXT NOT NULL
    );
    CREATE INDEX ix_sources_kb ON knowledge_sources(kb_id);

    CREATE TABLE knowledge_chunks (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        kb_id       TEXT NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
        source_id   TEXT NOT NULL REFERENCES knowledge_sources(id) ON DELETE CASCADE,
        chunk_index INTEGER NOT NULL,
        label       TEXT NOT NULL DEFAULT '',  -- which file inside a folder source this came from
        content     TEXT NOT NULL,
        embedding   BLOB NOT NULL
    );
    CREATE INDEX ix_chunks_kb ON knowledge_chunks(kb_id);

    ALTER TABLE sessions ADD COLUMN options_json TEXT NOT NULL DEFAULT '{}';
    """,
    # 6: MCP servers. A server is started as a program (transport 'stdio': command, args, env) or reached
    # at a web address (transport 'http': url, headers). env and headers may hold secrets such as API keys.
    # require_approval = ask before each run of this server's tools.
    """
    CREATE TABLE mcp_servers (
        id               TEXT PRIMARY KEY,
        name             TEXT NOT NULL UNIQUE COLLATE NOCASE,
        transport        TEXT NOT NULL CHECK (transport IN ('stdio', 'http')),
        command          TEXT NOT NULL DEFAULT '',
        args_json        TEXT NOT NULL DEFAULT '[]',
        env_json         TEXT NOT NULL DEFAULT '{}',
        url              TEXT NOT NULL DEFAULT '',
        headers_json     TEXT NOT NULL DEFAULT '{}',
        enabled          INTEGER NOT NULL DEFAULT 1,
        require_approval INTEGER NOT NULL DEFAULT 1,
        created_at       TEXT NOT NULL
    );
    """,
    # 7: scheduled jobs. A job is a saved prompt run through the assistant on a cron schedule (read in the
    # computer's local time; all timestamps stored in UTC). Each run produces a chat; last_session_id points at it.
    """
    CREATE TABLE scheduled_jobs (
        id                     TEXT PRIMARY KEY,
        name                   TEXT NOT NULL UNIQUE COLLATE NOCASE,
        cron                   TEXT NOT NULL,
        prompt                 TEXT NOT NULL,
        enabled                INTEGER NOT NULL DEFAULT 1,
        model                  TEXT NOT NULL DEFAULT '',
        allow_unattended_tools INTEGER NOT NULL DEFAULT 0,
        last_run_at            TEXT,
        next_run_at            TEXT,
        last_status            TEXT NOT NULL DEFAULT '',
        last_session_id        TEXT,
        running_since          TEXT,
        created_at             TEXT NOT NULL
    );
    CREATE INDEX ix_jobs_due ON scheduled_jobs(next_run_at);
    """,
    # 8: whether a tool call failed, kept so a reopened chat shows the real outcome of every call
    """
    ALTER TABLE messages ADD COLUMN tool_failed INTEGER NOT NULL DEFAULT 0;
    """,
    # 9: an optional description of a knowledge base, shown beside it in Settings
    """
    ALTER TABLE knowledge_bases ADD COLUMN description TEXT NOT NULL DEFAULT '';
    """,
    # 10: which model answered, saved with each reply that has statistics, so usage can be shown per model. NULL for
    # replies saved before this existed. The partial index keeps the usage queries (which only want messages that
    # carry statistics) fast however many other messages there are.
    """
    ALTER TABLE messages ADD COLUMN model TEXT;
    CREATE INDEX ix_messages_usage ON messages(created_at) WHERE stats_json IS NOT NULL;
    """,
    # 11: how long a scheduled job's last run took, in seconds (NULL until it has run since this existed)
    """
    ALTER TABLE scheduled_jobs ADD COLUMN last_duration_seconds REAL;
    """,
]


def apply_migrations(conn: sqlite3.Connection, migrations: Sequence[str] | None = None) -> int:
    """Bring the database up to date and return its schema version.

    ``migrations`` is the list to apply (default: the main database's). A knowledge base file has a list of its own.
    """
    migrations = MIGRATIONS if migrations is None else migrations
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    for number, sql in enumerate(migrations, start=1):
        if number <= current:
            continue
        # executescript runs several statements at once. Wrapping them in BEGIN/COMMIT makes the
        # migration all-or-nothing, and the version number is saved in the same transaction. (The
        # number cannot be passed as a ? parameter in a PRAGMA, which is why it is formatted in; it
        # comes from our own loop counter, never from outside input.)
        conn.executescript(f"BEGIN;\n{sql}\nPRAGMA user_version = {number};\nCOMMIT;")
    return conn.execute("PRAGMA user_version").fetchone()[0]
