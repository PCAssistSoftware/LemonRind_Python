"""Small text helpers for chats: automatic titles, search-query building and "5m ago" times.

They are plain functions with no database in sight, which keeps them easy to test.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta, tzinfo

from lemonrind.attachment_markers import title_source

DEFAULT_TITLE = "New chat"

# Characters that mark the start and end of a matched word in a search snippet. They are control characters that
# never occur in real chat text, so nothing a person typed can be mistaken for a marker.
SNIPPET_START = chr(2)
SNIPPET_END = chr(3)

type Parts = list[tuple[str, bool]]  # pieces of a text, each with "is this piece a match?"


def auto_title(text: str, max_length: int = 50) -> str:
    """Make a chat title from its first message: the first line, tidied and shortened at a word boundary."""
    first_line = next(
        (line.strip() for line in title_source(text).splitlines() if line.strip()), ""
    )
    title = " ".join(first_line.split())  # collapse runs of spaces and tabs
    if not title:
        return DEFAULT_TITLE
    if len(title) <= max_length:
        return title
    cut = title[:max_length]
    last_space = cut.rfind(" ")
    if last_space > max_length // 2:  # only back up to a space if that does not lose too much
        cut = cut[:last_space]
    return cut.rstrip(" ,.;:-") + "..."


def build_fts_query(text: str) -> str | None:
    """Turn what a person typed into a safe SQLite full-text (FTS5) query, or ``None`` if nothing usable.

    Each word becomes a quoted prefix term, and the terms are combined with AND, so ``hik manch`` finds
    chats that contain a word starting with "hik" *and* one starting with "manch". Quoting matters:
    FTS5 has its own query syntax (``AND``, ``-``, ``*``, parentheses...), and passing raw user text
    could be misread as that syntax, or be a syntax error. Inside quotes the text is just words; a literal
    quote character is written as two quotes.
    """
    words = text.split()
    if not words:
        return None
    return " ".join('"' + word.replace('"', '""') + '"*' for word in words)


def highlight_parts(text: str, query: str) -> Parts:
    """Split ``text`` into pieces, flagging the ones that match a word the person typed (ignoring upper/lower case).

    ``"Plan a hike"`` searched for ``"hik"`` gives ``[("Plan a ", False), ("hik", True), ("e", False)]``. Used to
    highlight a title or a tag, which are matched as plain substrings. An empty query highlights nothing.
    """
    words = sorted({word for word in query.split() if word}, key=len, reverse=True)  # longest first
    if not words or not text:
        return [(text, False)] if text else []
    pattern = re.compile("|".join(re.escape(word) for word in words), re.IGNORECASE)
    parts: Parts = []
    position = 0
    for found in pattern.finditer(text):
        if found.start() > position:
            parts.append((text[position : found.start()], False))
        parts.append((found.group(), True))
        position = found.end()
    if position < len(text):
        parts.append((text[position:], False))
    return parts


def snippet_from_text(text: str, query: str, *, words_around: int = 8) -> str | None:
    """A snippet of ``text`` around the first word that starts with a typed word, in the database's marker format.

    The same kind of result SQLite's own ``snippet()`` gives (matched words wrapped in ``SNIPPET_START`` and
    ``SNIPPET_END``, ``...`` where the text was cut), for the one case where the database's version cannot be
    used: a message whose stored text begins with an attachment marker (a file path) that must not be shown. Returns
    ``None`` when no word matches.
    """
    words = sorted({word for word in query.split() if word}, key=len, reverse=True)
    if not words:
        return None
    # (?<!\w) means "not directly after a letter or digit". A plain word boundary would not work for a typed word that
    # starts with punctuation, such as "(5".
    pattern = re.compile(
        r"(?<!\w)(?:" + "|".join(re.escape(word) for word in words) + ")", re.IGNORECASE
    )
    tokens = text.split()
    first = next((i for i, token in enumerate(tokens) if pattern.search(token)), None)
    if first is None:
        return None
    start, end = max(0, first - words_around), min(len(tokens), first + words_around + 1)
    window = " ".join(tokens[start:end])
    marked = pattern.sub(lambda m: f"{SNIPPET_START}{m.group()}{SNIPPET_END}", window)
    return ("..." if start > 0 else "") + marked + ("..." if end < len(tokens) else "")


def parse_snippet(raw: str) -> Parts:
    """Turn a search snippet from the database (matches wrapped in start/end markers) into flagged pieces."""
    parts: Parts = []
    for chunk in raw.split(SNIPPET_START):
        matched, closed, rest = chunk.partition(SNIPPET_END)
        if (
            closed
        ):  # this chunk began at a start marker: the text before the end marker is the match
            if matched:
                parts.append((matched, True))
            if rest:
                parts.append((rest, False))
        elif chunk:  # the text before the first start marker
            parts.append((chunk, False))
    return parts


def relative_time(then: datetime, now: datetime | None = None, tz: tzinfo | None = None) -> str:
    """A short "how long ago": ``now``, ``5m``, ``3h``, ``Yesterday``, ``4d``, then a date like ``2 Oct``.

    ``tz`` is the time zone used to decide what "yesterday" means (default: the computer's own).
    """
    now = now or datetime.now(UTC)
    age = now - then
    if age < timedelta(minutes=1):
        return "now"
    if age < timedelta(hours=1):
        return f"{int(age.total_seconds() // 60)}m"
    if age < timedelta(days=1):
        return f"{int(age.total_seconds() // 3600)}h"
    local_then = then.astimezone(tz)
    local_now = now.astimezone(tz)
    if local_then.date() == (local_now - timedelta(days=1)).date():
        return "Yesterday"
    if age < timedelta(days=7):
        return f"{age.days}d"
    return f"{local_then.day} {local_then:%b}"
