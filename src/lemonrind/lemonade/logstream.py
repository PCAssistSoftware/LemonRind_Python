"""Lemonade's live server log: a WebSocket that streams what the server itself is writing.

Lemonade documents ``WS /logs/stream`` on a separate *websocket port* (reported by ``/health`` as ``websocket_port``).
After connecting you send ``{"type": "logs.subscribe", "after_seq": null}``; Lemonade answers with one **snapshot**
(``logs.snapshot``: every entry it still holds, oldest first) and then one message per new line (``logs.entry``).
Each entry has a sequence number, a timestamp, a severity (``Info``, ``Warn``, ``Error``, ``Fatal``), a tag (which part
of the server wrote it) and the line itself.

This is what to look at when a model will not load, a request is slow, or something about the backend is odd: it is the
server's own account, not the app's (the app's own log is the other tab of the logs screen).

Python ideas used here:

* ``aiohttp``'s WebSocket client: ``async with session.ws_connect(url) as ws``, then ``async for message in ws``.
  A message split across several frames arrives whole, which the .NET client has to reassemble by hand.
* Callbacks passed in (``on_snapshot``, ``on_entry``): the screen decides what to do with each entry, this file only
  knows how to listen.
* A pydantic model with ``Field(alias=...)``-free names, because Lemonade's keys are already ``seq``, ``severity``...
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Iterable
from urllib.parse import urlsplit

import aiohttp
from pydantic import BaseModel

logger = logging.getLogger(__name__)

CONNECT_TIMEOUT_SECONDS = 10.0
SEVERITIES = ("Info", "Warn", "Error", "Fatal")


class LogEntry(BaseModel):
    seq: int = 0
    timestamp: str = ""
    severity: str = ""
    tag: str = ""
    line: str = ""

    @property
    def day(self) -> str:
        """The date part of the timestamp, ``2026-10-03``."""
        return self.timestamp[:10] if len(self.timestamp) >= 10 else ""

    @property
    def clock(self) -> str:
        """The time part of the timestamp, ``15:33:00.505``."""
        return self.timestamp[11:] if len(self.timestamp) > 11 else self.timestamp

    @property
    def message(self) -> str:
        """The text of the line without the timestamp, severity and tag that it repeats (they have columns of their own).

        Lines from the llama.cpp backend (tag ``Process``) also start with the backend's own clock and a one-letter level;
        those are dropped too.
        """
        text = _LEADING_FIELDS.sub("", self.line, count=1)
        if self.tag == "Process":
            text = _BACKEND_CLOCK.sub("", text, count=1)
        return text.strip() or self.line.strip()

    def render(self) -> str:
        """One line of text for the screen and for copying. Lemonade's ``line`` already holds the whole formatted line."""
        if self.line:
            return self.line
        tag = f"[{self.tag}] " if self.tag else ""
        return f"{self.timestamp} {self.severity.upper():<5} {tag}".strip()


_LEADING_FIELDS = re.compile(
    r"^\d{4}-\d\d-\d\d[ T]\d\d:\d\d:\d\d(?:\.\d+)?\s+\[[A-Za-z]+\]\s+\([^)]*\)\s*"
)
_BACKEND_CLOCK = re.compile(r"^\d+\.\d\d\.\d\d\d\.\d+\s+[A-Z]\s+")

# How serious each severity is, so "warnings and above" can be a comparison. Unknown severities count as Info.
_RANK = {"trace": 0, "debug": 0, "info": 1, "warn": 2, "warning": 2, "error": 3, "fatal": 4}
LEVELS = {"All": 0, "Warnings and errors": 2, "Errors only": 3}


def severity_rank(severity: str) -> int:
    return _RANK.get(severity.strip().lower(), 1)


def filter_entries(
    entries: Iterable[LogEntry], *, level: str = "All", text: str = ""
) -> list[LogEntry]:
    """The entries at least as serious as ``level`` ("All", "Warnings and errors" or "Errors only") that contain ``text``."""
    minimum = LEVELS.get(level, 0)
    needle = text.strip().lower()
    return [
        entry
        for entry in entries
        if severity_rank(entry.severity) >= minimum
        and (not needle or needle in entry.render().lower())
    ]


def websocket_url(base_url: str, port: int) -> str:
    """``ws://host:port/logs/stream`` for a Lemonade whose API is at ``base_url`` (``wss`` if that is ``https``)."""
    parts = urlsplit(base_url)
    scheme = "wss" if parts.scheme == "https" else "ws"
    host = parts.hostname or "localhost"
    if ":" in host:  # an IPv6 literal needs brackets in a URL
        host = f"[{host}]"
    return f"{scheme}://{host}:{port}/logs/stream"


def parse_message(text: str) -> tuple[list[LogEntry], bool] | None:
    """``(entries, is_snapshot)`` for a message from the server, or ``None`` for a kind we do not use."""
    try:
        data = json.loads(text)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    kind = data.get("type")
    if kind == "logs.snapshot":
        return [LogEntry.model_validate(item) for item in data.get("entries") or []], True
    if kind == "logs.entry" and data.get("entry"):
        return [LogEntry.model_validate(data["entry"])], False
    return None


async def stream_logs(
    base_url: str,
    port: int,
    api_key: str,
    on_snapshot: Callable[[list[LogEntry]], None],
    on_entry: Callable[[LogEntry], None],
) -> None:
    """Connect, subscribe, and call the callbacks until the connection ends or the task is cancelled."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    timeout = aiohttp.ClientTimeout(total=None, connect=CONNECT_TIMEOUT_SECONDS)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.ws_connect(
            websocket_url(base_url, port), headers=headers, max_msg_size=0
        ) as ws:
            await ws.send_json({"type": "logs.subscribe", "after_seq": None})
            async for message in ws:
                if message.type != aiohttp.WSMsgType.TEXT:
                    continue
                parsed = parse_message(message.data)
                if parsed is None:
                    continue
                entries, is_snapshot = parsed
                if is_snapshot:
                    on_snapshot(entries)
                else:
                    on_entry(entries[0])
