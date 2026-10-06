"""Shared test helpers (pytest finds this file automatically).

``make_chunk`` builds the same kind of object the OpenAI SDK produces for each streamed piece of a reply,
so the parsing code can be tested exactly as it will run, without a server.
"""

from __future__ import annotations

import contextlib
import importlib
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from nicegui.testing import User
from openai.types.chat import ChatCompletionChunk

from lemonrind.modules.base import ChatContext

NO_CHAT = ChatContext(options={})  # a chat with no per-chat options


def must[T](value: T | None) -> T:
    """Return ``value``, failing the test if it is ``None``: tells the type checker what the test already knows."""
    assert value is not None
    return value


def elements(user: User, marker: str) -> list[Any]:
    """The page elements carrying ``marker``, typed loosely: NiceGUI's base element class has no ``value`` or ``text``."""
    return list(user.find(marker=marker).elements)


def make_chunk(
    *,
    content: str | None = None,
    reasoning: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str | None = None,
    usage: dict[str, Any] | None = None,
    no_choices: bool = False,
    **top_level_extras: Any,
) -> ChatCompletionChunk:
    delta: dict[str, Any] = {}
    if content is not None:
        delta["content"] = content
    if reasoning is not None:
        delta["reasoning_content"] = reasoning  # Lemonade's extra field, kept in model_extra
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    data: dict[str, Any] = {
        "id": "chunk-1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "test-model",
        "choices": []
        if no_choices
        else [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        **top_level_extras,
    }
    if usage is not None:
        data["usage"] = usage
    return ChatCompletionChunk.model_validate(data)


class NullMemoryClient:
    """A client with no embedding model: what the memory module sees when Lemonade has none."""

    async def list_models(self, *, downloaded_only: bool = True) -> list:
        return []

    async def embed(self, texts, model):
        raise AssertionError("no embedding model, so embed must not be called")

    async def complete_json(self, messages, model, *, schema_name, schema) -> str:
        return '{"facts": []}'


async def open_section(user: User, key: str) -> None:
    """Open Settings and choose one of its sections in the navigation, as a user would."""
    try:
        user.find(marker="settings")
    except ValueError:  # no page opened yet in this test
        await user.open("/")
    user.find(marker="settings").click()
    await user.should_see(marker=f"nav-{key}")
    user.find(marker=f"nav-{key}").click()


@pytest.fixture(autouse=True)
def close_every_database_a_test_opens(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Close each ``Database`` a test opened when the test ends, so none is left for the garbage collector.

    Python warns (``ResourceWarning: unclosed database``) about a SQLite connection that is thrown away without
    ``close()``. Many tests build a database, a repository or a whole app context and simply let it go; this keeps
    them tidy without every test needing a ``finally`` block. A test that already closed its database is
    unaffected, because closing twice is harmless.

    The class is looked up fresh for each test (not imported once at the top), because the web-UI tests reload the
    application's modules between tests, and the patch must reach whichever copy is current.
    """
    module = importlib.import_module("lemonrind.storage.database")
    database_class = module.Database
    opened: list[Any] = []
    original_init = database_class.__init__

    def tracking_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        opened.append(self)

    monkeypatch.setattr(database_class, "__init__", tracking_init)
    yield
    for database in opened:
        # (one opened on another thread cannot be closed from this one: SQLite refuses, and that is fine here)
        with contextlib.suppress(sqlite3.ProgrammingError):
            database.close()


@pytest.fixture
def web(user: User, tmp_path: Path) -> Iterator[Any]:
    """An app context with a fake Lemonade and the page registered.

    It depends on ``user`` first: that fixture resets NiceGUI's page registry, and the page has to be registered
    after the reset. The data folder is ``tmp_path``, so nothing touches the real one.
    """
    from lemonrind.webui.context import set_context
    from lemonrind.webui.page import register_pages
    from tests.test_webui import FakeLemonade, make_context

    context = make_context(tmp_path, FakeLemonade())
    set_context(context)
    register_pages()
    yield context
    set_context(None)
