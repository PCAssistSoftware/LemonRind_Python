"""The terminal chat's module commands, and the tool lines shown while a reply streams."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from lemonrind.chats import ChatRepository
from lemonrind.config import Settings
from lemonrind.lemonade import ToolCall
from lemonrind.modules import ToolResult, build_registry
from lemonrind.storage import Database
from lemonrind.storage.migrations import MIGRATIONS, apply_migrations
from lemonrind.terminal.chatloop import ChatLoop
from lemonrind.terminal.streaming import describe_tool
from tests.conftest import NullMemoryClient
from tests.test_chatloop import FakeClient, ScriptedConsole
from tests.test_memory import FakeMemoryClient


def make_loop(lines: list[str], tmp_path: Path):
    settings = Settings()
    settings_file = tmp_path / "settings.json"
    console = ScriptedConsole(lines)
    loop = ChatLoop(
        client=FakeClient(),  # type: ignore[arg-type]
        settings=settings,
        settings_file=settings_file,
        repo=ChatRepository(Database(":memory:")),
        console=console,
        model="m",
        modules=build_registry(
            settings, tmp_path, db=Database(":memory:"), client=NullMemoryClient()
        ),
    )
    return loop, console, settings, settings_file


async def test_modules_lists_every_module_with_its_state_and_tools(tmp_path: Path):
    loop, console, _, _ = make_loop(["/modules", "/quit"], tmp_path)
    await loop.run(resume=False)

    for expected in (
        "utilities",
        "web_search",
        "filesystem",
        "current_time, calculate",
        "read_file",
    ):
        assert expected in console.output


async def test_module_off_switches_it_immediately_and_saves_the_setting(tmp_path: Path):
    loop, console, settings, settings_file = make_loop(
        ["/module web_search off", "/quit"], tmp_path
    )
    await loop.run(resume=False)

    assert settings.modules.enabled["web_search"] is False
    assert "web_search" not in [s["function"]["name"] for s in loop.modules.schemas()]  # type: ignore[union-attr]
    assert Settings.load(settings_file).modules.enabled["web_search"] is False
    assert "Web search is now off" in console.output


async def test_module_with_a_bad_key_shows_usage(tmp_path: Path):
    loop, console, settings, _ = make_loop(
        ["/module nope on", "/module utilities maybe", "/quit"], tmp_path
    )
    await loop.run(resume=False)

    assert console.output.count("Usage: /module KEY on|off") == 2
    assert settings.modules.enabled == {}


def test_tool_status_lines_show_progress_success_and_failure():
    call = ToolCall("c1", "search", '{"query": "otters"}')

    running = describe_tool(call.name, call.arguments, None).plain
    done = describe_tool(call.name, call.arguments, (ToolResult("abc"), 1.25)).plain
    failed = describe_tool(call.name, call.arguments, (ToolResult("nope", True), 0.1)).plain

    assert running.startswith('tool search {"query": "otters"}') and running.endswith("...")
    assert "done in 1.2s (3 characters)" in done
    assert "failed: nope" in failed


def test_a_version_1_database_is_upgraded_and_keeps_its_chats(tmp_path: Path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(f"BEGIN;\n{MIGRATIONS[0]}\nPRAGMA user_version = 1;\nCOMMIT;")
    conn.execute("INSERT INTO sessions VALUES ('s1', 'Old chat', NULL, '2026-01-01', '2026-01-01')")
    conn.execute(
        "INSERT INTO messages (session_id, role, content, created_at) VALUES ('s1', 'user', 'hi', '2026-01-01')"
    )
    conn.commit()
    conn.row_factory = sqlite3.Row

    assert apply_migrations(conn) == len(MIGRATIONS)
    conn.close()

    repo = ChatRepository(Database(path))
    (message,) = repo.list_messages("s1")
    assert (message.content, message.tool_calls, message.tool_call_id) == ("hi", (), None)


async def test_memory_commands_add_list_pin_and_forget(tmp_path: Path):
    settings = Settings()
    db = Database(":memory:")
    console = ScriptedConsole(
        [
            "/remember I live in Manchester",
            "/memories",
            "/pin 1",
            "/forget 1",
            "/memories",
            "/quit",
        ]
    )
    loop = ChatLoop(
        client=FakeClient(),  # type: ignore[arg-type]
        settings=settings,
        repo=ChatRepository(db),
        console=console,
        model="m",
        modules=build_registry(settings, tmp_path, db=db, client=FakeMemoryClient()),  # type: ignore[arg-type]
    )

    await loop.run(resume=False)

    output = console.output
    assert "Remembered." in output
    assert "1. " in output and "I live in Manchester" in output
    assert "Pinned." in output
    assert "Forgotten." in output
    assert "Nothing is remembered yet." in output


async def test_memory_commands_explain_when_the_module_is_off(tmp_path: Path):
    loop, console, settings, _ = make_loop(["/module memory off", "/memories", "/quit"], tmp_path)
    await loop.run(resume=False)
    assert "The Memory module is off" in console.output
