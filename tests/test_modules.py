"""Tests for the module system: the registry, and the Utilities, File system and Web search modules."""

from __future__ import annotations

import json
import re
from pathlib import Path

import httpx2 as httpx
import pytest

from lemonrind.config import Settings
from lemonrind.lemonade import ToolCall
from lemonrind.modules import (
    FileSystemModule,
    Module,
    ModuleRegistry,
    Tool,
    ToolError,
    UtilitiesModule,
    WebSearchModule,
    build_registry,
    tool_from_function,
)
from lemonrind.modules.filesystem import PathSandbox
from lemonrind.modules.utilities import evaluate
from lemonrind.storage import Database
from tests.conftest import NullMemoryClient


def call(name: str, **arguments) -> ToolCall:
    return ToolCall(id="c1", name=name, arguments=json.dumps(arguments))


# --- the registry ---------------------------------------------------------------------------------------


class RecordingModule(Module):
    """A module that records its lifecycle, so the registry's behaviour can be checked."""

    name = "Recording"
    config_key = "recording"
    description = "Test module."

    def __init__(self, settings: Settings, *, fail_on_start: bool = False) -> None:
        super().__init__(settings)
        self.events: list[str] = []
        self.fail_on_start = fail_on_start

    async def on_startup(self) -> None:
        self.events.append("start")
        if self.fail_on_start:
            raise RuntimeError("cannot start")

    async def on_shutdown(self) -> None:
        self.events.append("stop")

    def get_tools(self) -> list[Tool]:
        def echo(text: str) -> str:
            """Echo the text back."""
            return text

        def big() -> str:
            """Return a lot of text."""
            return "x" * 100

        return [tool_from_function(echo), tool_from_function(big)]


def test_only_enabled_modules_offer_tools():
    settings = Settings()
    registry = ModuleRegistry([RecordingModule(settings), UtilitiesModule(settings)])
    settings.modules.enabled["recording"] = False

    names = [schema["function"]["name"] for schema in registry.schemas()]

    assert names == [
        "current_time",
        "calculate",
        "read_more",
    ]  # read_more is built in, offered with any tool


def test_a_module_without_a_setting_uses_its_default():
    class OffByDefault(RecordingModule):
        config_key = "off"
        enabled_by_default = False

    settings = Settings()
    module = OffByDefault(settings)
    assert not module.enabled
    settings.modules.enabled["off"] = True
    assert module.enabled


async def test_reconcile_starts_and_stops_modules_as_settings_change():
    settings = Settings()
    module = RecordingModule(settings)
    registry = ModuleRegistry([module])

    await registry.start_enabled()
    await registry.reconcile()  # nothing changed: must not start twice
    settings.modules.enabled["recording"] = False
    await registry.reconcile()
    settings.modules.enabled["recording"] = True
    await registry.reconcile()
    await registry.stop_all()

    assert module.events == ["start", "stop", "start", "stop"]


async def test_a_module_that_fails_to_start_does_not_stop_the_others_and_is_retried():
    settings = Settings()
    broken = RecordingModule(settings, fail_on_start=True)
    healthy = UtilitiesModule(settings)
    registry = ModuleRegistry([broken, healthy])

    await registry.start_enabled()  # must not raise
    await (
        registry.reconcile()
    )  # the failed module is tried again, since it never counted as started

    assert broken.events == ["start", "start"]
    assert [t.name for t in registry.get_enabled_tools()][-3:] == [
        "current_time",
        "calculate",
        "read_more",
    ]


async def test_running_a_tool_finds_it_by_name_and_reports_unknown_names():
    registry = ModuleRegistry([RecordingModule(Settings())])

    ok = await registry.run(call("echo", text="hello"))
    unknown = await registry.run(call("nope"))

    assert ok.content == "hello" and not ok.is_error
    assert unknown.is_error and "echo" in unknown.content  # lists what is available


async def test_a_disabled_modules_tools_cannot_be_called():
    settings = Settings()
    settings.modules.enabled["recording"] = False
    registry = ModuleRegistry([RecordingModule(settings)])

    assert (await registry.run(call("echo", text="hi"))).is_error


async def test_long_results_are_cut_with_a_note_that_says_how_to_read_on():
    registry = ModuleRegistry([RecordingModule(Settings())], max_output_chars=30)

    result = await registry.run(call("big"))

    assert result.content.startswith("x" * 30)
    assert "70 more characters not shown" in result.content
    assert 'read_more with result="r1" and start=30' in result.content


async def test_read_more_gives_the_rest_piece_by_piece_until_the_end():
    registry = ModuleRegistry([RecordingModule(Settings())], max_output_chars=30)
    first = await registry.run(call("big"))  # "x" * 100: three pieces of 30 and a last of 10
    assert first.content.count("x") == 30

    second = await registry.run(call("read_more", result="r1", start=30))
    third = await registry.run(call("read_more", result="r1", start=60))
    last = await registry.run(call("read_more", result="r1", start=90))

    assert second.content.startswith("x" * 30) and "start=60" in second.content
    assert third.content.startswith("x" * 30) and "start=90" in third.content
    assert last.content.startswith("x" * 10) and "[End of result r1.]" in last.content
    assert "more characters" not in last.content
    assert not any(r.is_error for r in (first, second, third, last))


async def test_read_more_is_offered_only_when_some_other_tool_is_and_asks_for_no_permission():
    registry = ModuleRegistry([RecordingModule(Settings())])
    names = [schema["function"]["name"] for schema in registry.schemas()]
    assert "read_more" in names
    assert not registry.requires_approval(call("read_more", result="r1", start=0))

    off = Settings()
    off.modules.enabled["recording"] = False
    assert (
        ModuleRegistry([RecordingModule(off)]).schemas() == []
    )  # nothing to read on from: not offered either


async def test_read_more_explains_a_wrong_id_or_position_and_forgets_the_oldest_results():
    registry = ModuleRegistry([RecordingModule(Settings())], max_output_chars=30)
    for _ in range(25):  # 25 long results: only the newest 20 are kept
        await registry.run(call("big"))

    gone = await registry.run(call("read_more", result="r1", start=0))
    assert gone.is_error and "nothing saved as 'r1'" in gone.content and "r25" in gone.content
    beyond = await registry.run(call("read_more", result="r25", start=100))
    assert beyond.is_error and "between 0 and 99" in beyond.content
    negative = await registry.run(call("read_more", result="r25", start=-5))
    assert negative.is_error
    kept = await registry.run(call("read_more", result="r6", start=0))  # the oldest one still held
    assert not kept.is_error and kept.content.startswith("x" * 30)


async def test_results_are_not_cut_when_they_fit():
    registry = ModuleRegistry([RecordingModule(Settings())], max_output_chars=200)

    result = await registry.run(call("big"))  # 100 characters

    assert result.content == "x" * 100  # no note, nothing held
    assert (await registry.run(call("read_more", result="r1", start=0))).is_error


def test_build_registry_lists_every_module_with_its_own_key():
    registry = build_registry(
        Settings(), Path("data"), db=Database(":memory:"), client=NullMemoryClient()
    )

    assert [m.config_key for m in registry.modules] == [
        "utilities",
        "web_search",
        "web_reader",
        "filesystem",
        "coder",
        "memory",
        "knowledge",
        "mcp",
        "scheduler",
        "images",
        "backup",
    ]
    assert len({m.name for m in registry.modules}) == 11


# --- utilities --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2 + 3 * 4", 14),
        ("(2 + 3) * 4", 20),
        ("2 ** 10", 1024),
        ("-7 // 2", -4),
        ("10 % 4", 2),
        ("1234 * 5678", 7006652),
        ("sqrt(16) + abs(-2)", 6),
        ("max(1, 5, 3)", 5),
    ],
)
def test_the_calculator_does_arithmetic(expression, expected):
    assert evaluate(expression) == expected


def test_the_calculator_knows_constants():
    assert evaluate("round(pi, 2)") == 3.14


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo hi')",  # code execution
        "open('secrets.txt').read()",  # not an allowed function
        "().__class__",  # attribute access
        "[x for x in range(3)]",  # comprehension
        "'a' * 3",  # strings are not numbers
        "True + 1",  # booleans are not numbers here
        "lambda: 1",
        "9 ** 9 ** 9",  # would run for ages
        "1 / 0",
        "2 +",
        "x",
    ],
)
def test_the_calculator_refuses_anything_that_is_not_plain_arithmetic(expression):
    with pytest.raises(ToolError):
        evaluate(expression)


async def test_current_time_in_a_named_zone_and_for_an_unknown_one():
    module = UtilitiesModule(Settings())
    tool = {t.name: t for t in module.get_tools()}["current_time"]

    tokyo = await tool.run('{"timezone": "Asia/Tokyo"}')
    unknown = await tool.run('{"timezone": "Mars/Olympus"}')

    assert re.search(r"JST, UTC\+0900", tokyo.content)
    assert unknown.is_error and "Mars/Olympus" in unknown.content


# --- file system ------------------------------------------------------------------------------------------


@pytest.fixture
def files(tmp_path: Path) -> FileSystemModule:
    workspace = tmp_path / "workspace"
    (workspace / "notes").mkdir(parents=True)
    (workspace / "hello.txt").write_text("hello world", encoding="utf-8")
    (workspace / "notes" / "todo.txt").write_text("buy milk", encoding="utf-8")
    (workspace / "binary.dat").write_bytes(b"\x00\x01\x02")
    (tmp_path / "secret.txt").write_text("outside the workspace", encoding="utf-8")
    return FileSystemModule(Settings(), tmp_path)


def test_list_files_shows_folders_first_then_files(files: FileSystemModule):
    listing = files.list_files()

    assert listing.splitlines()[0] == "notes/"
    assert "hello.txt (11 bytes)" in listing
    assert files.list_files("notes") == "notes/todo.txt (8 bytes)"


def test_read_file_reads_text(files: FileSystemModule):
    assert files.read_file("notes/todo.txt") == "buy milk"


@pytest.mark.parametrize(
    "path",
    [
        "../secret.txt",
        "notes/../../secret.txt",
        "..\\secret.txt",
        "/etc/passwd",
        "C:\\Windows\\win.ini",
    ],
)
def test_the_sandbox_rejects_paths_that_leave_the_workspace(files: FileSystemModule, path: str):
    with pytest.raises(ToolError):
        files.read_file(path)
    with pytest.raises(ToolError):
        files.list_files(path)


def test_the_sandbox_resolves_inside_paths_and_normalises_dots(tmp_path: Path):
    sandbox = PathSandbox(tmp_path)
    assert sandbox.resolve("a/../b.txt") == (tmp_path / "b.txt").resolve()
    assert sandbox.resolve("") == tmp_path.resolve()


def test_read_file_errors_are_helpful(files: FileSystemModule):
    with pytest.raises(ToolError, match="list_files"):
        files.read_file("missing.txt")
    with pytest.raises(ToolError, match="binary"):
        files.read_file("binary.dat")
    with pytest.raises(ToolError, match="not a folder"):
        files.list_files("hello.txt")


def test_long_files_are_truncated(tmp_path: Path):
    settings = Settings()
    settings.modules.filesystem.max_read_chars = 1000
    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace" / "big.txt").write_text("a" * 5000, encoding="utf-8")

    text = FileSystemModule(settings, tmp_path).read_file("big.txt")

    assert text.startswith("a" * 1000) and "truncated at 1,000" in text


async def test_the_workspace_folder_is_created_on_startup(tmp_path: Path):
    module = FileSystemModule(Settings(), tmp_path)
    await module.on_startup()
    assert (tmp_path / "workspace").is_dir()


async def test_a_configured_root_replaces_the_default(tmp_path: Path):
    settings = Settings()
    settings.modules.filesystem.root = str(tmp_path / "elsewhere")
    module = FileSystemModule(settings, tmp_path)
    await module.on_startup()
    assert (tmp_path / "elsewhere").is_dir()


# --- web search -------------------------------------------------------------------------------------------


def searxng(handler) -> WebSearchModule:
    return WebSearchModule(Settings(), transport=httpx.MockTransport(handler))


async def test_web_search_formats_results_and_sends_the_query():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        results = [
            {
                "title": f"Result {n}",
                "url": f"https://example.com/{n}",
                "content": "Some  text\nhere",
            }
            for n in range(1, 8)
        ]
        return httpx.Response(200, json={"results": results})

    tool = searxng(handler).get_tools()[0]
    result = await tool.run('{"query": "otters & beavers", "max_results": 2}')

    assert seen[0].url.params["q"] == "otters & beavers"  # escaped by the client, not by us
    assert seen[0].url.params["format"] == "json"
    assert result.content.startswith("1. Result 1\n   https://example.com/1\n   Some text here")
    assert "3. " not in result.content  # honours max_results


async def test_web_search_with_no_results():
    tool = searxng(lambda r: httpx.Response(200, json={"results": []})).get_tools()[0]
    assert (await tool.run('{"query": "zzz"}')).content == "No results for 'zzz'."


async def test_web_search_explains_a_403_and_a_dead_server():
    forbidden = searxng(lambda r: httpx.Response(403)).get_tools()[0]
    assert "search.formats" in (await forbidden.run('{"query": "x"}')).content

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    dead = searxng(refuse).get_tools()[0]
    result = await dead.run('{"query": "x"}')
    assert result.is_error and "Could not search" in result.content


async def test_web_search_needs_an_address():
    settings = Settings()
    settings.modules.web_search.searxng_url = " "
    result = await WebSearchModule(settings).get_tools()[0].run('{"query": "x"}')
    assert result.is_error and "not set up" in result.content


async def test_pieces_end_at_a_line_never_in_the_middle_of_one():
    from lemonrind.modules.tool import Tool

    async def lines(_arguments: object) -> str:
        return "\n".join(f"line {n:03d} of the listing" for n in range(1, 40))

    class Lines(RecordingModule):
        def get_tools(self) -> list[Tool]:
            return [Tool("lines", "d", {"type": "object", "properties": {}}, lines)]

    registry = ModuleRegistry([Lines(Settings())], max_output_chars=200)

    first = await registry.run(call("lines"))

    body = first.content.split("\n[...")[0]
    assert body.endswith("listing\n")  # whole lines only: the piece stops after a line break
    assert body.count("\n") >= 5 and "start=" in first.content
