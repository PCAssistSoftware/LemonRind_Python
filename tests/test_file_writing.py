"""Writing files: the sandbox, atomic replacement, and the permission question (with its preview)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lemonrind.chats import ChatRepository, Conversation
from lemonrind.config import Settings
from lemonrind.lemonade import Finished, TextDelta, ToolCall, ToolCallsRequested
from lemonrind.lemonade.events import RequestStats
from lemonrind.modules import FileSystemModule, ModuleRegistry, ToolError
from lemonrind.storage import Database


@pytest.fixture
def files(tmp_path: Path) -> FileSystemModule:
    (tmp_path / "workspace").mkdir()
    return FileSystemModule(Settings(), tmp_path)


def tool(files: FileSystemModule, name: str = "write_file"):
    return {t.name: t for t in files.get_tools()}[name]


def write_call(path: str, content: str = "hello") -> ToolCall:
    return ToolCall("c1", "write_file", json.dumps({"path": path, "content": content}))


# --- the write itself ---------------------------------------------------------------------------------------------


def test_a_file_is_created_with_its_folders_and_then_replaced(
    files: FileSystemModule, tmp_path: Path
):
    first = files.write_file("notes/ideas/todo.txt", "buy milk\n")
    second = files.write_file("notes/ideas/todo.txt", "buy bread\n")

    assert first == "Created 'notes/ideas/todo.txt' (9 characters)."
    assert second == "Replaced 'notes/ideas/todo.txt' (10 characters)."
    assert (tmp_path / "workspace" / "notes" / "ideas" / "todo.txt").read_text(
        encoding="utf-8"
    ) == "buy bread\n"


def test_text_is_written_exactly_including_line_endings_and_unicode(
    files: FileSystemModule, tmp_path: Path
):
    text = "line one\r\nline two\ncafé 🍋\n"

    files.write_file("a.txt", text)

    assert (tmp_path / "workspace" / "a.txt").read_bytes() == text.encode(
        "utf-8"
    )  # no newline translation


def test_no_temporary_files_are_left_behind(files: FileSystemModule, tmp_path: Path):
    files.write_file("a.txt", "x")
    files.write_file("a.txt", "y")

    assert [p.name for p in (tmp_path / "workspace").iterdir()] == ["a.txt"]


def test_a_failed_write_leaves_the_old_file_intact_and_no_litter(
    files: FileSystemModule, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    files.write_file("a.txt", "precious")

    def explode(source, target):
        raise OSError("disk exploded")

    monkeypatch.setattr("os.replace", explode)
    with pytest.raises(OSError, match="exploded"):
        files.write_file("a.txt", "new text that never lands")

    assert (tmp_path / "workspace" / "a.txt").read_text(encoding="utf-8") == "precious"
    assert [p.name for p in (tmp_path / "workspace").iterdir()] == ["a.txt"]


@pytest.mark.parametrize(
    "path", ["../escape.txt", "a/../../escape.txt", "/etc/passwd", "C:\\Windows\\x.txt"]
)
def test_writes_outside_the_workspace_are_refused(
    files: FileSystemModule, tmp_path: Path, path: str
):
    with pytest.raises(ToolError):
        files.write_file(path, "nope")

    assert not (tmp_path / "escape.txt").exists()


def test_folders_and_oversized_content_are_refused(files: FileSystemModule, tmp_path: Path):
    (tmp_path / "workspace" / "docs").mkdir()
    files.settings.modules.filesystem.max_write_chars = 1000

    with pytest.raises(ToolError, match="is a folder"):
        files.write_file("docs", "x")
    with pytest.raises(ToolError, match="is a folder"):
        files.write_file("", "x")
    with pytest.raises(ToolError, match="limit"):
        files.write_file("big.txt", "x" * 1001)


# --- permission ---------------------------------------------------------------------------------------------------


def test_every_write_asks_by_default_but_reading_never_does(files: FileSystemModule):
    registry = ModuleRegistry([files])

    assert registry.requires_approval(write_call("a.txt"))
    assert not registry.requires_approval(ToolCall("c", "read_file", '{"path": "a.txt"}'))
    assert not registry.requires_approval(ToolCall("c", "list_files", "{}"))


def test_preapproved_folders_skip_the_question_but_only_inside_them(
    files: FileSystemModule, tmp_path: Path
):
    files.settings.modules.filesystem.preapproved_folders = ["scratch", "../outside"]
    registry = ModuleRegistry([files])

    assert not registry.requires_approval(write_call("scratch/a.txt"))
    assert not registry.requires_approval(write_call("scratch/deep/er/b.txt"))
    assert registry.requires_approval(write_call("other/a.txt"))
    assert registry.requires_approval(
        write_call("scratchy.txt")
    )  # not "inside scratch", merely similar
    assert registry.requires_approval(write_call("a.txt"))
    # a pre-approved folder that is not itself inside the workspace is ignored, never trusted
    assert (
        registry.requires_approval(write_call("../outside/a.txt")) is False
    )  # refused by the sandbox instead
    assert (tmp_path / "outside").exists() is False


def test_a_whole_workspace_can_be_preapproved_with_a_dot(files: FileSystemModule):
    files.settings.modules.filesystem.preapproved_folders = ["."]

    assert not ModuleRegistry([files]).requires_approval(write_call("anything/at/all.txt"))


def test_broken_arguments_ask_rather_than_assume(files: FileSystemModule):
    registry = ModuleRegistry([files])

    assert registry.requires_approval(ToolCall("c", "write_file", "{not json"))
    assert registry.requires_approval(ToolCall("c", "write_file", '{"path": 5}'))


def test_the_permission_question_describes_the_write_readably(
    files: FileSystemModule, tmp_path: Path
):
    registry = ModuleRegistry([files])

    new = registry.describe(write_call("notes/todo.txt", "line one\nline two"))
    assert new.startswith("Write 17 characters to notes/todo.txt\n(creates a new file)")
    assert "line one\nline two" in new  # real line breaks, not JSON "\n" escapes

    files.write_file("notes/todo.txt", "old content")
    replacing = registry.describe(write_call("notes/todo.txt", "new"))
    assert "REPLACES the existing file (11 bytes)" in replacing

    long = registry.describe(write_call("big.txt", "y" * 5000))
    assert "3,000 more characters" in long and long.count("y") == 2000
    assert "outside the workspace" in registry.describe(write_call("../x.txt"))
    assert registry.describe(ToolCall("c", "nonexistent", '{"a": 1}')) == '{"a": 1}'


def test_tools_without_a_preview_show_their_arguments_laid_out():
    from lemonrind.modules import tool_from_function

    def ping(host: str) -> str:
        """Ping."""
        return host

    assert tool_from_function(ping).describe('{"host": "a"}') == '{\n  "host": "a"\n}'
    assert tool_from_function(ping).describe("{broken") == "{broken"


# --- through a conversation ---------------------------------------------------------------------------------------


class WritingModel:
    """Asks to write a file, then thanks the user."""

    def __init__(self) -> None:
        self.calls = 0

    async def stream_chat(self, messages, model, **kwargs):
        self.calls += 1
        if self.calls == 1:
            yield ToolCallsRequested((write_call("report.txt", "Quarterly numbers"),))
            yield Finished(RequestStats(), "tool_calls")
        else:
            yield TextDelta("Saved.")
            yield Finished(RequestStats(), "stop")


@pytest.mark.parametrize("allowed", [True, False])
async def test_the_file_appears_only_after_the_user_allows_it(
    files: FileSystemModule, tmp_path: Path, allowed: bool
):
    registry = ModuleRegistry([files])
    conversation = Conversation(
        client=WritingModel(),
        repo=ChatRepository(Database(":memory:")),
        system_prompt="x",
        model="m",
        tools=registry,
    )
    shown: list[str] = []

    async def approve(call: ToolCall) -> bool:
        shown.append(conversation.describe_call(call))
        return allowed

    conversation.approve = approve

    await conversation.send("write the report")

    assert (tmp_path / "workspace" / "report.txt").exists() is allowed
    assert shown and "Write 17 characters to report.txt" in shown[0]
