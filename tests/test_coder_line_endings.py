"""Edits must work on files whatever their line-ending style (Windows \\r\\n or Unix \\n), and keep that style."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lemonrind.config import Settings
from lemonrind.lemonade import ToolCall
from lemonrind.modules import CoderModule, FileSystemModule, ModuleRegistry


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    root.mkdir()
    return root


@pytest.fixture
def coder(workspace: Path, tmp_path: Path) -> CoderModule:
    files = FileSystemModule(Settings(), tmp_path)
    return CoderModule(files.settings, files)


def test_edits_match_across_line_ending_styles_and_keep_the_files_own_style(
    coder: CoderModule, workspace: Path
):
    windows = workspace / "win.txt"
    windows.write_bytes(b"first\r\nsecond\r\nthird\r\n")
    unix = workspace / "unix.txt"
    unix.write_bytes(b"first\nsecond\nthird\n")

    coder.code_edit_file(
        "win.txt", "first\nsecond", "ONE\nTWO"
    )  # the model writes \n; the file has \r\n
    coder.code_edit_file("unix.txt", "first\r\nsecond", "ONE\nTWO")  # and the other way round

    assert windows.read_bytes() == b"ONE\r\nTWO\r\nthird\r\n"  # still all Windows endings
    assert unix.read_bytes() == b"ONE\nTWO\nthird\n"  # still all Unix endings


def test_a_file_mixing_both_styles_is_saved_in_whichever_it_uses_more(
    coder: CoderModule, workspace: Path
):
    mixed = workspace / "mixed.txt"
    mixed.write_bytes(b"a\r\nb\r\nc\nd\r\n")  # three Windows endings, one Unix

    coder.code_edit_file("mixed.txt", "a", "A")

    assert mixed.read_bytes() == b"A\r\nb\r\nc\r\nd\r\n"


def test_the_permission_question_matches_text_the_same_way(coder: CoderModule, workspace: Path):
    (workspace / "win.txt").write_bytes(b"first\r\nsecond\r\n")
    call = ToolCall(
        "c",
        "code_edit_file",
        json.dumps({"path": "win.txt", "old_text": "first\nsecond", "new_text": "x"}),
    )

    text = ModuleRegistry([coder]).describe(call)

    assert "this edit will fail" not in text and "-first" in text
