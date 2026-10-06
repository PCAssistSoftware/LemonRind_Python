"""Tests for the Coder module: finding, searching, outlining, reading, and checked edits."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lemonrind.config import Settings
from lemonrind.lemonade import ToolCall
from lemonrind.modules import CoderModule, FileSystemModule, ModuleRegistry, ToolError
from lemonrind.modules import coder as coder_module
from lemonrind.modules.coder import check_syntax, outline_python

APP = '''"""A small app."""

LIMIT = 10
NAME, OTHER = "app", "x"
lowercase = 1


@decorator(1)
def helper(a: int, *, b: str = "x") -> bool:
    """Check things.

    More detail here.
    """
    return True


async def fetch(url):
    return url


class Greeter(Base, metaclass=Meta):
    """Says hello."""

    def greet(self, name: str) -> str:
        return f"hi {name}"

    async def later(self) -> None:
        pass

    @property
    def title(self):
        return "t"


class Plain:
    pass
'''


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    (root / "src").mkdir(parents=True)
    (root / "src" / "app.py").write_text(APP, encoding="utf-8")
    (root / "src" / "util.py").write_text(
        "def double(x):\n    return x * 2\n\n\ndef triple(x):\n    return x * 3\n", encoding="utf-8"
    )
    (root / "README.md").write_text("# Title\nTODO: write docs\n", encoding="utf-8")
    (root / "data.json").write_text('{"a": 1}\n', encoding="utf-8")
    (root / ".venv").mkdir()
    (root / ".venv" / "ignored.py").write_text("TODO hidden\n", encoding="utf-8")
    (root / "node_modules").mkdir()
    (root / "node_modules" / "x.js").write_text("TODO hidden\n", encoding="utf-8")
    (root / "image.bin").write_bytes(b"\x00\x01TODO")
    return root


@pytest.fixture
def coder(project: Path, tmp_path: Path) -> CoderModule:
    files = FileSystemModule(Settings(), tmp_path)
    return CoderModule(files.settings, files)


# --- finding files -------------------------------------------------------------------------------------------------------


def test_glob_finds_files_by_pattern_and_skips_noise_folders(coder: CoderModule):
    assert coder.code_glob("**/*.py") == "src/app.py\nsrc/util.py"  # not .venv/ignored.py
    assert coder.code_glob("src/*.py") == "src/app.py\nsrc/util.py"
    assert coder.code_glob("*.md") == "README.md"
    assert coder.code_glob("**/*.rs") == "No files match '**/*.rs'."
    assert "x.js" not in coder.code_glob("**/*")


@pytest.mark.parametrize("pattern", ["", "  ", "../*", "/etc/*", "C:\\Windows\\*", "src/../../*"])
def test_glob_refuses_empty_absolute_and_climbing_patterns(coder: CoderModule, pattern):
    with pytest.raises(ToolError):
        coder.code_glob(pattern)


def test_glob_results_are_capped(coder: CoderModule, project: Path):
    for n in range(320):
        (project / f"f{n:03}.txt").write_text("x", encoding="utf-8")

    result = coder.code_glob("*.txt")

    assert result.count("\n") == 300 and "and 20 more" in result


# --- searching contents ------------------------------------------------------------------------------------------------------


def test_grep_returns_path_line_text_and_skips_noise_and_binary_files(coder: CoderModule):
    assert (
        coder.code_grep("TODO") == "README.md:2: TODO: write docs"
    )  # not the .venv, node_modules or binary copies


def test_grep_can_limit_files_and_ignore_case(coder: CoderModule):
    assert coder.code_grep("DEF (double|triple)", "src/*.py") == "No matches."
    result = coder.code_grep("DEF (double|triple)", "src/*.py", ignore_case=True)
    assert result == "src/util.py:1: def double(x):\nsrc/util.py:5: def triple(x):"


def test_grep_explains_a_bad_pattern_and_stops_at_the_cap(coder: CoderModule, project: Path):
    with pytest.raises(ToolError, match="not a valid regular expression"):
        coder.code_grep("(unclosed")
    with pytest.raises(ToolError):
        coder.code_grep("")
    with pytest.raises(ToolError, match="too long"):
        coder.code_grep("a" * 301)
    (project / "many.txt").write_text("hit\n" * 200, encoding="utf-8")

    result = coder.code_grep("hit", "many.txt")

    assert result.count("\n") == 150 and "stopped after 150 matches" in result


def test_grep_reports_a_pattern_that_is_too_slow(
    coder: CoderModule, monkeypatch: pytest.MonkeyPatch
):
    class Slow:
        def search(self, text, timeout=None):
            raise TimeoutError

    # Patch through the imported object, not a dotted-name string: the NiceGUI test fixture removes the
    # ``lemonrind`` package from sys.modules after a UI test, which breaks name-based lookups that run later.
    monkeypatch.setattr(coder_module.regex, "compile", lambda pattern, flags=0: Slow())

    with pytest.raises(ToolError, match="too slow"):
        coder.code_grep("anything")


# --- outlines ----------------------------------------------------------------------------------------------------------------


def test_the_outline_shows_signatures_and_first_docstring_lines_without_bodies():
    outline = outline_python(APP)

    assert "LIMIT = ...  # line 3" in outline and "NAME, OTHER = ...  # line 4" in outline
    assert "lowercase" not in outline  # only CONSTANTS are listed
    assert (
        "@decorator(1)\ndef helper(a: int, *, b: str = 'x') -> bool  # line 9" not in outline
    )  # (line numbers differ)
    assert "def helper(a: int, *, b: str='x') -> bool:  # line 10" not in outline
    assert "@decorator(1)" in outline and "def helper(" in outline
    assert '    """Check things."""' in outline and "More detail" not in outline
    assert "async def fetch(url)" in outline
    assert (
        "class Greeter(Base, metaclass=Meta):" in outline
        and "    def greet(self, name: str) -> str" in outline
    )
    assert "    async def later(self) -> None" in outline and "    @property" in outline
    assert "class Plain:" in outline
    assert "return" not in outline and "hi {name}" not in outline


def test_outline_through_the_tool_and_its_refusals(coder: CoderModule, project: Path):
    assert "class Greeter" in coder.code_outline("src/app.py")
    with pytest.raises(ToolError, match="only available for Python"):
        coder.code_outline("README.md")
    (project / "bad.py").write_text("def broken(:\n", encoding="utf-8")
    with pytest.raises(ToolError, match="syntax error"):
        coder.code_outline("bad.py")
    with pytest.raises(ToolError, match="not a file"):
        coder.code_outline("nope.py")
    assert outline_python("x = 1\n") == "(no classes, functions or constants)"


# --- reading ------------------------------------------------------------------------------------------------------------------


def test_reading_shows_line_numbers_and_ranges(coder: CoderModule):
    whole = coder.code_read_file("src/util.py")
    assert whole.startswith("src/util.py (lines 1-6 of 6)") and "1| def double(x):" in whole

    part = coder.code_read_file("src/util.py", 5, 6)
    assert part == "src/util.py (lines 5-6 of 6)\n5| def triple(x):\n6|     return x * 3"

    with pytest.raises(ToolError, match="only 6 lines"):
        coder.code_read_file("src/util.py", 99)
    with pytest.raises(ToolError, match="binary"):
        coder.code_read_file("image.bin")


def test_a_long_file_is_read_in_capped_pieces(coder: CoderModule, project: Path):
    (project / "long.txt").write_text(
        "\n".join(f"line {n}" for n in range(1, 2501)), encoding="utf-8"
    )

    first = coder.code_read_file("long.txt")
    rest = coder.code_read_file("long.txt", 2001)

    assert "(lines 1-2000 of 2500)" in first and "ask for another range" in first
    assert "(lines 2001-2500 of 2500)" in rest and "ask for another range" not in rest


# --- syntax checking -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "text", "ok"),
    [
        ("a.py", "x = 1\n", True), ("a.py", "def f(:\n", False),
        ("a.json", '{"a": 1}', True), ("a.json", '{"a": }', False),
        ("a.toml", "a = 1\n", True), ("a.toml", "a = = 1\n", False),
        ("a.csproj", '<Project Sdk="x"><PropertyGroup/></Project>', True),
        ("a.axaml", "<Window><Grid></Window>", False),
        ("a.xml", "", False),
        ("a.txt", "any ( [ { text", True), ("a.md", "# ok", True),
    ],
)  # fmt: skip
def test_syntax_is_checked_only_for_kinds_we_can_check(name, text, ok):
    assert (check_syntax(Path(name), text) is None) is ok


# --- editing -----------------------------------------------------------------------------------------------------------------


def test_an_edit_replaces_one_exact_block_and_nothing_else(coder: CoderModule, project: Path):
    result = coder.code_edit_file("src/util.py", "return x * 2", "return x + x")

    assert result == "Edited 'src/util.py': replaced 1 line(s) with 1."
    assert (project / "src" / "util.py").read_text(encoding="utf-8") == (
        "def double(x):\n    return x + x\n\n\ndef triple(x):\n    return x * 3\n"
    )


def test_an_edit_can_delete_a_block_and_handles_multiple_lines(coder: CoderModule, project: Path):
    coder.code_edit_file("src/util.py", "\n\ndef triple(x):\n    return x * 3\n", "")

    assert (project / "src" / "util.py").read_text(
        encoding="utf-8"
    ) == "def double(x):\n    return x * 2\n"


def test_ambiguous_missing_and_empty_edits_are_refused_with_advice(
    coder: CoderModule, project: Path
):
    before = (project / "src" / "util.py").read_text(encoding="utf-8")

    with pytest.raises(ToolError, match="matches 2 places"):
        coder.code_edit_file("src/util.py", "return x", "return y")
    with pytest.raises(ToolError, match="not found"):
        coder.code_edit_file("src/util.py", "no such text", "x")
    with pytest.raises(ToolError, match="empty"):
        coder.code_edit_file("src/util.py", "", "x")
    with pytest.raises(ToolError, match="not a file"):
        coder.code_edit_file("src/missing.py", "a", "b")

    assert (project / "src" / "util.py").read_text(encoding="utf-8") == before


def test_an_edit_that_breaks_syntax_is_refused_and_the_file_is_untouched(
    coder: CoderModule, project: Path
):
    before = (project / "src" / "util.py").read_text(encoding="utf-8")

    with pytest.raises(ToolError, match=r"syntax error \(line 1"):
        coder.code_edit_file("src/util.py", "def double(x):", "def double(x")
    with pytest.raises(ToolError, match="syntax error"):
        coder.code_edit_file("data.json", '"a": 1', '"a": ')

    assert (project / "src" / "util.py").read_text(encoding="utf-8") == before


def test_a_file_that_was_already_broken_can_still_be_edited_so_it_can_be_fixed(
    coder: CoderModule, project: Path
):
    (project / "broken.py").write_text("def f(:\n    pass\n", encoding="utf-8")

    coder.code_edit_file("broken.py", "def f(:", "def f():")  # fixes it

    assert (project / "broken.py").read_text(encoding="utf-8") == "def f():\n    pass\n"


def test_non_code_files_are_edited_without_a_syntax_check(coder: CoderModule, project: Path):
    coder.code_edit_file("README.md", "# Title", "# Title ((( unbalanced")

    assert "unbalanced" in (project / "README.md").read_text(encoding="utf-8")


# --- creating, deleting, moving ----------------------------------------------------------------------------------------------


def test_creating_makes_folders_checks_syntax_and_never_overwrites(
    coder: CoderModule, project: Path
):
    assert (
        coder.code_create_file("pkg/mod.py", "x = 1\ny = 2\n") == "Created 'pkg/mod.py' (2 lines)."
    )
    assert (project / "pkg" / "mod.py").read_text(encoding="utf-8") == "x = 1\ny = 2\n"

    with pytest.raises(ToolError, match="already exists"):
        coder.code_create_file("pkg/mod.py", "z = 3\n")
    with pytest.raises(ToolError, match="syntax error"):
        coder.code_create_file("pkg/bad.py", "def (:\n")
    assert not (project / "pkg" / "bad.py").exists()
    with pytest.raises(ToolError):
        coder.code_create_file("../escape.py", "x = 1\n")


def test_deleting_and_moving_work_inside_the_sandbox_only(coder: CoderModule, project: Path):
    coder.code_move_file("src/util.py", "lib/helpers/util2.py")
    assert (
        not (project / "src" / "util.py").exists()
        and (project / "lib" / "helpers" / "util2.py").is_file()
    )

    with pytest.raises(ToolError, match="already exists"):
        coder.code_move_file("README.md", "data.json")
    with pytest.raises(ToolError):
        coder.code_move_file("README.md", "../README.md")

    assert coder.code_delete_file("README.md") == "Deleted 'README.md'."
    assert coder.code_delete_folder("lib") == "Deleted the folder 'lib' and everything in it."
    assert not (project / "lib").exists()

    with pytest.raises(ToolError, match="cannot be deleted"):
        coder.code_delete_folder("")
    with pytest.raises(ToolError, match="not a folder"):
        coder.code_delete_folder("nope")
    with pytest.raises(ToolError, match="not a file"):
        coder.code_delete_file("src")


# --- permission --------------------------------------------------------------------------------------------------------------


def edit_call(path: str, old: str = "return x * 2", new: str = "return x + x") -> ToolCall:
    return ToolCall(
        "c", "code_edit_file", json.dumps({"path": path, "old_text": old, "new_text": new})
    )


def test_reading_and_searching_never_ask_but_changes_always_do(coder: CoderModule):
    registry = ModuleRegistry([coder])
    read_only = ("code_glob", "code_grep", "code_outline", "code_read_file")
    changing = (
        "code_edit_file",
        "code_create_file",
        "code_delete_file",
        "code_delete_folder",
        "code_move_file",
    )

    assert not any(
        registry.requires_approval(ToolCall("c", n, '{"path": "src/app.py", "pattern": "x"}'))
        for n in read_only
    )
    assert all(
        registry.requires_approval(ToolCall("c", n, '{"path": "src/app.py"}')) for n in changing
    )


def test_edits_and_creations_skip_the_question_in_preapproved_folders_but_deletes_never_do(
    coder: CoderModule,
):
    coder.settings.modules.filesystem.preapproved_folders = ["src"]
    registry = ModuleRegistry([coder])

    assert not registry.requires_approval(edit_call("src/util.py"))
    assert registry.requires_approval(edit_call("README.md"))
    assert not registry.requires_approval(
        ToolCall("c", "code_create_file", '{"path": "src/new.py", "content": ""}')
    )
    assert registry.requires_approval(ToolCall("c", "code_delete_file", '{"path": "src/util.py"}'))
    assert registry.requires_approval(
        ToolCall("c", "code_move_file", '{"path": "src/util.py", "new_path": "src/x.py"}')
    )


def test_the_edit_question_shows_a_real_diff(coder: CoderModule):
    text = ModuleRegistry([coder]).describe(edit_call("src/util.py"))

    assert text.startswith("Edit src/util.py\n\n--- src/util.py (before)\n+++ src/util.py (after)")
    assert "-    return x * 2" in text and "+    return x + x" in text


def test_the_question_warns_when_an_edit_cannot_succeed(coder: CoderModule):
    registry = ModuleRegistry([coder])

    assert "matches 2 places" in registry.describe(edit_call("src/util.py", old="return x"))
    assert "matches 0 places" in registry.describe(
        edit_call("src/util.py", old="nothing like this")
    )
    assert "Edit nope.py:" in registry.describe(edit_call("nope.py"))


def test_the_other_questions_say_what_is_at_stake(coder: CoderModule, project: Path):
    registry = ModuleRegistry([coder])
    (project / "pkg").mkdir()
    (project / "pkg" / "a.py").write_text("x = 1", encoding="utf-8")
    (project / "pkg" / "b.py").write_text("x = 2", encoding="utf-8")

    assert "PERMANENTLY DELETE the folder pkg and the 2 file(s) inside it" in registry.describe(
        ToolCall("c", "code_delete_folder", '{"path": "pkg"}')
    )
    assert "PERMANENTLY DELETE the file README.md" in registry.describe(
        ToolCall("c", "code_delete_file", '{"path": "README.md"}')
    )
    assert (
        registry.describe(ToolCall("c", "code_move_file", '{"path": "a", "new_path": "b"}'))
        == "Move a  ->  b"
    )
    created = registry.describe(
        ToolCall("c", "code_create_file", '{"path": "n.py", "content": "x = 1\\ny = 2"}')
    )
    assert created.startswith("Create n.py (2 lines)") and "y = 2" in created


# --- C# and Visual Basic (tree-sitter, no .NET compiler) ---------------------------------------------------------------------

CSHARP = """namespace Demo;

public class Greeter
{
    public string Greet(string name)
    {
        return $"Hello, {name}";
    }
}
"""

VB = """Public Class Greeter
    Public Function Greet(name As String) As String
        If name Is Nothing Then
            Return "Hello"
        End If
        Return "Hello, " & name
    End Function
End Class
"""


def test_a_csharp_edit_that_breaks_the_syntax_is_refused_and_the_file_is_untouched(
    coder: CoderModule, project: Path
):
    target = project / "Greeter.cs"
    target.write_text(CSHARP, encoding="utf-8")

    coder.code_edit_file("Greeter.cs", 'return $"Hello, {name}";', 'return $"Hi, {name}";')  # fine
    with pytest.raises(ToolError, match=r"syntax error \(near line"):
        coder.code_edit_file(
            "Greeter.cs", "public string Greet(string name)", "public string Greet(string name"
        )

    assert "Hi, {name}" in target.read_text(
        encoding="utf-8"
    ) and "(string name)" in target.read_text(encoding="utf-8")


def test_a_new_csharp_file_with_a_syntax_error_is_refused(coder: CoderModule, project: Path):
    with pytest.raises(ToolError, match="syntax error"):
        coder.code_create_file("Broken.cs", "public class A { void M() { int x = ; } ")
    assert not (project / "Broken.cs").exists()
    assert "Created" in coder.code_create_file("Fine.cs", CSHARP)


def test_a_visual_basic_edit_that_breaks_the_syntax_is_saved_with_a_warning(
    coder: CoderModule, project: Path
):
    target = project / "Greeter.vb"
    target.write_text(VB, encoding="utf-8")

    fine = coder.code_edit_file("Greeter.vb", 'Return "Hello, " & name', 'Return "Hi, " & name')
    assert "Warning" not in fine

    broken = coder.code_edit_file("Greeter.vb", "If name Is Nothing Then", "If name Is Nothing")
    assert "Warning: the Visual Basic checker" in broken and "near line" in broken
    assert "If name Is Nothing\n" in target.read_text(encoding="utf-8")  # VB is only warned about


def test_a_file_the_checker_already_misreads_can_still_be_edited_without_a_refusal(
    coder: CoderModule, project: Path
):
    target = project / "Old.cs"
    target.write_text(
        CSHARP.replace("return $", "return $$$ !!"), encoding="utf-8"
    )  # broken before the edit

    result = coder.code_edit_file(
        "Old.cs", "Greet(string name)", "Greet(string who)"
    )  # makes nothing worse

    assert result.startswith("Edited")


def test_dotnet_files_are_only_recognised_by_their_extension():
    from lemonrind.modules.dotnet_syntax import error_lines, kind_of

    assert kind_of(Path("A.cs")) == "csharp" and kind_of(Path("a.VB")) == "vb"
    assert kind_of(Path("a.py")) is None and kind_of(Path("a.csproj")) is None
    assert error_lines("csharp", CSHARP) == [] and error_lines("vb", VB) == []
    assert error_lines("csharp", "class { ") != []


def test_a_visual_basic_edit_that_drops_an_end_statement_names_the_missing_block(
    coder: CoderModule, project: Path
):
    (project / "Greeter.vb").write_text(VB, encoding="utf-8")

    result = coder.code_edit_file("Greeter.vb", "        End If\n", "")

    assert "'If' block seems not to be closed (End If missing)" in result


def test_visual_basic_block_balance_understands_the_forms_real_code_uses():
    from lemonrind.modules.dotnet_syntax import vb_balance, vb_block_notes

    balanced = """Imports System
Namespace Demo
    Public Interface IThing
        Sub Run()
        Function Name() As String
        Property Size As Integer
    End Interface

    Public Class Thing
        Implements IThing
        Public Property Label As String = "x"
        Private _n As Integer
        Public Property Count As Integer
            Get
                Return _n
            End Get
            Set(value As Integer)
                _n = value
            End Set
        End Property
        <Obsolete("x")> Public MustOverride Sub Nothing1()

        Public Async Function Go(a As Integer, b As Integer) As Task(Of Integer)
            If a > 0 AndAlso
               b > 0 Then
                Return 1
            ElseIf a = 0 Then
                Return 0
            Else
                Return 2
            End If
            If a = 1 Then Return 5
            For Each x In items
                Dim s = "End If ' not code"
                ' End If in a comment
            Next
            Do While a < 3
                a += 1
            Loop
            Dim f = Function(q) q + 1
            Dim g = Await Task.Run(Async Function() As Task(Of Integer)
                                       Return 1
                                   End Function)
            Using c = Open()
                Try
                    Select Case a
                        Case 1
                    End Select
                Finally
                End Try
            End Using
            Return 0
        End Function
    End Class
End Namespace
"""
    assert vb_balance(balanced) == ({}, {})
    assert vb_block_notes(None, balanced) == []
    # an edit that adds a whole block keeps things balanced; one that adds only half does not
    assert (
        vb_block_notes(
            balanced,
            balanced.replace(
                "Return 0\n        End Function",
                "If True Then\n Return 1\n End If\n Return 0\n        End Function",
            ),
        )
        == []
    )
    assert vb_block_notes(balanced, balanced.replace("Loop\n", "")) == [
        "a 'Do' block seems not to be closed (Loop missing)"
    ]
    assert vb_block_notes(balanced, balanced + "End Sub\n") == [
        "there is an extra 'End Sub' with no matching opening line"
    ]


def test_xml_that_declares_entities_is_never_parsed_so_an_expansion_bomb_costs_nothing():
    import time

    bomb = (
        '<?xml version="1.0"?><!DOCTYPE lolz [<!ENTITY lol "lol">'
        '<!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">'
        '<!ENTITY lol3 "&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;">]>'
        "<lolz>&lol3;</lolz>"
    )
    started = time.monotonic()

    assert (
        check_syntax(Path("evil.xml"), bomb) is None
    )  # not checked at all (neither refused nor expanded)
    assert (
        check_syntax(Path("logo.svg"), "<!DOCTYPE svg><svg><g></svg>") is None
    )  # a DOCTYPE is enough to skip it
    assert time.monotonic() - started < 1
    assert (
        check_syntax(Path("a.csproj"), "<Project><PropertyGroup></Project>") is not None
    )  # the usual check stays
