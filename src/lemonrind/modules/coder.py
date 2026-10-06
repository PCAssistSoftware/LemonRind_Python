"""The Coder module: code-aware tools for reading, searching and changing a project.

Point the File system module's workspace at a code project (``modules.filesystem.root``) and this module lets the
assistant work on it the way a programmer does: find files, search their contents, read just the part it needs,
get a file's *outline* without its bodies, and make **small precise edits** that are checked before they are saved.

The tools:

===================  ==============================================================================================
``code_glob``        find files by name pattern (``**/*.py``), skipping noise folders such as ``.git`` and ``.venv``
``code_grep``        search file *contents* with a regular expression (with a time limit), as ``path:line: text``
``code_outline``     a Python file's classes and function signatures, with no bodies, to learn its shape cheaply
``code_read_file``   read a file with line numbers, a line range at a time, capped so it cannot flood the context
``code_edit_file``   replace an exact block of text that occurs exactly once; asks permission and shows a diff
``code_create_file`` create a new file (never overwrites); asks permission
``code_delete_file`` / ``code_delete_folder`` / ``code_move_file``  always ask permission
===================  ==============================================================================================

Design points worth understanding:

* **Edits replace text, not line numbers.** A model that read a file a moment ago and says "change lines 40 to 45"
  will be wrong if anything shifted. "Replace this exact text with that" fails safely instead: the old text must match
  exactly once, so an ambiguous or stale edit is refused and the model is told to include more context.
* **Edits are syntax-checked.** For Python, JSON, TOML, C# and XML-based project and markup files (.csproj, .xaml) the result is parsed before it is saved (Visual Basic too, but only as a warning: see dotnet_syntax.py). An edit that
  would turn a valid file into a broken one is refused, with the line number. (A file that was *already* broken may
  be edited, so the model can fix it.)
* **The permission question shows a diff**, not "edit allowed?": what would actually change, line by line.
* **Everything stays in the sandbox** shared with the File system module, and creating or editing honours its
  ``preapproved_folders`` setting. Deleting and moving always ask.

Python ideas used here:

* The ``ast`` module: parse Python source into a tree to find classes and functions (for the outline) or to check
  that source is valid, without running it.
* ``difflib.unified_diff`` for the familiar ``+``/``-`` diff.
* ``regex`` (a drop-in for ``re``) because it supports a **timeout**, protecting against patterns that take forever.
* ``Path.rglob``/``Path.glob`` and ``fnmatch``-style patterns for finding files.
"""

from __future__ import annotations

import ast
import dataclasses
import difflib
import json
import shutil
import tomllib

# The standard XML parser is used only to check that a document is well formed, and never on text that declares a
# DOCTYPE or ENTITY (see _declares_entities), which is what the usual attacks rely on (hence the scanner exemptions).
import xml.etree.ElementTree as ET  # nosec B405
from collections.abc import Callable
from pathlib import Path

import regex

from lemonrind.config import Settings
from lemonrind.modules import dotnet_syntax
from lemonrind.modules.base import Module
from lemonrind.modules.filesystem import FileSystemModule, PathSandbox, write_text_atomic
from lemonrind.modules.tool import Tool, ToolError, tool_from_function

MAX_READ_LINES = 2000  # one read can never dump more than this into the model's context
MAX_GLOB_RESULTS = 300
MAX_GREP_RESULTS = 150
MAX_GREP_FILE_BYTES = 1_000_000  # larger files are skipped by grep (generated data, logs, binaries)
MAX_LINE_CHARS = 400  # a longer line is cut when shown, and matched only up to here
REGEX_TIMEOUT_SECONDS = 3.0
MAX_DIFF_LINES = 80
NOISE_FOLDERS = frozenset(
    {".git", ".hg", ".svn", ".idea", ".vs", ".vscode", "node_modules", "__pycache__", ".venv", "venv",
     "env", "bin", "obj", "dist", "build", ".mypy_cache", ".pytest_cache", ".ruff_cache", "target"}
)  # fmt: skip


# --- syntax checking -----------------------------------------------------------------------------------------------

# Project and markup files that are plain XML (so "is it well formed?" is a real check), including the .NET ones.
XML_SUFFIXES = frozenset(
    {".xml", ".csproj", ".vbproj", ".fsproj", ".props", ".targets", ".resx", ".config", ".xaml", ".axaml",
     ".slnx", ".svg", ".xsd", ".nuspec"}
)  # fmt: skip


def _declares_entities(text: str) -> bool:
    """Does an XML text have a ``<!DOCTYPE`` or ``<!ENTITY``?

    Entities are how a small XML file can expand into gigabytes (the "billion laughs" trick), and the text checked here
    is written by a model that may have been steered by a web page. Project and markup files never need them, so a
    document that has one is simply not checked (it is neither refused nor parsed). Without a DOCTYPE no entity can be
    declared, so everything else is safe to parse with the standard library.
    """
    lowered = text.lower()
    return "<!doctype" in lowered or "<!entity" in lowered


def check_syntax(path: Path, text: str) -> str | None:
    """``None`` if ``text`` is valid for the kind of file ``path`` names (or the kind is not checked); else a message."""
    suffix = path.suffix.lower()
    try:
        if suffix in (".py", ".pyi"):
            ast.parse(text)
        elif suffix == ".json":
            json.loads(text)
        elif suffix == ".toml":
            tomllib.loads(text)
        elif suffix in XML_SUFFIXES and not _declares_entities(text):
            ET.fromstring(text)  # nosec B314
    except ET.ParseError as error:  # checked before SyntaxError, which it is a kind of
        return str(error)  # for example "mismatched tag: line 1, column 54"
    except SyntaxError as error:
        return f"line {error.lineno}: {error.msg}"
    except json.JSONDecodeError as error:
        return f"line {error.lineno}: {error.msg}"
    except tomllib.TOMLDecodeError as error:
        return str(error)
    return None


# --- outlines -----------------------------------------------------------------------------------------------------------


def outline_python(source: str) -> str:
    """Classes, functions and their signatures from Python ``source``, with the first line of each docstring."""
    tree = ast.parse(source)
    lines: list[str] = []

    def doc(node: ast.AST, indent: str) -> None:
        text = ast.get_docstring(node)  # type: ignore[arg-type]
        if text:
            lines.append(f'{indent}    """{text.splitlines()[0]}"""')

    def function(node: ast.FunctionDef | ast.AsyncFunctionDef, indent: str) -> None:
        keyword = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
        for decorator in node.decorator_list:
            lines.append(f"{indent}@{ast.unparse(decorator)}")
        lines.append(
            f"{indent}{keyword} {node.name}({ast.unparse(node.args)}){returns}  # line {node.lineno}"
        )
        doc(node, indent)

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            bases = ", ".join(
                [ast.unparse(b) for b in node.bases]
                + [f"{k.arg}={ast.unparse(k.value)}" for k in node.keywords if k.arg]
            )
            lines.append(
                f"class {node.name}({bases}):  # line {node.lineno}"
                if bases
                else f"class {node.name}:  # line {node.lineno}"
            )
            doc(node, "")
            for member in node.body:
                if isinstance(member, ast.FunctionDef | ast.AsyncFunctionDef):
                    function(member, "    ")
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            function(node, "")
        elif isinstance(node, ast.Assign | ast.AnnAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)]
            if names and all(name.isupper() for name in names):  # only CONSTANTS are worth listing
                lines.append(f"{', '.join(names)} = ...  # line {node.lineno}")
    return "\n".join(lines) or "(no classes, functions or constants)"


# --- the module ---------------------------------------------------------------------------------------------------------


class CoderModule(Module):
    name = "Coder"
    config_key = "coder"
    description = "Code-aware tools over the workspace folder: find, search, outline, read, and checked edits."
    enabled_by_default = True

    def __init__(self, settings: Settings, files: FileSystemModule) -> None:
        super().__init__(settings)
        self._files = files  # shares its sandbox folder and its pre-approved folders

    # --- helpers ------------------------------------------------------------------------------------------------------

    @property
    def sandbox(self) -> PathSandbox:
        return self._files.sandbox

    def _existing_file(self, path: str) -> Path:
        target = self.sandbox.resolve(path)
        if not target.is_file():
            raise ToolError(
                f"'{path}' is not a file in the workspace. Use code_glob to find files."
            )
        return target

    @staticmethod
    def _load(target: Path) -> tuple[str, str]:
        """The file's text with ``\n`` line endings, and the line ending the file itself uses.

        A model writes ``\n`` between lines, but a file made on Windows has ``\r\n``, so a multi-line block copied
        from the model would never match it. Edits are therefore made on the normalised text, and the file's own
        line-ending style is put back when saving, so an edit never changes every line ending in the file.
        (A file mixing both styles is saved with whichever it uses more.)
        """
        data = target.read_bytes()
        if b"\x00" in data[:8000]:
            raise ToolError(f"'{target.name}' looks like a binary file, not text.")
        text = data.decode("utf-8", errors="replace")
        crlf = data.count(b"\r\n")
        lf = data.count(b"\n") - crlf
        return text.replace("\r\n", "\n"), ("\r\n" if crlf > lf else "\n")

    @staticmethod
    def _read(target: Path) -> str:
        data = target.read_bytes()
        if b"\x00" in data[:8000]:
            raise ToolError(f"'{target.name}' looks like a binary file, not text.")
        return data.decode("utf-8", errors="replace")

    def _visible(self, path: Path) -> bool:
        """Is this path outside every noise folder (``.git``, ``node_modules``, ``.venv``...)?"""
        relative = path.relative_to(self.sandbox.root)
        return not any(part in NOISE_FOLDERS for part in relative.parts[:-1])

    def _write_check(self, label: str, path: Path, new_text: str, old_text: str | None) -> str:
        """Refuse a write that would break a file's syntax (unless it was already broken before the edit).

        Returns a warning to add to the tool's answer, or ``""``: only Visual Basic gets a warning, because its
        checker is approximate (see ``dotnet_syntax.py``). C# problems refuse the write like the other languages.
        """
        problem = check_syntax(path, new_text)
        if problem is not None and (old_text is None or check_syntax(path, old_text) is None):
            raise ToolError(
                f"Not saved: this {label} would leave '{path.name}' with a syntax error ({problem}). "
                "Fix the new text and try again."
            )
        kind = dotnet_syntax.kind_of(path)
        if kind is None:
            return ""
        spots = dotnet_syntax.new_problems(kind, old_text, new_text)
        if kind in dotnet_syntax.REFUSING_KINDS:
            if spots:
                raise ToolError(
                    f"Not saved: this {label} would leave '{path.name}' with a syntax error "
                    f"({dotnet_syntax.describe(spots)}). Fix the new text and try again."
                )
            return ""
        findings = [f"a possible syntax problem {dotnet_syntax.describe(spots)}"] if spots else []
        findings += dotnet_syntax.vb_block_notes(old_text, new_text)
        if not findings:
            return ""
        return (
            "\nWarning: the Visual Basic checker (approximate) found: "
            + "; ".join(findings)
            + ". The change was saved; read that part again to be sure it is right."
        )

    # --- tools --------------------------------------------------------------------------------------------------------

    def get_tools(self) -> list[Tool]:
        def asking(
            tool: Tool, check: Callable[[dict], bool], preview: Callable[[dict], str]
        ) -> Tool:
            return dataclasses.replace(
                tool, requires_approval=True, approval_check=check, preview=preview
            )

        def always(tool: Tool, preview: Callable[[dict], str]) -> Tool:
            return dataclasses.replace(tool, requires_approval=True, preview=preview)

        return [
            tool_from_function(self.code_glob),
            tool_from_function(self.code_grep),
            tool_from_function(self.code_outline),
            tool_from_function(self.code_read_file),
            asking(
                tool_from_function(self.code_edit_file),
                self._edit_needs_approval,
                self._describe_edit,
            ),
            asking(
                tool_from_function(self.code_create_file),
                self._create_needs_approval,
                self._describe_create,
            ),
            always(tool_from_function(self.code_delete_file), self._describe_delete),
            always(tool_from_function(self.code_delete_folder), self._describe_delete),
            always(tool_from_function(self.code_move_file), self._describe_move),
        ]

    def code_glob(self, pattern: str) -> str:
        """Find files by name pattern inside the workspace. Common noise folders (.git, node_modules, .venv, ...) are skipped.

        Args:
            pattern: A glob such as "**/*.py" (every Python file, at any depth) or "src/*.py" (one folder only).
        """
        pattern = pattern.strip()
        if not pattern:
            raise ToolError("Give a glob pattern, for example '**/*.py'.")
        looks_absolute = (
            pattern.startswith(("/", "\\"))
            or Path(pattern).is_absolute()
            or bool(Path(pattern).drive)
        )
        if looks_absolute or ".." in Path(pattern).parts:
            raise ToolError(
                "The pattern must be relative to the workspace and may not contain '..'."
            )
        root = self.sandbox.root
        found = sorted(
            p
            for p in root.glob(pattern)
            if p.is_file() and self._visible(p) and p.resolve().is_relative_to(root)
        )
        if not found:
            return f"No files match '{pattern}'."
        shown = [p.relative_to(root).as_posix() for p in found[:MAX_GLOB_RESULTS]]
        extra = (
            f"\n... and {len(found) - MAX_GLOB_RESULTS} more (narrow the pattern)"
            if len(found) > MAX_GLOB_RESULTS
            else ""
        )
        return "\n".join(shown) + extra

    def code_grep(self, pattern: str, file_pattern: str = "", ignore_case: bool = False) -> str:
        """Search the CONTENTS of files with a regular expression. Returns matching lines as 'path:line: text'.

        Args:
            pattern: The regular expression to look for, e.g. "def \\w+_test" or "TODO|FIXME".
            file_pattern: Optionally limit the search to files matching a glob such as "**/*.py". Empty searches every text file.
            ignore_case: True to ignore upper/lower case.
        """
        if not pattern:
            raise ToolError("Give a regular expression to search for.")
        if len(pattern) > 300:
            raise ToolError("The pattern is too long (limit 300 characters).")
        try:
            compiled = regex.compile(pattern, regex.IGNORECASE if ignore_case else 0)
        except regex.error as error:
            raise ToolError(f"That is not a valid regular expression: {error}") from error

        root = self.sandbox.root
        candidates = root.glob(file_pattern.strip() or "**/*")
        results: list[str] = []
        for path in sorted(p for p in candidates if p.is_file() and self._visible(p)):
            if not path.resolve().is_relative_to(root) or path.stat().st_size > MAX_GREP_FILE_BYTES:
                continue
            try:
                text = self._read(path)
            except ToolError:
                continue  # binary: nothing to search
            for number, line in enumerate(text.splitlines(), start=1):
                try:
                    hit = compiled.search(line[:MAX_LINE_CHARS], timeout=REGEX_TIMEOUT_SECONDS)
                except TimeoutError:
                    raise ToolError(
                        f"That pattern is too slow (it took over {REGEX_TIMEOUT_SECONDS:.0f}s on one line). Simplify it."
                    ) from None
                if hit:
                    results.append(
                        f"{path.relative_to(root).as_posix()}:{number}: {line.strip()[:MAX_LINE_CHARS]}"
                    )
                    if len(results) >= MAX_GREP_RESULTS:
                        return (
                            "\n".join(results)
                            + f"\n... stopped after {MAX_GREP_RESULTS} matches (narrow the search)"
                        )
        return "\n".join(results) or "No matches."

    def code_outline(self, path: str) -> str:
        """Summarise a Python file's structure (classes, functions, their signatures, the first line of each docstring) with no bodies, so you can understand a file without reading all of it.

        Args:
            path: A .py file in the workspace.
        """
        target = self._existing_file(path)
        if target.suffix.lower() not in (".py", ".pyi"):
            raise ToolError(
                "Outlines are only available for Python files. Use code_read_file for other files."
            )
        try:
            return outline_python(self._read(target))
        except SyntaxError as error:
            raise ToolError(
                f"'{path}' has a syntax error, so it cannot be outlined (line {error.lineno}: {error.msg})."
            ) from error

    def code_read_file(self, path: str, start_line: int = 0, end_line: int = 0) -> str:
        """Read a text file with line numbers. Pass a line range to read just part of a large file.

        Args:
            path: The file, relative to the workspace root.
            start_line: First line to read (1-based). 0 means from the start.
            end_line: Last line to read, inclusive. 0 means as far as the limit of 2000 lines allows.
        """
        target = self._existing_file(path)
        lines = self._read(target).splitlines()
        total = len(lines)
        first = max(start_line, 1)
        last = min(end_line if end_line > 0 else total, total, first + MAX_READ_LINES - 1)
        if first > total:
            raise ToolError(f"'{path}' has only {total} lines.")
        width = len(str(last))
        body = "\n".join(
            f"{n:>{width}}| {lines[n - 1][: MAX_LINE_CHARS * 5]}" for n in range(first, last + 1)
        )
        note = (
            f"\n[showing lines {first}-{last}; the file has {total}: ask for another range to see more]"
            if last < total
            else ""
        )
        return f"{path} (lines {first}-{last} of {total})\n{body}{note}"

    # --- changing files -------------------------------------------------------------------------------------------------

    def code_edit_file(self, path: str, old_text: str, new_text: str) -> str:
        """Edit a file by replacing an exact block of old text with new text. Not line numbers, so it still works if the file changed since you read it. old_text must match EXACTLY ONCE: include enough surrounding lines to make it unique. Python, JSON, TOML, C# and XML-based project files (.csproj, .xaml and so on) are syntax-checked and an edit that breaks them is refused; Visual Basic gets an approximate check that only adds a warning. The user is asked to approve every edit and sees a diff.

        Args:
            path: The file to edit.
            old_text: The exact text to replace, including its indentation and line breaks.
            new_text: What to put in its place (may be empty to delete the block).
        """
        target = self._existing_file(path)
        original, newline = self._load(target)
        old_text, new_text = old_text.replace("\r\n", "\n"), new_text.replace("\r\n", "\n")
        if not old_text:
            raise ToolError("old_text is empty. To create a file use code_create_file.")
        count = original.count(old_text)
        if count == 0:
            raise ToolError(
                "old_text was not found in the file. Read it again with code_read_file and copy the text exactly, including spaces."
            )
        if count > 1:
            raise ToolError(
                f"old_text matches {count} places. Include more surrounding lines so it matches exactly once."
            )
        updated = original.replace(old_text, new_text, 1)
        warning = self._write_check("edit", target, updated, original)
        write_text_atomic(target, updated.replace("\n", newline))
        return f"Edited '{path}': replaced {len(old_text.splitlines()) or 1} line(s) with {len(new_text.splitlines())}.{warning}"

    def code_create_file(self, path: str, content: str) -> str:
        """Create a brand-new file. Fails if the file already exists (use code_edit_file to change one). Python, JSON, TOML, C# and XML-based project file content is syntax-checked first (Visual Basic gets an approximate check that only warns). The user is asked to approve every new file.

        Args:
            path: Where to create it, relative to the workspace root. Missing folders are created.
            content: The complete text of the new file.
        """
        limit = self.settings.modules.filesystem.max_write_chars
        if len(content) > limit:
            raise ToolError(
                f"That is {len(content):,} characters; the limit for one file is {limit:,}."
            )
        target = self.sandbox.resolve(path)
        if target.exists():
            raise ToolError(f"'{path}' already exists. Use code_edit_file to change it.")
        warning = self._write_check("file", target, content, None)
        write_text_atomic(target, content)
        return f"Created '{self.sandbox.display(target)}' ({len(content.splitlines())} lines).{warning}"

    def code_delete_file(self, path: str) -> str:
        """Permanently delete one file. The user is asked to approve every delete.

        Args:
            path: The file to delete.
        """
        target = self._existing_file(path)
        target.unlink()
        return f"Deleted '{path}'."

    def code_delete_folder(self, path: str) -> str:
        """Permanently delete a folder and everything in it. The user is asked to approve every delete.

        Args:
            path: The folder to delete (not the workspace root).
        """
        target = self.sandbox.resolve(path)
        if target == self.sandbox.root:
            raise ToolError("The workspace folder itself cannot be deleted.")
        if not target.is_dir():
            raise ToolError(f"'{path}' is not a folder in the workspace.")
        shutil.rmtree(target)
        return f"Deleted the folder '{path}' and everything in it."

    def code_move_file(self, path: str, new_path: str) -> str:
        """Move or rename a file. Never overwrites. The user is asked to approve every move.

        Args:
            path: The file to move.
            new_path: Its new location and name, relative to the workspace root.
        """
        source = self._existing_file(path)
        destination = self.sandbox.resolve(new_path)
        if destination.exists():
            raise ToolError(f"'{new_path}' already exists; nothing was moved.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.rename(destination)
        return f"Moved '{path}' to '{new_path}'."

    # --- permission: when to ask, and what to show ---------------------------------------------------------------------

    def _edit_needs_approval(self, data: dict) -> bool:
        path = data.get("path")
        return not (isinstance(path, str) and self._files.is_preapproved(path))

    _create_needs_approval = (
        _edit_needs_approval  # the same rule: ask unless the folder is pre-approved
    )

    def _describe_edit(self, data: dict) -> str:
        path, old, new = (
            str(data.get("path", "")),
            str(data.get("old_text", "")),
            str(data.get("new_text", "")),
        )
        try:
            target = self.sandbox.resolve(path)
            original, _ = self._load(target)
        except (ToolError, OSError) as error:
            return f"Edit {path}: {error}"
        old, new = old.replace("\r\n", "\n"), new.replace("\r\n", "\n")
        count = original.count(old) if old else 0
        if count != 1:
            return f"Edit {path}\n(this edit will fail: old_text matches {count} places, it must match exactly once)"
        updated = original.replace(old, new, 1)
        diff = list(
            difflib.unified_diff(
                original.splitlines(),
                updated.splitlines(),
                f"{path} (before)",
                f"{path} (after)",
                lineterm="",
                n=2,
            )
        )
        shown = diff[:MAX_DIFF_LINES]
        more = (
            f"\n[... {len(diff) - MAX_DIFF_LINES} more diff lines]"
            if len(diff) > MAX_DIFF_LINES
            else ""
        )
        return f"Edit {path}\n\n" + "\n".join(shown) + more

    def _describe_create(self, data: dict) -> str:
        path, content = str(data.get("path", "")), str(data.get("content", ""))
        shown = (
            content
            if len(content) <= 2000
            else content[:2000] + f"\n[... {len(content) - 2000:,} more characters]"
        )
        return f"Create {path} ({len(content.splitlines())} lines)\n\n{shown}"

    def _describe_delete(self, data: dict) -> str:
        path = str(data.get("path", ""))
        try:
            target = self.sandbox.resolve(path)
        except ToolError as error:
            return f"Delete {path}: {error}"
        if target.is_dir():
            count = sum(1 for p in target.rglob("*") if p.is_file())
            return f"PERMANENTLY DELETE the folder {path} and the {count} file(s) inside it. This cannot be undone."
        return f"PERMANENTLY DELETE the file {path}. This cannot be undone."

    def _describe_move(self, data: dict) -> str:
        return f"Move {data.get('path', '')}  ->  {data.get('new_path', '')}"
