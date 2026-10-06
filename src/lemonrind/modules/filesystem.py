"""The File system module: let the model read and write files in ONE folder, and nowhere else.

Giving a model access to your files is useful and risky. The safeguard is a *sandbox*: every path the
model supplies is resolved against a single root folder and rejected if it ends up outside. The model
(or a web page it read) might ask for ``../../Users/you/.ssh/id_rsa``; the sandbox turns that into an error.

**Writing** is the riskier half, so ``write_file`` asks your permission first (the approval mechanism from
step 8), showing the file name, whether it creates or overwrites, and the text it will write. Folders you list
in the ``preapproved_folders`` setting are exempt. A write is also bounded: only inside the sandbox, only
text, at most ``max_write_chars`` characters, and the file is replaced *atomically* (see ``write_file``).

Python ideas used here:

* ``pathlib.Path.resolve()`` - the real absolute path, with ``..`` and symbolic links followed. Checking
  the *resolved* path is what makes the sandbox hold.
* ``Path.is_relative_to`` - "is this path inside that folder?".
* ``on_startup`` - a module hook: the root folder is created when the module is switched on.
* ``tempfile`` + ``Path.replace`` - the standard recipe for **atomic writes** (below).
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

from lemonrind.config import Settings
from lemonrind.modules.base import Module
from lemonrind.modules.tool import Tool, ToolError, tool_from_function

MAX_LISTED = 200


def write_text_atomic(target: Path, content: str) -> None:
    """Write ``content`` to ``target`` so that a failure half way never leaves a half-written file.

    Everything goes to a temporary file next to the target, which is then swapped into place in one step. If
    anything fails (disk full, crash, power cut) the old file is untouched. (``Path.replace`` is atomic when both
    files are on the same disk, which is why the temporary file is created in the target's own folder.) The text
    is written exactly as given: no newline translation.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=target.parent, prefix=".write-", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
        Path(temporary).replace(target)
    except BaseException:
        with contextlib.suppress(OSError):
            Path(temporary).unlink()
        raise


class PathSandbox:
    """Turns a model-supplied path into a real path, and refuses anything outside the allowed folders.

    The **workspace** (``root``) is the default: a relative path means "inside the workspace". Any number of **extra
    roots** can be allowed as well; the model then uses a full path to reach into one. Whatever it asks for is resolved
    first (so ``..`` and symbolic links are followed) and only then checked against the allowed folders.
    """

    def __init__(self, root: Path, extra_roots: Sequence[Path] = ()) -> None:
        self.root = root.resolve()
        self.extra_roots = tuple(extra.resolve() for extra in extra_roots)

    @property
    def roots(self) -> tuple[Path, ...]:
        return (self.root, *self.extra_roots)

    def resolve(self, requested: str) -> Path:
        """The real path for ``requested``, or ``ToolError`` if it is not inside an allowed folder."""
        candidate = Path(requested.strip().strip('"'))
        if candidate.is_absolute():
            resolved = candidate.resolve()
            if any(resolved == root or resolved.is_relative_to(root) for root in self.roots):
                return resolved
            raise ToolError(
                "That full path is outside the allowed folders. Allowed: "
                + ", ".join(str(root) for root in self.roots)
                + "."
            )
        if (
            candidate.drive or candidate.root
        ):  # "C:file" or "\\file": neither relative nor a full path
            raise ToolError(
                "Use a path relative to the workspace folder, or a full path inside an allowed folder."
            )
        resolved = (self.root / candidate).resolve()
        if resolved != self.root and not resolved.is_relative_to(self.root):
            raise ToolError("That path is outside the workspace folder, which is not allowed.")
        return resolved

    def display(self, path: Path) -> str:
        """A path as the model should see it: relative to the workspace when inside it, else the full path."""
        if path == self.root or path.is_relative_to(self.root):
            relative = path.relative_to(self.root).as_posix()
            return relative if relative != "." else "(workspace root)"
        return path.as_posix()


class FileSystemModule(Module):
    name = "File system"
    config_key = "filesystem"
    description = (
        "Read and write files in one workspace folder (nothing outside it). Writes ask first."
    )

    def __init__(self, settings: Settings, data_dir: Path) -> None:
        super().__init__(settings)
        self._data_dir = data_dir

    @property
    def root(self) -> Path:
        configured = self.settings.modules.filesystem.root.strip()
        return Path(configured).expanduser() if configured else self._data_dir / "workspace"

    @property
    def extra_roots(self) -> list[Path]:
        return [
            Path(entry).expanduser()
            for entry in self.settings.modules.filesystem.allowed_roots
            if entry.strip()
        ]

    @property
    def sandbox(self) -> PathSandbox:
        """The rules for what the model may reach: the workspace, plus any extra allowed folders."""
        return PathSandbox(self.root, self.extra_roots)

    async def on_startup(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def get_tools(self) -> list[Tool]:
        write = dataclasses.replace(
            tool_from_function(self.write_file),
            requires_approval=True,
            approval_check=self._write_needs_approval,
            preview=self._describe_write,
        )
        tools = [tool_from_function(self.list_files), tool_from_function(self.read_file), write]
        return [self._with_folders(tool) for tool in tools]

    def _with_folders(self, tool: Tool) -> Tool:
        """Add the real folder names to a tool's description: a few tokens, instead of a separate "where am I" call."""
        roots = self.sandbox.roots
        if len(roots) == 1:
            note = f"The workspace is {roots[0]}."
        else:
            note = (
                f"The workspace is {roots[0]} (relative paths mean this). Other allowed folders, reached with a full "
                "path: " + ", ".join(str(root) for root in roots[1:]) + "."
            )
        return dataclasses.replace(tool, description=f"{tool.description}\n\n{note}")

    # --- writing ---------------------------------------------------------------------------------------------

    def write_file(self, path: str, content: str) -> str:
        """Create a text file in the workspace, or replace it if it already exists. The user is asked to approve every write.

        Args:
            path: Where to write, relative to the workspace root, e.g. "notes/todo.txt". Missing folders are created.
            content: The complete new text of the file.
        """
        limit = self.settings.modules.filesystem.max_write_chars
        if len(content) > limit:
            raise ToolError(
                f"That is {len(content):,} characters; the limit for one file is {limit:,}."
            )
        sandbox = self.sandbox
        target = sandbox.resolve(path)
        if target == sandbox.root or target.is_dir():
            raise ToolError(f"'{path}' is a folder. Give a file name.")
        existed = target.exists()
        write_text_atomic(target, content)
        verb = "Replaced" if existed else "Created"
        return f"{verb} '{sandbox.display(target)}' ({len(content):,} characters)."

    def _write_needs_approval(self, data: dict) -> bool:
        """Ask before a write, unless it lands inside a folder you pre-approved in the settings."""
        path = data.get("path")
        if not isinstance(path, str):
            return True
        try:
            self.sandbox.resolve(path)
        except ToolError:
            return False  # the write itself will be refused, so there is nothing to approve
        return not self.is_preapproved(path)

    def is_preapproved(self, path: str) -> bool:
        """Is ``path`` inside a folder listed in ``preapproved_folders``? (Other modules that write files ask the
        same question, so there is one rule.)"""
        sandbox = self.sandbox
        try:
            target = sandbox.resolve(path)
        except ToolError:
            return False
        for folder in self.settings.modules.filesystem.preapproved_folders:
            try:
                allowed = sandbox.resolve(folder)
            except ToolError:
                continue  # an entry pointing outside the workspace is ignored, never trusted
            if target == allowed or target.is_relative_to(allowed):
                return True
        return False

    def _describe_write(self, data: dict) -> str:
        """The text of the permission question: what will be written, where, and whether it replaces a file."""
        path, content = str(data.get("path", "")), str(data.get("content", ""))
        try:
            target = self.sandbox.resolve(path)
        except ToolError as error:
            return f"Write to '{path}': {error}"
        if target.is_file():
            effect = f"REPLACES the existing file ({target.stat().st_size:,} bytes)"
        else:
            effect = "creates a new file"
        shown = (
            content
            if len(content) <= 2000
            else content[:2000] + f"\n[... {len(content) - 2000:,} more characters]"
        )
        return f"Write {len(content):,} characters to {path}\n({effect})\n\n{shown}"

    def list_files(self, path: str = "") -> str:
        """List the files and folders in a folder of the workspace.

        Args:
            path: Folder to list, relative to the workspace root. Leave empty for the root itself.
        """
        sandbox = self.sandbox
        folder = sandbox.resolve(path)
        if not folder.is_dir():
            raise ToolError(f"'{path}' is not a folder in the workspace.")
        entries = sorted(folder.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        lines = []
        for entry in entries[:MAX_LISTED]:
            name = sandbox.display(entry)
            lines.append(
                f"{name}/" if entry.is_dir() else f"{name} ({entry.stat().st_size:,} bytes)"
            )
        if len(entries) > MAX_LISTED:
            lines.append(f"... and {len(entries) - MAX_LISTED} more")
        return "\n".join(lines) or "(empty folder)"

    def read_file(self, path: str) -> str:
        """Read a text file from the workspace.

        Args:
            path: File to read, relative to the workspace root, e.g. "notes/todo.txt".
        """
        sandbox = self.sandbox
        file = sandbox.resolve(path)
        if not file.is_file():
            raise ToolError(
                f"'{path}' is not a file in the workspace. Use list_files to see what exists."
            )
        limit = self.settings.modules.filesystem.max_read_chars
        with file.open("rb") as handle:
            data = handle.read(
                limit * 4 + 1
            )  # up to 4 bytes per character, plus one to detect "more"
        if b"\x00" in data[:8000]:
            raise ToolError(f"'{path}' looks like a binary file, not text.")
        text = data.decode("utf-8", errors="replace")
        if len(text) > limit:
            return text[:limit] + f"\n[... file truncated at {limit:,} characters]"
        return text
