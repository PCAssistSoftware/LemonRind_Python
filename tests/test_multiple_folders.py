"""More than one folder the assistant may use: the workspace by default, extra ones by full path, nothing else."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from nicegui.testing import User

from lemonrind.config import Settings
from lemonrind.modules import CoderModule, FileSystemModule, ToolError
from lemonrind.modules.filesystem import PathSandbox
from lemonrind.webui.context import AppContext
from tests.conftest import open_section
from tests.test_settings_sections import box, save, settle


@pytest.fixture
def setup(tmp_path: Path):
    """A workspace, one extra allowed folder, and a folder that is NOT allowed (with a file in each)."""
    workspace = tmp_path / "data" / "workspace"
    extra = tmp_path / "projects" / "alpha"
    forbidden = tmp_path / "secrets"
    for folder, name in (
        (workspace, "inside.txt"),
        (extra, "alpha.txt"),
        (forbidden, "passwords.txt"),
    ):
        folder.mkdir(parents=True)
        (folder / name).write_text(f"contents of {name}", encoding="utf-8")
    settings = Settings()
    settings.modules.filesystem.allowed_roots = [str(extra)]
    files = FileSystemModule(settings, tmp_path / "data")
    return files, workspace, extra, forbidden


# --- the sandbox ---------------------------------------------------------------------------------------------------


def test_relative_paths_mean_the_workspace_and_full_paths_work_inside_any_allowed_folder(setup):
    _, workspace, extra, _ = setup
    sandbox = PathSandbox(workspace, [extra])

    assert sandbox.resolve("inside.txt") == (workspace / "inside.txt").resolve()
    assert sandbox.resolve(str(extra / "alpha.txt")) == (extra / "alpha.txt").resolve()
    assert (
        sandbox.resolve(str(workspace / "inside.txt")) == (workspace / "inside.txt").resolve()
    )  # a full path inside it is fine too
    assert sandbox.resolve(str(extra)) == extra.resolve()  # the folder itself


def test_everything_outside_the_allowed_folders_is_refused_however_it_is_spelled(setup):
    _, workspace, extra, forbidden = setup
    sandbox = PathSandbox(workspace, [extra])

    attempts = [
        str(forbidden / "passwords.txt"),  # a full path elsewhere
        "../../secrets/passwords.txt",  # climbing out of the workspace
        str(extra / ".." / ".." / "secrets" / "passwords.txt"),  # climbing out of an extra folder
        str(workspace / ".." / ".." / "secrets"),
        str(extra.parent / "alpha-evil" / "x.txt"),  # shares a name prefix with an allowed folder
        "C:file.txt" if os.name == "nt" else "/etc/passwd",
    ]
    for attempt in attempts:
        with pytest.raises(ToolError):
            sandbox.resolve(attempt)


def test_a_relative_path_cannot_escape_the_workspace_into_an_extra_folder(setup):
    _, workspace, extra, _ = setup
    relative_to_extra = os.path.relpath(
        extra / "alpha.txt", workspace
    )  # e.g. ..\\..\\projects\\alpha\\alpha.txt

    with pytest.raises(ToolError, match="outside the workspace"):
        PathSandbox(workspace, [extra]).resolve(
            relative_to_extra
        )  # only a FULL path may reach an extra folder


@pytest.mark.skipif(
    os.name == "nt", reason="creating symbolic links needs special rights on Windows"
)
def test_a_symbolic_link_out_of_an_allowed_folder_is_followed_and_refused(setup):
    _, workspace, extra, forbidden = setup
    (workspace / "shortcut").symlink_to(forbidden, target_is_directory=True)

    with pytest.raises(ToolError):
        PathSandbox(workspace, [extra]).resolve("shortcut/passwords.txt")


def test_paths_are_shown_relative_to_the_workspace_or_in_full_for_an_extra_folder(setup):
    _, workspace, extra, _ = setup
    sandbox = PathSandbox(workspace, [extra])

    assert sandbox.display((workspace / "notes" / "a.txt").resolve()) == "notes/a.txt"
    assert sandbox.display(workspace.resolve()) == "(workspace root)"
    assert (
        sandbox.display((extra / "alpha.txt").resolve())
        == (extra / "alpha.txt").resolve().as_posix()
    )


# --- the tools -----------------------------------------------------------------------------------------------------


def test_the_file_tools_read_and_list_inside_an_extra_folder_and_not_elsewhere(setup):
    files, _, extra, forbidden = setup

    assert files.read_file(str(extra / "alpha.txt")) == "contents of alpha.txt"
    assert "alpha.txt" in files.list_files(str(extra))
    assert (
        files.read_file("inside.txt") == "contents of inside.txt"
    )  # the workspace still works as before
    with pytest.raises(ToolError, match="outside the allowed folders"):
        files.read_file(str(forbidden / "passwords.txt"))
    with pytest.raises(ToolError):
        files.list_files(str(forbidden))


def test_writing_to_an_extra_folder_works_and_still_asks_permission_unless_pre_approved(setup):
    files, _, extra, forbidden = setup
    write = {t.name: t for t in files.get_tools()}["write_file"]
    target = str(extra / "new.txt")

    assert (
        files.write_file(target, "hello")
        == f"Created '{Path(target).resolve().as_posix()}' (5 characters)."
    )
    assert (extra / "new.txt").read_text(encoding="utf-8") == "hello"
    assert write.needs_approval(
        json.dumps({"path": target, "content": "x"})
    )  # asks, like any write
    assert not write.needs_approval(
        json.dumps({"path": str(forbidden / "x.txt"), "content": "x"})
    )  # refused, so no question

    files.settings.modules.filesystem.preapproved_folders = [
        str(extra)
    ]  # a full path is accepted here too
    assert not write.needs_approval(json.dumps({"path": target, "content": "x"}))
    with pytest.raises(ToolError):
        files.write_file(str(forbidden / "x.txt"), "no")
    assert not (forbidden / "x.txt").exists()


def test_the_tool_descriptions_tell_the_model_which_folders_it_may_use(setup):
    files, workspace, extra, _ = setup

    descriptions = {t.name: t.description for t in files.get_tools()}

    for text in descriptions.values():
        assert str(workspace.resolve()) in text and str(extra.resolve()) in text
        assert "relative paths mean this" in text
    alone = FileSystemModule(Settings(), workspace.parent)  # no extra folders: a simpler note
    assert "The workspace is" in alone.get_tools()[0].description
    assert "Other allowed folders" not in alone.get_tools()[0].description


def test_the_coder_can_read_and_edit_a_file_in_an_extra_folder_by_full_path(setup):
    files, _, extra, forbidden = setup
    coder = CoderModule(files.settings, files)
    script = extra / "tool.py"
    script.write_text("x = 1\nprint(x)\n", encoding="utf-8")

    assert "x = 1" in coder.code_read_file(str(script))
    message = coder.code_edit_file(str(script), "x = 1", "x = 2")
    assert script.read_text(encoding="utf-8") == "x = 2\nprint(x)\n" and "tool.py" in message
    with pytest.raises(ToolError):
        coder.code_read_file(str(forbidden / "passwords.txt"))


# --- the settings screen -------------------------------------------------------------------------------------------


async def test_the_file_system_section_saves_the_extra_folders_and_rejects_a_relative_one(
    user: User, web: AppContext, tmp_path: Path
):
    await open_section(user, "filesystem")
    good = [str(tmp_path / "one"), str(tmp_path / "two")]

    box(user, "filesystem-extra").value = f"  {good[0]}\n\n{good[1]}  \n"  # type: ignore[attr-defined]
    await save(user)
    assert web.settings.modules.filesystem.allowed_roots == good

    await open_section(user, "filesystem")
    box(user, "filesystem-extra").value = "relative/folder"  # type: ignore[attr-defined]
    user.find(marker="settings-save").click()
    await settle()
    await user.should_see(marker="settings-save")  # still open
    assert user.notify.contains("is not a full path")
    assert web.settings.modules.filesystem.allowed_roots == good  # the earlier, good value was kept
