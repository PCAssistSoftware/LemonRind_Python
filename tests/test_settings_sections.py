"""Each section of the Settings dialog saves what it shows, plus the "view" dialogs in the right-hand panel."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import cast

import pytest
from nicegui import ui
from nicegui.testing import User

from lemonrind import config
from lemonrind.modules import MemoryModule
from lemonrind.modules.base import ChatContext
from lemonrind.webui.context import AppContext
from lemonrind.webui.inspectors import filter_problems, read_tail
from lemonrind.webui.settings_sections import (
    folder_size,
    human_size,
    storage_breakdown,
)
from tests.conftest import elements, open_section


def box(user: User, marker: str):
    """The one widget with this marker (so a test can read or set its value)."""
    (element,) = user.find(marker=marker).elements
    return element


async def save(user: User) -> None:
    user.find(marker="settings-save").click()
    await user.should_not_see(marker="settings-save")  # the dialog is gone: saving worked


async def settle() -> None:
    """Give a click handler that does async work (a notification, a worker thread) time to finish."""
    await asyncio.sleep(0.3)


def saved_settings(web: AppContext) -> dict:
    return json.loads(web.settings_file.read_text(encoding="utf-8"))


# --- Lemonade, Assistant, Interface -----------------------------------------------------------------------------


async def test_the_lemonade_section_saves_address_models_and_key_and_says_a_restart_is_needed(
    user: User, web: AppContext
):
    await open_section(user, "lemonade")
    await user.should_see("Default embedding model")

    box(user, "lemonade-url").value = "http://elsewhere:9999/v1"
    box(user, "lemonade-key").value = "secret-key"
    box(user, "lemonade-chat-model").value = "Fake-Chat"
    box(user, "lemonade-embedding-model").value = "Fake-Embed"
    await save(user)

    lemonade = web.settings.lemonade
    assert lemonade.base_url == "http://elsewhere:9999/v1/"  # the trailing slash is added
    assert (lemonade.api_key, lemonade.chat_model, lemonade.embedding_model) == (
        "secret-key",
        "Fake-Chat",
        "Fake-Embed",
    )
    assert saved_settings(web)["lemonade"]["api_key"] == "secret-key"
    assert user.notify.contains("Restart the app")


async def test_the_model_lists_come_from_lemonade_by_kind(user: User, web: AppContext):
    await user.open("/")
    await open_section(user, "lemonade")

    chat = box(user, "lemonade-chat-model")
    embedding = box(user, "lemonade-embedding-model")
    assert "Fake-Chat" in chat.options.values() and "Fake-Embed" not in chat.options.values()  # type: ignore[attr-defined]
    assert "Fake-Embed" in embedding.options.values()  # type: ignore[attr-defined]


async def test_the_assistant_section_holds_the_system_prompt(user: User, web: AppContext):
    await open_section(user, "assistant")

    box(user, "assistant-prompt").value = "You answer in rhyme."
    await save(user)

    assert web.settings.assistant.system_prompt == "You answer in rhyme."
    user.find(marker="view-prompt").click()
    await user.should_see("You answer in rhyme.")


async def test_the_interface_section_holds_the_theme_and_the_thinking_panel(
    user: User, web: AppContext
):
    await open_section(user, "interface")

    box(user, "settings-theme").value = "dark"
    box(user, "settings-thinking").value = True
    await save(user)

    assert (web.settings.ui.theme, web.settings.ui.thinking_open_by_default) == ("dark", True)
    assert saved_settings(web)["ui"]["theme"] == "dark"


# --- Web search, File system access, Image generation ------------------------------------------------------------


async def test_the_web_search_section_picks_the_engine_and_stores_the_keys(
    user: User, web: AppContext
):
    await open_section(user, "web_search")
    await user.should_see("Search and page-reading service")
    assert box(user, "websearch-engine").value == "SearXNG"
    assert set(box(user, "websearch-engine").options) == {"SearXNG", "Jina", "Tavily", "Firecrawl"}  # type: ignore[attr-defined]

    box(user, "websearch-engine").value = "Firecrawl"
    box(user, "websearch-url").value = "http://searx.local:8888/"
    box(user, "websearch-results").value = 8
    box(user, "websearch-jina-key").value = " jk "
    box(user, "websearch-tavily-key").value = "tk"
    box(user, "websearch-firecrawl-key").value = "fk"
    await save(user)

    config = web.settings.modules.web_search
    assert (config.engine, config.searxng_url, config.max_results) == (
        "Firecrawl",
        "http://searx.local:8888/",
        8,
    )
    assert (config.jina_api_key, config.tavily_api_key, config.firecrawl_api_key) == (
        "jk",
        "tk",
        "fk",
    )
    assert saved_settings(web)["modules"]["web_search"]["engine"] == "Firecrawl"
    reader = next(m for m in web.modules.modules if m.config_key == "web_reader")  # type: ignore[union-attr]
    assert "crawl_website" in [
        t.name for t in reader.get_tools()
    ]  # the key switched the crawl tool on


async def test_the_web_search_section_mentions_the_reader_when_it_is_off(
    user: User, web: AppContext
):
    web.settings.modules.enabled["web_reader"] = False

    await open_section(user, "web_search")

    await user.should_see("Web reader module")


async def test_the_file_system_section_saves_the_workspace_and_the_free_folders(
    user: User, web: AppContext
):
    await open_section(user, "filesystem")

    box(user, "filesystem-root").value = "  C:\\Projects\\demo  "
    box(user, "filesystem-free").value = "notes\n\n  drafts  \n"
    await save(user)

    files = web.settings.modules.filesystem
    assert files.root == "C:\\Projects\\demo"
    assert files.preapproved_folders == ["notes", "drafts"]  # blank lines and padding dropped


async def test_the_image_section_saves_the_model_size_and_the_optional_numbers(
    user: User, web: AppContext
):
    await open_section(user, "images")

    box(user, "images-width").value = 1024
    box(user, "images-steps").value = 20
    box(user, "images-cfg").value = 2.5
    box(user, "images-seed").value = 42
    await save(user)
    images = web.settings.modules.images
    assert (images.width, images.steps, images.cfg_scale, images.seed) == (1024, 20, 2.5, 42)

    await open_section(
        user, "images"
    )  # blank steps / CFG mean "the model's own default"; the seed is -1 (random), whether typed or left blank
    box(user, "images-steps").value = None
    box(user, "images-cfg").value = None
    box(user, "images-seed").value = None
    await save(user)
    assert (images.steps, images.cfg_scale, images.seed) == (None, None, -1)


# --- Memory ------------------------------------------------------------------------------------------------------


async def test_the_memory_section_has_the_compaction_slider_and_the_learning_switch(
    user: User, web: AppContext
):
    await open_section(user, "memory")
    await user.should_see("Conversation compaction")
    assert box(user, "memory-compaction").value == web.settings.assistant.compaction_percent

    box(user, "memory-compaction").value = 50
    box(user, "memory-learn").value = False
    await save(user)

    assert web.settings.assistant.compaction_percent == 50
    assert web.settings.modules.memory.extract_facts is False


async def test_the_memory_section_lists_pinned_and_other_facts_beside_the_settings(
    user: User, web: AppContext
):
    memory = cast(MemoryModule, next(m for m in web.modules.modules if m.config_key == "memory"))  # type: ignore[union-attr]
    memory.repo.add("The user's name is Darren.", pinned=True)
    memory.repo.add("The user likes pizza.", pinned=False)

    await open_section(user, "memory")

    await user.should_see("Pinned (always included)")
    await user.should_see("Other (used when relevant)")
    await user.should_see("Conversation compaction")


# --- Auto-backup --------------------------------------------------------------------------------------------------


async def test_the_backup_section_saves_its_settings_and_backs_up_on_request(
    user: User, web: AppContext, tmp_path: Path
):
    zips = (
        tmp_path.parent / f"{tmp_path.name}_zips"
    )  # outside the data folder (which is tmp_path in these tests)
    web.settings.modules.backup.backup_folder = str(zips)
    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace" / "a.txt").write_text("hello", encoding="utf-8")
    await open_section(user, "backup")
    await user.should_see("Last backup: never")

    user.find(marker="backup-now").click()
    for _ in range(50):  # zipping runs on a worker thread
        if list(zips.glob("*.zip")):
            break
        await asyncio.sleep(0.1)
    await settle()
    (made,) = zips.glob("lemonrind_backup_*.zip")
    assert user.notify.contains("Backed up")
    await user.should_see("Last backup:")
    assert "never" not in box(user, "backup-status").text  # type: ignore[attr-defined]
    assert made.stat().st_size > 0

    box(user, "backup-interval").value = 3
    box(user, "backup-folder").value = str(zips)
    await save(user)
    assert web.settings.modules.backup.interval_days == 3


async def test_a_value_the_settings_reject_keeps_the_dialog_open_and_undoes_the_save(
    user: User, web: AppContext
):
    await open_section(user, "backup")
    box(user, "assistant-prompt").value = "Changed before the bad value."
    box(user, "backup-interval").value = 1000  # the most it accepts is 365

    user.find(marker="settings-save").click()
    await settle()
    await user.should_see(marker="settings-save")  # still open

    assert user.notify.contains("interval_days")
    assert web.settings.modules.backup.interval_days == 7
    assert (
        web.settings.assistant.system_prompt != "Changed before the bad value."
    )  # everything was put back
    assert not web.settings_file.exists()


# --- Storage ------------------------------------------------------------------------------------------------------


async def test_the_storage_section_shows_the_folder_and_what_uses_the_space(
    user: User, web: AppContext, tmp_path: Path
):
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "x.png").write_bytes(b"x" * 2048)
    (tmp_path / "knowledge").mkdir(exist_ok=True)  # the app already made it
    (tmp_path / "knowledge" / "garden-1a2b3c4d.kb").write_bytes(b"k" * 5000)
    await open_section(user, "storage")

    await user.should_see("What is using the space")
    await user.should_see("Generated images")
    await user.should_see("Knowledge bases")  # the knowledge base files have a row of their own
    await user.should_see("2.0 KB")
    assert box(user, "storage-current").text == str(tmp_path)  # type: ignore[attr-defined]


async def test_choosing_another_data_folder_writes_the_pointer_and_asks_for_a_restart(
    user: User, web: AppContext, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    default = tmp_path / "default"
    monkeypatch.setattr(config, "default_data_dir", lambda: default)
    new_home = tmp_path / "elsewhere"
    await open_section(user, "storage")

    box(user, "storage-folder").value = str(new_home)
    await save(user)

    assert new_home.is_dir()
    assert config.read_data_location(default) == new_home.resolve()
    assert user.notify.contains("Restart the app to use the new data folder")

    await open_section(user, "storage")  # clearing the box goes back to the default
    assert box(user, "storage-folder").value == str(new_home.resolve())
    box(user, "storage-folder").value = ""
    await save(user)
    assert config.read_data_location(default) is None


async def test_an_unusable_data_folder_is_reported_and_nothing_is_saved(
    user: User, web: AppContext, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    default = tmp_path / "default"
    monkeypatch.setattr(config, "default_data_dir", lambda: default)
    blocker = tmp_path / "a-file"
    blocker.write_text("not a folder", encoding="utf-8")
    await open_section(user, "storage")

    box(user, "assistant-prompt").value = "Should not be kept."
    box(user, "storage-folder").value = str(
        blocker / "inside"
    )  # a folder cannot be made inside a file
    user.find(marker="settings-save").click()
    await settle()
    await user.should_see(marker="settings-save")

    assert user.notify.contains("cannot be used")
    assert config.read_data_location(default) is None
    assert web.settings.assistant.system_prompt != "Should not be kept."


def test_sizes_are_shown_in_friendly_units(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "sys.platform", "win32"
    )  # Windows counts 1 KB as 1,024 bytes, like its file manager
    assert (
        human_size(0) == "0 bytes"
        and human_size(1536) == "1.5 KB"
        and human_size(5 * 1024**2) == "5.0 MB"
    )
    monkeypatch.setattr("sys.platform", "linux")  # Linux and macOS count 1 kB as 1,000 bytes
    assert human_size(222_500) == "222.5 KB" and human_size(1536) == "1.5 KB"
    assert human_size(5_000_000) == "5.0 MB" and human_size(999) == "999 bytes"
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a").write_bytes(b"12345")
    (tmp_path / "b").write_bytes(b"123")
    assert (
        folder_size(tmp_path) == 8
        and folder_size(tmp_path / "b") == 3
        and folder_size(tmp_path / "none") == 0
    )


# --- Modules ------------------------------------------------------------------------------------------------------


async def test_the_modules_list_includes_auto_backup_and_it_starts_off(user: User, web: AppContext):
    await open_section(user, "modules")

    await user.should_see("Auto-backup")
    backup = [e for e in user.find(kind=ui.switch).elements if e.text == "Auto-backup"]
    assert len(backup) == 1 and backup[0].value is False


# --- the "view" dialogs -------------------------------------------------------------------------------------------


def test_the_end_of_a_log_file_is_read_without_loading_all_of_it(tmp_path: Path):
    path = tmp_path / "x.log"
    assert read_tail(path) == ""  # missing file
    path.write_text("\n".join(f"line {n}" for n in range(1000)), encoding="utf-8")

    tail = read_tail(path, lines=3)

    assert tail.splitlines() == ["line 997", "line 998", "line 999"]
    path.write_text(
        "A" * 300_000 + "\nlast line\n", encoding="utf-8"
    )  # bigger than the chunk that is read
    assert read_tail(path, lines=5).splitlines() == [
        "last line"
    ]  # the cut first line is dropped, not shown half


def test_only_warnings_and_errors_and_their_tracebacks_are_kept_by_the_filter():
    log = "\n".join(
        [
            "2026-10-03 10:00:00,000 INFO lemonrind.a: started",
            "2026-10-03 10:00:01,000 WARNING lemonrind.b: careful",
            "2026-10-03 10:00:02,000 INFO lemonrind.a: fine",
            "2026-10-03 10:00:03,000 ERROR lemonrind.c: broke",
            "Traceback (most recent call last):",
            "ValueError: boom",
            "2026-10-03 10:00:04,000 INFO lemonrind.a: recovered",
        ]
    )

    kept = filter_problems(log).splitlines()

    assert kept == [
        "2026-10-03 10:00:01,000 WARNING lemonrind.b: careful",
        "2026-10-03 10:00:03,000 ERROR lemonrind.c: broke",
        "Traceback (most recent call last):",
        "ValueError: boom",
    ]


async def test_the_four_view_buttons_are_in_the_right_hand_panel(user: User, web: AppContext):
    await user.open("/")

    for marker in ("view-prompt", "view-turn-context", "view-tools", "view-logs"):
        await user.should_see(marker=marker)


async def test_view_tools_sent_lists_every_tool_the_model_is_offered(user: User, web: AppContext):
    await user.open("/")

    user.find(marker="view-tools").click()

    await user.should_see("Tools sent")
    await user.should_see("calculate")  # a utilities tool, one panel per tool


async def test_view_turn_context_explains_when_nothing_was_added(user: User, web: AppContext):
    await user.open("/")

    user.find(marker="view-turn-context").click()

    await user.should_see("Turn context (preview)")
    await user.should_see("Current date/time:")  # the model is always told the date and time
    await user.should_see("Type a message first")


async def test_view_logs_shows_the_end_of_the_log_and_can_hide_the_chatter(
    user: User, web: AppContext, tmp_path: Path
):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "lemonrind.log").write_text(
        "2026-10-03 10:00:00,000 INFO lemonrind.a: routine start-up line\n"
        "2026-10-03 10:00:01,000 WARNING lemonrind.b: the memory lookup failed\n",
        encoding="utf-8",
    )
    await user.open("/")

    user.find(marker="view-logs").click()
    await user.should_see("routine start-up line")
    await user.should_see("the memory lookup failed")

    user.find(kind=ui.switch, content="Warnings and errors only").click()
    await user.should_not_see("routine start-up line")
    await user.should_see("the memory lookup failed")


class AddsContext:
    """A module registry stand-in whose only job is to add text to a message."""

    async def stable_context(self) -> str:
        return ""

    async def turn_context(self, user_text: str, *, chat: ChatContext) -> str:
        return f"Background about {user_text}"


async def test_the_apps_own_log_can_be_cleared_after_asking(
    user: User, web: AppContext, tmp_path: Path
):
    logs = tmp_path / "logs"
    logs.mkdir()
    log_file = logs / "lemonrind.log"
    log_file.write_text(
        "2026-10-03 10:00:00,000 WARNING lemonrind.b: the memory lookup failed\n", encoding="utf-8"
    )
    await user.open("/")
    user.find(marker="view-logs").click()
    await user.should_see("the memory lookup failed")

    user.find(marker="log-clear-file").click()
    await user.should_see("Delete everything in the app's log file")
    user.find(marker="dialog-cancel").click()
    assert "memory lookup" in log_file.read_text(encoding="utf-8")  # cancelling keeps it

    user.find(marker="log-clear-file").click()
    user.find(marker="dialog-ok").click()
    await user.should_see("(nothing logged yet)")
    assert log_file.read_text(encoding="utf-8") == ""


def test_the_storage_rows_add_up_to_the_total_and_explain_every_file(tmp_path: Path):
    (tmp_path / "lemonrind.db").write_bytes(b"d" * 4000)
    (tmp_path / "lemonrind.db-wal").write_bytes(
        b"w" * 220_000
    )  # the database's side files count as the database
    (tmp_path / "lemonrind.db-shm").write_bytes(b"s" * 32_000)
    (tmp_path / "settings.json").write_bytes(b"x" * 1900)
    (tmp_path / "session_secret").write_bytes(b"k" * 64)
    for folder, size in (("knowledge", 5000), ("images", 700), ("attachments", 300), ("logs", 73)):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "file").write_bytes(b"f" * size)
    (tmp_path / "workspace").mkdir()  # empty folders have no size

    rows, total = storage_breakdown(tmp_path)

    sizes = dict(rows)
    assert sizes["Database"] == 256_000
    assert sizes["Knowledge bases"] == 5000 and sizes["Generated images"] == 700
    assert (
        sizes["Attached pictures"] == 300 and sizes["Workspace files"] == 0 and sizes["Logs"] == 73
    )
    assert sizes["Other (settings and the rest)"] == 1964  # settings.json and the session secret
    assert sum(sizes.values()) == total == folder_size(tmp_path)


def test_with_nothing_unexplained_there_is_no_other_row(tmp_path: Path):
    (tmp_path / "lemonrind.db").write_bytes(b"d" * 10)

    rows, total = storage_breakdown(tmp_path)

    assert [label for label, _ in rows][0] == "Database"
    assert not any(label.startswith("Other") for label, _ in rows)
    assert total == 10


async def test_the_longest_tool_result_piece_is_set_in_modules_and_applies_at_once(
    user: User, web: AppContext
):
    await open_section(user, "modules")
    (box,) = elements(user, "modules-output-chars")
    assert box.value == web.settings.modules.max_tool_output_chars

    box.value = 80_000
    user.find(marker="settings-save").click()
    await settle()

    assert web.settings.modules.max_tool_output_chars == 80_000
    assert (
        web.modules is not None and web.modules.max_output_chars == 80_000
    )  # the next result already uses it
    assert saved_settings(web)["modules"]["max_tool_output_chars"] == 80_000


def test_the_default_piece_size_is_forty_thousand_characters():
    from lemonrind.modules.registry import DEFAULT_MAX_OUTPUT_CHARS

    assert config.Settings().modules.max_tool_output_chars == 40_000 == DEFAULT_MAX_OUTPUT_CHARS
