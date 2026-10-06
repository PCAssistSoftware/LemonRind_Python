"""The web UI's Settings dialog and the chat menu (rename, tag, move, export, delete), driven like a user would."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import cast

import pytest
from nicegui import Client, ui
from nicegui.testing import User

from lemonrind.modules import SchedulerModule
from lemonrind.webui.context import AppContext
from lemonrind.webui.dialogs import discard
from tests.conftest import elements, must, open_section
from tests.test_settings_sections import settle
from tests.test_webui import send


def one_chat(web: AppContext, title: str = "Otters") -> str:
    chat = web.repo.create_session(title=title)
    web.repo.add_message(chat.id, "user", "Tell me about otters")
    web.repo.add_message(chat.id, "assistant", "Otters are playful.")
    return chat.id


# --- Settings ----------------------------------------------------------------------------------------------------------------


async def test_saving_settings_updates_the_prompt_the_file_and_the_theme(
    user: User, web: AppContext
):
    await user.open("/")
    user.find(marker="settings").click()
    await user.should_see("Persona")

    user.find(marker="persona-identity").type("You are Rind, a careful assistant.")
    user.find(marker="persona-about").type("They are learning Python.")
    user.find(marker="settings-save").click()
    await user.should_not_see("Persona")

    assert web.settings.persona.identity == "You are Rind, a careful assistant."
    assert web.settings.persona.about_user == "They are learning Python."
    saved = json.loads(web.settings_file.read_text(encoding="utf-8"))
    assert saved["persona"]["identity"] == "You are Rind, a careful assistant."

    # the running chat now uses the new persona
    user.find(marker="view-prompt").click()
    await user.should_see("You are Rind, a careful assistant.")


async def test_cancelling_settings_changes_nothing(user: User, web: AppContext):
    await user.open("/")
    user.find(marker="settings").click()
    await user.should_see("Persona")
    user.find(marker="persona-identity").type("You are someone else.")
    user.find(marker="settings-cancel").click()
    await user.should_not_see("Persona")

    assert web.settings.persona.identity == ""
    assert not web.settings_file.exists()


async def test_modules_are_listed_with_their_tools_and_can_be_switched_off(
    user: User, web: AppContext
):
    assert (
        web.modules is not None and web.modules.schemas()
    )  # tools are offered to the model to begin with
    await user.open("/")
    user.find(marker="settings").click()

    await user.should_see("Modules")
    await user.should_see("web_search")  # each module's tools are named under its switch
    switches = [e for e in user.find(kind=ui.switch).elements if e.text == "Utilities"]
    assert len(switches) == 1
    switches[0].value = False
    user.find(marker="settings-save").click()
    await user.should_not_see(marker="settings-save")

    assert web.settings.modules.enabled["utilities"] is False
    names = {schema["function"]["name"] for schema in web.modules.schemas()}
    assert "calculate" not in names  # the change applied straight away


async def test_every_section_has_a_navigation_entry_and_the_buttons_follow_the_section(
    user: User, web: AppContext
):
    await user.open("/")
    user.find(marker="settings").click()
    for key in (
        "lemonade", "assistant", "persona", "interface", "storage", "web_search", "filesystem",
        "images", "memory", "backup", "modules", "knowledge", "mcp", "scheduler",
    ):  # fmt: skip
        await user.should_see(marker=f"nav-{key}")
    await user.should_see(marker="settings-save")  # the first section is a form: it has a Save

    user.find(
        marker="nav-scheduler"
    ).click()  # a live section acts at once, so there is nothing to save
    await user.should_not_see(marker="settings-save")
    assert {b.text for b in elements(user, "settings-cancel")} == {"Close"}

    user.find(
        marker="nav-memory"
    ).click()  # Memory has settings of its own (compaction), so it saves
    await user.should_see(marker="settings-save")
    assert {b.text for b in elements(user, "settings-cancel")} == {"Cancel"}


def knowledge_module(web: AppContext):
    return next(m for m in web.modules.modules if m.config_key == "knowledge")  # type: ignore[union-attr]


async def test_the_footer_has_a_knowledge_base_drop_down_that_attaches_one_to_the_chat(
    user: User, web: AppContext
):
    module = knowledge_module(web)
    handbook = module.repo.create_kb("Handbook")  # type: ignore[attr-defined]
    module.repo.create_kb("Recipes")  # type: ignore[attr-defined]
    await user.open("/")
    await user.should_see("Lemonade healthy")
    select = next(iter(user.find(marker="kb-select").elements))

    assert list(select.options.values()) == ["No knowledge base", "Handbook", "Recipes"]  # type: ignore[attr-defined]
    assert select.value == ""  # type: ignore[attr-defined]  # nothing attached yet

    select.value = handbook.id  # type: ignore[attr-defined]
    await settle()
    await send(user, "what does the handbook say?")
    await user.should_see("Paris.")

    (session,) = web.repo.list_sessions()
    assert (
        session.options.get("knowledge_base") == handbook.id
    )  # chosen before the chat existed, kept with it


async def test_the_drop_down_follows_the_chat_that_is_open_and_none_removes_the_knowledge_base(
    user: User, web: AppContext
):
    module = knowledge_module(web)
    handbook = module.repo.create_kb("Handbook")  # type: ignore[attr-defined]
    with_kb = web.repo.create_session(title="With handbook")
    web.repo.add_message(with_kb.id, "user", "hi")
    web.repo.add_message(with_kb.id, "assistant", "hello")
    web.repo.set_option(with_kb.id, "knowledge_base", handbook.id)
    await user.open("/")
    await user.should_see("hello")
    await settle()
    select = next(iter(user.find(marker="kb-select").elements))
    assert select.value == handbook.id  # type: ignore[attr-defined]

    select.value = ""  # type: ignore[attr-defined]
    await settle()
    assert web.repo.get_session(with_kb.id).options.get("knowledge_base") is None  # type: ignore[union-attr]

    user.find(marker="new-chat").click()
    await settle()
    assert select.value == ""  # type: ignore[attr-defined]  # a new chat starts with none


async def test_the_knowledge_drop_down_is_hidden_when_that_module_is_off_and_sees_new_knowledge_bases(
    user: User, web: AppContext
):
    web.settings.modules.enabled["knowledge"] = False
    assert web.modules is not None
    await web.modules.reconcile()
    await user.open("/")
    await user.should_see("Lemonade healthy")
    await user.should_not_see(marker="kb-select")


async def test_a_module_screen_shows_a_note_when_its_module_is_switched_off(
    user: User, web: AppContext
):
    web.settings.modules.enabled["knowledge"] = False
    assert web.modules is not None
    await web.modules.reconcile()
    await user.open("/")

    await open_section(user, "knowledge")

    await user.should_see("module is switched off. Switch it on under Modules")


async def test_a_settings_section_of_a_switched_off_module_still_works_but_says_so(
    user: User, web: AppContext
):
    web.settings.modules.enabled["web_search"] = False
    assert web.modules is not None
    await web.modules.reconcile()
    await user.open("/")

    await open_section(user, "web_search")

    await user.should_see("This module is currently disabled - see Modules to enable it")
    await user.should_see(marker="websearch-url")  # and the form is there


async def test_open_last_chat_in_the_jobs_section_closes_settings_and_shows_that_chat(
    user: User, web: AppContext
):
    chat_id = one_chat(web, "Digest")
    web.repo.add_tag(chat_id, "scheduled")
    modules = must(web.modules).modules
    job_module = cast(SchedulerModule, next(m for m in modules if m.config_key == "scheduler"))
    job = job_module.add_job("Digest", "0 9 * * 1", "Summarise")
    job_module.repo.record_run(job.id, status="ok", session_id=chat_id, next_run_at=None)
    other = web.repo.create_session(title="Another chat")
    web.repo.add_message(other.id, "user", "unrelated question")
    web.repo.add_message(other.id, "assistant", "unrelated answer")
    await user.open("/")
    await user.should_see("unrelated answer")  # the most recently updated chat is open

    await open_section(user, "scheduler")
    user.find(marker="job-open").click()

    await user.should_see("Otters are playful.")
    await user.should_not_see(marker="settings-save")  # the dialog went away


# --- the menu on each chat -------------------------------------------------------------------------------------------------


async def test_a_chat_can_be_renamed_and_a_blank_name_is_ignored(user: User, web: AppContext):
    chat_id = one_chat(web)
    await user.open("/")

    user.find(marker="menu-rename").click()
    await user.should_see(marker="dialog-field")
    user.find(marker="dialog-field").clear().type("River otters")
    user.find(marker="dialog-ok").click()
    await user.should_not_see(
        marker="dialog-field"
    )  # the dialog closed, so the answer has been handled
    assert must(web.repo.get_session(chat_id)).title == "River otters"

    user.find(marker="menu-rename").click()
    await user.should_see(marker="dialog-field")
    user.find(marker="dialog-field").clear().type("   ")
    user.find(marker="dialog-ok").click()
    await user.should_not_see(marker="dialog-field")
    assert must(web.repo.get_session(chat_id)).title == "River otters"


async def test_cancelling_a_dialog_does_nothing(user: User, web: AppContext):
    chat_id = one_chat(web)
    await user.open("/")

    user.find(marker="menu-tag").click()
    await user.should_see(marker="dialog-field")
    user.find(marker="dialog-field").type("wildlife")
    user.find(marker="dialog-cancel").click()

    assert must(web.repo.get_session(chat_id)).tags == ()


async def test_tags_can_be_added_and_removed(user: User, web: AppContext):
    chat_id = one_chat(web)
    await user.open("/")

    user.find(marker="menu-tag").click()
    await user.should_see(marker="dialog-field")
    user.find(marker="dialog-field").type("wildlife")
    user.find(marker="dialog-ok").click()
    await user.should_see("#wildlife")
    assert must(web.repo.get_session(chat_id)).tags == ("wildlife",)

    user.find(marker="menu-remove-tag").click()
    await user.should_not_see("#wildlife")
    assert must(web.repo.get_session(chat_id)).tags == ()


async def test_a_chat_can_be_moved_into_a_new_folder(user: User, web: AppContext):
    chat_id = one_chat(web)
    await user.open("/")

    user.find(marker="menu-move").click()
    await user.should_see(marker="dialog-field")
    user.find(marker="dialog-field").type("Animals")
    user.find(marker="dialog-ok").click()
    await user.should_see("Animals (1)")

    folder = must(web.repo.get_session(chat_id)).folder_id
    assert folder is not None and web.repo.list_folders()[0].name == "Animals"


async def test_deleting_a_chat_asks_first_and_then_clears_the_screen(user: User, web: AppContext):
    chat_id = one_chat(web)
    await user.open("/")
    await user.should_see("Otters are playful.")

    user.find(marker="menu-delete").click()
    await user.should_see("and all its messages?")
    user.find(marker="dialog-cancel").click()
    assert web.repo.get_session(chat_id) is not None

    user.find(marker="menu-delete").click()
    user.find(marker="dialog-ok").click()
    await user.should_see("No chats yet")
    await user.should_not_see(
        "Otters are playful."
    )  # the open chat was the deleted one, so the screen cleared
    assert web.repo.list_sessions() == []


async def test_a_chat_is_downloaded_as_a_markdown_file(user: User, web: AppContext):
    one_chat(web, "What about otters?")
    await user.open("/")

    user.find(marker="menu-export").click()
    download = await user.download.next()

    text = download.content.decode("utf-8")
    assert text.startswith("# What about otters?")
    assert "## You\n\nTell me about otters" in text
    assert "Otters are playful." in text


async def test_the_theme_button_cycles_through_the_three_themes(user: User, web: AppContext):
    await user.open("/")
    seen = [web.settings.ui.theme]
    for _ in range(3):
        user.find(marker="theme").click()
        seen.append(web.settings.ui.theme)

    assert seen[0] == seen[3] and len(set(seen)) == 3


async def test_tags_are_badges_on_their_own_line_under_the_time(user: User, web: AppContext):
    chat_id = one_chat(web)
    web.repo.add_tag(chat_id, "wildlife")
    await user.open("/")

    await user.should_see(kind=ui.badge, content="#wildlife")
    assert not any(
        "#wildlife" in (label.text or "") for label in user.find(kind=ui.item_label).elements
    )


# --- closing the tab while a dialog is open ---------------------------------------------------------------------------------------


class StubDialog:
    """Just enough of a dialog for ``discard``: whether it is deleted, which page it belongs to, and a counter."""

    def __init__(self, client_id: str, *, deleted: bool = False) -> None:
        self.is_deleted = deleted
        self.client = SimpleNamespace(id=client_id)
        self.deleted_times = 0

    def delete(self) -> None:
        self.deleted_times += 1


def test_a_finished_dialog_is_removed_unless_its_page_or_it_is_already_gone(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(Client, "instances", {"alive": object()})

    normal, page_gone, already = (
        StubDialog("alive"),
        StubDialog("closed-tab"),
        StubDialog("alive", deleted=True),
    )
    for dialog in (normal, page_gone, already):
        discard(dialog)  # type: ignore[arg-type]

    assert (normal.deleted_times, page_gone.deleted_times, already.deleted_times) == (1, 0, 0)


async def test_closing_the_page_while_settings_is_open_logs_nothing_bad(
    user: User, web: AppContext, caplog: pytest.LogCaptureFixture
):
    await user.open("/")
    user.find(marker="settings").click()
    await user.should_see(marker="settings-save")

    must(user.client).delete()  # the tab is closed: the page goes first...
    await settle()  # ...and the dialog's await returns afterwards

    assert not [r for r in caplog.records if r.levelname in ("WARNING", "ERROR")]


# --- the menu on each folder -------------------------------------------------------------------------------------------------


async def test_a_folder_left_empty_by_an_earlier_version_is_removed_at_start_up(
    user: User, web: AppContext
):
    web.repo.get_or_create_folder("tt")
    await user.open("/")
    await user.should_see("Lemonade healthy")

    await user.should_not_see("tt (0)")
    assert web.repo.list_folders() == []


async def test_a_folder_asks_before_it_is_deleted(user: User, web: AppContext):
    chat = web.repo.create_session(folder_id=web.repo.get_or_create_folder("tt").id)
    web.repo.add_message(chat.id, "user", "q")
    web.repo.add_message(chat.id, "assistant", "a")
    await user.open("/")
    await user.should_see("tt (1)")

    user.find(marker="menu-folder-delete").click()
    await user.should_see("Delete the folder 'tt'?")
    user.find(marker="dialog-cancel").click()
    assert [f.name for f in web.repo.list_folders()] == ["tt"]  # cancelling keeps it

    user.find(marker="menu-folder-delete").click()
    user.find(marker="dialog-ok").click()
    await user.should_not_see("tt (1)")
    assert web.repo.list_folders() == []


async def test_deleting_a_folder_keeps_its_chats_which_become_unfiled(user: User, web: AppContext):
    chat = web.repo.create_session(folder_id=web.repo.get_or_create_folder("Work").id)
    web.repo.add_message(chat.id, "user", "hello")
    web.repo.add_message(chat.id, "assistant", "hi there")
    await user.open("/")
    await user.should_see("Work (1)")

    user.find(marker="menu-folder-delete").click()
    await user.should_see("Its 1 chat will be kept and become unfiled.")
    user.find(marker="dialog-ok").click()
    await user.should_not_see("Work (1)")

    kept = web.repo.get_session(chat.id)
    assert kept is not None and kept.folder_id is None


async def test_a_folder_can_be_renamed_but_not_to_a_name_another_folder_has(
    user: User, web: AppContext
):
    first = web.repo.get_or_create_folder("Home")
    second = web.repo.get_or_create_folder("Work")
    for folder in (first, second):  # a folder exists only while it holds a chat
        web.repo.create_session(folder_id=folder.id)
    await user.open("/")

    user.find(marker="menu-folder-rename").click()  # the first folder listed is "Home"
    await user.should_see(marker="dialog-field")
    user.find(marker="dialog-field").clear().type("Garden")
    user.find(marker="dialog-ok").click()
    await user.should_see("Garden (1)")
    assert web.repo.list_folders()[0].name == "Garden" and first.id in {
        f.id for f in web.repo.list_folders()
    }

    with pytest.raises(ValueError, match="already exists"):
        web.repo.rename_folder(first.id, "work")  # names are compared ignoring case
    with pytest.raises(ValueError, match="needs a name"):
        web.repo.rename_folder(first.id, "  ")


async def test_new_chat_on_a_page_that_is_already_new_says_so(user: User, web: AppContext):
    await user.open("/")
    await user.should_see("Start a conversation")

    user.find(marker="new-chat").click()

    await user.should_see("This is already a new chat.")
