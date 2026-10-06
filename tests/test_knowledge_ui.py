"""The terminal /kb commands and the web Knowledge bases screen."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from nicegui import ui
from nicegui.testing import User

from lemonrind.chats import ChatRepository
from lemonrind.config import Settings
from lemonrind.modules import KnowledgeModule, build_registry
from lemonrind.modules.knowledge import OPTION_KEY, KnowledgeRepository
from lemonrind.storage import Database
from lemonrind.terminal.chatloop import ChatLoop
from lemonrind.webui.context import set_context
from lemonrind.webui.page import register_pages
from tests.conftest import elements, open_section
from tests.test_chatloop import FakeClient, ScriptedConsole
from tests.test_memory import FakeMemoryClient
from tests.test_settings_sections import settle
from tests.test_webui import FakeLemonade, make_context


def make_loop(lines: list[str], tmp_path: Path):
    settings = Settings()
    settings.modules.knowledge.min_similarity = 0.6
    db = Database(":memory:")
    console = ScriptedConsole(lines)
    modules = build_registry(settings, tmp_path, db=db, client=FakeMemoryClient())  # type: ignore[arg-type]
    loop = ChatLoop(
        client=FakeClient(),  # type: ignore[arg-type]
        settings=settings,
        repo=ChatRepository(db),
        console=console,
        model="m",
        modules=modules,
    )
    return loop, console, modules


async def test_kb_commands_create_fill_attach_and_remove(tmp_path: Path):
    notes = tmp_path / "menu.txt"
    notes.write_text("pizza food on fridays", encoding="utf-8")
    loop, console, modules = make_loop(
        [
            "/kb",  # none yet
            "/kb new Handbook",
            f"/kb add {notes}",
            "/kb note manchester city office",
            "/kb sources",
            "/kb remove 2",
            "/kb use none",
            "/kb sources",
            "/kb use handbook",  # names match ignoring case
            "/kb delete Handbook",
            "/quit",
        ],
        tmp_path,
    )

    await loop.run(resume=False)

    output = console.output
    assert "No knowledge bases yet" in output
    assert "Created 'Handbook' and attached it to this chat." in output
    assert "menu.txt" in output and "ready: 1 chunks" in output
    assert "Removed." in output
    assert "Detached." in output and "No knowledge base is attached" in output
    assert "Attached 'Handbook'." in output and "Deleted 'Handbook'." in output
    module = next(m for m in modules.modules if isinstance(m, KnowledgeModule))
    assert module.repo.list_kbs() == []
    assert OPTION_KEY not in loop.conversation.options  # deleting the attached one detaches it


async def test_kb_add_reports_what_it_cannot_read(tmp_path: Path):
    loop, console, _ = make_loop(
        ["/kb new K", "/kb add nowhere/at/all", "/kb add", "/quit"], tmp_path
    )

    await loop.run(resume=False)

    assert "Not a file, folder or web address" in console.output
    assert "Usage: /kb add PATH-OR-URL" in console.output


async def test_kb_commands_explain_when_the_module_is_off(tmp_path: Path):
    loop, console, _ = make_loop(["/module knowledge off", "/kb", "/quit"], tmp_path)

    await loop.run(resume=False)

    assert "The Knowledge bases module is off" in console.output


async def test_the_footer_shows_the_attached_knowledge_base_and_the_screen_lists_its_sources(
    user: User, tmp_path: Path
):
    context = make_context(tmp_path, FakeLemonade())
    context.modules = build_registry(  # this time with an embedding model available
        context.settings,
        tmp_path,
        db=context.db,
        client=FakeMemoryClient(),  # type: ignore[arg-type]
    )
    module = next(m for m in context.modules.modules if isinstance(m, KnowledgeModule))  # type: ignore[union-attr]
    kb = module.repo.create_kb("Handbook")
    await module.ingestor.add_text(kb.id, "menu.txt", "pizza food")
    chat = context.repo.create_session()
    context.repo.add_message(chat.id, "user", "hi")
    context.repo.add_message(chat.id, "assistant", "hello")
    context.repo.set_option(chat.id, OPTION_KEY, kb.id)
    set_context(context)
    register_pages()

    await user.open("/")
    await user.should_see("Lemonade healthy")
    await settle()
    select = elements(user, "kb-select")[0]
    assert (
        select.value == kb.id
    )  # the footer drop-down shows the knowledge base attached to this chat
    await open_section(user, "knowledge")

    await user.should_see("Attached to this chat")
    await user.should_see("menu.txt")
    await user.should_see("1 chunk")
    await user.should_see(kind=ui.upload)
    set_context(None)


async def test_a_new_knowledge_base_can_have_a_description_that_shows_beside_it(
    user: User, tmp_path: Path
):
    context = make_context(tmp_path, FakeLemonade())
    module = next(m for m in context.modules.modules if isinstance(m, KnowledgeModule))  # type: ignore[union-attr]
    set_context(context)
    register_pages()
    try:
        await user.open("/")
        await open_section(user, "knowledge")
        user.find(marker="kb-new").click()
        await user.should_see("Description (optional)")
        user.find(marker="dialog-field").type("Recipes")
        user.find(marker="dialog-description").type("Family dinners and baking")
        user.find(marker="dialog-ok").click()

        for _ in range(40):  # creating happens just after the click
            if module.repo.list_kbs():
                break
            await asyncio.sleep(0.05)
        await user.should_see(marker="kb-description")
        (kb,) = module.repo.list_kbs()
        assert (kb.name, kb.description) == ("Recipes", "Family dinners and baking")
    finally:
        set_context(None)


def test_knowledge_bases_kept_in_the_main_database_move_into_files_with_the_same_id(tmp_path: Path):
    from lemonrind.vectors import to_blob

    db = Database(tmp_path / "old.db")
    with db.transaction() as conn:  # rows exactly as the earlier versions wrote them
        conn.execute(
            "INSERT INTO knowledge_bases (id, name, description, created_at) VALUES (?, ?, ?, ?)",
            ("kb-1", "Old", "from before", "2026-01-01T00:00:00+00:00"),
        )
        conn.execute(
            "INSERT INTO knowledge_sources (id, kb_id, source_type, reference, display_name, status, chunk_count, created_at)"
            " VALUES ('s-1', 'kb-1', 'text', NULL, 'notes', 'ready', 2, '2026-01-01T00:00:00+00:00')"
        )
        for index, text in enumerate(("first chunk", "second chunk")):
            conn.execute(
                "INSERT INTO knowledge_chunks (kb_id, source_id, chunk_index, label, content, embedding)"
                " VALUES ('kb-1', 's-1', ?, '', ?, ?)",
                (index, text, to_blob([1.0, 0.0, 0.5])),
            )
    repo = KnowledgeRepository(tmp_path / "knowledge")

    assert repo.migrate_from_main_database(db) == 1

    kb = repo.get_kb("kb-1")  # the same id, so a chat that had it attached still has it
    assert kb is not None and (kb.name, kb.description) == ("Old", "from before")
    assert (kb.embedding_model, kb.embedding_dim) == ("", 3)  # the model was not recorded then
    assert [c.content for c, _ in repo.searchable_chunks("kb-1")] == ["first chunk", "second chunk"]
    assert [s.display_name for s in repo.list_sources("kb-1")] == ["notes"]
    assert (
        db.conn.execute("SELECT COUNT(*) FROM knowledge_bases").fetchone()[0] == 0
    )  # moved, not copied
    assert db.conn.execute("SELECT COUNT(*) FROM knowledge_chunks").fetchone()[0] == 0
    assert repo.migrate_from_main_database(db) == 0  # and doing it again finds nothing more
    assert [p.suffix for p in (tmp_path / "knowledge").iterdir()] == [".kb"]


# --- the file-per-knowledge-base features of the screen -----------------------------------------------------------


def context_with_embeddings(tmp_path: Path):
    context = make_context(tmp_path, FakeLemonade())
    context.modules = build_registry(
        context.settings,
        tmp_path,
        db=context.db,
        client=FakeMemoryClient(),  # type: ignore[arg-type]  # its one embedding model is called "embed"
    )
    module = next(m for m in context.modules.modules if isinstance(m, KnowledgeModule))  # type: ignore[union-attr]
    return context, module


def attach_to_open_chat(context, kb_id: str) -> None:
    chat = context.repo.create_session()
    context.repo.add_message(chat.id, "user", "hi")
    context.repo.add_message(chat.id, "assistant", "hello")
    context.repo.set_option(chat.id, OPTION_KEY, kb_id)


async def test_the_screen_shows_how_big_the_knowledge_base_is_and_which_model_made_it(
    user: User, tmp_path: Path
):
    context, module = context_with_embeddings(tmp_path)
    kb = module.repo.create_kb("Handbook")
    await module.ingestor.add_text(kb.id, "menu.txt", "pizza food on fridays")
    attach_to_open_chat(context, kb.id)
    set_context(context)
    register_pages()
    try:
        await user.open("/")
        await open_section(user, "knowledge")

        await user.should_see(marker="kb-details")
        (details,) = elements(user, "kb-details")
        assert "1 chunk," in details.text and "embedding model embed" in details.text
        await user.should_not_see(marker="kb-reembed")  # the model in use is the one that built it
    finally:
        set_context(None)


async def test_a_knowledge_base_from_another_model_shows_a_warning_and_re_embeds_on_request(
    user: User, tmp_path: Path
):
    context, module = context_with_embeddings(tmp_path)
    kb = module.repo.create_kb("Handbook")
    source = module.repo.create_source(kb.id, "text", None, "menu.txt")
    module.repo.add_chunks(kb.id, source.id, 0, ["pizza food"], [[1.0, 0.0]], model="old-model")
    module.repo.mark_ready(source.id, 1)
    attach_to_open_chat(context, kb.id)
    set_context(context)
    register_pages()
    try:
        await user.open("/")
        await open_section(user, "knowledge")

        await user.should_see("old-model")  # the warning names both models
        await user.should_see(marker="kb-reembed")
        user.find(marker="kb-reembed").click()
        await user.should_see("Embed all 1 chunks of 'Handbook' again")
        user.find(marker="dialog-ok").click()

        for _ in range(60):  # the work happens in the background
            await asyncio.sleep(0.1)
            stored = module.repo.get_kb(kb.id)
            if stored and stored.embedding_model == "embed":
                break
        assert module.repo.get_kb(kb.id).embedding_model == "embed"  # type: ignore[union-attr]
        await user.should_not_see(marker="kb-reembed")  # and the warning goes away
    finally:
        set_context(None)


async def test_a_knowledge_base_file_can_be_added_by_its_path(user: User, tmp_path: Path):
    context, module = context_with_embeddings(tmp_path)
    elsewhere = KnowledgeRepository(tmp_path / "other computer")
    original = elsewhere.create_kb("Shared handbook", "made elsewhere")
    source = elsewhere.create_source(original.id, "text", None, "notes")
    elsewhere.add_chunks(original.id, source.id, 0, ["pizza food"], [[1.0, 0.0]], model="embed")
    elsewhere.mark_ready(source.id, 1)
    carried = elsewhere.export_copy(original.id, tmp_path / "carried.kb")
    set_context(context)
    register_pages()
    try:
        await user.open("/")
        await open_section(user, "knowledge")

        user.find(marker="kb-import-path").type(str(carried))
        user.find(marker="kb-import").click()

        for _ in range(60):
            await asyncio.sleep(0.1)
            if module.repo.find_kb("Shared handbook"):
                break
        added = module.repo.find_kb("Shared handbook")
        assert added is not None and added.id == original.id
        await user.should_see("made elsewhere")  # shown straight away, and attached to the chat
        assert [c.content for c, _ in module.repo.searchable_chunks(added.id)] == ["pizza food"]

        user.find(marker="kb-import-path").type(str(carried))  # the same file again
        user.find(marker="kb-import").click()
        await user.should_see("already added")
    finally:
        set_context(None)


async def test_a_path_that_is_not_a_knowledge_base_is_explained(user: User, tmp_path: Path):
    context, module = context_with_embeddings(tmp_path)
    not_a_kb = tmp_path / "notes.kb"
    not_a_kb.write_text("not a database", encoding="utf-8")
    set_context(context)
    register_pages()
    try:
        await user.open("/")
        await open_section(user, "knowledge")

        user.find(marker="kb-import-path").type(str(not_a_kb))
        user.find(marker="kb-import").click()

        await user.should_see("not a knowledge base file")
        assert module.repo.list_kbs() == []
    finally:
        set_context(None)


async def test_the_download_button_makes_a_copy_that_can_be_added_elsewhere(
    user: User, tmp_path: Path
):
    import tempfile

    context, module = context_with_embeddings(tmp_path)
    kb = module.repo.create_kb("Handbook")
    await module.ingestor.add_text(kb.id, "menu.txt", "pizza food on fridays")
    attach_to_open_chat(context, kb.id)
    temp = Path(tempfile.gettempdir())
    before = set(temp.glob("lemonrind-export-*"))
    set_context(context)
    register_pages()
    try:
        await user.open("/")
        await open_section(user, "knowledge")

        user.find(marker="kb-download").click()
        made: list[Path] = []
        for _ in range(60):  # the copy is made on a worker thread, then handed to the browser
            await asyncio.sleep(0.1)
            made = [
                f for d in set(temp.glob("lemonrind-export-*")) - before for f in d.glob("*.kb")
            ]
            if made:
                break

        assert len(made) == 1 and made[0].name.startswith(
            "handbook-"
        )  # a real file in a temporary folder
        again = KnowledgeRepository(tmp_path / "second computer")
        added = again.import_file(made[0])  # and it can be added as it is, on another computer
        assert added.id == kb.id and len(again.searchable_chunks(added.id)) == 1
        again.close()
    finally:
        for folder in set(temp.glob("lemonrind-export-*")) - before:
            shutil.rmtree(folder, ignore_errors=True)
        set_context(None)


async def test_terminal_kb_export_import_and_reembed(tmp_path: Path):
    carried = tmp_path / "carried.kb"
    loop, console, modules = make_loop(
        [
            "/kb new Handbook",
            "/kb note pizza food on fridays",
            f"/kb export {carried}",
            "/kb delete Handbook",
            f"/kb import {carried}",  # the same knowledge base comes back, with its content
            f"/kb import {carried}",  # a second time it is refused
            "/kb import",
            "/kb import nowhere/at/all.kb",
            "/kb reembed",
            "/kb",
            "/quit",
        ],
        tmp_path,
    )

    await loop.run(resume=False)

    output = console.output
    assert (
        "Saved 'Handbook' to" in output and carried.is_file()
    )  # (a long path wraps, so only the start is checked)
    assert "Added 'Handbook' and attached it to this chat." in output
    assert "already added" in output
    assert "Usage: /kb import" in output and "No such file" in output
    assert "Re-embedded 1 chunks of 'Handbook'." in output
    assert "model embed" in output  # the list shows which embedding model built each one
    module = next(m for m in modules.modules if isinstance(m, KnowledgeModule))
    kb = module.repo.find_kb("Handbook")
    assert kb is not None and len(module.repo.searchable_chunks(kb.id)) == 1
