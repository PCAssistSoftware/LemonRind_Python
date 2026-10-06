"""The Knowledge bases section of Settings: create knowledge bases, add documents to them, and attach one to this chat.

One drop-down does two jobs: the knowledge base chosen in it is the one **attached to the current chat**, and
it is also the one whose sources are listed and added to below. Ingestion (reading, chunking and embedding)
can take a while for a large document, so it runs in the background and the list refreshes itself while
anything is still being read.

Python / NiceGUI ideas used here:

* ``ui.upload`` - the browser sends the file to the server; the handler gets a ``FileUpload`` with the
  original ``name`` and a ``save(path)`` method. We save it to a temporary folder, read it from there, and
  delete the folder afterwards (``try / finally`` guarantees the clean-up even if reading fails).
* ``asyncio.create_task`` plus a ``set`` of the running tasks (kept so they are not garbage-collected) for
  work that outlives the click that started it.
* ``ui.timer`` that polls: the section redraws its list while any source is still ``ingesting``.
* ``@ui.refreshable`` for the list, as in the other screens.
"""

from __future__ import annotations

import asyncio
import shutil
import sqlite3
import tempfile
from collections.abc import Coroutine
from pathlib import Path

from nicegui import ui
from nicegui.events import MultiUploadEventArguments

from lemonrind.chats import Conversation
from lemonrind.lemonade.client import LemonadeError
from lemonrind.modules.knowledge import (
    OPTION_KEY,
    DuplicateNameError,
    KnowledgeFileError,
    KnowledgeModule,
    Source,
    copy_database,
)
from lemonrind.modules.knowledge.extract import SUPPORTED_EXTENSIONS
from lemonrind.webui.dialogs import ask_confirm, ask_name_and_description

TYPE_ICONS = {"file": "description", "folder": "folder", "website": "language", "text": "notes"}
MAX_KB_FILE_BYTES = (
    4_000_000_000  # the biggest knowledge base file that can be uploaded through the browser
)
EXPORT_KEPT_SECONDS = (
    600  # a downloadable copy is deleted from the temporary folder after this long
)


def format_size(size: int) -> str:
    """``1.5 MB``, ``830 KB``..."""
    value = float(size)
    for unit in ("bytes", "KB", "MB", "GB"):
        if value < 1000 or unit == "GB":
            return f"{value:,.0f} {unit}" if unit in ("bytes", "KB") else f"{value:,.1f} {unit}"
        value /= 1000
    return f"{size} bytes"


def build_knowledge(module: KnowledgeModule, conversation: Conversation) -> None:
    """Draw the section into the current container (the Settings dialog puts it in a panel)."""
    repo = module.repo
    running: set[asyncio.Task] = set()
    ignore_change = False  # set while the code (not the user) changes the drop-down

    def attached_id() -> str | None:
        kb_id = conversation.options.get(OPTION_KEY)
        return kb_id if kb_id and repo.get_kb(kb_id) else None

    def in_background(work: Coroutine) -> None:
        task = asyncio.create_task(work)
        running.add(task)  # a task nobody references can be garbage-collected mid-run
        task.add_done_callback(running.discard)

    # --- the chooser --------------------------------------------------------------------------------------

    def fill_chooser() -> None:
        nonlocal ignore_change
        ignore_change = True  # setting options re-fires on_value_change; ignore that
        options = {"": "(none: no knowledge base for this chat)"}
        options.update({kb.id: kb.name for kb in repo.list_kbs()})
        chooser.set_options(options, value=attached_id() or "")
        ignore_change = False
        show_details()

    def show_details() -> None:
        """The description, how big it is and which embedding model made it; a warning if that model is not in use."""
        kb = repo.get_kb(attached_id() or "")
        description_label.text = kb.description if kb else ""
        description_label.set_visibility(bool(kb and kb.description))
        has_kb = kb is not None
        details_label.set_visibility(has_kb)
        for button in (download_button, delete_button):
            button.set_visibility(has_kb)
        if kb is None:
            warning_row.set_visibility(False)
            return
        stats = repo.stats(kb.id)
        model = (
            f"embedding model {kb.embedding_model} ({kb.embedding_dim} numbers per chunk)"
            if kb.embedding_model
            else "embedding model not recorded"
        )
        noun = "chunk" if stats.chunks == 1 else "chunks"
        details_label.text = f"{stats.chunks:,} {noun}, {format_size(stats.size_bytes)}, {model}"
        in_background(check_model(kb.id))

    async def check_model(kb_id: str) -> None:
        """Show the re-embed warning if the model now in use is not the one this knowledge base was built with."""
        kb = repo.get_kb(kb_id)
        if kb is None:
            return
        problem = await module.compatibility(kb)
        if attached_id() != kb_id:
            return  # a different knowledge base was chosen meanwhile
        warning_label.text = problem
        warning_row.set_visibility(bool(problem))
        reembed_button.set_enabled(kb_id not in module.reembedding)

    async def reembed() -> None:
        kb = repo.get_kb(attached_id() or "")
        if kb is None:
            return
        chunks = repo.stats(kb.id).chunks
        if await ask_confirm(
            "Re-embed knowledge base",
            f"Embed all {chunks:,} chunks of '{kb.name}' again with the embedding model in use now? "
            "The original documents are not needed. This can take a while.",
            ok_text="Re-embed",
        ):
            in_background(do_reembed(kb.id, kb.name))

    async def do_reembed(kb_id: str, name: str) -> None:
        task = asyncio.create_task(module.reembed(kb_id))
        await asyncio.sleep(0)  # let it start, so the screen can show that it is busy
        show_details()
        with status_box:
            ui.notify(f"Re-embedding '{name}'...", timeout=2500)
        try:
            count = await task
        except LemonadeError as error:
            with status_box:
                ui.notify(f"Could not re-embed '{name}': {error}", type="negative", multi_line=True)
        else:
            with status_box:
                ui.notify(f"Re-embedded {count:,} chunks of '{name}'.", type="positive")
        finally:
            show_details()
            sources.refresh()

    async def download() -> None:
        """Give the browser a self-contained copy of the knowledge base to keep or take to another computer."""
        kb = repo.get_kb(attached_id() or "")
        if kb is None or kb.path is None:
            return
        folder = Path(tempfile.mkdtemp(prefix="lemonrind-export-"))
        target = folder / kb.path.name
        await asyncio.to_thread(
            copy_database, kb.path, target
        )  # a worker thread: a big one takes a while
        ui.download.file(target, filename=f"{kb.name}{kb.path.suffix}")
        asyncio.get_running_loop().call_later(
            EXPORT_KEPT_SECONDS, lambda: shutil.rmtree(folder, ignore_errors=True)
        )

    async def add_kb_file(path: Path) -> None:
        """Add a knowledge base from a .kb file: check it, copy it into this app's folder, attach it to this chat."""
        plan = None
        try:
            plan = repo.plan_import(path)
            await asyncio.to_thread(copy_database, path, plan.destination)
            kb = repo.finish_import(plan)
        except KnowledgeFileError as error:
            ui.notify(str(error), type="negative", multi_line=True)
            return
        except (OSError, sqlite3.Error) as error:
            if plan is not None:
                plan.destination.unlink(missing_ok=True)  # do not leave half a copy behind
            ui.notify(f"Could not add that file: {error}", type="negative", multi_line=True)
            return
        conversation.set_option(OPTION_KEY, kb.id)
        fill_chooser()
        sources.refresh()
        add_panel.set_visibility(True)
        ui.notify(
            f"Added the knowledge base '{kb.name}' and attached it to this chat.", type="positive"
        )

    async def add_kb_from_path() -> None:
        text = (kb_path_input.value or "").strip().strip('"')
        if not text:
            return
        path = Path(text).expanduser()
        if not path.is_file():
            ui.notify("That file does not exist on this computer.", type="warning")
            return
        await add_kb_file(path)
        kb_path_input.value = ""

    async def add_kb_uploads(event: MultiUploadEventArguments) -> None:
        for upload in event.files:
            folder = Path(tempfile.mkdtemp(prefix="lemonrind-upload-"))
            try:
                path = folder / Path(upload.name).name
                await upload.save(path)
                await add_kb_file(path)
            finally:
                shutil.rmtree(folder, ignore_errors=True)
        kb_uploader.reset()

    def chosen(event) -> None:
        if ignore_change:
            return
        conversation.set_option(OPTION_KEY, event.value or None)
        show_details()
        sources.refresh()
        add_panel.set_visibility(attached_id() is not None)

    async def create() -> None:
        answer = await ask_name_and_description("New knowledge base")
        if answer is None or not answer[0].strip():
            return
        try:
            kb = repo.create_kb(*answer)
        except DuplicateNameError as error:
            ui.notify(str(error), type="warning")
            return
        conversation.set_option(OPTION_KEY, kb.id)
        fill_chooser()
        sources.refresh()
        add_panel.set_visibility(True)

    async def delete_kb() -> None:
        kb = repo.get_kb(attached_id() or "")
        if kb is None:
            return
        if await ask_confirm("Delete knowledge base", f"Delete '{kb.name}' and everything in it?"):
            repo.delete_kb(kb.id)
            conversation.set_option(OPTION_KEY, None)
            fill_chooser()
            sources.refresh()
            add_panel.set_visibility(False)

    # --- the sources -------------------------------------------------------------------------------------

    @ui.refreshable
    def sources() -> None:
        kb_id = attached_id()
        if kb_id is None:
            ui.label("Create a knowledge base, or pick one above, to add documents to it.").classes(
                "text-caption lr-muted"
            )
            return
        listed = repo.list_sources(kb_id)
        if not listed:
            ui.label("No sources yet. Add some below.").classes("text-caption lr-muted")
        for source in listed:
            source_row(source)

    def source_row(source: Source) -> None:
        with ui.row().classes("w-full items-center no-wrap gap-2"):
            ui.icon(TYPE_ICONS[source.source_type]).classes("lr-muted")
            with ui.column().classes("gap-0 flex-grow"):
                ui.label(source.display_name).classes("ellipsis")
                if source.status == "ingesting":
                    ui.label("Reading and embedding...").classes("text-caption lr-warn")
                elif source.status == "ready":
                    ui.label(
                        f"{source.chunk_count:,} chunk{'' if source.chunk_count == 1 else 's'}"
                    ).classes("text-caption lr-muted")
                else:
                    ui.label(source.error or "Failed").classes("text-caption text-negative")
            if source.status == "ingesting":
                ui.spinner(size="sm")
            ui.button(icon="delete", on_click=lambda s=source: remove(s)).props(
                "flat dense round color=negative"
            ).tooltip("Remove this source")

    def remove(source: Source) -> None:
        repo.delete_source(source.id)
        sources.refresh()

    was_busy = False

    def poll() -> None:
        """Redraw the list while anything is still being read, and once more when it has finished."""
        nonlocal was_busy
        kb_id = attached_id()
        busy = bool(running) or (
            kb_id is not None and any(s.status == "ingesting" for s in repo.list_sources(kb_id))
        )
        if busy or was_busy:
            sources.refresh()
        was_busy = busy

    # --- adding sources --------------------------------------------------------------------------------

    async def ingest_files(event: MultiUploadEventArguments) -> None:
        kb_id = attached_id()
        if kb_id is None:
            ui.notify("Attach a knowledge base first.", type="warning")
            return
        for upload in event.files:
            folder = Path(tempfile.mkdtemp(prefix="lemonrind-upload-"))
            path = folder / Path(upload.name).name  # .name drops any directory part of the name
            await upload.save(path)
            in_background(add_file_and_clean_up(kb_id, path, folder))
        uploader.reset()

    async def add_file_and_clean_up(kb_id: str, path: Path, folder: Path) -> None:
        try:
            await module.ingestor.add_file(kb_id, path)
        finally:
            shutil.rmtree(folder, ignore_errors=True)

    def add_folder() -> None:
        kb_id, text = attached_id(), (folder_input.value or "").strip().strip('"')
        if kb_id is None or not text:
            return
        path = Path(text).expanduser()
        if not path.is_dir():
            ui.notify("That folder does not exist on this computer.", type="warning")
            return
        in_background(module.ingestor.add_folder(kb_id, path))
        folder_input.value = ""

    def add_website() -> None:
        kb_id, url = attached_id(), (url_input.value or "").strip()
        if kb_id is None or not url:
            return
        in_background(module.ingestor.add_website(kb_id, url))
        url_input.value = ""

    def add_note() -> None:
        kb_id, text = attached_id(), (note_input.value or "").strip()
        if kb_id is None or not text:
            return
        in_background(
            module.ingestor.add_text(kb_id, (note_name.value or "").strip() or text[:40], text)
        )
        note_input.value = ""
        note_name.value = ""

    # --- the section ------------------------------------------------------------------------------------

    with ui.column().classes("w-full gap-2"):
        ui.label(
            "Collections of your own documents. The one chosen here is attached to this chat: the parts of "
            "it that match each message you send are shown to the assistant."
        ).classes("text-caption lr-muted")

        with ui.row().classes("w-full items-center no-wrap gap-1 q-mt-sm"):
            chooser = ui.select({}, label="Attached to this chat", on_change=chosen)
            chooser.classes("flex-grow").props("outlined dense")
            ui.button(icon="add", on_click=create).props("flat dense round").tooltip(
                "New knowledge base"
            ).mark("kb-new")
            download_button = ui.button(icon="download", on_click=download).props(
                "flat dense round"
            )
            download_button.tooltip(
                "Download this knowledge base as a file (to keep it or use it on another computer)"
            )
            download_button.mark("kb-download")
            delete_button = ui.button(icon="delete_forever", on_click=delete_kb).props(
                "flat dense round color=negative"
            )
            delete_button.tooltip("Delete this knowledge base")

        description_label = ui.label().classes("text-caption lr-muted")
        description_label.mark("kb-description")
        details_label = ui.label().classes("text-caption lr-muted")
        details_label.mark("kb-details")
        with ui.row().classes("w-full items-center no-wrap gap-2") as warning_row:
            ui.icon("warning", color="warning")
            warning_label = ui.label().classes("text-caption lr-warn flex-grow")
            reembed_button = ui.button("Re-embed", icon="refresh", on_click=reembed).props(
                "flat dense no-caps"
            )
            reembed_button.mark("kb-reembed")
        warning_row.set_visibility(False)
        status_box = ui.column().classes(
            "hidden"
        )  # only gives background tasks a place to show notices from
        ui.label("Sources").classes("text-weight-medium q-mt-sm")
        sources()

        with ui.column().classes("w-full gap-2") as add_panel:
            ui.separator()
            ui.label("Add").classes("text-weight-medium")
            accepted = ",".join(SUPPORTED_EXTENSIONS)
            uploader = (
                ui.upload(
                    on_multi_upload=ingest_files,
                    multiple=True,
                    auto_upload=True,
                    label=f"Drop files here or click to choose ({', '.join(SUPPORTED_EXTENSIONS)})",
                )
                .props(f"accept={accepted} flat bordered")
                .classes("w-full")
            )
            with ui.row().classes("w-full items-center no-wrap gap-1"):
                folder_input = (
                    ui.input(placeholder="A folder on this computer, e.g. C:\\Docs")
                    .props("outlined dense")
                    .classes("flex-grow")
                )
                ui.button("Add folder", on_click=add_folder).props("flat")
            with ui.row().classes("w-full items-center no-wrap gap-1"):
                url_input = (
                    ui.input(placeholder="A web page, e.g. https://example.com/guide")
                    .props("outlined dense")
                    .classes("flex-grow")
                )
                ui.button("Add page", on_click=add_website).props("flat")
            with ui.column().classes("w-full gap-1"):
                note_name = (
                    ui.input(placeholder="Note title (optional)")
                    .props("outlined dense")
                    .classes("w-full")
                )
                note_input = (
                    ui.textarea(placeholder="Paste or type a note")
                    .props("outlined dense autogrow")
                    .classes("w-full")
                )
                ui.button("Add note", on_click=add_note).props("flat")

        ui.separator()
        ui.label("Add a knowledge base from a file").classes("text-weight-medium")
        ui.label(
            "Use a .kb file made on another computer (the download button above makes one). It is copied into this "
            "app's own knowledge folder, so the original can be put away. If it was built with a different embedding "
            "model than the one in use here, you are offered a re-embed."
        ).classes("text-caption lr-muted")
        kb_uploader = (
            ui.upload(
                on_multi_upload=add_kb_uploads,
                multiple=False,
                auto_upload=True,
                max_file_size=MAX_KB_FILE_BYTES,
                label="Drop a .kb file here or click to choose",
            )
            .props("accept=.kb flat bordered")
            .classes("w-full")
        )
        with ui.row().classes("w-full items-center no-wrap gap-1"):
            kb_path_input = (
                ui.input(
                    placeholder="Or the path of a .kb file on this computer (best for very big ones)"
                )
                .props("outlined dense")
                .classes("flex-grow")
            )
            kb_path_input.mark("kb-import-path")
            ui.button("Add file", on_click=add_kb_from_path).props("flat").mark("kb-import")

    fill_chooser()
    add_panel.set_visibility(attached_id() is not None)
    ui.timer(1.0, poll)
