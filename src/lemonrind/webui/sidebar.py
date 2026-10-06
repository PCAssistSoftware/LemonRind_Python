"""The chat list in the left drawer: search, folders, and a menu on every chat.

Python / NiceGUI ideas used here:

* ``@ui.refreshable`` - marks a method that *draws* some widgets. Calling ``.refresh()`` on it throws away
  what it drew and draws it again with fresh data. It is the simplest way to keep a list in step with the
  database: change the data, call ``refresh()``.
* **Callbacks as constructor arguments** - the list does not know about the rest of the page. It is told
  (``on_open``, ``on_new``, ...) what to call when something happens, which keeps the pieces independent.
* ``itertools.groupby``-style grouping done with a plain dictionary.
"""

from __future__ import annotations

import html
from collections.abc import Callable

from nicegui import ui

from lemonrind.chats import ChatSession, Folder, export_markdown, relative_time, safe_filename
from lemonrind.chats.text import Parts, highlight_parts, parse_snippet
from lemonrind.webui.context import AppContext
from lemonrind.webui.dialogs import ask_confirm, ask_folder, ask_text

LIST_LIMIT = 200  # most chats shown at once


def group_by_folder(
    sessions: list[ChatSession], folders: list[Folder], *, include_empty_folders: bool
) -> list[tuple[Folder | None, list[ChatSession]]]:
    """Arrange chats for display: each folder (by name) with its chats, then the unfiled chats.

    A plain function with no widgets, so it is easy to test. ``sessions`` keeps its incoming order (most
    recently used first) inside each group.
    """
    by_folder: dict[str | None, list[ChatSession]] = {}
    for session in sessions:
        by_folder.setdefault(session.folder_id, []).append(session)

    groups: list[tuple[Folder | None, list[ChatSession]]] = []
    for folder in sorted(folders, key=lambda f: f.name.lower()):
        chats = by_folder.get(folder.id, [])
        if chats or include_empty_folders:
            groups.append((folder, chats))
    if by_folder.get(None):
        groups.append((None, by_folder[None]))
    return groups


def marked_html(parts: Parts, *, tag: str = "div") -> ui.html:
    """Show text with its matched pieces highlighted.

    Every piece is HTML-escaped before it goes in, so a chat that happens to contain ``<script>`` is shown as
    text, not run. That is why the sanitiser can be switched off here: the only tags in the result are the
    ``<mark>`` elements this function adds itself.
    """
    pieces = [
        f'<mark class="lr-mark">{html.escape(text)}</mark>' if matched else html.escape(text)
        for text, matched in parts
    ]
    return ui.html("".join(pieces), sanitize=False, tag=tag)


def menu_entry(
    text: str, icon: str, on_click: Callable[..., object], marker: str, *, danger: bool = False
) -> None:
    """One row of a context menu: an icon, then the words (``ui.menu_item`` has no icon of its own)."""
    with ui.menu_item(on_click=on_click).mark(marker):
        with ui.item_section().props("avatar").classes("min-w-0"):
            ui.icon(icon, size="sm").props("color=negative" if danger else "")
        with ui.item_section():
            ui.label(text).classes("text-negative" if danger else "")


class ChatList:
    def __init__(
        self,
        context: AppContext,
        *,
        current_id: Callable[[], str | None],
        on_new: Callable[[], None],
        on_open: Callable[[ChatSession], None],
        on_changed: Callable[[], None],
        on_deleted: Callable[[str], None],
    ) -> None:
        self._repo = context.repo
        self._current_id = current_id
        self._on_new = on_new
        self._on_open = on_open
        self._on_changed = on_changed  # a chat was renamed, filed or tagged
        self._on_deleted = on_deleted
        self._query = ""
        self._snippets: dict[
            str, str
        ] = {}  # chat id -> a piece of its best matching message (while searching)

    def build(self) -> None:
        new_chat = ui.button("New chat", icon="add", on_click=self._on_new)
        new_chat.props("color=primary").classes("w-full").mark("new-chat")
        search = ui.input(placeholder="Search chats").props("outlined dense clearable debounce=250")
        search.classes("w-full").mark("search")
        search.on_value_change(self._search_changed)
        self.render()

    def _search_changed(self, event) -> None:
        self._query = (event.value or "").strip()
        self.render.refresh()

    @ui.refreshable
    def render(self) -> None:
        searching = bool(self._query)
        if searching:
            hits = self._repo.search_hits(self._query)[:LIST_LIMIT]
            sessions = [hit.session for hit in hits]
            self._snippets = {hit.session.id: hit.snippet for hit in hits if hit.snippet}
        else:
            sessions = self._repo.list_sessions(limit=LIST_LIMIT)
            self._snippets = {}
        if not sessions and searching:
            ui.label("No chats found.").classes("text-caption lr-muted q-pa-sm")
            return

        groups = group_by_folder(
            sessions, self._repo.list_folders(), include_empty_folders=not searching
        )
        if not groups:
            ui.label("No chats yet. Type a message to start one.").classes(
                "text-caption lr-muted q-pa-sm"
            )
        with ui.list().props("dense").classes("w-full"):
            for folder, chats in groups:
                if folder is not None:
                    self._folder_header(folder, len(chats), searching)
                    if folder.collapsed and not searching:
                        continue
                elif len(groups) > 1:
                    ui.item_label("Unfiled").classes("text-weight-medium lr-unfiled")
                for chat in chats:
                    self._chat_item(chat)

    def _folder_header(self, folder: Folder, count: int, searching: bool) -> None:
        def toggle() -> None:
            self._repo.set_folder_collapsed(folder.id, not folder.collapsed)
            self.render.refresh()

        with ui.item(on_click=toggle).props("dense"):
            with ui.item_section().props("avatar"):
                ui.icon("chevron_right" if folder.collapsed and not searching else "expand_more")
            with ui.item_section():
                ui.item_label(f"{folder.name} ({count})").classes("text-weight-medium")
            with ui.item_section().props("side"):
                with ui.button(icon="more_vert").props("flat dense round size=sm").on("click.stop"):
                    with ui.menu():
                        menu_entry(
                            "Rename folder...",
                            "edit",
                            lambda f=folder: self._rename_folder(f),
                            "menu-folder-rename",
                        )
                        menu_entry(
                            "Delete folder",
                            "delete",
                            lambda f=folder, n=count: self._delete_folder(f, n),
                            "menu-folder-delete",
                            danger=True,
                        )

    def _chat_item(self, chat: ChatSession) -> None:
        selected = chat.id == self._current_id()
        with (
            ui.item(on_click=lambda c=chat: self._on_open(c))
            .props("clickable" + (" active" if selected else ""))
            .mark("chat-item")
        ):
            with ui.item_section():
                if self._query:
                    marked_html(highlight_parts(chat.title, self._query)).classes(
                        "text-weight-medium lr-one-line"
                    ).mark("chat-title")
                else:
                    ui.item_label(chat.title).props("lines=1").classes("text-weight-medium")
                if snippet := self._snippets.get(chat.id):
                    marked_html(parse_snippet(snippet)).classes("lr-snippet").mark("chat-snippet")
                ui.item_label(relative_time(chat.updated_at)).props("caption lines=1")
                if (
                    chat.tags
                ):  # on a line of their own, as small badges, so they do not blur into the time
                    with ui.row().classes("gap-1 q-mt-xs"):
                        for tag in chat.tags:
                            badge = ui.badge(f"#{tag}").props("outline").classes("lr-tag")
                            if self._query and any(
                                hit for _, hit in highlight_parts(tag, self._query)
                            ):
                                badge.classes("lr-mark")  # the search matched this tag
            with ui.item_section().props("side"):
                with ui.button(icon="more_vert").props("flat dense round size=sm").on("click.stop"):
                    with ui.menu():
                        menu_entry("Rename", "edit", lambda c=chat: self._rename(c), "menu-rename")
                        menu_entry(
                            "Move to folder...",
                            "drive_file_move",
                            lambda c=chat: self._move(c),
                            "menu-move",
                        )
                        menu_entry(
                            "Add tag...", "sell", lambda c=chat: self._add_tag(c), "menu-tag"
                        )
                        menu_entry(
                            "Export as Markdown",
                            "download",
                            lambda c=chat: self._export(c),
                            "menu-export",
                        )
                        for tag in chat.tags:
                            menu_entry(
                                f"Remove tag #{tag}",
                                "label_off",
                                lambda c=chat, t=tag: self._remove_tag(c, t),
                                "menu-remove-tag",
                            )
                        ui.separator()
                        menu_entry(
                            "Delete",
                            "delete",
                            lambda c=chat: self._delete(c),
                            "menu-delete",
                            danger=True,
                        )

    # --- actions on one chat ----------------------------------------------------------------------------------

    def _export(self, chat: ChatSession) -> None:
        """Download the chat as a Markdown file."""
        messages = self._repo.list_messages(chat.id)
        text = export_markdown(chat, messages)
        ui.download.content(text.encode("utf-8"), safe_filename(chat.title), "text/markdown")

    async def _rename(self, chat: ChatSession) -> None:
        title = await ask_text("Rename chat", "Title", chat.title, ok_text="Rename")
        if title and title.strip():
            self._repo.rename_session(chat.id, title)
            self._changed()

    async def _rename_folder(self, folder: Folder) -> None:
        name = await ask_text("Rename folder", "Name", folder.name, ok_text="Rename")
        if not name or not name.strip() or name.strip() == folder.name:
            return
        try:
            self._repo.rename_folder(folder.id, name)
        except ValueError as error:
            ui.notify(str(error), type="warning")
            return
        self._changed()

    async def _delete_folder(self, folder: Folder, count: int) -> None:
        kept = (
            f" Its {count} chat{'s' if count != 1 else ''} will be kept and become unfiled."
            if count
            else ""
        )
        if await ask_confirm("Delete folder", f"Delete the folder '{folder.name}'?{kept}"):
            self._repo.delete_folder(folder.id)
            self._changed()

    async def _move(self, chat: ChatSession) -> None:
        answer = await ask_folder(self._repo.list_folders(), chat.folder_id)
        if answer is None:
            return
        folder_id, new_name = answer
        if new_name:
            folder_id = self._repo.get_or_create_folder(new_name).id
        self._repo.move_to_folder(chat.id, folder_id)
        self._changed()

    async def _add_tag(self, chat: ChatSession) -> None:
        name = await ask_text("Add tag", "Tag", ok_text="Add")
        if name and name.strip():
            self._repo.add_tag(chat.id, name)
            self._changed()

    def _remove_tag(self, chat: ChatSession, tag: str) -> None:
        self._repo.remove_tag(chat.id, tag)
        self._changed()

    async def _delete(self, chat: ChatSession) -> None:
        if await ask_confirm("Delete chat", f"Delete '{chat.title}' and all its messages?"):
            self._repo.delete_session(chat.id)
            self._on_deleted(chat.id)
            self.render.refresh()

    def _changed(self) -> None:
        self.render.refresh()
        self._on_changed()
