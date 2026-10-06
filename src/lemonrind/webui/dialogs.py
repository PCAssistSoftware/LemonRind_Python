"""Small modal dialogs that *return an answer*: ``name = await ask_text(...)``.

NiceGUI dialogs can be awaited: the ``await`` pauses until the user presses a button, and ``dialog.submit(x)``
makes ``x`` the value of the ``await``. That turns "show a dialog, wait, then act" into straight-line code
instead of a chain of callbacks.

These dialogs are created under the page's layout rather than wherever the click happened. The chat list's menus
live inside a refreshable block that is rebuilt whenever a chat changes (for example when a reply finishes); a
dialog created inside one would be deleted by that rebuild while the user was still typing in it.
"""

from __future__ import annotations

from nicegui import Client, ui

from lemonrind.chats import Folder


def discard(dialog: ui.dialog) -> None:
    """Remove a finished dialog from the page, unless the page is already gone.

    If the person closes or reloads the tab while a dialog is open, the page (NiceGUI's *client*) is deleted first and the
    dialog's ``await`` then returns. Deleting an element of a client that no longer exists logs a warning and an error,
    so it is simply skipped: there is nothing left to tidy.
    """
    if dialog.is_deleted or dialog.client.id not in Client.instances:
        return
    dialog.delete()


def _on_page():
    """Use as ``with _on_page():`` so new elements become children of the page layout, which is never rebuilt."""
    return ui.context.client.layout


async def ask_text(title: str, label: str, value: str = "", *, ok_text: str = "OK") -> str | None:
    """Ask for one line of text. Returns ``None`` if the user cancels."""
    with _on_page(), ui.dialog() as dialog, ui.card().classes("w-96"):
        ui.label(title).classes("text-h6")
        field = ui.input(label, value=value).props("autofocus outlined").classes("w-full")
        field.mark("dialog-field")
        field.on("keydown.enter", lambda: dialog.submit(field.value))
        with ui.row().classes("w-full justify-end"):
            ui.button("Cancel", on_click=lambda: dialog.submit(None)).props("flat").mark(
                "dialog-cancel"
            )
            ui.button(ok_text, on_click=lambda: dialog.submit(field.value)).mark("dialog-ok")
    result = await dialog
    discard(dialog)
    return result


async def ask_name_and_description(
    title: str, *, ok_text: str = "Create"
) -> tuple[str, str] | None:
    """Ask for a name and an optional one-line description. Returns ``(name, description)``, or ``None`` if cancelled."""
    with _on_page(), ui.dialog() as dialog, ui.card().classes("w-96"):
        ui.label(title).classes("text-h6")
        name = ui.input("Name").props("autofocus outlined").classes("w-full").mark("dialog-field")
        description = (
            ui.input("Description (optional)")
            .props("outlined")
            .classes("w-full")
            .mark("dialog-description")
        )

        def submit() -> None:
            dialog.submit((name.value or "", description.value or ""))

        name.on("keydown.enter", submit)
        description.on("keydown.enter", submit)
        with ui.row().classes("w-full justify-end"):
            ui.button("Cancel", on_click=lambda: dialog.submit(None)).props("flat").mark(
                "dialog-cancel"
            )
            ui.button(ok_text, on_click=submit).mark("dialog-ok")
    result = await dialog
    discard(dialog)
    return result


async def ask_confirm(title: str, message: str, *, ok_text: str = "Delete") -> bool:
    """Ask a yes/no question. Returns ``True`` only if the user confirms."""
    with _on_page(), ui.dialog() as dialog, ui.card().classes("w-96"):
        ui.label(title).classes("text-h6")
        ui.label(message)
        with ui.row().classes("w-full justify-end"):
            ui.button("Cancel", on_click=lambda: dialog.submit(False)).props("flat").mark(
                "dialog-cancel"
            )
            ui.button(ok_text, on_click=lambda: dialog.submit(True)).props("color=negative").mark(
                "dialog-ok"
            )
    result = await dialog
    discard(dialog)
    return bool(result)


async def ask_folder(
    folders: list[Folder], current_id: str | None
) -> tuple[str | None, str] | None:
    """Pick a folder for a chat, or name a new one.

    Returns ``(folder_id, new_name)``: ``new_name`` is non-empty if the user typed a new folder name (then
    ``folder_id`` should be ignored); otherwise ``folder_id`` is the chosen folder, or ``None`` for "no
    folder". Returns ``None`` if the user cancels.
    """
    options: dict[str, str] = {"": "(no folder)", **{folder.id: folder.name for folder in folders}}
    with _on_page(), ui.dialog() as dialog, ui.card().classes("w-96"):
        ui.label("Move to folder").classes("text-h6")
        chosen = ui.select(options, value=current_id or "", label="Folder").classes("w-full")
        new_name = ui.input("...or a new folder").props("outlined").classes("w-full")
        new_name.mark("dialog-field")
        with ui.row().classes("w-full justify-end"):
            ui.button("Cancel", on_click=lambda: dialog.submit(None)).props("flat").mark(
                "dialog-cancel"
            )
            ui.button(
                "Move",
                on_click=lambda: dialog.submit((chosen.value or None, new_name.value.strip())),
            ).mark("dialog-ok")
    result = await dialog
    discard(dialog)
    return result


async def ask_tool_approval(tool: str, server: str | None, arguments: str) -> str:
    """Ask whether the assistant may run a tool. Returns ``"once"``, ``"chat"`` (allow it for the rest of
    this chat) or ``"deny"``.

    The dialog is ``persistent`` (clicking outside or pressing Escape does nothing), so the only ways out are
    the three buttons, and a stray click can never count as a yes. If the task waiting here is cancelled
    (the Stop button), the ``finally`` still removes the dialog.
    """
    where = f" from {server}" if server else ""
    with ui.dialog().props("persistent") as dialog, ui.card().classes("w-[34rem] max-w-full"):
        ui.label("Allow this tool?").classes("text-h6")
        ui.label(f"The assistant wants to run {tool}{where}.")
        ui.label("Arguments").classes("text-caption lr-muted")
        ui.label(arguments).classes("lr-tool-text")
        with ui.row().classes("w-full justify-end q-mt-sm"):
            ui.button("Deny", on_click=lambda: dialog.submit("deny")).props("flat color=negative")
            ui.button("Allow for this chat", on_click=lambda: dialog.submit("chat")).props("flat")
            ui.button("Allow once", on_click=lambda: dialog.submit("once")).props(
                "color=primary"
            ).mark("allow-once")
    try:
        result = await dialog
    finally:
        discard(dialog)
    return result or "deny"


async def ask_image_size(
    prompt: str, model: str, width: int, height: int, maximum: int
) -> tuple[int, int] | None:
    """Ask how big a picture to draw. Returns ``(width, height)``, or ``None`` if the user cancels.

    Drawing is slow and heavy (the image model may push the chat model out of memory), so it is confirmed first, with
    the description and model shown. Sizes are rounded to a multiple of 64 later, in the Images module.
    """
    with _on_page(), ui.dialog() as dialog, ui.card().classes("w-[28rem] max-w-full"):
        ui.label("Draw a picture").classes("text-h6")
        ui.label(f"Model: {model}").classes("text-caption lr-muted")
        ui.label(prompt[:300] + ("..." if len(prompt) > 300 else "")).classes("whitespace-pre-wrap")
        with ui.row().classes("w-full no-wrap gap-2"):
            wide = ui.number("Width", value=width, min=256, max=maximum, step=64, format="%d")
            wide.props("outlined dense").classes("flex-grow").mark("image-width")
            tall = ui.number("Height", value=height, min=256, max=maximum, step=64, format="%d")
            tall.props("outlined dense").classes("flex-grow").mark("image-height")
        ui.label(
            "This can take a few minutes, and Lemonade may unload the chat model to make room."
        ).classes("text-caption lr-muted")
        with ui.row().classes("w-full justify-end"):
            ui.button("Cancel", on_click=lambda: dialog.submit(None)).props("flat").mark(
                "dialog-cancel"
            )
            ui.button(
                "Draw",
                on_click=lambda: dialog.submit(
                    (int(wide.value or width), int(tall.value or height))
                ),
            ).props("color=primary").mark("dialog-ok")
    result = await dialog
    discard(dialog)
    return result
