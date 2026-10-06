"""The Memories section of Settings: review, edit, pin and delete what the assistant remembers about you.

Automatic learning gets things wrong now and then (a half-remembered detail, a fact that is no longer
true), so you need a place to correct it. The list shows two groups, as the assistant uses them:

* **Pinned** facts are always given to the model, in every chat.
* **Other** facts are only given when the message you send is about something related.

Python / NiceGUI ideas used here:

* ``@ui.refreshable`` again: any change calls ``render.refresh()`` and the list redraws from the database.
* ``lambda m=memory: ...`` - the ``m=memory`` default captures the *current* loop value. Without it every
  button would act on the last item of the loop (a classic closure trap).
"""

from __future__ import annotations

from nicegui import ui

from lemonrind.modules.memory import Memory, MemoryModule


def build_memories(module: MemoryModule) -> None:
    """Draw the section into the current container (the Settings dialog puts it in a panel)."""
    repo = module.repo

    async def add(text_input: ui.input, pin: ui.checkbox) -> None:
        saved = await module.save_fact(text_input.value or "", pinned=bool(pin.value))
        if saved is None and (text_input.value or "").strip():
            ui.notify("Something very similar is already remembered.", type="warning")
        text_input.value = ""
        render.refresh()

    async def save_edit(memory: Memory, field: ui.input) -> None:
        await module.edit_fact(memory.id, field.value or "")
        ui.notify("Saved", type="positive", timeout=1000)
        render.refresh()

    def toggle_pin(memory: Memory) -> None:
        repo.set_pinned(memory.id, not memory.pinned)
        render.refresh()

    def delete(memory: Memory) -> None:
        repo.delete(memory.id)
        render.refresh()

    @ui.refreshable
    def render() -> None:
        memories = repo.list_all()
        pinned = [m for m in memories if m.pinned]
        others = [m for m in memories if not m.pinned]
        if not memories:
            ui.label("Nothing is remembered yet. Facts you mention in chat appear here.").classes(
                "text-caption lr-muted"
            )
        for title, group in (
            ("Pinned (always included)", pinned),
            ("Other (used when relevant)", others),
        ):
            if not group:
                continue
            ui.label(title).classes("text-weight-medium q-mt-sm")
            for memory in group:
                with ui.row().classes("w-full items-center no-wrap gap-1"):
                    field = (
                        ui.input(value=memory.content).props("outlined dense").classes("flex-grow")
                    )
                    field.on("keydown.enter", lambda m=memory, f=field: save_edit(m, f))
                    ui.button(
                        icon="check", on_click=lambda m=memory, f=field: save_edit(m, f)
                    ).props("flat dense round").tooltip("Save the edited text")
                    ui.button(
                        icon="push_pin" if memory.pinned else "outlined_flag",
                        on_click=lambda m=memory: toggle_pin(m),
                    ).props(f"flat dense round {'color=primary' if memory.pinned else ''}").tooltip(
                        "Unpin" if memory.pinned else "Pin: always include this fact"
                    )
                    ui.button(icon="delete", on_click=lambda m=memory: delete(m)).props(
                        "flat dense round color=negative"
                    ).tooltip("Forget this")
                if not memory.has_embedding:
                    ui.label(
                        "No embedding yet (no embedding model was available), so it cannot be found by meaning."
                    ).classes("text-caption lr-warn")

    with ui.column().classes("w-full gap-2"):
        ui.label(
            "What the assistant remembers about you, across chats. Edit or delete anything that is wrong."
        ).classes("text-caption lr-muted")
        with ui.row().classes("w-full items-center gap-2 q-mt-sm"):
            new_text = (
                ui.input(placeholder="Add a fact, e.g. I prefer metric units")
                .props("outlined dense")
                .classes("flex-grow")
                .style("min-width: 8rem")
            )
            new_pin = ui.checkbox("Pin")
            add_button = ui.button("Add", on_click=lambda: add(new_text, new_pin)).props(
                "color=primary"
            )
            new_text.on("keydown.enter", lambda: add(new_text, new_pin))
            add_button.mark("memory-add")
        render()
