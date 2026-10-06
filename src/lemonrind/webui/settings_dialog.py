"""The Settings dialog: one place for every setting, with a navigation list down the left.

The sections follow the other editions' Settings screen, in the same order: Lemonade, Assistant, Persona, Interface,
Storage, Web search, File system access, Image generation, Memory, Auto-backup, Modules, Knowledge bases, MCP servers,
Scheduler.

Two kinds of section:

* **Forms** are read only when you press Save, so Cancel changes nothing. Each form is drawn by a ``build_*`` function in
  ``settings_sections.py`` that returns an *applier*; Save runs every applier, and if any value is unusable the dialog
  stays open and says why. Settings that the running app cannot apply live (the Lemonade address, the data folder) are
  saved and the user is told a restart is needed.
* **Live sections** (Knowledge bases, MCP servers, Scheduler) are the screens for those modules. They act on the database
  at once (add a source, switch a job off), so they have no Save: the button changes to Close while one is showing.

(The Memory section is both: its settings above the list need Save, the list of remembered facts below it acts at once.)

Python / NiceGUI ideas used here:

* ``ui.tab_panels`` driven by our own list of buttons instead of ``ui.tabs``: the panel is chosen with
  ``panels.set_value(name)`` in an ordinary click handler.
* A table of data (``SECTIONS``) that the navigation, the panels and the footer all read.
* A frozen ``dataclass`` (``SettingsOutcome``) as the return value, because the dialog has several things to report.
* Undoing a half-finished save by restoring a snapshot (``model_dump`` then ``model_validate``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from nicegui import ui
from pydantic import ValidationError

from lemonrind.chats import Conversation
from lemonrind.config import Settings
from lemonrind.modules import AutoBackupModule, McpModule, MemoryModule, Module
from lemonrind.modules.knowledge import KnowledgeModule
from lemonrind.modules.scheduler import SchedulerModule
from lemonrind.webui.context import AppContext
from lemonrind.webui.dialogs import discard
from lemonrind.webui.knowledge_dialog import build_knowledge
from lemonrind.webui.mcp_dialog import build_mcp
from lemonrind.webui.scheduler_dialog import build_scheduler
from lemonrind.webui.settings_sections import (
    Applier,
    ModelChoices,
    SettingsError,
    build_assistant,
    build_backup,
    build_filesystem,
    build_images,
    build_interface,
    build_lemonade,
    build_memory,
    build_modules,
    build_persona,
    build_storage,
    build_web_search,
)


@dataclass(frozen=True, slots=True)
class Section:
    key: str
    label: str
    icon: str
    live: bool = False  # True: acts at once, so no Save button


SECTIONS = [
    Section("lemonade", "Lemonade", "bolt"),
    Section("assistant", "Assistant", "smart_toy"),
    Section("persona", "Persona", "face"),
    Section("interface", "Interface", "palette"),
    Section("storage", "Storage", "storage"),
    Section("web_search", "Web search", "travel_explore"),
    Section("filesystem", "File system access", "folder_open"),
    Section("images", "Image generation", "image"),
    Section("memory", "Memory", "psychology"),
    Section("backup", "Auto-backup", "backup"),
    Section("modules", "Modules", "extension"),
    Section("knowledge", "Knowledge bases", "library_books", live=True),
    Section("mcp", "MCP servers", "hub", live=True),
    Section("scheduler", "Scheduler", "schedule", live=True),
]


@dataclass(frozen=True, slots=True)
class SettingsOutcome:
    saved: bool  # the forms were saved, so the page should re-apply the settings
    open_chat: str | None = None  # the user asked to jump to this chat (a scheduled run's result)
    notes: tuple[str, ...] = ()  # things to tell the user, such as "restart to use the new folder"


def _module(context: AppContext, key: str) -> Module | None:
    modules = context.modules
    return next((m for m in modules.modules if m.config_key == key), None) if modules else None


def _restore(settings: Settings, snapshot: dict) -> None:
    """Put every setting back as the snapshot had it (the same Settings object, so everything holding it sees this)."""
    restored = Settings.model_validate(snapshot)
    for name in Settings.model_fields:
        setattr(settings, name, getattr(restored, name))


async def open_settings(
    context: AppContext,
    models: ModelChoices,
    conversation: Conversation,
    *,
    initial: str = "lemonade",
) -> SettingsOutcome:
    """Show the dialog on the ``initial`` section and wait until it is closed."""
    settings = context.settings
    chat_to_open: str | None = None
    notes: list[str] = []
    appliers: dict[str, Applier] = {}

    def open_chat(session_id: str) -> None:
        nonlocal chat_to_open
        chat_to_open = session_id
        dialog.submit(False)

    def live_screen(key: str, kind: type[Module], build: Callable[[Module], None]) -> None:
        """A module's own screen, or a note when the module is switched off."""
        target = _module(context, key)
        if not isinstance(target, kind) or not target.enabled:
            name = target.name if target else key
            ui.label(
                f"The {name} module is switched off. Switch it on under Modules and press Save."
            ).classes("text-caption lr-muted")
            return
        build(target)

    sections = SECTIONS if context.modules is not None else [s for s in SECTIONS if not s.live]
    if initial not in {s.key for s in sections}:
        initial = "lemonade"

    async def save() -> None:
        snapshot = settings.model_dump()
        collected: list[str] = []
        try:
            for applier in appliers.values():
                if note := applier():
                    collected.append(note)
        except ValidationError as error:
            _restore(settings, snapshot)
            problem = error.errors()[0]
            ui.notify(f"{'.'.join(map(str, problem['loc']))}: {problem['msg']}", type="negative")
            return
        except SettingsError as error:
            _restore(settings, snapshot)
            ui.notify(str(error), type="negative", multi_line=True)
            return
        notes.extend(collected)
        dialog.submit(True)

    with ui.dialog() as dialog, ui.card() as card:
        card.classes("w-[72rem] max-h-[92vh] h-[46rem] gap-0 no-wrap q-pa-none")
        card.style("max-width: 96vw")  # Quasar caps dialogs at 560px unless told otherwise
        ui.label("Settings").classes("text-h6 q-px-md q-pt-sm q-pb-xs")
        with (
            ui.row().classes("w-full no-wrap gap-0 flex-grow items-stretch").style("min-height: 0")
        ):
            # --- navigation (icons only on a phone) ---------------------------------------------------
            nav_items: dict[str, ui.item] = {}
            with ui.column().classes("gap-0 q-pa-xs shrink-0 overflow-auto lr-settings-nav"):
                for section in sections:
                    with ui.item(on_click=lambda k=section.key: select(k)).props(
                        "clickable dense"
                    ) as item:
                        with ui.item_section().props("avatar").classes("min-w-0"):
                            ui.icon(section.icon)
                        with ui.item_section().classes("gt-xs"):
                            ui.item_label(section.label)
                    with item:  # the words show beside the icon, so a tooltip only helps where they are hidden (a phone)
                        ui.tooltip(section.label).classes("lt-sm")
                    item.mark(f"nav-{section.key}")
                    nav_items[section.key] = item

            # --- panels ----------------------------------------------------------------------------------
            with ui.column().classes("flex-grow overflow-auto q-pa-md gap-2").style("min-width: 0"):
                panels = ui.tab_panels(value=initial).props("animated=false").classes("w-full")
                with panels:
                    for section in sections:
                        with ui.tab_panel(section.key).classes("q-pa-none gap-2"):
                            ui.label(section.label).classes("text-h6")
                            match section.key:
                                case "lemonade":
                                    appliers[section.key] = build_lemonade(settings, models)
                                case "assistant":
                                    appliers[section.key] = build_assistant(settings)
                                case "persona":
                                    appliers[section.key] = build_persona(settings)
                                case "interface":
                                    appliers[section.key] = build_interface(settings)
                                case "storage":
                                    appliers[section.key] = build_storage(
                                        context.settings_file.parent
                                    )
                                case "web_search":
                                    appliers[section.key] = build_web_search(
                                        settings,
                                        _module(context, "web_search"),
                                        _module(context, "web_reader"),
                                    )
                                case "filesystem":
                                    appliers[section.key] = build_filesystem(
                                        settings, _module(context, "filesystem")
                                    )
                                case "images":
                                    appliers[section.key] = build_images(
                                        settings, models, _module(context, "images")
                                    )
                                case "memory":
                                    memory = _module(context, "memory")
                                    appliers[section.key] = build_memory(
                                        settings,
                                        memory if isinstance(memory, MemoryModule) else None,
                                    )
                                case "backup":
                                    backup = _module(context, "backup")
                                    appliers[section.key] = build_backup(
                                        settings,
                                        backup if isinstance(backup, AutoBackupModule) else None,
                                    )
                                case "modules":
                                    appliers[section.key] = build_modules(settings, context.modules)
                                case "knowledge":
                                    live_screen(
                                        "knowledge",
                                        KnowledgeModule,
                                        lambda m: build_knowledge(m, conversation),  # type: ignore[arg-type]
                                    )
                                case "mcp":
                                    live_screen("mcp", McpModule, lambda m: build_mcp(m))  # type: ignore[arg-type]
                                case "scheduler":
                                    live_screen(
                                        "scheduler",
                                        SchedulerModule,
                                        lambda m: build_scheduler(m, models.chat, open_chat),  # type: ignore[arg-type]
                                    )

        ui.separator()
        with ui.row().classes("w-full justify-end q-pa-sm gap-2"):
            cancel_button = ui.button("Cancel", on_click=lambda: dialog.submit(False)).props("flat")
            cancel_button.mark("settings-cancel")
            save_button = ui.button("Save", on_click=save).props("color=primary")
            save_button.mark("settings-save")

    def select(key: str) -> None:
        """Show one section; live sections have nothing to save, so the buttons change to a single Close."""
        panels.set_value(key)
        for name, item in nav_items.items():
            item.props(add="active" if name == key else "", remove="" if name == key else "active")
        live = next(s.live for s in sections if s.key == key)
        save_button.set_visibility(not live)
        cancel_button.set_text("Close" if live else "Cancel")

    select(initial)
    saved = await dialog
    if saved:
        context.save_settings()
    discard(dialog)
    return SettingsOutcome(bool(saved), chat_to_open, tuple(notes) if saved else ())
