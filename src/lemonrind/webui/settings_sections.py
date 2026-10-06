"""The form sections of the Settings dialog, one function each.

Every ``build_*`` function draws its section into the current container and returns an **applier**: a function that
reads the widgets and writes the values into the settings. The dialog calls all the appliers when Save is pressed, so
Cancel changes nothing, and a bad value (an unusable folder, say) is reported without closing the dialog.

An applier may return a short note ("Restart the app for the new address to take effect."); the page shows them after
saving. It may raise ``SettingsError`` (or let pydantic's validation error through) to say "do not save".

The sections follow the other editions' Settings: Lemonade, Assistant, Persona, Interface, Storage, Web search, File
system access, Image generation, Memory, Auto-backup, Modules, then the screens for knowledge bases, MCP servers and the
scheduler (those live in their own files, because they act on the database at once).

Python / NiceGUI ideas used here:

* **Returning a closure.** Each builder returns an inner function that remembers the widgets it created. The dialog never
  needs to know what is inside a section.
* ``ui.slider`` with a label that follows its value (``bind_text_from``).
* ``None`` for "not set": a blank number box means "use the model's own default", stored as ``None``.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from nicegui import ui

from lemonrind import config
from lemonrind.config import Settings
from lemonrind.modules import AutoBackupModule, Module, ModuleRegistry
from lemonrind.modules.backup import BackupError
from lemonrind.modules.memory import MemoryModule
from lemonrind.webui.memories_dialog import build_memories

Applier = Callable[[], str | None]

PERSONA_TONES = ["Default", "Casual", "Formal", "Warm", "Concise", "Direct"]
PERSONA_LENGTHS = ["Default", "Concise", "Detailed"]
PERSONA_EMOJI = ["Default", "None", "Sparing", "Frequent"]
THEME_LABELS = {"system": "Follow my system", "light": "Light", "dark": "Dark"}


class SettingsError(ValueError):
    """A value cannot be used. The message is fit to show the user."""


@dataclass(frozen=True, slots=True)
class ModelChoices:
    """The models Lemonade has downloaded, by kind, for the drop-downs."""

    chat: list[str] = field(default_factory=list)
    embedding: list[str] = field(default_factory=list)
    image: list[str] = field(default_factory=list)


def _note(text: str) -> None:
    ui.label(text).classes("text-caption lr-muted")


def _banner_if_off(module: Module | None) -> None:
    """The reminder the other editions show on a section whose module is switched off."""
    if module is not None and not module.enabled:
        ui.label(
            "This module is currently disabled - see Modules to enable it. This section still works either way."
        ).classes("text-caption lr-warn")


def _options(choices: list[str], current: str, blank_label: str) -> dict[str, str]:
    options = {"": blank_label, **{name: name for name in choices}}
    if current and current not in options:  # a model chosen earlier that Lemonade no longer lists
        options[current] = current
    return options


def _optional_int(value: float | None) -> int | None:
    return None if value in (None, "") else int(value)  # type: ignore[arg-type]


# --- Lemonade -------------------------------------------------------------------------------------------------


def build_lemonade(settings: Settings, models: ModelChoices) -> Applier:
    lemonade = settings.lemonade
    base_url = ui.input("Base URL (requires restart)", value=lemonade.base_url).props("outlined")
    base_url.classes("w-full").mark("lemonade-url")
    api_key = ui.input("API key (optional, requires restart)", value=lemonade.api_key)
    api_key.props("outlined type=password").classes("w-full").mark("lemonade-key")
    _note("Lemonade does not ask for a key unless it was started with one.")
    chat_model = (
        ui.select(
            _options(models.chat, lemonade.chat_model, "(whatever Lemonade has loaded)"),
            value=lemonade.chat_model,
            label="Default chat model",
        )
        .classes("w-full")
        .mark("lemonade-chat-model")
    )
    embedding_model = (
        ui.select(
            _options(
                models.embedding,
                lemonade.embedding_model,
                "(the first embedding model Lemonade has)",
            ),
            value=lemonade.embedding_model,
            label="Default embedding model",
        )
        .classes("w-full")
        .mark("lemonade-embedding-model")
    )
    _note(
        "Memories and knowledge bases are stored as numbers made by the embedding model, so after changing it the "
        "ones already saved will match less well. Takes effect after a restart."
    )

    def apply() -> str | None:
        notes = []
        if (
            base_url.value.strip() != lemonade.base_url.strip()
            or (api_key.value or "") != lemonade.api_key
        ):
            notes.append("Restart the app for the new Lemonade address or key to take effect.")
        if (embedding_model.value or "") != lemonade.embedding_model:
            notes.append("Restart the app to use the new embedding model.")
        lemonade.base_url = base_url.value
        lemonade.chat_model = chat_model.value or ""
        lemonade.embedding_model = embedding_model.value or ""
        lemonade.api_key = api_key.value or ""
        return " ".join(notes) or None

    return apply


# --- Assistant, Persona, Interface -------------------------------------------------------------------------


def build_assistant(settings: Settings) -> Applier:
    prompt = ui.textarea("System prompt", value=settings.assistant.system_prompt)
    prompt.props("outlined autogrow").classes("w-full").mark("assistant-prompt")
    _note(
        "The base instructions given to the model at the start of every request. Your persona, pinned memories and the "
        "summary of a long chat are added after it. Use 'View system prompt' in the right-hand panel to see the result."
    )

    def apply() -> str | None:
        settings.assistant.system_prompt = prompt.value
        return None

    return apply


def build_persona(settings: Settings) -> Applier:
    _note(
        "Who the assistant is, who it is talking to and how it writes. Anything left blank or at Default adds "
        "nothing to the prompt."
    )
    persona = settings.persona
    identity = ui.textarea("Identity (who the assistant is)", value=persona.identity)
    identity.props("outlined autogrow").classes("w-full").mark("persona-identity")
    about = ui.textarea("About you (what it should know about you)", value=persona.about_user)
    about.props("outlined autogrow").classes("w-full").mark("persona-about")
    with ui.row().classes("w-full no-wrap gap-2"):
        tone = ui.select(PERSONA_TONES, value=persona.tone, label="Tone").classes("flex-grow")
        length = ui.select(PERSONA_LENGTHS, value=persona.verbosity, label="Length").classes(
            "flex-grow"
        )
        emoji = ui.select(PERSONA_EMOJI, value=persona.emoji_usage, label="Emoji").classes(
            "flex-grow"
        )
    instructions = ui.textarea("Other instructions", value=persona.custom_instructions)
    instructions.props("outlined autogrow").classes("w-full")

    def apply() -> str | None:
        persona.identity = identity.value or ""
        persona.about_user = about.value or ""
        persona.tone = tone.value
        persona.verbosity = length.value
        persona.emoji_usage = emoji.value
        persona.custom_instructions = instructions.value or ""
        return None

    return apply


def build_interface(settings: Settings) -> Applier:
    theme = ui.select(THEME_LABELS, value=settings.ui.theme, label="Theme").classes("w-full")
    theme.mark("settings-theme")
    thinking = ui.switch(
        "Thinking panel open by default", value=settings.ui.thinking_open_by_default
    ).mark("settings-thinking")
    _note("Expand a reply's reasoning panel automatically instead of starting collapsed.")

    def apply() -> str | None:
        settings.ui.theme = theme.value
        settings.ui.thinking_open_by_default = thinking.value
        return None

    return apply


# --- Storage ---------------------------------------------------------------------------------------------------


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("bytes", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{int(value)} {unit}" if unit == "bytes" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"  # unreachable; keeps the type checker content


def folder_size(path: Path) -> int:
    """Total bytes in a file or folder (0 if it does not exist). Unreadable files count as 0."""
    if path.is_file():
        return path.stat().st_size
    total = 0
    for item in path.rglob("*") if path.is_dir() else []:
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def build_storage(data_dir: Path) -> Applier:
    default = config.default_data_dir()
    pointer = config.read_data_location(default)
    overridden = data_dir.resolve() != (pointer or default)

    ui.label("Data folder in use").classes("text-caption lr-muted")
    ui.label(str(data_dir)).classes("lr-tool-text").mark("storage-current")
    _note("It holds the database, settings, generated images, workspace files and logs.")
    if overridden:
        ui.label(
            "This session was started with --data-dir or LEMONRIND_DATA_DIR, which wins over the setting below."
        ).classes("text-caption lr-warn")
    location = ui.input(
        "Use this data folder from the next start (blank = the default)",
        value=str(pointer) if pointer else "",
    ).props("outlined")
    location.classes("w-full").mark("storage-folder")
    _note(
        f"The default is {default}. Changing it does not move anything: copy the contents of the current folder to the "
        "new one first (a backup zip is a handy way), or start fresh there. Requires restart."
    )

    ui.label("What is using the space").classes("text-weight-medium q-mt-sm")
    for label, relative in (
        ("Database", config.DATABASE_FILE_NAME),
        ("Generated images", "images"),
        ("Workspace files", "workspace"),
        ("Logs", "logs"),
    ):
        with ui.row().classes("w-full justify-between"):
            ui.label(label)
            size = folder_size(data_dir / relative)
            if relative == config.DATABASE_FILE_NAME:  # the database also has its -wal side file
                size += folder_size(data_dir / (relative + "-wal"))
            ui.label(human_size(size)).classes("lr-muted")
    with ui.row().classes("w-full justify-between"):
        ui.label("Everything").classes("text-weight-medium")
        ui.label(human_size(folder_size(data_dir))).classes("text-weight-medium")

    def apply() -> str | None:
        text = (location.value or "").strip()
        wanted = Path(os.path.expandvars(text)).expanduser().resolve() if text else None
        if wanted == pointer or (wanted is None and pointer is None):
            return None
        if wanted is not None and wanted != default:
            try:
                wanted.mkdir(parents=True, exist_ok=True)
            except OSError as error:
                raise SettingsError(f"That folder cannot be used: {error}") from error
        config.write_data_location(default, wanted)
        return "Restart the app to use the new data folder (nothing was moved)."

    return apply


# --- Web search, File system access, Image generation ----------------------------------------------------------


ENGINES = {
    "SearXNG": "SearXNG (your own server, no key; pages are fetched directly)",
    "Jina": "Jina (s.jina.ai and r.jina.ai)",
    "Tavily": "Tavily (needs an API key)",
    "Firecrawl": "Firecrawl (needs an API key; reads JavaScript pages and can crawl a site)",
}


def build_web_search(
    settings: Settings, module: Module | None, reader: Module | None = None
) -> Applier:
    _banner_if_off(module)
    if reader is not None and not reader.enabled:
        _note(
            "The Web reader module (read_webpage, crawl_website) uses these settings too and is switched off."
        )
    config_ = settings.modules.web_search
    engine = ui.select(ENGINES, value=config_.engine, label="Search and page-reading service")
    engine.classes("w-full").mark("websearch-engine")
    _note(
        "One choice serves both the web_search tool and the Web reader's read_webpage tool. Swapping the "
        "service never changes what the assistant sees, only who answers."
    )
    url = (
        ui.input("SearXNG base URL", value=config_.searxng_url).props("outlined").classes("w-full")
    )
    url.mark("websearch-url")
    _note("Needs the JSON output format switched on (search.formats in its settings.yml).")
    results = (
        ui.number(
            "Results per search (SearXNG)", value=config_.max_results, min=1, max=20, precision=0
        )
        .props("outlined dense")
        .classes("w-full")
        .mark("websearch-results")
    )
    jina = ui.input(
        "Jina API key (optional: reading works without one at a lower rate, searching needs one)",
        value=config_.jina_api_key,
    )
    jina.props("outlined type=password").classes("w-full").mark("websearch-jina-key")
    tavily = ui.input("Tavily API key", value=config_.tavily_api_key)
    tavily.props("outlined type=password").classes("w-full").mark("websearch-tavily-key")
    firecrawl = ui.input(
        "Firecrawl API key (also switches on the crawl_website tool)",
        value=config_.firecrawl_api_key,
    )
    firecrawl.props("outlined type=password").classes("w-full").mark("websearch-firecrawl-key")
    _note(
        "Keys are stored in settings.json in the data folder and sent only to that service. Content from any "
        "service is treated as untrusted: invisible and look-alike characters are stripped before the model reads it."
    )

    def apply() -> str | None:
        config_.engine = engine.value
        config_.searxng_url = url.value or ""
        config_.max_results = int(results.value or config_.max_results)
        config_.jina_api_key = (jina.value or "").strip()
        config_.tavily_api_key = (tavily.value or "").strip()
        config_.firecrawl_api_key = (firecrawl.value or "").strip()
        return None

    return apply


def build_filesystem(settings: Settings, module: Module | None) -> Applier:
    _banner_if_off(module)
    config_ = settings.modules.filesystem
    root = ui.input("Workspace folder", value=config_.root).props("outlined").classes("w-full")
    root.mark("filesystem-root")
    _note(
        "The folder the assistant works in by default: a relative path means a path inside it (and with the Coder it "
        "can edit it as a project). Empty = a 'workspace' folder inside the data folder."
    )
    extra = ui.textarea(
        "More folders the assistant may use (full paths, one per line)",
        value="\n".join(config_.allowed_roots),
    ).props("outlined autogrow")
    extra.classes("w-full").mark("filesystem-extra")
    _note(
        "Beyond the workspace, the assistant can only reach these folders (and everything inside them), by giving a full "
        "path. Anything else is refused. Searching with the Coder tools covers the workspace only."
    )
    free = ui.textarea(
        "Folders where writing needs no permission (one per line)",
        value="\n".join(config_.preapproved_folders),
    ).props("outlined autogrow")
    free.classes("w-full").mark("filesystem-free")
    _note(
        "Paths are relative to the workspace (or full paths inside an allowed folder); a single dot means the whole "
        "workspace. Everything else asks first, showing what would change. Use with care: a model can be tricked into writing."
    )

    def apply() -> str | None:
        extra_roots = [line.strip() for line in (extra.value or "").splitlines() if line.strip()]
        for entry in extra_roots:
            if not Path(entry).expanduser().is_absolute():
                raise SettingsError(
                    f"'{entry}' is not a full path. Give each extra folder as a full path."
                )
        config_.root = (root.value or "").strip()
        config_.allowed_roots = extra_roots
        config_.preapproved_folders = [
            line.strip() for line in (free.value or "").splitlines() if line.strip()
        ]
        return None

    return apply


def build_images(settings: Settings, models: ModelChoices, module: Module | None) -> Applier:
    _banner_if_off(module)
    config_ = settings.modules.images
    model = ui.select(
        _options(models.image, config_.model, "(the smallest image model Lemonade has downloaded)"),
        value=config_.model,
        label="Image model",
    ).classes("w-full")
    model.mark("images-model")
    with ui.row().classes("w-full no-wrap gap-2"):
        width = (
            ui.number(
                "Default width",
                value=config_.width,
                min=256,
                max=config_.max_side,
                step=64,
                precision=0,
            )
            .classes("flex-grow")
            .mark("images-width")
        )
        height = ui.number(
            "Default height",
            value=config_.height,
            min=256,
            max=config_.max_side,
            step=64,
            precision=0,
        ).classes("flex-grow")
    _note(
        "Bigger pictures take much longer and use more memory. Blank boxes below mean the model's own good defaults."
    )
    with ui.row().classes("w-full no-wrap gap-2"):
        steps = ui.number("Steps", value=config_.steps, min=1, max=200, precision=0).props(
            "clearable"
        )
        cfg = ui.number("CFG scale", value=config_.cfg_scale, min=0, max=30, step=0.5).props(
            "clearable"
        )
        seed = ui.number("Seed (-1 = random)", value=config_.seed, precision=0).props("clearable")
        for box, marker in ((steps, "images-steps"), (cfg, "images-cfg"), (seed, "images-seed")):
            box.classes("flex-grow").mark(marker)

    def apply() -> str | None:
        config_.model = model.value or ""
        config_.width = int(width.value or config_.width)
        config_.height = int(height.value or config_.height)
        config_.steps = _optional_int(steps.value)
        config_.cfg_scale = None if cfg.value in (None, "") else float(cfg.value)
        chosen_seed = _optional_int(seed.value)
        config_.seed = -1 if chosen_seed is None else chosen_seed  # blank means random too
        return None

    return apply


# --- Memory ----------------------------------------------------------------------------------------------------


def build_memory(settings: Settings, module: MemoryModule | None) -> Applier:
    _banner_if_off(module)
    ui.label("Conversation compaction").classes("text-weight-medium")
    slider = ui.slider(min=0, max=95, step=5, value=settings.assistant.compaction_percent)
    slider.props("label-always").mark("memory-compaction")
    summary = ui.label().classes("text-caption lr-muted")
    summary.bind_text_from(
        slider,
        "value",
        lambda v: (
            f"Summarise the older messages when the chat fills {int(v)}% of the model's memory."
            if v
            else "Never summarise: a chat that outgrows the model's memory will fail."
        ),
    )
    learn = ui.switch("Learn facts from my messages", value=settings.modules.memory.extract_facts)
    learn.mark("memory-learn")
    _note(
        "After each reply the assistant picks out lasting facts about you. Needs an embedding model in Lemonade to find them again."
    )

    if module is not None:
        ui.separator()
        _note("The settings above need Save. The list below acts at once.")
        build_memories(module)

    def apply() -> str | None:
        settings.assistant.compaction_percent = int(slider.value)
        settings.modules.memory.extract_facts = bool(learn.value)
        return None

    return apply


# --- Auto-backup -----------------------------------------------------------------------------------------------


def build_backup(settings: Settings, module: AutoBackupModule | None) -> Applier:
    config_ = settings.modules.backup
    _banner_if_off(module)
    _note(
        "Periodically zips your whole data folder (database, generated images, workspace files, settings) to a separate "
        "folder as a safety net, with no AI involved. Checked a few times a day; a new backup is made once the interval "
        "below has passed. The backup folder is kept outside the data folder on purpose: storing backups inside the "
        "folder being backed up would make each new backup include every previous one."
    )
    with ui.row().classes("w-full no-wrap gap-2"):
        interval = (
            ui.number(
                "Back up every N days", value=config_.interval_days, min=1, max=365, precision=0
            )
            .classes("flex-grow")
            .mark("backup-interval")
        )
        keep = ui.number(
            "Keep this many zips", value=config_.keep_count, min=1, max=100, precision=0
        ).classes("flex-grow")
    folder = (
        ui.input("Backup folder", value=config_.backup_folder).props("outlined").classes("w-full")
    )
    folder.mark("backup-folder")
    _note(
        "A relative folder is placed beside the data folder. Environment variables and ~ are expanded."
    )

    status = ui.label().classes("text-caption").mark("backup-status")

    def show_status() -> None:
        if module is None:
            status.text = ""
            return
        info = module.status()
        when = info.last.astimezone().strftime("%a %d %b %Y %H:%M") if info.last else "never"
        status.text = f"Last backup: {when}. {info.count} zip(s) kept in {info.folder}"

    async def back_up_now() -> None:
        if module is None:
            return
        try:
            result = await module.backup_now()
        except BackupError as error:
            ui.notify(str(error), type="negative", multi_line=True)
            return
        ui.notify(f"Backed up {result.files} files to {result.path.name}", type="positive")
        show_status()

    show_status()
    ui.button("Back up now", icon="backup", on_click=back_up_now).props("outline").mark(
        "backup-now"
    )
    _note(
        "Uses the folder and keep count already saved. Press Save first if you just changed them."
    )

    def apply() -> str | None:
        config_.interval_days = int(interval.value or config_.interval_days)
        config_.keep_count = int(keep.value or config_.keep_count)
        config_.backup_folder = (folder.value or "").strip() or "data_backups"
        return None

    return apply


# --- Modules ---------------------------------------------------------------------------------------------------


def build_modules(settings: Settings, registry: ModuleRegistry | None) -> Applier:
    if registry is None:
        _note("Modules are not available.")
        return lambda: None
    _note(
        "Each module gives the assistant some tools (or, for Auto-backup, does housekeeping). Switch them on or off here and "
        "press Save; the change applies straight away. Each module's own options are in its section."
    )
    switches: dict[str, ui.switch] = {}
    for module in registry.modules:
        with ui.column().classes("w-full gap-0"):
            switches[module.config_key] = ui.switch(module.name, value=module.enabled)
            tool_names = ", ".join(tool.name for tool in module.get_tools())
            tools_text = f" Tools: {tool_names}." if tool_names else ""
            ui.label(f"{module.description}{tools_text}").classes("text-caption lr-muted q-ml-xl")
    rounds = (
        ui.number(
            "Most tool rounds for one message",
            value=settings.modules.max_tool_rounds,
            min=1,
            max=100,
            precision=0,
        )
        .props("outlined dense")
        .classes("w-full q-mt-sm")
    )

    def apply() -> str | None:
        for key, switch in switches.items():
            settings.modules.enabled[key] = bool(switch.value)
        if rounds.value:
            settings.modules.max_tool_rounds = max(1, min(100, int(rounds.value)))
        return None

    return apply
