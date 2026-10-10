"""Settings: what the app needs to know to run, stored as JSON in the data folder.

Python ideas used here:

* ``pydantic.BaseModel`` - a class whose fields have types, defaults and validation. It is the Python
  counterpart of a C# settings class bound from configuration, except it also checks the values for you
  and can turn itself into JSON and back.
* ``field_validator`` - a method that runs when a field is set, to check or tidy the value.
* ``pathlib.Path`` - paths as objects (``folder / "file.json"``) instead of string concatenation.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Environment variable that overrides the default data folder (handy for tests and for running two
# copies side by side).
DATA_DIR_ENV = "LEMONRIND_DATA_DIR"
SETTINGS_FILE_NAME = "settings.json"
DATABASE_FILE_NAME = "lemonrind.db"
# A one-line file in the *default* data folder that says "the real data folder is over there" (set from Settings >
# Storage). It has to live in a place that is known before the settings file can be read, which is why the data
# folder is not simply a field inside settings.json.
LOCATION_FILE_NAME = "data_location.txt"


class LemonadeSettings(BaseModel):
    """How to reach Lemonade Server and which models to use."""

    # By default pydantic checks values only when an object is created. This makes it check (and run
    # the validators below) on every assignment too, e.g. ``settings.base_url = args.base_url``.
    model_config = ConfigDict(validate_assignment=True)

    base_url: str = "http://localhost:13305/v1/"
    api_key: str = ""  # Lemonade does not need one unless it was started with an API key
    chat_model: str = ""  # empty = use whatever Lemonade already has loaded
    embedding_model: str = ""

    # One reply from a local model can legitimately take many minutes: a big prompt needs time before
    # the first token and a long reply at ~60 tokens/second adds minutes more. A 5 minute limit cut off
    # a healthy 6 minute reply in the .NET editions, so the default here is 30 minutes.
    request_timeout_seconds: float = 1800.0

    @field_validator("base_url")
    @classmethod
    def _ensure_trailing_slash(cls, value: str) -> str:
        # A base URL without a trailing slash silently breaks every relative request built from it
        # ("http://host/v1" + "models" becomes "http://host/models", dropping "/v1"). Fixing it once
        # here means no other code has to remember. The .NET editions needed the same fix.
        value = value.strip()
        return value if value.endswith("/") else value + "/"


class AssistantSettings(BaseModel):
    """How the assistant behaves."""

    model_config = ConfigDict(validate_assignment=True)

    system_prompt: str = "You are a helpful local AI assistant running on the user's own machine."
    # When the conversation fills this percentage of the model's context window, the oldest messages are
    # summarised to make room. 0 turns it off.
    compaction_percent: int = Field(default=75, ge=0, le=95)
    # How many of the most recent exchanges (a message of yours and everything the assistant did in reply)
    # are always kept word for word.
    compaction_keep_turns: int = Field(default=3, ge=1, le=20)


class PersonaSettings(BaseModel):
    """Who the assistant is, who it is talking to, and how it writes. See ``chats/persona.py``."""

    model_config = ConfigDict(validate_assignment=True)

    identity: str = ""  # e.g. "You are Max, a dry-witted assistant."
    about_user: str = ""  # e.g. "I am Darren, a developer in Manchester."
    # The choices are fixed lists so a typo in settings.json is caught when the file is read.
    tone: Literal["Default", "Casual", "Formal", "Warm", "Concise", "Direct"] = "Default"
    verbosity: Literal["Default", "Concise", "Detailed"] = "Default"
    emoji_usage: Literal["Default", "None", "Sparing", "Frequent"] = "Default"
    custom_instructions: str = ""


class UiSettings(BaseModel):
    """How the web UI looks and behaves."""

    model_config = ConfigDict(validate_assignment=True)

    # "system" follows the operating system's light/dark setting.
    theme: Literal["system", "light", "dark"] = "system"
    thinking_open_by_default: bool = False  # expand a reply's thinking panel automatically
    # Widths in pixels of the chat list (left) and the stats panel (right): drag the edge of a panel to change them.
    left_panel_width: int = Field(default=300, ge=200, le=560)
    right_panel_width: int = Field(default=280, ge=220, le=520)


class WebSearchSettings(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    # A self-hosted SearXNG (a search engine that gathers results from many others). It must have the
    # JSON output format switched on (``search.formats`` in its settings.yml).
    searxng_url: str = "http://localhost:8888/"
    max_results: int = Field(default=5, ge=1, le=20)
    # Which service answers web_search and read_webpage. SearXNG is the default and reads pages by fetching them directly.
    engine: Literal["SearXNG", "Jina", "Tavily", "Firecrawl"] = "SearXNG"
    # Optional for Jina (reading works without one at a lower rate; searching needs one). Required for Tavily and Firecrawl.
    jina_api_key: str = ""
    tavily_api_key: str = ""
    firecrawl_api_key: str = ""  # also switches on the crawl_website tool


class FileSystemSettings(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    # The one folder the model may read and write. Empty = a "workspace" folder inside the data folder.
    root: str = ""
    # More folders the assistant may use, besides the workspace: full paths. Relative paths always mean the workspace;
    # a full path is allowed only if it lies inside the workspace or one of these.
    allowed_roots: list[str] = Field(default_factory=list)
    max_read_chars: int = Field(default=50_000, ge=1_000)
    # Writing a file asks your permission, except inside these folders (paths relative to the workspace;
    # "." means the whole workspace). Use with care: a model can be tricked into writing.
    preapproved_folders: list[str] = Field(default_factory=list)
    max_write_chars: int = Field(default=200_000, ge=1_000)  # the largest file the model may write


class MemorySettings(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    # After each reply, ask the model to pick out lasting facts about the user and remember them.
    extract_facts: bool = True
    # How similar (0 to 1) a stored fact must be to the message to count as relevant. Measured with Qwen3-
    # Embedding-0.6B: questions about a fact score 0.5 to 0.7, unrelated messages stay below 0.47. Other
    # embedding models spread their scores differently; raise it if unrelated facts keep appearing.
    min_similarity: float = Field(default=0.5, ge=0.0, le=1.0)
    max_relevant: int = Field(default=3, ge=1, le=20)
    # Above this a new fact is "the same thing said again" and is not saved twice.
    duplicate_similarity: float = Field(default=0.85, ge=0.0, le=1.0)


class KnowledgeSettings(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    # Documents are cut into chunks of about this many words, each repeating the last ``chunk_overlap``
    # words of the one before so that an answer near a cut still appears whole in one chunk.
    chunk_words: int = Field(default=400, ge=50, le=5000)
    chunk_overlap: int = Field(default=60, ge=0, le=1000)
    # How many chunks at most are added to a message, and how similar (0 to 1) each must be. Document
    # chunks are longer and more varied than a memory sentence, so their scores run lower than memory's.
    max_chunks: int = Field(default=4, ge=1, le=20)
    min_similarity: float = Field(default=0.35, ge=0.0, le=1.0)


class McpSettings(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    # How long to wait for a server to start and say hello. Generous: the first run of an npx server has to
    # download it, which can take a minute on a slow connection.
    connect_timeout_seconds: float = Field(default=90.0, ge=5.0)
    # How long one tool call may take before it is given up on.
    call_timeout_seconds: float = Field(default=120.0, ge=5.0)


class SchedulerSettings(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    # One job may run this long in total before it is stopped (a model can get stuck reasoning in a loop).
    job_timeout_seconds: float = Field(default=1800.0, ge=1.0)
    # How many request/tool/request cycles one run may use. A research job that searches and reads many pages
    # needs far more than a chat message does, and nobody is there to say "keep going".
    max_tool_rounds: int = Field(default=100, ge=1, le=500)
    # Cap on one reply's length (tokens, and a model's thinking counts), so a runaway generation cannot eat the
    # whole time limit. A model that thinks at length and then writes a long report needs room for both.
    max_output_tokens: int = Field(default=32768, ge=256)
    # When a reply stops at that limit with the report part written, the run asks the model to carry on from where it
    # stopped, up to this many times (0 = never; the run is then marked "cut short").
    max_continuations: int = Field(default=2, ge=0, le=10)
    # A job that was due while the app was closed runs once when the app starts, but only if it was missed by
    # less than this (APScheduler's misfire grace time; never less than a minute). A "Monday 9am" job should
    # not fire on Thursday evening just because the computer was off.
    missed_grace_minutes: int = Field(default=360, ge=0)


class BackupSettings(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    # A new backup is made once this many days have passed since the last one.
    interval_days: int = Field(default=7, ge=1, le=365)
    # Where the zips go. A relative folder is placed *beside* the data folder (never inside it, or every backup
    # would contain all the earlier ones). Environment variables and ~ are expanded.
    backup_folder: str = "data_backups"
    # The newest this many zips are kept; older ones are deleted after each backup.
    keep_count: int = Field(default=5, ge=1, le=100)


class ImageSettings(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    # Which image model to use. Empty = the smallest image model Lemonade has downloaded.
    model: str = ""
    # The size used when the assistant does not ask for one. Bigger takes much longer and uses more memory.
    width: int = Field(default=512, ge=256)
    height: int = Field(default=512, ge=256)
    max_side: int = Field(
        default=1536, ge=256
    )  # no side may exceed this, whatever the model asks for
    # Good starting values for the small, fast models. Clear a box (None) to leave that setting to the model.
    steps: int | None = Field(default=4, ge=1, le=200)
    cfg_scale: float | None = Field(default=1.0, ge=0.0, le=30.0)
    # -1 means "random" (the same as in the other editions); a fixed number makes the same prompt give the same picture.
    seed: int | None = -1


class ModulesSettings(BaseModel):
    """Which capability modules are on, and the limits that apply to tool use."""

    model_config = ConfigDict(validate_assignment=True)

    # config_key -> on/off. A module that is not listed uses its own default (see Module.enabled_by_default).
    enabled: dict[str, bool] = Field(default_factory=dict)
    # How many request/tool/request cycles one message may use before the model must answer.
    max_tool_rounds: int = Field(default=10, ge=1, le=100)
    # The longest piece of one tool result the model is given at once (about four characters to a token). A longer
    # result is held back and read on in pieces of this size with the built-in read_more tool.
    max_tool_output_chars: int = Field(default=40_000, ge=500, le=1_000_000)
    web_search: WebSearchSettings = Field(default_factory=WebSearchSettings)
    filesystem: FileSystemSettings = Field(default_factory=FileSystemSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    knowledge: KnowledgeSettings = Field(default_factory=KnowledgeSettings)
    mcp: McpSettings = Field(default_factory=McpSettings)
    scheduler: SchedulerSettings = Field(default_factory=SchedulerSettings)
    images: ImageSettings = Field(default_factory=ImageSettings)
    backup: BackupSettings = Field(default_factory=BackupSettings)


class Settings(BaseModel):
    """All settings. ``Field(default_factory=...)`` builds a fresh nested object for each instance."""

    lemonade: LemonadeSettings = Field(default_factory=LemonadeSettings)
    assistant: AssistantSettings = Field(default_factory=AssistantSettings)
    persona: PersonaSettings = Field(default_factory=PersonaSettings)
    ui: UiSettings = Field(default_factory=UiSettings)
    modules: ModulesSettings = Field(default_factory=ModulesSettings)

    @classmethod
    def load(cls, path: Path) -> Settings:
        """Read settings from ``path``; if the file does not exist yet, return the defaults."""
        if not path.exists():
            return cls()
        # model_validate_json parses AND validates, so a hand-edited file with a wrong type fails with
        # a clear message instead of misbehaving later. Fields missing from the file keep defaults.
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")


def _find_source_checkout() -> Path | None:
    """If this code is running from a source checkout, return the project folder, else ``None``.

    This file is ``<project>/src/lemonrind/config.py`` in a checkout, so the project folder is two levels
    up, and it is recognisable by the ``pyproject.toml`` next to ``src``. In an installed copy that file
    is not there. ``__file__`` is the path of the current source file.
    """
    project = Path(__file__).resolve().parents[2]
    return project if (project / "pyproject.toml").is_file() else None


_SOURCE_CHECKOUT = _find_source_checkout()


def user_data_dir() -> Path:
    """The usual per-user place for an installed program's data, which differs by operating system.

    Windows: ``%LOCALAPPDATA%\\LemonRind`` (the *Local* one, because the data can be large and should not roam with a
    network profile). macOS: ``~/Library/Application Support/LemonRind``. Linux and the rest: ``$XDG_DATA_HOME/lemonrind``,
    which is ``~/.local/share/lemonrind`` unless that variable says otherwise.
    """
    home = Path.home()
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA") or home / "AppData" / "Local") / "LemonRind"
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / "LemonRind"
    return Path(os.environ.get("XDG_DATA_HOME") or home / ".local" / "share") / "lemonrind"


def default_data_dir(*, source_checkout: Path | None = _SOURCE_CHECKOUT) -> Path:
    """The data folder used when nothing says otherwise.

    In a source checkout it is ``<project>/data``. An installed copy (``pipx install`` or ``uv tool install``) has no
    project folder, and a folder relative to wherever it was started from would give a different set of chats for
    every starting place, so it uses a fixed per-user folder instead (``user_data_dir``).
    """
    base = source_checkout / "data" if source_checkout is not None else user_data_dir()
    return base.expanduser().resolve()


def read_data_location(default: Path) -> Path | None:
    """The folder named in the default folder's ``data_location.txt``, or ``None`` if there is none (or it is blank)."""
    pointer = default / LOCATION_FILE_NAME
    try:
        text = pointer.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return Path(os.path.expandvars(text)).expanduser().resolve() if text else None


def write_data_location(default: Path, chosen: Path | None) -> None:
    """Remember ``chosen`` as the data folder from the next start; ``None`` (or the default itself) removes the pointer."""
    pointer = default / LOCATION_FILE_NAME
    if chosen is None or chosen.expanduser().resolve() == default:
        pointer.unlink(missing_ok=True)
        return
    default.mkdir(parents=True, exist_ok=True)
    pointer.write_text(str(chosen.expanduser().resolve()) + "\n", encoding="utf-8")


def resolve_data_dir(
    explicit: str | Path | None = None, *, source_checkout: Path | None = _SOURCE_CHECKOUT
) -> Path:
    """Pick the data folder. The first that applies wins:

    1. an explicit value (the ``--data-dir`` option),
    2. the ``LEMONRIND_DATA_DIR`` environment variable,
    3. the folder chosen in Settings > Storage (a ``data_location.txt`` in the default folder),
    4. ``<project>/data`` when running from a source checkout, so the same settings are used no matter
       which folder you start the app from,
    5. otherwise (an installed copy) a fixed per-user folder: see ``user_data_dir``.

    The whole app lives in one portable folder (database, settings, generated files), as in the other
    editions: copy it somewhere else and everything travels with it.
    """
    if explicit:
        return Path(explicit).expanduser().resolve()
    if env := os.environ.get(DATA_DIR_ENV):
        return Path(env).expanduser().resolve()
    default = default_data_dir(source_checkout=source_checkout)
    return read_data_location(default) or default


def settings_path(data_dir: Path) -> Path:
    return data_dir / SETTINGS_FILE_NAME


def database_path(data_dir: Path) -> Path:
    return data_dir / DATABASE_FILE_NAME
