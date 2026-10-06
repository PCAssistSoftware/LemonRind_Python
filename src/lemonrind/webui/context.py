"""The things every browser tab shares: settings, database, Lemonade client.

NiceGUI builds a fresh page (and a fresh ``ChatPage`` object) for each browser tab, but there is only one
database and one Lemonade client for the whole process. Those live here, created once at start-up.

Python ideas used here:

* A module-level variable as a simple "singleton": the module is imported once, so the variable exists once.
* ``@classmethod`` as an alternative constructor (``AppContext.create(...)``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lemonrind.chats import ChatRepository
from lemonrind.config import Settings, database_path, settings_path
from lemonrind.lemonade import LemonadeClient
from lemonrind.modules import ModuleRegistry, build_registry
from lemonrind.storage import Database


@dataclass(slots=True)
class AppContext:
    settings: Settings
    settings_file: Path
    db: Database
    client: LemonadeClient
    repo: ChatRepository
    modules: ModuleRegistry | None = None  # the capability modules (tools the model can use)

    @classmethod
    def create(cls, data_dir: Path, *, base_url: str | None = None) -> AppContext:
        """Load settings, open the database and build the Lemonade client."""
        settings_file = settings_path(data_dir)
        settings = Settings.load(settings_file)
        if not settings_file.exists():
            settings.save(settings_file)
        if base_url:
            settings.lemonade.base_url = base_url
        db = Database(database_path(data_dir))
        client = LemonadeClient(settings.lemonade)
        return cls(
            settings=settings,
            settings_file=settings_file,
            db=db,
            client=client,
            repo=ChatRepository(db),
            modules=build_registry(settings, data_dir, db=db, client=client),
        )

    def save_settings(self) -> None:
        self.settings.save(self.settings_file)

    async def aclose(self) -> None:
        if self.modules is not None:
            await self.modules.stop_all()
        await self.client.aclose()
        self.db.close()


_current: AppContext | None = None


def set_context(context: AppContext | None) -> None:
    global _current
    _current = context


def get_context() -> AppContext:
    if _current is None:
        raise RuntimeError("The web UI was started without an AppContext (call set_context first).")
    return _current
