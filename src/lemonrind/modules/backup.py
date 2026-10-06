"""The Auto-backup module: now and then, zip the whole data folder as a safety net.

Everything the app owns lives in one folder (the database, settings, generated images, workspace files), so a
backup is just a zip of that folder. No AI is involved and the module offers the model no tools; it is a module
only to get the on/off switch and the live start/stop that every module already has, rather than inventing a
second way of switching a feature on.

How it behaves (the same rules as the .NET editions):

* It looks a few times a day (the first look a minute after start-up), and makes a backup when the last one is
  older than ``interval_days``. No backup yet means one is due at once.
* The backup folder must be **outside** the data folder. A zip stored inside the folder it backs up would contain every
  earlier zip, growing without limit, so such a setting is refused with a warning.
* The newest ``keep_count`` zips are kept; older ones are deleted after each backup.
* **One unreadable file does not spoil the backup**: it is skipped and logged, and the rest is saved.
* A plain text file, ``last_backup.txt``, next to the zips records when the last backup finished. State this small does
  not need a database table.

One improvement on the .NET version: the database is copied with SQLite's own *backup API* instead of zipping the file
as it lies on disk. The app keeps the database open in WAL mode (recent writes sit in a side file), so a plain file
copy can be missing the newest changes or catch a page half-written. The backup API takes a consistent snapshot even
while the app is writing, and the zip contains that snapshot.

Python ideas used here:

* ``zipfile.ZipFile`` as a context manager, adding files one at a time with ``write``.
* ``sqlite3.Connection.backup``: copy a live database safely.
* ``tempfile.TemporaryDirectory`` for the snapshot, and writing the zip under a temporary name then ``Path.replace``
  so a half-written zip never looks like a real backup.
* ``asyncio.to_thread`` for the slow part, so the web page stays responsive while a large folder is zipped.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from lemonrind.config import DATABASE_FILE_NAME, BackupSettings, Settings
from lemonrind.modules.base import Module
from lemonrind.modules.tool import Tool

logger = logging.getLogger(__name__)

ZIP_PREFIX = "lemonrind_backup_"
MARKER_FILE_NAME = "last_backup.txt"
FIRST_CHECK_SECONDS = 60
CHECK_EVERY_HOURS = 6  # a backup is day-granular, so looking this often is plenty


class BackupError(Exception):
    """A backup could not be made. The message says why and is fit to show."""


@dataclass(frozen=True, slots=True)
class BackupResult:
    path: Path
    files: int
    skipped: int  # files that could not be read


@dataclass(frozen=True, slots=True)
class BackupStatus:
    folder: Path
    last: datetime | None  # when the last backup finished (UTC)
    count: int  # zips currently kept


def resolve_backup_folder(config: BackupSettings, data_dir: Path) -> Path:
    """The folder for the zips. A relative setting means "beside the data folder", not inside it."""
    chosen = Path(os.path.expandvars(config.backup_folder)).expanduser()
    return (chosen if chosen.is_absolute() else data_dir.parent / chosen).resolve()


def paths_overlap(data_dir: Path, backup_dir: Path) -> bool:
    """True if the backup folder is the data folder or somewhere inside it."""
    data, backup = data_dir.resolve(), backup_dir.resolve()
    return backup == data or data in backup.parents


def list_backups(backup_dir: Path) -> list[Path]:
    """The zips in the folder, newest first (their names start with the date and time, so name order is time order)."""
    if not backup_dir.is_dir():
        return []
    return sorted(backup_dir.glob(f"{ZIP_PREFIX}*.zip"), reverse=True)


def last_backup_time(backup_dir: Path) -> datetime | None:
    try:
        text = (backup_dir / MARKER_FILE_NAME).read_text(encoding="utf-8").strip()
        moment = datetime.fromisoformat(text)
    except (OSError, ValueError):
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


def is_backup_due(backup_dir: Path, interval_days: int, now: datetime) -> bool:
    last = last_backup_time(backup_dir)
    return last is None or (now - last).total_seconds() >= interval_days * 86400


def create_backup(
    data_dir: Path, backup_dir: Path, *, keep: int, now: datetime | None = None
) -> BackupResult:
    """Zip ``data_dir`` into ``backup_dir`` and prune old zips. Raises ``BackupError`` if it cannot be done."""
    if paths_overlap(data_dir, backup_dir):
        raise BackupError(
            f"The backup folder ({backup_dir}) is inside the data folder it is backing up. Choose a folder "
            "outside it, or every backup would contain all the earlier ones."
        )
    if not data_dir.is_dir():
        raise BackupError(
            f"The data folder ({data_dir}) does not exist, so there is nothing to back up."
        )
    stamp = (now or datetime.now()).astimezone()
    backup_dir.mkdir(parents=True, exist_ok=True)
    final = backup_dir / f"{ZIP_PREFIX}{stamp:%Y%m%d_%H%M%S}.zip"
    counter = 1
    while final.exists():  # two backups in the same second (a button pressed twice)
        final = backup_dir / f"{ZIP_PREFIX}{stamp:%Y%m%d_%H%M%S}_{counter}.zip"
        counter += 1
    partial = final.with_name(final.name + ".partial")

    files = skipped = 0
    database = data_dir / DATABASE_FILE_NAME
    try:
        with tempfile.TemporaryDirectory(prefix="lemonrind-backup-") as scratch:
            snapshot: Path | None = None
            if database.is_file():
                snapshot = Path(scratch) / DATABASE_FILE_NAME
                try:
                    _snapshot_database(database, snapshot)
                except sqlite3.Error:
                    logger.warning(
                        "Auto-backup: could not snapshot the database; copying the file as it is"
                    )
                    snapshot = None
            with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED) as archive:
                for path in sorted(data_dir.rglob("*")):
                    if not path.is_file():
                        continue
                    is_database_side_file = (
                        path.name.startswith(DATABASE_FILE_NAME) and path != database
                    )
                    if snapshot is not None and is_database_side_file:
                        continue  # the snapshot already contains what was in the -wal / -shm files
                    source = snapshot if (snapshot is not None and path == database) else path
                    if (
                        path.suffix == ".kb"
                    ):  # a knowledge base file the app may be writing to: snapshot it too
                        copy = Path(scratch) / f"knowledge-{files}.kb"
                        try:
                            _snapshot_database(path, copy)
                            source = copy
                        except sqlite3.Error:
                            logger.warning("Auto-backup: copying %s as it is", path.name)
                    try:
                        archive.write(source, path.relative_to(data_dir).as_posix())
                        files += 1
                    except OSError:
                        skipped += 1
                        logger.warning(
                            "Auto-backup: skipped a file that could not be read (%s)", path
                        )
        partial.replace(final)
    finally:
        partial.unlink(missing_ok=True)  # only still there if something above failed
    (backup_dir / MARKER_FILE_NAME).write_text(
        datetime.now(UTC).isoformat() + "\n", encoding="utf-8"
    )
    _prune(backup_dir, keep)
    logger.info("Auto-backup completed: %s (%d files, %d skipped)", final, files, skipped)
    return BackupResult(final, files, skipped)


def _snapshot_database(source: Path, target: Path) -> None:
    """Copy a database that may be in use into ``target``, consistently (see the module notes)."""
    live = sqlite3.connect(f"file:{source.as_posix()}?mode=ro", uri=True)
    copy = sqlite3.connect(target)
    try:
        live.backup(copy)
    finally:
        copy.close()
        live.close()


def _prune(backup_dir: Path, keep: int) -> None:
    for stale in list_backups(backup_dir)[max(1, keep) :]:
        try:
            stale.unlink()
        except (
            OSError
        ):  # locked by something else (a virus scan, say): the next backup's prune gets it
            logger.warning("Auto-backup: could not delete an old backup (%s)", stale)


class AutoBackupModule(Module):
    name = "Auto-backup"
    config_key = "backup"
    description = (
        "Periodically zips your whole data folder (including all settings) as a safety net. "
        "No AI involvement: purely operational."
    )
    enabled_by_default = False  # it writes files, so it is opt-in (as in the .NET editions)

    def __init__(self, settings: Settings, data_dir: Path) -> None:
        super().__init__(settings)
        self.data_dir = data_dir
        self._task: asyncio.Task | None = None
        self._lock = (
            asyncio.Lock()
        )  # one backup at a time (the button and the timer must not overlap)

    def get_tools(self) -> list[Tool]:
        return []  # nothing for the model: this is background housekeeping

    # --- lifecycle ---------------------------------------------------------------------------------------------

    async def on_startup(self) -> None:
        self._task = asyncio.create_task(self._loop(), name="auto-backup")

    async def on_shutdown(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            await asyncio.wait({task}, timeout=15)

    async def _loop(self) -> None:
        await asyncio.sleep(FIRST_CHECK_SECONDS)
        while True:
            try:
                await self.run_if_due()
            except Exception:  # one failed backup must not end the checking for good
                logger.exception("Auto-backup failed")
            await asyncio.sleep(CHECK_EVERY_HOURS * 3600)

    # --- doing it ----------------------------------------------------------------------------------------------

    @property
    def backup_dir(self) -> Path:
        return resolve_backup_folder(self.settings.modules.backup, self.data_dir)

    def status(self) -> BackupStatus:
        folder = self.backup_dir
        return BackupStatus(folder, last_backup_time(folder), len(list_backups(folder)))

    async def run_if_due(self, now: datetime | None = None) -> BackupResult | None:
        """Make a backup if one is due. Returns ``None`` when nothing was done (not due, or the folder is not allowed)."""
        config = self.settings.modules.backup
        folder = self.backup_dir
        if paths_overlap(self.data_dir, folder):
            logger.warning(
                "Auto-backup folder (%s) is inside the data folder (%s): skipping. Change it in Settings.",
                folder,
                self.data_dir,
            )
            return None
        if not self.data_dir.is_dir() or not is_backup_due(
            folder, config.interval_days, now or datetime.now(UTC)
        ):
            return None
        return await self.backup_now()

    async def backup_now(self) -> BackupResult:
        """Make a backup right now (the "Back up now" button), whether or not one is due."""
        config = self.settings.modules.backup
        async with self._lock:
            return await asyncio.to_thread(
                create_backup, self.data_dir, self.backup_dir, keep=config.keep_count
            )
