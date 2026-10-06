"""Auto-backup: when one is due, what goes in the zip, pruning, refusing a folder inside the data folder."""

from __future__ import annotations

import sqlite3
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from lemonrind.config import BackupSettings, Settings
from lemonrind.modules.backup import (
    MARKER_FILE_NAME,
    AutoBackupModule,
    BackupError,
    create_backup,
    is_backup_due,
    last_backup_time,
    list_backups,
    paths_overlap,
    resolve_backup_folder,
)

NOW = datetime(2026, 10, 3, 12, 0, tzinfo=UTC)


def make_data_dir(root: Path) -> Path:
    data = root / "data"
    (data / "workspace").mkdir(parents=True)
    (data / "settings.json").write_text('{"x": 1}', encoding="utf-8")
    (data / "workspace" / "notes.txt").write_text("hello", encoding="utf-8")
    db = sqlite3.connect(data / "lemonrind.db")
    db.execute("create table t (v text)")
    db.execute("insert into t values ('kept')")
    db.commit()
    db.close()
    return data


def names(zip_path: Path) -> set[str]:
    with zipfile.ZipFile(zip_path) as archive:
        return set(archive.namelist())


# --- the folder rules ------------------------------------------------------------------------------------------


def test_a_relative_backup_folder_sits_beside_the_data_folder_and_an_absolute_one_is_kept(
    tmp_path: Path,
):
    data = tmp_path / "data"

    assert resolve_backup_folder(BackupSettings(), data) == (tmp_path / "data_backups").resolve()
    elsewhere = tmp_path / "other" / "zips"
    assert (
        resolve_backup_folder(BackupSettings(backup_folder=str(elsewhere)), data)
        == elsewhere.resolve()
    )


def test_a_backup_folder_inside_the_data_folder_is_recognised(tmp_path: Path):
    data = tmp_path / "data"

    assert paths_overlap(data, data)
    assert paths_overlap(data, data / "backups")
    assert not paths_overlap(data, tmp_path / "data_backups")
    assert not paths_overlap(
        data, tmp_path / "data2"
    )  # a sibling that merely starts with the same letters


def test_a_backup_is_due_when_there_is_none_or_the_last_one_is_old_enough(tmp_path: Path):
    assert is_backup_due(tmp_path, 7, NOW)  # no marker file: never backed up

    (tmp_path / MARKER_FILE_NAME).write_text(
        (NOW - timedelta(days=3)).isoformat(), encoding="utf-8"
    )
    assert not is_backup_due(tmp_path, 7, NOW)
    assert is_backup_due(tmp_path, 3, NOW)  # exactly the interval has passed
    assert last_backup_time(tmp_path) == NOW - timedelta(days=3)

    (tmp_path / MARKER_FILE_NAME).write_text("not a date", encoding="utf-8")
    assert (
        is_backup_due(tmp_path, 7, NOW) and last_backup_time(tmp_path) is None
    )  # unreadable: just back up


# --- making a backup -------------------------------------------------------------------------------------------


def test_the_zip_holds_every_file_and_a_consistent_copy_of_the_database(tmp_path: Path):
    data = make_data_dir(tmp_path)
    live = sqlite3.connect(data / "lemonrind.db")  # the app keeps it open, in WAL mode
    live.execute("pragma journal_mode=wal")
    live.execute("insert into t values ('still in the wal file')")
    live.commit()

    result = create_backup(data, tmp_path / "data_backups", keep=5)

    assert result.files == 3 and result.skipped == 0
    assert names(result.path) == {
        "settings.json",
        "workspace/notes.txt",
        "lemonrind.db",
    }  # no -wal / -shm
    restored = tmp_path / "restored"
    with zipfile.ZipFile(result.path) as archive:
        archive.extract("lemonrind.db", restored)
    copy = sqlite3.connect(restored / "lemonrind.db")
    assert [row[0] for row in copy.execute("select v from t order by rowid")] == [
        "kept",
        "still in the wal file",
    ]
    copy.close()
    live.close()
    assert last_backup_time(tmp_path / "data_backups") is not None  # the marker was written
    assert not list((tmp_path / "data_backups").glob("*.partial"))


def test_old_zips_are_pruned_to_the_newest_few_and_two_in_one_second_do_not_clash(tmp_path: Path):
    data = make_data_dir(tmp_path)
    folder = tmp_path / "data_backups"
    moment = datetime(2026, 10, 3, 12, 0, 0)
    for minute in range(4):
        create_backup(data, folder, keep=3, now=moment + timedelta(minutes=minute))
    create_backup(data, folder, keep=3, now=moment + timedelta(minutes=3))  # the same second again

    kept = list_backups(folder)
    assert len(kept) == 3
    assert kept[0].name.endswith("_120300_1.zip")  # the second one made in that second got a suffix
    assert kept[1].name.endswith("_120300.zip") and kept[2].name.endswith(
        "_120200.zip"
    )  # the oldest two went


def test_a_backup_into_the_data_folder_is_refused_and_so_is_a_missing_data_folder(tmp_path: Path):
    data = make_data_dir(tmp_path)

    with pytest.raises(BackupError, match="inside the data folder"):
        create_backup(data, data / "backups", keep=5)
    with pytest.raises(BackupError, match="does not exist"):
        create_backup(tmp_path / "nowhere", tmp_path / "zips", keep=5)
    assert not (data / "backups").exists()


def test_an_unreadable_file_is_skipped_not_fatal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data = make_data_dir(tmp_path)
    real_write = zipfile.ZipFile.write

    def flaky(self, filename, arcname=None, *args, **kwargs):
        if arcname == "workspace/notes.txt":
            raise PermissionError("locked")
        return real_write(self, filename, arcname, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "write", flaky)

    result = create_backup(data, tmp_path / "zips", keep=5)

    assert result.skipped == 1 and result.files == 2
    assert "workspace/notes.txt" not in names(result.path)


# --- the module ------------------------------------------------------------------------------------------------


def make_module(tmp_path: Path) -> AutoBackupModule:
    return AutoBackupModule(Settings(), make_data_dir(tmp_path))


def test_the_module_is_off_by_default_and_offers_the_model_nothing(tmp_path: Path):
    module = make_module(tmp_path)

    assert not module.enabled and module.get_tools() == []


async def test_a_backup_is_made_when_due_and_not_again_until_the_interval_has_passed(
    tmp_path: Path,
):
    module = make_module(tmp_path)

    first = await module.run_if_due(NOW)
    again = await module.run_if_due(datetime.now(UTC))

    assert first is not None and again is None
    status = module.status()
    assert (
        status.count == 1
        and status.last is not None
        and status.folder == (tmp_path / "data_backups").resolve()
    )
    module.settings.modules.backup.interval_days = 1
    later = await module.run_if_due(datetime.now(UTC) + timedelta(days=2))
    assert later is not None and module.status().count == 2


async def test_back_up_now_ignores_the_interval(tmp_path: Path):
    module = make_module(tmp_path)

    await module.backup_now()
    await module.backup_now()

    assert module.status().count == 2


async def test_a_folder_inside_the_data_folder_is_skipped_quietly_by_the_timer_but_reported_by_the_button(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    module = make_module(tmp_path)
    module.settings.modules.backup.backup_folder = str(module.data_dir / "backups")

    with caplog.at_level("WARNING"):
        assert await module.run_if_due() is None
    assert "inside the data folder" in caplog.text
    with pytest.raises(BackupError):
        await module.backup_now()


async def test_the_module_starts_and_stops_its_checking_task(tmp_path: Path):
    module = make_module(tmp_path)

    await module.on_startup()
    assert module._task is not None and not module._task.done()
    await module.on_shutdown()

    assert module._task is None


def test_a_knowledge_base_file_is_backed_up_and_the_copy_can_be_added_back(tmp_path: Path):
    from lemonrind.modules.knowledge import KnowledgeRepository

    data = make_data_dir(tmp_path)
    repo = KnowledgeRepository(data / "knowledge")  # the app keeps it open while the backup runs
    kb = repo.create_kb("Handbook")
    source = repo.create_source(kb.id, "text", None, "note")
    repo.add_chunks(kb.id, source.id, 0, ["pizza on fridays"], [[1.0, 0.0]], model="m")
    repo.mark_ready(source.id, 1)

    result = create_backup(data, tmp_path / "data_backups", keep=5)

    kb_names = [n for n in names(result.path) if n.endswith(".kb")]
    assert len(kb_names) == 1 and kb_names[0].startswith("knowledge/handbook-")
    restored = tmp_path / "restored"
    with zipfile.ZipFile(result.path) as archive:
        archive.extract(kb_names[0], restored)
    there = KnowledgeRepository(tmp_path / "elsewhere")
    added = there.import_file(
        restored / kb_names[0]
    )  # a backup copy is a usable knowledge base file
    assert added.id == kb.id and [c.content for c, _ in there.searchable_chunks(added.id)] == [
        "pizza on fridays"
    ]
    there.close()
    repo.close()
