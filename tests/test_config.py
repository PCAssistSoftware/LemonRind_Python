"""Tests for settings: defaults, validation, saving/loading and finding the data folder."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from lemonrind import config
from lemonrind.config import (
    DATA_DIR_ENV,
    LOCATION_FILE_NAME,
    BackupSettings,
    LemonadeSettings,
    Settings,
    default_data_dir,
    read_data_location,
    resolve_data_dir,
    settings_path,
    user_data_dir,
    write_data_location,
)


def test_base_url_always_ends_with_a_slash():
    assert LemonadeSettings(base_url="http://host:1/v1").base_url == "http://host:1/v1/"
    assert LemonadeSettings(base_url="  http://host:1/v1/ ").base_url == "http://host:1/v1/"


def test_assigning_a_new_base_url_is_also_tidied():
    settings = LemonadeSettings()
    settings.base_url = "http://other:2/v1"  # validate_assignment makes the validator run here too
    assert settings.base_url == "http://other:2/v1/"


def test_save_then_load_round_trips(tmp_path: Path):
    path = settings_path(tmp_path)
    original = Settings()
    original.lemonade.chat_model = "Qwen3-8B-GGUF"
    original.save(path)
    assert Settings.load(path).lemonade.chat_model == "Qwen3-8B-GGUF"


def test_ui_theme_must_be_one_of_the_known_values(tmp_path: Path):
    path = tmp_path / "settings.json"
    path.write_text('{"ui": {"theme": "purple"}}', encoding="utf-8")
    with pytest.raises(ValidationError):
        Settings.load(path)
    assert Settings().ui.theme == "system"


def test_missing_file_gives_defaults(tmp_path: Path):
    assert Settings.load(tmp_path / "nope.json") == Settings()


def test_fields_missing_from_the_file_keep_their_defaults(tmp_path: Path):
    path = tmp_path / "settings.json"
    path.write_text('{"lemonade": {"chat_model": "X"}}', encoding="utf-8")
    loaded = Settings.load(path)
    assert loaded.lemonade.chat_model == "X"
    assert loaded.lemonade.base_url == LemonadeSettings().base_url


def test_wrong_type_in_the_file_is_rejected(tmp_path: Path):
    path = tmp_path / "settings.json"
    path.write_text('{"lemonade": {"request_timeout_seconds": "soon"}}', encoding="utf-8")
    with pytest.raises(ValidationError):
        Settings.load(path)


def test_data_dir_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    checkout = tmp_path / "checkout"
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path / "from-env"))
    assert (
        resolve_data_dir(tmp_path / "explicit", source_checkout=checkout)
        == (tmp_path / "explicit").resolve()
    )
    assert resolve_data_dir(source_checkout=checkout) == (tmp_path / "from-env").resolve()

    monkeypatch.delenv(DATA_DIR_ENV)
    assert resolve_data_dir(source_checkout=checkout) == (checkout / "data").resolve()
    assert resolve_data_dir(source_checkout=None) == user_data_dir().resolve()


def test_running_from_a_checkout_uses_the_project_data_folder(monkeypatch: pytest.MonkeyPatch):
    # The real project folder has a pyproject.toml, so launching from anywhere finds the same data.
    monkeypatch.delenv(DATA_DIR_ENV, raising=False)
    monkeypatch.chdir(Path.home())
    assert resolve_data_dir().name == "data"
    assert (resolve_data_dir().parent / "pyproject.toml").is_file()


# --- the data folder chosen in Settings > Storage -------------------------------------------------------------------


def test_a_chosen_data_folder_is_remembered_in_the_default_folder_and_used_next_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv(DATA_DIR_ENV, raising=False)
    checkout = tmp_path / "checkout"
    default = default_data_dir(source_checkout=checkout)
    assert (
        read_data_location(default) is None
        and resolve_data_dir(source_checkout=checkout) == default
    )

    elsewhere = tmp_path / "elsewhere"
    write_data_location(default, elsewhere)

    assert (default / LOCATION_FILE_NAME).read_text(encoding="utf-8").strip() == str(
        elsewhere.resolve()
    )
    assert read_data_location(default) == elsewhere.resolve()
    assert resolve_data_dir(source_checkout=checkout) == elsewhere.resolve()


def test_the_command_line_and_environment_still_beat_the_chosen_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    write_data_location(default_data_dir(source_checkout=checkout), tmp_path / "chosen")
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path / "from-env"))

    assert resolve_data_dir(source_checkout=checkout) == (tmp_path / "from-env").resolve()
    assert (
        resolve_data_dir(tmp_path / "explicit", source_checkout=checkout)
        == (tmp_path / "explicit").resolve()
    )


def test_choosing_nothing_or_the_default_removes_the_pointer(tmp_path: Path):
    default = tmp_path / "data"
    write_data_location(default, tmp_path / "elsewhere")

    write_data_location(default, None)
    assert read_data_location(default) is None

    write_data_location(default, tmp_path / "elsewhere")
    write_data_location(default, default)  # choosing the default is the same as choosing nothing
    assert not (default / LOCATION_FILE_NAME).exists()


def test_a_blank_or_missing_pointer_file_means_the_default(tmp_path: Path):
    default = tmp_path / "data"
    default.mkdir()
    assert read_data_location(default) is None
    (default / LOCATION_FILE_NAME).write_text("   ", encoding="utf-8")
    assert read_data_location(default) is None


def test_backup_settings_have_sensible_limits():
    assert BackupSettings().interval_days == 7 and BackupSettings().keep_count == 5
    with pytest.raises(ValidationError):
        BackupSettings(interval_days=0)
    with pytest.raises(ValidationError):
        BackupSettings(keep_count=1000)


def test_an_installed_copy_uses_a_fixed_per_user_folder_wherever_it_is_started_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv(DATA_DIR_ENV, raising=False)
    monkeypatch.setattr(
        config.Path, "home", lambda: tmp_path / "home"
    )  # nothing real is read or written
    monkeypatch.setattr(config.sys, "platform", "linux")
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()

    monkeypatch.chdir(first)
    here = resolve_data_dir(source_checkout=None)
    monkeypatch.chdir(second)
    there = resolve_data_dir(source_checkout=None)

    assert here == there == (tmp_path / "home" / ".local" / "share" / "lemonrind").resolve()


@pytest.mark.parametrize(
    ("platform", "variable", "expected"),
    [
        ("win32", "LOCALAPPDATA", ("custom", "LemonRind")),
        ("linux", "XDG_DATA_HOME", ("custom", "lemonrind")),
    ],
)
def test_the_per_user_folder_follows_the_systems_own_setting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
    variable: str,
    expected: tuple[str, str],
):
    monkeypatch.setattr(config.sys, "platform", platform)
    monkeypatch.setenv(variable, str(tmp_path / "custom"))
    assert user_data_dir() == tmp_path / Path(*expected)


def test_the_per_user_folder_has_sensible_defaults_without_those_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    home = tmp_path / "home"
    monkeypatch.setattr(config.Path, "home", lambda: home)
    for variable in ("LOCALAPPDATA", "XDG_DATA_HOME"):
        monkeypatch.delenv(variable, raising=False)

    monkeypatch.setattr(config.sys, "platform", "win32")
    assert user_data_dir() == home / "AppData" / "Local" / "LemonRind"
    monkeypatch.setattr(config.sys, "platform", "darwin")
    assert user_data_dir() == home / "Library" / "Application Support" / "LemonRind"
    monkeypatch.setattr(config.sys, "platform", "linux")
    assert user_data_dir() == home / ".local" / "share" / "lemonrind"
