"""Tests for the chat repository, run against a throwaway in-memory SQLite database."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from lemonrind.attachments import Attachment, stored_text
from lemonrind.chats import DEFAULT_TITLE, ChatNotFoundError, ChatRepository
from lemonrind.chats.text import SNIPPET_END, SNIPPET_START, parse_snippet
from lemonrind.lemonade.events import RequestStats
from lemonrind.storage import Database
from lemonrind.storage.migrations import MIGRATIONS
from tests.conftest import must


class FakeClock:
    """A clock that moves forward one minute every time it is read, so ordering is predictable."""

    def __init__(self) -> None:
        self._now = datetime(2026, 10, 2, 9, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        self._now += timedelta(minutes=1)
        return self._now


@pytest.fixture
def repo() -> ChatRepository:
    return ChatRepository(Database(":memory:"), clock=FakeClock())


# --- the database and its migrations ------------------------------------------------------------------


def test_new_database_is_at_the_latest_schema_version():
    assert Database(":memory:").schema_version == len(MIGRATIONS)


def test_reopening_a_file_database_keeps_data_and_does_not_rerun_migrations(tmp_path: Path):
    path = tmp_path / "sub" / "chats.db"  # the missing folder is created for us
    db = Database(path)
    session = ChatRepository(db).create_session("Keep me")
    db.close()

    reopened = Database(path)  # would fail with "table already exists" if migrations re-ran
    assert reopened.schema_version == len(MIGRATIONS)
    assert must(ChatRepository(reopened).get_session(session.id)).title == "Keep me"
    reopened.close()


def test_transaction_rolls_back_when_something_fails():
    db = Database(":memory:")
    with pytest.raises(RuntimeError), db.transaction() as conn:
        conn.execute("INSERT INTO folders (id, name) VALUES ('f1', 'Work')")
        raise RuntimeError("boom")
    assert db.conn.execute("SELECT count(*) FROM folders").fetchone()[0] == 0


# --- chats and messages ----------------------------------------------------------------------------------


def test_sessions_are_listed_most_recently_used_first(repo: ChatRepository):
    older = repo.create_session("Older")
    newer = repo.create_session("Newer")
    assert [s.id for s in repo.list_sessions()] == [newer.id, older.id]

    repo.add_message(older.id, "user", "bump")  # using a chat moves it to the top
    assert [s.id for s in repo.list_sessions()][0] == older.id
    assert [s.id for s in repo.list_sessions(limit=1)] == [older.id]


def test_messages_round_trip_with_reasoning_and_stats(repo: ChatRepository):
    session = repo.create_session()
    stats = RequestStats(
        prompt_tokens=41, input_tokens=40, output_tokens=149, tokens_per_second=27.2
    )
    repo.add_message(session.id, "user", "Capital of France?")
    repo.add_message(session.id, "assistant", "Paris.", reasoning="It is Paris.", stats=stats)

    user, assistant = repo.list_messages(session.id)
    assert (user.role, user.content, user.reasoning, user.stats) == (
        "user",
        "Capital of France?",
        None,
        None,
    )
    assert assistant.reasoning == "It is Paris."
    assert assistant.stats == stats


def test_first_user_message_names_the_chat_once(repo: ChatRepository):
    session = repo.create_session()
    repo.add_message(session.id, "user", "Plan a weekend hike near Manchester")
    assert must(repo.get_session(session.id)).title == "Plan a weekend hike near Manchester"

    repo.add_message(session.id, "user", "Something completely different")
    assert must(repo.get_session(session.id)).title == "Plan a weekend hike near Manchester"


def test_a_chat_with_its_own_title_is_not_renamed(repo: ChatRepository):
    session = repo.create_session("My project")
    repo.add_message(session.id, "user", "hello")
    assert must(repo.get_session(session.id)).title == "My project"


def test_an_assistant_message_does_not_name_the_chat(repo: ChatRepository):
    session = repo.create_session()
    repo.add_message(session.id, "assistant", "Hello!")
    assert must(repo.get_session(session.id)).title == DEFAULT_TITLE


def test_rename_and_missing_chats(repo: ChatRepository):
    session = repo.create_session()
    repo.rename_session(session.id, "  Renamed  ")
    assert must(repo.get_session(session.id)).title == "Renamed"
    with pytest.raises(ChatNotFoundError):
        repo.rename_session("nope", "x")
    with pytest.raises(ChatNotFoundError):
        repo.add_message("nope", "user", "x")
    assert repo.get_session("nope") is None


def test_deleting_a_chat_removes_its_messages_and_search_entries(repo: ChatRepository):
    session = repo.create_session()
    repo.add_message(session.id, "user", "zebra crossing")
    assert [s.id for s in repo.search("zebra")] == [session.id]

    repo.delete_session(session.id)
    assert repo.list_messages(session.id) == []
    assert repo.search("zebra") == []


def test_editing_a_message_keeps_the_search_index_correct(repo: ChatRepository):
    session = repo.create_session("Notes")  # a fixed title, so only the message text can match
    message = repo.add_message(session.id, "user", "alpha words")
    with repo._db.transaction() as conn:
        conn.execute("UPDATE messages SET content = 'beta words' WHERE id = ?", (message.id,))
    assert repo.search("alpha") == []
    assert [s.id for s in repo.search("beta")] == [session.id]


# --- folders and tags -----------------------------------------------------------------------------------


def test_folders_are_found_by_name_ignoring_case(repo: ChatRepository):
    work = repo.get_or_create_folder("Work")
    assert repo.get_or_create_folder("  work ").id == work.id
    assert [f.name for f in repo.list_folders()] == ["Work"]


def test_a_chat_can_be_filed_and_unfiled(repo: ChatRepository):
    folder = repo.get_or_create_folder("Work")
    session = repo.create_session()
    repo.move_to_folder(session.id, folder.id)
    assert must(repo.get_session(session.id)).folder_name == "Work"
    repo.move_to_folder(session.id, None)
    assert must(repo.get_session(session.id)).folder_id is None


def test_deleting_a_folder_keeps_its_chats(repo: ChatRepository):
    folder = repo.get_or_create_folder("Temp")
    session = repo.create_session("Keep")
    repo.move_to_folder(session.id, folder.id)
    repo.delete_folder(folder.id)
    kept = repo.get_session(session.id)
    assert kept is not None and kept.folder_id is None


def test_tags_are_shared_case_insensitive_and_idempotent(repo: ChatRepository):
    session = repo.create_session()
    repo.add_tag(session.id, "Code")
    repo.add_tag(session.id, "code")  # same tag
    repo.add_tag(session.id, "work")
    assert must(repo.get_session(session.id)).tags == ("Code", "work")
    assert repo.list_tags() == ["Code", "work"]

    repo.remove_tag(session.id, "CODE")
    assert must(repo.get_session(session.id)).tags == ("work",)


# --- search ----------------------------------------------------------------------------------------------


@pytest.fixture
def library(repo: ChatRepository) -> ChatRepository:
    hike = repo.create_session("Weekend hike ideas")
    repo.add_message(hike.id, "user", "Suggest a half-day hike near Manchester")
    repo.add_message(hike.id, "assistant", "Try Lyme Park to Bowstones.")
    regex = repo.create_session("Regex help")
    repo.add_message(regex.id, "user", "A regex for UK postcodes please")
    repo.add_tag(regex.id, "code")
    repo.move_to_folder(regex.id, repo.get_or_create_folder("Work").id)
    return repo


def titles(sessions) -> set[str]:
    return {s.title for s in sessions}


def test_search_matches_titles_messages_folders_and_tags(library: ChatRepository):
    assert titles(library.search("hike")) == {"Weekend hike ideas"}  # title and message
    assert titles(library.search("postcodes")) == {"Regex help"}  # message only
    assert titles(library.search("work")) == {"Regex help"}  # folder name
    assert titles(library.search("cod")) == {"Regex help"}  # tag (substring)


def test_search_hits_carry_a_snippet_of_the_matching_message(library: ChatRepository):
    (hit,) = library.search_hits("postcodes")
    assert hit.session.title == "Regex help"
    assert parse_snippet(hit.snippet or "") == [
        ("A regex for UK ", False),
        ("postcodes", True),
        (" please", False),
    ]


def test_a_chat_found_only_by_its_title_folder_or_tag_has_no_snippet(library: ChatRepository):
    assert [h.snippet for h in library.search_hits("work")] == [None]  # folder name
    assert [h.snippet for h in library.search_hits("cod")] == [None]  # tag


def test_the_snippet_comes_from_the_best_matching_message_and_never_from_a_tool_result(
    repo: ChatRepository,
):
    session = repo.create_session("Notes")
    repo.add_message(session.id, "user", "otter")
    repo.add_message(session.id, "assistant", "An otter is an otter is an otter, said the otter.")
    repo.add_message(session.id, "tool", "otter otter otter otter otter otter", tool_call_id="x")

    (hit,) = repo.search_hits("otter")

    assert hit.snippet is not None and f"{SNIPPET_START}otter{SNIPPET_END}" in hit.snippet
    assert "otter otter otter otter" not in (hit.snippet or "")  # the tool result is not offered


def test_search_hits_with_an_empty_query_list_every_chat_without_snippets(library: ChatRepository):
    hits = library.search_hits("  ")
    assert {h.session.title for h in hits} == {"Weekend hike ideas", "Regex help"}
    assert all(h.snippet is None for h in hits)


def test_a_snippet_never_shows_the_path_in_a_picture_marker(repo: ChatRepository):
    session = repo.create_session("Garden photo")
    marked = stored_text(
        "Does this bed look ready?",
        Attachment("image", "garden-bed.jpg", path=Path("C:/secret/data/garden-bed.jpg")),
    )
    repo.add_message(session.id, "user", marked)

    by_path = [h for h in repo.search_hits("garden-bed") if h.session.id == session.id]
    by_typed = [h for h in repo.search_hits("ready") if h.session.id == session.id]

    assert [h.snippet for h in by_path] in (
        [None],
        [],
    )  # matched only the hidden path: nothing safe to show
    (hit,) = by_typed
    assert "secret" not in (hit.snippet or "")
    assert parse_snippet(hit.snippet or "") == [
        ("Does this bed look ", False),
        ("ready", True),
        ("?", False),
    ]


def test_message_search_matches_word_beginnings(library: ChatRepository):
    assert titles(library.search("manch")) == {"Weekend hike ideas"}  # start of "Manchester"
    assert library.search("chester") == []  # middle of a word does not match message text


def test_several_words_must_all_match(library: ChatRepository):
    assert titles(library.search("hike manchester")) == {"Weekend hike ideas"}
    assert library.search("hike postcodes") == []


def test_odd_input_never_breaks_the_search(library: ChatRepository):
    for query in ['"', "-", "*", "AND", "(", 'a"b', "%", "_", "'; DROP TABLE sessions; --"]:
        library.search(query)  # must not raise
    assert len(library.list_sessions()) == 2  # and nothing was damaged


def test_like_wildcards_typed_by_the_user_are_literal(library: ChatRepository):
    assert library.search("%") == []
    assert library.search("_") == []


def test_empty_search_lists_everything(library: ChatRepository):
    assert len(library.search("   ")) == 2


def test_a_tool_messages_failure_is_stored_and_read_back():
    repo = ChatRepository(Database(":memory:"))
    chat = repo.create_session()
    repo.add_message(chat.id, "tool", "42", tool_call_id="a")
    repo.add_message(chat.id, "tool", "Error: no such tool", tool_call_id="b", tool_failed=True)

    ok, failed = repo.list_messages(chat.id)

    assert (ok.tool_failed, failed.tool_failed) == (False, True)


def test_the_sql_helpers_refuse_anything_but_the_fixed_pieces_they_were_written_for(tmp_path):
    from lemonrind.modules.mcp.repository import McpServerRepository
    from lemonrind.modules.memory.repository import MemoryRepository

    db = Database(":memory:")
    chat = ChatRepository(db).create_session()
    with pytest.raises(ValueError, match="allowed update"):
        ChatRepository(db)._update_session(chat.id, "title = 'x' WHERE 1=1 --", ())
    with pytest.raises(ValueError, match="updatable column"):
        McpServerRepository(db)._update("id", "name = 'x' WHERE 1=1 --", 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="allowed update"):
        MemoryRepository(db)._update("id", "content = 'x'; DROP TABLE memories; --", ())
