"""Tests for the pure text helpers: titles, search queries and relative times."""

from datetime import UTC, datetime, timedelta

from lemonrind.chats.text import (
    DEFAULT_TITLE,
    SNIPPET_END,
    SNIPPET_START,
    auto_title,
    build_fts_query,
    highlight_parts,
    parse_snippet,
    relative_time,
    snippet_from_text,
)


def test_auto_title_uses_the_first_non_empty_line_and_tidies_spaces():
    assert auto_title("\n\n  Hello    there  \nsecond line") == "Hello there"


def test_auto_title_shortens_at_a_word_boundary():
    text = "Please summarise the quarterly budget meeting notes for the whole team"
    title = auto_title(text, max_length=40)
    assert title == "Please summarise the quarterly budget..."
    assert len(title) <= 43


def test_auto_title_with_one_giant_word_cuts_hard():
    assert auto_title("x" * 100, max_length=10) == "xxxxxxxxxx..."


def test_auto_title_of_nothing_is_the_default():
    assert auto_title("   \n  ") == DEFAULT_TITLE


def test_fts_query_quotes_every_word_as_a_prefix():
    assert build_fts_query("hik manch") == '"hik"* "manch"*'


def test_fts_query_neutralises_operators_and_quotes():
    # Words that would be FTS5 syntax become harmless quoted text; embedded quotes are doubled.
    assert build_fts_query("AND -x (y)") == '"AND"* "-x"* "(y)"*'
    assert build_fts_query('say "hi"') == '"say"* """hi"""*'


def test_fts_query_of_nothing_is_none():
    assert build_fts_query("   ") is None


NOW = datetime(2026, 10, 2, 15, 0, tzinfo=UTC)


def test_relative_time_buckets():
    assert relative_time(NOW - timedelta(seconds=20), NOW, UTC) == "now"
    assert relative_time(NOW - timedelta(minutes=5), NOW, UTC) == "5m"
    assert relative_time(NOW - timedelta(hours=3), NOW, UTC) == "3h"
    assert relative_time(NOW - timedelta(days=4), NOW, UTC) == "4d"
    assert relative_time(NOW - timedelta(days=30), NOW, UTC) == "2 Sep"


def test_yesterday_means_the_previous_calendar_day():
    # Under 24 hours ago is still shown in hours, even across midnight; older than that but on the
    # previous calendar date is "Yesterday".
    assert relative_time(datetime(2026, 10, 1, 16, 0, tzinfo=UTC), NOW, UTC) == "23h"
    assert relative_time(datetime(2026, 10, 1, 10, 0, tzinfo=UTC), NOW, UTC) == "Yesterday"
    assert relative_time(datetime(2026, 10, 2, 12, 0, tzinfo=UTC), NOW, UTC) == "3h"


def test_a_time_slightly_in_the_future_is_just_now():
    assert relative_time(NOW + timedelta(seconds=5), NOW, UTC) == "now"


# --- highlighting ----------------------------------------------------------------------------------------


def test_highlight_parts_flags_each_typed_word_ignoring_case():
    assert highlight_parts("Plan a Hike near Hikes", "hik") == [
        ("Plan a ", False),
        ("Hik", True),
        ("e near ", False),
        ("Hik", True),
        ("es", False),
    ]
    assert highlight_parts("Weekend hike ideas", "hike week") == [
        ("Week", True),
        ("end ", False),
        ("hike", True),
        (" ideas", False),
    ]


def test_highlight_parts_with_nothing_to_find_returns_the_text_whole():
    assert highlight_parts("Plain title", "") == [("Plain title", False)]
    assert highlight_parts("Plain title", "zzz") == [("Plain title", False)]
    assert highlight_parts("", "abc") == []


def test_highlight_parts_treats_typed_characters_literally():
    # "a.c" must not behave as a pattern where the dot matches anything
    assert highlight_parts("abc a.c", "a.c") == [("abc ", False), ("a.c", True)]
    assert highlight_parts("(x)", "(") == [("(", True), ("x)", False)]


def test_parse_snippet_splits_on_the_marker_characters():
    raw = f"...a half-day {SNIPPET_START}hike{SNIPPET_END} near Manchester, {SNIPPET_START}hik{SNIPPET_END}ing..."
    assert parse_snippet(raw) == [
        ("...a half-day ", False),
        ("hike", True),
        (" near Manchester, ", False),
        ("hik", True),
        ("ing...", False),
    ]
    assert parse_snippet("no matches here") == [("no matches here", False)]
    assert parse_snippet("") == []


def test_snippet_from_text_cuts_around_the_first_match_and_marks_it():
    text = (
        "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen"
    )
    raw = snippet_from_text(text, "TEN", words_around=2)
    assert parse_snippet(raw or "") == [
        ("...eight nine ", False),
        ("ten", True),
        (" eleven twelve...", False),
    ]
    assert parse_snippet(snippet_from_text("short Hike text", "hik") or "") == [
        ("short ", False),
        ("Hik", True),
        ("e text", False),
    ]
    assert snippet_from_text("nothing here", "zzz") is None
    assert snippet_from_text("anything", "   ") is None
    assert (
        snippet_from_text("price (5.00) here", "(5") is not None
    )  # typed characters are not a pattern
