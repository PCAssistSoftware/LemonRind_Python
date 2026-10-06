"""Usage statistics: the arithmetic (made-up rows), the saved model name, and the queries that feed it."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta, timezone

import pytest

from lemonrind.chats import ChatRepository
from lemonrind.chats.usage import (
    ALL_MODELS,
    UNKNOWN_MODEL,
    ToolUse,
    UsageRequest,
    bucket_kind_for,
    build_report,
    compact,
    totals_of,
)
from lemonrind.lemonade import ToolCall
from lemonrind.lemonade.events import RequestStats
from lemonrind.storage import Database
from lemonrind.storage.migrations import MIGRATIONS, apply_migrations
from tests.test_tool_loop import RoundsStreamer, answer_round, make, tool_round

NOW = datetime(2026, 10, 6, 12, 0, tzinfo=UTC)


def request(
    days_ago: float = 0,
    *,
    model: str | None = "small",
    input_tokens: int = 100,
    output_tokens: int = 50,
    speed: float = 80.0,
    first: float = 1.0,
    session: str = "s1",
    title: str = "Chat",
    scheduled: bool = False,
    at: datetime | None = None,
) -> UsageRequest:
    return UsageRequest(
        at=at or NOW - timedelta(days=days_ago),
        session_id=session,
        title=title,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        tokens_per_second=speed,
        time_to_first_token=first,
        scheduled=scheduled,
    )


def report(requests, tools=(), **kwargs):
    kwargs.setdefault("since", None)
    kwargs.setdefault("tz", UTC)
    return build_report(list(requests), list(tools), until=NOW, **kwargs)


# --- the arithmetic ---------------------------------------------------------------------------------------------------


def test_totals_add_tokens_and_weight_the_speed_by_output_length():
    rows = [
        request(output_tokens=300, speed=100.0, first=2.0),
        request(output_tokens=100, speed=20.0, first=4.0),
    ]
    totals = totals_of(rows)
    assert (totals.requests, totals.input_tokens, totals.output_tokens) == (2, 200, 400)
    assert totals.tokens_per_second == pytest.approx((300 * 100 + 100 * 20) / 400)  # not a plain 60
    assert totals.time_to_first_token == pytest.approx(3.0)


def test_totals_ignore_zero_speeds_and_waits_and_cope_with_no_requests():
    totals = totals_of([request(speed=0.0, first=0.0), request(speed=50.0, first=2.0)])
    assert totals.tokens_per_second == 50.0 and totals.time_to_first_token == 2.0
    assert totals_of([]).requests == 0
    assert totals_of([request(speed=0.0, first=0.0)]).tokens_per_second == 0.0


@pytest.mark.parametrize(
    ("number", "text"),
    [(0, "0"), (950, "950"), (1200, "1.2 k"), (226_000, "226.0 k"), (1_840_000, "1.84 M")],
)
def test_compact_numbers(number, text):
    assert compact(number) == text


# --- the period, scheduled jobs and the model filter ---------------------------------------------------------------------


def test_only_requests_inside_the_period_count():
    rows = [request(1), request(5), request(40)]
    assert report(rows, since=NOW - timedelta(days=7)).totals.requests == 2
    assert report(rows).totals.requests == 3  # no start: everything
    assert report([request(-1)]).totals.requests == 0  # after "until"


def test_scheduled_jobs_can_be_left_out():
    rows = [request(1), request(1, scheduled=True), request(1, scheduled=True)]
    assert report(rows).totals.requests == 3
    assert report(rows, include_scheduled=False).totals.requests == 1


def test_the_model_filter_narrows_everything_but_still_lists_every_model_to_pick_from():
    rows = [
        request(model="small"),
        request(model="small"),
        request(model="big"),
        request(model=None),
    ]
    tools = [
        ToolUse(NOW, "search", False, "big", False),
        ToolUse(NOW, "search", False, "small", False),
    ]

    only_big = report(rows, tools, model="big")

    assert only_big.totals.requests == 1
    assert [row.model for row in only_big.models] == ["big"]
    assert [(t.name, t.calls) for t in only_big.tools] == [("search", 1)]
    assert only_big.available_models == ["big", "small", UNKNOWN_MODEL]
    assert report(rows, model=UNKNOWN_MODEL).totals.requests == 1
    assert ALL_MODELS == "All models"


def test_models_are_listed_busiest_first_with_the_slowest_start():
    rows = [
        request(model="b", first=3.0),
        request(model="a", first=1.0),
        request(model="a", first=9.5),
        request(model=None),
    ]
    models = report(rows).models
    assert [(m.model, m.totals.requests) for m in models] == [
        ("a", 2),
        ("b", 1),
        (UNKNOWN_MODEL, 1),
    ]
    assert models[0].slowest_start == 9.5


# --- the chart -----------------------------------------------------------------------------------------------------------


def test_a_short_period_is_charted_by_day_with_no_gaps():
    rows = [
        request(0, input_tokens=10, output_tokens=5),
        request(2, input_tokens=30, output_tokens=7),
    ]
    result = report(rows, since=NOW - timedelta(days=3))

    assert result.bucket_kind == "day"
    assert [b.start for b in result.buckets] == [date(2026, 10, d) for d in (3, 4, 5, 6)]
    assert [(b.input_tokens, b.output_tokens) for b in result.buckets] == [
        (0, 0),
        (30, 7),
        (0, 0),
        (10, 5),
    ]
    assert result.buckets[-1].label == "6 Oct"


def test_a_day_belongs_to_the_computers_time_zone_not_to_utc():
    # 23:30 UTC on the 5th is 01:30 on the 6th at UTC+2
    late = request(at=datetime(2026, 10, 5, 23, 30, tzinfo=UTC), input_tokens=77)
    plus_two = timezone(timedelta(hours=2))

    in_utc = report([late], since=NOW - timedelta(days=2), tz=UTC)
    local = report([late], since=NOW - timedelta(days=2), tz=plus_two)

    assert {b.start: b.input_tokens for b in in_utc.buckets}[date(2026, 10, 5)] == 77
    assert {b.start: b.input_tokens for b in local.buckets}[date(2026, 10, 6)] == 77


def test_the_chart_switches_to_weeks_then_months_for_longer_spans():
    assert bucket_kind_for(date(2026, 9, 1), date(2026, 10, 6)) == "day"  # 36 days
    assert bucket_kind_for(date(2026, 1, 1), date(2026, 10, 6)) == "week"
    assert bucket_kind_for(date(2024, 1, 1), date(2026, 10, 6)) == "month"

    weekly = report([request(100)], since=NOW - timedelta(days=200))
    assert weekly.bucket_kind == "week" and all(b.start.weekday() == 0 for b in weekly.buckets)
    assert weekly.buckets[-1].label.startswith("w/c ")

    monthly = report([request(0), request(500)], since=None)
    assert monthly.bucket_kind == "month"
    assert monthly.buckets[0].start.day == 1 and monthly.buckets[-1].label == "Oct 2026"
    months = [b.start for b in monthly.buckets]
    assert months == sorted(months) and len(months) == len(set(months))  # no gaps, no repeats


def test_all_time_with_no_use_is_a_single_empty_bucket_not_an_error():
    result = report([])
    assert result.totals.requests == 0 and len(result.buckets) == 1
    assert result.models == [] and result.tools == [] and result.chats == []


# --- tools and busiest chats ---------------------------------------------------------------------------------------------


def test_tools_show_calls_failures_and_the_failure_rate():
    uses = [
        ToolUse(NOW, "read_webpage", True, "m", False),
        ToolUse(NOW, "read_webpage", False, "m", False),
        ToolUse(NOW, "read_webpage", True, "m", False),
        ToolUse(NOW, "web_search", False, "m", False),
        ToolUse(NOW - timedelta(days=90), "write_file", False, "m", False),
    ]
    tools = report([], uses, since=NOW - timedelta(days=30)).tools
    assert [(t.name, t.calls, t.failed) for t in tools] == [
        ("read_webpage", 3, 2),
        ("web_search", 1, 0),
    ]
    assert tools[0].failure_rate == pytest.approx(2 / 3)


def test_busiest_chats_are_the_top_five_by_tokens():
    rows = [
        request(session=f"s{n}", title=f"Chat {n}", input_tokens=n * 100, output_tokens=0)
        for n in range(1, 8)
    ]
    chats = report(rows).chats
    assert [c.title for c in chats] == ["Chat 7", "Chat 6", "Chat 5", "Chat 4", "Chat 3"]
    assert chats[0].tokens == 700


# --- saving and reading it back ------------------------------------------------------------------------------------------


@pytest.fixture
def repo() -> ChatRepository:
    return ChatRepository(Database(":memory:"))


def test_the_model_is_saved_with_a_reply_and_read_back(repo: ChatRepository):
    session = repo.create_session("T")
    repo.add_message(
        session.id, "assistant", "hi", stats=RequestStats(output_tokens=3), model="small"
    )
    repo.add_message(session.id, "assistant", "no stats", model=None)

    saved, plain = repo.list_messages(session.id)

    assert saved.model == "small" and plain.model is None


def test_old_databases_gain_the_model_column_and_keep_their_replies():
    db = Database(":memory:", MIGRATIONS[:9])  # the schema as it was before the model was recorded
    stamp = datetime(2026, 10, 5, 12, 0, tzinfo=UTC).isoformat()
    with db.transaction() as conn:  # written by hand: the repository now writes the new column
        conn.execute(
            "INSERT INTO sessions (id, title, created_at, updated_at) VALUES ('old', 'Old chat', ?, ?)",
            (stamp, stamp),
        )
        conn.execute(
            "INSERT INTO messages (session_id, role, content, stats_json, created_at)"
            " VALUES ('old', 'assistant', 'old reply', '{\"output_tokens\": 9}', ?)",
            (stamp,),
        )

    assert apply_migrations(db.conn) == len(MIGRATIONS)  # brought up to date

    (row,) = ChatRepository(db).usage_requests()
    assert row.model is None and row.output_tokens == 9


def test_usage_requests_report_the_chat_title_the_scheduled_tag_and_respect_since(
    repo: ChatRepository,
):
    chat = repo.create_session("Hike")
    job = repo.create_session("Digest")
    repo.add_tag(job.id, "scheduled")
    repo.add_message(chat.id, "user", "question")  # no stats: not a model request
    repo.add_message(
        chat.id, "assistant", "a", stats=RequestStats(input_tokens=5, output_tokens=2), model="m"
    )
    repo.add_message(job.id, "assistant", "b", stats=RequestStats(output_tokens=1), model="m")

    rows = repo.usage_requests()

    assert [(r.title, r.model, r.scheduled, r.input_tokens) for r in rows] == [
        ("Hike", "m", False, 5),
        ("Digest", "m", True, 0),
    ]
    assert repo.usage_requests(since=datetime.now(UTC) + timedelta(days=1)) == []


def test_tool_uses_are_matched_to_their_calls_even_when_call_ids_repeat(repo: ChatRepository):
    session = repo.create_session("Tools")
    same_id = "call-1"
    repo.add_message(
        session.id, "assistant", "", tool_calls=[ToolCall(same_id, "web_search", "{}")], model="m1"
    )
    repo.add_message(session.id, "tool", "ok", tool_call_id=same_id)
    repo.add_message(
        session.id,
        "assistant",
        "",
        tool_calls=[ToolCall(same_id, "read_webpage", "{}")],
        model="m2",
    )
    repo.add_message(session.id, "tool", "broken", tool_call_id=same_id, tool_failed=True)
    other = repo.create_session("Other chat")
    repo.add_message(
        other.id, "tool", "orphan", tool_call_id="never-asked"
    )  # no matching call: skipped

    uses = repo.usage_tool_uses()

    assert [(u.name, u.failed, u.model) for u in uses] == [
        ("web_search", False, "m1"),
        ("read_webpage", True, "m2"),
    ]


async def test_a_conversation_saves_the_model_with_every_request_it_made():
    streamer = RoundsStreamer(tool_round("Looking. "), answer_round("Done."))
    conversation, repo = make(streamer)

    await conversation.send("Find otters")

    (session,) = repo.list_sessions()
    stored = repo.list_messages(session.id)
    assert [(m.role, m.model) for m in stored] == [
        ("user", None),
        ("assistant", "m"),  # the tool round
        ("tool", None),
        ("assistant", "m"),  # the answer
    ]
    assert len(repo.usage_requests()) == 2  # two requests: one for the tool, one for the answer
    assert [u.name for u in repo.usage_tool_uses()] == ["search"]
