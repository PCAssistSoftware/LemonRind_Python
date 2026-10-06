"""The usage screen in the browser: opening it, the filters, and a click on a busy chat."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from nicegui import ui
from nicegui.testing import User

from lemonrind.lemonade.events import RequestStats
from lemonrind.webui.context import AppContext
from lemonrind.webui.usage_dialog import PERIODS, period_start
from tests.conftest import elements
from tests.test_settings_sections import settle


def seed(web: AppContext) -> None:
    """Three chats: a recent one with two models, an old one, and a scheduled job's."""
    repo = web.repo
    clock = [datetime.now(UTC)]
    repo._clock = lambda: clock[0]  # back-dating rows is the point of this helper

    recent = repo.create_session("Recent chat")
    repo.add_message(recent.id, "user", "hello")
    repo.add_message(
        recent.id,
        "assistant",
        "hi",
        stats=RequestStats(input_tokens=1000, output_tokens=200, tokens_per_second=90.0),
        model="small",
    )
    repo.add_message(
        recent.id,
        "assistant",
        "again",
        stats=RequestStats(input_tokens=500, output_tokens=100, tokens_per_second=30.0),
        model="big",
    )
    job = repo.create_session("Morning digest")
    repo.add_tag(job.id, "scheduled")
    repo.add_message(
        job.id,
        "assistant",
        "the morning digest is ready",
        stats=RequestStats(input_tokens=4000, output_tokens=300, tokens_per_second=80.0),
        model="small",
    )
    clock[0] = datetime.now(UTC) - timedelta(days=60)
    old = repo.create_session("Old chat")
    repo.add_message(
        old.id,
        "assistant",
        "ancient",
        stats=RequestStats(input_tokens=777, output_tokens=7, tokens_per_second=10.0),
        model="legacy",
    )


def model_rows(user: User) -> dict[str, int]:
    (table,) = elements(user, "usage-models")
    return {row["model"]: row["requests"] for row in table.rows}


async def open_usage(user: User) -> None:
    await user.open("/")
    user.find(marker="view-usage").click()
    await user.should_see("Tokens and speed from the requests this app has saved")


async def test_the_usage_screen_opens_from_the_right_hand_panel_with_the_last_30_days(
    user: User, web: AppContext
):
    seed(web)

    await open_usage(user)

    assert model_rows(user) == {
        "small": 2,
        "big": 1,
    }  # the 60-day-old request is outside the default period
    await user.should_see(marker="usage-chart-tokens")
    await user.should_see(marker="usage-chart-speed")
    # a chart has only a height class, so in a short window the dialog's flex layout would squeeze it to nothing
    # unless it is told not to shrink (this was a real bug: the charts were blank until a control was changed)
    assert all(
        "shrink-0" in elements(user, m)[0].classes
        for m in ("usage-chart-tokens", "usage-chart-speed")
    )
    assert [row["model"] for row in elements(user, "usage-models")[0].rows] == [
        "small",
        "big",
    ]  # busiest first


async def test_with_no_saved_requests_the_screen_says_so_plainly(user: User, web: AppContext):
    await open_usage(user)

    await user.should_see("No requests in this period.")
    await user.should_not_see(marker="usage-models")


async def test_choosing_all_time_brings_in_the_old_request(user: User, web: AppContext):
    seed(web)
    await open_usage(user)

    (toggle,) = elements(user, "usage-period")
    toggle.value = "All time"
    await settle()

    assert model_rows(user) == {"small": 2, "big": 1, "legacy": 1}


async def test_the_model_filter_narrows_the_tables_and_scheduled_jobs_can_be_left_out(
    user: User, web: AppContext
):
    seed(web)
    await open_usage(user)

    (scheduled,) = elements(user, "usage-scheduled")
    scheduled.value = False
    await settle()
    assert model_rows(user) == {"small": 1, "big": 1}  # the digest is gone

    (picker,) = elements(user, "usage-model")
    picker.value = "big"
    await settle()
    assert model_rows(user) == {"big": 1}
    assert "small" in picker.options  # the picker still lists every model to switch to


async def test_a_model_with_no_requests_in_the_new_period_falls_back_to_all_models(
    user: User, web: AppContext
):
    seed(web)
    await open_usage(user)
    (toggle,) = elements(user, "usage-period")
    toggle.value = "All time"
    await settle()
    (picker,) = elements(user, "usage-model")
    picker.value = "legacy"  # only the 60-day-old request used it
    await settle()
    assert model_rows(user) == {"legacy": 1}

    toggle.value = "30 days"  # nothing from "legacy" in the last 30 days
    await settle()

    assert picker.value == "All models"
    assert model_rows(user) == {"small": 2, "big": 1}


async def test_a_busiest_chat_opens_that_chat(user: User, web: AppContext):
    seed(web)
    await open_usage(user)
    await user.should_not_see("the morning digest is ready")  # no chat is open yet

    user.find(kind=ui.button, content="Morning digest").click()  # 4,300 tokens: the busiest chat
    await settle()

    await user.should_see("the morning digest is ready")


def test_the_period_starts_at_midnight_counting_today_as_the_first_day():
    from datetime import timezone

    tz = timezone(timedelta(hours=2))
    now = datetime(2026, 10, 6, 1, 30, tzinfo=tz)

    start = period_start(7, now, tz)

    assert start == datetime(2026, 9, 30, 0, 0, tzinfo=tz)  # the 30th to the 6th is seven days
    assert period_start(None, now, tz) is None
    assert PERIODS["All time"] is None and PERIODS["12 months"] == 365
