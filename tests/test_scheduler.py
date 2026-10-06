"""Tests for the scheduler: cron arithmetic, storage, the APScheduler-driven module, the tools, and running a job."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from apscheduler.events import EVENT_JOB_MISSED, JobEvent

from lemonrind.chats import ChatRepository
from lemonrind.config import Settings
from lemonrind.lemonade import ChatEvent, Finished, TextDelta, ToolCall, ToolCallsRequested
from lemonrind.lemonade.client import LemonadeError
from lemonrind.lemonade.events import RequestStats
from lemonrind.lemonade.models import Health, LoadedModel, ModelInfo
from lemonrind.modules import Module, ModuleRegistry, Tool
from lemonrind.modules.scheduler import (
    CronError,
    DuplicateJobError,
    JobOutcome,
    JobRunner,
    SchedulerModule,
    SchedulerRepository,
)
from lemonrind.modules.scheduler.cron import (
    _weekday_names,
    format_local,
    next_run,
    normalise,
    upcoming,
)
from lemonrind.storage import Database

NOW = datetime(2026, 10, 2, 12, 0, tzinfo=UTC)

# --- cron -------------------------------------------------------------------------------------------------------


def test_expressions_are_tidied_and_checked():
    assert normalise("  0   19 * *  5 ") == "0 19 * * 5"
    for bad, message in [
        ("0 19 * *", "5 fields but this has 4"),
        ("* * * * * *", "5 fields but this has 6"),  # a sixth (seconds) field is not accepted
        ("61 * * * *", "not a valid cron expression"),
        ("", "5 fields"),
        ("every day", "5 fields"),
    ]:
        with pytest.raises(CronError, match=message):
            normalise(bad)


def test_the_next_run_is_in_the_future_at_the_local_wall_clock_time_asked_for():
    moment = next_run("30 8 * * *", NOW)

    assert moment is not None and moment > NOW and moment - NOW <= timedelta(hours=25)
    local = moment.astimezone()
    assert (local.hour, local.minute) == (8, 30)  # "8:30" means 8:30 where the computer is
    assert moment.tzinfo == UTC  # but it is returned in UTC, for storage


def test_runs_follow_each_other_and_never_repeat_the_start_moment():
    times = upcoming("*/15 * * * *", NOW, 4)

    assert len(times) == 4 and times == sorted(times) and times[0] > NOW
    assert {t - s for s, t in zip(times, times[1:], strict=False)} == {timedelta(minutes=15)}


def test_weekday_and_month_fields_work():
    friday = next_run("0 19 * * 5", NOW)
    assert friday is not None and friday.astimezone().weekday() == 4


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("5", "fri"),
        ("0", "sun"),
        ("7", "sun"),  # cron accepts 7 as Sunday too
        ("1-5", "mon,tue,wed,thu,fri"),
        ("*/2", "sun,tue,thu,sat"),
        ("1,3", "mon,wed"),
        ("5-7", "sun,fri,sat"),
        ("2/2", "tue,thu,sat"),  # "from Tuesday to the end of the week, every second day"
        ("mon-fri", "mon-fri"),  # names are the same in both systems
        ("1,fri", "fri,mon"),
    ],
)
def test_cron_weekday_numbers_become_day_names_because_the_library_counts_from_monday(
    field, expected
):
    assert _weekday_names(field) == expected


@pytest.mark.parametrize("bad", ["8", "3-1", "*/0", "1/0"])
def test_a_bad_weekday_is_reported_as_a_cron_error(bad):
    with pytest.raises(CronError, match="day-of-week"):
        _weekday_names(bad)


def test_a_misspelt_weekday_name_is_rejected_by_the_whole_check():
    with pytest.raises(CronError, match="not a valid cron expression"):
        normalise("0 9 * * funday")


def test_each_weekday_number_fires_on_that_day_not_the_day_after():
    for number, expected in [
        (0, 6),
        (1, 0),
        (2, 1),
        (5, 4),
        (6, 5),
        (7, 6),
    ]:  # python's weekday(): Monday is 0
        moment = next_run(f"0 12 * * {number}", NOW)
        assert moment is not None and moment.astimezone().weekday() == expected, number


def test_weekday_ranges_skip_the_weekend():
    days = {t.astimezone().weekday() for t in upcoming("0 9 * * 1-5", NOW, 14)}

    assert days == {0, 1, 2, 3, 4}


def test_restricting_both_day_of_month_and_day_of_week_is_rejected_with_an_explanation():
    with pytest.raises(CronError, match="ambiguous"):
        normalise("0 9 1 * 1")
    assert (
        next_run("0 9 1 * 1", NOW) is None
    )  # an old saved job like this is simply never scheduled
    assert normalise("0 9 1 * *") == "0 9 1 * *" and normalise("0 9 * * 1") == "0 9 * * 1"


def test_a_summer_time_change_keeps_the_local_wall_clock_time():
    # Whatever zone this computer is in, "9am" stays 9am on both sides of the year's clock changes.
    hours = {
        t.astimezone().hour for t in upcoming("0 9 * * *", datetime(2026, 3, 20, tzinfo=UTC), 20)
    }
    hours |= {
        t.astimezone().hour for t in upcoming("0 9 * * *", datetime(2026, 10, 20, tzinfo=UTC), 20)
    }

    assert hours == {9}


def test_a_schedule_that_can_never_fire_gives_none():
    assert next_run("0 0 31 2 *", NOW) is None  # the 31st of February
    assert upcoming("0 0 31 2 *", NOW, 3) == []


def test_times_are_shown_in_local_time():
    assert format_local(None) == "never"
    assert format_local(NOW) == NOW.astimezone().strftime("%a %d %b %H:%M")


# --- storage -----------------------------------------------------------------------------------------------------


def make_repo(clock=lambda: NOW) -> SchedulerRepository:
    return SchedulerRepository(Database(":memory:"), clock)


def test_jobs_are_stored_and_names_are_unique_ignoring_case():
    repo = make_repo()
    job = repo.create("Backup check", "0 9 * * 1", "Check the backups", NOW + timedelta(days=3))

    assert (job.enabled, job.allow_unattended_tools, job.last_status) == (
        True,
        False,
        "",
    )  # safe defaults
    with pytest.raises(DuplicateJobError):
        repo.create("backup CHECK", "0 9 * * 1", "x", None)
    assert repo.find("BACKUP check") is not None and [j.name for j in repo.list()] == [
        "Backup check"
    ]


def test_recording_a_run_clears_running_and_moves_the_schedule_on():
    repo = make_repo()
    job = repo.create("j", "0 9 * * *", "p", NOW)
    repo.mark_running(job.id)
    assert repo.get(job.id).running_since is not None  # type: ignore[union-attr]

    repo.record_run(job.id, status="ok", session_id="chat-1", next_run_at=NOW + timedelta(days=1))
    after = repo.get(job.id)

    assert after is not None
    assert (after.running_since, after.last_status, after.last_session_id) == (None, "ok", "chat-1")
    assert after.next_run_at == NOW + timedelta(days=1) and after.last_run_at == NOW


def test_a_manual_run_keeps_the_schedule_and_the_previous_chat_link_when_it_has_none():
    repo = make_repo()
    job = repo.create("j", "0 9 * * *", "p", NOW + timedelta(hours=2))
    repo.record_run(job.id, status="ok", session_id="chat-1", next_run_at=NOW + timedelta(hours=2))

    repo.record_run(
        job.id, status="failed", session_id=None, next_run_at=None, update_schedule=False
    )

    after = repo.get(job.id)
    assert after is not None and after.last_status == "failed"
    assert after.next_run_at == NOW + timedelta(hours=2)  # untouched
    assert after.last_session_id == "chat-1"  # a run with no chat does not erase the link


def test_runs_cut_off_by_a_restart_are_noted():
    repo = make_repo()
    job = repo.create("j", "* * * * *", "p", NOW)
    repo.mark_running(job.id)

    assert repo.fail_interrupted() == 1

    after = repo.get(job.id)
    assert after is not None and (after.running_since, after.last_status) == (None, "interrupted")


def test_enabling_stores_a_fresh_time_and_disabling_leaves_it():
    repo = make_repo()
    job = repo.create("j", "* * * * *", "p", NOW - timedelta(days=5))

    repo.set_enabled(job.id, False, None)
    assert repo.get(job.id).next_run_at == NOW - timedelta(days=5)  # type: ignore[union-attr]
    repo.set_enabled(job.id, True, NOW + timedelta(minutes=1))

    assert repo.get(job.id).next_run_at == NOW + timedelta(minutes=1)  # type: ignore[union-attr]


# --- the module: polling, missed runs, manual runs -----------------------------------------------------------------


@pytest.fixture(autouse=True)
async def _not_near_a_minute_boundary():
    """Several tests use an every-minute job on the real clock and count how often it runs. A test that straddles the
    turn of a minute would see a genuine extra firing, so wait out the last moments of a minute before each test."""
    now = datetime.now()
    second = now.second + now.microsecond / 1e6
    if second > 57.5:
        await asyncio.sleep(60.5 - second)


class FakeRunner:
    """Stands in for JobRunner: records which jobs it was asked to run."""

    def __init__(self, status: str = "ok", delay: float = 0.0) -> None:
        self.status, self.delay = status, delay
        self.ran: list[str] = []
        self.running = 0
        self.most_running = 0

    def clear_running_tags(self) -> None:
        pass

    async def run(self, job) -> JobOutcome:
        self.running += 1
        self.most_running = max(self.most_running, self.running)
        try:
            self.ran.append(job.name)
            await asyncio.sleep(self.delay)
            return JobOutcome(self.status, f"chat-for-{job.name}", "done")
        finally:
            self.running -= 1


class Clock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def make_module(runner=None, now: datetime = NOW):
    clock = Clock(now)
    settings = Settings()
    repo = SchedulerRepository(Database(":memory:"), clock)
    module = SchedulerModule(settings, repo, runner or FakeRunner(), clock=clock)  # type: ignore[arg-type]
    return module, repo, clock, settings


async def wait_for(condition, timeout: float = 5.0) -> None:
    """Wait for something the scheduler does on its own (it runs in the background)."""
    for _ in range(int(timeout / 0.02)):
        if condition():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("timed out waiting for the scheduler")


def real_module(runner=None):
    """A module on the real clock: APScheduler always uses the real time, so the due times must be real too."""
    return make_module(runner, now=datetime.now(UTC))


async def test_a_job_that_is_due_at_start_up_runs_and_is_rescheduled_from_after_the_run():
    module, repo, clock, _ = real_module()
    job = repo.create("daily", "0 9 * * *", "do it", clock.now - timedelta(minutes=5))

    await module.on_startup()
    try:
        await wait_for(lambda: module.runs_finished == 1)
    finally:
        await module.on_shutdown()

    after = repo.get(job.id)
    assert after is not None
    assert after.last_status == "ok" and after.last_session_id == "chat-for-daily"
    assert after.next_run_at is not None and after.next_run_at > clock.now  # not due again
    assert after.running_since is None
    assert module.last_outcome is not None and module.last_outcome[0] == "daily"
    assert module._scheduler is None  # stopped cleanly


async def test_a_job_that_is_not_yet_due_waits_and_a_disabled_one_is_never_scheduled():
    module, repo, clock, _ = real_module()
    waiting = repo.create("later", "0 9 * * *", "p", clock.now + timedelta(days=1))
    off = repo.create("off", "* * * * *", "p", clock.now - timedelta(minutes=5))
    repo.set_enabled(off.id, False, None)

    await module.on_startup()
    try:
        await asyncio.sleep(0.3)
        assert module._scheduler is not None
        assert module._scheduler.get_job(waiting.id) is not None
        assert module._scheduler.get_job(off.id) is None
    finally:
        await module.on_shutdown()

    assert module._runner.ran == []  # type: ignore[attr-defined]


async def test_jobs_run_one_at_a_time_in_due_order_even_with_a_manual_run():
    runner = FakeRunner(delay=0.1)
    module, repo, clock, _ = real_module(runner)
    repo.create("second", "* * * * *", "p", clock.now - timedelta(minutes=1))
    repo.create("first", "* * * * *", "p", clock.now - timedelta(minutes=10))
    manual = repo.create("manual", "0 9 * * 1", "p", clock.now + timedelta(days=3))

    await module.on_startup()
    try:
        await asyncio.sleep(0.05)  # both scheduled runs have started or are queued
        await module.run_now(manual.id)
        await wait_for(lambda: module.runs_finished == 3)
    finally:
        await module.on_shutdown()

    assert runner.ran[:2] == ["first", "second"] and sorted(runner.ran) == [
        "first",
        "manual",
        "second",
    ]
    assert runner.most_running == 1


async def test_a_run_that_was_missed_by_too_long_is_skipped_not_replayed():
    module, repo, clock, settings = real_module()
    settings.modules.scheduler.missed_grace_minutes = 60
    stale = repo.create("stale", "0 9 * * 1", "p", clock.now - timedelta(hours=5))
    recent = repo.create("recent", "0 9 * * 1", "p", clock.now - timedelta(minutes=30))

    await module.on_startup()
    try:
        await wait_for(
            lambda: (
                (repo.get(stale.id).last_status == "missed")  # type: ignore[union-attr]
                and (repo.get(recent.id).last_status == "ok")
            )  # type: ignore[union-attr]
        )
    finally:
        await module.on_shutdown()

    assert module._runner.ran == ["recent"]  # type: ignore[attr-defined]
    skipped = repo.get(stale.id)
    assert (
        skipped is not None and skipped.next_run_at is not None and skipped.next_run_at > clock.now
    )


async def test_many_missed_firings_of_one_job_are_coalesced_into_a_single_run():
    runner = FakeRunner()
    module, repo, clock, _ = real_module(runner)
    repo.create(
        "minutely", "* * * * *", "p", clock.now - timedelta(minutes=20)
    )  # twenty firings missed

    await module.on_startup()
    try:
        await wait_for(lambda: runner.ran)
        await asyncio.sleep(0.4)  # long enough for any extra, wrongly replayed runs
    finally:
        await module.on_shutdown()

    assert runner.ran == ["minutely"]


async def test_a_job_left_running_by_a_previous_session_is_cleaned_up_and_runs_again():
    runner = FakeRunner()
    module, repo, clock, _ = real_module(runner)
    stuck = repo.create("stuck", "* * * * *", "p", clock.now - timedelta(minutes=1))
    repo.mark_running(stuck.id)

    await module.on_startup()
    try:
        await wait_for(lambda: module.runs_finished == 1)
    finally:
        await module.on_shutdown()

    assert runner.ran == ["stuck"] and repo.get(stuck.id).last_status == "ok"  # type: ignore[union-attr]


async def test_run_now_keeps_the_normal_schedule():
    module, repo, _, _ = make_module()
    planned = NOW + timedelta(days=2)
    job = repo.create("weekly", "0 9 * * 1", "p", planned)

    outcome = await module.run_now(job.id)

    after = repo.get(job.id)
    assert outcome is not None and outcome.status == "ok"
    assert after is not None and after.next_run_at == planned and after.last_status == "ok"
    assert await module.run_now("no-such-job") is None
    assert module.runs_finished == 1


async def test_failures_and_timeouts_are_recorded_as_the_status():
    module, repo, clock, _ = real_module(FakeRunner(status="timed out"))
    job = repo.create("slow", "0 9 * * 1", "p", clock.now - timedelta(minutes=1))

    await module.on_startup()
    try:
        await wait_for(lambda: module.runs_finished == 1)
    finally:
        await module.on_shutdown()

    assert repo.get(job.id).last_status == "timed out"  # type: ignore[union-attr]


async def test_a_job_cancelled_at_shutdown_is_recorded_as_interrupted():
    runner = FakeRunner(delay=60)
    module, repo, clock, _ = real_module(runner)
    job = repo.create("long", "0 9 * * 1", "p", clock.now - timedelta(minutes=1))

    await module.on_startup()
    await wait_for(lambda: runner.running == 1)
    await module.on_shutdown()  # cancels the run and waits for it to record what happened

    after = repo.get(job.id)
    assert after is not None and after.last_status == "interrupted" and after.running_since is None


async def test_a_manual_run_cancelled_at_shutdown_is_recorded_as_interrupted_too():
    runner = FakeRunner(delay=60)
    module, repo, clock, _ = real_module(runner)
    job = repo.create("long", "0 9 * * 1", "p", clock.now + timedelta(days=1))

    await module.on_startup()
    module.start_now(job.id)
    await wait_for(lambda: runner.running == 1)
    await module.on_shutdown()

    assert repo.get(job.id).last_status == "interrupted"  # type: ignore[union-attr]


async def test_the_module_can_be_switched_off_and_on_again():
    runner = FakeRunner()
    module, repo, clock, _ = real_module(runner)
    repo.create("j", "0 9 * * 1", "p", clock.now + timedelta(days=1))

    for _ in range(2):  # Settings switches modules off and on while the app runs
        await module.on_startup()
        assert module._scheduler is not None and module._scheduler.running
        await module.on_shutdown()
        assert module._scheduler is None


async def test_changes_made_while_running_reach_the_scheduler_at_once():
    module, repo, clock, _ = real_module()
    await module.on_startup()
    try:
        assert module._scheduler is not None
        scheduler = module._scheduler

        job = module.add_job("Friday note", "0 19 * * 5", "p")  # added while running
        known = scheduler.get_job(job.id)
        assert known is not None and known.next_run_time.astimezone().weekday() == 4  # a Friday
        assert known.next_run_time.astimezone(UTC) == job.next_run_at

        module.edit_job(
            job.id,
            name="Friday note",
            schedule="30 8 * * 1",
            prompt="p",
            model="",
            allow_unattended_tools=False,
        )
        edited = scheduler.get_job(job.id)
        assert (
            edited is not None and edited.next_run_time.astimezone().weekday() == 0
        )  # now a Monday
        assert (
            edited.next_run_time.astimezone().hour,
            edited.next_run_time.astimezone().minute,
        ) == (8, 30)

        module.set_enabled(job.id, False)
        assert scheduler.get_job(job.id) is None
        module.set_enabled(job.id, True)
        back = scheduler.get_job(job.id)
        assert back is not None and back.next_run_time > datetime.now(
            UTC
        )  # from now, no catching up

        module.delete_job(job.id)
        assert scheduler.get_job(job.id) is None and repo.get(job.id) is None
    finally:
        await module.on_shutdown()


async def test_a_job_deleted_or_switched_off_after_it_was_scheduled_does_not_run():
    runner = FakeRunner()
    module, repo, clock, _ = real_module(runner)
    job = repo.create("gone", "0 9 * * 1", "p", clock.now + timedelta(days=1))

    await module._fire(job.id)  # the scheduler calls this when due: the job is on, so it runs
    repo.set_enabled(job.id, False, None)
    await module._fire(job.id)  # switched off in the meantime
    repo.delete(job.id)
    await module._fire(job.id)  # deleted in the meantime

    assert runner.ran == ["gone"]


async def test_a_saved_schedule_this_version_cannot_use_is_skipped_with_a_warning(caplog):
    module, repo, clock, _ = real_module()
    legacy = repo.create(
        "legacy", "0 9 1 * 1", "p", clock.now - timedelta(minutes=1)
    )  # day-of-month AND weekday
    fine = repo.create("fine", "0 9 * * 1", "p", clock.now + timedelta(days=1))

    with caplog.at_level("WARNING"):
        await module.on_startup()
    try:
        assert module._scheduler is not None
        assert (
            module._scheduler.get_job(legacy.id) is None
            and module._scheduler.get_job(fine.id) is not None
        )
    finally:
        await module.on_shutdown()

    assert "Not scheduling 'legacy'" in caplog.text


async def test_skipped_firings_are_recorded_for_missed_ones_and_ignored_for_unknown_jobs():
    module, repo, clock, _ = real_module()
    job = repo.create("j", "0 9 * * 1", "p", clock.now - timedelta(days=2))

    module._on_scheduler_event(JobEvent(EVENT_JOB_MISSED, job.id, "default"))
    module._on_scheduler_event(
        JobEvent(EVENT_JOB_MISSED, "no-such-job", "default")
    )  # must not raise

    after = repo.get(job.id)
    assert after is not None and after.last_status == "missed" and after.next_run_at is not None


# --- managing jobs and the model's tools ---------------------------------------------------------------------------


def test_adding_and_editing_jobs_validates_the_schedule_and_recomputes_the_next_run():
    module, repo, _, _ = make_module()

    job = module.add_job("Report", " 0  9 * * 1 ", "  Write the report ")
    assert (job.cron, job.prompt) == (
        "0 9 * * 1",
        "Write the report",
    ) and job.next_run_at is not None
    for bad in [("", "0 9 * * 1", "p"), ("x", "0 9 * * 1", " "), ("y", "nonsense", "p")]:
        with pytest.raises((ValueError, CronError)):
            module.add_job(*bad)
    with pytest.raises(DuplicateJobError):
        module.add_job("report", "0 9 * * 1", "again")

    original_next = job.next_run_at
    module.edit_job(
        job.id,
        name="Report",
        schedule="0 9 * * 1",
        prompt="New text",
        model="m",
        allow_unattended_tools=True,
    )
    same_schedule = repo.get(job.id)
    assert (
        same_schedule is not None and same_schedule.next_run_at == original_next
    )  # unchanged schedule
    assert (same_schedule.prompt, same_schedule.model, same_schedule.allow_unattended_tools) == (
        "New text",
        "m",
        True,
    )

    module.edit_job(
        job.id,
        name="Report",
        schedule="0 18 * * 5",
        prompt="x",
        model="",
        allow_unattended_tools=False,
    )
    assert repo.get(job.id).next_run_at != original_next  # type: ignore[union-attr]


def test_switching_a_job_back_on_does_not_catch_up_on_what_it_missed():
    module, repo, clock, _ = make_module()
    job = repo.create("j", "0 9 * * *", "p", NOW - timedelta(days=3))
    module.set_enabled(job.id, False)

    module.set_enabled(job.id, True)

    after = repo.get(job.id)
    assert (
        after is not None
        and after.enabled
        and after.next_run_at is not None
        and after.next_run_at > clock.now
    )


async def test_the_model_can_schedule_list_and_cancel_jobs_and_scheduling_needs_permission():
    module, repo, _, _ = make_module()
    registry = ModuleRegistry([module])
    tools = {t.name: t for t in module.get_tools()}

    assert (
        tools["schedule_job"].requires_approval
        and not tools["list_scheduled_jobs"].requires_approval
    )

    made = await tools["schedule_job"].run(
        '{"name": "Friday note", "cron_expression": "0 19 * * 5", "prompt": "Say hi"}'
    )
    assert "Scheduled 'Friday note'" in made.content and "(local time)" in made.content
    job = repo.find("Friday note")
    assert (
        job is not None and job.allow_unattended_tools is False
    )  # a model-made job never gets that
    assert (
        "'Friday note' (on): 0 19 * * 5" in (await tools["list_scheduled_jobs"].run("{}")).content
    )

    bad = await tools["schedule_job"].run(
        '{"name": "x", "cron_expression": "tomorrow", "prompt": "p"}'
    )
    duplicate = await tools["schedule_job"].run(
        '{"name": "friday NOTE", "cron_expression": "0 9 * * *", "prompt": "p"}'
    )
    assert bad.is_error and "5 fields" in bad.content
    assert duplicate.is_error and "already exists" in duplicate.content

    assert (
        "Deleted"
        in (
            await registry.run(ToolCall("c", "cancel_scheduled_job", '{"name": "Friday note"}'))
        ).content
    )
    assert (await tools["cancel_scheduled_job"].run('{"name": "Friday note"}')).is_error
    assert (await tools["list_scheduled_jobs"].run("{}")).content == "There are no scheduled jobs."


# --- running a job through the assistant ---------------------------------------------------------------------------


class FakeLemonade:
    """A client with one chat model and a scripted reply (or error), recording every request."""

    def __init__(self, *rounds: list[ChatEvent] | Exception) -> None:
        self.rounds = list(rounds)
        self.requests: list[dict] = []

    async def health(self) -> Health:
        return Health(
            status="ok", model_loaded="Chat", all_models_loaded=[LoadedModel(model_name="Chat")]
        )

    async def list_models(self, **kwargs) -> list[ModelInfo]:
        return [ModelInfo(id="Chat", labels=["chat"], downloaded=True)]

    async def stream_chat(
        self, messages, model, *, tools=None, max_tokens=None
    ) -> AsyncIterator[ChatEvent]:
        self.requests.append(
            {"messages": list(messages), "model": model, "max_tokens": max_tokens, "tools": tools}
        )
        step: Any = (
            self.rounds.pop(0)
            if self.rounds
            else [TextDelta("done"), Finished(RequestStats(), "stop")]
        )
        if isinstance(step, Exception):
            raise step
        for event in step:
            yield event


def reply(text: str) -> list[ChatEvent]:
    return [TextDelta(text), Finished(RequestStats(prompt_tokens=10, output_tokens=5), "stop")]


def make_runner(client, registry=None, settings=None):
    settings = settings or Settings()
    chats = ChatRepository(Database(":memory:"))
    clock = lambda: datetime(2026, 10, 2, 19, 0).astimezone()  # noqa: E731
    runner = JobRunner(settings, chats, client, lambda: registry, clock)
    return runner, chats, settings


def a_job(repo: SchedulerRepository, **kwargs):
    return repo.create(
        "Weekly digest", "0 9 * * 1", kwargs.pop("prompt", "Summarise the week"), None, **kwargs
    )


async def test_a_successful_run_leaves_a_tagged_chat_named_after_the_job():
    client = FakeLemonade(reply("Here is your digest."))
    runner, chats, settings = make_runner(client)
    job = a_job(make_repo())

    outcome = await runner.run(job)

    assert outcome.status == "ok" and outcome.session_id is not None
    session = chats.get_session(outcome.session_id)
    assert session is not None and session.title == "Weekly digest" and "scheduled" in session.tags
    assert [(m.role, m.content) for m in chats.list_messages(session.id)] == [
        ("user", "Summarise the week"),
        ("assistant", "Here is your digest."),
    ]
    request = client.requests[0]
    assert (
        request["max_tokens"] == settings.modules.scheduler.max_output_tokens
    )  # runaway generation is capped
    system = request["messages"][0]["content"]
    assert (
        "Current date and time: Friday 02 October 2026, 19:00" in system
        and "scheduled job" in system
    )


async def test_a_job_can_name_its_own_model_but_an_unknown_one_fails_clearly():
    client = FakeLemonade()
    runner, chats, _ = make_runner(client)
    outcome = await runner.run(a_job(make_repo(), model="Nonexistent"))

    assert outcome.status == "failed"
    session = chats.get_session(outcome.session_id)  # type: ignore[arg-type]
    assert session is not None and {"scheduled", "failed"} <= set(session.tags)
    assert "not downloaded" in chats.list_messages(session.id)[-1].content


async def test_a_lemonade_error_becomes_a_chat_with_the_prompt_the_reason_and_a_hint():
    client = FakeLemonade(
        LemonadeError(
            "Lemonade rejected the request: the prompt exceeds the available context size"
        )
    )
    runner, chats, _ = make_runner(client)

    outcome = await runner.run(a_job(make_repo()))

    assert outcome.status == "failed"
    messages = chats.list_messages(outcome.session_id)  # type: ignore[arg-type]
    assert messages[0].role == "user" and messages[0].content == "Summarise the week"
    assert (
        "This scheduled job failed" in messages[1].content
        and "larger context size" in messages[1].content
    )


async def test_a_model_that_answers_nothing_is_retried_once_then_reported():
    empty = [Finished(RequestStats(), "length")]
    client = FakeLemonade(empty, empty)
    runner, chats, _ = make_runner(client)

    outcome = await runner.run(a_job(make_repo()))

    assert outcome.status == "failed" and len(client.requests) == 2
    assert "no answer" in chats.list_messages(outcome.session_id)[-1].content  # type: ignore[arg-type]


async def test_an_empty_first_answer_is_retried_and_the_second_one_is_used():
    client = FakeLemonade([Finished(RequestStats(), "length")], reply("Second time lucky."))
    runner, chats, _ = make_runner(client)

    outcome = await runner.run(a_job(make_repo()))

    assert outcome.status == "ok" and len(client.requests) == 2
    assert chats.list_messages(outcome.session_id)[-1].content == "Second time lucky."  # type: ignore[arg-type]


async def test_a_job_that_runs_too_long_is_stopped_and_explained():
    class Slow(FakeLemonade):
        async def stream_chat(self, messages, model, **kwargs):
            yield TextDelta("Starting to write")
            await asyncio.sleep(60)
            yield Finished(RequestStats(), "stop")

    settings = Settings()
    settings.modules.scheduler.job_timeout_seconds = 1.0
    runner, chats, _ = make_runner(Slow(), settings=settings)

    outcome = await runner.run(a_job(make_repo()))

    assert outcome.status == "timed out"
    messages = chats.list_messages(outcome.session_id)  # type: ignore[arg-type]
    assert "time limit" in messages[-1].content  # the explanation is added to the chat
    assert "Starting to write" in " ".join(
        m.content for m in messages
    )  # and what had arrived is kept


class Risky(Module):
    name, config_key, description = "Risky", "risky", "needs permission"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.ran: list[dict] = []

    def get_tools(self) -> list[Tool]:
        async def danger(arguments: dict) -> str:
            self.ran.append(arguments)
            return "done"

        return [Tool("danger", "risky", {"type": "object"}, danger, None, requires_approval=True)]


def asks_for_danger() -> list[list[ChatEvent]]:
    call = ToolCall("c1", "danger", "{}")
    return [
        [ToolCallsRequested((call,)), Finished(RequestStats(), "tool_calls")],
        reply("All done."),
    ]


async def test_tools_that_need_permission_are_refused_in_an_unattended_job_unless_allowed():
    settings = Settings()
    for allowed, should_run in ((False, False), (True, True)):
        risky = Risky(settings)
        runner, _, _ = make_runner(
            FakeLemonade(*asks_for_danger()), ModuleRegistry([risky]), settings
        )

        outcome = await runner.run(
            a_job(make_repo(), allow_unattended_tools=allowed)
            if allowed
            else a_job(SchedulerRepository(Database(":memory:")))
        )

        assert outcome.status == "ok"
        assert bool(risky.ran) is should_run


async def test_reaching_the_round_limit_adds_a_warning_to_the_chat():
    settings = Settings()
    settings.modules.scheduler.max_tool_rounds = 2
    call = ToolCall("c1", "danger", "{}")
    looping = [
        [ToolCallsRequested((call,)), Finished(RequestStats(), "tool_calls")],
        reply("Out of rounds."),
    ]
    risky = Risky(settings)
    runner, chats, _ = make_runner(FakeLemonade(*looping), ModuleRegistry([risky]), settings)

    outcome = await runner.run(a_job(make_repo(), allow_unattended_tools=True))

    assert outcome.status == "ok"
    last = chats.list_messages(outcome.session_id)[-1]  # type: ignore[arg-type]
    assert "maximum of 2 tool rounds" in last.content


# --- the simple schedule picker -----------------------------------------------------------------------------------


def test_picker_choices_become_cron_expressions():
    from lemonrind.modules.scheduler.cron import SimpleSchedule, simple_expression

    assert simple_expression(SimpleSchedule("daily", 9, 0)) == "0 9 * * *"
    assert simple_expression(SimpleSchedule("weekly", 19, 30, weekday=5)) == "30 19 * * 5"
    assert simple_expression(SimpleSchedule("monthly", 7, 5, day=15)) == "5 7 15 * *"
    with pytest.raises(ValueError):
        simple_expression(SimpleSchedule("hourly", 1, 0))


def test_every_picker_choice_is_a_valid_expression_that_reads_back_the_same():
    from lemonrind.modules.scheduler.cron import (
        SimpleSchedule,
        normalise,
        read_simple,
        simple_expression,
    )

    for schedule in (
        SimpleSchedule("daily", 0, 0),
        SimpleSchedule("daily", 23, 55),
        *(SimpleSchedule("weekly", 9, 0, weekday=d) for d in range(7)),
        SimpleSchedule("monthly", 12, 30, day=1),
        SimpleSchedule("monthly", 12, 30, day=31),
    ):
        expression = simple_expression(schedule)
        assert normalise(expression) == expression
        read = read_simple(expression)
        assert read is not None and simple_expression(read) == expression


@pytest.mark.parametrize(
    "expression",
    [
        "*/15 * * * *",
        "0 9 * * 1-5",
        "0 */2 * * *",
        "0 9 1 6 *",
        "7 9 * * *",
        "0 24 * * *",
        "0 9 1 * 1",
        "nonsense",
        "0 9 * *",
        "0 9 32 * *",
    ],
)
def test_expressions_the_picker_cannot_express_stay_custom(expression: str):
    from lemonrind.modules.scheduler.cron import read_simple

    assert read_simple(expression) is None


def test_sunday_can_be_written_as_zero_or_seven():
    from lemonrind.modules.scheduler.cron import SimpleSchedule, read_simple

    assert read_simple("0 9 * * 7") == SimpleSchedule("weekly", 9, 0, weekday=0)
    assert read_simple("0 9 * * 0") == SimpleSchedule("weekly", 9, 0, weekday=0)


# --- the run's chat exists while the job works, and a run can be stopped -----------------------------------------------------


class SlowLemonade(FakeLemonade):
    """Waits (like a model taking minutes) until told to go on, then answers."""

    def __init__(self) -> None:
        super().__init__(reply("All done."))
        self.started = asyncio.Event()
        self.go = asyncio.Event()

    async def stream_chat(self, messages, model, *, tools=None, max_tokens=None):
        self.started.set()
        await self.go.wait()
        async for event in super().stream_chat(messages, model, tools=tools, max_tokens=max_tokens):
            yield event


async def test_the_chat_is_in_the_list_while_the_job_runs_and_the_running_tag_goes_when_it_ends():
    client = SlowLemonade()
    runner, chats, _ = make_runner(client)
    task = asyncio.create_task(runner.run(a_job(make_repo())))
    await client.started.wait()

    (during,) = chats.list_sessions()  # already there, minutes before the answer
    assert during.title == "Weekly digest" and {"scheduled", "running"} <= set(during.tags)

    client.go.set()
    outcome = await task
    (after,) = chats.list_sessions()  # the same chat, not a second one
    assert after.id == during.id == outcome.session_id
    assert "running" not in after.tags and "scheduled" in after.tags
    assert [m.role for m in chats.list_messages(after.id)] == ["user", "assistant"]


async def test_stopping_a_run_leaves_one_chat_saying_so_and_clears_the_running_tag():
    client = SlowLemonade()
    runner, chats, _ = make_runner(client)
    task = asyncio.create_task(runner.run(a_job(make_repo())))
    await client.started.wait()

    task.cancel()
    outcome = await task  # the runner reports the stop instead of raising it again

    assert outcome.status == "stopped"
    (session,) = chats.list_sessions()
    assert "running" not in session.tags and "failed" not in session.tags
    assert [(m.role, m.content) for m in chats.list_messages(session.id)] == [
        ("user", "Summarise the week"),
        ("assistant", "This run was stopped before it finished."),
    ]


def test_a_chat_still_tagged_running_at_start_up_is_cleaned_up_with_a_note():
    runner, chats, _ = make_runner(FakeLemonade())
    stale = chats.create_session("Old job")
    chats.add_tag(stale.id, "running")

    runner.clear_running_tags()

    again = chats.get_session(stale.id)
    assert again is not None and "running" not in again.tags
    assert (
        chats.list_messages(stale.id)[-1].content == "The app stopped while this job was running."
    )


class StoppableRunner(FakeRunner):
    """Like the real runner, turns a cancellation into a 'stopped' outcome."""

    async def run(self, job) -> JobOutcome:
        try:
            return await super().run(job)
        except asyncio.CancelledError:
            return JobOutcome("stopped", "chat-stopped", "Stopped.")


async def test_the_stop_button_ends_the_running_job_and_records_it_as_stopped():
    module, repo, clock, _ = real_module(StoppableRunner(delay=30))
    job = repo.create("slow", "0 9 * * 1", "p", clock.now + timedelta(days=3))

    assert module.stop_job(job.id) is False  # nothing is running yet
    module.start_now(job.id)
    await wait_for(lambda: module.runs_started == 1)

    assert module.stop_job(job.id) is True
    await wait_for(lambda: module.runs_finished == 1)

    stored = repo.get(job.id)
    assert stored is not None and stored.last_status == "stopped" and stored.running_since is None
    assert module.last_outcome is not None and module.last_outcome[1].status == "stopped"
    assert module.stop_job(job.id) is False  # and nothing is left to stop
