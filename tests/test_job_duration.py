"""A scheduled run says when it ended and how long it took: in its chat, in the job list and in the pop-up."""

from __future__ import annotations

import asyncio
import re

import pytest

from lemonrind.chats import format_duration
from lemonrind.modules.scheduler import JobOutcome
from tests.test_scheduler import FakeLemonade as SchedulerLemonade
from tests.test_scheduler import (
    FakeRunner,
    SlowLemonade,
    a_job,
    make_module,
    make_repo,
    make_runner,
    reply,
    run_messages,
)

END_NOTE = re.compile(r"^This run ended at \d\d:\d\d on \d\d/\d\d/\d{4} and took (.+)\.$")


@pytest.mark.parametrize(
    ("seconds", "words"),
    [
        (0, "0 s"),
        (0.4, "0 s"),
        (45, "45 s"),
        (60, "1 min 0 s"),
        (760, "12 min 40 s"),
        (3600, "1 h 0 min"),
        (7500, "2 h 5 min"),
        (-3, "0 s"),
    ],
)
def test_a_duration_is_written_the_way_a_person_would_say_it(seconds: float, words: str):
    assert format_duration(seconds) == words


async def test_a_finished_run_ends_its_chat_with_when_it_ended_and_how_long_it_took():
    runner, chats, _ = make_runner(SchedulerLemonade(reply("Here is your digest.")))

    outcome = await runner.run(a_job(make_repo()))

    assert outcome.seconds is not None and outcome.seconds >= 0
    assert run_messages(chats, outcome.session_id or "")[-1].content == "Here is your digest."
    closing = chats.list_messages(outcome.session_id or "")[-1]
    match = END_NOTE.match(closing.content)
    assert match is not None and closing.role == "assistant"
    assert (
        "ended at 19:00 on 02/10/2026" in closing.content
    )  # the test clock; dd/mm/yyyy as the user writes dates


async def test_a_failed_run_also_says_how_long_it_took():
    runner, chats, _ = make_runner(SchedulerLemonade(RuntimeError("boom")))

    outcome = await runner.run(a_job(make_repo()))

    assert outcome.status == "failed" and outcome.seconds is not None
    assert END_NOTE.match(chats.list_messages(outcome.session_id or "")[-1].content)


async def test_a_stopped_run_says_how_long_it_ran_before_it_was_stopped():
    client = SlowLemonade()
    runner, chats, _ = make_runner(client)
    task = asyncio.create_task(runner.run(a_job(make_repo())))
    await client.started.wait()
    task.cancel()

    outcome = await task

    assert outcome.status == "stopped"
    messages = chats.list_messages(outcome.session_id or "")
    assert messages[-2].content == "This run was stopped before it finished."
    assert END_NOTE.match(messages[-1].content)


async def test_the_module_saves_how_long_the_last_run_took():
    class TimedRunner(FakeRunner):
        async def run(self, job) -> JobOutcome:
            return JobOutcome("ok", "chat", "done", seconds=125.0)

    module, repo, _, _ = make_module(TimedRunner())
    job = repo.create("daily", "0 9 * * *", "do it", None)

    await module.run_now(job.id)

    saved = repo.get(job.id)
    assert saved is not None and saved.last_duration_seconds == 125.0
