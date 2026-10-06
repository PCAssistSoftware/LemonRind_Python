"""Watching a scheduled run while it works: the in-memory record of a run, and a page following it."""

from __future__ import annotations

import asyncio
from pathlib import Path

from nicegui.testing import User

from lemonrind.lemonade.events import PromptProgress, ReasoningDelta, TextDelta
from lemonrind.modules.scheduler.live import MAX_EVENTS, LiveRun, LiveRuns
from lemonrind.webui.context import set_context
from lemonrind.webui.page import register_pages
from tests.conftest import open_section  # noqa: F401  (kept next to the other UI helpers)
from tests.test_scheduler import SlowLemonade, a_job, make_repo, make_runner
from tests.test_scheduler_ui import scheduler
from tests.test_webui import FakeLemonade, make_context


def test_neighbouring_text_pieces_are_joined_and_only_the_latest_progress_is_kept():
    run = LiveRun("chat-1", "prompt")

    for piece in ("Hel", "lo ", "there"):
        run.publish(TextDelta(piece))
    run.publish(ReasoningDelta("hm"))
    run.publish(ReasoningDelta("m"))
    run.publish(PromptProgress(total=100, cache=0, processed=10, time_ms=1.0))
    run.publish(PromptProgress(total=100, cache=0, processed=60, time_ms=2.0))

    assert run.events[:2] == [TextDelta("Hello there"), ReasoningDelta("hmm")]
    assert len(run.events) == 3 and run.events[2].processed == 60  # type: ignore[attr-defined]


def test_a_very_long_run_keeps_only_its_most_recent_events():
    run = LiveRun("chat-1", "prompt")
    for number in range(MAX_EVENTS + 500):
        run.publish(PromptProgress(total=1, cache=0, processed=0, time_ms=0.0))
        run.publish(("marker", number))  # events that cannot be merged

    assert len(run.events) <= MAX_EVENTS
    assert run.events[-1] == ("marker", MAX_EVENTS + 499)  # the newest are the ones kept


def test_followers_get_every_event_and_can_stop_following():
    run = LiveRun("chat-1", "prompt")
    heard: list[object] = []
    stop = run.subscribe(heard.append, lambda: heard.append("done"))

    run.publish(TextDelta("a"))
    run.publish(TextDelta("b"))  # followers get the raw pieces, not the joined text
    stop()
    run.publish(TextDelta("c"))

    assert heard == [TextDelta("a"), TextDelta("b")]
    assert run.events == [TextDelta("abc")]


def test_a_follower_that_fails_does_not_disturb_the_run_or_the_others():
    run = LiveRun("chat-1", "prompt")
    heard: list[object] = []

    def broken(_event: object) -> None:
        raise RuntimeError("the page was closed")

    run.subscribe(broken, lambda: None)
    run.subscribe(heard.append, lambda: heard.append("done"))
    run.publish(TextDelta("still works"))
    run.finish()

    assert heard == [TextDelta("still works"), "done"]


def test_ending_a_run_tells_its_followers_and_forgets_it():
    runs = LiveRuns()
    run = runs.start("chat-1", "prompt")
    ended: list[str] = []
    run.subscribe(lambda event: None, lambda: ended.append("finished"))

    assert runs.get("chat-1") is run
    runs.end("chat-1")

    assert ended == ["finished"] and runs.get("chat-1") is None and run.finished
    runs.end("chat-1")  # ending twice is harmless


async def test_the_runner_streams_what_the_model_says_to_anyone_following():
    client = SlowLemonade()
    runner, chats, _ = make_runner(client)
    task = asyncio.create_task(runner.run(a_job(make_repo())))
    await client.started.wait()

    (session,) = chats.list_sessions()
    live = runner.live.get(session.id)
    assert live is not None and live.prompt == "Summarise the week"
    heard: list[object] = []
    finished: list[bool] = []
    live.subscribe(heard.append, lambda: finished.append(True))

    client.go.set()
    await task

    assert "All done." in "".join(e.text for e in heard if isinstance(e, TextDelta))
    assert finished == [True] and runner.live.get(session.id) is None


async def test_a_page_opened_on_a_running_job_shows_what_it_has_said_and_keeps_following(
    user: User, tmp_path: Path
):
    context = make_context(tmp_path, FakeLemonade())
    module = scheduler(context.modules)  # type: ignore[arg-type]
    running = context.repo.create_session("Nightly job")
    context.repo.add_tag(running.id, "scheduled")
    context.repo.add_tag(running.id, "running")
    live = module.live.start(running.id, "Write the nightly report")
    live.publish(TextDelta("Gathering the figures. "))
    set_context(context)
    register_pages()
    try:
        await user.open("/")
        await user.should_see("Lemonade healthy")
        user.find(marker="chat-item").click()

        await user.should_see("Write the nightly report")  # your prompt, as the first message
        await user.should_see("Gathering the figures.")  # what the job has said so far
        live.publish(TextDelta("Now writing the summary."))
        await asyncio.sleep(0.8)
        await user.should_see("Now writing the summary.")  # and what it says next, live
    finally:
        module.live.end(running.id)
        set_context(None)
