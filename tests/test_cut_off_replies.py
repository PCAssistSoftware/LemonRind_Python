"""A reply that stops at the output limit: a scheduled run says so instead of "ok", and the limits can be set."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path

from nicegui.testing import User

from lemonrind.chats import CONTINUE_PROMPT
from lemonrind.config import SchedulerSettings
from lemonrind.lemonade import ChatEvent, Finished, TextDelta
from lemonrind.lemonade.events import RequestStats
from lemonrind.webui.context import set_context
from lemonrind.webui.page import register_pages
from tests.conftest import elements
from tests.test_scheduler import FakeLemonade as SchedulerLemonade
from tests.test_scheduler import a_job, make_repo, make_runner, reply
from tests.test_settings_sections import open_section, settle
from tests.test_webui import FakeLemonade, make_context, send


def cut_off(text: str) -> list[ChatEvent]:
    """A reply that ends because it hit the token limit, not because the model finished."""
    return [
        TextDelta(text),
        Finished(RequestStats(prompt_tokens=10, output_tokens=16384), "length"),
    ]


# --- a scheduled run -------------------------------------------------------------------------------------------------------


async def test_a_run_whose_reply_hit_the_output_limit_is_marked_cut_short_with_a_note():
    runner, chats, settings = make_runner(
        SchedulerLemonade(cut_off("<table><tr><td>half a report"))
    )
    settings.modules.scheduler.max_continuations = 0

    outcome = await runner.run(a_job(make_repo()))

    assert outcome.status == "cut short" and outcome.session_id is not None
    last = chats.list_messages(outcome.session_id)[-1]
    assert last.role == "assistant"
    assert (
        f"cut off at the output limit ({settings.modules.scheduler.max_output_tokens:,} tokens"
        in last.content
    )
    assert "includes its thinking" in last.content and "Longest reply" in last.content


async def test_a_reply_that_finished_normally_is_still_ok():
    runner, chats, _ = make_runner(SchedulerLemonade(reply("All done.")))

    outcome = await runner.run(a_job(make_repo()))

    assert outcome.status == "ok"
    assert not any("cut off" in m.content for m in chats.list_messages(outcome.session_id or ""))


async def test_a_cut_off_reply_is_continued_automatically_and_the_run_is_ok():
    client = SchedulerLemonade(cut_off("<table><tr><td>first half"), reply("</td></tr></table>"))
    runner, chats, _ = make_runner(client)

    outcome = await runner.run(a_job(make_repo()))

    assert outcome.status == "ok" and outcome.session_id is not None
    saved = [(m.role, m.content) for m in chats.list_messages(outcome.session_id)]
    assert saved[:4] == [
        ("user", "Summarise the week"),
        ("assistant", "<table><tr><td>first half"),
        ("user", CONTINUE_PROMPT),
        ("assistant", "</td></tr></table>"),
    ]
    assert "continued automatically 1 time" in saved[-1][1] and "2 parts" in saved[-1][1]
    # the second request carried the first half, so the model can carry on from it
    assert {"role": "assistant", "content": "<table><tr><td>first half"} in client.requests[1][
        "messages"
    ]


async def test_continuing_stops_after_the_set_number_of_tries_and_says_so():
    client = SchedulerLemonade(
        cut_off("one "), cut_off("two "), cut_off("three "), reply("never asked")
    )
    runner, chats, settings = make_runner(client)

    outcome = await runner.run(a_job(make_repo()))

    assert settings.modules.scheduler.max_continuations == 2
    assert (
        outcome.status == "cut short" and len(client.requests) == 3
    )  # the first try and two continues
    last = chats.list_messages(outcome.session_id or "")[-1].content
    assert "even after 2 automatic continues" in last


async def test_with_continuing_off_a_cut_off_reply_is_not_asked_again():
    client = SchedulerLemonade(cut_off("half"), reply("never asked"))
    runner, _, settings = make_runner(client)
    settings.modules.scheduler.max_continuations = 0

    outcome = await runner.run(a_job(make_repo()))

    assert outcome.status == "cut short" and len(client.requests) == 1


async def test_a_continue_that_only_thinks_leaves_the_run_cut_short_not_failed():
    thinking_only = [Finished(RequestStats(prompt_tokens=10, output_tokens=16384), "length")]
    client = SchedulerLemonade(cut_off("the report so far"), thinking_only)
    runner, chats, _ = make_runner(client)

    outcome = await runner.run(a_job(make_repo()))

    assert (
        outcome.status == "cut short"
    )  # the report so far is kept, and the run says it is incomplete
    assert chats.list_messages(outcome.session_id or "")[1].content == "the report so far"


def test_the_default_longest_reply_leaves_room_for_thinking_and_a_report():
    limits = SchedulerSettings()
    assert (limits.max_output_tokens, limits.max_tool_rounds, limits.job_timeout_seconds) == (
        32768,
        100,
        1800.0,
    )


# --- the three boxes in Settings > Scheduler ---------------------------------------------------------------------------------


async def test_the_scheduler_limits_are_shown_changed_and_saved_at_once(user: User, tmp_path: Path):
    context = make_context(tmp_path, FakeLemonade())
    set_context(context)
    register_pages()
    try:
        await user.open("/")
        await open_section(user, "scheduler")
        (minutes,) = elements(user, "job-time-limit")
        (rounds,) = elements(user, "job-max-rounds")
        (continues,) = elements(user, "job-max-continues")
        assert continues.value == 2
        continues.value = 4
        (reply_box,) = elements(user, "job-max-reply")
        assert (minutes.value, rounds.value, reply_box.value) == (30, 100, 32768)

        rounds.value = 250
        reply_box.value = 65536
        for _ in range(40):
            if context.settings.modules.scheduler.max_output_tokens == 65536:
                break
            await asyncio.sleep(0.05)

        limits = context.settings.modules.scheduler
        assert (limits.max_tool_rounds, limits.max_output_tokens) == (250, 65536)
        assert limits.max_continuations == 4
        saved = json.loads(context.settings_file.read_text(encoding="utf-8"))["modules"][
            "scheduler"
        ]
        assert (saved["max_tool_rounds"], saved["max_output_tokens"]) == (250, 65536)

        rounds.value = 0  # outside the allowed range: ignored
        reply_box.value = 99999999
        await asyncio.sleep(0.2)
        assert (limits.max_tool_rounds, limits.max_output_tokens) == (250, 65536)
    finally:
        set_context(None)


# --- the chat -------------------------------------------------------------------------------------------------------------------


class CutOffLemonade(FakeLemonade):
    """Cuts the first reply off at the limit; any later reply is complete."""

    async def stream_chat(self, messages, model, **kwargs) -> AsyncIterator[ChatEvent]:
        self.requests.append([dict(m) for m in messages])
        if len(self.requests) == 1:
            events = cut_off("The beginning of a long answer that never")
        else:
            events = reply("...and here is the end.")
        for event in events:
            yield event


async def test_a_chat_reply_that_hit_the_output_limit_says_it_may_be_incomplete(
    user: User, tmp_path: Path
):
    context = make_context(tmp_path, CutOffLemonade())
    set_context(context)
    register_pages()
    try:
        await user.open("/")
        await user.should_see("Lemonade healthy")
        await settle()

        await send(user, "Write me a very long report")

        await user.should_see("The beginning of a long answer")
        await user.should_see("may be incomplete")
    finally:
        set_context(None)


async def test_the_continue_button_asks_the_model_to_carry_on_and_then_goes_away(
    user: User, tmp_path: Path
):
    lemonade = CutOffLemonade()
    context = make_context(tmp_path, lemonade)
    set_context(context)
    register_pages()
    try:
        await user.open("/")
        await user.should_see("Lemonade healthy")
        await settle()
        await send(user, "Write me a very long report")
        await user.should_see("may be incomplete")
        (button,) = elements(user, "continue-reply")
        assert button.visible

        user.find(marker="continue-reply").click()
        await user.should_see("...and here is the end.")

        assert button.visible is False  # pressed, so gone
        last = lemonade.requests[1][-1]  # (the page may add the time in front of it)
        assert last["role"] == "user" and last["content"].endswith(CONTINUE_PROMPT)
        await user.should_not_see(
            marker="continue-reply"
        )  # and the new reply finished normally, so it has none
    finally:
        set_context(None)
