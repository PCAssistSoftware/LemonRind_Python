"""The plan tool: the checklist itself, jobs being held to it, and the screen showing it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from nicegui.testing import User

from lemonrind.chats import ChatRepository, Conversation
from lemonrind.config import Settings
from lemonrind.lemonade import ChatEvent, Finished, ToolCall, ToolCallsRequested
from lemonrind.lemonade.events import RequestStats
from lemonrind.modules import ModuleRegistry
from lemonrind.modules.plan import (
    MAX_PLAN_NUDGES,
    MAX_PLAN_ONLY_ROUNDS,
    PLAN_TOOL,
    PlanModule,
    build_plan,
    plan_from_arguments,
)
from lemonrind.modules.tool import ToolError
from lemonrind.storage import Database
from lemonrind.webui.context import set_context
from lemonrind.webui.page import register_pages
from tests.test_scheduler import FakeLemonade as SchedulerLemonade
from tests.test_scheduler import a_job, make_repo, make_runner, run_messages
from tests.test_tool_loop import RoundsStreamer, answer_round
from tests.test_webui import FakeLemonade, make_context, send


def plan_call(steps: list[str], done: list[int] | None = None, current: int = 0) -> ToolCall:
    arguments = {"steps": steps, "done": done or [], "current": current}
    return ToolCall(f"plan-{len(steps)}-{len(done or [])}", PLAN_TOOL, json.dumps(arguments))


def plan_round(call: ToolCall) -> list[ChatEvent]:
    return [ToolCallsRequested((call,)), Finished(RequestStats(output_tokens=3), "tool_calls")]


STEPS = ["Read the reports", "Compare the figures", "Write the summary"]


# --- the checklist ---------------------------------------------------------------------------------------------------


def test_a_plan_shows_what_is_done_and_what_is_next():
    plan = build_plan(STEPS, (1,), 2)
    assert plan.render() == (
        "Plan (1 of 3 done):\n1. [done] Read the reports\n2. [now] Compare the figures\n"
        "3. [ ] Write the summary"
    )
    assert plan.unfinished == [2, 3] and not plan.complete
    assert build_plan(STEPS, (1, 2, 3)).complete


@pytest.mark.parametrize(
    ("steps", "done", "current"),
    [([], (), 0), (["a", " "], (), 0), (["a"] * 13, (), 0), (["a", "b"], (3,), 0), (["a"], (), 2)],
)
def test_a_plan_that_makes_no_sense_is_refused_with_a_reason(steps, done, current):
    with pytest.raises(ToolError):
        build_plan(steps, done, current)


def test_the_tool_answers_with_the_plan_and_what_to_do_next():
    tool = PlanModule(Settings()).get_tools()[0]
    assert tool.name == PLAN_TOOL
    text = PlanModule(Settings()).update_plan(STEPS, (1,), 0)
    assert "Plan (1 of 3 done)" in text and "Carry on with step 2" in text
    assert "Give your final answer now" in PlanModule(Settings()).update_plan(["a"], (1,), 0)


def test_the_plan_is_read_back_from_a_call_and_nonsense_gives_none():
    call = plan_call(STEPS, [1], 2)
    plan = plan_from_arguments(call.arguments)
    assert plan is not None and plan.current == 2 and plan.done == {1}
    assert plan_from_arguments("not json") is None
    assert plan_from_arguments('{"steps": []}') is None


# --- a job is held to its plan ---------------------------------------------------------------------------------------


class PlanTools:
    """Runs the real plan tool, the way the registry would."""

    def __init__(self) -> None:
        self.tool = PlanModule(Settings()).get_tools()[0]

    def requires_approval(self, call: ToolCall) -> bool:
        return False

    def schemas(self) -> list[dict]:
        return [self.tool.schema()]

    async def run(self, call: ToolCall):
        return await self.tool.run(call.arguments)


def conversation_with(streamer, *, enforce: bool) -> Conversation:
    conversation = Conversation(
        client=streamer,
        repo=ChatRepository(Database(":memory:")),
        system_prompt="Be brief.",
        model="m",
        tools=PlanTools(),
        max_rounds=20,
    )
    conversation.enforce_plan = enforce
    return conversation


async def test_a_job_that_stops_with_steps_open_is_sent_back_to_them():
    streamer = RoundsStreamer(
        plan_round(plan_call(STEPS, [1], 2)),
        answer_round("All done!"),  # too early: steps 2 and 3 are open
        plan_round(plan_call(STEPS, [1, 2, 3])),
        answer_round("Here is the summary."),
    )
    conversation = conversation_with(streamer, enforce=True)

    reply = await conversation.send("Do the weekly review")

    assert reply.text == "Here is the summary."
    reminder = streamer.requests[2][-1]
    assert reminder["role"] == "user" and "unfinished steps: 2, 3" in reminder["content"]
    assert conversation.plan is not None and conversation.plan.complete
    assert conversation.plan_nudges == 1


async def test_the_reminders_stop_after_a_few_tries_and_the_plan_stays_open():
    early = [answer_round("Done.") for _ in range(MAX_PLAN_NUDGES + 1)]
    streamer = RoundsStreamer(plan_round(plan_call(STEPS, [1], 2)), *early)
    conversation = conversation_with(streamer, enforce=True)

    reply = await conversation.send("Do the weekly review")

    assert reply.text == "Done."
    assert conversation.plan_nudges == MAX_PLAN_NUDGES
    assert conversation.plan is not None and not conversation.plan.complete


async def test_an_ordinary_chat_is_never_sent_back_to_its_plan():
    streamer = RoundsStreamer(plan_round(plan_call(STEPS, [1], 2)), answer_round("Quick answer."))
    conversation = conversation_with(streamer, enforce=False)

    reply = await conversation.send("Do the weekly review")

    assert reply.text == "Quick answer." and conversation.plan_nudges == 0
    assert len(streamer.requests) == 2


async def test_a_model_that_only_edits_its_plan_is_made_to_answer():
    rounds = [
        plan_round(plan_call(STEPS, [1], 2)) for _ in range(MAX_PLAN_ONLY_ROUNDS)
    ]  # the same edit over and over
    streamer = RoundsStreamer(*rounds, answer_round("Enough planning."))
    conversation = conversation_with(streamer, enforce=False)

    reply = await conversation.send("Plan it")

    assert reply.text == "Enough planning."
    assert (
        streamer.tools_offered[MAX_PLAN_ONLY_ROUNDS] is None
    )  # tools withheld on the last request


# --- the scheduled job runner ----------------------------------------------------------------------------------------


async def test_a_job_whose_plan_is_left_unfinished_is_marked_incomplete():
    settings = Settings()
    steps_open = plan_round(plan_call(STEPS, [1], 2))
    client = SchedulerLemonade(steps_open, *[answer_round("Done.") for _ in range(5)])
    runner, chats, _ = make_runner(client, ModuleRegistry([PlanModule(settings)]), settings)

    outcome = await runner.run(a_job(make_repo()))

    assert outcome.status == "incomplete" and outcome.session_id is not None
    last = run_messages(chats, outcome.session_id)[-1]
    assert "2 of the 3 steps of its own plan not done (steps 2, 3)" in last.content
    system = client.requests[0]["messages"][0]["content"]
    assert (
        f"a call to {PLAN_TOOL}" in system
    )  # the model is told to make a plan for a multi-step job


async def test_a_job_that_finishes_its_plan_is_ok():
    settings = Settings()
    client = SchedulerLemonade(
        plan_round(plan_call(STEPS, [1], 2)),
        plan_round(plan_call(STEPS, [1, 2, 3])),
        answer_round("The weekly digest."),
    )
    runner, _, _ = make_runner(client, ModuleRegistry([PlanModule(settings)]), settings)

    outcome = await runner.run(a_job(make_repo()))

    assert outcome.status == "ok"


async def test_with_the_plan_module_off_a_job_is_not_told_to_plan():
    settings = Settings()
    settings.modules.enabled["plan"] = False
    client = SchedulerLemonade(answer_round("Digest."))
    runner, _, _ = make_runner(client, ModuleRegistry([PlanModule(settings)]), settings)

    outcome = await runner.run(a_job(make_repo()))

    assert outcome.status == "ok"
    assert PLAN_TOOL not in client.requests[0]["messages"][0]["content"]


# --- the screen ------------------------------------------------------------------------------------------------------


class PlanningLemonade(FakeLemonade):
    """Plans on the first request, answers on the second."""

    async def stream_chat(self, messages, model, **kwargs):
        self.requests.append([dict(m) for m in messages])
        events = (
            plan_round(plan_call(STEPS, [1], 2))
            if len(self.requests) == 1
            else answer_round("Reviewed.")
        )
        for event in events:
            yield event


async def test_the_stats_panel_shows_the_plan_the_assistant_wrote(user: User, tmp_path: Path):
    lemonade = PlanningLemonade()
    context = make_context(tmp_path, lemonade)
    set_context(context)
    register_pages()
    try:
        await user.open("/")
        await user.should_see("Lemonade healthy")
        await send(user, "Do the weekly review")
        await user.should_see("Plan (1 of 3 done)")
        await user.should_see("Compare the figures")
    finally:
        set_context(None)
