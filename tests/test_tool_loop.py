"""Tests for the tool loop inside Conversation: tools run, results go back, the turn is saved as one unit."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import pytest

from lemonrind.chats import ChatRepository, Conversation
from lemonrind.lemonade import (
    ChatEvent,
    Finished,
    TextDelta,
    ToolCall,
    ToolCallsRequested,
)
from lemonrind.lemonade.events import RequestStats
from lemonrind.modules import ToolResult
from lemonrind.storage import Database


class RoundsStreamer:
    """Answers each request with the next list of events, and remembers what it was sent."""

    def __init__(self, *rounds: list[ChatEvent]) -> None:
        self.rounds = list(rounds)
        self.requests: list[list[dict]] = []
        self.tools_offered: list[object] = []

    async def stream_chat(self, messages, model, *, tools=None) -> AsyncIterator[ChatEvent]:
        self.requests.append(json.loads(json.dumps(list(messages))))  # a snapshot, not a live list
        self.tools_offered.append(tools)
        for event in self.rounds.pop(0):
            yield event


class FakeTools:
    def __init__(self, *, stall: bool = False, needs_approval: bool = False) -> None:
        self.stall = stall
        self.needs_approval = needs_approval
        self.ran: list[ToolCall] = []

    def requires_approval(self, call: ToolCall) -> bool:
        return self.needs_approval

    def schemas(self) -> list[dict]:
        return [{"type": "function", "function": {"name": "search"}}]

    async def run(self, call: ToolCall) -> ToolResult:
        self.ran.append(call)
        if self.stall:
            await asyncio.sleep(3600)
        return ToolResult(f"result for {call.arguments}")


SEARCH = ToolCall("call-1", "search", '{"q": "otters"}')
SEARCH_RESULT = 'result for {"q": "otters"}'


def tool_round(text: str = "", *, stats: RequestStats | None = None) -> list[ChatEvent]:
    events: list[ChatEvent] = [TextDelta(text)] if text else []
    return [
        *events,
        ToolCallsRequested((SEARCH,)),
        Finished(stats or RequestStats(output_tokens=3), "tool_calls"),
    ]


def answer_round(text: str, *, stats: RequestStats | None = None) -> list[ChatEvent]:
    return [TextDelta(text), Finished(stats or RequestStats(output_tokens=5), "stop")]


def make(streamer, tools=None, **kwargs) -> tuple[Conversation, ChatRepository]:
    repo = ChatRepository(Database(":memory:"))
    conversation = Conversation(
        client=streamer,
        repo=repo,
        system_prompt="Be brief.",
        model="m",
        tools=tools or FakeTools(),
        **kwargs,
    )
    return conversation, repo


async def test_tools_are_run_and_their_results_sent_back():
    streamer = RoundsStreamer(tool_round("Let me look. "), answer_round("Otters are mammals."))
    tools = FakeTools()
    conversation, _ = make(streamer, tools)
    seen: list = []

    reply = await conversation.send("Tell me about otters", seen.append)

    assert reply.text == "Otters are mammals."
    assert tools.ran == [SEARCH]
    # the second request contains the assistant's tool call and the tool's result
    second = streamer.requests[1]
    assert second[-2]["role"] == "assistant"
    assert second[-2]["tool_calls"][0]["id"] == "call-1"
    assert second[-2]["tool_calls"][0]["function"] == {
        "name": "search",
        "arguments": '{"q": "otters"}',
    }
    assert second[-1] == {"role": "tool", "tool_call_id": "call-1", "content": SEARCH_RESULT}
    # the front end was told about the tool run, in order, before the final answer
    kinds = [type(e).__name__ for e in seen]
    assert kinds.index("ToolStarted") < kinds.index("ToolFinished") < len(kinds) - 1
    assert streamer.tools_offered[0] == tools.schemas()


async def test_the_whole_turn_is_saved_and_reopens_with_the_same_history():
    streamer = RoundsStreamer(tool_round("Let me look. "), answer_round("Otters are mammals."))
    conversation, repo = make(streamer)

    await conversation.send("Tell me about otters")

    (session,) = repo.list_sessions()
    stored = repo.list_messages(session.id)
    assert [(m.role, m.content) for m in stored] == [
        ("user", "Tell me about otters"),
        ("assistant", "Let me look."),
        ("tool", SEARCH_RESULT),
        ("assistant", "Otters are mammals."),
    ]
    assert stored[1].tool_calls == (SEARCH,)
    assert stored[2].tool_call_id == "call-1"

    reopened = Conversation(
        client=RoundsStreamer(), repo=repo, system_prompt="Be brief.", model="m"
    )
    reopened.open(session)
    assert reopened.history[1:] == conversation.history[1:]  # rebuilt exactly as it was sent


async def test_stats_of_every_request_are_reported():
    first = RequestStats(output_tokens=3, prompt_tokens=100)
    last = RequestStats(output_tokens=5, prompt_tokens=180)
    streamer = RoundsStreamer(tool_round(stats=first), answer_round("Done.", stats=last))
    conversation, _ = make(streamer)

    reply = await conversation.send("go")

    assert reply.stats == last
    assert reply.round_stats == (first, last)


async def test_the_last_allowed_round_offers_no_tools():
    streamer = RoundsStreamer(tool_round(), tool_round(), answer_round("Final answer."))
    conversation, _ = make(streamer, max_rounds=3)

    reply = await conversation.send("go")

    assert reply.text == "Final answer."
    assert [bool(t) for t in streamer.tools_offered] == [True, True, False]


async def test_a_tool_request_on_the_last_round_is_treated_as_the_answer():
    streamer = RoundsStreamer(tool_round("Still calling tools."))
    tools = FakeTools()
    conversation, _ = make(streamer, tools, max_rounds=1)

    reply = await conversation.send("go")

    assert reply.text == "Still calling tools." and tools.ran == []


async def test_a_failure_in_a_later_round_discards_the_whole_turn():
    streamer = RoundsStreamer(
        tool_round("Let me look.")
    )  # the second request has no round: IndexError
    conversation, repo = make(streamer)

    with pytest.raises(IndexError):
        await conversation.send("go")

    assert len(conversation.history) == 1  # only the system prompt
    assert repo.list_sessions() == []


async def test_stopping_during_a_tool_discards_the_turn():
    conversation, repo = make(RoundsStreamer(tool_round("Looking.")), FakeTools(stall=True))

    task = asyncio.create_task(conversation.send("go"))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(conversation.history) == 1 and repo.list_sessions() == []


async def test_stopping_the_final_answer_keeps_the_tool_work_and_the_partial_text():
    class StopsInRoundTwo:
        def __init__(self) -> None:
            self.calls = 0

        async def stream_chat(self, messages, model, *, tools=None):
            self.calls += 1
            if self.calls == 1:
                for event in tool_round("Looking. "):
                    yield event
            else:
                yield TextDelta("Otters are")
                await asyncio.sleep(3600)

    conversation, repo = make(StopsInRoundTwo())
    task = asyncio.create_task(conversation.send("go"))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    (session,) = repo.list_sessions()
    assert [(m.role, m.content) for m in repo.list_messages(session.id)] == [
        ("user", "go"),
        ("assistant", "Looking."),
        ("tool", SEARCH_RESULT),
        ("assistant", "Otters are"),
    ]


async def test_broken_json_arguments_are_not_sent_back_to_the_server():
    bad = ToolCall("call-9", "search", "{broken")
    streamer = RoundsStreamer(
        [ToolCallsRequested((bad,)), Finished(RequestStats(), "tool_calls")], answer_round("ok")
    )
    conversation, _ = make(streamer)

    await conversation.send("go")

    assert streamer.requests[1][-2]["tool_calls"][0]["function"]["arguments"] == "{}"


async def test_search_ignores_tool_results():
    streamer = RoundsStreamer(tool_round(), answer_round("Nothing special."))
    conversation, repo = make(streamer)

    await conversation.send("go")

    assert repo.search("otters") == []  # "otters" only appears in the tool call result
    assert len(repo.search("special")) == 1


# --- permission to run a tool ----------------------------------------------------------------------------------


async def test_a_tool_that_needs_approval_runs_only_when_the_user_agrees():
    asked: list[ToolCall] = []

    async def approver(call: ToolCall) -> bool:
        asked.append(call)
        return True

    tools = FakeTools(needs_approval=True)
    conversation, _ = make(RoundsStreamer(tool_round(), answer_round("Done.")), tools)
    conversation.approve = approver

    reply = await conversation.send("go")

    assert asked == [SEARCH] and tools.ran == [SEARCH] and reply.text == "Done."


async def test_a_refused_tool_does_not_run_and_the_model_is_told_why():
    async def approver(call: ToolCall) -> bool:
        return False

    streamer = RoundsStreamer(tool_round(), answer_round("Understood."))
    tools = FakeTools(needs_approval=True)
    conversation, _ = make(streamer, tools)
    conversation.approve = approver
    seen: list = []

    reply = await conversation.send("go", seen.append)

    assert tools.ran == [] and reply.text == "Understood."
    refusal = streamer.requests[1][-1]
    assert refusal["role"] == "tool" and "did not allow" in refusal["content"]
    finished = next(e for e in seen if type(e).__name__ == "ToolFinished")
    assert finished.result.is_error


async def test_without_an_approver_a_tool_that_needs_permission_is_refused():
    tools = FakeTools(needs_approval=True)
    conversation, _ = make(RoundsStreamer(tool_round(), answer_round("ok")), tools)

    await conversation.send("go")

    assert tools.ran == []


async def test_a_broken_approver_counts_as_a_refusal():
    async def approver(call: ToolCall) -> bool:
        raise RuntimeError("dialog crashed")

    tools = FakeTools(needs_approval=True)
    conversation, _ = make(RoundsStreamer(tool_round(), answer_round("ok")), tools)
    conversation.approve = approver

    await conversation.send("go")

    assert tools.ran == []


async def test_tools_that_do_not_need_approval_never_ask():
    asked = []

    async def approver(call: ToolCall) -> bool:
        asked.append(call)
        return False

    tools = FakeTools()
    conversation, _ = make(RoundsStreamer(tool_round(), answer_round("ok")), tools)
    conversation.approve = approver

    await conversation.send("go")

    assert asked == [] and tools.ran == [SEARCH]


async def test_stopping_while_waiting_for_approval_discards_the_turn():
    async def approver(call: ToolCall) -> bool:
        await asyncio.sleep(3600)
        return True

    tools = FakeTools(needs_approval=True)
    conversation, repo = make(RoundsStreamer(tool_round("Looking.")), tools)
    conversation.approve = approver

    task = asyncio.create_task(conversation.send("go"))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert tools.ran == [] and repo.list_sessions() == []


# --- automatic tags ---------------------------------------------------------------------------------------------------


class OriginTools(FakeTools):
    """Tools that can say which module a tool belongs to (the real registry can)."""

    def module_for_tool(self, name: str):
        from types import SimpleNamespace

        return SimpleNamespace(config_key="mcp") if name.startswith("demo__") else None


@pytest.mark.parametrize(
    ("tool", "tag"),
    [("generate_image", "image"), ("code_edit_file", "coder"), ("demo__send_email", "mcp")],
)
async def test_a_chat_is_tagged_by_the_kind_of_tool_it_used(tool: str, tag: str):
    call = ToolCall("c1", tool, "{}")
    streamer = RoundsStreamer(
        [ToolCallsRequested((call,)), Finished(RequestStats(), "tool_calls")], answer_round("Done.")
    )
    conversation, repo = make(streamer, OriginTools())

    await conversation.send("do it")

    (session,) = repo.list_sessions()
    assert session.tags == (tag,)


async def test_an_ordinary_tool_or_no_tool_leaves_a_chat_untagged_and_a_chat_can_collect_several_tags():
    streamer = RoundsStreamer(tool_round(), answer_round("Found it."))
    conversation, repo = make(streamer, OriginTools())
    await conversation.send("search")
    assert repo.list_sessions()[0].tags == ()

    both = RoundsStreamer(
        [
            ToolCallsRequested(
                (ToolCall("a", "code_read_file", "{}"), ToolCall("b", "generate_image", "{}"))
            ),
            Finished(RequestStats(), "tool_calls"),
        ],
        answer_round("Both."),
    )
    conversation, repo = make(both, OriginTools())
    await conversation.send("do both")
    assert set(repo.list_sessions()[0].tags) == {"coder", "image"}
