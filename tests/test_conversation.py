"""Tests for Conversation: the shared send / stream / save logic, including the Stop button's behaviour."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from lemonrind.chats import ChatRepository, Conversation
from lemonrind.lemonade import ChatEvent, Finished, PromptProgress, ReasoningDelta, TextDelta
from lemonrind.lemonade.client import LemonadeError
from lemonrind.lemonade.events import RequestStats
from lemonrind.storage import Database


class ScriptedStreamer:
    """Plays back a fixed list of events. It matches the ChatStreamer protocol without inheriting from it."""

    def __init__(
        self, events: list[ChatEvent] | None = None, error: Exception | None = None
    ) -> None:
        self.events: list[ChatEvent] = (
            events if events is not None else [TextDelta("Hi"), Finished(RequestStats(), "stop")]
        )
        self.error = error
        self.requests: list[list[dict[str, str]]] = []

    async def stream_chat(self, messages, model, **kwargs) -> AsyncIterator[ChatEvent]:
        self.requests.append([dict(m) for m in messages])
        if self.error:
            raise self.error
        for event in self.events:
            yield event


class StallingStreamer:
    """Sends two pieces of text and then never finishes, like a long reply the user stops."""

    async def stream_chat(self, messages, model, **kwargs) -> AsyncIterator[ChatEvent]:
        yield TextDelta("Hel")
        yield TextDelta("lo")
        await asyncio.sleep(3600)
        yield Finished(RequestStats(), "stop")  # never reached


def make(streamer) -> tuple[Conversation, ChatRepository]:
    repo = ChatRepository(Database(":memory:"))
    return Conversation(client=streamer, repo=repo, system_prompt="Be brief.", model="m"), repo


async def test_a_reply_is_streamed_to_the_callback_and_saved():
    events = [
        PromptProgress(total=10, cache=0, processed=10, time_ms=5),
        ReasoningDelta("Hmm. "),
        TextDelta("Hel"),
        TextDelta("lo"),
        Finished(RequestStats(output_tokens=2), "stop"),
    ]
    conversation, repo = make(ScriptedStreamer(events))
    seen: list[ChatEvent] = []

    reply = await conversation.send("Hi there", seen.append)

    assert seen == events  # every event reached the front end, in order
    assert (reply.text, reply.reasoning, reply.finish_reason) == ("Hello", "Hmm.", "stop")
    assert reply.stats is not None and reply.stats.output_tokens == 2
    (session,) = repo.list_sessions()
    assert conversation.session is not None and conversation.session.id == session.id
    assert [(m.role, m.content) for m in repo.list_messages(session.id)] == [
        ("user", "Hi there"),
        ("assistant", "Hello"),
    ]


async def test_the_model_is_sent_the_system_prompt_and_the_whole_conversation():
    streamer = ScriptedStreamer()
    conversation, _ = make(streamer)
    await conversation.send("one")
    await conversation.send("two")
    assert [m["role"] for m in streamer.requests[1]] == ["system", "user", "assistant", "user"]
    assert streamer.requests[0][0] == {"role": "system", "content": "Be brief."}


async def test_changing_the_system_prompt_applies_to_the_next_message():
    streamer = ScriptedStreamer()
    conversation, _ = make(streamer)
    await conversation.send("one")
    conversation.set_system_prompt("Be silly.")
    await conversation.send("two")
    assert streamer.requests[1][0]["content"] == "Be silly."


async def test_a_failed_request_saves_nothing_and_forgets_the_question():
    conversation, repo = make(ScriptedStreamer(error=LemonadeError("down")))
    with pytest.raises(LemonadeError):
        await conversation.send("hello?")
    assert repo.list_sessions() == []
    assert conversation.history == [{"role": "system", "content": "Be brief."}]


async def test_a_reply_with_no_answer_is_returned_but_not_saved():
    events = [ReasoningDelta("thinking..."), Finished(RequestStats(), "length")]
    conversation, repo = make(ScriptedStreamer(events))
    reply = await conversation.send("hard question")
    assert reply.text == "" and reply.finish_reason == "length"
    assert repo.list_sessions() == [] and not conversation.is_saved


async def test_stop_keeps_the_part_of_the_answer_that_arrived():
    conversation, repo = make(StallingStreamer())
    arrived = asyncio.Event()

    def on_event(event: ChatEvent) -> None:
        if isinstance(event, TextDelta) and event.text == "lo":
            arrived.set()

    task = asyncio.create_task(conversation.send("Say hello", on_event))
    await arrived.wait()
    task.cancel()  # what the Stop button does
    with pytest.raises(asyncio.CancelledError):
        await task

    (session,) = repo.list_sessions()
    messages = repo.list_messages(session.id)
    assert [(m.role, m.content) for m in messages] == [
        ("user", "Say hello"),
        ("assistant", "Hello"),
    ]
    assert conversation.history[-1] == {"role": "assistant", "content": "Hello"}


async def test_stop_before_any_text_saves_nothing():
    class SilentStreamer:
        async def stream_chat(self, messages, model, **kwargs) -> AsyncIterator[ChatEvent]:
            await asyncio.sleep(3600)
            yield TextDelta("never")

    conversation, repo = make(SilentStreamer())
    task = asyncio.create_task(conversation.send("hello"))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert repo.list_sessions() == []
    assert conversation.history == [{"role": "system", "content": "Be brief."}]


async def test_open_rebuilds_the_history_and_new_starts_over():
    streamer = ScriptedStreamer()
    conversation, repo = make(streamer)
    await conversation.send("remember me")
    saved = conversation.session
    assert saved is not None

    conversation.new()
    assert conversation.session is None and len(conversation.history) == 1

    messages = conversation.open(saved)
    assert [m.role for m in messages] == ["user", "assistant"]
    assert [m["content"] for m in conversation.history[1:]] == ["remember me", "Hi"]
