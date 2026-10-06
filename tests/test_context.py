"""Modules adding context to a conversation: the system prompt, background for one message, and learning."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path

from nicegui import ui
from nicegui.testing import User

from lemonrind.chats import ChatRepository, ContextUsed, Conversation
from lemonrind.config import Settings
from lemonrind.lemonade import ChatEvent, Finished, TextDelta
from lemonrind.lemonade.events import RequestStats
from lemonrind.modules import Module, ModuleRegistry, Tool, build_registry
from lemonrind.modules.memory import MemoryModule
from lemonrind.storage import Database
from lemonrind.webui.context import AppContext
from tests.conftest import NO_CHAT, open_section
from tests.test_memory import FakeMemoryClient
from tests.test_webui import (
    FakeLemonade,
    make_context,
)


class RecordingStreamer:
    def __init__(self) -> None:
        self.requests: list[list[dict]] = []

    async def stream_chat(self, messages, model, **kwargs) -> AsyncIterator[ChatEvent]:
        self.requests.append(json.loads(json.dumps(list(messages))))
        yield TextDelta("Noted.")
        yield Finished(RequestStats(), "stop")


class FakeContext:
    """A context provider with fixed answers that records what happened to it."""

    def __init__(self, *, stable: str = "", turn: str = "", fail: bool = False) -> None:
        self.stable, self.turn, self.fail = stable, turn, fail
        self.learned: list[tuple[str, str, str]] = []
        self.release = asyncio.Event()

    async def stable_context(self) -> str:
        if self.fail:
            raise RuntimeError("stable broke")
        return self.stable

    async def turn_context(self, user_text: str, *, chat) -> str:
        if self.fail:
            raise RuntimeError("turn broke")
        return self.turn

    async def after_turn(self, user_text: str, reply_text: str, *, model: str) -> None:
        await self.release.wait()  # slow, so the test can check that the reply did not wait for it
        self.learned.append((user_text, reply_text, model))


def make(context: FakeContext) -> tuple[Conversation, RecordingStreamer, ChatRepository]:
    streamer = RecordingStreamer()
    repo = ChatRepository(Database(":memory:"))
    conversation = Conversation(
        client=streamer, repo=repo, system_prompt="Be brief.", model="m", context=context
    )
    return conversation, streamer, repo


async def test_stable_context_extends_the_system_prompt():
    conversation, streamer, _ = make(FakeContext(stable="Known facts about the user:\n- Darren"))

    await conversation.send("hi")

    assert streamer.requests[0][0] == {
        "role": "system",
        "content": "Be brief.\n\nKnown facts about the user:\n- Darren",
    }


async def test_the_system_prompt_follows_changes_in_the_stable_context():
    context = FakeContext(stable="fact one")
    conversation, streamer, _ = make(context)
    await conversation.send("first")

    context.stable = ""
    await conversation.send("second")

    assert streamer.requests[1][0]["content"] == "Be brief."


async def test_turn_context_goes_to_the_model_but_not_into_the_saved_chat():
    conversation, streamer, repo = make(FakeContext(turn="- The user likes pizza."))
    seen: list = []

    await conversation.send("What should I eat?", seen.append)

    sent = streamer.requests[0][-1]["content"]
    assert sent.startswith("(Background for the assistant")
    assert "- The user likes pizza." in sent and sent.endswith("What should I eat?")
    assert ContextUsed("- The user likes pizza.") in seen
    # the saved chat and the history hold exactly what was typed
    (session,) = repo.list_sessions()
    assert repo.list_messages(session.id)[0].content == "What should I eat?"
    assert conversation.history[1] == {"role": "user", "content": "What should I eat?"}


async def test_earlier_messages_are_sent_unchanged_on_later_turns():
    context = FakeContext(turn="background")
    conversation, streamer, _ = make(context)
    await conversation.send("one")
    context.turn = "different background"

    await conversation.send("two")

    assert streamer.requests[1][1] == {
        "role": "user",
        "content": "one",
    }  # no old background stuck to it


async def test_a_failing_context_provider_never_breaks_the_chat():
    conversation, _, repo = make(FakeContext(fail=True))

    reply = await conversation.send("hi")

    assert reply.text == "Noted." and len(repo.list_sessions()) == 1


async def test_learning_runs_in_the_background_with_the_finished_reply():
    context = FakeContext()
    conversation, _, _ = make(context)

    reply = await conversation.send("I like pizza")  # returns although after_turn is still waiting

    assert reply.text == "Noted." and context.learned == []
    context.release.set()
    await conversation.drain()
    assert context.learned == [("I like pizza", "Noted.", "m")]


async def test_a_reply_that_is_not_saved_is_not_learned_from():
    class Empty(RecordingStreamer):
        async def stream_chat(self, messages, model, **kwargs):
            yield Finished(RequestStats(), "length")

    context = FakeContext()
    context.release.set()
    conversation = Conversation(
        client=Empty(),
        repo=ChatRepository(Database(":memory:")),
        system_prompt="x",
        model="m",
        context=context,
    )

    await conversation.send("hi")
    await conversation.drain()

    assert context.learned == []


# --- the registry's hooks ----------------------------------------------------------------------------------


class Contributing(Module):
    config_key = "contrib"
    description = "test"

    def __init__(self, settings: Settings, name: str, text: str, *, fail: bool = False) -> None:
        super().__init__(settings)
        self.name = name
        self.config_key = name
        self._text, self._fail = text, fail
        self.learned: list[str] = []

    def get_tools(self) -> list[Tool]:
        return []

    async def stable_context(self) -> str:
        if self._fail:
            raise RuntimeError("broken")
        return self._text

    async def turn_context(self, user_text: str, *, chat) -> str:
        return f"{self._text}:{user_text}"

    async def after_turn(self, user_text: str, reply_text: str, *, model: str) -> None:
        self.learned.append(reply_text)


async def test_the_registry_combines_enabled_modules_and_skips_broken_or_disabled_ones():
    settings = Settings()
    settings.modules.enabled["off"] = False
    modules = [
        Contributing(settings, "a", "first"),
        Contributing(settings, "bad", "x", fail=True),
        Contributing(settings, "off", "hidden"),
        Contributing(settings, "b", "second"),
    ]
    registry = ModuleRegistry(modules)

    assert await registry.stable_context() == "first\n\nsecond"
    assert await registry.turn_context("hi", chat=NO_CHAT) == "first:hi\n\nx:hi\n\nsecond:hi"
    await registry.after_turn("q", "answer", model="m")
    assert [m.learned for m in modules] == [["answer"], ["answer"], [], ["answer"]]


# --- end to end with the real memory module ------------------------------------------------------------------


async def test_memory_flows_from_one_chat_into_the_next():
    settings = Settings()
    db = Database(":memory:")
    client = FakeMemoryClient(
        extraction=json.dumps(
            {"facts": [{"content": "The user lives in Manchester.", "pinned": False}]}
        )
    )
    registry = build_registry(settings, Path("data"), db=db, client=client)  # type: ignore[arg-type]
    streamer = RecordingStreamer()
    repo = ChatRepository(db)

    first = Conversation(
        client=streamer,
        repo=repo,
        system_prompt="Be brief.",
        model="m",
        tools=registry,
        context=registry,
    )
    await first.send("I live in Manchester")
    await first.drain()

    second = Conversation(
        client=streamer,
        repo=repo,
        system_prompt="Be brief.",
        model="m",
        tools=registry,
        context=registry,
    )
    await second.send("Which town do I live in?")

    assert "The user lives in Manchester." in streamer.requests[-1][-1]["content"]
    memory = next(m for m in registry.modules if isinstance(m, MemoryModule))
    assert len(memory.repo.list_all()) == 1


# --- the Memories screen --------------------------------------------------------------------------------------


async def test_the_memories_screen_lists_pinned_and_other_facts(user: User, tmp_path: Path):
    from lemonrind.webui.context import set_context
    from lemonrind.webui.page import register_pages

    lemonade = FakeLemonade()
    context: AppContext = make_context(tmp_path, lemonade)
    memory = next(m for m in context.modules.modules if isinstance(m, MemoryModule))  # type: ignore[union-attr]
    memory.repo.add("The user's name is Darren.", pinned=True)
    memory.repo.add("The user likes pizza.", pinned=False)
    set_context(context)
    register_pages()

    await user.open("/")
    await open_section(user, "memory")

    await user.should_see("Pinned (always included)")
    await user.should_see("Other (used when relevant)")
    await user.should_see(kind=ui.input, content="The user's name is Darren.")
    set_context(None)
