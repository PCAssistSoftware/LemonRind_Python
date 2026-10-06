"""Tests for the terminal chat loop, driven by scripted input and a fake model client.

``ScriptedConsole`` is a rich Console whose ``input`` reads from a list instead of the keyboard, and whose
output goes into a string we can inspect. ``FakeClient`` stands in for LemonadeClient and "streams" a
canned reply, recording what it was sent.
"""

from __future__ import annotations

import io
from collections.abc import AsyncIterator
from typing import cast

import pytest
from rich.console import Console

from lemonrind.chats import DEFAULT_TITLE, ChatRepository
from lemonrind.config import Settings
from lemonrind.lemonade import ChatEvent, Finished, ReasoningDelta, TextDelta
from lemonrind.lemonade.client import LemonadeError
from lemonrind.lemonade.events import RequestStats
from lemonrind.storage import Database
from lemonrind.terminal.chatloop import ChatLoop


class ScriptedConsole(Console):
    def __init__(self, lines: list[str]) -> None:
        super().__init__(file=io.StringIO(), width=120, force_terminal=False, color_system=None)
        self._lines = list(lines)

    def input(self, prompt="", **kwargs) -> str:  # replaces keyboard input
        if not self._lines:
            raise EOFError
        return self._lines.pop(0)

    @property
    def output(self) -> str:
        return cast(io.StringIO, self.file).getvalue()


class FakeClient:
    """Replies with a fixed answer (after some "thinking"), or fails if told to."""

    base_url = "http://fake/v1/"

    def __init__(self, answer: str = "Paris.", *, fail: bool = False) -> None:
        self.answer = answer
        self.fail = fail
        self.sent: list[list[dict[str, str]]] = []  # a copy of the messages for every request

    async def list_models(self, **kwargs) -> list:
        return []

    async def stream_chat(self, messages, model, **kwargs) -> AsyncIterator[ChatEvent]:
        self.sent.append([dict(m) for m in messages])
        if self.fail:
            raise LemonadeError("Lemonade is on fire")
        yield ReasoningDelta("Let me think. ")
        if self.answer:
            yield TextDelta(self.answer)
        yield Finished(RequestStats(output_tokens=3), "stop" if self.answer else "length")


def make_loop(
    lines: list[str], client: FakeClient | None = None, repo: ChatRepository | None = None
):
    console = ScriptedConsole(lines)
    client = client or FakeClient()
    repo = repo or ChatRepository(Database(":memory:"))
    loop = ChatLoop(client=client, settings=Settings(), repo=repo, console=console, model="m")  # type: ignore[arg-type]
    return loop, console, client, repo


# --- sending and saving ----------------------------------------------------------------------------------


async def test_first_message_creates_and_saves_the_chat():
    loop, console, client, repo = make_loop(["What is the capital of France?"])
    await loop.run()

    (session,) = repo.list_sessions()
    assert session.title == "What is the capital of France?"
    user, assistant = repo.list_messages(session.id)
    assert (user.role, user.content) == ("user", "What is the capital of France?")
    assert (assistant.role, assistant.content, assistant.reasoning) == (
        "assistant",
        "Paris.",
        "Let me think.",
    )
    assert assistant.stats is not None and assistant.stats.output_tokens == 3
    assert "Saved as: What is the capital of France?" in console.output


async def test_opening_and_leaving_without_a_message_saves_nothing():
    loop, _, _, repo = make_loop(["/new", "/quit"])
    await loop.run()
    assert repo.list_sessions() == []


async def test_a_failed_reply_is_not_saved_or_kept_in_the_history():
    client = FakeClient(fail=True)
    loop, console, _, repo = make_loop(["Hello?", "/quit"], client)
    await loop.run()
    assert repo.list_sessions() == []
    assert "Lemonade is on fire" in console.output
    assert loop.conversation.history == [loop.conversation._system_message]


async def test_an_empty_reply_is_not_saved():
    loop, console, _, repo = make_loop(["Think hard", "/quit"], FakeClient(answer=""))
    await loop.run()
    assert repo.list_sessions() == []
    assert "hit the token limit" in console.output


async def test_the_model_sees_the_whole_conversation_each_turn():
    loop, _, client, _ = make_loop(["first", "second"])
    await loop.run()
    roles = [[m["role"] for m in request] for request in client.sent]
    assert roles == [["system", "user"], ["system", "user", "assistant", "user"]]


# --- resuming and navigating -------------------------------------------------------------------------------


async def test_starting_again_reopens_the_latest_chat_with_its_history():
    repo = ChatRepository(Database(":memory:"))
    await make_loop(["remember the word banana"], repo=repo)[0].run()

    client = FakeClient("You said banana.")
    loop, console, _, _ = make_loop(["what did I say?"], client, repo)
    await loop.run(resume=True)

    earlier, answer, newest = (m["content"] for m in client.sent[0][1:])
    assert (earlier, answer) == ("remember the word banana", "Paris.")
    assert "Current date/time:" in newest and newest.endswith(
        "what did I say?"
    )  # the terminal is time-aware too
    assert "Opened: remember the word banana" in console.output
    assert len(repo.list_sessions()) == 1  # continued, not duplicated


async def test_resume_can_be_switched_off():
    repo = ChatRepository(Database(":memory:"))
    await make_loop(["old chat"], repo=repo)[0].run()
    loop, console, _, _ = make_loop([], repo=repo)
    await loop.run(resume=False)
    assert "Opened:" not in console.output


async def test_chats_then_open_switches_chat_and_history():
    repo = ChatRepository(Database(":memory:"))
    await make_loop(["about cats"], repo=repo)[0].run()
    await make_loop(["about dogs"], repo=repo)[0].run(resume=False)  # a second, newer chat

    loop, console, _, _ = make_loop(["/chats", "/open 2", "/quit"], repo=repo)
    await loop.run(resume=False)

    assert " 1. about dogs" in console.output and " 2. about cats" in console.output
    assert loop.session is not None and loop.session.title == "about cats"
    assert [m["content"] for m in loop.conversation.history[1:]] == ["about cats", "Paris."]


async def test_open_with_a_bad_number_is_explained():
    loop, console, _, _ = make_loop(["/open 7", "/open x", "/quit"])
    await loop.run(resume=False)
    assert console.output.count("Give a number") == 2


async def test_search_lists_matching_chats():
    repo = ChatRepository(Database(":memory:"))
    await make_loop(["tell me about zebras"], repo=repo)[0].run()
    await make_loop(["tell me about lions"], repo=repo)[0].run(resume=False)

    loop, console, _, _ = make_loop(["/search zebr", "/quit"], repo=repo)
    await loop.run(resume=False)
    assert "tell me about zebras" in console.output
    assert "tell me about lions" not in console.output


# --- organising ---------------------------------------------------------------------------------------------


async def test_rename_folder_and_tags():
    loop, _, _, repo = make_loop(
        ["hello", "/rename My plan", "/folder Work", "/tag urgent", "/tag Code"]
    )
    await loop.run()
    (session,) = repo.list_sessions()
    assert (session.title, session.folder_name, session.tags) == (
        "My plan",
        "Work",
        ("Code", "urgent"),
    )

    loop2, _, _, _ = make_loop(["/open 1", "/untag code", "/folder"], repo=repo)
    await loop2.run(resume=True)
    (session,) = repo.list_sessions()
    assert session.folder_name is None and session.tags == ("urgent",)


async def test_organising_commands_need_a_saved_chat():
    loop, console, _, repo = make_loop(["/rename x", "/tag y", "/folder z", "/quit"])
    await loop.run(resume=False)
    assert console.output.count("not saved yet") == 3
    assert repo.list_sessions() == []


@pytest.mark.parametrize(("answer", "kept"), [("y", False), ("n", True), ("", True)])
async def test_delete_asks_for_confirmation(answer: str, kept: bool):
    loop, _, _, repo = make_loop(["hello", "/delete", answer])
    await loop.run()
    assert bool(repo.list_sessions()) is kept
    if not kept:
        assert loop.session is None  # deleting the open chat leaves a fresh draft


async def test_unknown_commands_are_reported():
    loop, console, _, _ = make_loop(["/nonsense", "/quit"])
    await loop.run(resume=False)
    assert "Unknown command /nonsense" in console.output


async def test_a_default_titled_chat_gets_named_by_its_first_message():
    loop, _, _, repo = make_loop(["Name me please"])
    await loop.run()
    assert repo.list_sessions()[0].title != DEFAULT_TITLE
