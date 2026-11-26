"""A browser tab closed or reloaded while a reply is still running: the reply is saved and nothing is logged as an error.

NiceGUI deletes a page's client a few seconds after its tab goes away, but the task running the reply carries on.
These tests delete the client mid-reply (as NiceGUI does) and check what happens when the reply ends.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from nicegui.testing import User

from lemonrind.lemonade import ChatEvent, Finished, TextDelta, ToolCall, ToolCallsRequested
from lemonrind.lemonade.events import RequestStats
from lemonrind.webui.context import set_context
from lemonrind.webui.page import register_pages
from tests.conftest import must, open_section
from tests.test_queued_message import until_requested
from tests.test_settings_sections import settle
from tests.test_webui import FakeLemonade, make_context

STATS = RequestStats(prompt_tokens=20, input_tokens=20, output_tokens=5)


class HeldAnswer(FakeLemonade):
    """Starts answering, waits until the test lets it go on, then finishes the answer."""

    def __init__(self) -> None:
        super().__init__()
        self.release = asyncio.Event()

    async def stream_chat(self, messages, model, **kwargs) -> AsyncIterator[ChatEvent]:
        self.requests.append([dict(m) for m in messages])
        yield TextDelta("The first half, ")
        await self.release.wait()
        yield TextDelta("and the second half.")
        yield Finished(STATS, "stop")


class HeldToolAnswer(FakeLemonade):
    """Waits until the test lets it go on, then asks for a tool (new panels after the tab has gone)."""

    def __init__(self, tool: str = "calculate", arguments: str = '{"expression": "6 * 7"}') -> None:
        super().__init__()
        self.release = asyncio.Event()
        self.call = ToolCall("call-1", tool, arguments)

    async def stream_chat(self, messages, model, **kwargs) -> AsyncIterator[ChatEvent]:
        self.requests.append([dict(m) for m in messages])
        if any(m["role"] == "tool" for m in messages):
            yield TextDelta("The answer is 42.")
            yield Finished(STATS, "stop")
            return
        yield TextDelta("Let me work it out. ")
        await self.release.wait()
        yield ToolCallsRequested((self.call,))
        yield Finished(STATS, "tool_calls")


def errors(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Errors, plus NiceGUI's warning about a deleted page still being used (it means the same mistake)."""
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno >= logging.ERROR
        or "has been deleted but is still being used" in r.getMessage()
    ]


@pytest.mark.parametrize("by_enter", [False, True], ids=["send button", "Enter key"])
async def test_a_tab_closed_mid_reply_still_saves_the_chat_and_logs_no_error(
    user: User, tmp_path: Path, caplog: pytest.LogCaptureFixture, by_enter: bool
):
    lemonade = HeldAnswer()
    context = make_context(tmp_path, lemonade)
    set_context(context)
    register_pages()
    try:
        await user.open("/")
        await user.should_see("Lemonade healthy")
        await settle()
        box = user.find(marker="message-input").type("Tell me in two halves")
        if by_enter:
            box.trigger("keydown.enter.exact.prevent")
        else:
            user.find(marker="send").click()
        await until_requested(lemonade, 1)
        await asyncio.sleep(0.2)  # the first half has arrived

        must(user.client).delete()  # the tab is closed or reloaded
        lemonade.release.set()
        await asyncio.sleep(0.5)  # the reply finishes on its own

        (session,) = context.repo.list_sessions()
        assert [m.content for m in context.repo.list_messages(session.id)] == [
            "Tell me in two halves",
            "The first half, and the second half.",
        ]
        assert not errors(caplog)
    finally:
        lemonade.release.set()
        set_context(None)


async def test_a_tab_closed_before_a_tool_runs_still_saves_the_whole_turn(
    user: User, tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    lemonade = HeldToolAnswer()
    context = make_context(tmp_path, lemonade)
    set_context(context)
    register_pages()
    try:
        await user.open("/")
        await user.should_see("Lemonade healthy")
        await settle()
        user.find(marker="message-input").type("What is 6 times 7?")
        user.find(marker="send").click()
        await until_requested(lemonade, 1)

        must(user.client).delete()
        lemonade.release.set()
        await asyncio.sleep(0.5)

        assert len(lemonade.requests) == 2  # the tool ran and the model was asked again
        (session,) = context.repo.list_sessions()
        contents = [m.content for m in context.repo.list_messages(session.id)]
        assert contents[0] == "What is 6 times 7?"
        assert contents[-1] == "The answer is 42."
        assert not errors(caplog)
    finally:
        lemonade.release.set()
        set_context(None)


async def test_a_tool_that_needs_permission_after_the_tab_has_gone_is_refused_quietly(
    user: User, tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    lemonade = HeldToolAnswer("write_file", '{"path": "notes.txt", "content": "hello"}')
    context = make_context(tmp_path, lemonade)
    set_context(context)
    register_pages()
    try:
        await user.open("/")
        await user.should_see("Lemonade healthy")
        await settle()
        user.find(marker="message-input").type("Write a note")
        user.find(marker="send").click()
        await until_requested(lemonade, 1)

        must(user.client).delete()
        lemonade.release.set()
        await asyncio.sleep(0.5)

        assert len(lemonade.requests) == 2  # the model was told the tool was refused, and answered
        assert not list(
            tmp_path.rglob("notes.txt")
        )  # no one could approve it, so nothing was written
        (session,) = context.repo.list_sessions()
        assert context.repo.list_messages(session.id)[-1].content == "The answer is 42."
        assert not errors(caplog)
    finally:
        lemonade.release.set()
        set_context(None)


async def test_a_tab_closed_while_settings_are_open_logs_no_error(
    user: User, tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    lemonade = FakeLemonade()
    context = make_context(tmp_path, lemonade)
    context.settings.modules.enabled["knowledge"] = (
        True  # its drop-down is refreshed after Settings
    )
    set_context(context)
    register_pages()
    try:
        await open_section(user, "lemonade")
        must(user.client).delete()  # the open dialog's await returns as the page goes
        await settle()
        assert not errors(caplog)
    finally:
        set_context(None)
