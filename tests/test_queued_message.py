"""A message typed while a reply is being worked on: Enter queues it (and never stops the reply), then it is sent."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

from nicegui.testing import User

from lemonrind.lemonade import ChatEvent, Finished, TextDelta
from lemonrind.lemonade.events import RequestStats
from lemonrind.webui.context import set_context
from lemonrind.webui.page import register_pages
from tests.test_settings_sections import settle
from tests.test_webui import FakeLemonade, make_context, send


async def until_requested(lemonade: FakeLemonade, count: int) -> None:
    """Wait until the model has been asked ``count`` times (the first reply is then in flight)."""
    for _ in range(100):
        if len(lemonade.requests) >= count:
            return
        await asyncio.sleep(0.05)
    raise AssertionError("the model was never asked")


class HeldReply(FakeLemonade):
    """The first reply waits until the test lets it finish; every later one is immediate."""

    def __init__(self) -> None:
        super().__init__()
        self.release = asyncio.Event()

    async def stream_chat(self, messages, model, **kwargs) -> AsyncIterator[ChatEvent]:
        self.requests.append([dict(m) for m in messages])
        number = len(self.requests)
        yield TextDelta(f"Answer {number}. ")
        if number == 1:
            await self.release.wait()
        yield Finished(RequestStats(prompt_tokens=20, input_tokens=20, output_tokens=5), "stop")


async def test_enter_while_a_reply_runs_queues_the_message_and_sends_it_afterwards(
    user: User, tmp_path: Path
):
    lemonade = HeldReply()
    context = make_context(tmp_path, lemonade)
    set_context(context)
    register_pages()
    try:
        await user.open("/")
        await user.should_see("Lemonade healthy")
        await settle()

        await send(user, "First question")
        await until_requested(lemonade, 1)

        user.find(marker="message-input").type("And one more thing").trigger(
            "keydown.enter.exact.prevent"
        )
        await user.should_see("Queued, goes to the assistant at its next step")
        assert len(lemonade.requests) == 1  # Enter did not stop the reply or start another

        lemonade.release.set()
        await user.should_see("Answer 2.")  # sent by itself once the first reply ended

        users = [
            m["content"] for request in lemonade.requests for m in request if m["role"] == "user"
        ]
        assert any("And one more thing" in content for content in users)
        await user.should_not_see(marker="queued-row")
    finally:
        lemonade.release.set()
        set_context(None)


async def test_the_cross_takes_a_queued_message_back_into_the_box(user: User, tmp_path: Path):
    lemonade = HeldReply()
    context = make_context(tmp_path, lemonade)
    set_context(context)
    register_pages()
    try:
        await user.open("/")
        await user.should_see("Lemonade healthy")
        await settle()
        await send(user, "First question")
        await until_requested(lemonade, 1)
        user.find(marker="message-input").type("Changed my mind").trigger(
            "keydown.enter.exact.prevent"
        )
        await user.should_see("Queued, goes to the assistant")

        user.find(marker="queued-cancel").click()
        lemonade.release.set()
        await settle()

        assert len(lemonade.requests) == 1  # nothing was sent for the queued message
        box = user.find(marker="message-input").elements.pop()
        assert box.value == "Changed my mind"  # type: ignore[attr-defined]
    finally:
        lemonade.release.set()
        set_context(None)
