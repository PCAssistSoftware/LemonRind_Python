"""Tests for the web UI using NiceGUI's simulated browser (the ``user`` fixture).

The ``user`` fixture runs the real page code against a pretend browser: ``user.should_see("text")`` checks the
page contains that text, ``user.find(marker=...)`` finds a widget by the marker we gave it, and ``.click()`` /
``.type()`` act on it. No real browser, no network and no model are involved: the Lemonade client is a fake.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from nicegui import ui
from nicegui.testing import User

from lemonrind.chats import ChatRepository
from lemonrind.config import Settings
from lemonrind.lemonade import (
    ChatEvent,
    Finished,
    ReasoningDelta,
    TextDelta,
    ToolCall,
    ToolCallsRequested,
)
from lemonrind.lemonade.client import LemonadeError
from lemonrind.lemonade.events import RequestStats
from lemonrind.lemonade.models import Health, LoadedModel, ModelInfo
from lemonrind.modules import build_registry
from lemonrind.storage import Database
from lemonrind.webui.context import AppContext, set_context
from lemonrind.webui.page import register_pages
from tests.conftest import NullMemoryClient, elements, must


class FakeLemonade:
    """Stands in for LemonadeClient: one healthy chat model, and a canned streamed reply."""

    base_url = "http://fake/v1/"

    def __init__(self, answer: str = "Paris.", *, fail: bool = False) -> None:
        self.answer = answer
        self.fail = fail
        self.requests: list[list[dict[str, str]]] = []

    async def health(self) -> Health:
        return Health(
            status="ok",
            model_loaded="Fake-Chat",
            all_models_loaded=[LoadedModel(model_name="Fake-Chat")],
        )

    async def list_models(self, *, downloaded_only: bool = True) -> list[ModelInfo]:
        return [
            ModelInfo(id="Fake-Chat", labels=["chat"], downloaded=True, context_length=8000),
            ModelInfo(id="Fake-Embed", labels=["embeddings"], downloaded=True),
        ]

    async def load_model(self, name: str) -> None:
        pass

    async def tokenize(self, content: str) -> int:
        return max(1, len(content) // 4) if content else 0

    async def stream_chat(self, messages, model, **kwargs) -> AsyncIterator[ChatEvent]:
        self.requests.append([dict(m) for m in messages])
        if self.fail:
            raise LemonadeError("Lemonade is on fire")
        yield ReasoningDelta("Let me think. ")
        yield TextDelta(self.answer)
        yield Finished(RequestStats(prompt_tokens=20, input_tokens=20, output_tokens=5), "stop")

    async def aclose(self) -> None:
        pass


class ToolLemonade(FakeLemonade):
    """Asks for the calculator on the first request, then answers using its result."""

    async def stream_chat(self, messages, model, **kwargs) -> AsyncIterator[ChatEvent]:
        self.requests.append([dict(m) for m in messages])
        stats = RequestStats(prompt_tokens=20, input_tokens=20, output_tokens=5)
        if any(m["role"] == "tool" for m in messages):
            yield TextDelta("The answer is 42.")
            yield Finished(stats, "stop")
        else:
            call = ToolCall("call-1", "calculate", '{"expression": "6 * 7"}')
            yield ToolCallsRequested((call,))
            yield Finished(stats, "tool_calls")


def make_context(tmp_path: Path, lemonade: Any) -> AppContext:
    db = Database(":memory:")
    settings = Settings()
    return AppContext(
        settings=settings,
        settings_file=tmp_path / "settings.json",
        db=db,
        client=lemonade,  # type: ignore[arg-type]
        repo=ChatRepository(db),
        modules=build_registry(settings, tmp_path, db=db, client=NullMemoryClient()),
    )


@pytest.fixture
def lemonade() -> FakeLemonade:
    return FakeLemonade()


@pytest.fixture
def web(tmp_path: Path, lemonade: FakeLemonade):
    """An app context with a fake Lemonade, and the page registered.

    Depend on ``user`` *before* ``web`` in a test's arguments: the ``user`` fixture resets NiceGUI's page
    registry, and the page has to be registered after that reset.
    """
    context = make_context(tmp_path, lemonade)
    set_context(context)
    register_pages()
    yield context
    set_context(None)


async def send(user: User, text: str) -> None:
    user.find(marker="message-input").type(text)
    user.find(marker="send").click()


async def test_an_empty_app_shows_the_hint_and_a_healthy_model(user: User, web: AppContext):
    await user.open("/")
    await user.should_see("Start a conversation")
    await user.should_see("Lemonade healthy")
    await user.should_see("No chats yet")


async def test_sending_a_message_streams_the_reply_and_saves_the_chat(
    user: User, web: AppContext, lemonade: FakeLemonade
):
    await user.open("/")
    await send(user, "What is the capital of France?")

    await user.should_see("Paris.")
    await user.should_see("What is the capital of France?")
    await user.should_see("Thought for")  # the thinking panel collapsed to its summary
    await user.should_see("20 in / 5 out")  # the stats line under the reply

    (session,) = web.repo.list_sessions()
    assert session.title == "What is the capital of France?"
    assert [m.content for m in web.repo.list_messages(session.id)] == [
        "What is the capital of France?",
        "Paris.",
    ]
    assert [m["role"] for m in lemonade.requests[0]] == ["system", "user"]


async def test_the_latest_chat_is_reopened_with_its_messages(user: User, web: AppContext):
    chat = web.repo.create_session()
    web.repo.add_message(chat.id, "user", "Tell me about otters")
    web.repo.add_message(chat.id, "assistant", "Otters are playful.")

    await user.open("/")
    await user.should_see("Otters are playful.")
    await user.should_see("Tell me about otters")


async def test_the_next_message_continues_the_opened_chat(
    user: User, web: AppContext, lemonade: FakeLemonade
):
    chat = web.repo.create_session()
    web.repo.add_message(chat.id, "user", "remember: banana")
    web.repo.add_message(chat.id, "assistant", "Noted.")

    await user.open("/")
    await send(user, "what did I say?")
    await user.should_see("Paris.")

    earlier_one, earlier_two, newest = (m["content"] for m in lemonade.requests[0][1:])
    assert (earlier_one, earlier_two) == ("remember: banana", "Noted.")
    # the newest message carries the current date and time ahead of what was typed (and only in the request)
    assert "Current date/time:" in newest and newest.endswith("what did I say?")
    assert [m.content for m in web.repo.list_messages(web.repo.list_sessions()[0].id)][-2:] == [
        "what did I say?",
        "Paris.",
    ]
    assert len(web.repo.list_sessions()) == 1


async def test_new_chat_starts_empty_and_nothing_is_saved_until_a_reply(
    user: User, web: AppContext
):
    chat = web.repo.create_session()
    web.repo.add_message(chat.id, "user", "old question")
    web.repo.add_message(chat.id, "assistant", "old answer")

    await user.open("/")
    await user.should_see("old answer")
    user.find(marker="new-chat").click()
    await user.should_see("Start a conversation")
    await user.should_not_see("old answer")
    assert len(web.repo.list_sessions()) == 1  # still just the old chat


async def test_search_filters_the_chat_list(user: User, web: AppContext):
    for text in ("about zebras", "about lions"):
        chat = web.repo.create_session()
        web.repo.add_message(chat.id, "user", text)
        web.repo.add_message(chat.id, "assistant", "ok")

    await user.open("/")
    await user.should_see(kind=ui.item_label, content="about lions")
    user.find(marker="search").type("zebr")
    await user.should_see(marker="chat-snippet")  # the search results replace the plain labels
    # (a sidebar title is shown with its match highlighted, in a ui.html element rather than a plain label)
    assert [e.content for e in elements(user, "chat-title")] == [
        'about <mark class="lr-mark">zebr</mark>as'
    ]
    snippets = [e.content for e in elements(user, "chat-snippet")]
    assert len(snippets) == 1 and '<mark class="lr-mark">zebras</mark>' in snippets[0]
    assert not any("lions" in s for s in snippets)


async def test_search_highlights_the_title_and_never_runs_html_in_a_chat(
    user: User, web: AppContext
):
    chat = web.repo.create_session("A <b>bold</b> hike")
    web.repo.add_message(chat.id, "user", "see <script>alert(1)</script> hike")
    web.repo.add_message(chat.id, "assistant", "ok")

    await user.open("/")
    user.find(marker="search").type("hike")
    await user.should_see(marker="chat-snippet")

    (title,) = elements(user, "chat-title")
    (snippet,) = elements(user, "chat-snippet")
    assert title.content == 'A &lt;b&gt;bold&lt;/b&gt; <mark class="lr-mark">hike</mark>'
    assert "<script>" not in snippet.content and "&lt;script&gt;" in snippet.content
    assert '<mark class="lr-mark">hike</mark>' in snippet.content


@pytest.mark.parametrize("lemonade", [FakeLemonade(fail=True)])
async def test_a_failed_reply_is_explained_and_the_message_is_put_back(user: User, web: AppContext):
    await user.open("/")
    await send(user, "Will this work?")

    await user.should_see("Lemonade is on fire")
    assert web.repo.list_sessions() == []
    assert elements(user, "message-input")[0].value == "Will this work?"


@pytest.mark.parametrize("lemonade", [ToolLemonade()])
async def test_a_tool_call_shows_as_a_panel_and_is_stored_with_the_chat(
    user: User, web: AppContext, lemonade: ToolLemonade
):
    await user.open("/")
    await send(user, "What is 6 times 7?")

    await user.should_see("The answer is 42.")
    await user.should_see("calculate done")  # the panel's title once the tool finished
    assert lemonade.requests[1][-1]["content"] == "42"  # the result went back to the model

    # opening the page again rebuilds the same bubble, with its panel, from the database
    await user.open("/")
    await user.should_see("The answer is 42.")
    await user.should_see("calculate done")
    (session,) = web.repo.list_sessions()
    assert [m.role for m in web.repo.list_messages(session.id)] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]


# --- the stats panel: tool calls this turn, and the Modules tab ---------------------------------------------------------


class FailingToolLemonade(FakeLemonade):
    """Asks for a tool that does not exist, so the call fails, then answers."""

    async def stream_chat(self, messages, model, **kwargs) -> AsyncIterator[ChatEvent]:
        self.requests.append([dict(m) for m in messages])
        stats = RequestStats(prompt_tokens=20, input_tokens=20, output_tokens=5)
        if any(m["role"] == "tool" for m in messages):
            yield TextDelta("That tool does not exist.")
            yield Finished(stats, "stop")
        else:
            yield ToolCallsRequested((ToolCall("call-9", "no_such_tool", "{}"),))
            yield Finished(stats, "tool_calls")


def chip_colours(user: User) -> list[tuple[str, str]]:
    """(tool name, colour) for each chip in the stats panel."""
    return [
        (
            next(c for c in chip.default_slot.children if hasattr(c, "text") and c.text).text,
            chip.props["color"],
        )  # type: ignore[attr-defined]
        for chip in user.find(marker="tool-chip").elements
    ]


@pytest.mark.parametrize("lemonade", [ToolLemonade()])
async def test_a_successful_tool_call_shows_as_a_green_chip_and_survives_reopening(
    user: User, web: AppContext, lemonade: ToolLemonade
):
    await user.open("/")
    await user.should_see("No tool calls yet.")

    await send(user, "What is 6 times 7?")
    await user.should_see("The answer is 42.")

    assert chip_colours(user) == [("calculate", "positive")]
    await user.open("/")  # rebuilt from the saved chat
    await user.should_see("The answer is 42.")
    assert chip_colours(user) == [("calculate", "positive")]


@pytest.mark.parametrize("lemonade", [FailingToolLemonade()])
async def test_a_failed_tool_call_is_red_with_its_real_error_and_the_failure_is_saved(
    user: User, web: AppContext, lemonade: FailingToolLemonade
):
    await user.open("/")
    await send(user, "Use a tool that is not there")
    await user.should_see("That tool does not exist.")

    assert chip_colours(user) == [("no_such_tool", "negative")]
    (session,) = web.repo.list_sessions()
    tool_message = next(m for m in web.repo.list_messages(session.id) if m.role == "tool")
    assert tool_message.tool_failed is True and "There is no tool called" in tool_message.content

    await user.open("/")  # the failure is remembered, not forgotten when the chat is reopened
    await user.should_see("That tool does not exist.")
    assert chip_colours(user) == [("no_such_tool", "negative")]
    await user.should_see("no_such_tool failed")  # the panel in the chat agrees


@pytest.mark.parametrize("lemonade", [ToolLemonade()])
async def test_a_new_chat_clears_the_tool_chips(
    user: User, web: AppContext, lemonade: ToolLemonade
):
    await user.open("/")
    await send(user, "What is 6 times 7?")
    await user.should_see("The answer is 42.")
    assert chip_colours(user) == [("calculate", "positive")]

    user.find(marker="new-chat").click()

    await user.should_see("No tool calls yet.")
    await user.should_not_see(marker="tool-chip")


async def test_the_modules_tab_lists_every_module_and_swaps_the_buttons_for_a_way_into_settings(
    user: User, web: AppContext
):
    web.settings.modules.enabled["web_search"] = False
    await user.open("/")
    await user.should_see(marker="view-prompt")
    await user.should_not_see(marker="manage-modules")

    user.find(marker="tab-modules").click()

    await user.should_see(marker="module-card")
    names = {
        label.text
        for card in user.find(marker="module-card").elements
        for label in card.descendants()
        if hasattr(label, "text")
    }  # type: ignore[attr-defined]
    assert {"Web search", "Web reader", "File system", "Auto-backup", "Utilities"} <= names
    assert "Disabled" in names and "Enabled" in names  # web search was switched off above
    await user.should_see(marker="manage-modules")
    await user.should_not_see(marker="view-prompt")
    user.find(marker="manage-modules").click()
    await user.should_see(
        "Each module gives the assistant some tools"
    )  # Settings opened on the Modules section

    user.find(marker="settings-cancel").click()
    user.find(marker="tab-stats").click()
    await user.should_see(marker="view-prompt")
    await user.should_not_see(marker="manage-modules")


async def test_the_two_stats_tabs_look_like_a_switch_with_the_open_one_marked(
    user: User, web: AppContext
):
    await user.open("/")

    def active() -> list[str]:
        return [
            key
            for key in ("stats", "modules")
            if "lr-segment-active" in next(iter(user.find(marker=f"tab-{key}").elements)).classes  # type: ignore[attr-defined]
        ]

    assert active() == ["stats"]
    user.find(marker="tab-modules").click()
    await user.should_see(marker="module-card")
    assert active() == ["modules"]
    user.find(marker="tab-stats").click()
    assert active() == ["stats"]


async def test_the_button_that_toggles_the_right_hand_panel_is_named_for_both_tabs(
    user: User, web: AppContext
):
    await user.open("/")

    tooltips = [t.text for t in user.find(kind=ui.tooltip).elements]  # type: ignore[attr-defined]

    assert "Stats / Modules" in tooltips and "Stats" not in tooltips


def test_the_page_script_covers_a_stopped_server():
    from lemonrind.webui.styles import CSS, JS

    # NiceGUI's own "Connection lost" notice is watched; after a few seconds a full-page notice takes over
    assert "getElementById('popup')" in JS and "Lemon Rind has stopped" in JS
    assert ".lr-offline.lr-offline-on" in CSS


def test_side_panel_widths_have_defaults_and_stay_within_limits():
    import pytest
    from pydantic import ValidationError

    from lemonrind.config import UiSettings
    from lemonrind.webui.styles import JS

    ui_settings = UiSettings()
    assert (ui_settings.left_panel_width, ui_settings.right_panel_width) == (300, 280)
    with pytest.raises(ValidationError):
        UiSettings(left_panel_width=100)  # too narrow to be useful
    with pytest.raises(ValidationError):
        UiSettings(right_panel_width=900)  # would leave no room for the chat
    assert "lr_resize" in JS  # the drag handles report to the page under this event name


async def test_a_page_closed_while_lemonade_is_still_answering_causes_no_error(
    user: User, web: AppContext, lemonade: FakeLemonade, caplog
):
    gate = asyncio.Event()
    original = lemonade.list_models

    async def slow_list_models(*, downloaded_only: bool = True):
        await gate.wait()  # Lemonade is slow to answer
        return await original(downloaded_only=downloaded_only)

    lemonade.list_models = slow_list_models  # type: ignore[method-assign]
    await user.open("/")
    await asyncio.sleep(0.1)  # the page is up and its start-up check is waiting for the answer

    must(user.client).delete()  # the tab is closed or reloaded
    gate.set()
    await asyncio.sleep(0.3)

    assert not [r for r in caplog.records if r.levelname == "ERROR"]


async def test_a_reply_that_used_tools_shows_text_and_tool_panels_in_the_order_they_happened(
    user: User, web: AppContext
):
    from nicegui import ui

    from lemonrind.lemonade.events import ToolCall

    chat = web.repo.create_session()
    web.repo.add_message(chat.id, "user", "sum and time?")
    call_one = ToolCall("c1", "calculate", '{"expression": "6 * 7"}')
    web.repo.add_message(chat.id, "assistant", "First the sum.", tool_calls=[call_one])
    web.repo.add_message(chat.id, "tool", "42", tool_call_id="c1")
    call_two = ToolCall("c2", "current_time", "{}")
    web.repo.add_message(chat.id, "assistant", "Then the time.", tool_calls=[call_two])
    web.repo.add_message(chat.id, "tool", "15:40", tool_call_id="c2")
    web.repo.add_message(chat.id, "assistant", "All done.")

    await user.open("/")
    await user.should_see("All done.")

    markdown = next(iter(user.find(content="First the sum.").elements))
    flow = must(markdown.parent_slot).parent  # the column holding the reply, in time order
    kinds = [
        ("text:" + child.content) if isinstance(child, ui.markdown) else "tool"
        for child in flow.default_slot.children
    ]
    assert kinds == ["text:First the sum.", "tool", "text:Then the time.", "tool", "text:All done."]
