"""The stats panel folds many calls to one tool into one badge, keeping done and failed apart."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

from nicegui.testing import User

from lemonrind.lemonade import ChatEvent, Finished, TextDelta, ToolCall, ToolCallsRequested
from lemonrind.lemonade.events import RequestStats
from lemonrind.webui.bubbles import ToolChip
from lemonrind.webui.context import AppContext, set_context
from lemonrind.webui.page import register_pages
from lemonrind.webui.stats import group_chips
from tests.conftest import elements
from tests.test_webui import FakeLemonade, make_context, send


def chip(name: str, state: str = "ok", detail: str = "") -> ToolChip:
    return ToolChip(name, state, detail)


def test_calls_to_one_tool_become_one_group_with_a_count_in_first_use_order():
    chips = [chip("web_search"), chip("read_webpage"), chip("web_search"), chip("web_search")]

    groups = group_chips(chips)

    assert [(g.name, g.state, g.count) for g in groups] == [
        ("web_search", "ok", 3),
        ("read_webpage", "ok", 1),
    ]


def test_done_and_failed_calls_of_the_same_tool_stay_separate_and_failures_keep_their_errors():
    chips = [
        chip("read_webpage"),
        chip("read_webpage", "failed", "HTTP 404"),
        chip("read_webpage"),
        chip("read_webpage", "failed", "HTTP 404"),
        chip("read_webpage", "failed", "timed out"),
    ]

    groups = group_chips(chips)

    assert [(g.state, g.count) for g in groups] == [("ok", 2), ("failed", 3)]
    assert groups[1].details == ["HTTP 404", "timed out"]  # each kind of error once


def test_no_calls_gives_no_groups():
    assert group_chips([]) == []


class ManyCallsLemonade(FakeLemonade):
    """Twelve calculator calls in one request, one of which is a tool that does not exist."""

    async def stream_chat(self, messages, model, **kwargs) -> AsyncIterator[ChatEvent]:
        self.requests.append([dict(m) for m in messages])
        stats = RequestStats(prompt_tokens=20, input_tokens=20, output_tokens=5)
        if any(m["role"] == "tool" for m in messages):
            yield TextDelta("All done.")
            yield Finished(stats, "stop")
        else:
            calls = [
                ToolCall(f"call-{i}", "calculate", '{"expression": "1 + 1"}') for i in range(5)
            ]
            calls.append(ToolCall("call-x", "no_such_tool", "{}"))
            yield ToolCallsRequested(tuple(calls))
            yield Finished(stats, "tool_calls")


async def test_the_panel_shows_one_badge_per_tool_and_outcome_with_a_count(
    user: User, tmp_path: Path
):
    context: AppContext = make_context(tmp_path, ManyCallsLemonade())
    set_context(context)
    register_pages()
    await user.open("/")
    await send(user, "Do lots of sums")
    await user.should_see("All done.")

    badges = elements(user, "tool-chip")
    shown = {
        next(c.text for c in badge.default_slot.children if hasattr(c, "text") and c.text): (  # type: ignore[attr-defined]
            [c.text for c in badge.default_slot.children if hasattr(c, "text") and c.text][1:],  # type: ignore[attr-defined]
            badge.props["color"],
        )
        for badge in badges
    }
    assert shown == {"calculate": (["x 5"], "positive"), "no_such_tool": ([], "negative")}
    set_context(None)
