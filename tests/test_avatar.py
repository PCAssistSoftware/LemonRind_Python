"""The assistant's cat: what it does when (the brain), and where it sits on the page (the settings)."""

from __future__ import annotations

from typing import Any

from nicegui import ui
from nicegui.testing import User

from lemonrind.chats import ToolStarted
from lemonrind.lemonade import ReasoningDelta, TextDelta, ToolCall
from lemonrind.webui.avatar import (
    LISTEN_SECONDS,
    MOOD_SECONDS,
    SLEEP_AFTER_SECONDS,
    TALK_AFTER_SECONDS,
    AvatarBrain,
)
from lemonrind.webui.context import AppContext
from tests.test_settings_sections import box, open_section, save, saved_settings


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def brain() -> tuple[AvatarBrain, list[dict[str, Any]], Clock]:
    sent: list[dict[str, Any]] = []
    clock = Clock()
    return AvatarBrain(sent.append, clock), sent, clock


def states(sent: list[dict[str, Any]]) -> list[str]:
    return [m["value"] for m in sent if m["type"] == "state"]


# --- the brain -------------------------------------------------------------------------------------------------------


def test_typing_makes_it_listen_until_you_pause():
    cat, sent, clock = brain()
    cat.typing()
    assert cat.state == "listening"
    clock.now += LISTEN_SECONDS + 0.1
    cat.tick()
    assert states(sent) == ["listening", "idle"]


def test_a_reply_goes_thinking_speaking_and_back_to_idle_happy():
    cat, sent, clock = brain()
    cat.turn_started()
    assert cat.state == "thinking"
    cat.on_event(ReasoningDelta("hmm"))
    cat.on_event(TextDelta("Hello "))
    cat.on_event(TextDelta("there."))
    cat.tick()
    assert {"type": "text", "value": "Hello there."} in sent  # batched, not one message per piece
    cat.turn_finished(ok=True)
    assert {"type": "textEnd"} in sent and cat.mood == "happy"
    clock.now += TALK_AFTER_SECONDS + 0.1
    cat.tick()
    assert cat.state == "idle"
    clock.now += MOOD_SECONDS
    cat.tick()
    assert cat.mood == "neutral"


def test_a_tool_step_goes_back_to_thinking():
    cat, sent, _ = brain()
    cat.turn_started()
    cat.on_event(TextDelta("Let me look that up."))
    cat.on_event(ToolStarted(ToolCall("1", "search", "{}")))
    assert cat.state == "thinking"
    assert {"type": "text", "value": "Let me look that up."} in sent  # said before the tool ran


def test_a_failed_reply_is_annoying_and_stop_goes_idle():
    cat, _, _ = brain()
    cat.turn_started()
    cat.turn_finished(ok=False)
    assert (cat.state, cat.mood) == ("idle", "annoyed")
    cat.turn_started()
    cat.turn_stopped()
    assert cat.state == "idle" and not cat.busy


def test_it_sleeps_while_no_model_is_loaded_and_wakes_when_one_is():
    cat, sent, _ = brain()
    cat.set_model_ready(False)
    assert cat.state == "sleep"
    cat.typing()  # no model: typing does not wake it
    assert cat.state == "sleep"
    cat.set_model_ready(True)
    assert states(sent) == ["sleep", "idle"]


def test_a_reply_in_progress_is_not_put_to_sleep_by_a_model_check():
    cat, _, _ = brain()
    cat.turn_started()
    cat.set_model_ready(False)  # Lemonade is loading the model for this very reply
    assert cat.state == "thinking"


def test_it_dozes_off_after_a_quiet_spell_and_typing_wakes_it():
    cat, _, clock = brain()
    clock.now += SLEEP_AFTER_SECONDS + 1
    cat.tick()
    assert cat.state == "sleep"
    cat.typing()
    assert cat.state == "listening"


def test_a_freshly_loaded_page_is_told_the_state_and_mood():
    cat, sent, _ = brain()
    cat.set_model_ready(False)
    sent.clear()
    cat.resend()
    assert sent == [{"type": "state", "value": "sleep"}, {"type": "mood", "value": "neutral"}]


# --- the page --------------------------------------------------------------------------------------------------------


def panel_of(element: ui.element) -> str:
    """Which side panel an element is in: 'left', 'right' or ''."""
    current: ui.element | None = element
    while current is not None:
        name = type(current).__name__
        if name == "LeftDrawer":
            return "left"
        if name == "RightDrawer":
            return "right"
        current = current.parent_slot.parent if current.parent_slot else None
    return ""


async def test_the_cat_starts_at_the_top_left_and_settings_move_hide_and_change_it(
    user: User, web: AppContext
):
    await user.open("/")
    (avatar,) = user.find(marker="avatar").elements
    assert panel_of(avatar) == "left" and avatar.visible
    frame = avatar.default_slot.children[0]
    assert "cat=genie" in frame.props["src"]

    await open_section(user, "interface")
    box(user, "settings-avatar-position").value = "right"
    box(user, "settings-avatar-cat").value = "sid"
    await save(user)

    assert panel_of(avatar) == "right"
    assert "cat=sid" in frame.props["src"]
    assert saved_settings(web)["ui"]["avatar_position"] == "right"

    await open_section(user, "interface")
    box(user, "settings-avatar").value = False
    await save(user)
    assert not avatar.visible
