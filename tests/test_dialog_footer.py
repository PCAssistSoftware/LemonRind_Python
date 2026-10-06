"""The view dialogs keep their Close button outside the scrolling part, so it is always on screen."""

from __future__ import annotations

import pytest
from nicegui import ui
from nicegui.testing import User

from lemonrind.webui.context import AppContext
from tests.conftest import elements, must


def ancestors(element: ui.element):
    """The element's parents, nearest first, up to the top of its page."""
    parent = element.parent_slot.parent if element.parent_slot else None
    while parent is not None:
        yield parent
        parent = parent.parent_slot.parent if parent.parent_slot else None


@pytest.mark.parametrize("button", ["view-tools", "view-usage", "view-prompt"])
async def test_the_close_button_is_not_inside_the_scrolling_body(
    user: User, web: AppContext, button: str
):
    await user.open("/")
    user.find(marker=button).click()
    await user.should_see(marker="dialog-close")

    (close,) = elements(user, "dialog-close")

    scrolling = [a for a in ancestors(close) if "overflow-auto" in a.classes]
    assert scrolling == []  # nothing that scrolls contains the button
    card = next(a for a in ancestors(close) if isinstance(a, ui.card))
    bodies = [c for c in card.descendants() if "overflow-auto" in c.classes]
    assert (
        len(bodies) >= 1 and close not in bodies[0].descendants()
    )  # there is a body, and the button is not in it
    assert "max-h-[90vh]" in card.classes and must(
        card.parent_slot
    )  # the card cannot outgrow the window
