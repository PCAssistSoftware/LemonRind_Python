"""The assistant's face: an animated holographic cat that shows what the assistant is doing.

The cat itself is a self-contained web page (``assets/holo-cat.html``) shown in an ``<iframe>``. The page is told
what to do by messages (``{"type": "state", "value": "thinking"}``), so all this module has to do is decide which
message to send when:

* **listening** while you type,
* **thinking** from sending a message until the first words of the reply arrive, and while a tool runs,
* **speaking** while the reply streams in (the reply text is passed on, so the mouth moves with the words),
* **idle** otherwise, with the cat's own little habits,
* **sleep** when no chat model is loaded, or after a few minutes with nothing happening.

A reply that finishes well makes the cat happy for a moment; one that fails makes it annoyed.

The decisions live in ``AvatarBrain``, which only calls a ``send`` function, so they can be tested without a browser.
``Avatar`` is the screen part: the iframe, and the timers that drive the brain.

Python ideas used here:

* ``Literal`` types for the small fixed sets of states, moods and cats.
* Dependency injection: the brain is given ``send`` and ``clock`` functions instead of reaching for the screen and the
  system clock, which is what makes it easy to test.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from nicegui import app, ui

from lemonrind.chats import ToolStarted
from lemonrind.lemonade import ReasoningDelta, TextDelta

type AvatarState = Literal["idle", "listening", "thinking", "speaking", "sleep"]
type Mood = Literal["neutral", "happy", "curious", "annoyed"]

ASSETS = Path(__file__).parent / "assets"
URL_PREFIX = "/avatar"
PAGE = "holo-cat.html"

# The cats offered in Settings (the page also has a plain "classic" cat, kept but not offered).
CATS = {"genie": "Genie", "jess": "Jess", "sid": "Sid"}
POSITIONS = {"left": "Top of the chat list (left)", "right": "Top of the stats panel (right)"}

SLEEP_AFTER_SECONDS = 300  # nothing happening for this long: the cat goes to sleep
LISTEN_SECONDS = 4.0  # after the last key press, back to idle
MOOD_SECONDS = 6.0  # how long a happy or annoyed look lasts
TALK_AFTER_SECONDS = 2.5  # the mouth keeps going this long after the reply has arrived, then stops
ZOOM = 1.1  # the cat a little larger in its box: with no caption, the ring still fits
TEXT_SPEED = (
    45  # characters per second the mouth follows (the page's own default is a slow reading pace)
)


def serve_avatar() -> None:
    """Serve the cat page at ``/avatar`` (called once when the app starts)."""
    app.add_static_files(URL_PREFIX, ASSETS)


def avatar_url(cat: str) -> str:
    return f"{URL_PREFIX}/{PAGE}?debug=0&label=0&zoom={ZOOM}&cat={cat}&cps={TEXT_SPEED}"


class AvatarBrain:
    """Decides what the cat should be doing; ``send`` passes each message on to the page."""

    def __init__(
        self, send: Callable[[dict[str, Any]], None], clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._send = send
        self._clock = clock
        self.state: AvatarState = "idle"
        self.mood: Mood = "neutral"
        self.busy = False  # a reply is being worked on
        self.model_ready = (
            True  # a chat model is loaded (unknown counts as yes until a check says otherwise)
        )
        self._last_activity = clock()
        self._listening_until = 0.0
        self._mood_until = 0.0
        self._stop_talk_at = 0.0
        self._text = ""  # reply text not yet passed to the page

    # --- what happened ---------------------------------------------------------------------------------------

    def typing(self) -> None:
        """A key was pressed in the message box."""
        self._last_activity = self._clock()
        if self.busy or not self.model_ready:
            return
        self._listening_until = self._clock() + LISTEN_SECONDS
        self._set_state("listening")

    def turn_started(self) -> None:
        """A message was sent: think until the first words arrive."""
        self.busy = True
        self._last_activity = self._clock()
        self._text = ""
        self._stop_talk_at = 0.0
        self._send({"type": "textStop"})
        self._set_mood("neutral")
        self._set_state("thinking")

    def on_event(self, event: object) -> None:
        """One event from the conversation while the reply arrives."""
        match event:
            case TextDelta(text=piece):
                self._text += piece  # passed on in batches by ``tick``
                self.state = "speaking"  # the page switches itself when the first text arrives
            case ReasoningDelta() | ToolStarted():
                self.flush()
                if self.state == "speaking":
                    self._send({"type": "textStop"})
                self._set_state("thinking")

    def turn_finished(self, *, ok: bool) -> None:
        """The reply is complete (``ok``), or failed."""
        self.busy = False
        self._last_activity = self._clock()
        self.flush()
        if ok:
            self._send({"type": "textEnd"})
            self._stop_talk_at = self._clock() + TALK_AFTER_SECONDS
            self._set_mood("happy")
        else:
            self._send({"type": "textStop"})
            self._set_state("idle")
            self._set_mood("annoyed")
        self._mood_until = self._clock() + MOOD_SECONDS

    def turn_stopped(self) -> None:
        """The Stop button."""
        self.busy = False
        self._last_activity = self._clock()
        self._text = ""
        self._send({"type": "textStop"})
        self._set_state("idle")

    def set_model_ready(self, ready: bool) -> None:
        """Whether a chat model is loaded: with none, the cat sleeps."""
        self.model_ready = ready
        if self.busy:
            return  # sending a message loads the model; the reply decides the state
        if not ready:
            self._set_state("sleep")
        elif self.state == "sleep" and not self._sleepy():
            self._set_state("idle")

    # --- time passing ----------------------------------------------------------------------------------------

    def tick(self) -> None:
        """Called a few times a second: pass on new text, and end listening, moods, talking and wakefulness."""
        self.flush()
        now = self._clock()
        if self._stop_talk_at and now >= self._stop_talk_at:
            self._stop_talk_at = 0.0
            self._send({"type": "textStop"})
            self._set_state("idle")
        if self.state == "listening" and now >= self._listening_until:
            self._set_state("idle")
        if self.mood != "neutral" and now >= self._mood_until:
            self._set_mood("neutral")
        if not self.busy and self.state == "idle" and (not self.model_ready or self._sleepy()):
            self._set_state("sleep")

    def flush(self) -> None:
        if self._text:
            self._send({"type": "text", "value": self._text})
            self._text = ""

    def resend(self) -> None:
        """Tell a freshly loaded page the current state and mood."""
        self._send({"type": "state", "value": self.state})
        self._send({"type": "mood", "value": self.mood})

    # --- helpers ---------------------------------------------------------------------------------------------

    def _sleepy(self) -> bool:
        return self._clock() - self._last_activity >= SLEEP_AFTER_SECONDS

    def _set_state(self, state: AvatarState) -> None:
        if state != self.state:
            self.state = state
            self._send({"type": "state", "value": state})

    def _set_mood(self, mood: Mood) -> None:
        if mood != self.mood:
            self.mood = mood
            self._send({"type": "mood", "value": mood})


class Avatar:
    """The cat on screen: an iframe at the top of one of the side panels, driven by an ``AvatarBrain``."""

    def __init__(self, enabled: bool, cat: str) -> None:
        self.enabled = enabled
        self.cat = cat if cat in CATS else "genie"
        self.brain = AvatarBrain(self._post)
        self.root: ui.element | None = None
        self._frame: ui.element | None = None

    def build(self) -> None:
        """Create the cat in the current container (one of the panels)."""
        with ui.element("div").classes("lr-avatar").mark("avatar") as self.root:
            self._frame = ui.element("iframe").props(
                f'src="{avatar_url(self.cat)}" title="Assistant status"'
            )
        self.root.set_visibility(self.enabled)
        ui.timer(0.2, self.brain.tick)
        # The page needs a moment to load before it can take messages: then tell it where things stand.
        ui.timer(2.0, self.brain.resend, once=True)

    def move_to(self, container: ui.element) -> None:
        """Put the cat at the top of the other panel. A browser reloads a moved iframe, so it is told the state again."""
        if self.root is not None and self.root.parent_slot is not None:
            if self.root.parent_slot.parent is container:
                return
            self.root.move(container, target_index=0)
            ui.timer(2.0, self.brain.resend, once=True)

    def apply(self, *, enabled: bool, cat: str) -> None:
        """New settings: show or hide, and change the cat."""
        was_enabled = self.enabled
        self.enabled = enabled
        if self.root is not None:
            self.root.set_visibility(enabled)
        if cat in CATS and cat != self.cat and self._frame is not None:
            self.cat = cat
            # A new address (not just a message) so the page keeps this cat if it is ever reloaded.
            self._frame.props(f'src="{avatar_url(cat)}"')
            ui.timer(2.0, self.brain.resend, once=True)
        elif enabled and not was_enabled:
            self.brain.resend()  # messages are not sent while hidden: catch the page up

    def _post(self, message: dict[str, Any]) -> None:
        if not self.enabled or self._frame is None or self._frame.is_deleted:
            return
        code = (
            f"(function(f){{if(f&&f.contentWindow)f.contentWindow.postMessage({json.dumps(message)},'*');}})"
            f"(document.getElementById('c{self._frame.id}'))"
        )
        self._frame.client.run_javascript(code)
