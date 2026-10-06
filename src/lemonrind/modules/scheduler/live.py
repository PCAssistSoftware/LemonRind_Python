"""Watching a scheduled run while it works.

A scheduled job's messages are saved to its chat only when the run finishes (the same rule as any chat: nothing
is stored until there is a complete exchange). To let you *watch* a run, the runner also passes every event it gets
(thinking, text, tool calls and results) to a ``LiveRun``, which keeps them in memory and hands them to anyone
following the run. A page that opens the running chat first catches up from the kept events, then receives new ones
as they happen, and shows them with the same widgets as a normal chat. When the run ends the page switches to the
saved chat, and the in-memory copy is dropped.

Python ideas used here:

* Plain callbacks (``Callable``) as the observer pattern: ``subscribe`` returns a function that undoes it.
* ``dataclasses.replace``-style merging of tiny events: text arrives a few characters at a time, so neighbouring
  ``TextDelta`` events are joined, which keeps the memory use of a long run small.
* A ``dict`` keyed by chat id as the registry of runs that are going now.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from lemonrind.lemonade.events import PromptProgress, ReasoningDelta, TextDelta

logger = logging.getLogger(__name__)

MAX_EVENTS = (
    3000  # a run is capped here, oldest first, so a very long run cannot use unlimited memory
)
TRIM_TO = 2000


class LiveRun:
    """The events of one run that is going now, and the pages following it."""

    def __init__(self, session_id: str, prompt: str) -> None:
        self.session_id = session_id
        self.prompt = prompt
        self.events: list[object] = []
        self.finished = False
        self._followers: dict[int, tuple[Callable[[object], None], Callable[[], None]]] = {}
        self._next_id = 0

    def publish(self, event: object) -> None:
        """Keep an event and pass it to every follower. A follower that fails never disturbs the run."""
        self._keep(event)
        for _, (on_event, _on_finished) in list(self._followers.items()):
            try:
                on_event(event)
            except Exception:  # the page may have been closed: nothing here may stop the job
                logger.debug("A follower of a live run failed", exc_info=True)

    def _keep(self, event: object) -> None:
        last = self.events[-1] if self.events else None
        if isinstance(event, TextDelta | ReasoningDelta) and type(last) is type(event):
            self.events[-1] = type(event)(last.text + event.text)  # type: ignore[union-attr]
        elif isinstance(event, PromptProgress) and isinstance(last, PromptProgress):
            self.events[-1] = event  # only the latest progress line matters
        else:
            self.events.append(event)
        if len(self.events) > MAX_EVENTS:
            del self.events[: len(self.events) - TRIM_TO]

    def subscribe(
        self, on_event: Callable[[object], None], on_finished: Callable[[], None]
    ) -> Callable[[], None]:
        """Follow the run. Returns a function that stops following."""
        follower_id = self._next_id
        self._next_id += 1
        self._followers[follower_id] = (on_event, on_finished)
        return lambda: self._followers.pop(follower_id, None) and None

    def finish(self) -> None:
        self.finished = True
        for _, (_on_event, on_finished) in list(self._followers.items()):
            try:
                on_finished()
            except Exception:
                logger.debug("A follower of a live run failed", exc_info=True)
        self._followers.clear()


class LiveRuns:
    """The runs going now, by the id of their chat."""

    def __init__(self) -> None:
        self._runs: dict[str, LiveRun] = {}

    def start(self, session_id: str, prompt: str) -> LiveRun:
        run = LiveRun(session_id, prompt)
        self._runs[session_id] = run
        return run

    def get(self, session_id: str) -> LiveRun | None:
        return self._runs.get(session_id)

    def end(self, session_id: str) -> None:
        """The run is over: tell its followers, and forget it."""
        if run := self._runs.pop(session_id, None):
            run.finish()
