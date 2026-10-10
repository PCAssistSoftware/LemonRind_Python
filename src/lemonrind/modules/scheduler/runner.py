"""Running one scheduled job: its prompt goes through the assistant exactly as if you had typed it.

What "exactly as if you typed it" means here: the same ``Conversation`` that powers the chat (so the same
tools, memory and rules), in a **fresh chat of its own**, so a job firing never disturbs the chat you have open.
The chat is named after the job and tagged ``scheduled``, so it is easy to find.

What differs, because nobody is watching:

* **Limits.** A time limit for the whole run, a generous cap on tool rounds, and a cap on each reply's length,
  so a model stuck in a loop cannot run forever. When the round cap is reached, a note is added to the chat
  saying so, because a cut-off answer otherwise looks like a complete one.
* **No questions.** Tools that normally ask your permission cannot ask. They are *refused*, unless you switched
  on "may run tools without asking" for that job (an informed choice you make per job; a job the *assistant*
  created for you never gets it).
* **Failures become chats.** If the run fails (Lemonade down, a context overflow, a timeout), a chat is still
  created, holding your prompt and a plain explanation with a hint, so you can see what happened.
* **The date.** The model is told the current date and time, because a job such as "summarise this week" is
  meaningless without it.

Python ideas used here:

* ``async with asyncio.timeout(...)`` around a whole run; ``TimeoutError`` as the signal.
* A ``dataclass`` result type (``JobOutcome``) rather than returning a tuple.
* A **circular import** and its fix: ``chats`` needs ``modules`` and ``modules`` needs ``chats``. Python cannot
  finish importing either first. Importing one of them *inside the function that uses it* breaks the loop
  (``typing.TYPE_CHECKING`` keeps the names available for type hints).
* A *provider function* (``registry``) that is called when needed, which breaks a circular dependency (the
  scheduler is itself a module in the registry it needs to use).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from typing import TYPE_CHECKING, Any

from lemonrind.config import Settings
from lemonrind.lemonade.client import LemonadeError
from lemonrind.lemonade.selection import ModelSelectionError, pick_chat_model
from lemonrind.modules.registry import ModuleRegistry
from lemonrind.modules.scheduler.live import LiveRuns
from lemonrind.modules.scheduler.repository import ScheduledJob

if (
    TYPE_CHECKING
):  # only for the type hints below; see the note in ``run`` about why not a normal import
    from lemonrind.chats import ChatRepository, ChatSession, Conversation

logger = logging.getLogger(__name__)

SCHEDULED_TAG = "scheduled"
FAILED_TAG = "failed"
RUNNING_TAG = (
    "running"  # on a run's chat while it is still going, so the chat list shows it at once
)
ATTEMPTS = 2  # a model sometimes answers with nothing at all; one more try is usually enough


@dataclass(frozen=True, slots=True)
class JobOutcome:
    status: str  # "ok", "failed", "timed out"
    session_id: str | None  # the chat holding the result (or the explanation of the failure)
    message: str = ""  # a one-line summary for listings and notices
    seconds: float | None = None  # how long the run took


async def _allow_everything(call: Any) -> bool:
    return True


def explain_failure(error: str) -> str:
    """A hint for the most common reasons a run fails."""
    lowered = error.lower()
    if "exceeds the available context size" in lowered or (
        "context" in lowered and "exceed" in lowered
    ):
        return (
            "Try loading this model with a larger context size, switching off modules this job does not "
            "need, or shortening its prompt."
        )
    if "timed out" in lowered or "timeout" in lowered:
        return (
            "This usually means a request grew very large before Lemonade could answer, often a slow or "
            "overly broad tool call. Check the job's tools, or raise the request timeout in the settings."
        )
    return "Check that Lemonade is running a chat model, and that the modules this job uses are set up."


class JobRunner:
    def __init__(
        self,
        settings: Settings,
        chats: ChatRepository,
        client: Any,
        registry: Callable[[], ModuleRegistry | None],
        clock: Callable[[], datetime] = lambda: datetime.now().astimezone(),
    ) -> None:
        self._settings = settings
        self._chats = chats
        self._client = client  # a LemonadeClient: streaming chat, health and the model list
        self._registry = (
            registry  # a function, because the registry is built after the modules it holds
        )
        self._clock = clock
        self.live = LiveRuns()  # the runs going now, for a page that wants to watch one

    async def run(self, job: ScheduledJob) -> JobOutcome:
        """Run a job. Its chat is created *first* (tagged ``scheduled`` and ``running``) so it shows in the chat list while
        the job works, which can take minutes; the ``running`` tag is removed however the run ends.

        If the run is cancelled (the Stop button, or the app closing) a note says so in the chat and a ``stopped``
        outcome is returned instead of the cancellation being raised again: the caller that cancelled it already knows.
        """
        from lemonrind.chats import format_duration  # here, not at the top: see the note above

        session = self._chats.create_session(job.name)
        self._chats.add_tag(session.id, SCHEDULED_TAG)
        self._chats.add_tag(session.id, RUNNING_TAG)
        self.live.start(session.id, job.prompt, job.name)
        started = time.monotonic()
        try:
            try:
                outcome = await self._execute(job, session)
            except asyncio.CancelledError:
                self._add_note(job, session, "This run was stopped before it finished.")
                outcome = JobOutcome("stopped", session.id, "Stopped.")
            seconds = time.monotonic() - started
            ended = self._clock().astimezone()
            self._chats.add_message(
                session.id,
                "assistant",
                f"This run ended at {ended:%H:%M} on {ended:%d-%m-%Y} and took {format_duration(seconds)}.",
            )
            return replace(outcome, seconds=seconds)
        finally:
            self._chats.remove_tag(session.id, RUNNING_TAG)
            self.live.end(
                session.id
            )  # after everything is saved: followers then switch to the saved chat

    def clear_running_tags(self) -> None:
        """At start-up: a chat still tagged ``running`` belongs to a run that was cut off by the app closing."""
        for session in self._chats.list_sessions():
            if RUNNING_TAG in session.tags:
                self._chats.remove_tag(session.id, RUNNING_TAG)
                self._add_note(None, session, "The app stopped while this job was running.")

    async def _execute(self, job: ScheduledJob, session: ChatSession) -> JobOutcome:
        # Imported here, not at the top of the file: the chat package imports the modules package (for the
        # tool types), and the modules package imports this file, so a top-level import would be circular.
        from lemonrind.chats import CONTINUE_PROMPT, Conversation

        config = self._settings.modules.scheduler
        conversation: Conversation | None = None
        try:
            model = await self._choose_model(job)
            if (live_run := self.live.get(session.id)) is not None:
                live_run.model = model  # so the page can tell which model this run is using
            registry = self._registry()
            conversation = Conversation(
                client=self._client,
                repo=self._chats,
                system_prompt=self._system_prompt(),
                model=model,
                tools=registry,
                context=registry,
                max_rounds=config.max_tool_rounds,
            )
            conversation.max_output_tokens = config.max_output_tokens
            conversation.session = (
                session  # the messages are saved into the chat that already exists
            )
            if job.allow_unattended_tools:
                conversation.approve = (
                    _allow_everything  # otherwise tools that need permission are refused
                )

            reply = None
            continues = 0
            async with asyncio.timeout(config.job_timeout_seconds):
                for _ in range(ATTEMPTS):
                    live = self.live.get(session.id)
                    reply = await conversation.send(job.prompt, live.publish if live else None)
                    if reply.text:
                        break
                # A reply that stopped at the output limit is carried on, so a long report arrives whole.
                while (
                    reply is not None
                    and reply.text
                    and reply.finish_reason == "length"
                    and continues < config.max_continuations
                ):
                    continues += 1
                    live = self.live.get(session.id)
                    follow = await conversation.send(
                        CONTINUE_PROMPT, live.publish if live else None
                    )
                    if not follow.text:  # it used the whole limit thinking again: nothing to add
                        break
                    reply = follow
            await conversation.drain(60)  # let background learning (memory) finish
            if reply is None or not reply.text:
                return self._failure(job, session, "The model produced no answer.")

            self._chats.rename_session(session.id, job.name)
            if reply.finish_reason == "length":
                # The model hit the limit on one reply, so what it wrote stops part way (a report cut off in the
                # middle of a table, say). Calling that "ok" would hide it.
                tried = (
                    f" even after {continues} automatic continue{'s' if continues != 1 else ''}"
                    if continues
                    else ""
                )
                self._chats.add_message(
                    session.id,
                    "assistant",
                    f"Warning: the model's reply was cut off at the output limit ({config.max_output_tokens:,} tokens, "
                    f"which includes its thinking){tried}, so it is probably incomplete. Raise 'Longest reply' in "
                    "Settings > Scheduler, use a model that thinks less, or ask for a shorter report.",
                )
                return JobOutcome("cut short", session.id, "the reply reached the output limit")
            if continues:
                self._chats.add_message(
                    session.id,
                    "assistant",
                    f"Note: the reply reached the output limit and was continued automatically {continues} "
                    f"time{'s' if continues != 1 else ''}, so it arrives in {continues + 1} parts.",
                )
            if len(reply.round_stats) >= config.max_tool_rounds:
                self._chats.add_message(
                    session.id,
                    "assistant",
                    f"Warning: this job used its maximum of {config.max_tool_rounds} tool rounds, so it may "
                    "have stopped before finishing. Try splitting it into smaller jobs, or ask it to search "
                    "and read less.",
                )
            return JobOutcome("ok", session.id, "finished")
        except TimeoutError:
            minutes = config.job_timeout_seconds / 60
            return self._failure(
                job,
                session,
                f"This job did not finish within its time limit ({minutes:.0f} minutes) and was stopped. "
                "This usually means the model got stuck reasoning in a loop, or the job took unusually long.",
                status="timed out",
                hint="Try simplifying the prompt, switching off tools it does not need, or asking for a "
                "shorter answer.",
            )
        except (LemonadeError, ModelSelectionError) as error:
            return self._failure(job, session, str(error), hint=explain_failure(str(error)))
        except Exception as error:  # a bug: still leave an honest record rather than nothing
            logger.exception("Scheduled job %s crashed", job.name)
            return self._failure(job, session, f"{type(error).__name__}: {error}")

    # --- helpers ---------------------------------------------------------------------------------------------

    async def _choose_model(self, job: ScheduledJob) -> str:
        """The job's own model if it names one, else the usual rule for picking a chat model."""
        health = await self._client.health()
        models = await self._client.list_models()
        return pick_chat_model(
            job.model, health, models, default=self._settings.lemonade.chat_model
        )

    def _system_prompt(self) -> str:
        from lemonrind.chats import (
            build_system_prompt,  # imported here for the circular-import reason in ``run``
        )

        now = self._clock()
        return (
            f"{build_system_prompt(self._settings)}\n\n"
            f"Current date and time: {now:%A %d %B %Y, %H:%M} ({now.tzname()}). "
            "You are running as a scheduled job: nobody is watching, so do the task completely and finish with "
            "a clear final answer."
        )

    def _failure(
        self,
        job: ScheduledJob,
        session: ChatSession,
        reason: str,
        *,
        status: str = "failed",
        hint: str = "",
    ) -> JobOutcome:
        """Record a failed run as a chat: your prompt, then what went wrong. (If the run had already got as far
        as saving part of a chat, the explanation is added to that chat instead.)"""
        text = f"This scheduled job failed: {reason.rstrip('. ')}."
        if status == "timed out":
            text = reason
        if hint:
            text += f"\n\n{hint}"
        self._add_note(job, session, text)
        self._chats.add_tag(session.id, FAILED_TAG)
        return JobOutcome(status, session.id, reason)

    def _add_note(self, job: ScheduledJob | None, session: ChatSession, text: str) -> None:
        """Add an assistant message to a run's chat (with your prompt first if nothing had been saved yet)."""
        if job is not None and not self._chats.list_messages(session.id):
            self._chats.add_message(session.id, "user", job.prompt)
        self._chats.add_message(session.id, "assistant", text)
        if job is not None:
            self._chats.rename_session(session.id, job.name)
