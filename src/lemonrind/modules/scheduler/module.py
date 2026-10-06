"""The Scheduler module: run saved prompts on a schedule.

You can create a job from a chat ("remind me every Monday at 9 to check the backups": the model uses the
``schedule_job`` tool) or from the Scheduler screen. When a job is due, ``JobRunner`` sends its prompt
through the assistant in a chat of its own.

**APScheduler decides when jobs fire.** The jobs themselves still live in our own ``scheduled_jobs`` table (the
screens, the model's tools and the run history all read it), and APScheduler holds an in-memory copy of "what to
run, and when". At start-up every enabled job is loaded into it; adding, editing, switching or deleting a job
updates both. What APScheduler brings over the small polling loop this replaced:

* **It sleeps until the next job is due** instead of waking every half minute to look.
* **Misfire handling is built in.** A job that came due while the app was closed is handed over with its old due time;
  APScheduler runs it once if it is overdue by less than ``misfire_grace_time`` (our ``missed_grace_minutes``), and
  otherwise skips it and tells us (``EVENT_JOB_MISSED``), which we record as ``missed``. Several missed firings are
  *coalesced* into one run.
* **A job cannot overlap itself** (``max_instances=1``): if a run is still going when the next one is due, that firing
  is skipped, not queued.
* Time zones, jitter and persistent job stores are available for later.

What is still ours:

* **One job at a time across all jobs.** A local model has a limited number of generation slots; two jobs (or a job and
  you) at once would only slow each other down. A lock serialises every run, including manual "Run now".
* **Stopping.** Closing the app cancels a running job and records it as ``interrupted``.
* The jobs only run while the app is running (no background service, no system scheduler).

The tools ask for your permission (``schedule_job`` is marked as needing approval): a job runs unattended and
repeatedly, so a prompt-injected "schedule something sneaky" must not happen silently.

Python ideas used here:

* Using a **scheduling library**: a scheduler object, triggers, job options and event listeners.
* ``asyncio.Lock`` to serialise work; ``try / finally`` to record state even when a run is cancelled at shutdown.
* Injecting **time** (``clock``) so tests can say "it is now Friday 19:00".
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime

from apscheduler.events import EVENT_JOB_MAX_INSTANCES, EVENT_JOB_MISSED, JobEvent
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from lemonrind.config import Settings
from lemonrind.modules.base import Module
from lemonrind.modules.scheduler import cron
from lemonrind.modules.scheduler.repository import (
    DuplicateJobError,
    ScheduledJob,
    SchedulerRepository,
)
from lemonrind.modules.scheduler.runner import JobOutcome, JobRunner
from lemonrind.modules.tool import Tool, ToolError, tool_from_function

MIN_GRACE_SECONDS = (
    60  # a firing is always a few milliseconds late; a grace of 0 would call every one "missed"
)

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(UTC)


class SchedulerModule(Module):
    name = "Scheduler"
    config_key = "scheduler"
    description = "Run saved prompts through the assistant on a cron schedule (for example every Friday at 19:00)."

    def __init__(
        self,
        settings: Settings,
        repo: SchedulerRepository,
        runner: JobRunner,
        *,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        super().__init__(settings)
        self.repo = repo
        self._runner = runner
        self._clock = clock
        self._scheduler: AsyncIOScheduler | None = None  # exists only while the module is running
        self._runs: set[asyncio.Task] = (
            set()
        )  # runs started by the scheduler, so shutdown can cancel them
        self._run_lock = asyncio.Lock()  # one job at a time
        self._background: set[asyncio.Task] = set()  # manual runs started from a screen
        self.runs_started = 0  # increases when a run begins, so screens can show its chat at once
        self.runs_finished = (
            0  # increases after every run: screens watch it to know when to refresh
        )
        self._active: tuple[str, asyncio.Task] | None = None  # (job id, the task doing its run)
        self._stopped: str | None = None  # the job whose Stop button was pressed
        self.last_outcome: tuple[str, JobOutcome] | None = (
            None  # (job name, outcome) of the latest run
        )

    # --- lifecycle ---------------------------------------------------------------------------------------------

    async def on_startup(self) -> None:
        self.repo.fail_interrupted()  # a run cut off by the app closing last time
        self._runner.clear_running_tags()
        # Created here, not in __init__: the scheduler attaches itself to the event loop that is running now.
        scheduler = AsyncIOScheduler(
            timezone=cron.local_timezone(),
            job_defaults={"coalesce": True, "max_instances": 1},
        )
        scheduler.add_listener(self._on_scheduler_event, EVENT_JOB_MISSED | EVENT_JOB_MAX_INSTANCES)
        self._scheduler = scheduler
        for job in self.repo.list():
            self._schedule(job, resume=True)
        scheduler.start()

    async def on_shutdown(self) -> None:
        scheduler, self._scheduler = self._scheduler, None
        if scheduler is not None:
            scheduler.shutdown(wait=False)  # also cancels runs that are in progress
        tasks = [*self._runs, *self._background]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.wait(
                tasks, timeout=15
            )  # lets a running job record that it was interrupted

    def _grace_seconds(self) -> int:
        minutes = self.settings.modules.scheduler.missed_grace_minutes
        return max(MIN_GRACE_SECONDS, minutes * 60)

    def _schedule(self, job: ScheduledJob, *, resume: bool = False) -> None:
        """Tell APScheduler about a job (replacing what it knew), if the module is running and the job is on.

        With ``resume`` (used at start-up) the job keeps the due time stored from last time, which may be in the
        past: that is how a firing missed while the app was closed is noticed. Otherwise APScheduler works out
        the next time from now, so a job that is switched on or edited never "catches up".
        """
        if self._scheduler is None or not job.enabled:
            return
        try:
            trigger = cron.trigger_for(job.cron)
        except cron.CronError as error:  # saved by an older version, for instance
            logger.warning("Not scheduling '%s': %s", job.name, error)
            return
        options = {"next_run_time": job.next_run_at} if resume and job.next_run_at else {}
        self._scheduler.add_job(
            self._fire,
            trigger,
            args=[job.id],
            id=job.id,
            name=job.name,
            replace_existing=True,
            misfire_grace_time=self._grace_seconds(),
            **options,
        )

    def _unschedule(self, job_id: str) -> None:
        if self._scheduler is not None and self._scheduler.get_job(job_id) is not None:
            self._scheduler.remove_job(job_id)

    def _next_time(self, job: ScheduledJob) -> datetime | None:
        """When the job fires next: APScheduler's own answer if it knows the job, else worked out from the clock."""
        if self._scheduler is not None:
            known = self._scheduler.get_job(job.id)
            if known is not None and known.next_run_time is not None:
                return known.next_run_time.astimezone(UTC)
        return cron.next_run(job.cron, self._clock())

    def _on_scheduler_event(self, event: JobEvent) -> None:
        """APScheduler skipped a firing: record it (this runs on the event loop thread, where SQLite is usable)."""
        job = self.repo.get(event.job_id)
        if job is None:
            return
        if event.code == EVENT_JOB_MISSED:
            self.repo.record_run(
                job.id, status="missed", session_id=None, next_run_at=self._next_time(job)
            )
            logger.info("Skipped missed job %s (it was due %s)", job.name, job.next_run_at)
        else:
            logger.info("Skipped a run of %s: the previous run is still going", job.name)

    # --- running jobs ------------------------------------------------------------------------------------------

    async def _fire(self, job_id: str) -> None:
        """What APScheduler calls when a job is due."""
        task = asyncio.current_task()
        if task is not None:
            self._runs.add(task)
        try:
            job = self.repo.get(job_id)
            if job is not None and job.enabled:  # deleted or switched off since it was scheduled
                await self._run(job, scheduled=True)
        except asyncio.CancelledError:
            # The app is closing (Ctrl+C) or the module was switched off while the job ran. This is the top of the
            # task, so ending quietly is right: re-raising would make APScheduler log it as a failed job, with a
            # long traceback, for what is a normal stop. The run's own chat already says it was stopped.
            logger.info("The run of job %s was cancelled", job_id)
        finally:
            if task is not None:
                self._runs.discard(task)

    async def run_now(self, job_id: str) -> JobOutcome | None:
        """Run a job right now (the "Run now" button). Its normal schedule is left as it was."""
        job = self.repo.get(job_id)
        if job is None:
            return None
        return await self._run(job, scheduled=False)

    @property
    def live(self):  # type: ignore[no-untyped-def]
        """The runs going now (``LiveRuns``): a page can follow one while it works."""
        return self._runner.live

    def stop_job(self, job_id: str) -> bool:
        """Stop the run of this job if it is running now. Returns whether there was one to stop."""
        if self._active is None or self._active[0] != job_id or self._active[1].done():
            return False
        self._stopped = job_id
        self._active[1].cancel()
        return True

    def start_now(self, job_id: str) -> None:
        """Start "Run now" in the background and return at once (for a button press)."""
        task = asyncio.create_task(self.run_now(job_id))
        self._background.add(task)  # keep a reference so the task is not garbage-collected mid-run
        task.add_done_callback(self._background.discard)

    async def _run(self, job: ScheduledJob, *, scheduled: bool) -> JobOutcome:
        async with self._run_lock:
            self.repo.mark_running(job.id)
            self.runs_started += 1
            outcome = JobOutcome("interrupted", None, "The app stopped while this job was running.")
            # The run is its own task so the Stop button can cancel just that, not whatever started it.
            work = asyncio.create_task(self._runner.run(job))
            self._active = (job.id, work)
            try:
                outcome = await work
                return outcome
            except asyncio.CancelledError:
                task = asyncio.current_task()
                if self._stopped == job.id and task is not None and task.cancelling() == 0:
                    # The Stop button cancelled the work and nothing is cancelling this task: not an app shutdown.
                    outcome = JobOutcome("stopped", None, "Stopped.")
                    return outcome
                raise
            finally:
                # also when cancelled at shutdown: the job must not stay "running" forever
                self._active = None
                self._stopped = None
                self.repo.record_run(
                    job.id,
                    status=outcome.status,
                    session_id=outcome.session_id,
                    next_run_at=self._next_time(job) if scheduled else None,
                    update_schedule=scheduled,
                )
                self.last_outcome = (job.name, outcome)
                self.runs_finished += 1

    # --- managing jobs (the screens and the tools) ---------------------------------------------------------------

    def add_job(
        self,
        name: str,
        schedule: str,
        prompt: str,
        *,
        model: str = "",
        allow_unattended_tools: bool = False,
    ) -> ScheduledJob:
        """Create a job. Raises ``CronError`` for a bad schedule and ``DuplicateJobError`` for a repeated name."""
        name, prompt = name.strip(), prompt.strip()
        if not name or not prompt:
            raise ValueError("A job needs a name and a prompt.")
        expression = cron.normalise(schedule)
        job = self.repo.create(
            name,
            expression,
            prompt,
            cron.next_run(expression, self._clock()),
            model=model,
            allow_unattended_tools=allow_unattended_tools,
        )
        self._schedule(job)
        return job

    def edit_job(
        self,
        job_id: str,
        *,
        name: str,
        schedule: str,
        prompt: str,
        model: str,
        allow_unattended_tools: bool,
    ) -> None:
        job = self.repo.get(job_id)
        if job is None:
            raise ValueError("That job no longer exists.")
        expression = cron.normalise(schedule)
        next_at = (
            job.next_run_at if expression == job.cron else cron.next_run(expression, self._clock())
        )
        self.repo.update(
            job_id,
            name=name.strip(),
            cron=expression,
            prompt=prompt.strip(),
            model=model,
            allow_unattended_tools=allow_unattended_tools,
            next_run_at=next_at,
        )
        edited = self.repo.get(job_id)
        if edited is not None:
            self._unschedule(job_id)
            self._schedule(edited)

    def set_enabled(self, job_id: str, enabled: bool) -> None:
        job = self.repo.get(job_id)
        if job is None:
            return
        self.repo.set_enabled(job_id, enabled, cron.next_run(job.cron, self._clock()))
        self._unschedule(job_id)
        switched = self.repo.get(job_id)
        if enabled and switched is not None:
            self._schedule(switched)  # from now: no catching up on what was missed while it was off

    def delete_job(self, job_id: str) -> None:
        self._unschedule(job_id)
        self.repo.delete(job_id)

    # --- tools (so the model can schedule things from a chat) --------------------------------------------------
    #
    # These are ``async def`` on purpose, although they never wait for anything. Plain (non-async) tools are run
    # on a worker thread so a slow one cannot freeze the app, but a SQLite connection may only be used by the
    # thread that created it. An ``async`` tool runs on the main thread, where the database lives.

    def get_tools(self) -> list[Tool]:
        schedule = tool_from_function(self.schedule_job)
        # A job runs unattended and repeatedly, so creating one is something you should be asked about.
        schedule = Tool(
            schedule.name,
            schedule.description,
            schedule.parameters,
            schedule.func,
            schedule.args_model,
            requires_approval=True,
        )
        return [
            schedule,
            tool_from_function(self.list_scheduled_jobs),
            tool_from_function(self.cancel_scheduled_job),
        ]

    async def schedule_job(self, name: str, cron_expression: str, prompt: str) -> str:
        """Create a scheduled job that runs a prompt through the assistant (with its tools: web search, files, email and so on) on a recurring schedule.

        Args:
            name: A short unique name for the job, e.g. "Monday backup check".
            cron_expression: A standard 5-field cron expression (minute hour day-of-month month day-of-week) in the user's local time, e.g. "0 19 * * 5" for 7pm every Friday.
            prompt: What the assistant should do each time, written as a complete instruction that makes sense with no other context.
        """
        try:
            job = self.add_job(
                name, cron_expression, prompt
            )  # never given permission to use tools unasked
        except (cron.CronError, DuplicateJobError, ValueError) as error:
            raise ToolError(f"Could not schedule that: {error}") from error
        if job.next_run_at is None:
            return (
                f"Scheduled '{job.name}', but that schedule never fires again, so it will not run."
            )
        return (
            f"Scheduled '{job.name}'. Next run: {cron.format_local(job.next_run_at)} (local time)."
        )

    async def list_scheduled_jobs(self) -> str:
        """List every scheduled job with its schedule and when it runs next."""
        jobs = self.repo.list()
        if not jobs:
            return "There are no scheduled jobs."
        return "\n".join(
            f"- '{job.name}' ({'on' if job.enabled else 'off'}): {job.cron}, next run "
            f"{cron.format_local(job.next_run_at) if job.enabled else 'n/a'}"
            + (f", last run {job.last_status}" if job.last_status else "")
            for job in jobs
        )

    async def cancel_scheduled_job(self, name: str) -> str:
        """Permanently delete the scheduled job with this exact name.

        Args:
            name: The job's name, as shown by list_scheduled_jobs.
        """
        job = self.repo.find(name)
        if job is None:
            raise ToolError(f"There is no scheduled job called '{name}'.")
        self.delete_job(job.id)
        return f"Deleted the scheduled job '{job.name}'."
