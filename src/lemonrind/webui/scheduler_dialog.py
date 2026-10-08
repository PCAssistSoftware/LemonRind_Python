"""The Scheduled jobs section of Settings: saved prompts that run by themselves on a schedule.

Each job is a prompt plus a cron schedule. The list shows when a job runs next, how its last run ended, and
lets you run it right now, edit it, switch it off or delete it. The result of every run is a chat (tagged
``scheduled``): the "open last chat" button jumps to it.

The editing form shows the next few run times as you type the schedule, so a mistake such as "0 9 * * 1"
(Monday) meant as Friday shows up before you save.

Python / NiceGUI ideas used here:

* One form function (``edit_job``) for both "new" and "edit": it gets ``job=None`` for new.
* ``ui.timer`` polling plus a counter on the module (``runs_finished``) to know when to redraw.
* ``datetime`` arithmetic with time zones: stored times are UTC, shown in local time.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from nicegui import ui

from lemonrind.modules.scheduler import CronError, DuplicateJobError, ScheduledJob, SchedulerModule
from lemonrind.modules.scheduler.cron import (
    HELP_TEXT,
    MINUTE_STEP,
    SimpleSchedule,
    format_local,
    normalise,
    read_simple,
    simple_expression,
    upcoming,
)
from lemonrind.webui.dialogs import ask_confirm, discard

FREQUENCY_CHOICES = {"daily": "Daily", "weekly": "Weekly", "monthly": "Monthly", "custom": "Custom"}
# Cron numbers its days from Sunday = 0; the list starts on Monday because that is how most people read a week.
WEEKDAY_CHOICES = {
    1: "Monday",
    2: "Tuesday",
    3: "Wednesday",
    4: "Thursday",
    5: "Friday",
    6: "Saturday",
    0: "Sunday",
}
HOUR_CHOICES = {hour: f"{hour:02d}" for hour in range(24)}
MINUTE_CHOICES = {minute: f"{minute:02d}" for minute in range(0, 60, MINUTE_STEP)}
DAY_CHOICES = {day: str(day) for day in range(1, 32)}
DEFAULT_SCHEDULE = "0 9 * * *"  # every day at 09:00

STATUS_COLOURS = {
    "ok": "positive",
    "failed": "negative",
    "timed out": "negative",
    "interrupted": "warning",
    "stopped": "warning",
    "missed": "warning",
}


def build_scheduler(
    module: SchedulerModule,
    chat_models: list[str],
    open_chat: Callable[[str], None],
    *,
    time_limit_minutes: float | None = None,
    on_time_limit: Callable[[float], None] | None = None,
) -> None:
    """Draw the section into the current container. ``open_chat(session_id)`` jumps to a run's chat.

    ``time_limit_minutes`` and ``on_time_limit`` add a box for how long one run may take: the box shows the current
    limit, and ``on_time_limit`` is called with the new number of minutes whenever it is changed to a valid value.
    """
    seen_runs = module.runs_finished

    @ui.refreshable
    def jobs() -> None:
        listed = module.repo.list()
        if not listed:
            ui.label(
                "No jobs yet. Press New job, or ask the assistant to schedule something."
            ).classes("text-caption lr-muted")
        for job in listed:
            job_row(job)

    def job_row(job: ScheduledJob) -> None:
        with ui.column().classes("w-full gap-0 q-pa-sm lr-tool"):
            with ui.row().classes("w-full items-center no-wrap gap-2"):
                ui.switch(value=job.enabled, on_change=lambda e, j=job: toggle(j, e.value)).tooltip(
                    "Run on schedule"
                )
                with ui.column().classes("gap-0 flex-grow"):
                    ui.label(job.name).classes("text-weight-medium")
                    ui.label(
                        f"{job.cron}   next: {format_local(job.next_run_at) if job.enabled else 'off'}"
                    ).classes("text-caption lr-muted")
                if job.running_since is not None:
                    ui.spinner(size="sm")
                if job.running_since is not None:
                    ui.button(icon="stop", on_click=lambda j=job: stop(j)).props(
                        "flat dense round color=negative"
                    ).tooltip("Stop this run").mark("job-stop")
                else:
                    ui.button(icon="play_arrow", on_click=lambda j=job: run_now(j)).props(
                        "flat dense round"
                    ).tooltip("Run now").mark("job-run")
                if job.last_session_id:
                    ui.button(icon="open_in_new", on_click=lambda j=job: open_last(j)).props(
                        "flat dense round"
                    ).tooltip("Open the last run's chat").mark("job-open")
                ui.button(icon="edit", on_click=lambda j=job: edit(j)).props(
                    "flat dense round"
                ).tooltip("Edit")
                ui.button(icon="delete", on_click=lambda j=job: remove(j)).props(
                    "flat dense round color=negative"
                ).tooltip("Delete")
            if job.running_since is not None:
                ui.label("Running now...").classes("text-caption lr-warn")
            elif job.last_status:
                when = (
                    job.last_run_at.astimezone().strftime("%a %d %b %H:%M")
                    if job.last_run_at
                    else ""
                )
                ui.label(f"Last run: {job.last_status}  {when}").classes(
                    f"text-caption text-{STATUS_COLOURS.get(job.last_status, 'grey')}"
                )
            if job.allow_unattended_tools:
                ui.label("May run tools that normally ask permission, without asking.").classes(
                    "text-caption lr-warn"
                )

    def toggle(job: ScheduledJob, enabled: bool) -> None:
        module.set_enabled(job.id, enabled)
        jobs.refresh()

    def run_now(job: ScheduledJob) -> None:
        module.start_now(job.id)
        ui.notify(f"Running '{job.name}' now. Its chat appears when it finishes.", timeout=2500)
        jobs.refresh()

    def stop(job: ScheduledJob) -> None:
        if module.stop_job(job.id):
            ui.notify(f"Stopping '{job.name}'...", timeout=2000)
        jobs.refresh()

    def open_last(job: ScheduledJob) -> None:
        if job.last_session_id:
            open_chat(job.last_session_id)

    async def remove(job: ScheduledJob) -> None:
        if await ask_confirm("Delete job", f"Delete '{job.name}'? Chats it already made are kept."):
            module.delete_job(job.id)
            jobs.refresh()

    async def edit(job: ScheduledJob | None) -> None:
        if await edit_job(module, job, chat_models):
            jobs.refresh()

    def poll() -> None:
        """Redraw while a job is running and once more after any run finished (to show how it ended)."""
        nonlocal seen_runs
        running = any(j.running_since is not None for j in module.repo.list())
        if running or module.runs_finished != seen_runs:
            seen_runs = module.runs_finished
            jobs.refresh()

    with ui.column().classes("w-full gap-2"):
        ui.label(
            "Jobs run while this app is running, one at a time. Each run is saved as a chat tagged "
            "'scheduled'. Times are your computer's local time."
        ).classes("text-caption lr-muted")
        jobs()
        ui.button("New job", icon="add", on_click=lambda: edit(None)).props("color=primary").mark(
            "job-new"
        )
        if time_limit_minutes is not None and on_time_limit is not None:
            _time_limit_box(time_limit_minutes, on_time_limit)

    ui.timer(1.5, poll)


TIME_LIMIT_MIN_MINUTES = 1
TIME_LIMIT_MAX_MINUTES = (
    24 * 60
)  # a day: more than that is far more likely a typing slip than a plan


def _time_limit_box(minutes: float, on_change: Callable[[float], None]) -> None:
    """The "how long may one run take" box. A value outside the allowed range is ignored (the box shows a warning)."""

    def changed(event) -> None:
        value = event.value
        if value is None or not (TIME_LIMIT_MIN_MINUTES <= value <= TIME_LIMIT_MAX_MINUTES):
            return  # still typing, or out of range: the box's own hint explains the range
        on_change(float(value))

    ui.separator().classes("q-mt-sm")
    ui.number(
        "Time limit for one run (minutes)",
        value=round(minutes),
        min=TIME_LIMIT_MIN_MINUTES,
        max=TIME_LIMIT_MAX_MINUTES,
        step=10,
        format="%d",
        on_change=changed,
    ).props("outlined dense").classes("w-64").mark("job-time-limit")
    ui.label(
        "A run that takes longer than this is stopped. It applies from the next run and is saved at once. "
        f"Allowed: {TIME_LIMIT_MIN_MINUTES} to {TIME_LIMIT_MAX_MINUTES} minutes (a day). The default is 30."
    ).classes("text-caption lr-muted")


async def edit_job(
    module: SchedulerModule, job: ScheduledJob | None, chat_models: list[str]
) -> bool:
    """The form for a new or existing job. Returns ``True`` if it was saved."""

    def preview() -> None:
        try:
            times = upcoming(normalise(schedule.value or ""), datetime.now(UTC), 3)
        except CronError as error:
            preview_label.text = str(error)
            preview_label.classes(replace="text-caption text-negative")
            return
        shown = ", ".join(format_local(t) for t in times) or "never (this date does not exist)"
        preview_label.text = f"Next runs: {shown}"
        preview_label.classes(replace="text-caption text-positive")

    with (
        ui.dialog() as dialog,
        ui.card().classes("w-[40rem] max-w-full max-h-[90vh] overflow-auto"),
    ):
        ui.label("Edit job" if job else "New job").classes("text-h6")
        name = (
            ui.input("Name", value=job.name if job else "")
            .props("outlined dense")
            .classes("w-full")
        )
        name.mark("job-name")
        start = job.cron if job else DEFAULT_SCHEDULE
        simple = read_simple(start) or SimpleSchedule("daily", 9, 0)
        with ui.row().classes("w-full items-center gap-2 no-wrap"):
            frequency = ui.select(
                FREQUENCY_CHOICES,
                value=simple.frequency if read_simple(start) else "custom",
                label="Repeat",
            ).props("outlined dense")
            frequency.classes("w-32").mark("job-frequency")
            weekday = ui.select(WEEKDAY_CHOICES, value=simple.weekday, label="Day").props(
                "outlined dense"
            )
            weekday.classes("w-36").mark("job-weekday")
            day = ui.select(DAY_CHOICES, value=simple.day, label="Day of month").props(
                "outlined dense"
            )
            day.classes("w-32").mark("job-day")
            hour = ui.select(HOUR_CHOICES, value=simple.hour, label="Hour").props("outlined dense")
            hour.classes("w-24").mark("job-hour")
            minute = ui.select(MINUTE_CHOICES, value=simple.minute, label="Minute").props(
                "outlined dense"
            )
            minute.classes("w-24").mark("job-minute")
        schedule = (
            ui.input("Schedule (cron)", value=start).props("outlined dense").classes("w-full")
        )
        schedule.mark("job-schedule")
        syncing = False  # True while the pickers and the text are being set from each other

        def show_pickers() -> None:
            """Which pickers make sense for the chosen frequency (none at all for Custom)."""
            custom = frequency.value == "custom"
            weekday.set_visibility(frequency.value == "weekly")
            day.set_visibility(frequency.value == "monthly")
            hour.set_visibility(not custom)
            minute.set_visibility(not custom)

        def pickers_changed() -> None:
            """A picker moved: write the matching expression into the text box (Custom leaves the text alone)."""
            nonlocal syncing
            show_pickers()
            if frequency.value == "custom":
                return
            syncing = True
            schedule.value = simple_expression(
                SimpleSchedule(frequency.value, hour.value, minute.value, weekday.value, day.value)
            )
            syncing = False

        def text_changed() -> None:
            """The text was typed: show it in the pickers if they can express it, otherwise switch to Custom."""
            nonlocal syncing
            preview()
            if syncing:
                return
            syncing = True
            if (found := read_simple((schedule.value or "").strip())) is not None:
                frequency.value, hour.value, minute.value = (
                    found.frequency,
                    found.hour,
                    found.minute,
                )
                weekday.value, day.value = found.weekday, found.day
            else:
                frequency.value = "custom"
            syncing = False
            show_pickers()

        for picker in (frequency, weekday, day, hour, minute):
            picker.on_value_change(lambda _: None if syncing else pickers_changed())
        schedule.on_value_change(lambda _: text_changed())
        show_pickers()
        preview_label = ui.label().classes("text-caption")
        ui.label(HELP_TEXT).classes("text-caption lr-muted")
        prompt = (
            ui.textarea("What should the assistant do each time?", value=job.prompt if job else "")
            .props("outlined autogrow")
            .classes("w-full")
            .mark("job-prompt")
        )
        model = ui.select(
            {"": "(whatever model is selected)", **{m: m for m in chat_models}},
            value=job.model if job else "",
            label="Model",
        ).classes("w-full")
        unattended = ui.switch(
            "Allow tools that normally ask permission to run without asking",
            value=job.allow_unattended_tools if job else False,
        )
        ui.label(
            "Nobody is there to answer a permission question, so by default such tools (for example MCP servers "
            "that send email) are refused in a scheduled run. Switch this on only for a job you trust."
        ).classes("text-caption lr-muted")
        with ui.row().classes("w-full justify-end q-mt-sm lr-sticky-footer"):
            ui.button("Cancel", on_click=lambda: dialog.submit(False)).props("flat")
            ui.button("Save", on_click=lambda: dialog.submit(True)).props("color=primary").mark(
                "job-save"
            )
    preview()

    while True:  # stay open on a validation error so nothing typed is lost
        if not await dialog:
            discard(dialog)
            return False
        try:
            if job is None:
                module.add_job(
                    name.value or "",
                    schedule.value or "",
                    prompt.value or "",
                    model=model.value or "",
                    allow_unattended_tools=bool(unattended.value),
                )
            else:
                module.edit_job(
                    job.id,
                    name=name.value or "",
                    schedule=schedule.value or "",
                    prompt=prompt.value or "",
                    model=model.value or "",
                    allow_unattended_tools=bool(unattended.value),
                )
        except (CronError, DuplicateJobError, ValueError) as error:
            ui.notify(str(error), type="negative", multi_line=True)
            dialog.open()
            continue
        discard(dialog)
        return True
