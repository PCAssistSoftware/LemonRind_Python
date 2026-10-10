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

from lemonrind.config import SchedulerSettings
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
    "cut short": "warning",
    "missed": "warning",
}


def build_scheduler(
    module: SchedulerModule,
    chat_models: list[str],
    open_chat: Callable[[str], None],
    *,
    limits: SchedulerSettings | None = None,
    save: Callable[[], None] | None = None,
) -> None:
    """Draw the section into the current container. ``open_chat(session_id)`` jumps to a run's chat.

    ``limits`` (the scheduler's settings) and ``save`` add boxes for how long a run may take, how many tool rounds
    it may use and how long one reply may be. Each box shows the current value; a change to a valid value is made
    at once and ``save`` is called (this section has no Save button).
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
                ).tooltip("Edit").mark("job-edit")
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
        if limits is not None and save is not None:
            _limit_boxes(limits, save)

    ui.timer(1.5, poll)


def model_choices(chat_models: list[str], current: str) -> dict[str, str]:
    """The models to pick from in a job's form, always including the one the job already uses.

    A job keeps the model's name it was saved with. If that model has since been removed or renamed on Lemonade it is
    no longer in the list, and a drop-down cannot hold a value that is not among its options (it raises an error, which
    used to make such a job impossible to open). So the saved name is added back, marked, so the job can still be edited
    and the person can see why it needs another model.
    """
    options = {"": "(whatever model is selected)", **{name: name for name in chat_models}}
    if current and current not in options:
        options[current] = f"{current} (not listed by Lemonade now)"
    return options


TIME_LIMIT_MIN_MINUTES = 1
TIME_LIMIT_MAX_MINUTES = (
    24 * 60
)  # a day: more than that is far more likely a typing slip than a plan
ROUNDS_MIN, ROUNDS_MAX = 1, 500
REPLY_MIN, REPLY_MAX = 256, 131_072
CONTINUES_MIN, CONTINUES_MAX = 0, 10


def _limit_boxes(limits: SchedulerSettings, save: Callable[[], None]) -> None:
    """The boxes for how long a run may take, how many rounds it may use and how long one reply may be.

    A value outside a box's range is ignored (the box's own hint says what is allowed), and a valid one is applied to
    the settings at once and saved.
    """

    def box(
        label: str,
        value: float,
        *,
        low: int,
        high: int,
        step: int,
        marker: str,
        apply: Callable[[int], None],
        hint: str,
    ) -> None:
        def changed(event) -> None:
            number = event.value
            if number is None or not (low <= number <= high):
                return  # still typing, or out of range
            apply(int(number))
            save()

        ui.number(
            label, value=round(value), min=low, max=high, step=step, format="%d", on_change=changed
        ).props("outlined dense").classes("w-64").mark(marker)
        ui.label(hint).classes("text-caption lr-muted")

    ui.separator().classes("q-mt-sm")
    ui.label("Limits for one run").classes("text-weight-medium")
    box(
        "Time limit (minutes)",
        limits.job_timeout_seconds / 60,
        low=TIME_LIMIT_MIN_MINUTES,
        high=TIME_LIMIT_MAX_MINUTES,
        step=10,
        marker="job-time-limit",
        apply=lambda minutes: setattr(limits, "job_timeout_seconds", minutes * 60.0),
        hint="A run that takes longer than this is stopped. "
        f"Allowed: {TIME_LIMIT_MIN_MINUTES} to {TIME_LIMIT_MAX_MINUTES} minutes (a day). The default is 30.",
    )
    box(
        "Most tool rounds",
        limits.max_tool_rounds,
        low=ROUNDS_MIN,
        high=ROUNDS_MAX,
        step=10,
        marker="job-max-rounds",
        apply=lambda rounds: setattr(limits, "max_tool_rounds", rounds),
        hint="Each time the model uses tools and then carries on is one round. A run that reaches this many stops and "
        f"says so. Allowed: {ROUNDS_MIN} to {ROUNDS_MAX}. The default is 100.",
    )
    box(
        "Longest reply (tokens)",
        limits.max_output_tokens,
        low=REPLY_MIN,
        high=REPLY_MAX,
        step=4096,
        marker="job-max-reply",
        apply=lambda tokens: setattr(limits, "max_output_tokens", tokens),
        hint="The most the model may write in one go, counting its thinking too. A reply that reaches it is cut off, and "
        "the run is marked 'cut short'. A model that thinks a lot needs a bigger number, and a longer reply takes "
        f"longer to write. Allowed: {REPLY_MIN:,} to {REPLY_MAX:,}. The default is 32,768.",
    )
    box(
        "Automatic continues",
        limits.max_continuations,
        low=CONTINUES_MIN,
        high=CONTINUES_MAX,
        step=1,
        marker="job-max-continues",
        apply=lambda times: setattr(limits, "max_continuations", times),
        hint="When a reply is cut off at the longest reply, the run asks the model to carry on from where it stopped, "
        f"up to this many times. 0 turns it off. Allowed: {CONTINUES_MIN} to {CONTINUES_MAX}. The default is 2.",
    )
    ui.label("Changes apply from the next run and are saved at once.").classes(
        "text-caption lr-muted"
    )


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
        ui.card().classes("w-[40rem] max-w-full max-h-[90vh] no-wrap gap-0 q-pa-none"),
    ):
        ui.label("Edit job" if job else "New job").classes("text-h6 q-px-md q-pt-md q-pb-sm")
        # The form scrolls inside its own box; the title above and the buttons below stay put. (Direct children may not
        # shrink, or a tall multi-line box would be squeezed and its text would spill over the buttons.)
        with (
            ui.column()
            .classes("w-full grow overflow-auto q-px-md q-pb-md gap-2 lr-form-body")
            .style("min-height: 0")
        ):
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
                hour = ui.select(HOUR_CHOICES, value=simple.hour, label="Hour").props(
                    "outlined dense"
                )
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
                    SimpleSchedule(
                        frequency.value, hour.value, minute.value, weekday.value, day.value
                    )
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
                ui.textarea(
                    "What should the assistant do each time?", value=job.prompt if job else ""
                )
                .props("outlined autogrow")
                .classes("w-full")
                .mark("job-prompt")
            )
            model = ui.select(
                model_choices(chat_models, job.model if job else ""),
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
        ui.separator()
        with ui.row().classes("w-full justify-end q-pa-sm gap-2 shrink-0"):
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
