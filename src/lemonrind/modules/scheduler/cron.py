"""Cron expressions: "when should this job run?"

A **cron expression** is five fields: ``minute hour day-of-month month day-of-week``. A ``*`` means "every",
``*/15`` means "every 15th", ``1-5`` is a range, ``1,15`` a list.

    0 19 * * 5      7pm every Friday
    30 8 * * 1-5    8:30am Monday to Friday
    0 */2 * * *     every two hours, on the hour
    0 9 1 * *       9am on the first of each month

The expression is read in the **computer's own local time zone** (so "9am" is 9am where you are, and clock
changes for summer time are handled), while everything *stored* is in UTC. That split is the standard way to
do it: UTC for storage and comparison (no ambiguity), local time for what a person means.

APScheduler's ``CronTrigger`` does the arithmetic and, in ``module.py``, also decides when jobs fire. This file
puts the app's rules around it, and it exists because of two places where APScheduler 3.x does *not* behave like
the cron you know:

* **Day-of-week numbers.** In classic cron ``0`` is Sunday and ``5`` is Friday. In APScheduler 3.x ``0`` is
  *Monday* and ``5`` is *Saturday* (a historical mistake the project keeps for compatibility). Passing ``0 19 * * 5``
  straight through would run on Saturdays. ``_weekday_names`` therefore turns cron's numbers into day names
  (``fri``), which mean the same everywhere.
* **Day-of-month together with day-of-week.** Classic cron runs when *either* matches ("the 1st, or any Monday");
  APScheduler runs only when *both* match. Rather than silently change what such an expression means, it is
  rejected with an explanation (two jobs say it clearly).

Two further decisions of ours:

* only the classic **five fields** are accepted (jobs here start a full model run, so seconds make no sense);
* a text that does not parse is reported with a message, not an exception, so a tool or form can show it.

Python ideas used here:

* **Naive and aware datetimes.** An *aware* ``datetime`` carries its time zone; the trigger works in the local zone
  and ``astimezone(UTC)`` converts the answer for storage.
* ``Iterator`` / generators: ``upcoming`` yields as many future times as asked for.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, tzinfo

from apscheduler.triggers.cron import CronTrigger
from tzlocal import get_localzone

FIELD_COUNT = 5

HELP_TEXT = (
    "Five fields: minute hour day-of-month month day-of-week. Examples: '0 19 * * 5' is 7pm every Friday, "
    "'30 8 * * 1-5' is 8:30am on weekdays, '0 */2 * * *' is every two hours."
)

_DAY_NAMES = (
    "sun",
    "mon",
    "tue",
    "wed",
    "thu",
    "fri",
    "sat",
)  # classic cron numbering: 0 is Sunday


class CronError(ValueError):
    """The text is not a usable cron expression. The message says why and is fit to show."""


def local_timezone() -> tzinfo:
    """The computer's own time zone, as the object APScheduler wants."""
    return get_localzone()


def _weekday_names(field: str) -> str:
    """Turn cron's day-of-week numbers into day names, which APScheduler reads the way cron does.

    ``5`` becomes ``fri``, ``1-5`` becomes ``mon,tue,wed,thu,fri``, ``*/2`` becomes ``sun,tue,thu,sat``. Both ``0`` and
    ``7`` mean Sunday in cron. Parts that are already names (``mon-fri``) are passed through untouched.
    """
    names: list[str] = []
    numbers: set[int] = set()
    for part in field.split(","):
        if any(char.isalpha() for char in part):
            names.append(part)  # "mon-fri" and the like: the same in both systems
            continue
        span, slash, step_text = part.partition("/")
        try:
            step = int(step_text) if slash else 1
            if span == "*":
                first, last = 0, 6
            elif "-" in span:
                low, high = span.split("-", 1)
                first, last = int(low), int(high)
            else:
                first = int(span)
                last = 6 if slash else first  # "a/n" means "from a to the end, every n"
        except ValueError as error:
            raise CronError(f"'{field}' is not a valid day-of-week. {HELP_TEXT}") from error
        if step < 1 or not (0 <= first <= 7 and 0 <= last <= 7 and first <= last):
            raise CronError(f"'{field}' is not a valid day-of-week. {HELP_TEXT}")
        numbers.update(day % 7 for day in range(first, last + 1, step))
    return ",".join([*names, *(_DAY_NAMES[day] for day in sorted(numbers))])


def trigger_for(expression: str) -> CronTrigger:
    """The APScheduler trigger for a (normalised) expression, in the computer's time zone."""
    minute, hour, day, month, weekday = expression.split()
    if day != "*" and weekday != "*":
        raise CronError(
            "Restricting both the day of the month and the day of the week is ambiguous (classic cron runs "
            "on either, but this scheduler would need both). Use one of them, or make two jobs."
        )
    try:
        return CronTrigger(
            minute=minute,
            hour=hour,
            day=day,
            month=month,
            day_of_week=_weekday_names(weekday) if weekday != "*" else "*",
            timezone=local_timezone(),
        )
    except ValueError as error:  # APScheduler's own message names the offending field and limit
        raise CronError(
            f"'{expression}' is not a valid cron expression: {error}. {HELP_TEXT}"
        ) from error


def normalise(text: str) -> str:
    """Tidy spacing and check the expression. Raises ``CronError`` if it cannot be used."""
    parts = text.split()
    if len(parts) != FIELD_COUNT:
        raise CronError(
            f"A cron expression has {FIELD_COUNT} fields but this has {len(parts)}. {HELP_TEXT}"
        )
    expression = " ".join(parts)
    trigger_for(expression)  # raises CronError if it cannot be used
    return expression


def next_run(expression: str, after: datetime) -> datetime | None:
    """The next time (UTC) the expression fires after ``after``, or ``None`` if it never will again
    (for example the 31st of February, or an expression this version cannot use)."""
    try:
        trigger = trigger_for(expression)
    except CronError:
        return None
    # The trigger counts a start that matches exactly as "now"; one microsecond on makes it strictly after.
    found = trigger.get_next_fire_time(None, after + timedelta(microseconds=1))
    return found.astimezone(UTC) if found else None


def upcoming(expression: str, after: datetime, count: int) -> list[datetime]:
    """The next ``count`` firing times (UTC), for showing a preview next to a schedule."""
    times: list[datetime] = []
    moment: datetime | None = after
    while len(times) < count and moment is not None:
        moment = next_run(expression, moment)
        if moment is not None:
            times.append(moment)
    return times


def format_local(moment: datetime | None) -> str:
    """``Fri 09 Oct 19:00`` in the computer's local time, or ``never``."""
    if moment is None:
        return "never"
    return moment.astimezone().strftime("%a %d %b %H:%M")


# --- the simple schedule picker ----------------------------------------------------------------------------------
# The form offers Daily / Weekly / Monthly pickers for the common cases and a free text box for everything else. These
# two functions turn the picker's choices into an expression and back, so opening a job made with the picker shows
# the picker again, while a hand-written expression (``*/15 * * * *``) stays in "Custom".

FREQUENCIES = ("daily", "weekly", "monthly")
MINUTE_STEP = (
    5  # the picker offers 00, 05, ... 55: enough for real schedules without a 60-item list
)


@dataclass(frozen=True, slots=True)
class SimpleSchedule:
    frequency: str  # "daily", "weekly" or "monthly"
    hour: int
    minute: int
    weekday: int = 1  # cron numbering: 0 Sunday ... 6 Saturday (only for "weekly")
    day: int = 1  # day of the month, 1 to 31 (only for "monthly")


def simple_expression(schedule: SimpleSchedule) -> str:
    """The cron expression for a picker choice, for example ``0 9 * * 5`` for Fridays at 09:00."""
    time_fields = f"{schedule.minute} {schedule.hour}"
    match schedule.frequency:
        case "daily":
            return f"{time_fields} * * *"
        case "weekly":
            return f"{time_fields} * * {schedule.weekday}"
        case "monthly":
            return f"{time_fields} {schedule.day} * *"
    raise ValueError(f"Unknown frequency {schedule.frequency!r}")


def read_simple(expression: str) -> SimpleSchedule | None:
    """The picker choice an expression corresponds to, or ``None`` if the picker cannot express it.

    Only plain numbers count (``*/15`` or ``1-5`` are Custom), and the minute must be one the picker offers.
    """
    parts = expression.split()
    if len(parts) != FIELD_COUNT or not all(part.isascii() for part in parts):
        return None
    minute, hour, day, month, weekday = parts
    if month != "*" or not minute.isdigit() or not hour.isdigit():
        return None
    minute_number, hour_number = int(minute), int(hour)
    if hour_number > 23 or minute_number % MINUTE_STEP or minute_number > 59:
        return None
    if day == "*" and weekday == "*":
        return SimpleSchedule("daily", hour_number, minute_number)
    if day == "*" and weekday.isdigit() and int(weekday) <= 7:
        return SimpleSchedule("weekly", hour_number, minute_number, weekday=int(weekday) % 7)
    if weekday == "*" and day.isdigit() and 1 <= int(day) <= 31:
        return SimpleSchedule("monthly", hour_number, minute_number, day=int(day))
    return None
