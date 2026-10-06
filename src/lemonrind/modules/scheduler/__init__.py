"""The Scheduler module: run saved prompts through the assistant on a cron schedule."""

from lemonrind.modules.scheduler.cron import CronError
from lemonrind.modules.scheduler.module import SchedulerModule
from lemonrind.modules.scheduler.repository import (
    DuplicateJobError,
    ScheduledJob,
    SchedulerRepository,
)
from lemonrind.modules.scheduler.runner import JobOutcome, JobRunner

__all__ = [
    "CronError",
    "DuplicateJobError",
    "JobOutcome",
    "JobRunner",
    "ScheduledJob",
    "SchedulerModule",
    "SchedulerRepository",
]
