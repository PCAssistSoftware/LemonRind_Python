"""Storing scheduled jobs.

A job is a **saved prompt plus a schedule**. When it is due the app sends that prompt through the normal
assistant (tools and all) in a fresh chat of its own, tagged ``scheduled``, so the result is a chat you can read
like any other.

Besides what you wrote, a job records what happened: when it last ran and how it ended (``ok``, ``failed``,
``timed out``, ``interrupted``, ``missed``), which chat holds the output, and when it is next due. ``running_since``
is set while a run is in progress; if the app stops during a run it is still set at the next start, which is how
that interrupted run is recognised.

Timestamps are ISO 8601 text in UTC, as everywhere else in this project.
"""

from __future__ import annotations

import builtins
import sqlite3
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from lemonrind.storage import Database


class DuplicateJobError(ValueError):
    """A job with that name already exists."""


@dataclass(frozen=True, slots=True)
class ScheduledJob:
    id: str
    name: str
    cron: str
    prompt: str
    enabled: bool
    model: str  # "" = whatever model is selected when it runs
    # Tools that normally ask your permission (MCP servers, ...) cannot ask when nobody is there. They are
    # refused unless you switch this on for the job, which says "I accept that this runs them unattended".
    allow_unattended_tools: bool
    last_run_at: datetime | None
    next_run_at: datetime | None
    last_status: str  # "" until the first run
    last_session_id: str | None
    running_since: datetime | None
    created_at: datetime


def now_utc() -> datetime:
    return datetime.now(UTC)


def _when(text: str | None) -> datetime | None:
    return datetime.fromisoformat(text) if text else None


def _iso(moment: datetime | None) -> str | None:
    return moment.astimezone(UTC).isoformat() if moment else None


_SELECT = (
    "SELECT id, name, cron, prompt, enabled, model, allow_unattended_tools, last_run_at, next_run_at,"
    " last_status, last_session_id, running_since, created_at FROM scheduled_jobs"
)


def _from_row(row: sqlite3.Row) -> ScheduledJob:
    return ScheduledJob(
        id=row["id"],
        name=row["name"],
        cron=row["cron"],
        prompt=row["prompt"],
        enabled=bool(row["enabled"]),
        model=row["model"],
        allow_unattended_tools=bool(row["allow_unattended_tools"]),
        last_run_at=_when(row["last_run_at"]),
        next_run_at=_when(row["next_run_at"]),
        last_status=row["last_status"],
        last_session_id=row["last_session_id"],
        running_since=_when(row["running_since"]),
        created_at=datetime.fromisoformat(row["created_at"]),
    )


class SchedulerRepository:
    def __init__(self, db: Database, clock: Callable[[], datetime] = now_utc) -> None:
        self._db = db
        self._clock = clock

    def create(
        self,
        name: str,
        cron: str,
        prompt: str,
        next_run_at: datetime | None,
        *,
        model: str = "",
        allow_unattended_tools: bool = False,
    ) -> ScheduledJob:
        job_id = str(uuid.uuid4())
        try:
            with self._db.transaction() as conn:
                conn.execute(
                    "INSERT INTO scheduled_jobs (id, name, cron, prompt, enabled, model,"
                    " allow_unattended_tools, next_run_at, created_at) VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?)",
                    (
                        job_id,
                        name.strip(),
                        cron,
                        prompt,
                        model,
                        int(allow_unattended_tools),
                        _iso(next_run_at),
                        self._clock().isoformat(),
                    ),
                )
        except sqlite3.IntegrityError as error:  # UNIQUE name
            raise DuplicateJobError(f"A job called '{name}' already exists.") from error
        job = self.get(job_id)
        assert job is not None
        return job

    def update(
        self,
        job_id: str,
        *,
        name: str,
        cron: str,
        prompt: str,
        model: str,
        allow_unattended_tools: bool,
        next_run_at: datetime | None,
    ) -> None:
        try:
            with self._db.transaction() as conn:
                conn.execute(
                    "UPDATE scheduled_jobs SET name = ?, cron = ?, prompt = ?, model = ?,"
                    " allow_unattended_tools = ?, next_run_at = ? WHERE id = ?",
                    (
                        name.strip(),
                        cron,
                        prompt,
                        model,
                        int(allow_unattended_tools),
                        _iso(next_run_at),
                        job_id,
                    ),
                )
        except sqlite3.IntegrityError as error:
            raise DuplicateJobError(f"A job called '{name}' already exists.") from error

    def get(self, job_id: str) -> ScheduledJob | None:
        row = self._db.conn.execute(f"{_SELECT} WHERE id = ?", (job_id,)).fetchone()
        return _from_row(row) if row else None

    def find(self, name: str) -> ScheduledJob | None:
        row = self._db.conn.execute(f"{_SELECT} WHERE name = ?", (name.strip(),)).fetchone()
        return _from_row(row) if row else None

    def list(self) -> builtins.list[ScheduledJob]:
        rows = self._db.conn.execute(f"{_SELECT} ORDER BY name")
        return [_from_row(row) for row in rows]

    def set_enabled(self, job_id: str, enabled: bool, next_run_at: datetime | None) -> None:
        """Switch a job on or off. Switching on stores a fresh ``next_run_at`` (so the job does not "catch
        up" on what it missed while off); switching off leaves the stored time alone."""
        with self._db.transaction() as conn:
            if enabled:
                conn.execute(
                    "UPDATE scheduled_jobs SET enabled = 1, next_run_at = ? WHERE id = ?",
                    (_iso(next_run_at), job_id),
                )
            else:
                conn.execute("UPDATE scheduled_jobs SET enabled = 0 WHERE id = ?", (job_id,))

    def mark_running(self, job_id: str) -> None:
        with self._db.transaction() as conn:
            conn.execute(
                "UPDATE scheduled_jobs SET running_since = ? WHERE id = ?",
                (self._clock().isoformat(), job_id),
            )

    def record_run(
        self,
        job_id: str,
        *,
        status: str,
        session_id: str | None,
        next_run_at: datetime | None,
        update_schedule: bool = True,
    ) -> None:
        """Note how a run ended and (unless it was a manual "run now") when the job is next due."""
        with self._db.transaction() as conn:
            if update_schedule:
                conn.execute(
                    "UPDATE scheduled_jobs SET running_since = NULL, last_run_at = ?, last_status = ?,"
                    " last_session_id = COALESCE(?, last_session_id), next_run_at = ? WHERE id = ?",
                    (self._clock().isoformat(), status, session_id, _iso(next_run_at), job_id),
                )
            else:
                conn.execute(
                    "UPDATE scheduled_jobs SET running_since = NULL, last_run_at = ?, last_status = ?,"
                    " last_session_id = COALESCE(?, last_session_id) WHERE id = ?",
                    (self._clock().isoformat(), status, session_id, job_id),
                )

    def fail_interrupted(self) -> int:
        """Runs still marked as running when the app starts were cut off when it stopped."""
        with self._db.transaction() as conn:
            cursor = conn.execute(
                "UPDATE scheduled_jobs SET running_since = NULL, last_status = 'interrupted'"
                " WHERE running_since IS NOT NULL"
            )
            return cursor.rowcount

    def delete(self, job_id: str) -> None:
        with self._db.transaction() as conn:
            conn.execute("DELETE FROM scheduled_jobs WHERE id = ?", (job_id,))
