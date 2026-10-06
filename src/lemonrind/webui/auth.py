"""An optional password for the web UI.

By default the web server listens on ``127.0.0.1``: only this computer can reach it, so no password is needed. If you
start it with ``--host 0.0.0.0`` so a phone or another computer can use it, **everyone on that network can** read every
chat and memory and, worse, make the assistant run its tools (write files, send email through MCP servers, run
scheduled jobs). So a password is available, and ``lemonrind-web`` warns loudly if you listen on the network without one.

Set it with the ``LEMONRIND_PASSWORD`` environment variable (preferred: command-line arguments show up in process lists)
or ``--password``. Then every page asks for it once, and the browser remembers it in a cookie.

How it works, in three small parts:

* ``password_ok`` compares what was typed with the real password in **constant time** (``hmac.compare_digest``). An
  ordinary ``==`` stops at the first different character, and the tiny timing difference can leak the password to
  someone who measures many attempts.
* ``LoginThrottle`` stops guessing: after 5 wrong tries from one address, that address is locked out for 30 seconds,
  and each wrong try costs a one-second pause too.
* ``AuthMiddleware`` runs before every request: no logged-in session, no page, just a redirect to ``/login``.
  (*Middleware* is code that wraps every request, like a doorman.) NiceGUI's own internal files and the websocket are
  let through, because a page cannot load without passing the doorman first, and those need a page's client id.

The session cookie is signed with a secret that is generated once and kept in the data folder (``session_secret``), so
logging in survives a restart. This is a *single shared password*, not user accounts: it keeps casual visitors and
other devices on your network out, and is not a substitute for HTTPS on an untrusted network (the password and cookie
travel unencrypted over plain HTTP, so use it on a network you trust, or put the app behind a TLS proxy).

Python ideas used here:

* ``hmac.compare_digest``, ``secrets.token_urlsafe`` (the right tools for secrets: never ``==`` or ``random``).
* A ``BaseHTTPMiddleware`` subclass (Starlette's way to wrap requests).
* A small class holding state with an injected clock (``time.monotonic``), so the lockout can be tested without waiting.
"""

from __future__ import annotations

import asyncio
import hmac
import secrets
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from urllib.parse import quote

from nicegui import app, ui
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import RedirectResponse, Response

SESSION_KEY = "authenticated"
# The name of a file, not a secret (hence the scanner exemption).
SECRET_FILE = "session_secret"  # nosec B105
LOGIN_PATH = "/login"
# NiceGUI's own files (scripts, styles) and its websocket. A page cannot load without passing the middleware, and the
# websocket needs the client id that only an already-loaded page has.
ALWAYS_ALLOWED_PREFIXES = ("/_nicegui",)


def password_ok(candidate: str, expected: str) -> bool:
    """Compare a typed password with the real one without leaking, through timing, how much of it was right."""
    return hmac.compare_digest(candidate.encode("utf-8"), expected.encode("utf-8"))


def safe_redirect(target: str | None) -> str:
    """Where to go after logging in: a path on *this* site, never somewhere else.

    The address to return to arrives in the login link, so it can be forged (``/login?redirect_to=https://evil.example``).
    Anything that is not a plain path on this site becomes ``/``.
    """
    if (
        not target
        or not target.startswith("/")
        or target.startswith("//")
        or "\\" in target
        or "\n" in target
    ):
        return "/"
    return target


def load_or_create_secret(data_dir: Path) -> str:
    """The key that signs the login cookie, generated once and kept so that logins survive a restart."""
    path = data_dir / SECRET_FILE
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    data_dir.mkdir(parents=True, exist_ok=True)
    secret = secrets.token_urlsafe(48)
    path.write_text(secret, encoding="utf-8")
    return secret


class LoginThrottle:
    """Lock an address out for a while after too many wrong passwords."""

    def __init__(
        self,
        max_failures: int = 5,
        lockout_seconds: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max, self._lockout, self._clock = max_failures, lockout_seconds, clock
        self._failures: dict[str, int] = {}
        self._locked_until: dict[str, float] = {}

    def seconds_locked(self, who: str) -> float:
        """How many more seconds ``who`` must wait (0 if they may try)."""
        remaining = self._locked_until.get(who, 0.0) - self._clock()
        if remaining <= 0:
            self._locked_until.pop(who, None)
            return 0.0
        return remaining

    def record_failure(self, who: str) -> None:
        self._failures[who] = self._failures.get(who, 0) + 1
        if self._failures[who] >= self._max:
            self._locked_until[who] = self._clock() + self._lockout
            self._failures[who] = 0

    def record_success(self, who: str) -> None:
        self._failures.pop(who, None)


class AuthMiddleware(BaseHTTPMiddleware):
    """Send visitors who are not logged in to the login page."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        path = request.url.path
        allowed = path == LOGIN_PATH or path.startswith(ALWAYS_ALLOWED_PREFIXES)
        if not allowed and not app.storage.user.get(SESSION_KEY, False):
            target = request.url.path + (f"?{request.url.query}" if request.url.query else "")
            return RedirectResponse(
                f"{LOGIN_PATH}?redirect_to={quote(target, safe='/')}"
            )  # encoded: it may hold ? and &
        return await call_next(request)


async def _penalty() -> None:
    """The pause after a wrong password (its own function so tests can skip the wait)."""
    await asyncio.sleep(1)


def install_login(password: str) -> None:
    """Require ``password`` for every page. Call once, before the server starts."""
    throttle = LoginThrottle()
    app.add_middleware(AuthMiddleware)

    @ui.page(LOGIN_PATH)
    def login(redirect_to: str = "/") -> None:
        async def try_login() -> None:
            who = (
                ui.context.client.request.client.host
                if ui.context.client.request.client
                else "unknown"
            )
            if (wait := throttle.seconds_locked(who)) > 0:
                ui.notify(
                    f"Too many wrong passwords. Try again in {wait:.0f} seconds.", type="negative"
                )
                return
            if password_ok(box.value or "", password):
                throttle.record_success(who)
                app.storage.user[SESSION_KEY] = True
                ui.navigate.to(safe_redirect(redirect_to))
            else:
                throttle.record_failure(who)
                await _penalty()  # each wrong guess costs a second
                ui.notify("Wrong password.", type="negative")
                box.value = ""

        with ui.card().classes("absolute-center w-80"):
            ui.label("Lemon Rind").classes("text-h6")
            box = ui.input("Password", password=True, password_toggle_button=True).props(
                "autofocus outlined"
            )
            box.classes("w-full").on("keydown.enter", try_login).mark("login-password")
            ui.button("Log in", on_click=try_login).props("color=primary").classes("w-full").mark(
                "login-button"
            )
