"""The optional web password: the small pieces, then the whole login flow in a running page."""

from __future__ import annotations

from pathlib import Path

import pytest
from nicegui import ui
from nicegui.testing import User
from starlette.middleware import Middleware

from lemonrind.webui import auth
from lemonrind.webui.auth import (
    LoginThrottle,
    load_or_create_secret,
    password_ok,
    safe_redirect,
)


def test_only_the_exact_password_is_accepted():
    assert password_ok("hunter2", "hunter2")
    assert not password_ok("hunter", "hunter2")
    assert not password_ok("", "hunter2")
    assert password_ok("pässwörd", "pässwörd")  # non-ASCII works too


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("/chat/5", "/chat/5"),
        ("/?x=1&y=2", "/?x=1&y=2"),
        (None, "/"),
        ("", "/"),
        ("https://evil.example", "/"),
        ("//evil.example", "/"),
        ("/\\evil.example", "/"),
        ("/ok\nSet-Cookie: x=1", "/"),
        ("evil", "/"),
    ],
)
def test_after_login_you_can_only_be_sent_to_a_page_on_this_site(target, expected):
    assert safe_redirect(target) == expected


def test_the_signing_secret_is_made_once_and_then_reused(tmp_path: Path):
    first = load_or_create_secret(tmp_path / "data")  # the folder does not exist yet
    second = load_or_create_secret(tmp_path / "data")

    assert first == second and len(first) >= 40
    assert (tmp_path / "data" / "session_secret").read_text(encoding="utf-8") == first


def test_five_wrong_passwords_lock_an_address_out_for_a_while_then_it_may_try_again():
    now = [100.0]
    throttle = LoginThrottle(max_failures=5, lockout_seconds=30, clock=lambda: now[0])

    for _ in range(4):
        throttle.record_failure("1.2.3.4")
    assert throttle.seconds_locked("1.2.3.4") == 0
    throttle.record_failure("1.2.3.4")
    assert throttle.seconds_locked("1.2.3.4") == 30
    assert throttle.seconds_locked("5.6.7.8") == 0  # someone else is not affected

    now[0] += 12
    assert throttle.seconds_locked("1.2.3.4") == 18
    now[0] += 19
    assert throttle.seconds_locked("1.2.3.4") == 0


def test_a_correct_password_clears_the_count_of_wrong_ones():
    throttle = LoginThrottle(max_failures=3, lockout_seconds=30, clock=lambda: 0.0)

    throttle.record_failure("a")
    throttle.record_failure("a")
    throttle.record_success("a")
    throttle.record_failure("a")
    throttle.record_failure("a")

    assert throttle.seconds_locked("a") == 0  # two, not four, wrong tries in a row


# --- the flow in a running page -----------------------------------------------------------------------------------------------


async def test_without_the_password_you_are_sent_to_the_login_page_and_with_it_you_get_in(
    user: User, monkeypatch
):
    monkeypatch.setattr(auth, "_penalty", _no_wait)  # a wrong guess normally costs a second
    # In the real app, ui.run() adds NiceGUI's own middleware *after* ours, which puts ours inside it (it needs the
    # request tracking NiceGUI sets up). The test fixture has already run ui.run(), so add ours innermost by hand.
    monkeypatch.setattr(
        auth.app, "add_middleware", lambda cls: auth.app.user_middleware.append(Middleware(cls))
    )
    monkeypatch.setattr(
        auth.app, "middleware_stack", None
    )  # make Starlette rebuild its chain with ours in it

    @ui.page("/")
    def home() -> None:
        ui.label("The secret home page")

    auth.install_login("open sesame")

    await user.open("/")
    await user.should_see(marker="login-password")
    await user.should_not_see("The secret home page")

    user.find(marker="login-password").type("wrong").trigger("keydown.enter")
    await user.should_see("Wrong password.")

    user.find(marker="login-password").type("open sesame").trigger("keydown.enter")
    await user.should_see("The secret home page")


async def _no_wait() -> None:
    return None
