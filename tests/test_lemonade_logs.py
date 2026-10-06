"""Lemonade's live server log: parsing, filtering, the WebSocket client against a real local server, and the screen."""

from __future__ import annotations

import asyncio
import json

import aiohttp
import pytest
from aiohttp import WSMsgType
from aiohttp import web as aioweb
from aiohttp.test_utils import TestServer
from nicegui.testing import User

from lemonrind.lemonade.logstream import (
    LogEntry,
    filter_entries,
    parse_message,
    severity_rank,
    stream_logs,
    websocket_url,
)
from lemonrind.webui import inspectors
from lemonrind.webui.context import AppContext
from lemonrind.webui.inspectors import SHOWN_LINES, render_log_html
from tests.test_model_ui import box
from tests.test_settings_sections import settle

ENTRIES = [
    LogEntry(
        seq=1,
        timestamp="10:00:00",
        severity="Info",
        tag="Server",
        line="10:00:00 [Info] (Server) started",
    ),
    LogEntry(
        seq=2,
        timestamp="10:00:01",
        severity="Warn",
        tag="Process",
        line="10:00:01 [Warn] (Process) slow load",
    ),
    LogEntry(
        seq=3,
        timestamp="10:00:02",
        severity="Error",
        tag="Server",
        line="10:00:02 [Error] (Server) <boom> & more",
    ),
    LogEntry(
        seq=4,
        timestamp="10:00:03",
        severity="Fatal",
        tag="Server",
        line="10:00:03 [Fatal] (Server) gone",
    ),
]

# --- parsing and filtering ----------------------------------------------------------------------------------------


def test_the_log_address_uses_the_websocket_port_and_the_matching_scheme():
    assert websocket_url("http://ubuntu:13305/api/v1/", 9000) == "ws://ubuntu:9000/logs/stream"
    assert (
        websocket_url("https://lemon.example/v1/", 9443) == "wss://lemon.example:9443/logs/stream"
    )
    assert websocket_url("http://[::1]:13305/v1/", 9000) == "ws://[::1]:9000/logs/stream"


def test_snapshots_and_single_entries_are_parsed_and_everything_else_is_ignored():
    snapshot = parse_message(
        json.dumps(
            {"type": "logs.snapshot", "entries": [{"seq": 1, "line": "a"}, {"seq": 2, "line": "b"}]}
        )
    )
    entry = parse_message(
        json.dumps({"type": "logs.entry", "entry": {"seq": 3, "severity": "Error", "line": "c"}})
    )

    assert (
        snapshot is not None and snapshot[1] is True and [e.line for e in snapshot[0]] == ["a", "b"]
    )
    assert entry is not None and entry[1] is False and entry[0][0].severity == "Error"
    for ignored in (
        '{"type": "logs.pong"}',
        "not json",
        "[1, 2]",
        '{"type": "logs.entry"}',
        '{"entries": []}',
    ):
        assert parse_message(ignored) is None


def test_severities_are_ranked_so_warnings_and_above_is_a_comparison():
    assert [
        severity_rank(s) for s in ("Debug", "Info", "Warn", "Warning", "Error", "Fatal", "Weird")
    ] == [0, 1, 2, 2, 3, 4, 1]


def test_entries_are_filtered_by_level_and_by_text_ignoring_case():
    assert [e.seq for e in filter_entries(ENTRIES)] == [1, 2, 3, 4]
    assert [e.seq for e in filter_entries(ENTRIES, level="Warnings and errors")] == [2, 3, 4]
    assert [e.seq for e in filter_entries(ENTRIES, level="Errors only")] == [3, 4]
    assert [e.seq for e in filter_entries(ENTRIES, text="SLOW")] == [2]
    assert [e.seq for e in filter_entries(ENTRIES, level="Errors only", text="gone")] == [4]
    assert filter_entries(ENTRIES, text="nothing like this") == []


def test_an_entry_shows_lemonades_own_line_or_builds_one_when_it_has_none():
    assert ENTRIES[0].render() == "10:00:00 [Info] (Server) started"
    assert (
        LogEntry(timestamp="10:00:09", severity="Warn", tag="Net").render()
        == "10:00:09 WARN  [Net]"
    )


REAL = [
    LogEntry(seq=1, timestamp="2026-10-02 23:59:58.100", severity="Info", tag="Server",
             line="2026-10-02 23:59:58.100 [Info] (Server) POST /api/v1/tokenize"),
    LogEntry(seq=2, timestamp="2026-10-03 00:00:01.250", severity="Info", tag="Process",
             line="2026-10-03 00:00:01.250 [Info] (Process) 0.00.029.290 I slot release: id 0 | task 1044"),
    LogEntry(seq=3, timestamp="2026-10-03 00:00:02.000", severity="Error", tag="WrappedServer",
             line="2026-10-03 00:00:02.000 [Error] (WrappedServer) <boom> & more"),
]  # fmt: skip


def test_a_log_entry_is_split_into_clock_day_and_a_message_without_the_repeated_prefix():
    server, process, error = REAL

    assert (server.day, server.clock) == ("2026-10-02", "23:59:58.100")
    assert (
        server.message == "POST /api/v1/tokenize"
    )  # the timestamp, severity and tag now have their own columns
    assert (
        process.message == "slot release: id 0 | task 1044"
    )  # the backend's own clock and level letter go too
    assert error.message == "<boom> & more"
    plain = LogEntry(line="a line in some other shape")  # nothing to strip: shown as it is
    assert plain.message == "a line in some other shape" and plain.day == "" and plain.clock == ""


def test_the_log_is_drawn_as_a_table_with_a_header_columns_and_date_rows():
    html_text = render_log_html(REAL)

    assert html_text.startswith(
        '<div class="lr-log-head"><span>Time</span><span>Level</span><span>Source</span><span>Message</span></div>'
    )
    assert html_text.count("lr-log-row ") == 3  # one row per entry
    assert html_text.count("lr-log-day") == 2  # a date row at the start and where the day changes
    assert '<span class="lr-log-time">23:59:58.100</span>' in html_text
    assert '<span class="lr-log-lvl lr-log-lvl-error">ERROR</span>' in html_text
    assert (
        'class="lr-log-row lr-log-row-error"' in html_text
        and 'class="lr-log-tag" title="WrappedServer">WrappedServer</span>' in html_text
    )
    assert (
        "&lt;boom&gt; &amp; more" in html_text and "<boom>" not in html_text
    )  # a log line can never inject markup
    assert (
        "0.00.029.290" not in html_text and "[Info]" not in html_text
    )  # the duplicated prefix is gone


def test_only_the_latest_lines_are_drawn():
    many = [
        LogEntry(
            seq=n,
            timestamp="2026-10-03 10:00:00.000",
            severity="Info",
            tag="T",
            line=f"2026-10-03 10:00:00.000 [Info] (T) line {n}",
        )
        for n in range(SHOWN_LINES + 50)
    ]

    html_text = render_log_html(many)

    assert html_text.count("lr-log-row ") == SHOWN_LINES
    assert f"line {SHOWN_LINES + 49}</span>" in html_text and "line 0</span>" not in html_text


# --- the WebSocket client ------------------------------------------------------------------------------------------


async def test_the_client_subscribes_then_delivers_the_snapshot_and_live_entries_and_sends_the_key():
    seen: dict = {}

    async def handler(request: aioweb.Request) -> aioweb.WebSocketResponse:
        seen["auth"] = request.headers.get("Authorization")
        ws = aioweb.WebSocketResponse()
        await ws.prepare(request)
        subscribe = await ws.receive()
        assert subscribe.type == WSMsgType.TEXT
        seen["subscribe"] = json.loads(subscribe.data)
        await ws.send_json(
            {"type": "logs.snapshot", "entries": [e.model_dump() for e in ENTRIES[:2]]}
        )
        await ws.send_json({"type": "logs.entry", "entry": ENTRIES[2].model_dump()})
        await ws.send_json(
            {"type": "logs.something-new", "entry": {}}
        )  # a message kind we do not know: skipped
        await ws.send_str("garbage")
        await ws.send_json({"type": "logs.entry", "entry": ENTRIES[3].model_dump()})
        await ws.close()
        return ws

    app = aioweb.Application()
    app.router.add_get("/logs/stream", handler)
    snapshot: list[LogEntry] = []
    live: list[LogEntry] = []
    async with TestServer(app) as server:
        await stream_logs(
            f"http://127.0.0.1:{server.port}/api/v1/",
            server.port,
            "secret",
            snapshot.extend,
            live.append,
        )

    assert seen["subscribe"] == {"type": "logs.subscribe", "after_seq": None}
    assert seen["auth"] == "Bearer secret"
    assert [e.seq for e in snapshot] == [1, 2] and [e.seq for e in live] == [3, 4]


async def test_a_refused_connection_raises_so_the_screen_can_say_why():
    with pytest.raises(aiohttp.ClientError):
        await stream_logs("http://127.0.0.1:9/", 9, "", lambda _: None, lambda _: None)


# --- the screen ----------------------------------------------------------------------------------------------------


async def test_the_logs_screen_says_when_lemonade_reports_no_log_port(user: User, web: AppContext):
    await user.open("/")

    user.find(marker="view-logs").click()

    await user.should_see("Lemonade server")
    await user.should_see("This app")
    await user.should_see("Lemonade did not report a log port")


async def test_the_logs_screen_shows_the_streamed_lines_and_filters_them(
    user: User, web: AppContext, monkeypatch: pytest.MonkeyPatch
):
    from lemonrind.lemonade.models import Health, LoadedModel

    async def health_with_port() -> Health:
        return Health(
            status="ok",
            model_loaded="Fake-Chat",
            websocket_port=9000,
            all_models_loaded=[LoadedModel(model_name="Fake-Chat")],
        )

    async def fake_stream(base_url, port, api_key, on_snapshot, on_entry) -> None:
        assert port == 9000
        on_snapshot(ENTRIES[:2])
        on_entry(ENTRIES[2])
        await asyncio.sleep(30)  # stays "connected" until the screen closes and cancels it

    await user.open("/")
    monkeypatch.setattr(web.client, "health", health_with_port)
    monkeypatch.setattr(inspectors, "stream_logs", fake_stream)

    user.find(marker="view-logs").click()
    await user.should_see("Streaming Lemonade's log")
    await asyncio.sleep(1.0)  # the screen redraws on a timer
    content = box(user, "lemonade-log").content  # type: ignore[attr-defined]
    assert "started" in content and "slow load" in content and "&lt;boom&gt;" in content

    level = next(
        e for e in user.find(kind=inspectors.ui.select).elements if e.props.get("label") == "Show"
    )
    level.value = "Errors only"
    await asyncio.sleep(1.0)
    content = box(user, "lemonade-log").content  # type: ignore[attr-defined]
    assert "&lt;boom&gt;" in content and "started" not in content and "slow load" not in content


async def test_a_dropped_connection_is_reported_on_the_screen_not_raised(
    user: User, web: AppContext, monkeypatch: pytest.MonkeyPatch
):
    from lemonrind.lemonade.models import Health

    async def health_with_port() -> Health:
        return Health(status="ok", websocket_port=9000)

    async def broken(*args, **kwargs) -> None:
        raise ConnectionRefusedError("nothing is listening")

    await user.open("/")
    monkeypatch.setattr(web.client, "health", health_with_port)
    monkeypatch.setattr(inspectors, "stream_logs", broken)

    user.find(marker="view-logs").click()
    await settle()

    await user.should_see("Disconnected: ConnectionRefusedError: nothing is listening")
