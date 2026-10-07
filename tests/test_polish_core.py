"""Logging to a file, exporting chats, the web app's start-up wiring, and the terminal model picker."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from lemonrind.chats import (
    ChatRepository,
    ChatSession,
    StoredMessage,
    export_markdown,
    safe_filename,
)
from lemonrind.chats.export import MAX_RESULT_CHARS
from lemonrind.chats.models import ChatSession as Session
from lemonrind.config import Settings
from lemonrind.lemonade import ToolCall
from lemonrind.lemonade.client import LemonadeError
from lemonrind.lemonade.models import Health, LoadedModel, ModelInfo
from lemonrind.logging_setup import configure_logging, log_path
from lemonrind.storage import Database
from lemonrind.terminal.chatloop import ChatLoop
from lemonrind.terminal.modelpicker import choose_model, show_models, switch_model
from lemonrind.webui import app as web_app
from lemonrind.webui.context import AppContext
from tests.test_chatloop import FakeClient, ScriptedConsole

NOW = datetime(2026, 10, 2, 12, 0, tzinfo=UTC)

# --- logging -------------------------------------------------------------------------------------------------------------------


@pytest.fixture
def clean_root_logger():
    """Put the root logger back as it was, so these tests do not change logging for the rest of the run."""
    root = logging.getLogger()
    before, level = list(root.handlers), root.level
    yield root
    for handler in list(root.handlers):
        if handler not in before:
            root.removeHandler(handler)
            handler.close()
    root.setLevel(level)


def test_warnings_from_any_module_land_in_the_log_file(tmp_path: Path, clean_root_logger):
    path = configure_logging(tmp_path, to_screen=False)

    logging.getLogger("lemonrind.modules.memory").warning("Could not embed text", exc_info=False)
    logging.getLogger("httpx").info(
        "GET http://x 200 OK"
    )  # chatty libraries are kept at WARNING and above
    for handler in clean_root_logger.handlers:
        handler.flush()

    text = path.read_text(encoding="utf-8")
    assert path == log_path(tmp_path) and path.parent.name == "logs"
    assert "WARNING lemonrind.modules.memory: Could not embed text" in text
    assert "GET http://x" not in text


def test_configuring_twice_does_not_write_every_line_twice(tmp_path: Path, clean_root_logger):
    configure_logging(tmp_path, to_screen=False)
    path = configure_logging(tmp_path, to_screen=False)

    logging.getLogger("lemonrind.test").warning("only once")
    for handler in clean_root_logger.handlers:
        handler.flush()

    assert path.read_text(encoding="utf-8").count("only once") == 1


def test_the_screen_is_used_only_when_asked_and_verbose_adds_debug_detail(
    tmp_path: Path, clean_root_logger
):
    def screen_handlers() -> list[logging.Handler]:
        return [h for h in clean_root_logger.handlers if type(h) is logging.StreamHandler]

    configure_logging(tmp_path, to_screen=False)
    assert not screen_handlers() and clean_root_logger.level == logging.INFO

    configure_logging(tmp_path, to_screen=True)
    assert [h.level for h in screen_handlers()] == [logging.WARNING]

    configure_logging(tmp_path, to_screen=False, verbose=True)
    assert [h.level for h in screen_handlers()] == [
        logging.DEBUG
    ] and clean_root_logger.level == logging.DEBUG


# --- exporting a chat ------------------------------------------------------------------------------------------------------------


def make_session(**kwargs) -> ChatSession:
    values: dict[str, Any] = {
        "id": "s",
        "title": "Otters & rivers",
        "created_at": NOW,
        "updated_at": NOW,
    } | kwargs
    return Session(**values)


def message(role: str, content: str, **kwargs) -> StoredMessage:
    return StoredMessage(id=1, session_id="s", role=role, content=content, created_at=NOW, **kwargs)  # type: ignore[arg-type]


def test_a_chat_becomes_a_readable_markdown_document():
    session = make_session(folder_name="Wildlife", tags=("otters", "rivers"))
    messages = [
        message("user", "Tell me about otters"),
        message(
            "assistant",
            "Searching.",
            reasoning="Let me think.",
            tool_calls=(ToolCall("c1", "web_search", '{"query": "otters"}'),),
        ),
        message("tool", "1. Otters are mammals", tool_call_id="c1"),
        message("assistant", "Otters are playful mammals."),
    ]

    text = export_markdown(session, messages)

    assert text.startswith("# Otters & rivers\n\n*Started ")
    assert "folder: Wildlife | tags: #otters, #rivers*" in text
    assert "## You\n\nTell me about otters" in text
    assert "<details><summary>Thinking</summary>\n\nLet me think.\n\n</details>" in text
    assert '> Used the tool **web_search** with `{"query": "otters"}`' in text
    assert (
        "<details><summary>Result of web_search</summary>\n\n```\n1. Otters are mammals\n```"
        in text
    )
    assert (
        text.index("Tell me about otters")
        < text.index("Searching.")
        < text.index("Otters are playful")
    )
    assert text.endswith("Otters are playful mammals.\n")


def test_long_results_are_shortened_and_code_fences_cannot_be_broken_out_of():
    messages = [
        message("assistant", "", tool_calls=(ToolCall("c1", "read_file", "{not json"),)),
        message("tool", "x" * (MAX_RESULT_CHARS + 500), tool_call_id="c1"),
    ]

    text = export_markdown(make_session(), messages)

    assert "[... 500 more characters]" in text
    assert "with `{not json`" in text  # unreadable arguments are shown as they were

    fenced = export_markdown(
        make_session(), [message("tool", "before\n```\ninner block\n```\nafter", tool_call_id="c1")]
    )
    assert "````\nbefore" in fenced  # the outer fence is longer than any run of backticks inside it


def test_a_summarised_chat_includes_the_summary_and_an_unusual_tool_result_is_labelled():
    session = make_session(summary="They discussed rivers.", summary_upto=3)

    text = export_markdown(session, [message("tool", "orphan result", tool_call_id="unknown")])

    assert "Result of tool" in text  # a result whose call is not in the export still gets a label
    assert "They discussed rivers." in text and "summarised to save space" in text


def test_an_attached_picture_becomes_an_image_link_and_a_document_a_collapsed_block(tmp_path: Path):
    picture = tmp_path / "img_1.jpg"
    messages = [
        message("user", f"[Attached image: {picture}]\n\nwhat is this?"),
        message("assistant", "A square."),
        message(
            "user",
            "[Attached file: report.pdf]\n\nRevenue rose.\n\n[End of attached file]\n\nsummarise",
        ),
    ]

    text = export_markdown(make_session(), messages)

    assert f"![attached image]({picture.as_uri()})\n\nwhat is this?" in text
    assert "[Attached image:" not in text and "[End of attached file]" not in text
    assert "*Attached file: report.pdf*" in text
    assert (
        "<details><summary>Text of report.pdf</summary>\n\n```\nRevenue rose.\n```\n\n</details>\n\nsummarise"
        in text
    )


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("What is 6 x 7?", "What-is-6-x-7.md"),
        ("  ", "chat.md"),
        ("!!!", "chat.md"),
        ("a" * 100, "a" * 60 + ".md"),
        ("naïve café", "na-ve-caf.md"),
    ],
)
def test_titles_become_safe_file_names(title, expected):
    assert safe_filename(title) == expected


async def test_the_terminal_exports_the_current_chat_to_a_file(tmp_path: Path):
    repo = ChatRepository(Database(":memory:"))
    chat = repo.create_session()
    repo.add_message(chat.id, "user", "Hello there")
    repo.add_message(chat.id, "assistant", "Hi!")
    console = ScriptedConsole(["/export", f"/export {tmp_path / 'out' / 'mine.md'}", "/quit"])
    settings_file = tmp_path / "settings.json"
    loop = ChatLoop(
        client=FakeClient(),  # type: ignore[arg-type]
        settings=Settings(),
        settings_file=settings_file,
        repo=repo,
        console=console,
        model="m",
    )

    await loop.run(resume=True)

    default = tmp_path / "exports" / "Hello-there.md"
    assert default.is_file() and "## You\n\nHello there" in default.read_text(encoding="utf-8")
    assert (tmp_path / "out" / "mine.md").is_file()
    assert console.output.count("Saved to") == 2


async def test_exporting_an_unsaved_chat_explains(tmp_path: Path):
    console = ScriptedConsole(["/export", "/quit"])
    loop = ChatLoop(
        client=FakeClient(),  # type: ignore[arg-type]
        settings=Settings(),
        repo=ChatRepository(Database(":memory:")),
        console=console,
        model="m",
    )

    await loop.run(resume=False)

    assert "not saved yet" in console.output


# --- the web app's start-up ------------------------------------------------------------------------------------------------------


def test_the_web_app_defaults_to_this_computer_only_and_offers_a_password():
    args = web_app.build_parser().parse_args([])

    assert (args.host, args.port, args.password, args.no_browser) == (
        "127.0.0.1",
        8080,
        None,
        False,
    )
    assert (
        web_app.build_parser().parse_args(["--password", "x", "--host", "0.0.0.0"]).password == "x"
    )


async def test_the_app_context_opens_the_database_builds_modules_and_closes_them(tmp_path: Path):
    context = AppContext.create(tmp_path / "data", base_url="http://elsewhere:9999/v1")

    assert (
        context.settings.lemonade.base_url == "http://elsewhere:9999/v1/"
    )  # the trailing slash is added
    assert (tmp_path / "data" / "settings.json").is_file()  # created on first run
    assert context.modules is not None and len(context.modules.modules) == 11
    assert context.repo.list_sessions() == []

    context.save_settings()
    await context.aclose()  # closes the client, the modules and the database without error


def test_main_wires_everything_up_before_starting_the_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    started = {}
    monkeypatch.setattr(web_app.ui, "run", lambda **kwargs: started.update(kwargs))
    monkeypatch.setattr(web_app, "register_pages", lambda: started.update(pages=True))
    monkeypatch.setattr(web_app.app, "on_startup", lambda fn: started.update(on_startup=fn))
    monkeypatch.setattr(web_app.app, "on_shutdown", lambda fn: started.update(on_shutdown=fn))
    monkeypatch.setattr(
        web_app.app,
        "add_static_files",
        lambda *a, **k: started.setdefault("static", []).append(a[0]),
    )
    root = logging.getLogger()
    before = list(root.handlers)

    web_app.main(["--data-dir", str(tmp_path), "--no-browser", "--port", "9123"])

    for handler in list(root.handlers):
        if handler not in before:
            root.removeHandler(handler)
            handler.close()
    assert started["port"] == 9123 and started["host"] == "127.0.0.1" and started["show"] is False
    assert started["pages"] and callable(started["on_startup"]) and callable(started["on_shutdown"])
    assert started["static"] == [
        "/generated",
        "/attached",
    ]  # generated pictures, and pictures you attached
    assert (tmp_path / "session_secret").is_file() and started["storage_secret"]
    assert "WARNING" not in capsys.readouterr().out  # loopback only: no warning about the network


def test_listening_on_the_network_without_a_password_is_warned_about(
    tmp_path: Path, monkeypatch, capsys
):
    monkeypatch.setattr(web_app.ui, "run", lambda **kwargs: None)
    monkeypatch.setattr(web_app, "register_pages", lambda: None)
    monkeypatch.setattr(web_app.app, "on_startup", lambda fn: None)
    monkeypatch.setattr(web_app.app, "on_shutdown", lambda fn: None)
    monkeypatch.setattr(web_app.app, "add_static_files", lambda *a, **k: None)
    monkeypatch.delenv("LEMONRIND_PASSWORD", raising=False)
    root = logging.getLogger()
    before = list(root.handlers)

    web_app.main(["--data-dir", str(tmp_path), "--host", "0.0.0.0"])

    for handler in list(root.handlers):
        if handler not in before:
            root.removeHandler(handler)
            handler.close()
    assert "WARNING: listening on the network with no password" in capsys.readouterr().out


def test_ctrl_c_stops_the_web_app_quietly(tmp_path: Path, monkeypatch, capsys):
    def interrupted(**kwargs):
        raise KeyboardInterrupt  # what asyncio re-raises after the server has shut down on Ctrl+C

    monkeypatch.setattr(web_app.ui, "run", interrupted)
    monkeypatch.setattr(web_app, "register_pages", lambda: None)
    monkeypatch.setattr(web_app.app, "on_startup", lambda fn: None)
    monkeypatch.setattr(web_app.app, "on_shutdown", lambda fn: None)
    monkeypatch.setattr(web_app.app, "add_static_files", lambda *a, **k: None)
    root = logging.getLogger()
    before = list(root.handlers)

    web_app.main(["--data-dir", str(tmp_path), "--no-browser"])  # must return, not raise

    for handler in list(root.handlers):
        if handler not in before:
            root.removeHandler(handler)
            handler.close()
    assert "Lemon Rind stopped." in capsys.readouterr().out


# --- the terminal model picker ---------------------------------------------------------------------------------------------------


class PickerClient:
    base_url = "http://fake/v1/"

    def __init__(
        self, *, loaded: str | None = None, load_error: str | None = None, down: bool = False
    ) -> None:
        self.loaded, self.load_error, self.down = loaded, load_error, down
        self.load_calls: list[str] = []

    async def health(self) -> Health:
        if self.down:
            raise LemonadeError("Could not reach Lemonade")
        models = [LoadedModel(model_name=self.loaded)] if self.loaded else []
        return Health(status="ok", model_loaded=self.loaded, all_models_loaded=models)

    async def list_models(self, **kwargs):
        return [
            ModelInfo(id="Chat-A", labels=["chat"], downloaded=True, context_length=8000),
            ModelInfo(id="Chat-B", labels=["chat", "tool-calling"], downloaded=True),
            ModelInfo(id="Embed", labels=["embeddings"], downloaded=True),
        ]

    async def load_model(self, name: str) -> None:
        self.load_calls.append(name)
        if self.load_error:
            raise LemonadeError(self.load_error)


def quiet_console() -> Console:
    import io

    return Console(file=io.StringIO(), width=100, force_terminal=False, color_system=None)


async def test_the_picker_uses_a_loaded_chat_model_without_reloading_it():
    client, console = PickerClient(loaded="Chat-A"), quiet_console()

    assert await choose_model(client, "", console) == "Chat-A"  # type: ignore[arg-type]
    assert client.load_calls == [] and "Connected to Lemonade" in console.file.getvalue()


async def test_the_picker_loads_the_requested_or_default_model_and_reports_trouble():
    client, console = PickerClient(), quiet_console()
    assert await choose_model(client, "Chat-A", console) == "Chat-A"  # type: ignore[arg-type]
    assert await choose_model(PickerClient(), "", console) == "Chat-B"  # type: ignore[arg-type]  # tool-capable default
    assert client.load_calls == ["Chat-A"] and "Loaded Chat-A" in console.file.getvalue()

    for broken, message in (
        (PickerClient(down=True), "Could not reach"),
        (PickerClient(load_error="boom"), "boom"),
    ):
        console = quiet_console()
        assert await choose_model(broken, "Chat-A", console) is None  # type: ignore[arg-type]
        assert message in console.file.getvalue()
    unknown = quiet_console()
    assert await choose_model(PickerClient(), "Nope", unknown) is None  # type: ignore[arg-type]
    assert "not downloaded" in unknown.file.getvalue()


async def test_models_are_listed_and_switching_checks_the_name_and_survives_a_failed_load():
    client, console = PickerClient(), quiet_console()

    await show_models(client, "Chat-A", console)  # type: ignore[arg-type]
    listing = console.file.getvalue()
    assert "Chat-A" in listing and "8,000 tokens of context" in listing and "Embed" not in listing

    assert await switch_model(client, "Chat-A", "Chat-B", console) == "Chat-B"  # type: ignore[arg-type]
    assert await switch_model(client, "Chat-A", "", console) == "Chat-A"  # type: ignore[arg-type]
    assert await switch_model(client, "Chat-A", "Embed", console) == "Chat-A"  # type: ignore[arg-type]  # not a chat model
    failing = PickerClient(load_error="out of memory")
    assert await switch_model(failing, "Chat-A", "Chat-B", console) == "Chat-A"  # type: ignore[arg-type]
    assert (
        "Usage: /model NAME" in console.file.getvalue()
        and "out of memory" in console.file.getvalue()
    )


def test_the_address_to_open_turns_a_wildcard_host_into_this_computer():
    assert web_app.page_url("127.0.0.1", 8080) == "http://127.0.0.1:8080"
    assert web_app.page_url("0.0.0.0", 9000) == "http://127.0.0.1:9000"
    assert web_app.page_url("192.168.1.5", 8080) == "http://192.168.1.5:8080"
    assert web_app.page_url("::1", 8080) == "http://[::1]:8080"


def test_on_linux_the_browser_is_started_with_its_chatter_thrown_away(monkeypatch):
    import subprocess

    launched: list[dict] = []
    monkeypatch.setattr(web_app.sys, "platform", "linux")
    monkeypatch.setattr(web_app.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        web_app.subprocess,
        "Popen",
        lambda command, **options: launched.append({"command": command, **options}),
    )
    monkeypatch.setattr(web_app.webbrowser, "open", lambda url: launched.append({"fallback": url}))

    web_app.open_browser("http://127.0.0.1:8080")

    (call,) = launched
    assert call["command"] == ["xdg-open", "http://127.0.0.1:8080"]
    assert call["stdout"] == call["stderr"] == subprocess.DEVNULL


def test_without_xdg_open_or_on_windows_the_standard_browser_opener_is_used(monkeypatch):
    opened: list[str] = []
    monkeypatch.setattr(web_app.webbrowser, "open", opened.append)
    monkeypatch.setattr(web_app.shutil, "which", lambda name: None)
    monkeypatch.setattr(web_app.sys, "platform", "linux")
    web_app.open_browser("http://127.0.0.1:1")
    monkeypatch.setattr(web_app.sys, "platform", "win32")
    web_app.open_browser("http://127.0.0.1:2")

    assert opened == ["http://127.0.0.1:1", "http://127.0.0.1:2"]


def test_the_web_app_opens_the_browser_after_start_up_unless_told_not_to(
    tmp_path: Path, monkeypatch, capsys
):
    callbacks: list = []
    opened: list[str] = []
    monkeypatch.setattr(web_app.ui, "run", lambda **kwargs: None)
    monkeypatch.setattr(web_app, "register_pages", lambda: None)
    monkeypatch.setattr(web_app.app, "on_startup", callbacks.append)
    monkeypatch.setattr(web_app.app, "on_shutdown", lambda fn: None)
    monkeypatch.setattr(web_app.app, "add_static_files", lambda *a, **k: None)
    monkeypatch.setattr(web_app, "open_browser", opened.append)
    root = logging.getLogger()
    before = list(root.handlers)

    web_app.main(["--data-dir", str(tmp_path), "--port", "9124"])
    for callback in callbacks:  # what the server runs once it is up
        if getattr(callback, "__name__", "") == "<lambda>":
            callback()
    quiet = len(callbacks)
    callbacks.clear()
    web_app.main(["--data-dir", str(tmp_path), "--port", "9124", "--no-browser"])

    for handler in list(root.handlers):
        if handler not in before:
            root.removeHandler(handler)
            handler.close()
    assert opened == ["http://127.0.0.1:9124"]
    assert len(callbacks) == quiet - 1  # one start-up callback fewer without a browser
    assert "Open http://127.0.0.1:9124" in capsys.readouterr().out


def test_the_image_defaults_match_the_other_editions():
    from lemonrind.config import ImageSettings

    images = ImageSettings()
    assert (images.width, images.height) == (512, 512)
    assert (images.steps, images.cfg_scale, images.seed) == (
        4,
        1.0,
        -1,
    )  # a seed of -1 means random


def test_the_log_can_be_cleared_while_the_app_keeps_logging_to_it(tmp_path: Path):
    from lemonrind.logging_setup import clear_logs, configure_logging, log_path

    root = logging.getLogger()
    before = list(root.handlers)
    try:
        configure_logging(tmp_path, to_screen=False)
        logging.getLogger("lemonrind.test").warning("before clearing")
        older = log_path(tmp_path).with_name("lemonrind.log.1")
        older.write_text("an older copy", encoding="utf-8")
        assert "before clearing" in log_path(tmp_path).read_text(encoding="utf-8")

        clear_logs(tmp_path)
        logging.getLogger("lemonrind.test").warning("after clearing")

        text = log_path(tmp_path).read_text(encoding="utf-8")
        assert "before clearing" not in text and "after clearing" in text
        assert not older.exists()
    finally:
        for handler in list(root.handlers):
            if handler not in before:
                root.removeHandler(handler)
                handler.close()


def test_a_harmless_windows_connection_reset_is_not_logged_but_real_errors_are(tmp_path: Path):
    from lemonrind.logging_setup import configure_logging, log_path

    root = logging.getLogger()
    before = list(root.handlers)
    try:
        configure_logging(tmp_path, to_screen=False)
        asyncio_log = logging.getLogger("asyncio")
        try:
            raise ConnectionResetError(10054, "An existing connection was forcibly closed")
        except ConnectionResetError:
            asyncio_log.exception(
                "Exception in callback _ProactorBasePipeTransport._call_connection_lost()"
            )
        try:
            raise ValueError("a real problem")
        except ValueError:
            asyncio_log.exception("Exception in callback something_else()")

        text = log_path(tmp_path).read_text(encoding="utf-8")
        assert "_call_connection_lost" not in text and "a real problem" in text
    finally:
        for handler in list(root.handlers):
            if handler not in before:
                root.removeHandler(handler)
                handler.close()


def test_a_busy_port_is_noticed_and_a_free_one_is_not():
    import socket

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        busy = listener.getsockname()[1]
        assert web_app.port_in_use("127.0.0.1", busy)
        assert web_app.port_in_use(
            "0.0.0.0", busy
        )  # a wildcard address is checked through this computer
    assert not web_app.port_in_use("127.0.0.1", busy)  # released again: free


def test_starting_on_a_busy_port_says_what_to_do_and_starts_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    import socket

    started: dict = {}
    monkeypatch.setattr(web_app.ui, "run", lambda **kwargs: started.update(kwargs))
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        busy = listener.getsockname()[1]

        with pytest.raises(SystemExit) as stopped:
            web_app.main(
                ["--port", str(busy), "--no-browser", "--data-dir", str(tmp_path / "data")]
            )

    message = str(stopped.value)
    assert f"Port {busy} is already in use" in message and f"--port {busy + 10}" in message
    assert started == {}  # the server was never started
    assert not (tmp_path / "data").exists()  # and nothing was created before the check
