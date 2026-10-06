"""The MCP screens: the terminal /mcp commands, the web servers screen, and the tool permission dialog."""

from __future__ import annotations

import json
import sys
from collections.abc import AsyncIterator
from pathlib import Path

from nicegui.testing import User

from lemonrind.chats import ChatRepository
from lemonrind.config import Settings
from lemonrind.lemonade import ChatEvent, Finished, TextDelta, ToolCall, ToolCallsRequested
from lemonrind.lemonade.events import RequestStats
from lemonrind.modules import McpModule, Module, Tool, build_registry
from lemonrind.storage import Database
from lemonrind.terminal.chatloop import ChatLoop
from lemonrind.webui.context import set_context
from lemonrind.webui.page import register_pages
from tests.conftest import NullMemoryClient, open_section
from tests.test_chatloop import FakeClient, ScriptedConsole
from tests.test_webui import FakeLemonade, make_context

DEMO_SERVER = Path(__file__).parent / "mcp_demo_server.py"
DEMO_JSON = json.dumps(
    {"mcpServers": {"demo": {"command": sys.executable, "args": [str(DEMO_SERVER)]}}}
)


def mcp_module(modules) -> McpModule:
    return next(m for m in modules.modules if isinstance(m, McpModule))


# --- the terminal -------------------------------------------------------------------------------------------------


async def test_mcp_commands_import_list_switch_and_remove(tmp_path: Path):
    settings = Settings()
    settings.modules.mcp.connect_timeout_seconds = 30.0
    db = Database(":memory:")
    modules = build_registry(settings, tmp_path, db=db, client=NullMemoryClient())  # type: ignore[arg-type]
    console = ScriptedConsole(
        [
            "/mcp",
            f"/mcp import {DEMO_JSON}",
            "/mcp ask demo off",
            "/mcp off demo",
            "/mcp on demo",
            "/mcp remove demo",
            "/mcp import not json",
            "/mcp on nobody",
            "/quit",
        ]
    )
    loop = ChatLoop(
        client=FakeClient(),  # type: ignore[arg-type]
        settings=settings,
        repo=ChatRepository(db),
        console=console,
        model="m",
        modules=modules,
    )
    await modules.start_enabled()
    try:
        await loop.run(resume=False)
    finally:
        await modules.stop_all()

    output = console.output
    assert "No MCP servers yet" in output
    assert "demo  connected" in output and "tools: echo, add, explode, pixel" in output
    assert "asks first" in output and "runs freely" in output  # before and after /mcp ask
    assert "demo  off" in output
    assert "Removed 'demo'." in output
    assert "not valid JSON" in output
    assert "No MCP server with that name" in output
    assert mcp_module(modules).repo.list() == []


async def test_the_terminal_asks_before_running_a_tool_that_needs_permission(tmp_path: Path):
    class Risky(Module):
        name, config_key, description = "Risky", "risky", "test"
        ran: list[str] = []

        def get_tools(self) -> list[Tool]:
            async def danger(arguments: dict) -> str:
                self.ran.append(arguments["what"])
                return "done"

            return [
                Tool(
                    "danger",
                    "Does something risky",
                    {"type": "object"},
                    danger,
                    None,
                    requires_approval=True,
                )
            ]

    class Asking:
        """A model that asks for the risky tool first, then writes its answer."""

        def __init__(self) -> None:
            self.calls = 0

        async def list_models(self, **kwargs) -> list:
            return []

        async def stream_chat(self, messages, model, **kwargs) -> AsyncIterator[ChatEvent]:
            self.calls += 1
            if self.calls == 1:
                yield ToolCallsRequested(
                    (ToolCall("c1", "danger", '{"what": "delete everything"}'),)
                )
                yield Finished(RequestStats(), "tool_calls")
            else:
                yield TextDelta("Finished.")
                yield Finished(RequestStats(), "stop")

    from lemonrind.modules import ModuleRegistry

    for answer, should_run in (("y", True), ("n", False), ("", False)):
        risky = Risky(Settings())
        risky.ran = []
        client = Asking()
        console = ScriptedConsole(["Please do it", answer, "/quit"])
        loop = ChatLoop(
            client=client,  # type: ignore[arg-type]
            settings=Settings(),
            repo=ChatRepository(Database(":memory:")),
            console=console,
            model="m",
            modules=ModuleRegistry([risky]),
        )

        await loop.run(resume=False)

        assert risky.ran == (["delete everything"] if should_run else [])
        assert "The assistant wants to run danger" in console.output
        assert loop.conversation.approve is None  # restored after the reply


# --- the web ------------------------------------------------------------------------------------------------------


async def test_the_web_screen_adds_a_server_from_pasted_json(user: User, tmp_path: Path):
    context = make_context(tmp_path, FakeLemonade())
    module = mcp_module(context.modules)  # type: ignore[arg-type]
    set_context(context)
    register_pages()
    try:
        await user.open("/")
        await open_section(user, "mcp")
        await user.should_see("Paste the JSON from the server")
        user.find(marker="mcp-json").type(DEMO_JSON)
        user.find(marker="mcp-add").click()

        await user.should_see("demo")
        await user.should_see("Ask me before running its tools")
        assert [s.config.name for s in module.statuses()] == ["demo"]
    finally:
        await module.on_shutdown()
        set_context(None)


async def test_the_web_screen_explains_bad_json(user: User, tmp_path: Path):
    context = make_context(tmp_path, FakeLemonade())
    set_context(context)
    register_pages()
    try:
        await user.open("/")
        await open_section(user, "mcp")
        await user.should_see("Add servers")  # the dialog is built by an async handler: wait for it
        user.find(marker="mcp-json").type("{broken")
        user.find(marker="mcp-add").click()

        await user.should_see("not valid JSON")
    finally:
        set_context(None)


class RiskyModule(Module):
    name, config_key, description = "Risky", "risky", "A tool that must be approved."

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.ran: list[dict] = []

    def get_tools(self) -> list[Tool]:
        async def danger(arguments: dict) -> str:
            self.ran.append(arguments)
            return "the risky thing happened"

        return [
            Tool(
                "danger",
                "Does something risky",
                {"type": "object"},
                danger,
                None,
                requires_approval=True,
            )
        ]


class AskingLemonade(FakeLemonade):
    """Asks for the risky tool on every message that has no tool result yet."""

    async def stream_chat(self, messages, model, **kwargs) -> AsyncIterator[ChatEvent]:
        stats = RequestStats(prompt_tokens=20, input_tokens=20, output_tokens=5)
        if any(m["role"] == "tool" for m in messages[-2:]):
            yield TextDelta("All done.")
            yield Finished(stats, "stop")
        else:
            yield ToolCallsRequested((ToolCall("c1", "danger", '{"target": "everything"}'),))
            yield Finished(stats, "tool_calls")


async def test_the_web_asks_permission_and_runs_the_tool_only_when_allowed(
    user: User, tmp_path: Path
):
    from lemonrind.modules import ModuleRegistry

    context = make_context(tmp_path, AskingLemonade())
    risky = RiskyModule(context.settings)
    context.modules = ModuleRegistry([risky])
    set_context(context)
    register_pages()
    try:
        await user.open("/")
        user.find(marker="message-input").type("do the risky thing")
        user.find(marker="send").click()

        await user.should_see("Allow this tool?")
        await user.should_see("The assistant wants to run danger")
        await user.should_see("everything")  # the arguments are shown
        assert risky.ran == []  # nothing runs while the question is open
        user.find(marker="allow-once").click()

        await user.should_see("All done.")
        assert risky.ran == [{"target": "everything"}]
    finally:
        set_context(None)
