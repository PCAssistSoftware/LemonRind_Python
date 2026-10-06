"""Tests for the MCP module: importing server definitions, converting results, and real connections.

The connection tests start ``tests/mcp_demo_server.py`` as a real child process and talk MCP to it over
standard input and output, exactly as the app does with any server, so the whole path is exercised.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import pytest
from mcp.types import (
    AudioContent,
    BlobResourceContents,
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    ResourceLink,
    TextContent,
    TextResourceContents,
)

from lemonrind.config import Settings
from lemonrind.lemonade import ToolCall
from lemonrind.modules import ModuleRegistry, Tool
from lemonrind.modules.mcp import (
    DuplicateServerError,
    McpModule,
    McpServerRepository,
    ServerDefinition,
    ServerImportError,
    from_command_line,
    parse_servers,
)
from lemonrind.modules.mcp.connection import McpConnection, describe_error, result_to_tool_result
from lemonrind.modules.mcp.module import slugify, tool_name
from lemonrind.storage import Database

DEMO_SERVER = Path(__file__).parent / "mcp_demo_server.py"


def demo(name: str = "Demo") -> ServerDefinition:
    return ServerDefinition(name, "stdio", command=sys.executable, args=(str(DEMO_SERVER),))


def make_module(tmp_path: Path | None = None) -> McpModule:
    settings = Settings()
    settings.modules.mcp.connect_timeout_seconds = 30.0
    return McpModule(
        settings,
        McpServerRepository(Database(":memory:")),
        log_dir=tmp_path / "logs" if tmp_path else None,
    )


# --- importing server definitions ----------------------------------------------------------------------------


def test_the_mcp_servers_block_is_the_main_shape():
    text = """{"mcpServers": {
        "postmark": {"command": "npx", "args": ["-y", "server-postmark"], "env": {"TOKEN": "abc", "N": 5}},
        "docs": {"url": "https://example.com/mcp", "headers": {"Authorization": "Bearer x"}}
    }}"""

    postmark, docs = parse_servers(text)

    assert (postmark.name, postmark.transport, postmark.command) == ("postmark", "stdio", "npx")
    assert postmark.args == ("-y", "server-postmark") and postmark.env == {"TOKEN": "abc", "N": "5"}
    assert (docs.transport, docs.url, docs.headers) == (
        "http",
        "https://example.com/mcp",
        {"Authorization": "Bearer x"},
    )


@pytest.mark.parametrize(
    "text",
    [
        '{"servers": {"a": {"command": "x"}}}',  # the VS Code wrapper
        '{"a": {"command": "x"}}',  # a bare name-to-server mapping
        '{"name": "a", "command": "x"}',  # one server written out on its own
    ],
)
def test_other_common_shapes_are_accepted(text):
    (server,) = parse_servers(text)
    assert (server.name, server.command) == ("a", "x")


def test_a_bare_server_without_a_name_gets_the_default_name():
    (server,) = parse_servers('{"command": "npx", "args": ["-y", "thing"]}', default_name="thing")
    assert server.name == "thing"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("{not json", "not valid JSON"),
        ("[1, 2]", "Expected a JSON object"),
        ("{}", "Expected a JSON object"),
        ('{"mcpServers": {}}', "at least one server"),
        ('{"mcpServers": {"a": 5}}', "must be an object"),
        ('{"mcpServers": {"a": {"args": []}}}', "needs either"),
        ('{"mcpServers": {"a": {"command": "x", "args": "-y"}}}', "list"),
        ('{"mcpServers": {"a": {"command": "x", "env": []}}}', "must be an object"),
        ('{"mcpServers": {"a": {"url": "ftp://x"}}}', "http://"),
        ('{"mcpServers": {"a": {"type": "sse", "url": "https://x"}}}', "SSE"),
    ],
)
def test_bad_json_is_explained_not_guessed_at(text, message):
    with pytest.raises(ServerImportError, match=message):
        parse_servers(text)


def test_a_command_line_is_split_like_a_shell_keeping_windows_paths():
    server = from_command_line(
        "local", r'"C:\Program Files\Python\python.exe" C:\tools\server.py --flag'
    )

    assert server.command == r"C:\Program Files\Python\python.exe"
    assert server.args == (r"C:\tools\server.py", "--flag")
    with pytest.raises(ServerImportError):
        from_command_line("x", "   ")
    with pytest.raises(ServerImportError):
        from_command_line("  ", "npx thing")


# --- storage ----------------------------------------------------------------------------------------------------


def test_servers_are_stored_with_their_settings_and_names_are_unique():
    repo = McpServerRepository(Database(":memory:"))
    config = repo.add(
        ServerDefinition("Mail", "stdio", command="npx", args=("-y", "m"), env={"K": "v"})
    )

    assert (config.enabled, config.require_approval) == (True, True)  # safe defaults
    assert (
        config.summary == "npx -y m" and "v" not in config.summary
    )  # no secrets in the one-line summary
    with pytest.raises(DuplicateServerError):
        repo.add(ServerDefinition("mail", "stdio", command="x"))

    repo.set_enabled(config.id, False)
    repo.set_require_approval(config.id, False)
    stored = repo.get(config.id)
    assert stored is not None and (stored.enabled, stored.require_approval) == (False, False)
    assert stored.env == {"K": "v"} and repo.find("MAIL") is not None
    repo.delete(config.id)
    assert repo.list() == []


# --- converting what a server sends back -----------------------------------------------------------------------


def reply(*content, is_error=False, structured=None) -> CallToolResult:
    return CallToolResult(content=list(content), is_error=is_error, structured_content=structured)


def test_text_blocks_are_joined_and_the_error_flag_is_kept():
    result = result_to_tool_result(
        reply(TextContent(type="text", text="a"), TextContent(type="text", text="b"))
    )
    assert (result.content, result.is_error) == ("a\nb", False)
    assert result_to_tool_result(
        reply(TextContent(type="text", text="bad"), is_error=True)
    ).is_error


def test_other_kinds_of_content_become_short_descriptions_never_raw_data():
    result = result_to_tool_result(
        reply(
            ImageContent(type="image", data="A" * 500, mime_type="image/png"),
            AudioContent(type="audio", data="B" * 40, mime_type="audio/wav"),
            EmbeddedResource(
                type="resource",
                resource=TextResourceContents(uri="file:///a.txt", text="inline text"),
            ),
            EmbeddedResource(
                type="resource", resource=BlobResourceContents(uri="file:///b.bin", blob="AAAA")
            ),
            ResourceLink(type="resource_link", name="doc", uri="https://example.com/doc"),
        )
    )

    assert "image result (image/png, 500 characters of data) not shown" in result.content
    assert "audio result (audio/wav" in result.content
    assert "inline text" in result.content
    assert "binary resource file:///b.bin not shown" in result.content
    assert "resource link: https://example.com/doc" in result.content
    assert "AAAA" not in result.content  # the data itself is never sent to the model


def test_empty_replies_fall_back_to_structured_content_or_a_note():
    assert result_to_tool_result(reply(structured={"total": 3})).content == '{"total": 3}'
    assert result_to_tool_result(reply()).content == "(the tool returned nothing)"


def test_errors_inside_exception_groups_are_found_and_missing_commands_named():
    assert describe_error(ExceptionGroup("g", [ValueError("inner")])) == "ValueError: inner"
    assert "Could not start 'npx-missing'" in describe_error(
        FileNotFoundError(2, "x"), "npx-missing"
    )
    assert "Timed out" in describe_error(TimeoutError())


def test_tool_names_are_prefixed_and_made_safe():
    assert slugify("My Server!") == "my_server" and slugify("!!!") == "server"
    assert tool_name("mail", "send email") == "mail__send_email"
    assert len(tool_name("mail", "x" * 100)) == 64


# --- real connections -------------------------------------------------------------------------------------------


async def test_a_server_connects_and_its_tools_work_with_a_prefix(tmp_path: Path):
    module = make_module(tmp_path)
    await module.on_startup()
    try:
        await module.add_servers([demo()])
        await module.wait_settled(60)

        (state,) = module.statuses()
        assert state.status == "connected" and state.error is None
        assert set(state.tool_names) == {"echo", "add", "explode", "pixel"}
        tools = {t.name: t for t in module.get_tools()}
        assert set(tools) == {"demo__echo", "demo__add", "demo__explode", "demo__pixel"}
        assert tools["demo__echo"].description.startswith("[Demo] Repeat the text back")
        assert tools["demo__add"].parameters["required"] == ["a", "b"]  # the server's own schema
        assert "$schema" not in tools["demo__add"].parameters

        assert (await tools["demo__echo"].run('{"text": "hi"}')).content == "Echo: hi"
        assert (await tools["demo__add"].run('{"a": 2, "b": 3}')).content == "5"
        failed = await tools["demo__explode"].run("{}")
        assert failed.is_error and "explode" in failed.content
        assert "image result" in (await tools["demo__pixel"].run("{}")).content
        assert (
            module.server_name_for("demo__echo") == "Demo"
            and module.server_name_for("nope") is None
        )
    finally:
        await module.on_shutdown()


async def test_a_wrong_argument_is_rejected_by_the_server_and_reported_as_an_error(tmp_path: Path):
    module = make_module(tmp_path)
    await module.on_startup()
    try:
        await module.add_servers([demo()])
        await module.wait_settled(60)
        tools = {t.name: t for t in module.get_tools()}

        result = await tools["demo__add"].run('{"a": "many", "b": 3}')

        assert result.is_error
    finally:
        await module.on_shutdown()


async def test_one_broken_server_does_not_affect_the_others(tmp_path: Path):
    module = make_module(tmp_path)
    await module.on_startup()
    try:
        await module.add_servers(
            [demo("Good"), ServerDefinition("Bad", "stdio", command="no-such-command-xyz")]
        )
        await module.wait_settled(60)

        by_name = {s.config.name: s for s in module.statuses()}
        assert by_name["Good"].status == "connected"
        assert by_name["Bad"].status == "failed" and "Could not start 'no-such-command-xyz'" in (
            by_name["Bad"].error or ""
        )
        assert {t.name for t in module.get_tools()} == {
            f"good__{n}" for n in ("echo", "add", "explode", "pixel")
        }
    finally:
        await module.on_shutdown()


async def test_servers_can_be_switched_off_restarted_and_removed_one_at_a_time(tmp_path: Path):
    module = make_module(tmp_path)
    await module.on_startup()
    try:
        first, second = await module.add_servers([demo("One"), demo("Two")])
        await module.wait_settled(60)
        assert len(module.get_tools()) == 8

        await module.set_enabled(first.id, False)
        assert {t.name.split("__")[0] for t in module.get_tools()} == {"two"}
        assert {s.config.name: s.status for s in module.statuses()}["One"] == "stopped"

        await module.set_enabled(first.id, True)
        await module.wait_settled(60)
        assert len(module.get_tools()) == 8

        await module.restart(second.id)
        await module.wait_settled(60)
        assert len(module.get_tools()) == 8

        await module.remove(first.id)
        assert {t.name.split("__")[0] for t in module.get_tools()} == {"two"}
        assert [s.config.name for s in module.statuses()] == ["Two"]
    finally:
        await module.on_shutdown()


async def test_the_approval_setting_applies_to_the_next_call(tmp_path: Path):
    module = make_module(tmp_path)
    await module.on_startup()
    try:
        (config,) = await module.add_servers([demo()])
        await module.wait_settled(60)
        registry = ModuleRegistry([module])
        call = ToolCall("c", "demo__echo", '{"text": "x"}')
        assert registry.requires_approval(call)  # new servers ask first

        module.set_require_approval(config.id, False)

        assert not registry.requires_approval(call)
        assert not registry.requires_approval(ToolCall("c", "unknown_tool", "{}"))
        assert (await registry.run(call)).content == "Echo: x"
    finally:
        await module.on_shutdown()


async def test_servers_added_while_the_module_is_not_running_wait_for_start_up(tmp_path: Path):
    module = make_module(tmp_path)
    await module.add_servers([demo()])  # on_startup has not run yet
    assert module.get_tools() == [] and module.statuses()[0].status == "stopped"

    await module.on_startup()  # now every enabled stored server connects
    try:
        await module.wait_settled(60)
        assert len(module.get_tools()) == 4
    finally:
        await module.on_shutdown()


async def test_two_servers_whose_names_look_alike_get_different_prefixes(tmp_path: Path):
    module = make_module(tmp_path)
    await module.on_startup()
    try:
        await module.add_servers([demo("My Server"), demo("my-server")])
        await module.wait_settled(60)

        names = [t.name for t in module.get_tools()]

        assert len(names) == len(set(names)) == 8
        assert {n.split("__")[0] for n in names} == {"my_server", "my_server_2"}
    finally:
        await module.on_shutdown()


async def test_shutdown_stops_every_server_process(tmp_path: Path):
    module = make_module(tmp_path)
    await module.on_startup()
    await module.add_servers([demo()])
    await module.wait_settled(60)

    await module.on_shutdown()

    assert module.get_tools() == []
    assert module.statuses()[0].status == "stopped"


# --- connection edge cases -----------------------------------------------------------------------------------------


async def test_a_server_that_never_answers_times_out_with_a_clear_message(tmp_path: Path):
    repo = McpServerRepository(Database(":memory:"))
    config = repo.add(
        ServerDefinition(
            "Hangs", "stdio", command=sys.executable, args=("-c", "import time; time.sleep(30)")
        )
    )
    connection = McpConnection(config, log_path=tmp_path / "hangs.log", connect_timeout=1.5)

    connection.start()
    assert await connection.wait_settled(20)

    assert connection.status == "failed" and "Timed out" in (connection.error or "")
    await connection.stop()


async def test_stopping_a_connection_that_is_still_connecting_is_quick(tmp_path: Path):
    repo = McpServerRepository(Database(":memory:"))
    config = repo.add(
        ServerDefinition(
            "Hangs", "stdio", command=sys.executable, args=("-c", "import time; time.sleep(30)")
        )
    )
    connection = McpConnection(config, connect_timeout=60)
    connection.start()
    await asyncio.sleep(0.5)

    started = time.monotonic()
    await connection.stop()

    assert time.monotonic() - started < 10 and connection.status == "stopped"


async def test_a_server_that_prints_an_error_and_exits_shows_what_it_said(tmp_path: Path):
    repo = McpServerRepository(Database(":memory:"))
    config = repo.add(
        ServerDefinition(
            "Crashes",
            "stdio",
            command=sys.executable,
            args=("-c", "import sys; sys.stderr.write('API key missing\\n'); sys.exit(3)"),
        )
    )
    connection = McpConnection(config, log_path=tmp_path / "crash.log", connect_timeout=30)

    connection.start()
    await connection.wait_settled(30)

    assert connection.status == "failed" and "API key missing" in (connection.error or "")
    await connection.stop()


async def test_calling_a_tool_on_a_server_that_is_not_connected_is_an_error_result():
    repo = McpServerRepository(Database(":memory:"))
    config = repo.add(demo())
    connection = McpConnection(config)

    result = await connection.call_tool("echo", {"text": "x"})

    assert result.is_error and "not connected" in result.content


async def test_tools_without_a_validation_model_get_the_arguments_dictionary_untouched():
    seen = []

    async def run(arguments: dict) -> str:
        seen.append(arguments)
        return "ok"

    tool = Tool(name="t", description="d", parameters={"type": "object"}, func=run)

    assert (await tool.run('{"anything": [1, 2], "goes": {"here": true}}')).content == "ok"
    assert seen == [{"anything": [1, 2], "goes": {"here": True}}]
    assert tool.requires_approval is False


def test_servers_get_the_network_settings_and_node_default_but_not_your_other_secrets():
    from lemonrind.modules.mcp.connection import server_environment

    parent = {
        "HTTPS_PROXY": "http://proxy:3128",
        "NODE_EXTRA_CA_CERTS": "C:/cert.pem",
        "OPENAI_API_KEY": "secret",
    }

    env = server_environment({"TOKEN": "t"}, parent)

    assert (
        env
        == {
            "HTTPS_PROXY": "http://proxy:3128",
            "NODE_EXTRA_CA_CERTS": "C:/cert.pem",
            "TOKEN": "t",
            "NODE_USE_SYSTEM_CA": "1",  # the default that avoids slow starts behind an HTTPS-inspecting antivirus
        }
    )
    assert "OPENAI_API_KEY" not in env


def test_configured_and_inherited_values_win_over_the_node_default():
    from lemonrind.modules.mcp.connection import server_environment

    assert server_environment({"NODE_USE_SYSTEM_CA": "0"}, {})["NODE_USE_SYSTEM_CA"] == "0"
    assert server_environment({}, {"NODE_USE_SYSTEM_CA": "0"})["NODE_USE_SYSTEM_CA"] == "0"
