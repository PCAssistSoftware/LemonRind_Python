"""The MCP servers module: use tools that other programs provide.

This module owns the live connections (``McpConnection``, one per enabled server) and presents what the
servers offer as ordinary ``Tool`` objects, so the tool loop treats them exactly like the built-in tools.

Things worth knowing:

* **Start-up never waits.** Connecting to a server can take a minute (the first run of an ``npx`` server downloads
  it). ``on_startup`` only *starts* the connections and returns, so the app is usable at once; a server's tools
  appear as soon as it has connected, and a message sent earlier simply sees fewer tools. One server failing
  (wrong command, bad token) never affects the others; it shows up as "failed" with the reason.
* **Changes are live and per server.** Adding, enabling, disabling, restarting or removing one server touches only
  that connection, with no restart of the app or of the other servers.
* **Tool names are prefixed** with the server's name (``postmark__send_email``), because two servers may both
  offer a tool called ``search``, and because it tells the model (and you, in the chat) where a call goes.
* **Tools ask permission by default.** An MCP tool can send an email or delete files, and a model can be
  tricked into calling it by text in a web page or document. Each server has a setting, on by default, to ask you
  before every call. (See the ``approve`` hook on ``Conversation``.)

Python ideas used here:

* Dynamic tools: ``get_tools`` builds its answer from whatever is connected *right now*, so the module needs no
  list of tools written in advance. A closure (``make_runner``) captures which server and tool each ``Tool``
  belongs to.
* ``asyncio.gather`` to stop all connections at the same time.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lemonrind.config import Settings
from lemonrind.modules.base import Module
from lemonrind.modules.mcp.connection import McpConnection, RemoteTool, Status
from lemonrind.modules.mcp.repository import (
    McpServerConfig,
    McpServerRepository,
    ServerDefinition,
)
from lemonrind.modules.tool import Tool, ToolError

MAX_TOOL_NAME = 64  # the limit chat APIs put on a function name


@dataclass(frozen=True, slots=True)
class ServerStatus:
    """What a screen needs to show about one server."""

    config: McpServerConfig
    status: Status  # "stopped" also covers a disabled server
    error: str | None
    tool_names: tuple[str, ...]  # as the server calls them (without our prefix)


def slugify(name: str) -> str:
    """``'My Server!'`` becomes ``'my_server'``: safe to use inside a function name."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "server"


def tool_name(slug: str, remote_name: str) -> str:
    """The name the model sees: server prefix, two underscores, the server's own name for the tool."""
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", remote_name)
    return f"{slug}__{cleaned}"[:MAX_TOOL_NAME]


class McpModule(Module):
    name = "MCP servers"
    config_key = "mcp"
    description = (
        "Connect to MCP servers (email, files, databases, ...) and use the tools they offer."
    )

    def __init__(
        self, settings: Settings, repo: McpServerRepository, *, log_dir: Path | None = None
    ) -> None:
        super().__init__(settings)
        self.repo = repo
        self._log_dir = log_dir
        self._connections: dict[str, McpConnection] = {}  # by server id
        self._running = False  # between on_startup and on_shutdown

    # --- lifecycle (the module hooks) ------------------------------------------------------------------------

    async def on_startup(self) -> None:
        self._running = True
        for config in self.repo.list():
            if config.enabled:
                self._connect(config)

    async def on_shutdown(self) -> None:
        self._running = False
        connections = list(self._connections.values())
        self._connections.clear()
        await asyncio.gather(*(connection.stop() for connection in connections))

    def _connect(self, config: McpServerConfig) -> None:
        mcp = self.settings.modules.mcp
        log_path = self._log_dir / f"mcp-{slugify(config.name)}.log" if self._log_dir else None
        connection = McpConnection(
            config,
            log_path=log_path,
            connect_timeout=mcp.connect_timeout_seconds,
            call_timeout=mcp.call_timeout_seconds,
        )
        self._connections[config.id] = connection
        connection.start()

    async def _disconnect(self, server_id: str) -> None:
        if connection := self._connections.pop(server_id, None):
            await connection.stop()

    # --- managing servers (used by the screens) -------------------------------------------------------------

    async def add_servers(self, definitions: list[ServerDefinition]) -> list[McpServerConfig]:
        """Store the servers (they ask permission before running tools) and connect them if the module runs."""
        added = []
        for definition in definitions:
            config = self.repo.add(definition)
            added.append(config)
            if self._running:
                self._connect(config)
        return added

    async def set_enabled(self, server_id: str, enabled: bool) -> None:
        self.repo.set_enabled(server_id, enabled)
        await self._disconnect(server_id)
        if enabled and self._running and (config := self.repo.get(server_id)):
            self._connect(config)

    def set_require_approval(self, server_id: str, require: bool) -> None:
        self.repo.set_require_approval(server_id, require)
        if (connection := self._connections.get(server_id)) and (
            config := self.repo.get(server_id)
        ):
            connection.config = config  # takes effect on the very next tool call

    async def restart(self, server_id: str) -> None:
        await self._disconnect(server_id)
        if self._running and (config := self.repo.get(server_id)) and config.enabled:
            self._connect(config)

    async def remove(self, server_id: str) -> None:
        await self._disconnect(server_id)
        self.repo.delete(server_id)

    def statuses(self) -> list[ServerStatus]:
        result = []
        for config in self.repo.list():
            connection = self._connections.get(config.id)
            result.append(
                ServerStatus(
                    config=config,
                    status=connection.status if connection else "stopped",
                    error=connection.error if connection else None,
                    tool_names=tuple(t.name for t in connection.tools) if connection else (),
                )
            )
        return result

    async def wait_settled(self, timeout: float | None = None) -> None:
        """Wait until every connection has finished connecting (successfully or not)."""
        await asyncio.gather(*(c.wait_settled(timeout) for c in list(self._connections.values())))

    # --- tools -----------------------------------------------------------------------------------------------

    def get_tools(self) -> list[Tool]:
        tools: list[Tool] = []
        used: set[str] = set()
        for connection in self._connections.values():
            if connection.status != "connected":
                continue
            slug = self._slug_for(connection.config, used)
            for remote in connection.tools:
                tools.append(self._make_tool(connection, slug, remote))
        return tools

    def _slug_for(self, config: McpServerConfig, used: set[str]) -> str:
        """The prefix for this server's tools, made unique if two servers' names slugify alike."""
        slug, number = slugify(config.name), 2
        while slug in used:
            slug, number = f"{slugify(config.name)}_{number}", number + 1
        used.add(slug)
        return slug

    def _make_tool(self, connection: McpConnection, slug: str, remote: RemoteTool) -> Tool:
        runner = self._runner(connection, remote.name)
        return Tool(
            name=tool_name(slug, remote.name),
            description=f"[{connection.config.name}] {remote.description}".strip(),
            parameters=remote.schema,
            func=runner,
            args_model=None,  # the server validates its own arguments
            requires_approval=connection.config.require_approval,
        )

    @staticmethod
    def _runner(
        connection: McpConnection, remote_name: str
    ) -> Callable[[dict[str, Any]], Awaitable[str]]:
        async def run(arguments: dict[str, Any]) -> str:
            result = await connection.call_tool(remote_name, arguments)
            if result.is_error:
                raise ToolError(result.content)
            return result.content

        return run

    def server_name_for(self, tool: str) -> str | None:
        """Which server a (prefixed) tool name belongs to, for showing in a permission question."""
        used: set[str] = set()
        for connection in self._connections.values():
            slug = self._slug_for(connection.config, used)
            if any(tool_name(slug, remote.name) == tool for remote in connection.tools):
                return connection.config.name
        return None
