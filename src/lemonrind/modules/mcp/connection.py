"""One live connection to one MCP server.

**The Model Context Protocol (MCP)** is an open standard for giving AI applications tools. A *server* offers
tools (send an email, query a database, read a calendar); a *client* (this app) connects, asks what tools exist,
and calls them. Because the protocol is standard, any MCP server works with any MCP client, with no
code written for each one.

Two ways to reach a server:

* **stdio**: the app starts the server as a child program (``npx -y some-server``, ``python server.py``) and
  they exchange JSON messages through the child's standard input and output;
* **http** ("streamable HTTP"): the server is already running somewhere and is reached at a web address.

Why a background task per connection? The ``mcp`` library keeps a connection open with ``async with`` blocks
(``async with Client(...) as client:``), and the connection only lives as long as that block. We want it to live as
long as the app does, and Python's async machinery requires that a connection be opened *and* closed by the
same task. So each connection gets one long-running task that opens it, announces "ready", waits for a stop
signal, and closes it, while other tasks simply call ``client.call_tool`` in the meantime.

Python ideas used here:

* ``contextlib.AsyncExitStack``: enter several ``async with`` contexts in code that is not a ``with`` block, and
  close them all in the right order at the end.
* ``asyncio.Event`` as a signal between tasks (``_ready``, ``_stop``); ``asyncio.timeout`` to bound a wait.
* ``ExceptionGroup``: libraries built on task groups raise several errors at once; ``describe_error`` finds
  the one worth showing.
* Importing the heavy ``mcp`` package only when a connection is actually made.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections.abc import Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from lemonrind.modules.mcp.repository import McpServerConfig
from lemonrind.modules.tool import ToolResult

logger = logging.getLogger(__name__)

type Status = Literal["stopped", "connecting", "connected", "failed"]

LOG_TAIL_LINES = (
    6  # how much of a failing server's own error output is added to the message shown to you
)


@dataclass(frozen=True, slots=True)
class RemoteTool:
    """A tool as the server describes it."""

    name: str
    description: str
    schema: dict[str, Any]  # JSON schema of the arguments, written by the server


# The ``mcp`` library starts a server with only a short whitelist of the parent's environment variables, so
# that your other secrets (API keys in the environment, say) are not handed to every server. That whitelist
# leaves out the settings a program needs to reach the internet through a proxy or an HTTPS-inspecting
# antivirus, which makes ``npx`` servers extremely slow to start (about 70 seconds here). These are passed on.
PASSED_THROUGH = (
    "NODE_EXTRA_CA_CERTS", "NODE_USE_SYSTEM_CA", "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE",
    "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy",
)  # fmt: skip


def server_environment(configured: Mapping[str, str], parent: Mapping[str, str]) -> dict[str, str]:
    """The extra environment variables for a server: the network settings above, then the ones you configured
    (which win), and ``NODE_USE_SYSTEM_CA=1`` unless either already says otherwise. That last one makes Node
    trust the operating system's certificate store, where an antivirus installs its own certificate."""
    env = {name: parent[name] for name in PASSED_THROUGH if name in parent}
    env.update(configured)
    env.setdefault("NODE_USE_SYSTEM_CA", "1")
    return env


def describe_error(error: BaseException, command: str = "") -> str:
    """A short readable description, looking inside ``ExceptionGroup`` (several errors at once)."""
    if isinstance(error, BaseExceptionGroup):
        return describe_error(error.exceptions[0], command)
    if isinstance(error, FileNotFoundError):
        return f"Could not start '{command or error.filename or 'the command'}'. Is it installed and on the PATH?"
    if isinstance(error, TimeoutError):
        return "Timed out. (The first run of an npx server downloads it, which can take a minute.)"
    return f"{type(error).__name__}: {error}"


def result_to_tool_result(result: Any) -> ToolResult:
    """Turn the server's reply into text for the model.

    A reply is a list of *content blocks*: text, images, audio, embedded resources, links. A language model
    reading this app's messages can only use text, so other kinds become a short description (never the raw
    base64 data, which would waste thousands of tokens).
    """
    parts: list[str] = []
    for block in result.content or []:
        kind = getattr(block, "type", "")
        if kind == "text":
            parts.append(block.text)
        elif kind in ("image", "audio"):
            parts.append(
                f"[{kind} result ({block.mime_type}, {len(block.data):,} characters of data) not shown]"
            )
        elif kind == "resource":
            resource = block.resource
            text = getattr(resource, "text", None)
            parts.append(
                text if text is not None else f"[binary resource {resource.uri} not shown]"
            )
        elif kind == "resource_link":
            parts.append(f"[resource link: {block.uri}]")
        else:
            parts.append(f"[{kind or 'unknown'} content not shown]")
    if not parts and getattr(result, "structured_content", None):
        parts.append(json.dumps(result.structured_content, default=str))
    return ToolResult("\n".join(parts) or "(the tool returned nothing)", bool(result.is_error))


class McpConnection:
    def __init__(
        self,
        config: McpServerConfig,
        *,
        log_path: Path | None = None,
        connect_timeout: float = 90.0,
        call_timeout: float = 120.0,
    ) -> None:
        self.config = config
        self.status: Status = "stopped"
        self.error: str | None = None
        self.tools: list[RemoteTool] = []
        self._log_path = log_path
        self._connect_timeout = connect_timeout
        self._call_timeout = call_timeout
        self._client: Any = None
        self._task: asyncio.Task | None = None
        self._ready = asyncio.Event()  # set when connecting has finished, successfully or not
        self._stop = asyncio.Event()  # set to ask the task to close the connection

    # --- lifecycle -----------------------------------------------------------------------------------------

    def start(self) -> None:
        """Begin connecting in the background and return at once."""
        if self._task is not None:
            return
        self.status = "connecting"
        self._task = asyncio.create_task(self._run(), name=f"mcp-{self.config.name}")

    async def wait_settled(self, timeout: float | None = None) -> bool:
        """Wait until connecting has finished (connected or failed). False if ``timeout`` ran out first."""
        try:
            async with asyncio.timeout(timeout):
                await self._ready.wait()
        except TimeoutError:
            return False
        return True

    async def stop(self) -> None:
        """Close the connection (and the server program, for stdio) and wait for that to finish."""
        self._stop.set()
        task, self._task = self._task, None
        if task is not None:
            if self.status == "connecting":  # nothing to say goodbye to yet: abandon the attempt
                task.cancel()
            # asyncio.wait never raises the task's own outcome (a cancelled or failed task is simply "done"),
            # unlike ``await task``, and gives up after a timeout without cancelling anything itself.
            _, still_running = await asyncio.wait({task}, timeout=15)
            if still_running:
                task.cancel()
        self.status, self._client = "stopped", None

    async def _run(self) -> None:
        from mcp import Client  # imported here: the package is slow to import and often not needed

        try:
            async with AsyncExitStack() as stack:
                if self._log_path is not None:  # the server's own error output goes to a log file
                    self._log_path.parent.mkdir(parents=True, exist_ok=True)
                    log = stack.enter_context(
                        self._log_path.open("a", encoding="utf-8", errors="replace")
                    )
                    log.write(
                        f"\n--- {datetime.now():%Y-%m-%d %H:%M:%S} connecting to {self.config.name}\n"
                    )
                    log.flush()
                else:
                    log = stack.enter_context(Path(os.devnull).open("w"))  # noqa: SIM115
                try:
                    async with asyncio.timeout(self._connect_timeout):
                        transport = await self._open_transport(stack, log)
                        client = await stack.enter_async_context(
                            Client(transport, read_timeout_seconds=self._call_timeout)
                        )
                        listing = await self._list_all_tools(client)
                except BaseException as error:
                    if isinstance(error, asyncio.CancelledError):
                        raise
                    self.status = "failed"
                    self.error = describe_error(error, self.config.command) + self._log_tail()
                    logger.warning(
                        "MCP server %s failed to connect: %s", self.config.name, self.error
                    )
                    return
                self.tools = listing
                self._client = client
                self.status = "connected"
                self._ready.set()
                await self._stop.wait()  # stay connected until asked to leave
        except Exception as error:  # trouble while closing is not worth surfacing
            logger.debug("MCP server %s closed with an error", self.config.name, exc_info=error)
        finally:
            self._client = None
            if self.status != "failed":
                self.status = "stopped"
            self._ready.set()

    async def _open_transport(self, stack: AsyncExitStack, log: Any) -> Any:
        config = self.config
        if config.transport == "http":
            import httpx2 as httpx
            from mcp.client.streamable_http import streamable_http_client

            http = await stack.enter_async_context(
                httpx.AsyncClient(
                    headers=dict(config.headers), timeout=self._call_timeout, follow_redirects=True
                )
            )
            return streamable_http_client(config.url, http_client=http)

        from mcp import StdioServerParameters
        from mcp.client.stdio import stdio_client

        env = server_environment(config.env, os.environ)
        params = StdioServerParameters(command=config.command, args=list(config.args), env=env)
        return stdio_client(params, errlog=log)

    @staticmethod
    async def _list_all_tools(client: Any) -> list[RemoteTool]:
        tools: list[RemoteTool] = []
        cursor = None
        while True:  # a server may return its tools a page at a time
            page = await client.list_tools(cursor=cursor)
            for tool in page.tools:
                schema = dict(tool.input_schema or {"type": "object", "properties": {}})
                schema.pop("$schema", None)  # a meta-keyword some model servers cannot parse
                tools.append(
                    RemoteTool(tool.name, (tool.description or tool.title or "").strip(), schema)
                )
            cursor = page.next_cursor
            if not cursor:
                return tools

    def _log_tail(self) -> str:
        """The last lines the server printed to its error output: usually the real reason for a failure."""
        if self._log_path is None or not self._log_path.exists():
            return ""
        lines = [
            line
            for line in self._log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.strip() and not line.startswith("--- ")
        ]
        tail = " | ".join(re.sub(r"\s+", " ", line).strip() for line in lines[-LOG_TAIL_LINES:])
        return f" (server said: {tail[:400]})" if tail else ""

    # --- using it ------------------------------------------------------------------------------------------

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if self.status != "connected" or self._client is None:
            return ToolResult(f"The MCP server '{self.config.name}' is not connected.", True)
        try:
            result = await self._client.call_tool(
                name, arguments, read_timeout_seconds=self._call_timeout
            )
        except TimeoutError:
            return ToolResult(
                f"'{name}' did not answer within {self._call_timeout:.0f} seconds.", True
            )
        except Exception as error:  # the server crashed, or rejected the call at protocol level
            return ToolResult(
                f"The MCP server '{self.config.name}' failed: {describe_error(error)}", True
            )
        return result_to_tool_result(result)
