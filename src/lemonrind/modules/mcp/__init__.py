"""MCP servers: use the tools other programs provide, through the Model Context Protocol."""

from lemonrind.modules.mcp.importer import ServerImportError, from_command_line, parse_servers
from lemonrind.modules.mcp.module import McpModule, ServerStatus
from lemonrind.modules.mcp.repository import (
    DuplicateServerError,
    McpServerConfig,
    McpServerRepository,
    ServerDefinition,
)

__all__ = [
    "DuplicateServerError",
    "ServerImportError",
    "McpModule",
    "McpServerConfig",
    "McpServerRepository",
    "ServerDefinition",
    "ServerStatus",
    "from_command_line",
    "parse_servers",
]
