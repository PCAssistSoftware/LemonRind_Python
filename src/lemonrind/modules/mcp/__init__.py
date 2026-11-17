"""MCP servers: use the tools other programs provide, through the Model Context Protocol."""

from lemonrind.modules.mcp.importer import (
    LEMONADE_SERVER_NAME,
    ServerImportError,
    from_command_line,
    lemonade_gateway,
    parse_servers,
)
from lemonrind.modules.mcp.module import McpModule, ServerStatus
from lemonrind.modules.mcp.repository import (
    DuplicateServerError,
    McpServerConfig,
    McpServerRepository,
    ServerDefinition,
)

__all__ = [
    "LEMONADE_SERVER_NAME",
    "DuplicateServerError",
    "ServerImportError",
    "McpModule",
    "McpServerConfig",
    "McpServerRepository",
    "ServerDefinition",
    "ServerStatus",
    "from_command_line",
    "lemonade_gateway",
    "parse_servers",
]
