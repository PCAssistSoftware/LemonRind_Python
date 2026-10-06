"""A tiny MCP server used by the tests and for trying the MCP module by hand.

Run it as a program (``python tests/mcp_demo_server.py``) and it speaks MCP over its standard input and
output, which is exactly how the app talks to servers started with a ``command``. It offers four tools that
cover the cases the client must handle: normal text results, structured arguments, an error, and a
non-text result.
"""

from __future__ import annotations

import base64

from mcp.server.mcpserver import MCPServer
from mcp.types import ImageContent

server = MCPServer("demo")


@server.tool()
def echo(text: str) -> str:
    """Repeat the text back."""
    return f"Echo: {text}"


@server.tool()
def add(a: int, b: int) -> int:
    """Add two whole numbers."""
    return a + b


@server.tool()
def explode(reason: str = "no reason") -> str:
    """Always fails (to test error handling)."""
    raise RuntimeError(f"explosion: {reason}")


@server.tool()
def pixel() -> list[ImageContent]:
    """Return a 1x1 image (to test non-text results)."""
    png = base64.b64encode(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
            "1f15c4890000000d49444154789c6360000002000001e221bc330000000049454e44ae426082"
        )
    ).decode()
    return [ImageContent(type="image", data=png, mime_type="image/png")]


if __name__ == "__main__":
    server.run("stdio")
