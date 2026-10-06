"""Reading server settings from the JSON that MCP servers' own documentation tells you to paste.

Nearly every MCP server's README has a JSON block like this to paste into an MCP-aware desktop app or editor::

    {
      "mcpServers": {
        "postmark": {
          "command": "npx",
          "args": ["-y", "@modelcontextprotocol/server-postmark"],
          "env": {"POSTMARK_SERVER_TOKEN": "..."}
        }
      }
    }

Asking you to retype that into separate form fields would be silly, so *pasting it* is the main way to add a
server (the same lesson the .NET editions learned from using them). This module turns pasted text into
``ServerDefinition`` objects and is strict about what it accepts: the text is typed or pasted by a person and
the result will be used to *start programs*, so a clear error beats a guess.

Accepted shapes: ``{"mcpServers": {...}}`` (the most common), ``{"servers": {...}}`` (the other widely used one), a plain
``{"name": {...}}`` mapping, or one bare server object (``{"command": ...}``, optionally with a ``"name"``).

Python ideas used here:

* ``json.loads`` and checking the *shape* of what came back with ``isinstance``, because JSON can be anything.
* ``shlex.split`` for splitting a command line the way a shell would (used by the terminal ``/mcp add``).
* A custom exception with a message fit to show.
"""

from __future__ import annotations

import json
import re
import shlex
from typing import Any

from lemonrind.modules.mcp.repository import ServerDefinition


class ServerImportError(ValueError):
    """The pasted text cannot be turned into servers. The message says what is wrong."""


def parse_servers(text: str, *, default_name: str = "server") -> list[ServerDefinition]:
    """Turn pasted JSON into server definitions. Raises ``ServerImportError`` if it cannot."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise ServerImportError(
            f"That is not valid JSON ({error.msg} at line {error.lineno})."
        ) from error
    if not isinstance(data, dict) or not data:
        raise ServerImportError('Expected a JSON object such as {"mcpServers": {...}}.')

    for wrapper in ("mcpServers", "servers"):
        if wrapper in data:
            data = data[wrapper]
            if not isinstance(data, dict) or not data:
                raise ServerImportError(
                    f"'{wrapper}' must be an object with at least one server in it."
                )
            break
    else:
        if "command" in data or "url" in data:  # one server written out on its own
            data = {str(data.get("name") or default_name): data}

    definitions = []
    for name, entry in data.items():
        if not isinstance(entry, dict):
            raise ServerImportError(f"The entry for '{name}' must be an object.")
        definitions.append(_definition(str(name), entry))
    return definitions


def _definition(name: str, entry: dict[str, Any]) -> ServerDefinition:
    name = name.strip()
    if not name:
        raise ServerImportError("A server needs a name.")
    kind = str(entry.get("type", "")).lower()
    if kind == "sse":
        raise ServerImportError(
            f"'{name}' uses the older SSE transport, which is not supported. Use a stdio command or a "
            "streamable HTTP address."
        )

    if "url" in entry or kind in ("http", "streamable-http", "streamable_http"):
        url = str(entry.get("url", "")).strip()
        if not url.startswith(("http://", "https://")):
            raise ServerImportError(f"'{name}': the url must start with http:// or https://")
        return ServerDefinition(
            name=name,
            transport="http",
            url=url,
            headers=_string_map(name, "headers", entry.get("headers")),
        )

    command = str(entry.get("command", "")).strip()
    if not command:
        raise ServerImportError(f'\'{name}\' needs either a "command" (to start it) or a "url".')
    args = entry.get("args", [])
    if not isinstance(args, list):
        raise ServerImportError(f"'{name}': \"args\" must be a list of text values.")
    return ServerDefinition(
        name=name,
        transport="stdio",
        command=command,
        args=tuple(str(arg) for arg in args),
        env=_string_map(name, "env", entry.get("env")),
    )


def _string_map(server: str, key: str, value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ServerImportError(f"'{server}': \"{key}\" must be an object.")
    return {str(k): str(v) for k, v in value.items()}  # numbers and booleans become their text


def from_command_line(name: str, command_line: str) -> ServerDefinition:
    """``npx -y some-server`` typed in the terminal becomes a definition (split like a shell would)."""
    try:
        parts = shlex.split(
            command_line, posix=False
        )  # posix=False keeps Windows backslashes intact
    except ValueError as error:
        raise ServerImportError(f"Could not read that command line: {error}") from error
    parts = [part.strip('"') for part in parts]
    if not parts:
        raise ServerImportError("Give the command that starts the server.")
    return ServerDefinition(
        name=_clean_name(name), transport="stdio", command=parts[0], args=tuple(parts[1:])
    )


def _clean_name(name: str) -> str:
    cleaned = re.sub(r"\s+", " ", name).strip()
    if not cleaned:
        raise ServerImportError("A server needs a name.")
    return cleaned
