"""The MCP servers section of Settings: add servers by pasting their JSON, see whether they are connected, switch them.

MCP servers' documentation gives you a JSON block to paste into other apps; pasting that same block here is the
whole "add a server" step. Each server is then listed with its live state (connecting, connected with N tools,
or failed with the reason), and can be switched on or off, restarted, removed, or told to run its tools without
asking first. Everything takes effect at once, for that server only.

Python / NiceGUI ideas used here:

* ``ui.switch(value=..., on_change=...)`` with ``lambda`` default arguments (``c=config``) to capture the
  current loop item.
* A ``ui.timer`` that redraws the list while any server is still connecting, as in the knowledge screen.
"""

from __future__ import annotations

from nicegui import ui

from lemonrind.modules.mcp import (
    DuplicateServerError,
    McpModule,
    ServerImportError,
    ServerStatus,
    parse_servers,
)
from lemonrind.webui.dialogs import ask_confirm

EXAMPLE = (
    '{"mcpServers": {"everything": {"command": "npx", "args": ["-y", '
    '"@modelcontextprotocol/server-everything"]}}}'
)


def build_mcp(module: McpModule) -> None:
    """Draw the section into the current container (the Settings dialog puts it in a panel)."""

    @ui.refreshable
    def servers() -> None:
        statuses = module.statuses()
        if not statuses:
            ui.label("No servers yet. Paste a server's JSON below.").classes(
                "text-caption lr-muted"
            )
        for item in statuses:
            server_row(item)

    def server_row(item: ServerStatus) -> None:
        config = item.config
        with ui.column().classes("w-full gap-0 q-pa-sm lr-tool"):
            with ui.row().classes("w-full items-center no-wrap gap-2"):
                ui.switch(
                    value=config.enabled,
                    on_change=lambda e, c=config: toggle(c.id, e.value),
                ).tooltip("Connect to this server")
                # min-width 0 lets this column shrink below its text, so a long command line wraps instead of
                # pushing the row wider than the dialog
                with ui.column().classes("gap-0 flex-grow").style("min-width: 0"):
                    ui.label(config.name).classes("text-weight-medium")
                    ui.label(config.summary).classes("text-caption lr-muted lr-wrap")
                if item.status == "connecting":
                    ui.spinner(size="sm")
                ui.button(icon="refresh", on_click=lambda c=config: restart(c.id)).props(
                    "flat dense round"
                ).tooltip("Reconnect")
                ui.button(icon="delete", on_click=lambda c=config: remove(c)).props(
                    "flat dense round color=negative"
                ).tooltip("Remove this server")
            if not config.enabled:
                ui.label("Off").classes("text-caption lr-muted")
            elif item.status == "connecting":
                ui.label("Connecting... (the first run of an npx server downloads it)").classes(
                    "text-caption lr-warn"
                )
            elif item.status == "failed":
                ui.label(item.error or "Failed").classes("text-caption text-negative lr-wrap")
            elif item.status == "connected":
                names = ", ".join(item.tool_names) or "no tools"
                ui.label(f"Connected. Tools: {names}").classes("text-caption text-positive")
            ui.switch(
                "Ask me before running its tools",
                value=config.require_approval,
                on_change=lambda e, c=config: module.set_require_approval(c.id, e.value),
            ).props("dense")

    async def toggle(server_id: str, enabled: bool) -> None:
        await module.set_enabled(server_id, enabled)
        servers.refresh()

    async def restart(server_id: str) -> None:
        await module.restart(server_id)
        servers.refresh()

    async def remove(config) -> None:
        if await ask_confirm("Remove server", f"Remove '{config.name}'?", ok_text="Remove"):
            await module.remove(config.id)
            servers.refresh()

    async def import_json() -> None:
        try:
            added = await module.add_servers(parse_servers(json_input.value or ""))
        except (ServerImportError, DuplicateServerError) as error:
            ui.notify(str(error), type="negative", multi_line=True)
            return
        json_input.value = ""
        ui.notify(f"Added {len(added)} server(s). They ask before running tools.", type="positive")
        servers.refresh()

    was_connecting = False

    def poll() -> None:
        """Redraw while any server is connecting, and once more when it has settled (to show the result)."""
        nonlocal was_connecting
        connecting = any(s.status == "connecting" and s.config.enabled for s in module.statuses())
        if connecting or was_connecting:
            servers.refresh()
        was_connecting = connecting

    with ui.column().classes("w-full gap-2"):
        ui.label(
            "MCP servers give the assistant extra tools: email, files, databases, calendars and more. "
            "Only add servers you trust: their tools can act on your behalf."
        ).classes("text-caption lr-muted")
        ui.label("Add servers").classes("text-weight-medium")
        ui.label(
            'Paste the JSON from the server\'s documentation (the "mcpServers" block). Example:'
        ).classes("text-caption lr-muted")
        ui.label(EXAMPLE).classes("lr-tool-text")
        json_input = (
            ui.textarea(placeholder="Paste server JSON here")
            .props("outlined autogrow")
            .classes("w-full")
            .mark("mcp-json")
        )
        ui.button("Add", on_click=import_json).props("color=primary").mark("mcp-add")
        ui.separator()
        ui.label("Your servers").classes("text-weight-medium")
        servers()

    ui.timer(1.0, poll)
