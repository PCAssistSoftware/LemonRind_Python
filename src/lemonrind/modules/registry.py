"""The registry: knows every module, and is what the conversation talks to when the model wants a tool.

``ModuleRegistry`` does three jobs:

1. **Lifecycle** - start the enabled modules at launch, stop them at shutdown, and ``reconcile`` after
   Settings changed so a module that was just switched on starts and one switched off stops, with no restart.
2. **Offering tools** - ``schemas()`` is the list of tool descriptions sent with every request: the tools of
   the *enabled* modules only.
3. **Running tools** - ``run(call)`` finds the tool the model named and runs it.

It never mentions a specific module. Adding a module to the app means adding it to the list passed in.

Python ideas used here:

* ``logging`` - the standard library's way to record warnings and errors (``logger.warning(..., exc_info=True)``
  keeps the full traceback). One module failing to start is logged and must not stop the others.
* ``time.monotonic`` - a clock for measuring durations that never jumps backwards (unlike the wall clock).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

from lemonrind.lemonade.events import ToolCall
from lemonrind.modules.base import ChatContext, Module
from lemonrind.modules.tool import Tool, ToolResult

logger = logging.getLogger(__name__)

DEFAULT_MAX_OUTPUT_CHARS = 12_000


class ModuleRegistry:
    def __init__(
        self, modules: Iterable[Module], *, max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS
    ):
        self._modules = list(modules)
        self._started: set[str] = set()  # config keys of modules whose start-up has run
        self.max_output_chars = max_output_chars

    # --- the modules ---------------------------------------------------------------------------------

    @property
    def modules(self) -> list[Module]:
        """Every module, enabled or not (the Settings screen lists these)."""
        return list(self._modules)

    # --- lifecycle -------------------------------------------------------------------------------------

    async def start_enabled(self) -> None:
        await self.reconcile()

    async def stop_all(self) -> None:
        for module in self._modules:
            if module.config_key in self._started:
                await self._guard(module, "shut down", module.on_shutdown)
                self._started.discard(module.config_key)

    async def reconcile(self) -> None:
        """Make the running modules match the settings: start what was switched on, stop what was switched off."""
        for module in self._modules:
            running = module.config_key in self._started
            if module.enabled and not running:
                if await self._guard(module, "start", module.on_startup):
                    self._started.add(module.config_key)
            elif not module.enabled and running:
                await self._guard(module, "shut down", module.on_shutdown)
                self._started.discard(module.config_key)

    @staticmethod
    async def _guard(module: Module, action: str, hook) -> bool:
        try:
            await hook()
        except Exception:
            logger.warning("Module %s failed to %s", module.name, action, exc_info=True)
            return False
        return True

    # --- tools -------------------------------------------------------------------------------------------

    def get_enabled_tools(self) -> list[Tool]:
        return [tool for module in self._modules if module.enabled for tool in module.get_tools()]

    def schemas(self) -> list[dict]:
        """The tool descriptions to send with a request (empty when no module offers a tool)."""
        return [tool.schema() for tool in self.get_enabled_tools()]

    # --- context for the model ---------------------------------------------------------------------------
    #
    # These hooks are best-effort: a module that fails here is logged and skipped, because a broken
    # memory lookup must never stop you chatting.

    async def stable_context(self) -> str:
        """Text for the system prompt, from every enabled module that has some."""
        parts = []
        for module in self._modules:
            if module.enabled:
                parts.append(await self._safely(module, "stable_context", module.stable_context()))
        return "\n\n".join(part for part in parts if part)

    async def turn_context(self, user_text: str, *, chat: ChatContext) -> str:
        """Text for this one message, from every enabled module that has some."""
        parts = []
        for module in self._modules:
            if module.enabled:
                parts.append(
                    await self._safely(
                        module, "turn_context", module.turn_context(user_text, chat=chat)
                    )
                )
        return "\n\n".join(part for part in parts if part)

    async def after_turn(self, user_text: str, reply_text: str, *, model: str) -> None:
        for module in self._modules:
            if module.enabled:
                await self._safely(
                    module, "after_turn", module.after_turn(user_text, reply_text, model=model)
                )

    @staticmethod
    async def _safely(module: Module, hook_name: str, awaitable):
        try:
            return await awaitable
        except Exception:
            logger.warning("Module %s failed in %s", module.name, hook_name, exc_info=True)
            return ""

    def requires_approval(self, call: ToolCall) -> bool:
        """Must the user agree before this call runs? Unknown tools do not (they only produce an error)."""
        for tool in self.get_enabled_tools():
            if tool.name == call.name:
                return tool.needs_approval(call.arguments)
        return False

    def describe(self, call: ToolCall) -> str:
        """The text for a permission question about ``call`` (the tool's own preview, or its arguments)."""
        for tool in self.get_enabled_tools():
            if tool.name == call.name:
                return tool.describe(call.arguments)
        return call.arguments

    def module_for_tool(self, tool_name: str) -> Module | None:
        for module in self._modules:
            if module.enabled and any(tool.name == tool_name for tool in module.get_tools()):
                return module
        return None

    async def run(self, call: ToolCall) -> ToolResult:
        """Run the tool the model asked for. Unknown names and failures come back as error results."""
        tools = {tool.name: tool for tool in self.get_enabled_tools()}
        tool = tools.get(call.name)
        if tool is None:
            available = ", ".join(sorted(tools)) or "none"
            return ToolResult(
                f"There is no tool called '{call.name}'. Available tools: {available}.", True
            )
        result = await tool.run(call.arguments)
        return self._truncate(result)

    def _truncate(self, result: ToolResult) -> ToolResult:
        """Cap a huge result: it all goes into the model's limited context window."""
        if len(result.content) <= self.max_output_chars:
            return result
        omitted = len(result.content) - self.max_output_chars
        text = (
            result.content[: self.max_output_chars]
            + f"\n[... {omitted:,} more characters not shown]"
        )
        return ToolResult(text, result.is_error)
