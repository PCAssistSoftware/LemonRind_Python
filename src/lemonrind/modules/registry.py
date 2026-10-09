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
from collections import OrderedDict
from collections.abc import Iterable

from lemonrind.lemonade.events import ToolCall
from lemonrind.modules.base import ChatContext, Module
from lemonrind.modules.tool import Tool, ToolError, ToolResult, tool_from_function

logger = logging.getLogger(__name__)

DEFAULT_MAX_OUTPUT_CHARS = 40_000
READ_MORE = (
    "read_more"  # the name of the built-in tool that reads on through a result that was cut short
)
HELD_RESULTS = 20  # how many cut-short results are kept for read_more (the newest)


class ModuleRegistry:
    def __init__(
        self, modules: Iterable[Module], *, max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS
    ):
        self._modules = list(modules)
        self._started: set[str] = set()  # config keys of modules whose start-up has run
        self.max_output_chars = max_output_chars
        self._held: OrderedDict[str, str] = (
            OrderedDict()
        )  # result id -> its whole text, oldest first
        self._held_count = 0  # only ever grows, so an id is never reused
        self._read_more_tool = tool_from_function(self.read_more, name=READ_MORE)

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
        tools = [tool for module in self._modules if module.enabled for tool in module.get_tools()]
        if tools:  # reading on is only worth offering when there is some other tool whose result can be cut short
            tools.append(self._read_more_tool)
        return tools

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
        if tool.name == READ_MORE:
            return (
                result  # already one piece of the right size, with its own note about what is left
            )
        return self._truncate(result)

    # --- long results ---------------------------------------------------------------------------------------------

    def _truncate(self, result: ToolResult) -> ToolResult:
        """Give the model one piece of a long result, and keep the whole so it can ask for the rest.

        A long result goes into the model's limited context window, so only the first ``max_output_chars`` are sent.
        The rest is not thrown away: the whole text is held under a short id, and the note at the end of the piece tells
        the model how to read on with ``read_more``.
        """
        if len(result.content) <= self.max_output_chars:
            return result
        self._held_count += 1
        result_id = f"r{self._held_count}"
        self._held[result_id] = result.content
        while len(self._held) > HELD_RESULTS:
            self._held.popitem(last=False)  # forget the oldest
        return ToolResult(self._piece(result_id, result.content, 0), result.is_error)

    def _piece(self, result_id: str, content: str, start: int) -> str:
        """``max_output_chars`` of ``content`` from ``start``, with a note on how to continue if there is more."""
        end = start + self.max_output_chars
        if end < len(content):
            end = self._tidy_end(content, start, end)
        piece = content[start:end]
        if end >= len(content):
            return piece + (f"\n[End of result {result_id}.]" if start else "")
        return (
            piece
            + f"\n[... {len(content) - end:,} more characters not shown. To read on, call {READ_MORE} with "
            f'result="{result_id}" and start={end}.]'
        )

    @staticmethod
    def _tidy_end(content: str, start: int, end: int) -> int:
        """Move a piece's end back to the end of a line (or at least a word), so a piece never stops half way through
        a word and the next one never starts in the middle of one. Only the last tenth of the piece is searched, so a
        text with no breaks at all is simply cut at the limit."""
        earliest = end - max(1, (end - start) // 10)
        for separator in ("\n", " "):
            found = content.rfind(separator, earliest, end)
            if found != -1:
                return found + 1
        return end

    def read_more(self, result: str, start: int) -> str:
        """Read on through a tool result that was cut short. A result that is too long is shown only in part, and ends with a note giving the result id and the number to use here as start.

        Args:
            result: The id from the note at the end of the cut-short result, for example r3.
            start: The character number to continue from, as given in that note.
        """
        content = self._held.get(result.strip())
        if content is None:
            kept = ", ".join(self._held) or "none"
            raise ToolError(
                f"There is nothing saved as '{result}'. Only the last {HELD_RESULTS} long results are kept "
                f"(now: {kept}); run the tool again to get a fresh copy."
            )
        if not 0 <= start < len(content):
            raise ToolError(
                f"start must be between 0 and {len(content) - 1:,} for result {result}."
            )
        self._held.move_to_end(result.strip())  # still in use: keep it longer
        return self._piece(result.strip(), content, start)
