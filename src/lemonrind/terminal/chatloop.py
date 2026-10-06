"""The interactive chat: reading what you type, running ``/commands``, and saving conversations.

Python ideas used here:

* A **class that holds state** (``ChatLoop``): the current chat, the history sent to the model, the last
  numbered list. Methods read and change that state through ``self``.
* ``textwrap.shorten`` - the standard library's "cut text to a length at a word boundary".
* ``rich.text.Text`` for printing user-supplied text safely. ``console.print("[red]")`` would treat
  square brackets in a chat message as styling instructions; a ``Text`` object never does.
* Default state for "a chat that does not exist yet": ``self.session is None`` means a draft. Nothing is
  written to the database until the first message is actually sent, so opening and closing the app never
  leaves empty chats behind.
"""

from __future__ import annotations

import asyncio
import textwrap
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

from pydantic import ValidationError
from rich.console import Console
from rich.text import Text

from lemonrind.attachments import Attachment, AttachmentError, attachments_dir, prepare
from lemonrind.chats import (
    DEFAULT_TITLE,
    ChatRepository,
    ChatSession,
    Conversation,
    StoredMessage,
    build_system_prompt,
    export_markdown,
    relative_time,
    safe_filename,
)
from lemonrind.config import Settings
from lemonrind.lemonade import LemonadeClient
from lemonrind.lemonade.client import LemonadeError
from lemonrind.modules import (
    KnowledgeModule,
    McpModule,
    MemoryModule,
    ModuleRegistry,
    SchedulerModule,
)
from lemonrind.modules.knowledge import OPTION_KEY as KNOWLEDGE_OPTION
from lemonrind.modules.knowledge import DuplicateNameError, KnowledgeFileError, copy_database
from lemonrind.modules.mcp import (
    DuplicateServerError,
    ServerImportError,
    from_command_line,
    parse_servers,
)
from lemonrind.modules.memory import Memory
from lemonrind.modules.scheduler import CronError, DuplicateJobError
from lemonrind.modules.scheduler.cron import format_local, upcoming
from lemonrind.terminal.modelpicker import show_models, switch_model
from lemonrind.terminal.streaming import stream_reply

HELP_TEXT = """[bold]Chats[/bold]
  [cyan]/chats[/cyan]            list your recent chats (numbered)
  [cyan]/search WORDS[/cyan]     find chats by title, folder, tag or message text
  [cyan]/open N[/cyan]           open chat number N from the last list
  [cyan]/new[/cyan]              start a fresh chat
  [cyan]/rename TITLE[/cyan]     rename this chat
  [cyan]/delete [N][/cyan]       delete this chat, or chat number N from the last list
  [cyan]/folder [NAME][/cyan]    put this chat in a folder (no name: take it out of its folder)
  [cyan]/tag NAME[/cyan]         tag this chat        [cyan]/untag NAME[/cyan]  remove a tag
[bold]Modules (what the model can do)[/bold]
  [cyan]/modules[/cyan]          list the modules and their tools, and whether each is on
  [cyan]/module KEY on|off[/cyan] switch a module on or off (takes effect immediately)
[bold]Memory[/bold]
  [cyan]/memories[/cyan]         list what the assistant remembers about you (numbered)
  [cyan]/remember TEXT[/cyan]    remember a fact       [cyan]/forget N[/cyan]  forget memory number N
  [cyan]/pin N[/cyan]            toggle whether memory N is always included
[bold]Knowledge bases (your documents)[/bold]
  [cyan]/kb[/cyan]               list knowledge bases (the one attached to this chat is marked)
  [cyan]/kb new NAME[/cyan]      create one          [cyan]/kb delete NAME[/cyan]  delete one
  [cyan]/kb use NAME|none[/cyan] attach one to this chat, or detach
  [cyan]/kb add PATH-OR-URL[/cyan]  add a file, a folder or a web page to the attached one
  [cyan]/kb note TEXT[/cyan]     add a pasted note      [cyan]/kb sources[/cyan]  list its sources
  [cyan]/kb remove N[/cyan]      remove source number N
  [cyan]/kb export PATH[/cyan]   save the attached one as a .kb file (to keep it, or use it on another computer)
  [cyan]/kb import PATH[/cyan]   add a knowledge base from a .kb file          [cyan]/kb reembed[/cyan]  rebuild it for the current embedding model
[bold]MCP servers (tools from other programs)[/bold]
  [cyan]/mcp[/cyan]              list servers, whether they are connected, and their tools
  [cyan]/mcp add NAME COMMAND [ARGS][/cyan]   add a server started by a command, e.g. npx -y some-server
  [cyan]/mcp import JSON|FILE[/cyan]  add servers from pasted MCP JSON (one line) or a .json file
  [cyan]/mcp on|off|restart|remove NAME[/cyan]   manage one server (changes apply immediately)
  [cyan]/mcp ask NAME on|off[/cyan]  ask before running its tools (on by default)
[bold]Scheduled jobs (prompts that run by themselves)[/bold]
  [cyan]/jobs[/cyan]             list jobs, when they run next and how the last run went
  [cyan]/jobs add NAME | CRON | PROMPT[/cyan]   e.g. /jobs add Digest | 0 9 * * 1 | Summarise the news
  [cyan]/jobs run|on|off|remove NAME[/cyan]    run a job now (waits), switch it, or delete it
  [cyan]/jobs allow NAME on|off[/cyan]  let it run tools that normally ask permission, unattended
[bold]Attachments (pictures and documents)[/bold]
  [cyan]/attach PATH[/cyan]      attach a picture (needs a vision model) or a document to your next message
  [cyan]/detach[/cyan]           remove the attachment
[bold]Export[/bold]
  [cyan]/export [FILE][/cyan]    save this chat as a Markdown file (default: the data folder's exports folder)
[bold]Persona[/bold]
  [cyan]/persona[/cyan]          show the persona (who the assistant is, how it writes)
  [cyan]/persona FIELD VALUE[/cyan]  set identity, about, tone, length, emoji or instructions (empty value clears it)
  [cyan]/prompt[/cyan]           show the full system prompt the model is sent
[bold]Models[/bold]
  [cyan]/models[/cyan]           list the chat models Lemonade has downloaded
  [cyan]/model NAME[/cyan]       switch to another model (loads it if needed)
[bold]Other[/bold]
  [cyan]/help[/cyan]             show this help
  [cyan]/quit[/cyan]             leave (Ctrl+C also works)"""

NO_VISION_MESSAGE = (
    "The current model does not support image input. Switch to a vision model with /model first "
    "(the picture was not attached, or was kept for later)."
)
LIST_LIMIT = 20  # how many chats /chats and /search show
RECAP_MESSAGES = 4  # how many recent messages are shown when a chat is opened
RECAP_WIDTH = 300  # characters per recap message


class ChatLoop:
    def __init__(
        self,
        *,
        client: LemonadeClient,
        settings: Settings,
        settings_file: Path | None = None,
        repo: ChatRepository,
        console: Console,
        model: str,
        modules: ModuleRegistry | None = None,
    ) -> None:
        self.client = client
        self.settings = settings
        self.settings_file = settings_file
        self.modules = modules
        self.repo = repo
        self.console = console
        # The shared "send, stream, save" logic (also used by the web UI) lives in Conversation.
        self.conversation = Conversation(
            client=client,
            repo=repo,
            system_prompt=build_system_prompt(settings),
            model=model,
            tools=modules,
            context=modules,
            max_rounds=settings.modules.max_tool_rounds,
            compaction_percent=settings.assistant.compaction_percent,
            compaction_keep_turns=settings.assistant.compaction_keep_turns,
        )
        self.conversation.time_awareness = True  # the model is always told the real date and time
        self._attachment: Attachment | None = None  # waiting to go with the next message
        self._listing: list[ChatSession] = []  # the numbered list from the last /chats or /search
        self._memory_listing: list[Memory] = []  # the numbered list from the last /memories

    @property
    def model(self) -> str:
        return self.conversation.model

    @property
    def session(self) -> ChatSession | None:
        return self.conversation.session

    # --- the main loop -------------------------------------------------------------------------------

    async def run(self, *, resume: bool = True) -> None:
        """Chat until the user quits. With ``resume`` the most recent saved chat is opened first."""
        self.console.print(
            f"Using [bold]{self.model}[/bold]. Type [cyan]/help[/cyan] for commands.\n"
        )
        await self._learn_context_window()
        if resume and (latest := self.repo.list_sessions(limit=1)):
            self._open(latest[0])

        while True:
            try:
                # input() blocks, so run it on a worker thread to keep the event loop free.
                text = (
                    await asyncio.to_thread(self.console.input, "[bold green]You:[/bold green] ")
                ).strip()
            except EOFError:  # Ctrl+Z / Ctrl+D or the end of piped input
                return
            if not text:
                continue
            if text.startswith("/"):
                if not await self._command(text):
                    return
            else:
                await self._send(text)

    async def _send(self, text: str) -> None:
        """Send one message and show the streamed reply (the conversation saves it)."""
        was_default_title = self.session is None or self.session.title == DEFAULT_TITLE
        if (
            self._attachment is not None
            and self._attachment.is_image
            and not await self._model_sees()
        ):
            self.console.print(
                Text(NO_VISION_MESSAGE, style="yellow")
            )  # the model was switched after attaching
            return
        attachment, self._attachment = (
            self._attachment,
            None,
        )  # one attachment goes with one message
        reply = await stream_reply(self.conversation, text, self.console, attachment=attachment)
        if reply is None and attachment is not None:
            self._attachment = (
                attachment  # the message did not go: keep the attachment for the next try
            )
        if reply and reply.text and was_default_title and self.session:
            self.console.print(Text(f"Saved as: {self.session.title}", style="dim"))

    async def _learn_context_window(self) -> None:
        """Find out how much the current model can read at once, so old messages can be summarised in time."""
        try:
            models = await self.client.list_models()
        except LemonadeError:
            return  # no limit known: compaction simply does not trigger
        for model in models:
            if model.id == self.model:
                self.conversation.context_window = model.context_length
                return

    # --- commands ---------------------------------------------------------------------------------------

    async def _command(self, text: str) -> bool:
        """Run a ``/command``. Returns ``False`` when the user asked to quit."""
        command, _, argument = text.partition(" ")
        argument = argument.strip()
        match command.lower():
            case "/quit" | "/exit":
                return False
            case "/help":
                self.console.print(HELP_TEXT)
            case "/new":
                self._new()
            case "/chats":
                self._show_listing(self.repo.list_sessions(limit=LIST_LIMIT))
            case "/search":
                if argument:
                    self._show_listing(self.repo.search(argument)[:LIST_LIMIT])
                else:
                    self.console.print("Usage: /search WORDS")
            case "/open":
                if session := self._pick(argument):
                    self._open(session)
            case "/rename":
                self._rename(argument)
            case "/delete":
                await self._delete(argument)
            case "/folder":
                self._folder(argument)
            case "/tag":
                self._tag(argument, add=True)
            case "/untag":
                self._tag(argument, add=False)
            case "/modules":
                self._show_modules()
            case "/module":
                await self._switch_module(argument)
            case "/export":
                self._export(argument)
            case "/attach":
                await self._attach(argument)
            case "/detach":
                self._attachment = None
                self.console.print(Text("Attachment removed.", style="dim"))
            case "/persona":
                await self._persona(argument)
            case "/prompt":
                self.console.print(Text(await self.conversation.current_system_prompt()))
            case "/jobs":
                await self._jobs(argument)
            case "/mcp":
                await self._mcp(argument)
            case "/kb":
                await self._kb(argument)
            case "/memories":
                self._show_memories()
            case "/remember":
                await self._remember(argument)
            case "/forget":
                self._forget(argument)
            case "/pin":
                self._pin(argument)
            case "/models":
                await show_models(self.client, self.model, self.console)
            case "/model":
                self.conversation.model = await switch_model(
                    self.client, self.model, argument, self.console
                )
                await self._learn_context_window()
            case _:
                self.console.print(f"[red]Unknown command {command}.[/red] Try /help.")
        return True

    def _show_modules(self) -> None:
        if self.modules is None:
            self.console.print("[dim]No modules are loaded.[/dim]")
            return
        for module in self.modules.modules:
            state = "[green]on [/green]" if module.enabled else "[dim]off[/dim]"
            tools = ", ".join(tool.name for tool in module.get_tools())
            self.console.print(
                Text.from_markup(f" {state} ")
                + Text.assemble((f"{module.config_key:<12}", "bold"), f" {module.description}")
            )
            self.console.print(Text(f"        tools: {tools}", style="dim"))
        self.console.print("[dim]Use /module KEY on|off to change one.[/dim]")

    async def _switch_module(self, argument: str) -> None:
        key, _, state = argument.partition(" ")
        known = {m.config_key: m for m in self.modules.modules} if self.modules else {}
        if key not in known or state.strip().lower() not in {"on", "off"}:
            self.console.print("Usage: /module KEY on|off   (see /modules for the keys)")
            return
        assert self.modules is not None
        self.settings.modules.enabled[key] = state.strip().lower() == "on"
        if self.settings_file is not None:
            self.settings.save(self.settings_file)
        await self.modules.reconcile()  # starts or stops the module right now
        self.console.print(Text(f"{known[key].name} is now {state.strip().lower()}.", style="dim"))

    # --- attachments -----------------------------------------------------------------------------------------

    async def _attach(self, argument: str) -> None:
        if not argument:
            self.console.print(
                "Usage: /attach PATH   (a picture, PDF, Word, Excel, text or Markdown file)"
            )
            return
        source = Path(argument.strip('"')).expanduser()
        if not source.is_file():
            self.console.print(Text(f"No such file: {source}", style="red"))
            return
        folder = attachments_dir(self.settings_file.parent if self.settings_file else Path("data"))
        try:
            attachment = await asyncio.to_thread(prepare, source, folder)
        except AttachmentError as error:
            self.console.print(Text(str(error), style="red"))
            return
        if attachment.is_image and not await self._model_sees():
            self.console.print(Text(NO_VISION_MESSAGE, style="yellow"))
            return
        self._attachment = attachment
        self.console.print(
            Text(f"Attached {attachment.name}. It goes with your next message.", style="dim")
        )

    async def _model_sees(self) -> bool:
        """Does the current model accept pictures? (Lemonade labels such models ``vision``.)"""
        try:
            models = await self.client.list_models()
        except LemonadeError:
            return True  # cannot tell: let the server decide
        return any(m.id == self.model and m.supports_vision for m in models)

    # --- export ----------------------------------------------------------------------------------------------

    def _export(self, argument: str) -> None:
        if (session := self._require_saved()) is None:
            return
        text = export_markdown(session, self.repo.list_messages(session.id))
        if argument:
            target = Path(argument.strip('"')).expanduser()
        else:
            folder = self.settings_file.parent if self.settings_file else Path("data")
            target = folder / "exports" / safe_filename(session.title)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        self.console.print(Text(f"Saved to {target}", style="dim"))

    # --- persona ---------------------------------------------------------------------------------------------

    PERSONA_FIELDS: ClassVar[dict[str, str]] = {
        "identity": "identity",
        "about": "about_user",
        "tone": "tone",
        "length": "verbosity",
        "emoji": "emoji_usage",
        "instructions": "custom_instructions",
    }

    async def _persona(self, argument: str) -> None:
        persona = self.settings.persona
        if argument:
            name, _, value = argument.partition(" ")
            field = self.PERSONA_FIELDS.get(name.lower())
            if field is None:
                self.console.print(
                    f"Usage: /persona FIELD VALUE   (fields: {', '.join(self.PERSONA_FIELDS)})"
                )
                return
            try:
                setattr(
                    persona,
                    field,
                    value.strip()
                    or ("Default" if field in {"tone", "verbosity", "emoji_usage"} else ""),
                )
            except ValidationError as error:
                allowed = error.errors()[0].get("ctx", {}).get("expected", "a different value")
                self.console.print(f"[red]That is not one of the allowed values: {allowed}[/red]")
                return
            if self.settings_file is not None:
                self.settings.save(self.settings_file)
            self.conversation.set_system_prompt(build_system_prompt(self.settings))
        for label, field in self.PERSONA_FIELDS.items():
            shown = getattr(persona, field) or "(not set)"
            self.console.print(Text.assemble((f"{label:<13}", "bold"), str(shown)))

    # --- scheduled jobs ---------------------------------------------------------------------------------

    def _scheduler(self) -> SchedulerModule | None:
        found = None
        if self.modules is not None:
            found = next((m for m in self.modules.modules if isinstance(m, SchedulerModule)), None)
        if found is None or not found.enabled:
            self.console.print("[dim]The Scheduler module is off. Use /module scheduler on.[/dim]")
            return None
        return found

    async def _jobs(self, argument: str) -> None:
        if (module := self._scheduler()) is None:
            return
        action, _, rest = argument.partition(" ")
        action, rest = action.lower(), rest.strip()
        if action == "":
            self._show_jobs(module)
        elif action == "add":
            parts = [part.strip() for part in rest.split("|")]
            if len(parts) != 3:
                self.console.print(
                    "Usage: /jobs add NAME | CRON | PROMPT   (separated by | characters)"
                )
                return
            try:
                job = module.add_job(*parts)
            except (CronError, DuplicateJobError, ValueError) as error:
                self.console.print(f"[red]{error}[/red]")
                return
            times = ", ".join(format_local(t) for t in upcoming(job.cron, datetime.now(UTC), 3))
            self.console.print(Text(f"Added '{job.name}'. Next runs: {times}", style="dim"))
        elif action in {"run", "on", "off", "remove", "allow"}:
            name, _, option = rest.partition(" ")
            found_job = module.repo.find(name)
            if found_job is None:
                self.console.print("[red]No job with that name. See /jobs.[/red]")
                return
            await self._job_action(module, found_job, action, option.strip().lower())
        else:
            self.console.print("Usage: /jobs [add|run|on|off|remove|allow] ...  (see /help)")

    async def _job_action(self, module: SchedulerModule, job, action: str, option: str) -> None:
        if action == "run":
            with self.console.status(f"Running '{job.name}' (this can take minutes)..."):
                outcome = await module.run_now(job.id)
            status = outcome.status if outcome else "gone"
            self.console.print(
                Text(
                    f"Finished: {status}. The result is a chat tagged 'scheduled' (/chats).",
                    style="dim",
                )
            )
        elif action in {"on", "off"}:
            module.set_enabled(job.id, action == "on")
            self.console.print(Text(f"'{job.name}' is now {action}.", style="dim"))
        elif action == "remove":
            module.delete_job(job.id)
            self.console.print(Text(f"Removed '{job.name}'.", style="dim"))
        elif option in {"on", "off"}:  # allow NAME on|off
            module.edit_job(
                job.id,
                name=job.name,
                schedule=job.cron,
                prompt=job.prompt,
                model=job.model,
                allow_unattended_tools=option == "on",
            )
            self.console.print(Text(f"Unattended tools for '{job.name}': {option}.", style="dim"))
        else:
            self.console.print("Usage: /jobs allow NAME on|off")

    def _show_jobs(self, module: SchedulerModule) -> None:
        jobs = module.repo.list()
        if not jobs:
            self.console.print(
                "[dim]No jobs yet. /jobs add NAME | CRON | PROMPT creates one.[/dim]"
            )
        for job in jobs:
            colour = {"ok": "green", "failed": "red", "timed out": "red"}.get(
                job.last_status, "yellow"
            )
            self.console.print(
                Text.assemble(
                    (job.name, "bold"),
                    "  ",
                    (job.cron, "cyan"),
                    (f"  next: {format_local(job.next_run_at) if job.enabled else 'off'}", "dim"),
                    "  ",
                    (job.last_status or "never run", colour),
                    (
                        "  [may run tools unattended]" if job.allow_unattended_tools else "",
                        "yellow",
                    ),
                )
            )

    # --- MCP servers -----------------------------------------------------------------------------------

    def _mcp_module(self) -> McpModule | None:
        found = None
        if self.modules is not None:
            found = next((m for m in self.modules.modules if isinstance(m, McpModule)), None)
        if found is None or not found.enabled:
            self.console.print("[dim]The MCP servers module is off. Use /module mcp on.[/dim]")
            return None
        return found

    async def _mcp(self, argument: str) -> None:
        if (module := self._mcp_module()) is None:
            return
        action, _, rest = argument.partition(" ")
        rest = rest.strip()
        action = action.lower()

        if action == "":
            self._show_mcp(module)
        elif action == "add":
            name, _, command_line = rest.partition(" ")
            try:
                definition = from_command_line(name, command_line)
                await module.add_servers([definition])
            except (ServerImportError, DuplicateServerError) as error:
                self.console.print(f"[red]{error}[/red]")
                return
            await self._mcp_connect_and_show(module)
        elif action == "import":
            text = rest
            if rest.lower().endswith(".json") and Path(rest.strip('"')).expanduser().is_file():
                text = Path(rest.strip('"')).expanduser().read_text(encoding="utf-8")
            try:
                await module.add_servers(parse_servers(text))
            except (ServerImportError, DuplicateServerError) as error:
                self.console.print(f"[red]{error}[/red]")
                return
            await self._mcp_connect_and_show(module)
        elif action in {"on", "off", "restart", "remove", "ask"}:
            await self._mcp_manage(module, action, rest)
        else:
            self.console.print(
                "Usage: /mcp [add|import|on|off|restart|remove|ask] ...  (see /help)"
            )

    async def _mcp_manage(self, module: McpModule, action: str, rest: str) -> None:
        name, _, option = rest.partition(" ")
        config = module.repo.find(name)
        if config is None:
            self.console.print("[red]No MCP server with that name. See /mcp.[/red]")
            return
        if action == "on" or action == "off":
            await module.set_enabled(config.id, action == "on")
        elif action == "restart":
            await module.restart(config.id)
        elif action == "remove":
            await module.remove(config.id)
            self.console.print(Text(f"Removed '{config.name}'.", style="dim"))
            return
        elif option.strip().lower() in {"on", "off"}:  # ask NAME on|off
            module.set_require_approval(config.id, option.strip().lower() == "on")
        else:
            self.console.print("Usage: /mcp ask NAME on|off")
            return
        await self._mcp_connect_and_show(module)

    async def _mcp_connect_and_show(self, module: McpModule) -> None:
        with self.console.status("Connecting (the first run of an npx server downloads it)..."):
            await module.wait_settled(self.settings.modules.mcp.connect_timeout_seconds + 5)
        self._show_mcp(module)

    def _show_mcp(self, module: McpModule) -> None:
        statuses = module.statuses()
        if not statuses:
            self.console.print(
                "[dim]No MCP servers yet. /mcp add NAME COMMAND ... or /mcp import JSON adds one.[/dim]"
            )
        for item in statuses:
            colour = {"connected": "green", "failed": "red", "connecting": "yellow"}.get(
                item.status, "dim"
            )
            state = item.status if item.config.enabled else "off"
            ask = "asks first" if item.config.require_approval else "runs freely"
            self.console.print(
                Text.assemble(
                    (f"{item.config.name}", "bold"),
                    "  ",
                    (state, colour),
                    (f"  {ask}  {item.config.summary}", "dim"),
                )
            )
            if item.error:
                self.console.print(Text(f"    {item.error}", style="red"))
            elif item.tool_names:
                self.console.print(Text("    tools: " + ", ".join(item.tool_names), style="dim"))

    # --- knowledge bases -------------------------------------------------------------------------------

    def _knowledge(self) -> KnowledgeModule | None:
        found = None
        if self.modules is not None:
            found = next((m for m in self.modules.modules if isinstance(m, KnowledgeModule)), None)
        if found is None or not found.enabled:
            self.console.print(
                "[dim]The Knowledge bases module is off. Use /module knowledge on.[/dim]"
            )
            return None
        return found

    async def _kb(self, argument: str) -> None:
        if (module := self._knowledge()) is None:
            return
        action, _, rest = argument.partition(" ")
        rest = rest.strip()
        repo = module.repo
        attached = self.conversation.options.get(KNOWLEDGE_OPTION)

        match action.lower():
            case "":
                kbs = repo.list_kbs()
                if not kbs:
                    self.console.print(
                        "[dim]No knowledge bases yet. /kb new NAME creates one.[/dim]"
                    )
                for kb in kbs:
                    sources = repo.list_sources(kb.id)
                    marker = "*" if kb.id == attached else " "
                    model = f", model {kb.embedding_model}" if kb.embedding_model else ""
                    self.console.print(
                        Text.assemble(
                            f" {marker} ",
                            (kb.name, "bold"),
                            (f"  {len(sources)} sources{model}", "dim"),
                        )
                    )
            case "new":
                try:
                    kb = repo.create_kb(rest)
                except (ValueError, DuplicateNameError) as error:
                    self.console.print(f"[red]{error}[/red]")
                    return
                self.conversation.set_option(KNOWLEDGE_OPTION, kb.id)
                self.console.print(
                    Text(f"Created '{kb.name}' and attached it to this chat.", style="dim")
                )
            case "use":
                if rest.lower() == "none":
                    self.conversation.set_option(KNOWLEDGE_OPTION, None)
                    self.console.print(Text("Detached.", style="dim"))
                elif found := repo.find_kb(rest):
                    self.conversation.set_option(KNOWLEDGE_OPTION, found.id)
                    self.console.print(Text(f"Attached '{found.name}'.", style="dim"))
                else:
                    self.console.print("[red]No knowledge base with that name.[/red]")
            case "delete":
                if doomed := repo.find_kb(rest):
                    repo.delete_kb(doomed.id)
                    if doomed.id == attached:
                        self.conversation.set_option(KNOWLEDGE_OPTION, None)
                    self.console.print(Text(f"Deleted '{doomed.name}'.", style="dim"))
                else:
                    self.console.print("[red]No knowledge base with that name.[/red]")
            case "add" | "note":
                await self._kb_add(module, action.lower(), rest)
            case "sources":
                if kb := self._attached_kb(module):
                    self._show_sources(module, kb.id)
            case "export":
                if kb := self._attached_kb(module):
                    await self._kb_export(module, kb, rest)
            case "import":
                await self._kb_import(module, rest)
            case "reembed":
                if kb := self._attached_kb(module):
                    await self._kb_reembed(module, kb)
            case "remove":
                if kb := self._attached_kb(module):
                    sources = repo.list_sources(kb.id)
                    if rest.isdigit() and 1 <= int(rest) <= len(sources):
                        repo.delete_source(sources[int(rest) - 1].id)
                        self.console.print(Text("Removed.", style="dim"))
                    else:
                        self.console.print("Give a source number from [cyan]/kb sources[/cyan].")
            case _:
                self.console.print(
                    "Usage: /kb [new|use|delete|add|note|sources|remove|export|import|reembed] ...  (see /help)"
                )

    def _attached_kb(self, module: KnowledgeModule):
        kb_id = self.conversation.options.get(KNOWLEDGE_OPTION)
        kb = module.repo.get_kb(kb_id) if kb_id else None
        if kb is None:
            self.console.print(
                "[dim]No knowledge base is attached. Use /kb new NAME or /kb use NAME.[/dim]"
            )
        return kb

    def _show_sources(self, module: KnowledgeModule, kb_id: str) -> None:
        sources = module.repo.list_sources(kb_id)
        if not sources:
            self.console.print("[dim]No sources yet. Use /kb add PATH-OR-URL.[/dim]")
        for number, source in enumerate(sources, start=1):
            detail = (
                f"{source.chunk_count} chunks" if source.status == "ready" else (source.error or "")
            )
            colour = {"ready": "green", "failed": "red"}.get(source.status, "yellow")
            self.console.print(
                Text.assemble(
                    f"{number:>3}. ",
                    (source.source_type.ljust(7), "dim"),
                    source.display_name,
                    "  ",
                    (f"{source.status}: {detail}", colour),
                )
            )

    async def _kb_add(self, module: KnowledgeModule, action: str, rest: str) -> None:
        if (kb := self._attached_kb(module)) is None:
            return
        if not rest:
            self.console.print("Usage: /kb add PATH-OR-URL   or   /kb note TEXT")
            return
        ingestor = module.ingestor
        with self.console.status("Reading and embedding (large documents take a while)..."):
            if action == "note":
                await ingestor.add_text(kb.id, rest[:40], rest)
            elif rest.lower().startswith(("http://", "https://")):
                await ingestor.add_website(kb.id, rest)
            elif (path := Path(rest.strip('"')).expanduser()).is_dir():
                await ingestor.add_folder(kb.id, path)
            elif path.is_file():
                await ingestor.add_file(kb.id, path)
            else:
                self.console.print(f"[red]Not a file, folder or web address: {rest}[/red]")
                return
        self._show_sources(module, kb.id)

    async def _kb_export(self, module: KnowledgeModule, kb, rest: str) -> None:
        if not rest:
            self.console.print("Usage: /kb export PATH   (a file name ending in .kb, or a folder)")
            return
        target = Path(rest.strip('"')).expanduser()
        if target.is_dir():
            target = target / f"{kb.name}.kb"
        await asyncio.to_thread(copy_database, kb.path, target)
        self.console.print(Text(f"Saved '{kb.name}' to {target}", style="dim"))

    async def _kb_import(self, module: KnowledgeModule, rest: str) -> None:
        if not rest:
            self.console.print("Usage: /kb import PATH-TO-A-.kb-FILE")
            return
        path = Path(rest.strip('"')).expanduser()
        if not path.is_file():
            self.console.print(f"[red]No such file: {path}[/red]")
            return
        try:
            plan = module.repo.plan_import(path)
            await asyncio.to_thread(copy_database, path, plan.destination)
            kb = module.repo.finish_import(plan)
        except KnowledgeFileError as error:
            self.console.print(f"[red]{error}[/red]")
            return
        self.conversation.set_option(KNOWLEDGE_OPTION, kb.id)
        self.console.print(Text(f"Added '{kb.name}' and attached it to this chat.", style="dim"))
        if problem := await module.compatibility(kb):
            self.console.print(f"[yellow]{problem}[/yellow]")

    async def _kb_reembed(self, module: KnowledgeModule, kb) -> None:
        try:
            with self.console.status("Embedding every chunk again (this takes a while)..."):
                count = await module.reembed(kb.id)
        except LemonadeError as error:
            self.console.print(f"[red]Could not re-embed: {error}[/red]")
            return
        self.console.print(Text(f"Re-embedded {count:,} chunks of '{kb.name}'.", style="dim"))

    # --- memory ------------------------------------------------------------------------------------------

    def _memory(self) -> MemoryModule | None:
        """The memory module, if it exists and is switched on (else say so)."""
        found = None
        if self.modules is not None:
            found = next((m for m in self.modules.modules if isinstance(m, MemoryModule)), None)
        if found is None or not found.enabled:
            self.console.print("[dim]The Memory module is off. Use /module memory on.[/dim]")
            return None
        return found

    def _show_memories(self) -> None:
        if (memory := self._memory()) is None:
            return
        self._memory_listing = memory.repo.list_all()
        if not self._memory_listing:
            self.console.print("[dim]Nothing is remembered yet.[/dim]")
            return
        for number, item in enumerate(self._memory_listing, start=1):
            kind = ("pinned", "magenta") if item.pinned else ("      ", "dim")
            note = "" if item.has_embedding else "  (no embedding yet)"
            self.console.print(
                Text.assemble(f"{number:>3}. ", kind, "  ", item.content, (note, "dim"))
            )
        self.console.print("[dim]/forget N removes one, /pin N toggles pinned.[/dim]")

    def _pick_memory(self, argument: str) -> Memory | None:
        if not argument.isdigit() or not 1 <= int(argument) <= len(self._memory_listing):
            self.console.print("Give a number from the last [cyan]/memories[/cyan] list.")
            return None
        return self._memory_listing[int(argument) - 1]

    async def _remember(self, text: str) -> None:
        if not text:
            self.console.print("Usage: /remember TEXT")
        elif memory := self._memory():
            saved = await memory.save_fact(text, pinned=False)
            self.console.print(
                Text(
                    "Remembered." if saved else "Already remembered something very similar.",
                    style="dim",
                )
            )

    def _forget(self, argument: str) -> None:
        if (memory := self._memory()) and (item := self._pick_memory(argument)):
            memory.repo.delete(item.id)
            self._memory_listing = [m for m in self._memory_listing if m.id != item.id]
            self.console.print(Text("Forgotten.", style="dim"))

    def _pin(self, argument: str) -> None:
        if (memory := self._memory()) and (item := self._pick_memory(argument)):
            memory.repo.set_pinned(item.id, not item.pinned)
            self.console.print(
                Text("Pinned." if not item.pinned else "No longer pinned.", style="dim")
            )
            self._show_memories()

    def _new(self) -> None:
        self.conversation.new()
        self.console.print(
            "[dim]Started a new chat (it is saved when you send the first message).[/dim]"
        )

    def _open(self, session: ChatSession) -> None:
        """Make ``session`` the current chat and show where the conversation was."""
        messages = self.conversation.open(session)
        self.console.print(
            Text(f"Opened: {session.title} ({len(messages)} messages)", style="bold")
        )
        self._recap(messages[-RECAP_MESSAGES:])

    def _recap(self, messages: list[StoredMessage]) -> None:
        """Print the last few messages so you can see where the conversation was."""
        for message in messages:
            if message.role == "tool":
                continue  # tool results are long and for the model; the answer that follows covers them
            who, style = ("You", "green") if message.role == "user" else ("Assistant", "cyan")
            content = message.content or (
                "(used " + ", ".join(c.name for c in message.tool_calls) + ")"
                if message.tool_calls
                else ""
            )
            shortened = textwrap.shorten(content, width=RECAP_WIDTH, placeholder=" ...")
            self.console.print(Text.assemble((f"{who}: ", f"bold {style}"), (shortened, "dim")))
        if messages:
            self.console.print()

    def _show_listing(self, sessions: list[ChatSession]) -> None:
        self._listing = sessions
        if not sessions:
            self.console.print("[dim]No chats found.[/dim]")
            return
        for number, chat in enumerate(sessions, start=1):
            current = "*" if self.session and chat.id == self.session.id else " "
            details = [relative_time(chat.updated_at)]
            if chat.folder_name:
                details.append(f"in {chat.folder_name}")
            details.extend(f"#{tag}" for tag in chat.tags)
            self.console.print(
                Text.assemble(
                    f"{current}{number:>3}. ",
                    (chat.title, "bold"),
                    ("   " + "  ".join(details), "dim"),
                )
            )
        self.console.print("[dim]Use /open N to open one.[/dim]")

    def _pick(self, argument: str) -> ChatSession | None:
        """Turn "3" into the third chat of the last list."""
        if not argument.isdigit() or not 1 <= int(argument) <= len(self._listing):
            self.console.print(
                "Give a number from the last [cyan]/chats[/cyan] or [cyan]/search[/cyan] list."
            )
            return None
        return self._listing[int(argument) - 1]

    def _require_saved(self) -> ChatSession | None:
        if self.session is None:
            self.console.print("[dim]This chat is not saved yet: send a message first.[/dim]")
        return self.session

    def _rename(self, title: str) -> None:
        if not title:
            self.console.print("Usage: /rename NEW TITLE")
        elif session := self._require_saved():
            self.repo.rename_session(session.id, title)
            self.conversation.refresh_session()
            self.console.print(Text(f"Renamed to: {title}", style="dim"))

    async def _delete(self, argument: str) -> None:
        target = self._pick(argument) if argument else self._require_saved()
        if target is None:
            return
        answer = await asyncio.to_thread(
            self.console.input, f"Delete '{target.title}' and all its messages? [y/N] "
        )
        if answer.strip().lower() not in {"y", "yes"}:
            self.console.print("[dim]Kept.[/dim]")
            return
        self.repo.delete_session(target.id)
        self._listing = [s for s in self._listing if s.id != target.id]
        self.console.print("[dim]Deleted.[/dim]")
        if self.session and self.session.id == target.id:
            self._new()

    def _folder(self, name: str) -> None:
        if session := self._require_saved():
            folder = self.repo.get_or_create_folder(name) if name else None
            self.repo.move_to_folder(session.id, folder.id if folder else None)
            self.conversation.refresh_session()
            self.console.print(
                Text(
                    f"Filed in: {folder.name}" if folder else "Taken out of its folder.",
                    style="dim",
                )
            )

    def _tag(self, name: str, *, add: bool) -> None:
        if not name:
            self.console.print(f"Usage: /{'tag' if add else 'untag'} NAME")
        elif session := self._require_saved():
            (self.repo.add_tag if add else self.repo.remove_tag)(session.id, name)
            self.conversation.refresh_session()
            tags = ", ".join(self.session.tags) if self.session and self.session.tags else "(none)"
            self.console.print(Text(f"Tags: {tags}", style="dim"))
