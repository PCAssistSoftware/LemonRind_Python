"""The ``lemonrind`` command: start-up, then hand over to the terminal chat.

The terminal chat is the first (and simplest) front end for the Lemon Rind core. It exists to prove the
core works before any web UI is built, and it stays useful afterwards as a quick way to test Lemonade from
a shell. Run it with ``lemonrind`` or ``python -m lemonrind``.

Python ideas used here:

* ``argparse`` - the standard library's command-line option parser.
* ``asyncio.run(main())`` - starts the async world; everything below it can ``await``.
* ``with`` / ``async with`` - open the database and the Lemonade client, and close both however we leave.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence

from rich.console import Console

from lemonrind import __version__
from lemonrind.chats import ChatRepository
from lemonrind.config import Settings, database_path, resolve_data_dir, settings_path
from lemonrind.lemonade import LemonadeClient
from lemonrind.logging_setup import configure_logging
from lemonrind.modules import build_registry
from lemonrind.storage import Database
from lemonrind.terminal.chatloop import ChatLoop
from lemonrind.terminal.modelpicker import choose_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lemonrind", description="Chat with a local Lemonade Server."
    )
    parser.add_argument("--base-url", help="Lemonade API address, e.g. http://localhost:13305/v1/")
    parser.add_argument(
        "--model", help="model to use (default: from settings, else the one loaded)"
    )
    parser.add_argument(
        "--data-dir",
        help="data folder (default: $LEMONRIND_DATA_DIR, else the project's data folder)",
    )
    parser.add_argument(
        "--new", action="store_true", help="start a new chat instead of reopening the latest one"
    )
    parser.add_argument("--verbose", action="store_true", help="show debug logging")
    parser.add_argument("--version", action="version", version=f"lemonrind {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: the ``lemonrind`` command and ``python -m lemonrind`` both end up here."""
    args = build_parser().parse_args(argv)
    # Make sure emoji and accents in replies print correctly even when output is piped on Windows.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\nBye.")
        return 130


async def run(args: argparse.Namespace) -> int:
    console = Console()

    data_dir = resolve_data_dir(args.data_dir)
    # Warnings go to a log file (quiet failures must still leave a trace); --verbose also shows them here.
    configure_logging(data_dir, to_screen=False, verbose=args.verbose)
    path = settings_path(data_dir)
    settings = Settings.load(path)
    if path.exists():
        console.print(f"[dim]Settings: {path}[/dim]")
    else:
        settings.save(path)
        console.print(f"[dim]Created {path} - edit it to point at your Lemonade Server.[/dim]")
    if args.base_url:
        settings.lemonade.base_url = args.base_url

    db_path = database_path(data_dir)
    console.print(f"[dim]Chats: {db_path}[/dim]")

    # Database has __enter__/__exit__, so ``with`` closes it however we leave the block.
    with Database(db_path) as db:
        async with LemonadeClient(settings.lemonade) as client:
            model = await choose_model(client, args.model or settings.lemonade.chat_model, console)
            if model is None:
                return 1
            # The modules (utilities, web search, files, ...) the model may use through tools.
            modules = build_registry(settings, data_dir, db=db, client=client)
            await modules.start_enabled()
            try:
                loop = ChatLoop(
                    client=client,
                    settings=settings,
                    settings_file=path,
                    repo=ChatRepository(db),
                    console=console,
                    model=model,
                    modules=modules,
                )
                await loop.run(resume=not args.new)
            finally:
                await loop.conversation.drain(30)  # let background learning finish before exiting
                await modules.stop_all()
    return 0
