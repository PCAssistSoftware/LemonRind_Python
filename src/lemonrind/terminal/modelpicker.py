"""Choosing, loading and switching the chat model.

Python ideas used here:

* ``asyncio.create_task`` - start the model load in the background while the foreground code keeps a
  spinner with a stopwatch running.
"""

from __future__ import annotations

import asyncio
import time

from rich.console import Console

from lemonrind.lemonade import LemonadeClient
from lemonrind.lemonade.client import LemonadeError
from lemonrind.lemonade.selection import ModelSelectionError, default_is_missing, pick_chat_model


async def choose_model(
    client: LemonadeClient, requested: str, console: Console, *, default: str = ""
) -> str | None:
    """Check Lemonade is reachable, pick a chat model, and load it if it is not loaded already."""
    try:
        health = await client.health()
        models = await client.list_models()
    except LemonadeError as error:
        console.print(f"[red]{error}[/red]")
        return None
    console.print(f"[green]Connected to Lemonade[/green] [dim]({client.base_url})[/dim]")

    try:
        picked = pick_chat_model(requested, health, models, default=default)
    except ModelSelectionError as error:
        console.print(f"[red]{error}[/red]")
        return None
    if not requested and default_is_missing(default, models):
        console.print(
            f"[yellow]Your default model '{default}' is no longer on this Lemonade, so {picked} is used instead. "
            "Change the default in Settings > Lemonade (web) or choose one with /model.[/yellow]"
        )
    requested = picked
    loaded = {m.model_name for m in health.all_models_loaded}
    if requested not in loaded:
        try:
            await load_with_status(client, requested, console)
        except LemonadeError as error:
            console.print(f"[red]{error}[/red]")
            console.print(
                "[dim]Pick another model with --model NAME (see /models once connected).[/dim]"
            )
            return None
    return requested


async def load_with_status(client: LemonadeClient, name: str, console: Console) -> None:
    """Load a model while showing a spinner with the real elapsed seconds.

    Lemonade's load call has no progress reporting: it simply returns when the model is ready. So this
    shows an honest stopwatch instead of a guessed percentage.
    """
    started = time.monotonic()
    with console.status(f"Loading {name}...") as status:
        # Start the load in the background, then wake once a second to refresh the stopwatch.
        load = asyncio.create_task(client.load_model(name))
        while not load.done():
            elapsed = int(time.monotonic() - started)
            status.update(f"Loading {name}... {elapsed}s (large models can take a while)")
            await asyncio.wait({load}, timeout=1)
        await load  # re-raises LemonadeError if loading failed
    console.print(f"[green]Loaded {name}[/green] in {time.monotonic() - started:.0f}s")


async def show_models(client: LemonadeClient, current: str, console: Console) -> None:
    models = [m for m in await client.list_models() if m.category == "chat"]
    for m in models:
        marker = "[green]*[/green]" if m.id == current else " "
        context = f"  [dim]{m.context_length:,} tokens of context[/dim]" if m.context_length else ""
        console.print(f" {marker} {m.id}{context}")


async def switch_model(client: LemonadeClient, current: str, wanted: str, console: Console) -> str:
    """Load ``wanted`` and return it; return ``current`` unchanged if that is not possible."""
    if not wanted:
        console.print("Usage: /model NAME   (see /models)")
        return current
    names = {m.id for m in await client.list_models() if m.category == "chat"}
    if wanted not in names:
        console.print(
            f"[red]'{wanted}' is not one of the downloaded chat models.[/red] Try /models."
        )
        return current
    try:
        await load_with_status(client, wanted, console)
    except LemonadeError as error:
        console.print(f"[red]{error}[/red]")
        return current
    return wanted
