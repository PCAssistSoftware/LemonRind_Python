"""Start the web UI: ``lemonrind-web`` or ``python -m lemonrind.webui``.

Python ideas used here:

* ``app.on_shutdown`` - register a function NiceGUI calls when the server stops (to close the database).
* ``ui.run`` - starts the web server (uvicorn) and blocks until it stops.
"""

from __future__ import annotations

import argparse
import os
import shutil

# Security scanners flag any use of subprocess; here it only starts the browser (see open_browser).
import subprocess  # nosec B404
import sys
import webbrowser
from collections.abc import Sequence
from pathlib import Path

from nicegui import app, ui

from lemonrind import __version__
from lemonrind.attachments import ATTACHED_URL, attachments_dir
from lemonrind.config import resolve_data_dir
from lemonrind.logging_setup import configure_logging
from lemonrind.modules import ImagesModule
from lemonrind.modules.images import URL_PREFIX
from lemonrind.webui.auth import install_login, load_or_create_secret
from lemonrind.webui.context import AppContext, set_context
from lemonrind.webui.page import register_pages

# The name of an environment variable, not a password (hence the scanner exemption).
PASSWORD_ENV = "LEMONRIND_PASSWORD"  # nosec B105
LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lemonrind-web", description="Lemon Rind in your browser."
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="address to listen on (default: this computer only; 0.0.0.0 shares it with your network:"
        " set a password first, see --password)",
    )
    parser.add_argument(
        "--password",
        help="require this password to use the web UI (better: set the LEMONRIND_PASSWORD environment"
        " variable, because command-line arguments are visible to other programs)",
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--base-url", help="Lemonade API address, e.g. http://localhost:13305/v1/")
    parser.add_argument(
        "--data-dir",
        help="data folder (default: $LEMONRIND_DATA_DIR, else the project's data folder)",
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="do not open a browser tab on start-up"
    )
    parser.add_argument("--version", action="version", version=f"lemonrind {__version__}")
    return parser


def serve_images(context: AppContext) -> None:
    """Serve the generated pictures folder at ``/generated`` so a chat's Markdown image links work."""
    if context.modules is None:
        return
    module = next((m for m in context.modules.modules if isinstance(m, ImagesModule)), None)
    if module is not None:
        module.images_dir.mkdir(parents=True, exist_ok=True)  # the route needs the folder to exist
        app.add_static_files(URL_PREFIX, module.images_dir)


def serve_attachments(data_dir: Path) -> None:
    """Serve the folder of pictures attached to messages at ``/attached`` so a chat can show them again."""
    folder = attachments_dir(data_dir)
    folder.mkdir(parents=True, exist_ok=True)
    app.add_static_files(ATTACHED_URL, folder)


def page_url(host: str, port: int) -> str:
    """The address to open for a server listening on ``host``: a wildcard address means "this computer"."""
    # Only a comparison: nothing is bound to the wildcard address here (hence the scanner exemption).
    shown = "127.0.0.1" if host in ("", "0.0.0.0", "::") else host  # nosec B104
    return f"http://[{shown}]:{port}" if ":" in shown else f"http://{shown}:{port}"


def open_browser(url: str) -> None:
    """Open ``url`` in the default browser, keeping the browser's own console chatter out of this terminal.

    A browser started by Python inherits the terminal, and on Linux browsers print warnings there (D-Bus, GTK, push
    services) that look like errors but are not. ``xdg-open`` (or ``open`` on macOS) with its output thrown away
    avoids that; anywhere else the standard ``webbrowser`` module is used.
    """
    opener = {"linux": "xdg-open", "darwin": "open"}.get(sys.platform)
    if opener and shutil.which(opener):
        # A fixed program name and one URL, no shell involved (hence the scanner exemption).
        subprocess.Popen(  # nosec B603
            [opener, url],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return
    webbrowser.open(url)


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    data_dir = resolve_data_dir(args.data_dir)
    configure_logging(data_dir, to_screen=True)
    context = AppContext.create(data_dir, base_url=args.base_url)
    password = args.password or os.environ.get(PASSWORD_ENV, "")
    if password:
        install_login(password)
    elif args.host not in LOCAL_HOSTS:
        print(
            "WARNING: listening on the network with no password. Anyone who can reach this address can read your "
            f"chats and make the assistant run its tools. Set {PASSWORD_ENV} (or --password) first."
        )
    set_context(context)
    if context.modules is not None:
        app.on_startup(context.modules.start_enabled)  # modules start once the server is up
    app.on_shutdown(context.aclose)
    serve_images(context)
    serve_attachments(data_dir)
    register_pages()

    print(f"Settings: {context.settings_file}")
    url = page_url(args.host, args.port)
    print(f"Open {url}")
    if not args.no_browser:
        app.on_startup(
            lambda: open_browser(url)
        )  # once the server is up, so the page loads at the first try
    try:
        ui.run(
            title="Lemon Rind",
            host=args.host,
            port=args.port,
            favicon="🍋",
            reload=False,  # auto-reload on code changes needs extra set-up; off keeps start-up simple
            show=False,  # the browser is opened by open_browser above
            show_welcome_message=False,
            storage_secret=load_or_create_secret(data_dir),  # signs the login cookie
        )
    except KeyboardInterrupt:
        # Ctrl+C. The server has already shut down properly (the modules and the database are closed by the shutdown
        # hook); Python's asyncio re-raises the interrupt once that is done, which would print a long, alarming traceback.
        print("\nLemon Rind stopped.")
