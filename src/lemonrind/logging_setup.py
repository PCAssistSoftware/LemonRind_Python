"""Where warnings and errors go: a log file in the data folder.

Several parts of the app handle trouble *quietly* on purpose: a memory lookup that fails must not stop a chat, an MCP
server that cannot connect must not stop the app, a scheduled job that crashes must not stop the next one. "Quietly" must
not mean "invisibly", or nobody can find out why something does not work. These failures are written with Python's
``logging`` module, and this file decides where that output lands:

* **a log file**, ``data/logs/lemonrind.log``, always (it keeps three older copies of about 1 MB each, so it can never
  grow without limit);
* **the screen** only for the web server (its console is where you expect messages), or in the terminal chat when you
  pass ``--verbose``. In the terminal chat, log lines on screen would be mixed into the conversation.

Python ideas used here:

* The ``logging`` module's structure: *loggers* (named, one per module, created with ``logging.getLogger(__name__)``)
  produce records, *handlers* decide where they go (a file, the screen), *formatters* decide how a line looks.
  Configure once, at the root; every module's logger then inherits it.
* ``RotatingFileHandler``: starts a new file when the current one is full, deleting the oldest.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_FILE_NAME = "lemonrind.log"
FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_MARKER = "_lemonrind_handler"  # lets us recognise (and not duplicate) handlers we added ourselves


def log_path(data_dir: Path) -> Path:
    return data_dir / "logs" / LOG_FILE_NAME


class _QuietConnectionReset(logging.Filter):
    """Drop one harmless Windows message: asyncio logging "Exception in callback _call_connection_lost" with a
    ``ConnectionResetError``. It means a browser tab (or Lemonade) closed a connection abruptly while the app was closing
    its side of it; there is nothing to fix, and the traceback only alarms."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name == "asyncio" and record.exc_info:
            error = record.exc_info[1]
            if isinstance(error, ConnectionResetError | ConnectionAbortedError):
                return False
        return True


def clear_logs(data_dir: Path) -> None:
    """Empty the log file and delete its older rotated copies (``lemonrind.log.1`` ...).

    The file is emptied through the open logging handler where there is one, so the app keeps writing to it normally.
    """
    path = log_path(data_dir)
    for handler in logging.getLogger().handlers:
        if (
            isinstance(handler, RotatingFileHandler)
            and Path(handler.baseFilename) == path.resolve()
        ):
            handler.acquire()
            try:
                if handler.stream is not None:
                    handler.stream.seek(0)
                    handler.stream.truncate()
                    handler.stream.flush()
            finally:
                handler.release()
            break
    else:
        if path.exists():
            path.write_text("", encoding="utf-8")
    for older in path.parent.glob(LOG_FILE_NAME + ".*"):
        older.unlink(missing_ok=True)


def configure_logging(data_dir: Path, *, to_screen: bool, verbose: bool = False) -> Path:
    """Send the app's log to a file (and optionally the screen). Safe to call more than once. Returns the log file."""
    path = log_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    for handler in list(
        root.handlers
    ):  # remove what an earlier call added, so lines are never written twice
        if getattr(handler, _MARKER, False):
            root.removeHandler(handler)
            handler.close()

    level = logging.DEBUG if verbose else logging.INFO
    root.setLevel(level)
    formatter = logging.Formatter(FORMAT)

    file_handler = RotatingFileHandler(path, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.addFilter(_QuietConnectionReset())
    setattr(file_handler, _MARKER, True)
    root.addHandler(file_handler)

    if to_screen or verbose:
        screen = logging.StreamHandler()
        screen.setLevel(logging.DEBUG if verbose else logging.WARNING)
        screen.setFormatter(formatter)
        screen.addFilter(_QuietConnectionReset())
        setattr(screen, _MARKER, True)
        root.addHandler(screen)

    # Libraries that report every request at INFO would bury the useful lines.
    for noisy in ("httpx", "httpx2", "httpcore", "openai", "uvicorn.access", "watchfiles"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return path
