"""Every part of the app must be importable on its own, in a fresh interpreter.

Python runs a module's imports the first time anything imports it, so a *circular* import (A needs B, B needs
A) only fails when A happens to be imported first. Tests that import several things in a convenient order can
miss it, while the real entry point (which imports in its own order) hits it. Importing each module in its own
fresh process, as a user would, checks every order.
"""

import subprocess
import sys

import pytest

ENTRY_POINTS = [
    "lemonrind.cli",
    "lemonrind.webui.app",
    "lemonrind.chats",
    "lemonrind.modules",
    "lemonrind.modules.scheduler",
    "lemonrind.modules.mcp",
    "lemonrind.modules.knowledge",
    "lemonrind.modules.memory",
    "lemonrind.lemonade",
    "lemonrind.terminal.chatloop",
    "lemonrind.webui.page",
]


@pytest.mark.parametrize("module", ENTRY_POINTS)
def test_each_module_imports_cleanly_on_its_own(module: str):
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"], capture_output=True, text=True, timeout=120
    )

    assert result.returncode == 0, result.stderr[-800:]
