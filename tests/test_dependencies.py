"""Every third-party package the code imports is declared in pyproject.toml (so a fresh install works)."""

from __future__ import annotations

import ast
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The name you import is not always the name you install.
INSTALL_NAME = {"PIL": "pillow", "docx": "python-docx", "httpx2": "httpx2"}


def declared() -> set[str]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    names = set()
    for requirement in data["project"]["dependencies"]:
        name = requirement.split(";")[0]
        for separator in "<>=!~[ ":
            name = name.split(separator)[0]
        names.add(name.strip().lower().replace("_", "-"))
    return names


def imported_third_party() -> dict[str, str]:
    """top-level module name -> a file that imports it, for everything that is neither the standard library nor ours."""
    found: dict[str, str] = {}
    for path in (ROOT / "src" / "lemonrind").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules = [node.module]
            else:
                continue
            for module in modules:
                top = module.split(".")[0]
                if top not in sys.stdlib_module_names and top != "lemonrind":
                    found.setdefault(top, path.name)
    return found


def test_every_package_the_code_imports_is_declared():
    names = declared()
    missing = {
        top: file
        for top, file in imported_third_party().items()
        if INSTALL_NAME.get(top, top).lower().replace("_", "-") not in names
    }

    assert missing == {}, f"imported but not declared in pyproject.toml: {missing}"


def test_the_declared_list_has_no_leftovers_from_removed_features():
    imported = {
        INSTALL_NAME.get(top, top).lower().replace("_", "-") for top in imported_third_party()
    }
    leftovers = (
        declared() - imported - {"tzdata"}
    )  # tzdata is used by zoneinfo, never imported by name

    assert leftovers == set(), f"declared but never imported: {leftovers}"
