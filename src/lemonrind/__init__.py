"""Lemon Rind: a local AI agent for Lemonade Server (Python edition).

A folder containing an ``__init__.py`` file is a Python *package*: other code can import from it
(``from lemonrind.config import Settings``). The file can be empty; it is the marker that says "this
folder is importable". The ``src/`` layout (code under ``src/lemonrind`` rather than directly in the
project root) stops tests from accidentally importing the source folder instead of the installed package.
"""

__version__ = "0.2.5"
