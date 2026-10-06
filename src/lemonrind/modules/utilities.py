"""The Utilities module: the current time and a calculator.

Two small tools that show why tools exist at all: a language model cannot know what time it is, and it
is unreliable at arithmetic. Asking code is exact.

Python ideas used here:

* ``zoneinfo`` - the standard library's time zone database ("Europe/London").
* ``ast`` - parses Python source into a tree *without running it*. The calculator walks that tree and
  allows only numbers and arithmetic. The tempting shortcut, ``eval(expression)``, would let the model
  (or text it read on a web page) run any Python code on your computer. **Never ``eval`` text you do
  not control.**
* A dictionary of functions as a lookup table (``_FUNCTIONS``), and ``operator`` module functions
  (``operator.add``) so ``+`` can be stored in data.
"""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from lemonrind.modules.base import Module
from lemonrind.modules.tool import Tool, ToolError, tool_from_function

_BINARY: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY: dict[type, Callable[[Any], Any]] = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
}
_CONSTANTS = {"pi": math.pi, "e": math.e}
_MAX_EXPONENT = 1000  # 9**9**9 would otherwise keep the computer busy for a very long time
_MAX_LENGTH = 500


def evaluate(expression: str) -> float | int:
    """Evaluate an arithmetic expression safely. Raises ``ToolError`` for anything not allowed."""
    if len(expression) > _MAX_LENGTH:
        raise ToolError("The expression is too long.")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as error:
        raise ToolError(f"Not a valid expression: {error.msg}") from error
    try:
        return _evaluate_node(tree.body)
    except ZeroDivisionError as error:
        raise ToolError("Division by zero.") from error
    except (ValueError, OverflowError, TypeError) as error:
        raise ToolError(f"The calculation is not possible: {error}") from error


def _evaluate_node(node: ast.expr) -> float | int:
    match node:
        case ast.Constant(value=int() | float() as value) if type(value) in (int, float):
            return value  # (the second test excludes True and False, which are ints too)
        case ast.Name(id=name) if name in _CONSTANTS:
            return _CONSTANTS[name]
        case ast.UnaryOp(op=op, operand=operand) if type(op) in _UNARY:
            return _UNARY[type(op)](_evaluate_node(operand))
        case ast.BinOp(left=left, op=op, right=right) if type(op) in _BINARY:
            left_value, right_value = _evaluate_node(left), _evaluate_node(right)
            if isinstance(op, ast.Pow) and abs(right_value) > _MAX_EXPONENT:
                raise ToolError(f"The exponent is too large (the limit is {_MAX_EXPONENT}).")
            return _BINARY[type(op)](left_value, right_value)
        case ast.Call(func=ast.Name(id=name), args=args, keywords=[]) if name in _FUNCTIONS:
            return _FUNCTIONS[name](*(_evaluate_node(arg) for arg in args))
        case _:
            raise ToolError(
                "Only numbers, + - * / // % **, parentheses, pi, e and the functions "
                + ", ".join(_FUNCTIONS)
                + " are allowed."
            )


class UtilitiesModule(Module):
    name = "Utilities"
    config_key = "utilities"
    description = "The current date and time, and a calculator."

    def get_tools(self) -> list[Tool]:
        return [tool_from_function(self.current_time), tool_from_function(self.calculate)]

    def current_time(self, timezone: str = "") -> str:
        """Get the current date and time.

        Args:
            timezone: An IANA time zone name such as "Europe/London" or "America/New_York". Leave empty for this computer's own time zone.
        """
        try:
            moment = datetime.now(ZoneInfo(timezone)) if timezone else datetime.now().astimezone()
        except (ZoneInfoNotFoundError, ValueError) as error:
            raise ToolError(
                f"Unknown time zone '{timezone}'. Use a name like 'Europe/London'."
            ) from error
        return f"{moment:%A, %d %B %Y, %H:%M:%S} ({moment.tzname()}, UTC{moment:%z})"

    def calculate(self, expression: str) -> str:
        """Evaluate an arithmetic expression exactly. Use this instead of doing arithmetic yourself.

        Args:
            expression: For example "(12.5 * 4) / 3" or "sqrt(2) ** 2". Supports + - * / // % **, parentheses, pi, e, and sqrt, sin, cos, tan, log, log10, exp, abs, round, min, max.
        """
        return str(evaluate(expression))
