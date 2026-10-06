"""Syntax checks for C# and Visual Basic files, without the .NET compiler.

The .NET editions of this app use Roslyn (the compiler as a library) to check an edit. There is no Roslyn for Python, so
this uses **tree-sitter** instead: a parser library with ready-made grammars (``tree-sitter-c-sharp`` and
``tree-sitter-vb-dotnet``, both installed as ordinary wheels, no .NET needed). A tree-sitter parser never refuses
a file; it builds a syntax tree and marks the places it could not make sense of with ``ERROR`` (or "missing") nodes.
So "is there a syntax problem?" means "does the tree contain error nodes?".

How far to trust it differs by language, and that decides what the Coder does with the answer:

* **C#**: the grammar read every C# file of the sibling projects without a single false alarm, so a new problem
  *refuses* the edit, as for Python.
* **Visual Basic**: the grammar is younger and misses some newer constructs (checked against the sibling project, it
  misread about one file in five: multi-line string literals, tuple returns, some generic calls). A refusal could
  block a valid edit, so a VB problem is only a **warning** added to the tool's answer.

Either way what is compared is the number of error spots *before* and *after* the edit, not "zero": a file the grammar
already misreads is not blocked, and only an edit that makes things worse is flagged.

Python ideas used here:

* ``functools.cache`` to build each language object once, and importing inside a function so a missing optional
  parser only switches the check off instead of stopping the app from starting.
* Walking a tree with a small recursive function that only descends where there is something to find.
"""

from __future__ import annotations

import collections
import functools
import re
from pathlib import Path

KINDS = {".cs": "csharp", ".csx": "csharp", ".vb": "vb"}
REFUSING_KINDS = ("csharp",)  # a new problem in these refuses the edit; the others only warn
MAX_SPOTS_SHOWN = 3


def kind_of(path: Path) -> str | None:
    """``"csharp"`` or ``"vb"`` for a file of that language, else ``None``."""
    return KINDS.get(path.suffix.lower())


@functools.cache
def _language(kind: str):  # type: ignore[no-untyped-def]
    """The tree-sitter language for ``kind``, or ``None`` if its parser is not installed."""
    try:
        from tree_sitter import Language

        if kind == "csharp":
            import tree_sitter_c_sharp

            return Language(tree_sitter_c_sharp.language())
        import tree_sitter_vb_dotnet

        return Language(tree_sitter_vb_dotnet.language())
    except ImportError:
        return None


def error_lines(kind: str, text: str) -> list[int]:
    """The (1-based) line of every place the parser could not understand, in order; empty if the text is fine
    (or the parser is not available)."""
    language = _language(kind)
    if language is None:
        return []
    from tree_sitter import Parser

    tree = Parser(language).parse(text.encode("utf-8"))
    lines: list[int] = []

    def walk(node) -> None:  # type: ignore[no-untyped-def]
        if node.type == "ERROR" or node.is_missing:
            lines.append(node.start_point[0] + 1)
            return  # one report per broken region is enough
        if node.has_error:
            for child in node.children:
                walk(child)

    if tree.root_node.has_error:
        walk(tree.root_node)
    return lines


def new_problems(kind: str, before: str | None, after: str) -> list[int]:
    """The lines of syntax problems an edit *added*: ``[]`` unless ``after`` has more error spots than ``before``.

    ``before`` is ``None`` for a new file. (Line numbers refer to ``after``.)
    """
    found = error_lines(kind, after)
    already = len(error_lines(kind, before)) if before is not None else 0
    return found if len(found) > already else []


def describe(lines: list[int]) -> str:
    """``near line 12`` or ``near lines 12, 40 and 41`` (at most three shown)."""
    shown = [str(n) for n in dict.fromkeys(lines)][:MAX_SPOTS_SHOWN]
    if len(shown) == 1:
        return f"near line {shown[0]}"
    return "near lines " + ", ".join(shown[:-1]) + " and " + shown[-1]


# --- Visual Basic block balance ----------------------------------------------------------------------------------------
# The Visual Basic grammar is lenient about one thing that matters most when a model edits code: a block that is opened
# and never closed (an `If ... Then` without its `End If`, a `For` without `Next`). This second check reads the file
# very simply: it strips strings and comments, joins continued lines, looks at how each statement *starts*, and keeps a
# stack of the blocks that are open. At the end, whatever is still open is "unclosed", and an `End X` with nothing to
# close is "stray". Like the parser check it compares before and after an edit, so quirks of the simple reader
# (a multi-line lambda, say) do not matter unless the edit changes them.

_STRINGS_AND_COMMENTS = re.compile(
    r'("(?:[^"]|"")*")|(\'[^\n]*)|(^[ \t]*REM\b[^\n]*)', re.IGNORECASE | re.MULTILINE
)
_CONTINUATION = re.compile(r"[ \t]_[ \t]*\n")
_ATTRIBUTES = re.compile(r"^(?:<[^>]*>\s*)+")
_MODIFIERS = frozenset(
    [
        "public",
        "private",
        "protected",
        "friend",
        "shared",
        "shadows",
        "overloads",
        "overrides",
        "overridable",
        "notoverridable",
        "mustoverride",
        "mustinherit",
        "notinheritable",
        "readonly",
        "writeonly",
        "default",
        "partial",
        "async",
        "iterator",
        "static",
        "narrowing",
        "widening",
        "custom",
    ]
)
_BLOCKS_CLOSED_BY_END = frozenset(
    [
        "class",
        "module",
        "structure",
        "interface",
        "enum",
        "namespace",
        "sub",
        "function",
        "property",
        "operator",
        "while",
        "select",
        "try",
        "with",
        "using",
        "synclock",
        "if",
        "get",
        "set",
        "addhandler",
        "removehandler",
        "raiseevent",
        "event",
    ]
)
_ALWAYS_OPENERS = frozenset(
    [
        "class",
        "module",
        "structure",
        "interface",
        "enum",
        "namespace",
        "operator",
        "while",
        "try",
        "with",
        "using",
        "synclock",
        "do",
        "for",
    ]
)


_CONTINUING_ENDINGS = (",", "(", "&", "+", "-", "*", "/", "=", "<", ">", ".", "{")
_CONTINUING_WORDS = frozenset(
    {"andalso", "orelse", "and", "or", "xor", "mod", "is", "isnot", "like", "in"}
)


def _join_implicit_continuations(lines: list[str]) -> list[str]:
    """Join lines Visual Basic continues without an underscore: after an operator or comma, or before a leading ``.``."""
    joined: list[str] = []
    carry = ""
    for raw in lines:
        line = raw.strip()
        if carry:
            line = carry + " " + line
            carry = ""
        elif line.startswith(".") and joined:
            joined[-1] = joined[-1] + " " + line
            continue
        last_word = line.rsplit(None, 1)[-1].lower() if line else ""
        if line and (line.endswith(_CONTINUING_ENDINGS) or last_word in _CONTINUING_WORDS):
            carry = line
            continue
        joined.append(line)
    if carry:
        joined.append(carry)
    return joined


def _opens_lambda_block(words: list[str]) -> bool:
    """Is this statement a multi-line lambda being opened, such as ``Run(Sub(x)`` with its body on the lines below?"""
    text = " ".join(words)
    matches = list(re.finditer(r"\b(?:sub|function)\s*\(", text))
    if not matches or " end " in f" {text} ":
        return False
    match = matches[-1]  # the last lambda on the line is the one whose body can continue below
    depth, position = 1, match.end()
    while position < len(text) and depth:
        depth += {"(": 1, ")": -1}.get(text[position], 0)
        position += 1
    if depth:
        return False  # the parameter list does not close on this line
    rest = text[position:].strip()
    return rest == "" or rest.startswith("as ")  # "Async Function() As Task(Of String)"


def _statements(text: str) -> list[list[str]]:
    """Each statement as lowercase words, with strings, comments, attributes, modifiers and line continuations removed."""
    cleaned = _STRINGS_AND_COMMENTS.sub(lambda m: '""' if m.group(1) is not None else "", text)
    cleaned = _CONTINUATION.sub(" ", cleaned)
    result: list[list[str]] = []
    for raw in _join_implicit_continuations(cleaned.split("\n")):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = _ATTRIBUTES.sub("", line).strip()
        words = re.findall(r"[A-Za-z_][A-Za-z_0-9]*|\S", line)
        lowered = [w.lower() for w in words]
        start = 0
        while start < len(lowered) and lowered[start] in _MODIFIERS:
            start += 1
        if start < len(lowered):
            result.append(
                lowered[start:] + (["<must>"] if "mustoverride" in lowered[:start] else [])
            )
    return result


def vb_balance(text: str) -> tuple[collections.Counter[str], collections.Counter[str]]:
    """``(unclosed, stray)``: how many blocks of each kind were left open, and how many ``End X`` had nothing to close."""
    statements = _statements(text)
    stack: list[str] = []
    unclosed: collections.Counter[str] = collections.Counter()
    stray: collections.Counter[str] = collections.Counter()

    def close(kind: str, shown_as: str) -> None:
        """Close the innermost ``kind`` block; blocks opened inside it and never closed are reported as unclosed."""
        if kind not in stack:
            stray[shown_as] += 1
            return
        while stack[-1] != kind:
            unclosed[stack.pop()] += 1
        stack.pop()

    for index, words in enumerate(statements):
        first = words[0]
        in_interface = bool(stack) and stack[-1] == "interface"
        must = words[-1] == "<must>"
        if first == "end" and len(words) > 1 and words[1] in _BLOCKS_CLOSED_BY_END:
            close(words[1], words[1])
        elif first in ("loop", "next"):
            close("do" if first == "loop" else "for", first)
        elif first in _ALWAYS_OPENERS:
            stack.append(first)
        elif first == "select" and words[1:2] == ["case"]:
            stack.append("select")
        elif first == "if" and "then" in words and words[-1] == "then":
            stack.append("if")  # a block If; "If x Then y" on one line has something after Then
        elif (
            first in ("sub", "function")
            and len(words) > 1
            and words[1] != "("  # "Function(x) ..." is a lambda, not a declaration
            and not in_interface
            and not must
        ):
            stack.append(first)
        elif first == "property" and not in_interface and not must:
            following = statements[index + 1][0] if index + 1 < len(statements) else ""
            if following in (
                "get",
                "set",
                "readonly",
                "writeonly",
            ):  # a full property, not an auto-property
                stack.append("property")
        elif first in ("get", "set", "addhandler", "removehandler", "raiseevent") and stack[
            -1:
        ] == ["property"]:
            if len(words) == 1 or words[1] == "(":
                stack.append(first)
        elif first == "event" and stack[-1:] != ["interface"] and "custom" in words:
            stack.append("event")
        if first != "end" and _opens_lambda_block(words):
            stack.append("function" if "function" in words else "sub")
    unclosed.update(stack)
    return unclosed, stray


def vb_block_notes(before: str | None, after: str) -> list[str]:
    """Plain sentences about blocks an edit left open or closed twice (nothing if the edit made no difference)."""
    open_after, stray_after = vb_balance(after)
    open_before, stray_before = (
        vb_balance(before) if before is not None else (collections.Counter(), collections.Counter())
    )
    notes = []
    for kind, count in open_after.items():
        if count > open_before.get(kind, 0):
            closer = {"do": "Loop", "for": "Next"}.get(kind, "End " + kind.capitalize())
            notes.append(f"a '{kind.capitalize()}' block seems not to be closed ({closer} missing)")
    for kind, count in stray_after.items():
        if count > stray_before.get(kind, 0):
            closer = kind.capitalize() if kind in ("loop", "next") else "End " + kind.capitalize()
            notes.append(f"there is an extra '{closer}' with no matching opening line")
    return notes
