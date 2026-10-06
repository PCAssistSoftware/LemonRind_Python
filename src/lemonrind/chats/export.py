"""Exporting a chat as a Markdown document you can keep, print, or paste somewhere.

A chat is stored as rows in a database, which is no good as something to hand to a person. This turns one into a plain
Markdown file: the title, the date, the tags, then the conversation with each message under a heading.

What is included and how:

* your messages and the assistant's replies, in order, as ``## You`` and ``## Assistant``;
* the model's **thinking**, folded into a collapsible ``<details>`` block (readers who do not want it can skip it);
* **tool calls** as a short quoted line (what was asked of which tool), and the tool's result folded and *shortened*
  (a web page or a file can be thousands of lines);
* the **summary** of a long chat's older part, if it has one, as a note at the end.

It is a pure function (a chat in, text out), which is what makes it easy to test; the screens only choose where the text
goes (a download in the browser, a file from the terminal).

Python ideas used here:

* Building a ``list`` of lines and joining it once, the standard way to assemble text.
* ``re.sub`` to turn a title into a safe file name.
* ``datetime.astimezone()`` to show UTC times in your own time zone.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path

from lemonrind.attachment_markers import parse_file_attachment, parse_image_marker
from lemonrind.chats.models import ChatSession, StoredMessage

MAX_RESULT_CHARS = 1500  # a tool's result is cut to this many characters in the export
MAX_ARGUMENT_CHARS = 300


def safe_filename(title: str, *, extension: str = ".md") -> str:
    """``'What is 6 x 7?'`` becomes ``'What-is-6-x-7.md'``: letters, digits, dashes only, never empty, not too long."""
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", title).strip("-")[:60].strip("-")
    return (cleaned or "chat") + extension


def _fence(text: str) -> str:
    """Wrap text in a code block, using a longer fence than any run of backticks inside it."""
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}\n{text}\n{fence}"


def _shorten(text: str, limit: int) -> str:
    return (
        text
        if len(text) <= limit
        else text[:limit] + f"\n[... {len(text) - limit:,} more characters]"
    )


def _image_link(path: Path) -> str:
    """A Markdown image pointing at the stored copy, so a viewer that opens local files shows the picture inline."""
    try:
        address = path.as_uri()
    except ValueError:  # a relative path has no file address
        address = path.as_posix()
    return f"![attached image]({address})"


def _user_lines(content: str) -> list[str]:
    """Your message as Markdown lines: an attached picture becomes an image link, a document a collapsed block."""
    image, typed = parse_image_marker(content)
    if image is not None:
        return [_image_link(image), "", typed, ""]
    if (parts := parse_file_attachment(content)) is not None:
        return [
            f"*Attached file: {parts.name}*",
            "",
            f"<details><summary>Text of {parts.name}</summary>",
            "",
            _fence(_shorten(parts.text, MAX_RESULT_CHARS)),
            "",
            "</details>",
            "",
            parts.typed,
            "",
        ]
    return [content, ""]


def export_markdown(session: ChatSession, messages: Sequence[StoredMessage]) -> str:
    """The chat as a Markdown document."""
    lines = [f"# {session.title}", ""]
    created = session.created_at.astimezone().strftime("%A %d %B %Y, %H:%M")
    details = [f"Started {created}"]
    if session.folder_name:
        details.append(f"folder: {session.folder_name}")
    if session.tags:
        details.append("tags: " + ", ".join(f"#{tag}" for tag in session.tags))
    lines += ["*" + " | ".join(details) + "*", ""]

    call_names: dict[str, str] = {}  # tool call id -> tool name, to label the results
    for message in messages:
        if message.role == "user":
            lines += ["## You", ""]
            lines += _user_lines(message.content)
        elif message.role == "assistant":
            lines += ["## Assistant", ""]
            if message.reasoning:
                lines += [
                    "<details><summary>Thinking</summary>",
                    "",
                    message.reasoning,
                    "",
                    "</details>",
                    "",
                ]
            for call in message.tool_calls:
                call_names[call.id] = call.name
                try:
                    shown = json.dumps(json.loads(call.arguments), ensure_ascii=False)
                except json.JSONDecodeError:
                    shown = call.arguments
                lines += [
                    f"> Used the tool **{call.name}** with `{_shorten(shown, MAX_ARGUMENT_CHARS)}`",
                    "",
                ]
            if message.content:
                lines += [message.content, ""]
        elif message.role == "tool":
            name = call_names.get(message.tool_call_id or "", "tool")
            body = _fence(_shorten(message.content, MAX_RESULT_CHARS))
            lines += [
                f"<details><summary>Result of {name}</summary>",
                "",
                body,
                "",
                "</details>",
                "",
            ]

    if session.summary:
        lines += [
            "---",
            "",
            "*Earlier messages in this chat were summarised to save space. The summary the assistant was given "
            "instead of them:*",
            "",
            session.summary,
            "",
        ]
    return "\n".join(lines).rstrip() + "\n"
