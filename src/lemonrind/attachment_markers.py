"""The text markers that record an attachment inside a saved message (see ``attachments.py`` for the whole picture).

Kept in a module of its own, with no heavy imports, because several places only need to *read* a marker (building a chat's
title, showing a stored message) and should not have to load the picture and document libraries to do so.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

IMAGE_PREFIX = "[Attached image: "
FILE_PREFIX = "[Attached file: "

FILE_END = "[End of attached file]"

_FILE_MARKER = re.compile(r"\[Attached file: ([^\]\r\n]+)\]\n\n")


def parse_image_marker(content: str) -> tuple[Path | None, str]:
    """``(image path, the typed text)`` if the message starts with an image marker, else ``(None, content)``."""
    if not content.startswith(IMAGE_PREFIX):
        return None, content
    end = content.find("]")
    if end < 0:
        return None, content
    return Path(content[len(IMAGE_PREFIX) : end]), content[end + 1 :].lstrip("\r\n")


@dataclass(frozen=True, slots=True)
class FileParts:
    name: str  # the document's file name
    text: str  # the text taken from it
    typed: str  # what the person typed alongside it


def parse_file_attachment(content: str) -> FileParts | None:
    """The three parts of a message that starts with a document marker, or ``None`` if it does not.

    The stored form is ``[Attached file: NAME]``, a blank line, the text, a blank line, ``[End of attached file]``, a blank
    line, then what was typed. (A message saved without the end line, by an older version, is shown as all document text.)
    """
    match = _FILE_MARKER.match(content) if content.startswith(FILE_PREFIX) else None
    if match is None:
        return None
    body = content[match.end() :]
    divider = f"\n\n{FILE_END}"
    if divider in body:
        text, _, typed = body.partition(divider)
        return FileParts(match.group(1), text, typed.lstrip("\r\n"))
    return FileParts(match.group(1), body, "")


def parse_file_marker(content: str) -> str | None:
    """The attached document's name if the message starts with a document marker."""
    parts = parse_file_attachment(content)
    return parts.name if parts else None


def title_source(content: str) -> str:
    """The part of a stored message to build a chat title from: what was typed, not the marker or the document."""
    path, rest = parse_image_marker(content)
    if path is not None:
        return rest
    parts = parse_file_attachment(content)
    if parts is not None:
        return parts.typed or parts.name
    return content
