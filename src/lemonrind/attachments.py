"""Attaching a picture or a document to a message.

* **A picture** goes to a model that can see (one labelled ``vision`` by Lemonade). It is first shrunk to at most 1568
  pixels on its longest side and saved as a JPEG in the data folder: past about that size vision models stop finding more
  detail, so a bigger picture only costs time and context. The small copy is what is stored with the chat, so the chat
  still shows the picture later even if the original moves.
* **A document** (PDF, Word, Excel, text, Markdown) is turned into text and placed in the message, capped at 50,000
  characters so one attachment cannot swamp the model's whole context window.

How an attachment lives in the saved chat (the same scheme as the .NET editions, so the idea carries across): the message
text is stored with a **marker** at the front, because the database column holds text and an image cannot live in it.

    [Attached image: C:\\...\\attachments\\img_20261003_101500_ab12cd34.jpg]

    what is in this picture?

    [Attached file: report.pdf]

    <the extracted text>

    [End of attached file]

    summarise this

When a chat is reopened the marker is read back: an image marker becomes a real image part again (if the file still exists),
a file marker is simply text. Only the **newest** image in a chat is sent to the model; earlier ones go as their text alone,
because a picture costs thousands of tokens and the model rarely needs the old ones again.

Python ideas used here:

* **Pillow** (``PIL``): ``Image.open``, ``ImageOps.exif_transpose`` (honour a phone's rotation flag), ``resize``, ``save``.
* ``base64`` and a **data URL** (``data:image/jpeg;base64,...``): how an image travels inside a JSON chat request.
* A small frozen ``dataclass`` for "what is attached", so the chat code takes one object rather than three loose values.
"""

from __future__ import annotations

import base64
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

from lemonrind.attachment_markers import (
    FILE_END,
    FILE_PREFIX,
    IMAGE_PREFIX,
)
from lemonrind.modules.knowledge.extract import SUPPORTED_EXTENSIONS as TEXT_EXTENSIONS
from lemonrind.modules.knowledge.extract import ExtractionError, extract_text

ATTACHMENTS_FOLDER = "attachments"  # inside the data folder
ATTACHED_URL = "/attached"  # where the web app serves that folder
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp")
MAX_LONGEST_EDGE = 1568
JPEG_QUALITY = 85
MAX_FILE_CHARACTERS = 50_000

ACCEPT = ",".join((*IMAGE_EXTENSIONS, *TEXT_EXTENSIONS))  # for the file picker's ``accept`` filter


class AttachmentError(Exception):
    """The attachment cannot be used. The message says why and is fit to show."""


@dataclass(frozen=True, slots=True)
class Attachment:
    """What is attached to the next message. ``path`` is the saved small copy of a picture; ``text`` is a document's text."""

    kind: str  # "image" or "file"
    name: str  # the file name to show
    path: Path | None = None  # an image's stored copy
    text: str = ""  # a document's extracted text

    @property
    def is_image(self) -> bool:
        return self.kind == "image"


def attachments_dir(data_dir: Path) -> Path:
    return data_dir / ATTACHMENTS_FOLDER


def attached_url(path: Path) -> str:
    """The web address of a stored picture (the folder is served at ``/attached``)."""
    return f"{ATTACHED_URL}/{path.name}"


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def supported_text(path: Path) -> bool:
    return path.suffix.lower() in TEXT_EXTENSIONS


def save_resized_image(source: Path, folder: Path, *, now: datetime | None = None) -> Path:
    """Shrink the picture to a JPEG of at most ``MAX_LONGEST_EDGE`` pixels and save it in ``folder``; return its path."""
    try:
        with Image.open(source) as opened:
            picture = ImageOps.exif_transpose(
                opened
            )  # a phone photo's "rotate me" flag, applied for real
            picture.load()
    except (UnidentifiedImageError, OSError) as error:
        raise AttachmentError(f"'{source.name}' could not be read as a picture: {error}") from error

    if picture.mode in ("RGBA", "LA", "P"):  # JPEG has no transparency: put the picture on white
        picture = picture.convert("RGBA")
        background = Image.new("RGB", picture.size, (255, 255, 255))
        background.paste(picture, mask=picture.getchannel("A"))
        picture = background
    elif picture.mode != "RGB":
        picture = picture.convert("RGB")

    longest = max(picture.size)
    if longest > MAX_LONGEST_EDGE:
        scale = MAX_LONGEST_EDGE / longest
        picture = picture.resize(
            (max(1, round(picture.width * scale)), max(1, round(picture.height * scale))),
            Image.Resampling.LANCZOS,
        )

    folder.mkdir(parents=True, exist_ok=True)
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%d_%H%M%S")
    target = folder / f"img_{stamp}_{uuid.uuid4().hex[:8]}.jpg"
    picture.save(target, "JPEG", quality=JPEG_QUALITY)
    return target


def read_document(path: Path) -> str:
    """The text of a document, capped. Raises ``AttachmentError`` with a readable message (a scanned PDF says so)."""
    try:
        text = extract_text(path)
    except ExtractionError as error:
        raise AttachmentError(str(error)) from error
    if len(text) > MAX_FILE_CHARACTERS:
        text = (
            text[:MAX_FILE_CHARACTERS]
            + f"\n[... truncated - the file is longer than the {MAX_FILE_CHARACTERS:,}-character limit for a one-off attachment ...]"
        )
    return text


def prepare(source: Path, images_folder: Path) -> Attachment:
    """Turn a chosen file into an ``Attachment``: a picture is shrunk and saved, a document is read."""
    if is_image(source):
        return Attachment("image", source.name, path=save_resized_image(source, images_folder))
    if supported_text(source):
        return Attachment("file", source.name, text=read_document(source))
    supported = ", ".join((*IMAGE_EXTENSIONS, *TEXT_EXTENSIONS))
    raise AttachmentError(
        f"'{source.name}' is not a type that can be attached. Supported: {supported}"
    )


# --- how an attachment is written into, and read back out of, the saved message ----------------------------------


def stored_text(text: str, attachment: Attachment | None) -> str:
    """The message as saved: the typed text, with the attachment's marker (and a document's text) in front."""
    if attachment is None:
        return text
    if attachment.is_image:
        return f"{IMAGE_PREFIX}{attachment.path}]\n\n{text}"
    return f"{FILE_PREFIX}{attachment.name}]\n\n{attachment.text}\n\n{FILE_END}\n\n{text}"


def image_part(path: Path) -> dict[str, Any]:
    """The picture as a chat-request part: a data URL holding the JPEG bytes in base64."""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}


def wire_content(text: str, attachment: Attachment | None) -> str | list[dict[str, Any]]:
    """The message content to send to the model: a plain string, or text plus a picture for an image attachment."""
    if attachment is not None and attachment.is_image and attachment.path is not None:
        return [{"type": "text", "text": text}, image_part(attachment.path)]
    return stored_text(text, attachment)  # a document's text goes inline


def has_image(message: dict[str, Any]) -> bool:
    content = message.get("content")
    return isinstance(content, list) and any(
        isinstance(part, dict) and part.get("type") == "image_url" for part in content
    )


def text_of(message: dict[str, Any]) -> str:
    """A message's text, whether its content is a string or a list of parts."""
    content = message.get("content")
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return content or ""
