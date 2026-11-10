"""Pictures in a knowledge base: a vision model reads the picture, and its description is what gets searched.

A knowledge base can only search text, so a picture is turned into text first. The model that is **loaded right
now** is asked to write out any words in the picture and to describe what it shows; that description is chunked
and embedded like any other document, and it is also kept whole on the source so the person can read it (and ask
for it to be read again) from the knowledge base screen.

Which model? Whatever chat model Lemonade has loaded, provided it can see pictures (Lemonade labels those
``vision``). Nothing is loaded or swapped for the purpose: loading another model can push out the one a chat or a
scheduled job is using. If the loaded model cannot see pictures the person is told which ones can.

The description is the model's reading, not the picture itself: good for screenshots, scanned text, charts and
diagrams, weaker for a photograph with nothing to read.

Python ideas used here:

* ``tempfile.TemporaryDirectory`` as a context manager: the shrunk copy of the picture exists only inside the
  ``with`` block and is removed afterwards, even on an error.
* ``asyncio.to_thread`` for the picture shrinking (Pillow is blocking work).
"""

from __future__ import annotations

import asyncio
import base64
import tempfile
from pathlib import Path

from lemonrind.lemonade.client import LemonadeClient, LemonadeError

PICTURE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")

PICTURE_PROMPT = (
    "Describe this picture so that someone who cannot see it could find it by searching later. "
    "First write out, word for word, any text that appears in it. "
    "Then describe what it shows: the objects, people, layout, and any charts, tables or numbers. "
    "Be factual and specific, and do not guess at anything that is not visible."
)

MAX_DESCRIPTION_TOKENS = 4096  # a thinking model's thinking counts towards this, so it needs room


class PictureError(Exception):
    """A picture could not be read. The message is fit to show the person."""


def is_picture(path: Path) -> bool:
    return path.suffix.lower() in PICTURE_EXTENSIONS


async def loaded_vision_model(client: LemonadeClient) -> str:
    """The name of the loaded chat model that can see pictures, or ``PictureError`` saying what to do."""
    try:
        health = await client.health()
        models = await client.list_models()
    except LemonadeError as error:
        raise PictureError(f"Could not ask Lemonade which model is loaded: {error}") from error

    by_name = {model.id.lower(): model for model in models}
    loaded = [loaded.model_name for loaded in health.all_models_loaded]
    if health.model_loaded:
        loaded.append(health.model_loaded)
    chat_loaded = [
        name for name in loaded if (info := by_name.get(name.lower())) and info.category == "chat"
    ]
    for name in chat_loaded:
        if by_name[name.lower()].supports_vision:
            return name

    can_see = sorted(model.id for model in models if model.supports_vision)
    examples = f" Models that can: {', '.join(can_see[:5])}." if can_see else ""
    if chat_loaded:
        raise PictureError(
            f"The loaded model '{chat_loaded[0]}' cannot read pictures. Load one that can and add the picture "
            f"again.{examples}"
        )
    raise PictureError(
        f"No chat model is loaded. Load one that can read pictures and add the picture again.{examples}"
    )


async def shrink_picture(path: Path) -> bytes:
    """The picture as a JPEG of at most 1568 pixels on its longest side (what vision models read well), as bytes."""
    # Imported here, not at the top: attachments imports this package (for the list of readable text files).
    from lemonrind.attachments import AttachmentError, save_resized_image

    try:
        with tempfile.TemporaryDirectory() as folder:
            shrunk = await asyncio.to_thread(save_resized_image, path, Path(folder))
            return await asyncio.to_thread(shrunk.read_bytes)
    except AttachmentError as error:
        raise PictureError(str(error)) from error
    except OSError as error:
        raise PictureError(f"Could not read '{path.name}': {error}") from error


def data_url(image: bytes) -> str:
    """A picture as a ``data:`` web address, which is how it travels inside a chat request or a web page."""
    return "data:image/jpeg;base64," + base64.b64encode(image).decode("ascii")


async def describe_picture(client: LemonadeClient, image: bytes, name: str) -> tuple[str, str]:
    """Ask the loaded vision model to read the picture. Returns ``(description, model name)``."""
    model = await loaded_vision_model(client)
    part = {"type": "image_url", "image_url": {"url": data_url(image)}}
    try:
        text = await client.describe_image(
            model, part, PICTURE_PROMPT, max_tokens=MAX_DESCRIPTION_TOKENS
        )
    except LemonadeError as error:
        raise PictureError(str(error)) from error
    if not text.strip():
        raise PictureError(
            f"'{model}' did not write a description of '{name}' (a model that thinks may have used its whole "
            "allowance thinking). Try again, or load a model that thinks less."
        )
    return text.strip(), model
