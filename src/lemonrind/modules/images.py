"""The Images module: the assistant can draw, using an image model on your Lemonade Server.

Ask "draw me a lemon on a table" and the model calls ``generate_image`` with a prompt it writes itself. Because an
image model is big (it may push the chat model out of memory) and slow (minutes, not seconds), the tool **asks you
first**, showing the prompt and the size, using the permission mechanism from step 8. If you allow it, Lemonade
makes the picture, the PNG is saved in ``data/images/``, and the tool's answer tells the model to put a Markdown
image link in its reply, which the web page then shows inline.

How the picture gets from the file to the chat: the tool returns ``![description](/generated/img_....png)``. The web
app serves the folder ``data/images`` at the address ``/generated`` (see ``webui/app.py``), so that link is an
ordinary image link, and the chat's Markdown renderer displays it. It is stored in the chat as text, so reopening
the chat shows the same picture.

Python ideas used here:

* ``round`` and ``min``/``max`` to *clamp* a number into a range (image sizes must be sensible multiples).
* ``secrets.token_hex`` for a short unguessable-enough file name; ``datetime.strftime`` for a sortable one.
* Writing binary data with ``Path.write_bytes``, and ``base64`` to decode what the server sends.
"""

from __future__ import annotations

import dataclasses
import re
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any

from lemonrind.config import Settings
from lemonrind.lemonade.selection import pick_image_model
from lemonrind.modules.base import Module
from lemonrind.modules.tool import Tool, ToolError, tool_from_function

MIN_SIDE, STEP = 256, 64  # image models want sizes that divide evenly; 64 is safe for all of them
MAX_PROMPT_CHARS = 2000
URL_PREFIX = "/generated"  # where the web app serves the images folder


@dataclasses.dataclass(frozen=True, slots=True)
class DrawnImage:
    """A picture that has been made and saved."""

    name: str  # the file name inside data/images
    width: int
    height: int
    model: str
    markdown: str  # ``![description](/generated/name)``: what a reply contains to show the picture


def clamp_size(width: int, height: int, maximum: int) -> tuple[int, int]:
    """Round each side to a multiple of 64 and keep it between 256 and ``maximum``."""

    def fix(side: int) -> int:
        return max(MIN_SIDE, min(maximum, round(side / STEP) * STEP))

    return fix(width), fix(height)


class ImagesModule(Module):
    name = "Images"
    config_key = "images"
    description = "Draw pictures with an image model on Lemonade (asks you first)."

    def __init__(self, settings: Settings, data_dir: Path, client: Any) -> None:
        super().__init__(settings)
        self._client = client  # needs list_models() and generate_image()
        self.images_dir = data_dir / "images"

    async def on_startup(self) -> None:
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def get_tools(self) -> list[Tool]:
        return [
            dataclasses.replace(
                tool_from_function(self.generate_image),
                requires_approval=True,
                preview=self._describe,
            )
        ]

    def _size(self, data: dict) -> tuple[int, int]:
        config = self.settings.modules.images
        width = data.get("width") or config.width
        height = data.get("height") or config.height
        try:
            return clamp_size(int(width), int(height), config.max_side)
        except (TypeError, ValueError):
            return clamp_size(config.width, config.height, config.max_side)

    def _describe(self, data: dict) -> str:
        """The permission question: what will be drawn, how big, and why it is worth a second thought."""
        width, height = self._size(data)
        model = self.settings.modules.images.model or "the image model Lemonade has"
        return (
            f"Generate an image ({width} x {height}) with {model}\n"
            "(this can take a few minutes, and Lemonade may unload the chat model to make room)\n\n"
            f"Prompt: {str(data.get('prompt', ''))[:MAX_PROMPT_CHARS]}"
        )

    async def generate_image(
        self, prompt: str, width: int | None = None, height: int | None = None
    ) -> str:
        """Draw a picture from a text description using a local image model. The user is asked to confirm first, and it can take a few minutes. On success you get Markdown image syntax: include that exact Markdown, unchanged, in your reply so the user sees the picture.

        Args:
            prompt: A detailed description of the picture: subject, style, lighting, composition. Write it in English.
            width: Optional width in pixels (default from the settings; rounded to a multiple of 64).
            height: Optional height in pixels (default from the settings; rounded to a multiple of 64).
        """
        drawn = await self.draw(prompt, width=width, height=height)
        return (
            f"Image generated ({drawn.width} x {drawn.height}, {drawn.model}) and saved as {drawn.name}. Include this "
            f"exact Markdown in your reply so the user sees it: {drawn.markdown}"
        )

    async def draw(
        self,
        prompt: str,
        *,
        model: str | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> DrawnImage:
        """Make and save a picture. ``model`` is the image model to use; ``None`` picks the one in the settings.

        This is the part shared by the ``generate_image`` tool (the assistant asks for a picture) and the web page
        (you pick an image model in the model list and type a description). Raises ``ToolError`` for a bad request.
        """
        prompt = prompt.strip()
        if not prompt:
            raise ToolError("Give a description of the picture to draw.")
        if len(prompt) > MAX_PROMPT_CHARS:
            raise ToolError(
                f"The prompt is too long ({len(prompt):,} characters; the limit is {MAX_PROMPT_CHARS:,})."
            )

        config = self.settings.modules.images
        if model is None:
            model = pick_image_model(config.model, await self._client.list_models())
        if model is None:
            raise ToolError("Lemonade has no image model downloaded, so nothing can be drawn.")
        size = self._size({"width": width, "height": height})

        png = await self._client.generate_image(
            prompt,
            model,
            width=size[0],
            height=size[1],
            steps=config.steps,
            cfg_scale=config.cfg_scale,
            seed=None if config.seed in (None, -1) else config.seed,  # -1 = random
        )
        self.images_dir.mkdir(parents=True, exist_ok=True)
        name = f"img_{datetime.now():%Y%m%d_%H%M%S}_{secrets.token_hex(4)}.png"
        (self.images_dir / name).write_bytes(png)

        alt = re.sub(r"[\[\]\r\n]+", " ", prompt)[:80].strip()
        return DrawnImage(name, size[0], size[1], model, f"![{alt}]({URL_PREFIX}/{name})")
