"""Drawing straight from the model list: pick an image model, describe a picture, choose the size, get the picture."""

from __future__ import annotations

import asyncio

from nicegui.testing import User

from lemonrind.lemonade.models import ModelInfo
from lemonrind.modules import ImagesModule
from lemonrind.webui.context import AppContext
from tests.conftest import elements
from tests.test_images import PNG
from tests.test_settings_sections import settle
from tests.test_webui import FakeLemonade


class DrawingLemonade(FakeLemonade):
    """A Lemonade with a chat model and an image model, that can make a (fake) picture."""

    def __init__(self) -> None:
        super().__init__()
        self.drawn: list[dict] = []

    async def list_models(self, *, downloaded_only: bool = True) -> list[ModelInfo]:
        return [
            ModelInfo(id="Fake-Chat", labels=["chat"], downloaded=True, context_length=8000),
            ModelInfo(id="Draw-It", labels=["image"], downloaded=True),
        ]

    async def generate_image(self, prompt: str, model: str, **options) -> bytes:
        self.drawn.append({"prompt": prompt, "model": model, **options})
        return PNG


def images_module(web: AppContext, lemonade: DrawingLemonade) -> ImagesModule:
    module = next(m for m in web.modules.modules if isinstance(m, ImagesModule))  # type: ignore[union-attr]
    module._client = lemonade  # the page's own fake knows about image models
    return module


async def pick(user: User, model: str) -> None:
    (select,) = elements(user, "model-select")
    select.value = model
    await settle()


async def open_page(user: User, web: AppContext, lemonade: DrawingLemonade) -> ImagesModule:
    web.client = lemonade  # type: ignore[assignment]
    module = images_module(web, lemonade)
    await user.open("/")
    await user.should_see("Lemonade healthy")
    await settle()
    return module


async def test_an_image_model_picked_in_the_list_draws_what_you_type_and_saves_it(
    user: User, web: AppContext, tmp_path
):
    lemonade = DrawingLemonade()
    module = await open_page(user, web, lemonade)
    module.images_dir.mkdir(parents=True, exist_ok=True)

    await pick(user, "Draw-It")
    user.find(marker="message-input").type("a lemon on a table")
    user.find(marker="send").click()
    await user.should_see("Draw a picture")  # the size question comes first
    await user.should_see("Model: Draw-It")
    user.find(marker="dialog-ok").click()

    await user.should_see(
        marker="save-image"
    )  # the picture is drawn, with its Save and Copy buttons
    await user.should_see(marker="copy-image")
    assert lemonade.requests == []  # no chat request: the text was a picture description
    (drawing,) = lemonade.drawn
    assert drawing["model"] == "Draw-It" and (drawing["width"], drawing["height"]) == (512, 512)
    assert [p.name for p in module.images_dir.glob("*.png")] != []

    (session,) = web.repo.list_sessions()
    user_message, assistant = web.repo.list_messages(session.id)
    assert user_message.content == "a lemon on a table"
    assert assistant.content.startswith("![a lemon on a table](/generated/img_")
    assert "image" in session.tags  # the chat is tagged like any other chat that drew something


async def test_the_size_asked_for_is_the_size_drawn(user: User, web: AppContext):
    lemonade = DrawingLemonade()
    await open_page(user, web, lemonade)
    await pick(user, "Draw-It")
    user.find(marker="message-input").type("wide")
    user.find(marker="send").click()
    await user.should_see("Draw a picture")
    (width,) = elements(user, "image-width")
    width.value = 1024
    user.find(marker="dialog-ok").click()
    await user.should_see(marker="save-image")

    assert (lemonade.drawn[0]["width"], lemonade.drawn[0]["height"]) == (1024, 512)


async def test_cancelling_the_size_question_draws_nothing_and_keeps_the_text(
    user: User, web: AppContext
):
    lemonade = DrawingLemonade()
    await open_page(user, web, lemonade)
    await pick(user, "Draw-It")
    user.find(marker="message-input").type("never mind")
    user.find(marker="send").click()
    await user.should_see("Draw a picture")
    user.find(marker="dialog-cancel").click()
    await asyncio.sleep(0.2)

    assert lemonade.drawn == [] and web.repo.list_sessions() == []
    (box,) = elements(user, "message-input")
    assert box.value == "never mind"


async def test_picking_a_chat_model_again_goes_back_to_chatting(user: User, web: AppContext):
    lemonade = DrawingLemonade()
    await open_page(user, web, lemonade)
    await pick(user, "Draw-It")
    await pick(user, "Fake-Chat")
    user.find(marker="message-input").type("hello")
    user.find(marker="send").click()

    await user.should_see("Paris.")  # a normal chat reply
    assert lemonade.drawn == [] and len(lemonade.requests) == 1


async def test_an_image_model_cannot_be_picked_while_the_images_module_is_off(
    user: User, web: AppContext
):
    lemonade = DrawingLemonade()
    module = await open_page(user, web, lemonade)
    web.settings.modules.enabled[module.config_key] = False

    await pick(user, "Draw-It")
    await user.should_see("Switch the Images module on")
    user.find(marker="message-input").type("hello")
    user.find(marker="send").click()
    await user.should_see("Paris.")  # still chatting with the chat model
    assert lemonade.drawn == []


async def test_a_failed_drawing_is_explained_and_the_description_comes_back(
    user: User, web: AppContext
):
    lemonade = DrawingLemonade()
    await open_page(user, web, lemonade)

    async def broken(prompt: str, model: str, **options) -> bytes:
        from lemonrind.lemonade.client import LemonadeError

        raise LemonadeError("the image model fell over")

    lemonade.generate_image = broken  # type: ignore[method-assign]
    await pick(user, "Draw-It")
    user.find(marker="message-input").type("try again")
    user.find(marker="send").click()
    await user.should_see("Draw a picture")
    user.find(marker="dialog-ok").click()

    await user.should_see("the image model fell over")
    assert web.repo.list_sessions() == []  # nothing saved for a failed picture
    (box,) = elements(user, "message-input")
    assert box.value == "try again"
