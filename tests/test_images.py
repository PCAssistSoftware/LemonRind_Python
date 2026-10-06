"""Tests for image generation: the client call, choosing a model, the tool, and showing the picture in the chat."""

from __future__ import annotations

import base64
import re
from pathlib import Path

import httpx2 as httpx
import pytest
from nicegui import app
from nicegui.testing import User

from lemonrind.config import LemonadeSettings, Settings
from lemonrind.lemonade import LemonadeClient, ToolCall
from lemonrind.lemonade.client import LemonadeError
from lemonrind.lemonade.models import ModelInfo
from lemonrind.lemonade.selection import pick_image_model
from lemonrind.modules import ImagesModule, ModuleRegistry
from lemonrind.modules.images import clamp_size
from tests.test_webui import FakeLemonade, make_context

PNG = b"\x89PNG\r\n\x1a\n" + b"fake image bytes"

MODELS = [
    ModelInfo(id="Chat", labels=["chat"], downloaded=True, size=8.0),
    ModelInfo(id="Big-Image", labels=["image"], downloaded=True, size=33.0),
    ModelInfo(id="Small-Image", labels=["image", "edit"], downloaded=True, size=17.0),
]


# --- the client ---------------------------------------------------------------------------------------------------


def client_with(handler) -> LemonadeClient:
    transport = httpx.MockTransport(handler)
    return LemonadeClient(
        LemonadeSettings(base_url="http://test/v1/"),
        openai_client=object(),
        http_client=httpx.AsyncClient(base_url="http://test/v1/", transport=transport),
    )


async def test_the_client_asks_for_inline_base64_and_decodes_the_picture():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(PNG).decode()}]})

    png = await client_with(handler).generate_image("a lemon", "Img", width=512, height=768, seed=7)

    assert png == PNG
    assert seen[0].url.path == "/v1/images/generations"
    body = httpx.Response(200, content=seen[0].content).json()
    assert body == {
        "model": "Img",
        "prompt": "a lemon",
        "size": "512x768",
        "response_format": "b64_json",
        "seed": 7,
    }


async def test_optional_settings_are_sent_only_when_given():
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(httpx.Response(200, content=request.content).json())
        return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(PNG).decode()}]})

    await client_with(handler).generate_image(
        "x", "m", width=256, height=256, steps=4, cfg_scale=1.5
    )

    assert bodies[0]["steps"] == 4 and bodies[0]["cfg_scale"] == 1.5 and "seed" not in bodies[0]


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(500, json={"error": {"message": "out of memory"}}), "out of memory"),
        (httpx.Response(404, json={"error": "no such model"}), "no such model"),
        (httpx.Response(502, text="<html>bad gateway</html>"), "HTTP 502"),
        (httpx.Response(200, json={"data": []}), "did not contain an image"),
        (httpx.Response(200, json={"oops": 1}), "did not contain an image"),
        (
            httpx.Response(200, json={"data": [{"b64_json": "!!!not base64!!!"}]}),
            "did not contain an image",
        ),
    ],
)
async def test_failures_become_readable_errors(response, message):
    with pytest.raises(LemonadeError, match=message):
        await client_with(lambda request: response).generate_image("x", "m", width=256, height=256)


async def test_an_unreachable_server_is_reported():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    with pytest.raises(LemonadeError, match="Could not reach Lemonade"):
        await client_with(handler).generate_image("x", "m", width=256, height=256)


# --- choosing a model and a size ----------------------------------------------------------------------------------------


def test_the_requested_image_model_wins_else_the_smallest_and_chat_models_never_count():
    assert pick_image_model("Big-Image", MODELS) == "Big-Image"
    assert pick_image_model("", MODELS) == "Small-Image"
    assert pick_image_model("Chat", MODELS) == "Small-Image"  # a chat model cannot draw
    assert pick_image_model("", MODELS[:1]) is None


@pytest.mark.parametrize(
    ("asked", "expected"),
    [
        ((1000, 700), (1024, 704)),
        ((10, 10), (256, 256)),
        ((9000, 300), (1536, 320)),
        ((512, 512), (512, 512)),
    ],
)
def test_sizes_are_rounded_to_64_and_limited(asked, expected):
    assert clamp_size(*asked, maximum=1536) == expected


# --- the tool -----------------------------------------------------------------------------------------------------------


class FakeImageClient:
    def __init__(self, *, models=MODELS, error: Exception | None = None) -> None:
        self.models, self.error = models, error
        self.requests: list[dict] = []

    async def list_models(self, **kwargs):
        return self.models

    async def generate_image(self, prompt, model, **kwargs) -> bytes:
        self.requests.append({"prompt": prompt, "model": model, **kwargs})
        if self.error:
            raise self.error
        return PNG


def make_module(tmp_path: Path, client=None):
    settings = Settings()
    client = client or FakeImageClient()
    return ImagesModule(settings, tmp_path, client), client, settings


async def test_the_tool_saves_the_picture_and_hands_back_markdown_to_show_it(tmp_path: Path):
    module, client, _ = make_module(tmp_path)
    tool = module.get_tools()[0]

    result = await tool.run('{"prompt": "a lemon [on] a table", "width": 1000, "height": 700}')

    assert not result.is_error
    match = re.search(
        r"!\[(.*?)\]\(/generated/(img_\d{8}_\d{6}_[0-9a-f]{8}\.png)\)", result.content
    )
    assert match is not None
    assert "[" not in match.group(1) and "lemon" in match.group(
        1
    )  # brackets cannot break the Markdown
    assert (tmp_path / "images" / match.group(2)).read_bytes() == PNG
    assert (
        client.requests[0] | {}
        == {
            "prompt": "a lemon [on] a table",
            "model": "Small-Image",
            "width": 1024,
            "height": 704,
            "steps": 4,  # the new defaults: 4 steps, CFG 1, and a seed of -1 meaning random (so none is sent)
            "cfg_scale": 1.0,
            "seed": None,
        }
    )
    assert "1024 x 704, Small-Image" in result.content


async def test_the_configured_model_size_and_overrides_are_used(tmp_path: Path):
    module, client, settings = make_module(tmp_path)
    settings.modules.images.model = "Big-Image"
    settings.modules.images.width = 512
    settings.modules.images.height = 512
    settings.modules.images.steps = 8
    settings.modules.images.seed = 42

    await module.get_tools()[0].run('{"prompt": "x"}')

    assert client.requests[0] | {} == {
        "prompt": "x", "model": "Big-Image", "width": 512, "height": 512, "steps": 8, "cfg_scale": 1.0, "seed": 42,
    }  # fmt: skip


@pytest.mark.parametrize(
    ("arguments", "message"),
    [('{"prompt": "   "}', "Give a description"), ('{"prompt": "' + "x" * 2001 + '"}', "too long")],
)
async def test_empty_and_overlong_prompts_are_refused_before_any_work(
    tmp_path: Path, arguments, message
):
    module, client, _ = make_module(tmp_path)

    result = await module.get_tools()[0].run(arguments)

    assert result.is_error and message in result.content and client.requests == []


async def test_no_image_model_and_server_errors_come_back_as_error_results(tmp_path: Path):
    none, _, _ = make_module(tmp_path, FakeImageClient(models=MODELS[:1]))
    broken, _, _ = make_module(
        tmp_path, FakeImageClient(error=LemonadeError("Image generation failed: out of memory"))
    )

    assert "no image model" in (await none.get_tools()[0].run('{"prompt": "x"}')).content
    failed = await broken.get_tools()[0].run('{"prompt": "x"}')
    assert failed.is_error and "out of memory" in failed.content
    assert list((tmp_path / "images").glob("*")) == []  # nothing saved for a failed picture


async def test_drawing_asks_permission_and_the_question_shows_prompt_and_size(tmp_path: Path):
    module, _, settings = make_module(tmp_path)
    settings.modules.images.model = "Big-Image"
    registry = ModuleRegistry([module])
    call = ToolCall("c", "generate_image", '{"prompt": "a lemon", "width": 1000}')

    assert registry.requires_approval(call)
    text = registry.describe(call)
    assert "Generate an image (1024 x 512) with Big-Image" in text and "Prompt: a lemon" in text
    assert "unload the chat model" in text


async def test_the_images_folder_is_created_at_start_up(tmp_path: Path):
    module, _, _ = make_module(tmp_path)
    await module.on_startup()
    assert (tmp_path / "images").is_dir()


# --- showing it in the web chat ----------------------------------------------------------------------------------------


async def test_the_web_app_serves_generated_pictures_where_the_chat_links_to_them(
    user: User, tmp_path: Path
):
    from lemonrind.webui.app import serve_images
    from lemonrind.webui.context import set_context
    from lemonrind.webui.page import register_pages

    context = make_context(tmp_path, FakeLemonade())
    module = next(m for m in context.modules.modules if isinstance(m, ImagesModule))  # type: ignore[union-attr]
    module.images_dir.mkdir(parents=True, exist_ok=True)
    (module.images_dir / "img_test.png").write_bytes(PNG)
    chat = context.repo.create_session()
    context.repo.add_message(chat.id, "user", "draw a lemon")
    context.repo.add_message(
        chat.id, "assistant", "Here it is: ![a lemon](/generated/img_test.png)"
    )
    set_context(context)
    serve_images(context)
    register_pages()
    try:
        await user.open("/")

        response = await user.http_client.get("/generated/img_test.png")
        assert response.status_code == 200 and response.content == PNG
        missing = await user.http_client.get("/generated/nope.png")
        assert missing.status_code == 404
        traversal = await user.http_client.get("/generated/../settings.json")
        assert traversal.status_code in (400, 404)
    finally:
        set_context(None)
        app.routes[:] = [r for r in app.routes if getattr(r, "path", "") != "/generated"]
