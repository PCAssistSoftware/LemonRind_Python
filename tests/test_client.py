"""Tests for LemonadeClient using fakes in place of the network.

Python does not need interfaces for this: the client only calls ``openai_client.chat.completions.create``
and ``http_client.get/post``, so any object with those methods will do.
"""

from __future__ import annotations

import httpx2 as httpx
import openai
import pytest

from lemonrind.config import LemonadeSettings
from lemonrind.lemonade import (
    ChatEvent,
    Finished,
    LemonadeClient,
    PromptProgress,
    ReasoningDelta,
    TextDelta,
)
from lemonrind.lemonade.client import LemonadeError
from tests.conftest import make_chunk


class FakeCompletions:
    """Stands in for ``client.chat.completions``: ``create`` returns an async iterable of chunks."""

    def __init__(self, chunks=None, error: Exception | None = None):
        self._chunks = chunks or []
        self._error = error
        self.last_request: dict | None = None

    async def create(self, **request):
        self.last_request = request
        if self._error:
            raise self._error

        async def stream():
            for chunk in self._chunks:
                yield chunk

        return stream()


class FakeOpenAI:
    def __init__(self, completions: FakeCompletions):
        self.chat = type("Chat", (), {"completions": completions})()


def make_client(completions: FakeCompletions, handler=None) -> LemonadeClient:
    transport = httpx.MockTransport(handler or (lambda request: httpx.Response(404)))
    return LemonadeClient(
        LemonadeSettings(base_url="http://test/v1/"),
        openai_client=FakeOpenAI(completions),
        http_client=httpx.AsyncClient(base_url="http://test/v1/", transport=transport),
    )


async def collect(client: LemonadeClient) -> list[ChatEvent]:
    return [event async for event in client.stream_chat([{"role": "user", "content": "hi"}], "m")]


async def test_stream_yields_progress_thinking_text_then_finished():
    completions = FakeCompletions(
        [
            make_chunk(prompt_progress={"total": 10, "cache": 0, "processed": 10, "time_ms": 40}),
            make_chunk(reasoning="Hmm. "),
            make_chunk(content="Hi"),
            make_chunk(content=" there", finish_reason="stop"),
            make_chunk(
                no_choices=True,
                usage={"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
                timings={
                    "prompt_n": 10,
                    "prompt_ms": 40.0,
                    "predicted_n": 4,
                    "predicted_per_second": 20.0,
                },
            ),
        ]
    )
    events = await collect(make_client(completions))

    kinds = [type(e) for e in events]
    assert kinds == [PromptProgress, ReasoningDelta, TextDelta, TextDelta, Finished]
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "Hi there"
    finished = events[-1]
    assert finished.finish_reason == "stop"
    assert finished.stats.output_tokens == 4 and finished.stats.tokens_per_second == 20.0


async def test_request_asks_lemonade_for_progress_and_usage():
    completions = FakeCompletions([make_chunk(content="x", finish_reason="stop")])
    await collect(make_client(completions))
    request = completions.last_request
    assert request["stream"] is True
    assert request["extra_body"] == {"return_progress": True}
    assert request["stream_options"] == {"include_usage": True}


async def test_sdk_timeout_becomes_a_readable_lemonade_error():
    timeout = openai.APITimeoutError(
        request=httpx.Request("POST", "http://test/v1/chat/completions")
    )
    client = make_client(FakeCompletions(error=timeout))
    with pytest.raises(LemonadeError, match="timed out"):
        await collect(client)


async def test_health_and_models_parse_lemonade_json():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/health"):
            return httpx.Response(
                200,
                json={
                    "status": "ok",
                    "model_loaded": "Qwen3-8B-GGUF",
                    "all_models_loaded": [{"model_name": "Qwen3-8B-GGUF", "device": "gpu"}],
                    "something_new": True,  # unknown fields must be ignored
                },
            )
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": "Qwen3-8B-GGUF", "labels": ["chat", "tool-calling"], "downloaded": True},
                    {"id": "Some-Embedding", "labels": ["embeddings"], "downloaded": True},
                    {"id": "Not-Downloaded", "labels": ["chat"], "downloaded": False},
                ]
            },
        )

    client = make_client(FakeCompletions(), handler)
    health = await client.health()
    assert health.is_ok and health.all_models_loaded[0].model_name == "Qwen3-8B-GGUF"

    models = await client.list_models()
    assert [m.id for m in models] == ["Qwen3-8B-GGUF", "Some-Embedding"]
    assert [m.category for m in models] == ["chat", "embedding"]
    assert len(await client.list_models(downloaded_only=False)) == 3


async def test_unreachable_server_gives_a_friendly_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = make_client(FakeCompletions(), handler)
    with pytest.raises(LemonadeError, match="Could not reach Lemonade"):
        await client.health()


async def test_load_model_reports_failure():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "error", "message": "out of memory"})

    client = make_client(FakeCompletions(), handler)
    with pytest.raises(LemonadeError, match="out of memory"):
        await client.load_model("Big-Model")
