"""Tests that talk to a real Lemonade Server. Skipped unless you opt in:

    $env:LEMONRIND_LIVE = "1"; $env:LEMONRIND_BASE_URL = "http://localhost:13305/v1/"; pytest -m live

They send a tiny prompt, so they are quick, but they do use the model.
"""

from __future__ import annotations

import os

import pytest

from lemonrind.config import LemonadeSettings
from lemonrind.lemonade import Finished, LemonadeClient, TextDelta

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("LEMONRIND_LIVE") != "1", reason="set LEMONRIND_LIVE=1 to run"
    ),
]


def live_settings() -> LemonadeSettings:
    return LemonadeSettings(
        base_url=os.environ.get("LEMONRIND_BASE_URL", "http://localhost:13305/v1/")
    )


async def test_server_is_healthy_and_has_a_chat_model():
    async with LemonadeClient(live_settings()) as client:
        assert (await client.health()).is_ok
        assert any(m.category == "chat" for m in await client.list_models())


async def test_a_short_reply_streams_and_reports_stats():
    async with LemonadeClient(live_settings()) as client:
        chat_models = [m.id for m in await client.list_models() if m.category == "chat"]
        model = os.environ.get("LEMONRIND_MODEL") or chat_models[0]
        events = [
            e
            async for e in client.stream_chat(
                [{"role": "user", "content": "Reply with the single word: pong"}],
                model,
                max_tokens=400,
            )
        ]
    assert isinstance(events[-1], Finished)
    assert events[-1].stats.output_tokens > 0
    assert any(isinstance(e, TextDelta) for e in events)
