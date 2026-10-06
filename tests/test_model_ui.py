"""The grouped model picker, the model details popup, the token meter and its breakdown, and time awareness."""

from __future__ import annotations

import pytest
from nicegui.testing import User

from lemonrind.chats import ChatRepository, Conversation
from lemonrind.chats.conversation import time_awareness_text
from lemonrind.lemonade.client import LemonadeError
from lemonrind.lemonade.models import ModelInfo
from lemonrind.storage import Database
from lemonrind.webui.context import AppContext
from lemonrind.webui.model_ui import HEADER_PREFIX, compute_breakdown, group_models, is_header
from tests.test_conversation import ScriptedStreamer
from tests.test_settings_sections import AddsContext, box, settle

MODELS = [
    ModelInfo(
        id="zeta-chat",
        labels=["chat", "tool-calling"],
        downloaded=True,
        size=8.0,
        context_length=40000,
    ),
    ModelInfo(
        id="Alpha-Chat",
        labels=["chat", "vision"],
        downloaded=True,
        size=16.9,
        max_context_window=262144,
    ),
    ModelInfo(id="Draw-It", labels=["image"], downloaded=True, size=17.7),
    ModelInfo(id="Embed-It", labels=["embeddings"], downloaded=True),
    ModelInfo(id="Mystery", labels=["custom"], downloaded=True),
]

# --- grouping ------------------------------------------------------------------------------------------------------


def test_models_are_grouped_chat_image_embedding_other_and_sorted_without_empty_groups():
    groups = group_models(MODELS)

    assert [(key, [m.id for m in members]) for key, _, members in groups] == [
        ("chat", ["Alpha-Chat", "zeta-chat"]),  # sorted without regard to case
        ("image", ["Draw-It"]),
        ("embedding", ["Embed-It"]),
        ("other", ["Mystery"]),
    ]
    assert [header for _, header, _ in groups] == [
        "Chat models",
        "Image models",
        "Embedding models",
        "Other models",
    ]
    assert [key for key, _, _ in group_models([MODELS[0]])] == [
        "chat"
    ]  # only the groups that exist
    assert group_models([]) == []


def test_group_headers_are_recognised_and_model_names_are_not():
    assert is_header(HEADER_PREFIX + "chat") and not is_header("Alpha-Chat") and not is_header(None)


def test_model_capabilities_are_read_from_the_labels_and_either_context_field():
    assert MODELS[1].supports_vision and not MODELS[0].supports_vision
    assert (
        MODELS[0].context_window == 40000
        and MODELS[1].context_window == 262144
        and MODELS[3].context_window is None
    )


async def test_the_picker_shows_group_headers_that_cannot_be_chosen(user: User, web: AppContext):
    await user.open("/")
    await user.should_see("Lemonade healthy")
    await settle()
    select = box(user, "model-select")

    labels = [option["label"] for option in select._props["options"]]  # type: ignore[attr-defined]
    assert labels == ["Chat models", "Fake-Chat", "Embedding models", "Fake-Embed"]
    flags = [(o["label"], o.get("disable", False)) for o in select._props["options"]]  # type: ignore[attr-defined]
    assert flags == [
        ("Chat models", True),
        ("Fake-Chat", False),
        ("Embedding models", True),
        ("Fake-Embed", False),
    ]
    assert select.value == "Fake-Chat"  # type: ignore[attr-defined]


async def test_choosing_a_group_header_changes_nothing(user: User, web: AppContext):
    await user.open("/")
    await user.should_see("Lemonade healthy")
    await settle()
    select = box(user, "model-select")

    select.value = HEADER_PREFIX + "embedding"  # type: ignore[attr-defined]
    await settle()

    assert select.value == "Fake-Chat"  # type: ignore[attr-defined]


# --- the details popup -----------------------------------------------------------------------------------------------


async def test_the_info_button_shows_catalog_and_live_details_and_every_model(
    user: User, web: AppContext
):
    await user.open("/")

    user.find(marker="model-info").click()

    await user.should_see("Fake-Chat")
    await user.should_see("Largest context window")
    await user.should_see("8,000 tokens")
    await user.should_see("Currently loaded (live)")
    await user.should_see("All models on this server (2)")


async def test_the_info_button_says_so_when_the_model_is_not_loaded(
    user: User, web: AppContext, monkeypatch: pytest.MonkeyPatch
):
    from lemonrind.lemonade.models import Health

    async def nothing_loaded() -> Health:
        return Health(status="ok", model_loaded=None, all_models_loaded=[])

    await user.open("/")
    monkeypatch.setattr(web.client, "health", nothing_loaded)

    user.find(marker="model-info").click()

    await user.should_see("Not currently loaded.")


# --- the meter and the breakdown -------------------------------------------------------------------------------------


async def test_the_token_meter_appears_after_a_reply_and_opens_the_breakdown(
    user: User, web: AppContext
):
    await user.open("/")
    await user.should_not_see(marker="context-meter")  # nothing to show before the first reply

    user.find(marker="message-input").type("What is the capital of France?")
    user.find(marker="send").click()
    await user.should_see("Paris.")
    await user.should_see(marker="context-meter")
    assert box(user, "context-tip").text == (  # type: ignore[attr-defined]
        "Context used: 25 / 8,000 tokens (0%). Click for the breakdown."
    )
    assert box(user, "context-meter").value == 0  # type: ignore[attr-defined]

    user.find(marker="context-meter").click()
    await user.should_see("Context usage breakdown")
    await user.should_see("System prompt")
    await user.should_see("Conversation (summary, pinned facts, messages)")
    await user.should_see("Free")


class CountingClient:
    """Counts tokens as one per four characters, like the fake server does."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    async def tokenize(self, content: str) -> int:
        if self.fail:
            raise LemonadeError("tokenizer unreachable")
        return len(content) // 4 if content else 0


async def make_conversation(*, compaction: int = 0) -> Conversation:
    conversation = Conversation(
        client=ScriptedStreamer(), repo=ChatRepository(Database(":memory:")), system_prompt="x" * 400, model="m",
        compaction_percent=compaction,
    )  # fmt: skip
    await conversation.send("hello there")
    return conversation


async def test_the_breakdown_counts_the_system_prompt_history_and_what_is_being_typed():
    conversation = await make_conversation(compaction=75)

    info = await compute_breakdown(
        CountingClient(), conversation, window=10_000, tokens_now=None, typing="y" * 40
    )  # type: ignore[arg-type]

    assert info.exact and info.system_prompt == 100  # 400 characters
    assert info.history > 0 and info.tools == 0 and info.pending == 10
    assert (
        info.now == info.system_prompt + info.tools + info.history
    )  # no reply figure known: the sum
    assert info.threshold == 7500 and info.free == 10_000 - info.now and not info.would_overflow


async def test_the_breakdown_uses_the_servers_own_figure_and_warns_when_the_next_message_would_not_fit():
    conversation = await make_conversation()

    info = await compute_breakdown(
        CountingClient(), conversation, window=1000, tokens_now=900, typing="z" * 800
    )  # type: ignore[arg-type]

    assert info.now == 900 and info.pending == 200 and info.would_overflow
    assert round(info.percent) == 90 and info.threshold == 0  # compaction off: no marker


async def test_the_breakdown_falls_back_to_an_estimate_when_the_tokenizer_is_unreachable():
    conversation = await make_conversation()

    info = await compute_breakdown(
        CountingClient(fail=True), conversation, window=5000, tokens_now=None
    )  # type: ignore[arg-type]

    assert not info.exact and info.system_prompt == 100  # 400 characters divided by four


async def test_the_breakdown_notices_a_summarised_chat():
    conversation = await make_conversation()
    assert conversation.session is not None
    before = await compute_breakdown(CountingClient(), conversation, window=1000, tokens_now=None)  # type: ignore[arg-type]
    conversation._repo.set_summary(conversation.session.id, "They said hello.", 1)
    conversation.session = conversation._repo.get_session(conversation.session.id)

    after = await compute_breakdown(CountingClient(), conversation, window=1000, tokens_now=None)  # type: ignore[arg-type]

    assert before.summarised is False and after.summarised is True


# --- the turn context preview and time awareness -------------------------------------------------------------------


async def test_the_turn_context_preview_follows_what_is_typed(user: User, web: AppContext):
    await user.open("/")
    user.find(marker="message-input").type("tell me about otters")

    user.find(marker="view-turn-context").click()

    await user.should_see("Worked out for the message you are typing now.")
    await user.should_see("Current date/time:")


def test_the_time_line_names_the_day_date_time_and_zone():
    line = time_awareness_text()

    assert line.startswith("Current date/time: ") and line.count(":") >= 2
    assert time_awareness_text().split("(")[-1].endswith(")")


async def test_time_awareness_is_added_to_each_message_but_never_saved_or_announced():
    streamer = ScriptedStreamer()
    repo = ChatRepository(Database(":memory:"))
    conversation = Conversation(client=streamer, repo=repo, system_prompt="s", model="m")
    conversation.time_awareness = True
    events: list = []

    await conversation.send("What day is it?", events.append)

    sent = streamer.requests[0][-1]["content"]
    assert "Current date/time:" in sent and sent.endswith("What day is it?")
    (session,) = repo.list_sessions()
    assert [m.content for m in repo.list_messages(session.id)][
        0
    ] == "What day is it?"  # the saved chat is clean
    assert not any(
        type(event).__name__ == "ContextUsed" for event in events
    )  # only module text is announced


async def test_time_awareness_is_off_by_default_so_other_callers_are_unchanged():
    streamer = ScriptedStreamer()
    conversation = Conversation(
        client=streamer, repo=ChatRepository(Database(":memory:")), system_prompt="s", model="m"
    )

    await conversation.send("hello")

    assert streamer.requests[0][-1]["content"] == "hello"


async def test_the_time_line_comes_before_what_modules_add_and_the_preview_matches():
    streamer = ScriptedStreamer()
    conversation = Conversation(
        client=streamer, repo=ChatRepository(Database(":memory:")), system_prompt="s", model="m",
        context=AddsContext(),  # type: ignore[arg-type]
    )  # fmt: skip
    conversation.time_awareness = True

    preview = await conversation.preview_turn_context("otters")
    await conversation.send("otters")

    assert preview.startswith("Current date/time:") and preview.endswith("Background about otters")
    assert "Background about otters" in streamer.requests[0][-1]["content"]


async def test_the_turn_context_preview_changes_when_something_is_typed_and_says_no_knowledge_base_is_attached(
    user: User, web: AppContext
):
    await user.open("/")
    await user.should_see("Lemonade healthy")
    await settle()

    user.find(marker="view-turn-context").click()  # nothing typed: only the time line, plus a hint
    await user.should_see("Type a message first")
    await user.should_not_see("No knowledge base is attached")

    user.find(marker="message-input").type("Summarise my notes")
    user.find(marker="view-turn-context").click()  # something typed: modules are asked about it
    await user.should_see("Worked out for the message you are typing now.")
    await user.should_see(
        "No knowledge base is attached"
    )  # even though the words "knowledge base" were not typed


async def test_the_preview_skips_the_modules_when_nothing_is_typed():
    conversation = Conversation(
        client=ScriptedStreamer(),
        repo=ChatRepository(Database(":memory:")),
        system_prompt="s",
        model="m",
        context=AddsContext(),  # type: ignore[arg-type]
    )
    conversation.time_awareness = True

    empty = await conversation.preview_turn_context("   ")
    typed = await conversation.preview_turn_context("otters")

    assert empty.startswith("Current date/time:") and "Background" not in empty
    assert typed.endswith("Background about otters")
