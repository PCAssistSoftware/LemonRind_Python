"""Tests for conversation compaction: where to cut, what is summarised, and how the chat carries on."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest

from lemonrind.chats import (
    ChatRepository,
    CompactionFinished,
    CompactionStarted,
    Conversation,
    StoredMessage,
)
from lemonrind.chats.compaction import (
    SUMMARY_HEADER,
    SYSTEM_PROMPT,
    TOOL_RESULT_CHARS,
    build_transcript,
    split_for_compaction,
)
from lemonrind.lemonade import ChatEvent, Finished, TextDelta, ToolCall
from lemonrind.lemonade.client import LemonadeError
from lemonrind.lemonade.events import RequestStats
from lemonrind.storage import Database

NOW = datetime(2026, 10, 2, tzinfo=UTC)


def msg(role: str, content: str, *, id: int = 0, **extra) -> StoredMessage:
    return StoredMessage(id=id, session_id="s", role=role, content=content, created_at=NOW, **extra)  # type: ignore[arg-type]


# --- where to cut -----------------------------------------------------------------------------------------------


def turns(count: int) -> list[StoredMessage]:
    out = []
    for n in range(count):
        out += [msg("user", f"q{n}"), msg("assistant", f"a{n}")]
    return out


def test_the_cut_is_at_the_start_of_the_oldest_turn_that_is_kept():
    messages = turns(5)  # user messages at 0, 2, 4, 6, 8

    assert split_for_compaction(messages, keep_turns=2) == 6
    assert split_for_compaction(messages, keep_turns=4) == 2


def test_too_few_turns_means_nothing_to_fold():
    assert split_for_compaction(turns(3), keep_turns=3) is None
    assert split_for_compaction(turns(2), keep_turns=5) is None
    assert split_for_compaction([], keep_turns=1) is None


def test_a_reply_that_used_tools_is_never_split():
    messages = [
        msg("user", "old"),
        msg("assistant", "calling", tool_calls=(ToolCall("c1", "search", "{}"),)),
        msg("tool", "result", tool_call_id="c1"),
        msg("assistant", "done"),
        msg("user", "recent"),
        msg("assistant", "ok"),
    ]

    cut = split_for_compaction(messages, keep_turns=1)

    assert cut == 4  # the whole tool exchange is folded, together
    assert messages[cut].role == "user"


# --- the transcript ---------------------------------------------------------------------------------------------


def test_the_transcript_labels_speakers_tools_and_includes_the_earlier_summary():
    messages = [
        msg("user", "find otters"),
        msg("assistant", "", tool_calls=(ToolCall("c1", "web_search", '{"query": "otters"}'),)),
        msg("tool", "x" * (TOOL_RESULT_CHARS + 500), tool_call_id="c1"),
        msg("assistant", "Otters are mammals."),
    ]

    text = build_transcript("They discussed rivers.", messages)

    assert text.startswith("Earlier summary: They discussed rivers.")
    assert "User: find otters" in text
    assert 'Assistant used the tool web_search with {"query": "otters"}' in text
    assert "Result of web_search: " + "x" * TOOL_RESULT_CHARS + " [...]" in text  # shortened
    assert "x" * (TOOL_RESULT_CHARS + 1) not in text
    assert "Assistant: Otters are mammals." in text


# --- the conversation -------------------------------------------------------------------------------------------


class CompactingStreamer:
    """Plays the model: ordinary replies report a nearly full context; summary requests are recognised."""

    def __init__(self, *, prompt_tokens: int = 800, fail_summary: bool = False) -> None:
        self.prompt_tokens = prompt_tokens
        self.fail_summary = fail_summary
        self.requests: list[list[dict]] = []
        self.summary_requests: list[list[dict]] = []

    async def stream_chat(self, messages, model, **kwargs) -> AsyncIterator[ChatEvent]:
        snapshot = json.loads(json.dumps(list(messages)))
        if snapshot[0]["content"] == SYSTEM_PROMPT:
            self.summary_requests.append(snapshot)
            if self.fail_summary:
                raise LemonadeError("cannot summarise")
            yield TextDelta(f"SUMMARY {len(self.summary_requests)}")
            yield Finished(RequestStats(), "stop")
            return
        self.requests.append(snapshot)
        yield TextDelta(f"reply {len(self.requests)}")
        yield Finished(RequestStats(prompt_tokens=self.prompt_tokens, output_tokens=10), "stop")


def make(streamer: CompactingStreamer, **kwargs) -> tuple[Conversation, ChatRepository]:
    repo = ChatRepository(Database(":memory:"))
    settings = {"compaction_percent": 75, "compaction_keep_turns": 1, "context_window": 1000}
    conversation = Conversation(
        client=streamer, repo=repo, system_prompt="Be brief.", model="m", **{**settings, **kwargs}
    )
    return conversation, repo


async def test_a_nearly_full_chat_has_its_oldest_turns_summarised_on_the_next_message():
    streamer = CompactingStreamer()
    conversation, repo = make(streamer)
    events: list = []
    await conversation.send("first question")
    await conversation.send("second question")
    assert streamer.summary_requests == []  # one turn to fold is not enough yet (keep_turns=1)

    await conversation.send("third question", events.append)

    kinds = [type(e) for e in events]
    assert CompactionStarted in kinds and CompactionFinished(2) in events
    assert kinds.index(CompactionStarted) < kinds.index(CompactionFinished)

    (summary_request,) = streamer.summary_requests
    assert "User: first question" in summary_request[1]["content"]
    assert "second question" not in summary_request[1]["content"]  # kept word for word instead

    sent = streamer.requests[-1]  # what the model got for the third question
    assert sent[0]["content"] == f"Be brief.\n\n{SUMMARY_HEADER}SUMMARY 1"
    assert [m["content"] for m in sent[1:]] == ["second question", "reply 2", "third question"]

    # everything is still stored and shown; the summary just marks what is no longer sent
    session = conversation.session
    assert session is not None and session.summary == "SUMMARY 1"
    stored = repo.list_messages(session.id)
    assert len(stored) == 6
    assert session.summary_upto == stored[1].id  # the first turn's answer


async def test_reopening_a_compacted_chat_sends_the_summary_and_only_the_later_messages():
    streamer = CompactingStreamer()
    conversation, repo = make(streamer)
    for text in ("one", "two", "three"):
        await conversation.send(text)
    session = repo.get_session(conversation.session.id)  # type: ignore[union-attr]
    assert session is not None

    reopened = Conversation(client=streamer, repo=repo, system_prompt="Be brief.", model="m")
    messages = reopened.open(session)

    assert len(messages) == 6  # all of them are returned for display
    assert reopened.history[0]["content"].endswith("SUMMARY 1")
    assert [m["content"] for m in reopened.history[1:]] == ["two", "reply 2", "three", "reply 3"]


async def test_a_second_compaction_folds_the_first_summary_in():
    streamer = CompactingStreamer()
    conversation, _ = make(streamer)
    for text in ("one", "two", "three", "four"):
        await conversation.send(text)

    assert len(streamer.summary_requests) == 2
    assert "Earlier summary: SUMMARY 1" in streamer.summary_requests[1][1]["content"]
    assert conversation.session is not None and conversation.session.summary == "SUMMARY 2"


@pytest.mark.parametrize(
    "settings",
    [
        {"compaction_percent": 0},  # switched off
        {"context_window": None},  # the model's limit is unknown
        {"context_window": 100_000},  # plenty of room
    ],
)
async def test_no_compaction_when_off_unknown_or_roomy(settings):
    streamer = CompactingStreamer()
    conversation, _ = make(streamer, **settings)

    for text in ("one", "two", "three", "four"):
        await conversation.send(text)

    assert streamer.summary_requests == []


async def test_a_failed_summary_does_not_stop_the_chat_and_is_retried_later():
    streamer = CompactingStreamer(fail_summary=True)
    conversation, repo = make(streamer)
    for text in ("one", "two"):
        await conversation.send(text)

    reply = await conversation.send("three")  # compaction fails quietly

    assert reply.text == "reply 3"
    assert conversation.session is not None and conversation.session.summary == ""
    assert len(streamer.summary_requests) == 1

    streamer.fail_summary = False
    await conversation.send("four")  # the next message tries again, and succeeds
    assert conversation.session is not None
    assert repo.get_session(conversation.session.id).summary == "SUMMARY 2"  # type: ignore[union-attr]


async def test_starting_a_new_chat_forgets_the_summary():
    streamer = CompactingStreamer()
    conversation, _ = make(streamer)
    for text in ("one", "two", "three"):
        await conversation.send(text)

    conversation.new()

    assert conversation.history == [{"role": "system", "content": "Be brief."}]


async def test_summarising_does_not_reorder_the_chat_list():
    streamer = CompactingStreamer()
    conversation, repo = make(streamer)
    await conversation.send("one")
    session_id = conversation.session.id  # type: ignore[union-attr]
    before = repo.get_session(session_id).updated_at  # type: ignore[union-attr]

    repo.set_summary(session_id, "s", 1)

    assert repo.get_session(session_id).updated_at == before  # type: ignore[union-attr]


async def test_the_web_page_marks_where_a_chat_was_summarised(user, tmp_path):
    from lemonrind.webui.context import set_context
    from lemonrind.webui.page import register_pages
    from tests.test_webui import FakeLemonade, make_context

    context = make_context(tmp_path, FakeLemonade())
    chat = context.repo.create_session()
    context.repo.add_message(chat.id, "user", "old question")
    last_old = context.repo.add_message(chat.id, "assistant", "old answer")
    context.repo.add_message(chat.id, "user", "recent question")
    context.repo.add_message(chat.id, "assistant", "recent answer")
    context.repo.set_summary(chat.id, "They talked about old things.", last_old.id)
    set_context(context)
    register_pages()

    await user.open("/")

    await user.should_see("summarised to save space")
    await user.should_see("recent answer")
    set_context(None)
