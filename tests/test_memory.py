"""Tests for vectors, the memory store, fact extraction, and the Memory module's behaviour."""

from __future__ import annotations

import json

import numpy as np
import pytest

from lemonrind.config import Settings
from lemonrind.lemonade.client import LemonadeError
from lemonrind.lemonade.models import ModelInfo
from lemonrind.modules.memory import MemoryModule, MemoryRepository
from lemonrind.modules.memory.extraction import build_messages, parse_facts
from lemonrind.storage import Database
from lemonrind.vectors import from_blob, normalise, rank, to_blob
from tests.conftest import NO_CHAT

# --- vectors -----------------------------------------------------------------------------------------------


def test_normalise_gives_unit_length_and_leaves_a_zero_vector_alone():
    assert float(np.linalg.norm(normalise([3.0, 4.0]))) == pytest.approx(1.0)
    assert normalise([0.0, 0.0]).tolist() == [0.0, 0.0]


def test_a_vector_survives_a_round_trip_through_bytes():
    blob = to_blob([3.0, 4.0])
    assert len(blob) == 8  # two float32 numbers
    assert from_blob(blob).tolist() == pytest.approx([0.6, 0.8])


def test_rank_orders_by_similarity_and_honours_the_limit():
    stored = [normalise([1, 0]), normalise([1, 1]), normalise([0, 1])]

    ranked = rank([1, 0.1], stored)

    assert [i for i, _ in ranked] == [0, 1, 2]
    assert ranked[0][1] > ranked[1][1] > ranked[2][1]
    assert len(rank([1, 0.1], stored, limit=2)) == 2
    assert rank([1, 0], []) == []


# --- a fake Lemonade for the module --------------------------------------------------------------------------

# Each "topic" is an axis of the pretend embedding space; a text scores on every topic whose words it uses.
TOPICS = [
    ("manchester", "live", "city", "town"),
    ("pizza", "food", "eat"),
    ("darren", "name"),
    ("metric", "units"),
]


class FakeMemoryClient:
    def __init__(self, *, with_embedding_model: bool = True, extraction: str = '{"facts": []}'):
        self.with_embedding_model = with_embedding_model
        self.extraction = extraction
        self.extraction_error: Exception | None = None
        self.extraction_calls: list[list[dict]] = []

    async def list_models(self, *, downloaded_only: bool = True) -> list[ModelInfo]:
        if not self.with_embedding_model:
            return [ModelInfo(id="chat", labels=["chat"], downloaded=True)]
        return [ModelInfo(id="embed", labels=["embeddings"], downloaded=True)]

    async def embed(self, texts, model):
        assert model == "embed"
        return [
            [float(sum(word in text.lower() for word in words)) + 0.01 for words in TOPICS]
            for text in texts
        ]

    async def complete_json(self, messages, model, *, schema_name, schema) -> str:
        self.extraction_calls.append(messages)
        if self.extraction_error:
            raise self.extraction_error
        return self.extraction


def make_module(client: FakeMemoryClient | None = None, settings: Settings | None = None):
    client = client or FakeMemoryClient()
    settings = settings or Settings()
    settings.modules.memory.min_similarity = (
        0.6  # the fake's scores: related texts 1.0, unrelated 0.5
    )
    repo = MemoryRepository(Database(":memory:"))
    return MemoryModule(settings, repo, client), repo, client


# --- the repository -------------------------------------------------------------------------------------------


def test_repository_lists_pinned_first_and_tracks_embeddings():
    repo = MemoryRepository(Database(":memory:"))
    repo.add("ordinary", pinned=False, embedding=[1, 0])
    repo.add("core", pinned=True)

    listed = repo.list_all()

    assert [(m.content, m.pinned, m.has_embedding) for m in listed] == [
        ("core", True, False),
        ("ordinary", False, True),
    ]
    assert [m.content for m in repo.list_pinned()] == ["core"]
    assert [m.content for m, _ in repo.with_embeddings()] == ["ordinary"]
    assert [m.content for m in repo.without_embeddings()] == ["core"]


def test_editing_text_replaces_the_embedding_and_pinning_toggles():
    repo = MemoryRepository(Database(":memory:"))
    memory = repo.add("old text", pinned=False, embedding=[1, 0])

    repo.set_content(memory.id, "  new text ", None)  # no embedding available for the new text
    repo.set_pinned(memory.id, True)

    updated = repo.get(memory.id)
    assert updated is not None
    assert (updated.content, updated.pinned, updated.has_embedding) == ("new text", True, False)


def test_delete_removes_a_memory():
    repo = MemoryRepository(Database(":memory:"))
    memory = repo.add("x", pinned=False)
    repo.delete(memory.id)
    assert repo.list_all() == [] and repo.get(memory.id) is None


# --- parsing the model's extraction --------------------------------------------------------------------------


def test_parse_facts_reads_valid_json_and_drops_blank_facts():
    text = json.dumps(
        {
            "facts": [
                {"content": "The user is Darren.", "pinned": True},
                {"content": "  ", "pinned": False},
            ]
        }
    )
    (fact,) = parse_facts(text)
    assert (fact.content, fact.pinned) == ("The user is Darren.", True)


@pytest.mark.parametrize(
    "text", ["", "not json at all", '{"facts": "nope"}', "[1, 2]", '{"other": 1}']
)
def test_unusable_extraction_answers_mean_no_facts(text):
    assert parse_facts(text) == []


def test_the_extraction_prompt_warns_against_remembering_tool_output():
    system, user = build_messages("hello", "hi there")
    assert "INSIDE a document" in system["content"]
    assert (
        'User said: "hello"' in user["content"]
        and 'Assistant replied: "hi there"' in user["content"]
    )


# --- the hooks -----------------------------------------------------------------------------------------------


async def test_pinned_facts_go_into_the_stable_context_and_nothing_when_there_are_none():
    module, repo, _ = make_module()
    assert await module.stable_context() == ""

    repo.add("The user's name is Darren.", pinned=True)
    repo.add("The user likes pizza.", pinned=False)

    assert (
        await module.stable_context() == "Known facts about the user:\n- The user's name is Darren."
    )


async def test_turn_context_adds_only_related_unpinned_facts():
    module, _, _ = make_module()
    await module.save_fact("The user lives in Manchester.", pinned=False)
    await module.save_fact("The user likes pizza.", pinned=False)
    await module.save_fact(
        "The user lives in a city.", pinned=True
    )  # pinned: not repeated per turn

    context = await module.turn_context("Which town do I live in?", chat=NO_CHAT)

    assert (
        context
        == "Possibly relevant things you know about the user:\n- The user lives in Manchester."
    )
    assert await module.turn_context("Tell me a joke.", chat=NO_CHAT) == ""


async def test_a_near_duplicate_fact_is_not_saved_twice():
    module, repo, _ = make_module()

    first = await module.save_fact("The user lives in Manchester.", pinned=False)
    second = await module.save_fact("The user lives in the city of Manchester town.", pinned=True)
    third = await module.save_fact("The user likes pizza.", pinned=False)

    assert first is not None and second is None and third is not None
    assert len(repo.list_all()) == 2


async def test_editing_a_fact_updates_its_embedding():
    module, repo, _ = make_module()
    memory = await module.save_fact("The user likes pizza.", pinned=False)
    assert memory is not None

    await module.edit_fact(memory.id, "The user lives in Manchester.")

    assert "Manchester" in await module.turn_context("which town do I live in", chat=NO_CHAT)
    assert await module.turn_context("any food I eat", chat=NO_CHAT) == ""


async def test_without_an_embedding_model_facts_are_kept_but_not_searchable_until_one_appears():
    client = FakeMemoryClient(with_embedding_model=False)
    module, repo, _ = make_module(client)

    saved = await module.save_fact("The user lives in Manchester.", pinned=False)
    assert saved is not None and not saved.has_embedding
    assert await module.turn_context("which town do I live in", chat=NO_CHAT) == ""

    client.with_embedding_model = True  # Lemonade now has one
    module._embedder.reset()
    assert await module.embed_missing() == 1
    assert "Manchester" in await module.turn_context("which town do I live in", chat=NO_CHAT)


async def test_learning_after_a_turn_saves_the_extracted_facts():
    client = FakeMemoryClient(
        extraction=json.dumps(
            {"facts": [{"content": "The user's name is Darren.", "pinned": True}]}
        )
    )
    module, repo, _ = make_module(client)

    await module.after_turn("Hi, I'm Darren", "Nice to meet you.", model="chat")

    (memory,) = repo.list_all()
    assert (memory.content, memory.pinned) == ("The user's name is Darren.", True)
    assert client.extraction_calls[0][1]["content"].startswith('User said: "Hi, I\'m Darren"')


async def test_learning_can_be_switched_off_and_survives_failures():
    settings = Settings()
    settings.modules.memory.extract_facts = False
    module, repo, client = make_module(settings=settings)
    await module.after_turn("Hi, I'm Darren", "Hello", model="chat")
    assert client.extraction_calls == []

    settings.modules.memory.extract_facts = True
    client.extraction_error = LemonadeError("down")
    await module.after_turn("Hi", "Hello", model="chat")  # must not raise
    client.extraction_error = None
    client.extraction = "garbage"
    await module.after_turn("Hi", "Hello", model="chat")  # must not raise either
    assert repo.list_all() == []


async def test_the_remember_and_recall_tools():
    module, _, _ = make_module()
    tools = {tool.name: tool for tool in module.get_tools()}

    saved = await tools["remember"].run(
        '{"fact": "The user lives in Manchester.", "always_include": true}'
    )
    again = await tools["remember"].run('{"fact": "The user lives in Manchester city."}')
    empty = await tools["remember"].run('{"fact": " "}')
    recalled = await tools["recall"].run('{"query": "which town"}')

    assert saved.content == "Remembered."
    assert "already remember" in again.content
    assert empty.is_error
    assert "- The user lives in Manchester. (always known)" in recalled.content


async def test_recall_with_nothing_stored():
    module, _, _ = make_module()
    result = await {t.name: t for t in module.get_tools()}["recall"].run('{"query": "anything"}')
    assert result.content == "Nothing is remembered yet."
