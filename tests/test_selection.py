"""Tests for the shared chat-model choice."""

import pytest

from lemonrind.lemonade.models import Health, LoadedModel, ModelInfo
from lemonrind.lemonade.selection import ModelSelectionError, pick_chat_model

MODELS = [
    ModelInfo(id="Chat-A", labels=["chat"], downloaded=True),
    ModelInfo(id="Chat-B", labels=["chat", "tool-calling"], downloaded=True),
    ModelInfo(id="Embed", labels=["embeddings"], downloaded=True),
]


def health(loaded: str | None) -> Health:
    models = [LoadedModel(model_name=loaded)] if loaded else []
    return Health(status="ok", model_loaded=loaded, all_models_loaded=models)


def test_a_requested_model_wins():
    assert pick_chat_model("Chat-B", health("Chat-A"), MODELS) == "Chat-B"


def test_a_requested_model_must_be_downloaded():
    with pytest.raises(ModelSelectionError, match="Chat-A, Chat-B"):
        pick_chat_model("Missing", health(None), MODELS)


def test_with_nothing_requested_a_loaded_chat_model_is_used():
    assert pick_chat_model("", health("Chat-B"), MODELS) == "Chat-B"


def test_a_loaded_embedding_model_is_not_mistaken_for_the_chat_model():
    # (Chat-B is the one labelled tool-calling, so it is the default)
    assert pick_chat_model("", health("Embed"), MODELS) == "Chat-B"


def test_with_nothing_loaded_a_tool_capable_chat_model_is_used():
    assert pick_chat_model("", health(None), MODELS) == "Chat-B"


def test_no_chat_models_is_an_error():
    only_embedding = [ModelInfo(id="Embed", labels=["embeddings"], downloaded=True)]
    with pytest.raises(ModelSelectionError, match="no downloaded chat models"):
        pick_chat_model("", health("Embed"), only_embedding)


def test_a_chat_model_still_loaded_beside_the_newest_embedding_model_is_preferred():
    # Embed was loaded last (so it is "the loaded model"), but Chat-B is still in memory too.
    both = Health(
        status="ok",
        model_loaded="Embed",
        all_models_loaded=[LoadedModel(model_name="Embed"), LoadedModel(model_name="Chat-B")],
    )
    assert pick_chat_model("", both, MODELS) == "Chat-B"


def test_the_best_embedding_model_is_the_requested_one_else_the_first():
    from lemonrind.lemonade.selection import pick_embedding_model

    models = [*MODELS, ModelInfo(id="Embed-2", labels=["embeddings"], downloaded=True)]
    assert pick_embedding_model("Embed-2", models) == "Embed-2"
    assert pick_embedding_model("", models) == "Embed"
    assert pick_embedding_model("Chat-A", models) == "Embed"  # a chat model cannot embed
    assert pick_embedding_model("", MODELS[:2]) is None


def test_with_nothing_loaded_the_smallest_tool_capable_chat_model_is_chosen():
    models = [
        ModelInfo(id="Image-Mislabelled", labels=["custom", "chat"], downloaded=True, size=9.0),
        ModelInfo(id="Big-Tools", labels=["chat", "tool-calling"], downloaded=True, size=17.0),
        ModelInfo(id="Small-Tools", labels=["chat", "tool-calling"], downloaded=True, size=8.0),
        ModelInfo(id="Tiny-NoTools", labels=["chat"], downloaded=True, size=1.0),
    ]
    assert pick_chat_model("", health(None), models) == "Small-Tools"


def test_without_any_tool_capable_model_the_smallest_chat_model_is_chosen():
    models = [
        ModelInfo(id="B", labels=["chat"], downloaded=True, size=5.0),
        ModelInfo(id="A", labels=["chat"], downloaded=True),  # size unknown: sorts last
    ]
    assert pick_chat_model("", health(None), models) == "B"
