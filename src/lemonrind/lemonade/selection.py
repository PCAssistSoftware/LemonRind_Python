"""Which chat model to use: a rule shared by every front end.

Kept as a plain function (no printing, no network) so it can be tested directly, and so the terminal and
the web UI cannot drift apart.
"""

from __future__ import annotations

from collections.abc import Sequence

from lemonrind.lemonade.models import Health, ModelInfo


class ModelSelectionError(Exception):
    """No usable chat model could be chosen; the message says why and is fit to show the user."""


def pick_embedding_model(requested: str, models: Sequence[ModelInfo]) -> str | None:
    """The embedding model to use, or ``None`` if Lemonade has none downloaded.

    The ``requested`` one (from the settings) wins if it exists; otherwise the first downloaded embedding model.
    """
    embedding_models = [m.id for m in models if m.category == "embedding"]
    if requested and requested in embedding_models:
        return requested
    return embedding_models[0] if embedding_models else None


def pick_image_model(requested: str, models: Sequence[ModelInfo]) -> str | None:
    """The image model to draw with, or ``None`` if Lemonade has none downloaded.

    The ``requested`` one wins if it exists; otherwise the *smallest* image model (it loads and draws fastest).
    """
    images = [m for m in models if m.category == "image"]
    if requested and requested in {m.id for m in images}:
        return requested
    if not images:
        return None
    return min(images, key=lambda m: m.size if m.size is not None else float("inf")).id


def _best_default(models: Sequence[ModelInfo]) -> str:
    """The model to start with when nothing is loaded or requested.

    This app leans on tool calling, so models labelled ``tool-calling`` come first, and the smallest of them
    wins because it loads fastest. (Choosing the first in the list instead once picked an image model that
    the server also labels "chat".)
    """
    chat = [m for m in models if m.category == "chat"]
    capable = [m for m in chat if "tool-calling" in {label.lower() for label in m.labels}]
    pool = capable or chat
    return min(pool, key=lambda m: m.size if m.size is not None else float("inf")).id


def default_is_missing(default: str, models: Sequence[ModelInfo]) -> bool:
    """Is a saved default model named, but no longer on this Lemonade (removed, renamed or replaced)?"""
    return bool(default) and default not in {m.id for m in models}


def pick_chat_model(
    requested: str, health: Health, models: Sequence[ModelInfo], *, default: str = ""
) -> str:
    """Return the id of the chat model to use.

    * ``requested`` (a model asked for by name: ``--model``, or a scheduled job's own model) wins if Lemonade has it
      downloaded, and is an error if not: the person named it, so quietly using another would be wrong.
    * ``default`` (the default chat model saved in Settings) is used if Lemonade has it. If it is gone, it is
      ignored and the choice below is made instead, because a setting made months ago must not stop the app working
      the day that model is removed. Callers use ``default_is_missing`` to tell the person.
    * With nothing named, use the model Lemonade has loaded *if it is a chat model*. The "loaded" model
      is simply whatever was loaded most recently, which can be an embedding model used by a background
      job, so it is only trusted when it really is a chat model; failing that, any loaded chat model.
    * Otherwise fall back to the smallest tool-capable chat model (see ``_best_default``).
    """
    chat_models = [m.id for m in models if m.category == "chat"]
    if requested:
        if requested not in {m.id for m in models}:
            available = ", ".join(chat_models) or "none"
            raise ModelSelectionError(
                f"Model '{requested}' is not downloaded on this Lemonade. Chat models available: {available}"
            )
        return requested
    if default and default in {m.id for m in models}:
        return default
    if health.model_loaded in chat_models:
        return health.model_loaded or ""
    # The newest model is not a chat model (the memory module just used the embedding model, say), but
    # a chat model may well still be loaded alongside it: using that avoids loading another one.
    for loaded in health.all_models_loaded:
        if loaded.model_name in chat_models:
            return loaded.model_name
    if chat_models:
        return _best_default(models)
    raise ModelSelectionError("Lemonade has no downloaded chat models.")
