"""Typed shapes for the JSON that Lemonade's management endpoints return (health, model list).

Python ideas used here:

* A pydantic model turns a JSON dictionary into a typed object: ``Health.model_validate(data)``. Fields
  Lemonade sends that we do not declare are simply ignored, so a newer Lemonade adding fields never
  breaks us (the same effect as ignoring unknown properties when deserializing in C#).
* ``ConfigDict(protected_namespaces=())`` - pydantic reserves names starting with ``model_`` for its own
  methods and warns about fields like ``model_name``. Lemonade's JSON uses exactly those names, so we
  switch the warning off for these classes.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RecipeOptions(BaseModel):
    """The options a model was started with."""

    ctx_size: int | None = (
        None  # the context window actually in use (can be smaller than the model's maximum)
    )
    llamacpp_args: str | None = None  # extra command-line arguments for the llama.cpp backend


class LoadedModel(BaseModel):
    """One model Lemonade currently has loaded in memory."""

    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    device: str | None = None
    recipe: str | None = None  # which backend runs it, e.g. "llamacpp"
    max_context_window: int | None = None
    checkpoint: str | None = None  # where the weights come from
    type: str | None = None  # "llm", "embedding", ...
    status: str | None = None
    backend_health: str | None = None
    is_busy: bool = False
    recipe_options: RecipeOptions | None = None
    launch_command: list[str] = Field(
        default_factory=list
    )  # the real command line the backend was started with


class Health(BaseModel):
    """GET /health: is Lemonade up, and what is loaded right now."""

    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: str | None = None
    websocket_port: int | None = None
    all_models_loaded: list[LoadedModel] = Field(default_factory=list)

    @property
    def is_ok(self) -> bool:
        return self.status == "ok"


class ModelInfo(BaseModel):
    """One entry from GET /models."""

    id: str
    labels: list[str] = Field(default_factory=list)
    downloaded: bool = False
    size: float | None = None  # gigabytes, as Lemonade reports it
    context_length: int | None = None
    max_context_window: int | None = None
    recipe: str | None = None  # the backend that runs it ("llamacpp", "sd-cpp", ...)
    checkpoint: str | None = None
    source: str | None = None
    recipe_options: dict[str, Any] = Field(
        default_factory=dict
    )  # defaults for this model (sampling, image size...)

    @property
    def context_window(self) -> int | None:
        """The model's largest context window, whichever of the two fields Lemonade filled in."""
        return self.context_length or self.max_context_window

    @property
    def supports_vision(self) -> bool:
        return "vision" in {label.lower() for label in self.labels}

    @property
    def category(self) -> str:
        """Group used by model pickers: chat, image, embedding or other."""
        labels = {label.lower() for label in self.labels}
        if labels & {"embeddings", "embedding"}:
            return "embedding"
        if "image" in labels:
            return "image"
        if labels & {"chat", "llm"}:
            return "chat"
        return "other"
