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


class GpuStats(BaseModel):
    """One graphics chip in GET /system-stats."""

    id: str | None = None
    name: str = ""
    gpu_percent: float | None = None
    memory_total_gb: float | None = None
    memory_used_gb: float | None = None
    memory_type: str | None = None  # "shared" when the chip uses the computer's main memory


class SystemStats(BaseModel):
    """GET /system-stats: how busy the machine Lemonade runs on is right now. Any figure can be missing."""

    cpu_percent: float | None = None
    gpu_percent: float | None = None
    npu_percent: float | None = None
    memory_gb: float | None = None  # main memory in use
    primary_gpu_id: str | None = None
    gpus: list[GpuStats] = Field(default_factory=list)

    @property
    def primary_gpu(self) -> GpuStats | None:
        """The graphics chip Lemonade reports as its main one (else the first)."""
        for gpu in self.gpus:
            if gpu.id is not None and gpu.id == self.primary_gpu_id:
                return gpu
        return self.gpus[0] if self.gpus else None


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
