"""The Lemonade client: health, models, loading a model, and streaming chat.

Two libraries are used on purpose:

* the ``openai`` SDK for the chat call, because Lemonade speaks the OpenAI API and the SDK handles the
  streaming protocol, and
* ``httpx2`` (the HTTP client the SDK itself is built on) for Lemonade's own extra endpoints that the
  OpenAI API does not have (``/health``, ``/models`` details, ``/load``).

Python ideas used here:

* ``async def`` / ``await`` - functions that can wait for the network without blocking everything else.
* An *async generator* (``async def`` containing ``yield``): ``stream_chat`` hands back events one at a
  time as they arrive, and callers consume it with ``async for``. Think of C# ``IAsyncEnumerable<T>``.
* Dependency injection by plain arguments: the constructor accepts ready-made clients so tests can pass
  in fakes. No DI container, no interfaces: anything with the right methods works ("duck typing").
* A small *accumulator* class (``ToolCallAssembler``): a tool call arrives in pieces spread over many
  chunks and is put back together here.
* Pure functions (``parse_chunk``, ``build_stats``) hold the parsing logic so it can be tested without a
  network or a model.
"""

from __future__ import annotations

import base64
import time
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx2 as httpx
import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionChunk

from lemonrind.config import LemonadeSettings
from lemonrind.lemonade.events import (
    ChatEvent,
    Finished,
    PromptProgress,
    ReasoningDelta,
    RequestStats,
    TextDelta,
    ToolCall,
    ToolCallsRequested,
)
from lemonrind.lemonade.models import Health, ModelInfo


class LemonadeError(Exception):
    """Anything that went wrong talking to Lemonade, with a message fit to show the user."""


@dataclass(frozen=True, slots=True)
class ToolCallFragment:
    """One piece of a tool call. The first piece carries the id and name, later ones more arguments text."""

    index: int  # which of the (possibly several) tool calls in this reply the piece belongs to
    id: str | None
    name: str | None
    arguments: str


class ToolCallAssembler:
    """Collects ``ToolCallFragment`` pieces and builds the finished ``ToolCall`` objects.

    The model writes ``{"city": "Paris"}`` a few characters at a time, so the arguments text of a call is
    the pieces joined in order. Two calls in one reply are told apart by ``index``.
    """

    def __init__(self) -> None:
        self._parts: dict[int, dict[str, Any]] = {}

    def add(self, fragment: ToolCallFragment) -> None:
        part = self._parts.setdefault(fragment.index, {"id": "", "name": "", "arguments": []})
        if fragment.id:
            part["id"] = fragment.id
        if fragment.name:
            part["name"] = fragment.name
        part["arguments"].append(fragment.arguments)

    def build(self) -> tuple[ToolCall, ...]:
        calls = []
        for index in sorted(self._parts):
            part = self._parts[index]
            if not part["name"]:
                continue  # a fragment with no name is not a usable call
            # Every request needs an id to pair a result with its call. Servers normally supply one;
            # if one does not, make one up.
            call_id = part["id"] or f"call_{uuid.uuid4().hex[:12]}"
            calls.append(ToolCall(call_id, part["name"], "".join(part["arguments"])))
        return tuple(calls)

    def __bool__(self) -> bool:
        return bool(self._parts)


@dataclass(slots=True)
class ParsedChunk:
    """What one streamed JSON chunk contained. Every field is optional; most chunks hold just one."""

    progress: PromptProgress | None = None
    reasoning: str | None = None
    text: str | None = None
    tool_calls: list[ToolCallFragment] = field(default_factory=list)
    finish_reason: str | None = None
    usage: Any = None  # the OpenAI "usage" object, present on the final chunk
    timings: dict[str, Any] | None = None  # llama.cpp's own timing numbers, final chunk


def parse_chunk(chunk: ChatCompletionChunk) -> ParsedChunk:
    """Pull the interesting parts out of one chunk.

    Lemonade adds fields the OpenAI standard does not have, and the SDK keeps those in ``model_extra``:

    * ``prompt_progress`` (top level) - sent while the prompt is processed, when ``return_progress`` is on
    * ``reasoning_content`` (inside the delta) - the model's thinking, kept apart from the answer
    * ``timings`` (top level, last chunk) - prompt and generation speed measured by llama.cpp
    """
    parsed = ParsedChunk(usage=chunk.usage)

    extra = chunk.model_extra or {}
    if (progress := extra.get("prompt_progress")) is not None:
        parsed.progress = PromptProgress(
            total=int(progress.get("total", 0)),
            cache=int(progress.get("cache", 0)),
            processed=int(progress.get("processed", 0)),
            time_ms=float(progress.get("time_ms", 0.0)),
        )
    if (timings := extra.get("timings")) is not None:
        parsed.timings = timings

    if chunk.choices:
        choice = chunk.choices[0]
        parsed.finish_reason = choice.finish_reason
        parsed.text = choice.delta.content or None
        for piece in choice.delta.tool_calls or []:
            function = piece.function
            parsed.tool_calls.append(
                ToolCallFragment(
                    index=piece.index,
                    id=piece.id,
                    name=function.name if function else None,
                    arguments=(function.arguments or "") if function else "",
                )
            )
        delta_extra = choice.delta.model_extra or {}
        # llama.cpp calls it reasoning_content; some other servers say just "reasoning".
        parsed.reasoning = (
            delta_extra.get("reasoning_content") or delta_extra.get("reasoning") or None
        )
    return parsed


def build_stats(
    usage: Any,
    timings: Mapping[str, Any] | None,
    measured_first_token_seconds: float,
) -> RequestStats:
    """Combine the final chunk's ``usage`` and ``timings`` into one ``RequestStats``.

    Prefers llama.cpp's own timings (measured on the server, so network delay is excluded) and falls back
    to the OpenAI ``usage`` numbers and our own stopwatch when a server does not send them.
    """
    timings = timings or {}
    cached = 0
    prompt_tokens = 0
    output_tokens = 0
    if usage is not None:
        prompt_tokens = usage.prompt_tokens or 0
        output_tokens = usage.completion_tokens or 0
        details = getattr(usage, "prompt_tokens_details", None)
        cached = (getattr(details, "cached_tokens", 0) or 0) if details else 0

    input_tokens = int(timings.get("prompt_n", max(prompt_tokens - cached, 0)))
    if not prompt_tokens:
        prompt_tokens = input_tokens + int(timings.get("cache_n", 0))
    if not output_tokens:
        output_tokens = int(timings.get("predicted_n", 0))

    prompt_ms = timings.get("prompt_ms")
    return RequestStats(
        prompt_tokens=prompt_tokens,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        tokens_per_second=float(timings.get("predicted_per_second", 0.0)),
        time_to_first_token=(prompt_ms / 1000.0)
        if prompt_ms is not None
        else measured_first_token_seconds,
    )


class LemonadeClient:
    """Talks to one Lemonade Server. Create it once, use it for the life of the app, then ``aclose()``."""

    def __init__(
        self,
        settings: LemonadeSettings,
        *,
        openai_client: AsyncOpenAI | Any | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings
        timeout = settings.request_timeout_seconds
        # max_retries=0 matters. The SDK's default is to quietly retry a timed-out request, which
        # resubmits the same huge prompt to a local server that has no spare capacity to run it twice
        # (in the .NET editions this crashed Lemonade). Retrying, if ever wanted, is the app's decision.
        self._openai = openai_client or AsyncOpenAI(
            base_url=settings.base_url,
            api_key=settings.api_key
            or "lemonade",  # the SDK insists on a value; Lemonade ignores it
            timeout=timeout,
            max_retries=0,
        )
        headers = {"Authorization": f"Bearer {settings.api_key}"} if settings.api_key else {}
        self._http = http_client or httpx.AsyncClient(
            base_url=settings.base_url, headers=headers, timeout=timeout
        )

    @property
    def base_url(self) -> str:
        return self._settings.base_url

    # --- lifetime ------------------------------------------------------------------------------

    async def aclose(self) -> None:
        await self._http.aclose()
        close = getattr(self._openai, "close", None)
        if close is not None:
            await close()

    # ``async with LemonadeClient(...) as client:`` closes the client automatically, like C# ``await using``.
    async def __aenter__(self) -> LemonadeClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    # --- Lemonade's own endpoints ----------------------------------------------------------------

    async def health(self) -> Health:
        return Health.model_validate(await self._get_json("health"))

    async def list_models(self, *, downloaded_only: bool = True) -> list[ModelInfo]:
        data = await self._get_json("models")
        models = [ModelInfo.model_validate(item) for item in data.get("data", [])]
        return [m for m in models if m.downloaded] if downloaded_only else models

    async def tokenize(self, content: str) -> int:
        """How many tokens ``content`` is, counted by the loaded model's own tokenizer (exact, not characters / 4)."""
        if not content:
            return 0
        try:
            response = await self._http.post("tokenize", json={"content": content})
            response.raise_for_status()
            return len(response.json().get("tokens", []))
        except (httpx.HTTPError, ValueError) as error:
            raise LemonadeError(
                f"Could not count tokens ({type(error).__name__}: {error})"
            ) from error

    async def load_model(self, model_name: str) -> None:
        """Load a model into memory. Blocks until it is ready, which can take a long time for a big model."""
        try:
            response = await self._http.post("load", json={"model_name": model_name})
            response.raise_for_status()
            result = response.json()
        except httpx.HTTPError as error:
            raise LemonadeError(f"Could not load '{model_name}': {error}") from error
        if result.get("status") != "success":
            raise LemonadeError(f"Lemonade failed to load '{model_name}': {result.get('message')}")

    async def _get_json(self, path: str) -> dict[str, Any]:
        try:
            response = await self._http.get(path)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as error:
            raise LemonadeError(
                f"Could not reach Lemonade at {self._settings.base_url} ({type(error).__name__}: {error})"
            ) from error

    # --- chat --------------------------------------------------------------------------------------

    async def stream_chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        model: str,
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[ChatEvent]:
        """Send a conversation and yield ``ChatEvent`` objects as the reply streams in.

        ``messages`` use the OpenAI format: ``{"role": "user", "content": "Hello"}``. ``tools`` are the
        tool descriptions (JSON schema) the model may call; when it does, a ``ToolCallsRequested`` event
        arrives just before ``Finished``.
        """
        request: dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "stream": True,
            # Ask for the usage numbers in the last chunk (otherwise a stream has none).
            "stream_options": {"include_usage": True},
            # llama.cpp extension: send prompt_progress chunks while the prompt is processed. It is not
            # part of the OpenAI API, so it travels in extra_body, which the SDK merges into the JSON.
            "extra_body": {"return_progress": True},
        }
        if tools:
            request["tools"] = list(tools)
        if max_tokens is not None:
            request["max_tokens"] = max_tokens
        if temperature is not None:
            request["temperature"] = temperature

        started = time.monotonic()
        first_token_seconds: float | None = None
        finish_reason: str | None = None
        usage: Any = None
        timings: dict[str, Any] | None = None
        assembler = ToolCallAssembler()

        try:
            stream = await self._openai.chat.completions.create(**request)
            async for chunk in stream:
                parsed = parse_chunk(chunk)

                if parsed.progress is not None:
                    yield parsed.progress
                if (parsed.reasoning or parsed.text) and first_token_seconds is None:
                    first_token_seconds = (
                        time.monotonic() - started
                    )  # stopwatch fallback for the stats
                if parsed.reasoning:
                    yield ReasoningDelta(parsed.reasoning)
                if parsed.text:
                    yield TextDelta(parsed.text)
                for fragment in parsed.tool_calls:
                    assembler.add(fragment)

                finish_reason = parsed.finish_reason or finish_reason
                usage = parsed.usage or usage
                timings = parsed.timings or timings
        except openai.OpenAIError as error:
            raise LemonadeError(_describe_openai_error(error, model)) from error

        if assembler and (calls := assembler.build()):
            yield ToolCallsRequested(calls)
        yield Finished(
            stats=build_stats(usage, timings, first_token_seconds or 0.0),
            finish_reason=finish_reason,
        )

    # --- images ---------------------------------------------------------------------------------------------------

    async def generate_image(
        self,
        prompt: str,
        model: str,
        *,
        width: int,
        height: int,
        steps: int | None = None,
        cfg_scale: float | None = None,
        seed: int | None = None,
    ) -> bytes:
        """Draw a picture and return it as PNG bytes. Slow: the first call also loads the image model.

        Uses Lemonade's OpenAI-style ``/images/generations`` endpoint, asking for the picture inline as base64
        (``b64_json``) rather than as a link. ``steps``, ``cfg_scale`` and ``seed`` are sent only when set.
        """
        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "size": f"{width}x{height}",
            "response_format": "b64_json",
        }
        for key, value in (("steps", steps), ("cfg_scale", cfg_scale), ("seed", seed)):
            if value is not None:
                body[key] = value
        try:
            response = await self._http.post("images/generations", json=body)
            response.raise_for_status()
            data = response.json()
            return base64.b64decode(data["data"][0]["b64_json"])
        except httpx.HTTPStatusError as error:
            raise LemonadeError(
                f"Image generation failed: {_server_message(error.response)}"
            ) from error
        except httpx.HTTPError as error:
            raise LemonadeError(
                f"Could not reach Lemonade for the image: {type(error).__name__}: {error}"
            ) from error
        except (KeyError, IndexError, ValueError, TypeError) as error:
            raise LemonadeError("Lemonade's answer did not contain an image.") from error

    # --- embeddings and one-shot JSON ---------------------------------------------------------------

    async def embed(self, texts: Sequence[str], model: str) -> list[list[float]]:
        """Turn each text into an embedding vector (a list of numbers) using an embedding model."""
        try:
            response = await self._openai.embeddings.create(model=model, input=list(texts))
        except openai.OpenAIError as error:
            raise LemonadeError(_describe_openai_error(error, model)) from error
        return [item.embedding for item in sorted(response.data, key=lambda d: d.index)]

    async def complete_json(
        self,
        messages: Sequence[Mapping[str, Any]],
        model: str,
        *,
        schema_name: str,
        schema: Mapping[str, Any],
    ) -> str:
        """One reply (not streamed) that the server forces to match a JSON schema. Returns the JSON text.

        Used for background jobs that want data back instead of prose, such as fact extraction. The
        ``response_format`` makes llama.cpp constrain generation to the schema, so the text parses.
        """
        try:
            response = await self._openai.chat.completions.create(
                model=model,
                messages=list(messages),
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": schema_name, "schema": dict(schema), "strict": True},
                },
            )
        except openai.OpenAIError as error:
            raise LemonadeError(_describe_openai_error(error, model)) from error
        return response.choices[0].message.content or ""


def _server_message(response: httpx.Response) -> str:
    """The reason a server gave for refusing a request: its JSON ``error.message`` if it has one, else the status."""
    try:
        error = response.json().get("error")
        message = error.get("message") if isinstance(error, dict) else error
        if message:
            return str(message)
    except (ValueError, AttributeError):
        pass
    return f"HTTP {response.status_code}"


def _describe_openai_error(error: openai.OpenAIError, model: str) -> str:
    """Turn an SDK exception into a sentence a person can act on."""
    if isinstance(error, openai.APITimeoutError):
        return "The request timed out and Lemonade never replied in time."
    if isinstance(error, openai.APIConnectionError):
        return f"Could not connect to Lemonade: {error}"
    message = getattr(error, "message", None) or str(error)
    return f"Lemonade rejected the request for model '{model}': {message}"
