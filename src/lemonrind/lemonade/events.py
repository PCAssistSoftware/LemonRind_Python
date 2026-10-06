"""The typed events a streaming reply is turned into.

Lemonade (like OpenAI) sends a reply as many small JSON chunks. The client reads those chunks and
produces *events* that the rest of the app can understand without knowing anything about the wire format:

* ``PromptProgress`` - the server is still reading the prompt (large prompts take a while)
* ``ReasoningDelta`` - a piece of the model's "thinking"
* ``TextDelta``      - a piece of the visible answer
* ``ToolCallsRequested`` - the model wants one or more tools run before it answers
* ``Finished``       - the reply is complete, with its statistics

Python ideas used here:

* ``@dataclass(frozen=True, slots=True)`` - a small immutable data class; Python writes the constructor,
  ``__repr__`` and equality for you. Think of a C# ``record``.
* ``type X = A | B | C`` - a type alias for "one of these". The events are plain classes with no common
  base class; code tells them apart with ``match`` (Python's pattern matching) or ``isinstance``.
* ``@property`` - a computed value that is read like a field.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PromptProgress:
    """llama.cpp's own progress while it processes the prompt, before any text is generated.

    ``cache`` tokens were already in the server's KV cache from an earlier request, so they never
    needed processing: they are excluded from both ``processed`` and ``total`` when working out progress.
    """

    total: int
    cache: int
    processed: int
    time_ms: float

    @property
    def percent(self) -> int | None:
        """Whole-number progress, or ``None`` when everything came from the cache (nothing to wait for)."""
        remaining_total = self.total - self.cache
        if remaining_total <= 0:
            return None
        return round(100 * (self.processed - self.cache) / remaining_total)

    @property
    def eta_seconds(self) -> int | None:
        """Estimated seconds left, or ``None`` while there is too little data for a sensible guess."""
        remaining_total = self.total - self.cache
        done = self.processed - self.cache
        if remaining_total <= 0 or done <= 0 or self.time_ms < 500:
            return None
        return max(0, round((self.time_ms / 1000) * (remaining_total / done - 1)))

    def describe(self) -> str | None:
        """Text for a status line, e.g. ``Processing prompt: 62% (ETA: 8s)``."""
        percent = self.percent
        if percent is None:
            return None
        eta = self.eta_seconds
        return f"Processing prompt: {percent}%" + (f" (ETA: {eta}s)" if eta is not None else "")


@dataclass(frozen=True, slots=True)
class ReasoningDelta:
    text: str


@dataclass(frozen=True, slots=True)
class TextDelta:
    text: str


@dataclass(frozen=True, slots=True)
class RequestStats:
    """Numbers about one finished reply.

    ``prompt_tokens`` is the full prompt size including tokens served from the cache. ``input_tokens``
    is only what had to be processed this time. The context-usage figure wants the first; "how much work
    did this turn cost" wants the second.
    """

    prompt_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    tokens_per_second: float = 0.0
    time_to_first_token: float = 0.0  # seconds

    def plus(self, later: RequestStats) -> RequestStats:
        """Combine two requests of one turn (a reply that needed tools takes several requests).

        Work done adds up. The prompt size and the speed are the *later* request's, because the context
        window is as full as the last prompt made it, and the first-token time stays the first request's:
        it is how long you waited before anything appeared.
        """
        return RequestStats(
            prompt_tokens=later.prompt_tokens,
            input_tokens=self.input_tokens + later.input_tokens,
            output_tokens=self.output_tokens + later.output_tokens,
            tokens_per_second=later.tokens_per_second,
            time_to_first_token=self.time_to_first_token,
        )

    def summary(self) -> str:
        """One line for display, e.g. ``40 in / 149 out | 27.2 tok/s | first token 0.07s | prompt 41 tokens``."""
        return (
            f"{self.input_tokens:,} in / {self.output_tokens:,} out | "
            f"{self.tokens_per_second:.1f} tok/s | first token {self.time_to_first_token:.2f}s | "
            f"prompt {self.prompt_tokens:,} tokens"
        )


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One tool the model asked for. ``arguments`` is the raw JSON text exactly as the model wrote it."""

    id: str
    name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class ToolCallsRequested:
    """Sent once, after the model has finished writing all of its tool calls for this request."""

    calls: tuple[ToolCall, ...]


@dataclass(frozen=True, slots=True)
class Finished:
    stats: RequestStats
    # "stop" (finished normally), "length" (hit the token cap), ...
    finish_reason: str | None = None


type ChatEvent = PromptProgress | ReasoningDelta | TextDelta | ToolCallsRequested | Finished
