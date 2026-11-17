"""Will a model fit in the memory of the machine Lemonade runs on?

A rough answer from the model's file size and the memory the server reports. It does not count the working space
a long conversation needs on top (the context window), so "fits" means the weights fit, not that every context size
will. Kept free of any screen code so the rule is easy to test.

Python ideas used here: a frozen dataclass for a result, and ``re`` to pull a number out of text like "122.20 GB".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from lemonrind.lemonade.models import SystemStats

# The share of memory a model's file may take and still leave room for the context, the desktop and everything
# else on the machine. A rule of thumb, not a measurement.
COMFORTABLE_SHARE = 0.7

type Verdict = Literal["fits", "tight", "too_big", "unknown"]


@dataclass(frozen=True, slots=True)
class MemoryFit:
    verdict: Verdict
    message: str


def physical_memory_gb(system_info: dict[str, Any]) -> float | None:
    """The machine's main memory in GB, from the ``Physical Memory`` text in /system-info ("122.20 GB")."""
    text = system_info.get("Physical Memory")
    if not isinstance(text, str):
        return None
    match = re.search(r"(\d+(?:\.\d+)?)\s*(GB|MB|TB)", text, re.IGNORECASE)
    if match is None:
        return None
    value = float(match.group(1))
    unit = match.group(2).upper()
    return value / 1024 if unit == "MB" else value * 1024 if unit == "TB" else value


def assess_fit(size_gb: float | None, stats: SystemStats | None, ram_gb: float | None) -> MemoryFit:
    """Compare a model's file size with the graphics memory (when the server has a graphics chip) or main memory."""
    if size_gb is None:
        return MemoryFit("unknown", "The model's size is not known.")

    gpu = stats.primary_gpu if stats is not None else None
    if gpu is not None and gpu.memory_total_gb:
        capacity = gpu.memory_total_gb
        where = (
            "graphics memory (shared with the main memory)"
            if gpu.memory_type == "shared"
            else "graphics memory"
        )
    elif ram_gb:
        capacity = ram_gb
        where = "main memory"
    else:
        return MemoryFit("unknown", "The server did not say how much memory it has.")

    share = size_gb / capacity
    summary = f"{size_gb:.1f} GB of {capacity:.0f} GB of {where} ({share:.0%})"
    if share <= COMFORTABLE_SHARE:
        return MemoryFit(
            "fits",
            f"Fits: {summary}. That is the model file alone: the context (KV cache) and working space need more.",
        )
    if share <= 1.0:
        return MemoryFit(
            "tight",
            f"Tight: {summary}. That is the model file alone, and the context (KV cache) and working space need more, so there is little room.",
        )
    return MemoryFit(
        "too_big", f"Too big: {summary}. It could only run partly from slower memory, if at all."
    )
