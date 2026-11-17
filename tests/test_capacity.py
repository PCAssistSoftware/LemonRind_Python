"""Whether a model fits the server's memory, and the readings it is judged from."""

from __future__ import annotations

import pytest

from lemonrind.lemonade.capacity import assess_fit, physical_memory_gb
from lemonrind.lemonade.models import GpuStats, SystemStats


def stats(total: float | None, *, kind: str = "dedicated") -> SystemStats:
    return SystemStats(
        primary_gpu_id="g0",
        gpus=[GpuStats(id="g0", memory_total_gb=total, memory_used_gb=1.0, memory_type=kind)],
    )


@pytest.mark.parametrize(
    ("size", "verdict"),
    [(10.0, "fits"), (70.0, "fits"), (71.0, "tight"), (100.0, "tight"), (101.0, "too_big")],
)
def test_the_verdict_follows_the_share_of_graphics_memory(size: float, verdict: str):
    assert assess_fit(size, stats(100.0), None).verdict == verdict


def test_shared_memory_is_named_as_such():
    fit = assess_fit(12.0, stats(121.0, kind="shared"), None)
    assert fit.verdict == "fits"
    assert "shared with the main memory" in fit.message


def test_without_a_graphics_chip_main_memory_is_the_limit():
    fit = assess_fit(30.0, SystemStats(), 32.0)
    assert fit.verdict == "tight" and "main memory" in fit.message


def test_missing_figures_give_unknown_not_a_guess():
    assert assess_fit(None, stats(100.0), None).verdict == "unknown"
    assert assess_fit(5.0, SystemStats(), None).verdict == "unknown"
    assert assess_fit(5.0, stats(None), None).verdict == "unknown"


@pytest.mark.parametrize(
    ("text", "gb"),
    [("122.20 GB", 122.2), ("2048 MB", 2.0), ("1 TB", 1024.0), ("lots", None), (None, None)],
)
def test_physical_memory_is_read_from_the_text(text: object, gb: float | None):
    assert physical_memory_gb({"Physical Memory": text}) == gb
