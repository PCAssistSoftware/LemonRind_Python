"""Tests for the prompt-progress arithmetic (a plain class, so the tests are plain functions)."""

from lemonrind.lemonade.events import PromptProgress


def test_percent_ignores_tokens_served_from_the_cache():
    # 1000 token prompt, 400 already cached, 700 processed so far -> 300 of the 600 that needed work.
    progress = PromptProgress(total=1000, cache=400, processed=700, time_ms=2000)
    assert progress.percent == 50


def test_everything_cached_means_nothing_to_report():
    progress = PromptProgress(total=500, cache=500, processed=500, time_ms=10)
    assert progress.percent is None
    assert progress.describe() is None


def test_eta_needs_enough_elapsed_time_and_some_progress():
    assert PromptProgress(total=100, cache=0, processed=0, time_ms=0).eta_seconds is None
    assert (
        PromptProgress(total=100, cache=0, processed=10, time_ms=100).eta_seconds is None
    )  # too early
    # 25% done after 2s -> 6s still to go.
    assert PromptProgress(total=100, cache=0, processed=25, time_ms=2000).eta_seconds == 6


def test_describe_formats_percent_and_eta():
    progress = PromptProgress(total=100, cache=0, processed=25, time_ms=2000)
    assert progress.describe() == "Processing prompt: 25% (ETA: 6s)"
    assert (
        PromptProgress(total=100, cache=0, processed=25, time_ms=100).describe()
        == "Processing prompt: 25%"
    )
