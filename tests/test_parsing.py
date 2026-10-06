"""Tests for turning raw streamed chunks into parsed pieces and final statistics."""

from lemonrind.lemonade.client import build_stats, parse_chunk
from tests.conftest import make_chunk


def test_progress_chunk_is_recognised():
    chunk = make_chunk(prompt_progress={"total": 14, "cache": 2, "processed": 7, "time_ms": 30})
    parsed = parse_chunk(chunk)
    assert parsed.progress is not None
    assert (parsed.progress.total, parsed.progress.cache, parsed.progress.processed) == (14, 2, 7)
    assert parsed.text is None and parsed.reasoning is None


def test_reasoning_and_answer_text_are_kept_apart():
    assert parse_chunk(make_chunk(reasoning="Okay, ")).reasoning == "Okay, "
    answer = parse_chunk(make_chunk(content="Hello"))
    assert answer.text == "Hello" and answer.reasoning is None


def test_empty_content_is_treated_as_no_text():
    # llama.cpp sends content "" on some chunks; that must not count as the first answer token.
    assert parse_chunk(make_chunk(content="")).text is None


def test_finish_reason_and_final_usage_chunk():
    assert parse_chunk(make_chunk(finish_reason="length")).finish_reason == "length"
    final = make_chunk(
        no_choices=True,
        usage={"prompt_tokens": 14, "completion_tokens": 300, "total_tokens": 314},
        timings={
            "prompt_n": 14,
            "prompt_ms": 53.1,
            "predicted_n": 300,
            "predicted_per_second": 27.2,
        },
    )
    parsed = parse_chunk(final)
    assert parsed.usage.completion_tokens == 300
    assert parsed.timings is not None and parsed.timings["predicted_n"] == 300


def test_build_stats_prefers_server_timings():
    final = make_chunk(
        no_choices=True,
        usage={
            "prompt_tokens": 1000,
            "completion_tokens": 200,
            "total_tokens": 1200,
            "prompt_tokens_details": {"cached_tokens": 800},
        },
        timings={
            "prompt_n": 200,
            "prompt_ms": 1500.0,
            "predicted_n": 200,
            "predicted_per_second": 40.0,
        },
    )
    parsed = parse_chunk(final)
    stats = build_stats(parsed.usage, parsed.timings, measured_first_token_seconds=9.9)
    assert stats.prompt_tokens == 1000  # whole prompt, cache included
    assert stats.input_tokens == 200  # only what had to be processed
    assert stats.output_tokens == 200
    assert stats.tokens_per_second == 40.0
    assert stats.time_to_first_token == 1.5  # the server's figure, not our stopwatch


def test_build_stats_falls_back_to_our_own_numbers():
    stats = build_stats(usage=None, timings=None, measured_first_token_seconds=2.5)
    assert stats.time_to_first_token == 2.5
    assert stats.output_tokens == 0 and stats.tokens_per_second == 0.0
