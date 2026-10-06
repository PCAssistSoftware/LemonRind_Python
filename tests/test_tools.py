"""Tests for turning functions into tools, and for the tool-call plumbing in the Lemonade client."""

from __future__ import annotations

import json
from typing import Literal

from lemonrind.lemonade import Finished, LemonadeClient, ToolCall, ToolCallsRequested
from lemonrind.lemonade.client import ToolCallAssembler, ToolCallFragment, parse_chunk
from lemonrind.modules import ToolError, tool_from_function
from tests.conftest import make_chunk
from tests.test_client import FakeCompletions, make_client


def get_weather(city: str, units: Literal["metric", "imperial"] = "metric", days: int = 1) -> str:
    """Look up the weather for a city.

    Args:
        city: Name of the city, e.g. "Paris".
        units: Which units to use.
        days: How many days ahead
            (a wrapped description line).
    """
    return f"{city}:{units}:{days}"


# --- building a tool from a function ---------------------------------------------------------------------


def test_the_schema_is_built_from_the_signature_and_docstring():
    tool = tool_from_function(get_weather)

    assert tool.name == "get_weather"
    assert tool.description == "Look up the weather for a city."
    schema = tool.schema()["function"]["parameters"]
    assert schema["required"] == ["city"]  # only the parameter without a default
    assert schema["properties"]["city"] == {
        "type": "string",
        "description": 'Name of the city, e.g. "Paris".',
    }
    assert schema["properties"]["units"]["enum"] == ["metric", "imperial"]
    assert schema["properties"]["days"]["type"] == "integer"
    assert (
        schema["properties"]["days"]["description"]
        == "How many days ahead (a wrapped description line)."
    )
    assert "title" not in json.dumps(schema)  # pydantic's titles are stripped


def test_an_argument_named_title_survives_the_cleanup():
    def make_note(title: str) -> str:
        """Make a note."""
        return title

    schema = tool_from_function(make_note).parameters
    assert "title" in schema["properties"]


async def test_running_a_tool_validates_and_converts_the_arguments():
    tool = tool_from_function(get_weather)

    assert (await tool.run('{"city": "Rome", "days": "3"}')).content == "Rome:metric:3"  # "3" -> 3
    assert (
        await tool.run('{"city": "Rome", "extra": true}')
    ).content == "Rome:metric:1"  # extras ignored


async def test_bad_arguments_become_error_results_not_exceptions():
    tool = tool_from_function(get_weather)

    broken = await tool.run("{not json")
    missing = await tool.run("{}")
    wrong_type = await tool.run('{"city": "Rome", "days": "many"}')
    not_object = await tool.run("[1, 2]")

    assert all(r.is_error for r in (broken, missing, wrong_type, not_object))
    assert "not valid JSON" in broken.content
    assert "city" in missing.content
    assert "days" in wrong_type.content


async def test_tools_without_arguments_accept_empty_text():
    def ping() -> str:
        """Ping."""
        return "pong"

    assert (await tool_from_function(ping).run("")).content == "pong"


async def test_async_and_sync_functions_both_work_and_non_text_results_become_json():
    async def fetch(n: int) -> dict:
        """Fetch."""
        return {"n": n}

    result = await tool_from_function(fetch).run('{"n": 4}')
    assert json.loads(result.content) == {"n": 4}


async def test_tool_errors_and_unexpected_exceptions_are_reported_to_the_model():
    def refuses(x: int) -> str:
        """Refuses."""
        raise ToolError("No, because of reasons.")

    def crashes(x: int) -> str:
        """Crashes."""
        return str(1 / 0)

    refused = await tool_from_function(refuses).run('{"x": 1}')
    crashed = await tool_from_function(crashes).run('{"x": 1}')

    assert (refused.content, refused.is_error) == ("No, because of reasons.", True)
    assert crashed.is_error and "ZeroDivisionError" in crashed.content


# --- reassembling streamed tool calls -----------------------------------------------------------------------


def test_fragments_are_joined_into_calls_in_index_order():
    assembler = ToolCallAssembler()
    assert not assembler  # nothing yet
    for fragment in [
        ToolCallFragment(0, "id-a", "search", '{"q'),
        ToolCallFragment(1, "id-b", "calc", ""),
        ToolCallFragment(0, None, None, 'uery": "x"}'),
        ToolCallFragment(1, None, None, '{"e": "1+1"}'),
    ]:
        assembler.add(fragment)

    assert assembler
    assert assembler.build() == (
        ToolCall("id-a", "search", '{"query": "x"}'),
        ToolCall("id-b", "calc", '{"e": "1+1"}'),
    )


def test_a_call_without_an_id_gets_one_and_a_nameless_fragment_is_dropped():
    assembler = ToolCallAssembler()
    assembler.add(ToolCallFragment(0, None, "search", "{}"))
    assembler.add(ToolCallFragment(1, "orphan", None, "{}"))

    (call,) = assembler.build()
    assert call.name == "search" and call.id.startswith("call_")


def test_parse_chunk_reads_tool_call_fragments():
    chunk = make_chunk(
        tool_calls=[
            {
                "index": 0,
                "id": "abc",
                "type": "function",
                "function": {"name": "search", "arguments": '{"q'},
            }
        ]
    )

    (fragment,) = parse_chunk(chunk).tool_calls
    assert fragment == ToolCallFragment(0, "abc", "search", '{"q')


async def test_stream_chat_sends_the_tools_and_yields_the_assembled_calls_before_finished():
    def piece(**function):
        return make_chunk(tool_calls=[{"index": 0, **function}])

    completions = FakeCompletions(
        [
            piece(id="id1", type="function", function={"name": "search", "arguments": '{"query":'}),
            piece(function={"arguments": ' "otters"}'}),
            make_chunk(finish_reason="tool_calls"),
        ]
    )
    client: LemonadeClient = make_client(completions)
    tools = [{"type": "function", "function": {"name": "search"}}]

    events = [
        e async for e in client.stream_chat([{"role": "user", "content": "hi"}], "m", tools=tools)
    ]

    assert completions.last_request["tools"] == tools
    assert isinstance(events[-1], Finished) and events[-1].finish_reason == "tool_calls"
    assert events[-2] == ToolCallsRequested((ToolCall("id1", "search", '{"query": "otters"}'),))


async def test_stream_chat_omits_tools_when_there_are_none():
    completions = FakeCompletions([make_chunk(content="hi", finish_reason="stop")])
    client = make_client(completions)

    [e async for e in client.stream_chat([], "m")]

    assert "tools" not in completions.last_request
