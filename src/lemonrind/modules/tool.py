"""A tool the model can call: a plain Python function plus the description the model reads.

To let a model call your code you must give it, for each tool, a *description* in JSON schema: the tool's
name, what it is for, and the name and type of every argument. Writing that by hand for every function is
tedious and easy to get out of step with the code, so ``tool_from_function`` **derives it from the function
itself**: the name, the docstring, the parameter names, the type hints and the defaults.

Python ideas used here:

* ``inspect`` - looks at a function while the program runs (its parameters, docstring, whether it is
  ``async``). The same idea as .NET reflection, but simpler.
* **Type hints are available at run time** (``typing.get_type_hints``), so ``city: str`` can be turned
  into ``{"type": "string"}``.
* ``pydantic.create_model`` - builds a model class *from data* while the program runs. We build one with a
  field per parameter, and then get two things for free: the JSON schema, and validation of whatever
  arguments the model sends.
* ``asyncio.to_thread`` - runs an ordinary (blocking) function on a worker thread so it cannot freeze the app.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
import typing
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

logger = logging.getLogger(__name__)


class ToolError(Exception):
    """Raise this inside a tool when something went wrong that the *model* should be told about.

    The message goes back to the model as the tool's result, so write it as a hint ("no such file;
    use list_files to see what exists"), not a stack trace. Any other exception is reported too, but
    with less friendly wording.
    """


@dataclass(frozen=True, slots=True)
class ToolResult:
    """What a tool run produced: text for the model, and whether it counts as a failure."""

    content: str
    is_error: bool = False


@dataclass(frozen=True, slots=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema of the arguments
    func: Callable[..., Any]
    # The model that validates the arguments. Tools built from a Python function have one. A tool whose
    # description comes from somewhere else (an MCP server) has ``None``: its ``func`` then receives the
    # arguments dictionary as it is, and the server that owns the tool does the validating.
    args_model: type[BaseModel] | None = None
    # True for tools that can do things you may not want done without being asked (send an email, change
    # a file): the conversation then asks the front end for permission before every run.
    requires_approval: bool = False
    # Optional refinements of the permission rule. ``approval_check`` decides *per call* from the arguments (a
    # write into a folder you pre-approved needs no question), overriding ``requires_approval``. ``preview``
    # turns the arguments into the text shown in the permission question: far easier to judge than raw JSON.
    approval_check: Callable[[dict[str, Any]], bool] | None = None
    preview: Callable[[dict[str, Any]], str] | None = None

    def needs_approval(self, arguments: str) -> bool:
        """Must the user agree before a call with these raw JSON ``arguments`` runs?"""
        if self.approval_check is None:
            return self.requires_approval
        try:
            data = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError:
            return (
                True  # cannot judge it, so ask (the run itself will complain about the arguments)
            )
        return self.approval_check(data) if isinstance(data, dict) else True

    def describe(self, arguments: str) -> str:
        """Text for the permission question: the tool's own preview, or the arguments laid out for reading."""
        try:
            data = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError:
            return arguments
        if self.preview is not None and isinstance(data, dict):
            try:
                return self.preview(data)
            except Exception:  # a broken preview must not stop the question being asked
                logger.debug("The preview of tool %s failed", self.name, exc_info=True)
        return json.dumps(data, indent=2, ensure_ascii=False)

    def schema(self) -> dict[str, Any]:
        """The description in the shape the OpenAI chat API wants in its ``tools`` list."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    async def run(self, arguments: str) -> ToolResult:
        """Run the tool with the model's raw JSON ``arguments``. Never raises (except for cancellation).

        Every kind of failure becomes a ``ToolResult`` with ``is_error=True``: the model reads it and can
        correct itself (fix an argument, try another approach) instead of the whole chat crashing.
        """
        try:
            data = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError as error:
            return ToolResult(f"The arguments were not valid JSON ({error}).", True)
        if not isinstance(data, dict):
            return ToolResult("The arguments must be a JSON object.", True)

        if self.args_model is None:
            positional: tuple[Any, ...] = (data,)
            kwargs: dict[str, Any] = {}
        else:
            try:
                validated = self.args_model.model_validate(data)
            except ValidationError as error:
                return ToolResult(f"Invalid arguments: {_describe_validation_error(error)}", True)
            positional = ()
            kwargs = {name: getattr(validated, name) for name in self.args_model.model_fields}

        try:
            if inspect.iscoroutinefunction(self.func):
                value = await self.func(*positional, **kwargs)
            else:
                value = await asyncio.to_thread(self.func, *positional, **kwargs)
        except ToolError as error:
            return ToolResult(str(error), True)
        except Exception as error:  # a bug or an unexpected failure inside the tool
            return ToolResult(f"The tool failed: {type(error).__name__}: {error}", True)
        # (asyncio.CancelledError is not an Exception subclass, so Stop still gets through.)
        return ToolResult(value if isinstance(value, str) else json.dumps(value, default=str))


def tool_from_function(
    func: Callable[..., Any], *, name: str | None = None, description: str | None = None
) -> Tool:
    """Build a ``Tool`` from a function, using its signature and docstring.

    Docstring format (Google style)::

        def get_weather(city: str, units: str = "metric") -> str:
            \"\"\"Look up the current weather for a city.

            Args:
                city: Name of the city, e.g. "Paris".
                units: "metric" or "imperial".
            \"\"\"

    The text before ``Args:`` is the tool's description, and each ``name: text`` line under ``Args:``
    describes that argument. A parameter without a default value is required.
    """
    doc = inspect.getdoc(func) or ""
    summary, argument_docs = _parse_docstring(doc)
    hints = typing.get_type_hints(func)

    fields: dict[str, Any] = {}
    for parameter_name, parameter in inspect.signature(func).parameters.items():
        annotation = hints.get(parameter_name, str)
        default = ... if parameter.default is inspect.Parameter.empty else parameter.default
        fields[parameter_name] = (
            annotation,
            Field(default, description=argument_docs.get(parameter_name)),
        )
    model = create_model(
        f"{func.__name__}_arguments", __config__=ConfigDict(extra="ignore"), **fields
    )

    return Tool(
        name=name or func.__name__,
        description=description or summary or func.__name__,
        parameters=_clean_schema(model.model_json_schema()),
        func=func,
        args_model=model,
    )


def _parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """Split a docstring into its description and the per-argument descriptions."""
    parts = re.split(r"(?m)^(?:Args|Arguments):\s*$", doc, maxsplit=1)
    head = parts[0]
    tail = parts[1] if len(parts) > 1 else ""
    descriptions: dict[str, str] = {}
    current: str | None = None
    for line in tail.splitlines():
        if not line.strip():
            continue
        if not line.startswith((" ", "\t")):  # a new unindented section ends the Args block
            break
        match = re.match(r"\s+(\w+)(?:\s*\([^)]*\))?:\s*(.*)", line)
        if match:
            current = match.group(1)
            descriptions[current] = match.group(2).strip()
        elif current:  # a wrapped continuation line
            descriptions[current] += " " + line.strip()
    return " ".join(head.split()), descriptions


def _clean_schema(schema: Any) -> Any:
    """Remove the ``title`` entries pydantic adds (they only cost the model tokens).

    The names inside ``properties`` are the tool's argument names, so an argument that happens to be
    called "title" must survive: only *schema keywords* named title are dropped.
    """
    if isinstance(schema, dict):
        return {
            key: (
                {name: _clean_schema(sub) for name, sub in value.items()}
                if key == "properties"
                else _clean_schema(value)
            )
            for key, value in schema.items()
            if key != "title"
        }
    if isinstance(schema, list):
        return [_clean_schema(item) for item in schema]
    return schema


def _describe_validation_error(error: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in item['loc']) or 'arguments'}: {item['msg']}"
        for item in error.errors()
    )
