"""A short checklist the assistant keeps while it works through a task with several steps.

A model working through many tool calls tends to lose the thread: it repeats a step, skips one, or stops after the
second and writes as if the job were done. Asked to write its steps down and tick them off, it stays on track
better, and you can see at a glance how far it has got.

The tool is stateless: each call carries the whole plan, so the plan lives in the conversation itself (the model
sees its own earlier calls) and nothing has to be kept per chat. The screens read the same call to draw the
checklist, and a scheduled job reads it to see whether the model stopped with steps left.

Python ideas used here:

* A frozen dataclass with ``slots`` for a small immutable value, and ``@property`` for answers worked out from
  its fields.
* ``tuple[int, ...]`` as a parameter type: the tool's JSON schema then says "an array of whole numbers".
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from lemonrind.modules.base import Module
from lemonrind.modules.tool import Tool, ToolError, tool_from_function

PLAN_TOOL = "update_plan"
MAX_STEPS = 12
MAX_STEP_LENGTH = 200
# A job that tries to finish with steps left is sent back to them this many times, then left alone.
MAX_PLAN_NUDGES = 3
# After this many rounds of only editing the plan, tools are withheld so the model has to answer in words.
MAX_PLAN_ONLY_ROUNDS = 8


@dataclass(frozen=True, slots=True)
class Plan:
    steps: tuple[str, ...]
    done: frozenset[int]  # step numbers, counting from 1
    current: int  # the step being worked on, or 0 for none

    @property
    def unfinished(self) -> list[int]:
        return [n for n in range(1, len(self.steps) + 1) if n not in self.done]

    @property
    def complete(self) -> bool:
        return not self.unfinished

    def status(self, number: int) -> str:
        if number in self.done:
            return "done"
        return "now" if number == self.current else "todo"

    def render(self) -> str:
        """The plan as text: what the model gets back from the tool, and what is saved in the chat."""
        marks = {"done": "[done]", "now": "[now]", "todo": "[ ]"}
        lines = [f"Plan ({len(self.done)} of {len(self.steps)} done):"]
        lines += [f"{n}. {marks[self.status(n)]} {text}" for n, text in enumerate(self.steps, 1)]
        return "\n".join(lines)


def build_plan(steps: list[str], done: tuple[int, ...] = (), current: int = 0) -> Plan:
    """Check what the model sent and make a Plan of it. Raises ``ToolError`` with a message the model can act on."""
    cleaned = [" ".join(step.split()) for step in steps]
    if not cleaned or any(not step for step in cleaned):
        raise ToolError("Give at least one step, and no empty steps.")
    if len(cleaned) > MAX_STEPS:
        raise ToolError(f"Keep the plan to {MAX_STEPS} steps or fewer: join small steps together.")
    cleaned = [step[:MAX_STEP_LENGTH] for step in cleaned]
    count = len(cleaned)
    bad = [n for n in done if not 1 <= n <= count]
    if bad or not 0 <= current <= count:
        raise ToolError(
            f"Step numbers count from 1 to {count}. 'done' lists finished steps; 'current' is the step "
            "you are on now (0 for none)."
        )
    return Plan(tuple(cleaned), frozenset(done), current)


def plan_from_arguments(arguments: str) -> Plan | None:
    """The plan inside the JSON arguments of an ``update_plan`` call, or None if they do not make one."""
    try:
        data = json.loads(arguments)
        return build_plan(
            [str(step) for step in data["steps"]],
            tuple(int(n) for n in data.get("done") or ()),
            int(data.get("current") or 0),
        )
    except (ValueError, KeyError, TypeError, ToolError):
        return None


def plan_reminder(plan: Plan) -> str:
    """What a job that tried to finish early is told: the steps still open."""
    open_steps = ", ".join(str(n) for n in plan.unfinished)
    return (
        f"[Reminder] Your plan still has unfinished steps: {open_steps}. Carry on with them, and call "
        f"{PLAN_TOOL} as you finish each. If a step is no longer needed, call {PLAN_TOOL} with a shorter "
        "plan that says so, then give your final answer."
    )


class PlanModule(Module):
    name = "Plan"
    config_key = "plan"
    description = "Lets the assistant keep a short checklist while it works through a task with several steps."

    def get_tools(self) -> list[Tool]:
        return [tool_from_function(self.update_plan, name=PLAN_TOOL)]

    def update_plan(self, steps: list[str], done: tuple[int, ...] = (), current: int = 0) -> str:
        """Write down, or update, your plan for a task that needs several steps and tools. Use it only when there really are several steps, not for a simple question. Send the whole plan every time.

        Args:
            steps: Every step of the plan, in order, each a short sentence.
            done: The numbers (counting from 1) of the steps that are finished.
            current: The number of the step you are working on now, or 0 if none.
        """
        plan = build_plan(steps, done, current)
        if plan.complete:
            advice = "Every step is done. Give your final answer now."
        else:
            next_step = plan.current or plan.unfinished[0]
            advice = f"Carry on with step {next_step}, and update the plan as you finish each step."
        return f"{plan.render()}\n{advice}"
