import asyncio
from enum import Enum
from typing import Literal, Callable
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from synaptic import Agent, EventBus, Tool, ToolCall, ToolRegistry, autotool, LocalExecutor
from synaptic.agent import _execution


class Mode(str, Enum):
    READ = "read"


class Target(BaseModel):
    path: str


def test_nested_postponed_union_literal_enum_schema_and_validation():
    @autotool(autobind=False)
    def rich(
        target: Target,
        labels: list[dict[str, int]],
        mode: Mode = Mode.READ,
        limit: int | None = None,
        choice: Literal["a", "b"] = "a",
    ) -> str:
        return target.path

    schema = rich.declaration["parameters"]
    assert "$defs" in schema
    invocation = rich.prepare(ToolCall("rich", {"target": {"path": "ok"}, "labels": [{"x": 1}]}))
    assert invocation.arguments["mode"] == "read"
    assert LocalExecutor().execute(invocation) == "ok"
    with pytest.raises(Exception):
        rich.run(target={"path": "ok"}, labels=[{"x": "invalid"}])


def test_unsupported_annotation_fails_and_outputs_are_validated():
    with pytest.raises(TypeError):

        @autotool(autobind=False)
        def bad(value: Callable[[str], str]):
            return value

    @autotool(autobind=False)
    def invalid() -> int:
        return "not an integer"

    with pytest.raises(Exception):
        LocalExecutor().execute(invalid.prepare(ToolCall("invalid", {})))


@pytest.mark.parametrize("mode", ["sync", "async", "external"])
def test_policy_and_effect_arguments_remain_equal_under_mutation_attempts(mode):
    observed, executed = [], []
    tool = Tool(
        "write",
        {"parameters": {"type": "object"}},
        lambda path, nested: executed.append((path, nested)),
        default_params={"path": "allowed", "nested": {"x": [1]}},
        add_to_registry=False,
    )
    agent = Agent.__new__(Agent)
    agent.events, agent.signals = EventBus(), None
    agent._registry = ToolRegistry()
    agent._registry.register(tool)
    agent.model = SimpleNamespace(blacklist=[])

    def policy(name, args):
        observed.append(args)
        with pytest.raises(TypeError):
            args["path"] = "forbidden"
        with pytest.raises(TypeError):
            args["nested"]["x"][0] = 999
        tool.default_params["path"] = "changed"
        return True

    agent.permission = policy
    agent.executor = LocalExecutor()
    call = ToolCall("write", {})
    agent.events.on("tool_start", lambda proposal: proposal.args.update(path="forbidden"))
    if mode == "sync":
        result = _execution.execute(agent, call)
    elif mode == "async":
        result = asyncio.run(_execution.aexecute(agent, call))
    else:

        class External:
            def execute(self, invocation):
                return invocation.execute()

        agent.executor = External()
        result = _execution.execute(agent, call)
    assert result["call_id"] == call.id
    assert observed[0]["path"] == executed[0][0] == "allowed"
    assert executed[0][1] == {"x": [1]}


def test_definition_snapshots_and_argument_overrides():
    tool = Tool("echo", {"parameters": {"type": "object"}}, lambda x=1: x, add_to_registry=False)
    snapshot = tool.prepare(ToolCall("echo", {"x": 2}))
    tool.declaration["parameters"]["type"] = "string"
    with pytest.raises(AttributeError):
        tool.function = lambda **kwargs: 0
    assert LocalExecutor().execute(snapshot) == 2
