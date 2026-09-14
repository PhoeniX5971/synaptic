"""Tests for Tool, ToolCall, autotool, and tool registries."""
import asyncio
import pytest

import synaptic.core.tool as tool_module
from synaptic.core.tool import TOOL_REGISTRY, Tool, ToolCall, ToolRegistry, autotool, register_callback


@pytest.fixture(autouse=True)
def clean_registry():
    saved_registry = dict(TOOL_REGISTRY)
    callbacks = tool_module._default_registry._callbacks
    saved_callbacks = list(callbacks)
    yield
    TOOL_REGISTRY.clear()
    TOOL_REGISTRY.update(saved_registry)
    callbacks.clear()
    callbacks.extend(saved_callbacks)


class TestToolCall:
    def test_attributes(self):
        tc = ToolCall(name="add", args={"a": 1, "b": 2})
        assert tc.name == "add"
        assert tc.args == {"a": 1, "b": 2}

    def test_get_arg(self):
        tc = ToolCall(name="x", args={"k": 99})
        assert tc.get_arg("k") == 99
        assert tc.get_arg("missing") is None

    def test_list_args(self):
        tc = ToolCall(name="x", args={"a": 1, "b": 2})
        assert set(tc.list_args()) == {"a", "b"}

    def test_repr(self):
        tc = ToolCall(name="fn", args={"x": 5})
        assert "fn" in repr(tc)


class TestToolSync:
    def test_run(self):
        t = Tool(
            name="add",
            declaration={"name": "add", "description": "", "parameters": {}},
            function=lambda a, b: a + b,
            add_to_registry=False,
        )
        assert t.run(a=3, b=4) == 7

    def test_default_params_merged(self):
        t = Tool(
            name="greet",
            declaration={"name": "greet", "description": "", "parameters": {}},
            function=lambda name="World": f"Hello {name}",
            default_params={"name": "World"},
            add_to_registry=False,
        )
        assert t.run() == "Hello World"
        assert t.run(name="Alice") == "Hello Alice"

    def test_is_async_false(self):
        t = Tool(
            name="sync",
            declaration={},
            function=lambda: None,
            add_to_registry=False,
        )
        assert t.is_async is False

    def test_adds_to_registry_by_default(self):
        assert "reg_tool" not in TOOL_REGISTRY
        Tool(
            name="reg_tool",
            declaration={"name": "reg_tool", "description": "", "parameters": {}},
            function=lambda: None,
        )
        assert "reg_tool" in TOOL_REGISTRY

    def test_not_added_when_disabled(self):
        Tool(
            name="no_reg",
            declaration={},
            function=lambda: None,
            add_to_registry=False,
        )
        assert "no_reg" not in TOOL_REGISTRY

    def test_non_callable_raises(self):
        with pytest.raises(ValueError):
            Tool(name="bad", declaration={}, function="not_callable")  # type: ignore


class TestToolAsync:
    def test_is_async_true(self):
        async def fn():
            return 42

        t = Tool(name="async_fn", declaration={}, function=fn, add_to_registry=False)
        assert t.is_async is True

    def test_run_async(self):
        async def double(x):
            return x * 2

        t = Tool(name="double", declaration={}, function=double, add_to_registry=False)
        result = asyncio.run(t.run(x=5))
        assert result == 10


class TestRegistryCallbacks:
    def test_callback_fired_on_registration(self):
        fired = []
        register_callback(lambda: fired.append(1))
        Tool(name="cb_tool", declaration={}, function=lambda: None)
        assert fired == [1]

    def test_multiple_callbacks(self):
        counts = [0, 0]
        register_callback(lambda: counts.__setitem__(0, counts[0] + 1))
        register_callback(lambda: counts.__setitem__(1, counts[1] + 1))
        Tool(name="multi_tool", declaration={}, function=lambda: None)
        assert counts == [1, 1]

    def test_scoped_registry_callback(self):
        registry = ToolRegistry()
        fired = []
        registry.on_change(lambda: fired.append(1))
        Tool(name="scoped", declaration={}, function=lambda: None, registry=registry)
        assert fired == [1]
        assert "scoped" in registry.all()
        assert "scoped" not in TOOL_REGISTRY

class TestAutotool:
    def test_generates_correct_schema(self):
        @autotool(
            description="Add two numbers",
            param_descriptions={"a": "first", "b": "second"},
            autobind=False,
        )
        def add(a: int, b: int) -> int:
            return a + b

        decl = add.declaration
        assert decl["name"] == "add"
        assert decl["description"] == "Add two numbers"
        assert decl["parameters"]["properties"]["a"]["type"] == "integer"
        assert decl["parameters"]["properties"]["b"]["description"] == "second"
        assert "a" in decl["parameters"]["required"]
        assert "b" in decl["parameters"]["required"]

    def test_string_type(self):
        @autotool(description="greet", autobind=False)
        def greet(name: str) -> str:
            return f"Hi {name}"

        assert greet.declaration["parameters"]["properties"]["name"]["type"] == "string"

    def test_float_type(self):
        @autotool(description="scale", autobind=False)
        def scale(x: float) -> float:
            return x

        assert scale.declaration["parameters"]["properties"]["x"]["type"] == "number"

    def test_bool_type(self):
        @autotool(description="flag", autobind=False)
        def flag(on: bool) -> bool:
            return on

        assert flag.declaration["parameters"]["properties"]["on"]["type"] == "boolean"

    def test_optional_param_not_required(self):
        @autotool(description="opt", autobind=False)
        def opt(required: str, optional: str = "default") -> str:
            return required

        required_list = opt.declaration["parameters"].get("required", [])
        assert "required" in required_list
        assert "optional" not in required_list

    def test_tool_is_callable(self):
        @autotool(description="mul", autobind=False)
        def mul(a: int, b: int) -> int:
            return a * b

        assert mul.run(a=3, b=4) == 12
