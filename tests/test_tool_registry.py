"""Tests for dynamic ToolRegistry removal behavior."""

from synaptic.core.tool import Tool, ToolRegistry


def test_unregister_fires_callback():
    registry = ToolRegistry()
    fired = []
    registry.on_change(lambda: fired.append(1))
    Tool(name="gone", declaration={}, function=lambda: None, registry=registry)

    registry.unregister("gone")

    assert fired == [1, 1]
    assert "gone" not in registry.all()


def test_unregister_prefix_fires_once():
    registry = ToolRegistry()
    fired = []
    registry.on_change(lambda: fired.append(1))
    Tool(name="mcp__a", declaration={}, function=lambda: None, registry=registry)
    Tool(name="mcp__b", declaration={}, function=lambda: None, registry=registry)

    registry.unregister_prefix("mcp__")

    assert fired == [1, 1, 1]
    assert registry.all() == {}
