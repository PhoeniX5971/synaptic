import pytest

from synaptic import Tool, ToolRegistry


def test_read_only_snapshot_disposal_and_reentrancy():
    errors, seen = [], []
    registry = ToolRegistry(on_observer_error=errors.append)
    snapshot = registry.all()
    subscription = registry.on_change(lambda: seen.append(registry.version))
    bad = registry.on_change(lambda: registry.register(Tool("nested", {}, add_to_registry=False)))
    registry.register(Tool("one", {}, add_to_registry=False))
    assert seen == [1]
    assert len(errors) == 1
    assert "nested" not in registry.all()
    assert snapshot == {}
    with pytest.raises(TypeError):
        registry.all()["injected"] = None
    subscription.close()
    subscription.close()
    bad.close()
    registry.register(Tool("two", {}, add_to_registry=False))
    assert seen == [1]


def test_owned_alias_cannot_be_rebound_by_another_instance():
    registry = ToolRegistry()
    tool = Tool("same", {}, add_to_registry=False)
    registry.replace_owned("owner-a", [tool])
    with pytest.raises(ValueError):
        registry.replace_owned("owner-b", [tool])
    with pytest.raises(ValueError):
        registry.register(tool)
    assert registry.all()["same"] is tool
