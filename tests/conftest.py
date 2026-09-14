"""Shared fixtures for all tests."""
import pytest
import synaptic.core.tool as tool_module
from synaptic.core.tool import TOOL_REGISTRY


@pytest.fixture(autouse=True)
def isolate_tool_registry():
    """
    Saves and restores global tool state around every test.
    """
    saved_registry = dict(TOOL_REGISTRY)
    callbacks = tool_module._default_registry._callbacks
    saved_callbacks = list(callbacks)
    yield
    TOOL_REGISTRY.clear()
    TOOL_REGISTRY.update(saved_registry)
    callbacks.clear()
    callbacks.extend(saved_callbacks)
