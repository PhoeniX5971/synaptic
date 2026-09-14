from unittest.mock import patch

import pytest

from synaptic import Model, Provider, Tool, ToolRegistry


def tool(description):
    return Tool(
        "echo",
        {
            "name": "echo",
            "description": description,
            "parameters": {"type": "object", "properties": {}},
        },
        add_to_registry=False,
    )


def test_binding_replacement_and_registry_removal_update_provider_schema():
    with (
        patch("synaptic.providers.openai_.model.OpenAI"),
        patch("synaptic.providers.openai_.model.AsyncOpenAI"),
    ):
        registry = ToolRegistry()
        model = Model(Provider.OPENAI, "fixture", tool_registry=registry)
    first, replacement = tool("first"), tool("replacement")
    model.bind_tools([first, first])
    assert len(model.tools) == 1
    with pytest.raises(ValueError, match="replace=True"):
        model.bind_tools([replacement])
    assert model.llm.openai_tools[0]["function"]["description"] == "first"
    model.bind_tools([replacement], replace=True)
    assert model.llm.openai_tools[0]["function"]["description"] == "replacement"
    model.unbind_tools("echo")
    assert model.llm.openai_tools == []
    registry.register(first)
    assert model.tools == [first]
    # Binding an unrelated tool must not freeze the registry's merged view.
    other = Tool("other", {"name": "other"}, add_to_registry=False)
    model.bind_tools([other])
    registry.unregister("echo")
    assert model.tools == [other]
    assert [t["function"]["name"] for t in model.llm.openai_tools] == ["other"]
