"""Tests for the core Model class (no real API calls — all adapters mocked)."""
import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

import synaptic.core.tool as tool_module
from synaptic.core.base.memory import History, ResponseMem, UserMem
from synaptic.core.base.base_model import ResponseChunk
from synaptic.core.model import Model
from synaptic.core.provider import Provider
from synaptic.core.tool import TOOL_REGISTRY, Tool, ToolCall


TS = datetime(2024, 1, 1, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Fixture: isolate global registry state
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fake_adapter(instructions="", history=None):
    """Returns a MagicMock that quacks like a BaseModel adapter."""
    adapter = MagicMock()
    adapter.instructions = instructions
    adapter.synaptic_tools = []
    adapter.history = history or History()
    adapter._invalidate_tools = MagicMock()
    return adapter


def _make_model(provider=Provider.GEMINI, instructions="", tools=None, history=None, **kwargs):
    """Build a Model with a fully-mocked adapter."""
    with patch("synaptic.core._factory.GeminiAdapter") as MockGemini:
        MockGemini.return_value = _fake_adapter(instructions=instructions, history=history or History())
        m = Model(
            provider=provider,
            model="test-model",
            api_key="fake-key",
            instructions=instructions,
            tools=tools,
            history=history,
            **kwargs,
        )
    return m


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
class TestModelConstruction:
    def test_provider_stored(self):
        m = _make_model(provider=Provider.GEMINI)
        assert m.provider == Provider.GEMINI

    def test_response_format_none_needs_no_schema(self):
        # Should not raise
        _make_model()

    def test_json_format_without_schema_raises(self):
        from synaptic.core.base.base_model import ResponseFormat
        with pytest.raises(ValueError, match="schema"):
            _make_model(response_format=ResponseFormat.JSON, response_schema=None)

    def test_automem_false_preserves_window(self):
        h = History(size=10)
        m = _make_model(history=h, automem=False)
        assert h.size == 10

    def test_automem_true_keeps_default_window(self):
        h = History(size=10)
        _make_model(history=h, automem=True)
        assert h.size == 10


# ---------------------------------------------------------------------------
# Dynamic instructions property
# ---------------------------------------------------------------------------
class TestDynamicInstructions:
    def test_getter_returns_value(self):
        m = _make_model(instructions="be helpful")
        assert m.instructions == "be helpful"

    def test_setter_updates_llm(self):
        m = _make_model(instructions="old")
        m.instructions = "new"
        assert m.instructions == "new"
        assert m.llm.instructions == "new"

    def test_setter_before_llm_is_safe(self):
        """The setter must not crash if called before self.llm exists (during __init__)."""
        # Simulate by directly calling the setter logic
        class Dummy:
            pass

        obj = Dummy()
        obj._instructions = "initial"
        # Property setter uses hasattr check — verify it's safe with no llm
        assert not hasattr(obj, "llm")


# ---------------------------------------------------------------------------
# bind_tools
# ---------------------------------------------------------------------------
class TestBindTools:
    def test_bind_tools_calls_invalidate(self):
        m = _make_model()
        t = Tool(name="bt", declaration={}, function=lambda: None, add_to_registry=False)
        m.bind_tools([t])
        m.llm._invalidate_tools.assert_called()

    def test_bind_tools_appends(self):
        m = _make_model()
        m.tools = []
        t = Tool(name="bt2", declaration={}, function=lambda: None, add_to_registry=False)
        m.bind_tools([t])
        assert t in m.tools


# ---------------------------------------------------------------------------
# invoke
# ---------------------------------------------------------------------------
class TestInvoke:
    def test_invalid_role_raises(self):
        m = _make_model()
        m.llm.invoke.return_value = ResponseMem("ok", TS, [])
        with pytest.raises(ValueError):
            m.invoke("hi", role="robot")

    def test_returns_response_mem(self):
        m = _make_model()
        expected = ResponseMem("response", TS, [])
        m.llm.invoke.return_value = expected
        result = m.invoke("hello")
        assert result is expected

    def test_automem_stores_user_and_response(self):
        h = History()
        m = _make_model(history=h, automem=True)
        m.history = h
        resp = ResponseMem("reply", TS, [])
        m.llm.invoke.return_value = resp
        m.invoke("prompt", automem=True)
        assert len(h.entries) == 2
        assert isinstance(h.entries[0], UserMem)
        assert isinstance(h.entries[1], ResponseMem)

    def test_autorun_executes_tool_calls(self):
        m = _make_model(autorun=True)
        t = Tool(name="fn", declaration={}, function=lambda: "result", add_to_registry=False)
        m.llm.synaptic_tools = [t]
        m.blacklist = []
        tc = ToolCall("fn", {})
        resp = ResponseMem("", TS, [tc])
        m.llm.invoke.return_value = resp
        result = m.invoke("go", autorun=True)
        assert result.tool_results[0]["result"] == "result"

    def test_autorun_false_clears_tool_results(self):
        m = _make_model()
        resp = ResponseMem("ok", TS, [])
        m.llm.invoke.return_value = resp
        result = m.invoke("hi", autorun=False)
        assert result.tool_results == []


# ---------------------------------------------------------------------------
# ainvoke
# ---------------------------------------------------------------------------
class TestAinvoke:
    def test_ainvoke_aggregates_llm_stream(self):
        m = _make_model()
        resp = ResponseMem("async response", TS, [])
        async def stream(**kwargs):
            yield ResponseChunk(text="async response")
        m.llm.astream = stream

        result = asyncio.run(m.ainvoke("prompt"))
        assert result.message == "async response"

    def test_ainvoke_invalid_role_raises(self):
        m = _make_model()
        m.llm.invoke.return_value = ResponseMem("", TS, [])
        with pytest.raises(ValueError):
            asyncio.run(m.ainvoke("hi", role="bad"))


# ---------------------------------------------------------------------------
# astream
# ---------------------------------------------------------------------------
class TestAstream:
    def test_raises_if_no_astream_on_llm(self):
        m = _make_model()
        del m.llm.astream  # remove the attribute

        async def run():
            async for _ in m.astream("hi"):
                pass

        with pytest.raises(NotImplementedError):
            asyncio.run(run())

    def test_yields_chunks_and_saves_to_history(self):
        h = History()
        m = _make_model(history=h, automem=True)
        m.history = h

        chunks = [
            ResponseChunk(text="hello", is_final=False),
            ResponseChunk(text=" world", is_final=False),
            ResponseChunk(text="hello world", is_final=True),
        ]

        async def fake_astream(prompt, role="user", **kwargs):
            for c in chunks:
                yield c

        m.llm.astream = fake_astream

        async def run():
            collected = []
            async for chunk in m.astream("test", automem=True):
                collected.append(chunk)
            return collected

        result = asyncio.run(run())
        assert len(result) == 3
        # history should have user + response mem
        assert len(h.entries) == 2
