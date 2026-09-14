"""Tests for core memory classes: Memory, UserMem, ResponseMem, History."""
import pytest
from datetime import datetime, timezone

from synaptic.core.base.memory import History, Memory, ResponseMem, UserMem
from synaptic.core.tool import ToolCall


TS = datetime(2024, 1, 1, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------
class TestMemory:
    def test_attributes(self):
        m = Memory(message="hello", created=TS, role="user")
        assert m.message == "hello"
        assert m.created == TS
        assert m.role == "user"

    def test_repr(self):
        m = Memory(message="hi", created=TS, role="assistant")
        assert "assistant" in repr(m)
        assert "hi" in repr(m)


# ---------------------------------------------------------------------------
# UserMem
# ---------------------------------------------------------------------------
class TestUserMem:
    def test_default_role(self):
        m = UserMem(message="ping", created=TS)
        assert m.role == "user"

    def test_custom_role(self):
        m = UserMem(message="ping", created=TS, role="system")
        assert m.role == "system"


# ---------------------------------------------------------------------------
# ResponseMem
# ---------------------------------------------------------------------------
class TestResponseMem:
    def test_default_role(self):
        m = ResponseMem(message="pong", created=TS, tool_calls=[])
        assert m.role == "assistant"

    def test_tool_calls_stored(self):
        tc = ToolCall(name="add", args={"a": 1, "b": 2})
        m = ResponseMem(message="", created=TS, tool_calls=[tc])
        assert len(m.tool_calls) == 1
        assert m.tool_calls[0].name == "add"

    def test_list_tool_calls(self):
        tcs = [ToolCall("add", {}), ToolCall("greet", {})]
        m = ResponseMem(message="", created=TS, tool_calls=tcs)
        assert m.list_tool_calls() == ["add", "greet"]

    def test_get_tool_call(self):
        tc = ToolCall("add", {"a": 5})
        m = ResponseMem(message="", created=TS, tool_calls=[tc])
        found = m.get_tool_call("add")
        assert found.args == {"a": 5}

    def test_get_tool_call_missing_raises(self):
        m = ResponseMem(message="", created=TS, tool_calls=[])
        with pytest.raises(IndexError):
            m.get_tool_call("nonexistent")

    def test_tool_results_default_empty(self):
        m = ResponseMem(message="", created=TS, tool_calls=[])
        assert m.tool_results == []

    def test_repr_contains_role(self):
        m = ResponseMem(message="ok", created=TS, tool_calls=[])
        assert "assistant" in repr(m)


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------
class TestHistory:
    def test_starts_empty(self):
        h = History()
        assert h.entries == []

    def test_add(self):
        h = History()
        h.add(UserMem(message="a", created=TS))
        assert len(h.entries) == 1

    def test_rolling_window_trims_oldest(self):
        h = History(size=3)
        for i in range(5):
            h.add(UserMem(message=str(i), created=TS))
        assert len(h.entries) == 3
        # oldest messages 0 and 1 were dropped
        assert h.entries[0].message == "2"

    def test_window_resize(self):
        h = History(size=10)
        for i in range(10):
            h.add(UserMem(message=str(i), created=TS))
        h.window(3)
        assert len(h.entries) == 3

    def test_initial_list(self):
        m = UserMem(message="seed", created=TS)
        h = History(memory_list=[m])
        assert len(h.entries) == 1
        assert h.entries[0].message == "seed"
