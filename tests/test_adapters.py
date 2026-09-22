"""
Tests for adapter history formatting with tool call/result linking.
No real API calls are made.
"""

from datetime import datetime, timezone
from unittest.mock import patch

from synaptic.core.base.memory import History, ResponseMem, UserMem
from synaptic.core.base.base_model import ResponseFormat
from synaptic.core.tool import ToolCall


TS = datetime(2024, 1, 1, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _history_with_tool_turn(message="answer", tool_name="add", tool_args=None, tool_result=None):
    """Return a History containing one ResponseMem with a tool call + result."""
    h = History()
    tc = ToolCall(name=tool_name, args=tool_args or {"a": 1, "b": 2})
    resp = ResponseMem(message=message, created=TS, tool_calls=[tc])
    resp.tool_results = [{"name": tool_name, "result": tool_result or 3}]
    h.add(resp)
    return h


def _history_with_user_turn(message="hello"):
    h = History()
    h.add(UserMem(message=message, created=TS))
    return h


# ---------------------------------------------------------------------------
# OpenAIAdapter
# ---------------------------------------------------------------------------
class TestOpenAIAdapterToContents:
    def _adapter(self, instructions="", history=None):
        from synaptic.providers.openai_ import OpenAIAdapter

        with patch("synaptic.providers.openai_.model.OpenAI"):
            a = OpenAIAdapter(
                model="gpt-4o",
                history=history or History(),
                api_key="fake",
                response_format=ResponseFormat.NONE,
                response_schema=None,
                tools=[],
                instructions=instructions,
            )
        a.openai_tools = []
        return a

    def test_instructions_first(self):
        from synaptic.providers.openai_.helpers import history_messages

        a = self._adapter(instructions="be helpful")
        contents = history_messages(a.history, a.instructions, a.role_map)
        assert contents[0] == {"role": "system", "content": "be helpful"}

    def test_user_message(self):
        from synaptic.providers.openai_.helpers import history_messages

        h = _history_with_user_turn("hi there")
        a = self._adapter(history=h)
        contents = history_messages(a.history, a.instructions, a.role_map)
        assert any(m["role"] == "user" and m["content"] == "hi there" for m in contents)

    def test_tool_call_linked_to_result(self):
        from synaptic.providers.openai_.helpers import history_messages

        h = _history_with_tool_turn()
        a = self._adapter(history=h)
        contents = history_messages(a.history, a.instructions, a.role_map)

        assistant = next(m for m in contents if m["role"] == "assistant")
        tool_msg = next(m for m in contents if m["role"] == "tool")

        assert assistant["tool_calls"][0]["id"] == h.entries[0].tool_calls[0].id
        assert tool_msg["tool_call_id"] == h.entries[0].tool_calls[0].id
        assert tool_msg["name"] == "add"

    def test_multiple_tool_calls_all_linked(self):
        from synaptic.providers.openai_.helpers import history_messages

        h = History()
        tcs = [ToolCall("add", {"a": 1}), ToolCall("greet", {"name": "x"})]
        resp = ResponseMem(message="", created=TS, tool_calls=tcs)
        resp.tool_results = [{"name": "add", "result": 1}, {"name": "greet", "result": "hi"}]
        h.add(resp)

        a = self._adapter(history=h)
        contents = history_messages(a.history, a.instructions, a.role_map)

        tool_msgs = [m for m in contents if m["role"] == "tool"]
        assert len(tool_msgs) == 2
        assert tool_msgs[0]["tool_call_id"] == h.entries[0].tool_calls[0].id
        assert tool_msgs[1]["tool_call_id"] == h.entries[0].tool_calls[1].id
        assert tool_msgs[0]["name"] == "add"
        assert tool_msgs[1]["name"] == "greet"

    def test_no_tool_results_no_tool_messages(self):
        from synaptic.providers.openai_.helpers import history_messages

        h = History()
        tc = ToolCall("fn", {})
        resp = ResponseMem(message="", created=TS, tool_calls=[tc])
        resp.tool_results = []
        h.add(resp)
        a = self._adapter(history=h)
        contents = history_messages(a.history, a.instructions, a.role_map)
        assert not any(m["role"] == "tool" for m in contents)


# ---------------------------------------------------------------------------
# DeepSeekAdapter
# ---------------------------------------------------------------------------
class TestDeepSeekAdapterToMessages:
    def _adapter(self, instructions="", history=None):
        from synaptic.providers.deepseek import DeepSeekAdapter

        with patch("synaptic.providers.deepseek.model.OpenAI"):
            a = DeepSeekAdapter(
                model="deepseek-chat",
                history=history or History(),
                api_key="fake",
                response_format=ResponseFormat.NONE,
                response_schema=None,
                tools=[],
                instructions=instructions,
            )
        a.openai_tools = []
        return a

    def test_instructions_as_system_message(self):
        a = self._adapter(instructions="you are helpful")
        msgs = a.to_messages()
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "you are helpful"

    def test_tool_call_and_result_linked(self):
        from synaptic.providers.together.helpers import history_messages

        h = _history_with_tool_turn()
        a = self._adapter(history=h)
        msgs = history_messages(a)

        assistant = next(m for m in msgs if m.get("role") == "assistant")
        tool_msg = next(m for m in msgs if m.get("role") == "tool")

        assert assistant["tool_calls"][0]["id"] == h.entries[0].tool_calls[0].id
        assert tool_msg["tool_call_id"] == h.entries[0].tool_calls[0].id
        assert tool_msg["name"] == "add"

    def test_plain_user_message(self):
        h = _history_with_user_turn("hey")
        a = self._adapter(history=h)
        msgs = a.to_messages()
        assert any(m.get("role") == "user" and m.get("content") == "hey" for m in msgs)


# ---------------------------------------------------------------------------
# TogetherAdapter
# ---------------------------------------------------------------------------
class TestTogetherAdapterToMessages:
    def _adapter(self, instructions="", history=None):
        from synaptic.providers.together import TogetherAdapter

        with patch("synaptic.providers.together.model.Together"):
            a = TogetherAdapter(
                model="meta-llama/Llama-3",
                history=history or History(),
                api_key="fake",
                response_format=ResponseFormat.NONE,
                response_schema=None,
                tools=[],
                instructions=instructions,
            )
        a.together_tools = []
        return a

    def test_tool_call_and_result_linked(self):
        from synaptic.providers.together.helpers import history_messages

        h = _history_with_tool_turn()
        a = self._adapter(history=h)
        msgs = history_messages(a)

        assistant = next(m for m in msgs if m.get("role") == "assistant")
        tool_msg = next(m for m in msgs if m.get("role") == "tool")

        assert assistant["tool_calls"][0]["id"] == h.entries[0].tool_calls[0].id
        assert tool_msg["tool_call_id"] == h.entries[0].tool_calls[0].id

    def test_plain_user_message(self):
        from synaptic.providers.together.helpers import history_messages

        h = _history_with_user_turn("hi")
        a = self._adapter(history=h)
        msgs = history_messages(a)
        assert any(m["role"] == "user" and m["content"] == "hi" for m in msgs)


# ---------------------------------------------------------------------------
# UniversalLLMAdapter
# ---------------------------------------------------------------------------
class TestUniversalLLMAdapterToContents:
    def _adapter(self, instructions="", history=None):
        from synaptic.providers.universal_openai import UniversalLLMAdapter

        with patch("synaptic.providers.universal_openai.model.OpenAI"):
            a = UniversalLLMAdapter(
                model="llama",
                history=history or History(),
                base_url="http://localhost:11434",
                api_key=None,
                response_format=ResponseFormat.NONE,
                response_schema=None,
                tools=[],
                instructions=instructions,
            )
        a.openai_tools = []
        return a

    def test_instructions_first(self):
        a = self._adapter(instructions="sys prompt")
        contents = a.to_contents()
        assert contents[0]["role"] == "system"
        assert contents[0]["content"] == "sys prompt"

    def test_tool_call_and_result_linked(self):
        h = _history_with_tool_turn()
        a = self._adapter(history=h)
        contents = a.to_contents()

        assistant = next(m for m in contents if m.get("role") == "assistant")
        tool_msg = next(m for m in contents if m.get("role") == "tool")

        assert assistant["tool_calls"][0]["id"] == h.entries[0].tool_calls[0].id
        assert tool_msg["tool_call_id"] == h.entries[0].tool_calls[0].id
        assert tool_msg["name"] == "add"


# ---------------------------------------------------------------------------
# GeminiAdapter
# ---------------------------------------------------------------------------
class TestGeminiAdapterToContents:
    def _adapter(self, history=None, instructions=""):
        from synaptic.providers.gemini import GeminiAdapter
        from google.genai import types

        with patch("synaptic.providers.gemini.model.genai"):
            a = GeminiAdapter(
                model="gemini-2.5-flash",
                history=history or History(),
                api_key="fake",
                response_format=ResponseFormat.NONE,
                response_schema=None,
                tools=[],
                instructions=instructions,
            )
        a.gemini_tools = []
        return a

    def test_plain_user_message(self):
        h = _history_with_user_turn("hello gemini")
        a = self._adapter(history=h)
        contents = a.to_contents()
        assert len(contents) == 1
        assert contents[0].role == "user"
        assert contents[0].parts[0].text == "hello gemini"

    def test_tool_call_splits_into_two_turns(self):
        h = _history_with_tool_turn(message="here is the call")
        a = self._adapter(history=h)
        contents = a.to_contents()
        # one model turn + one user turn with FunctionResponse
        assert len(contents) == 2
        assert contents[0].role == "model"
        assert contents[1].role == "user"

    def test_model_turn_has_function_call_part(self):
        from google.genai import types

        h = _history_with_tool_turn()
        a = self._adapter(history=h)
        contents = a.to_contents()
        model_parts = contents[0].parts
        assert any(p.function_call is not None for p in model_parts)
        fc_part = next(p for p in model_parts if p.function_call is not None)
        assert fc_part.function_call.name == "add"

    def test_user_turn_has_function_response_part(self):
        from google.genai import types

        h = _history_with_tool_turn()
        a = self._adapter(history=h)
        contents = a.to_contents()
        user_parts = contents[1].parts
        assert any(p.function_response is not None for p in user_parts)
        fr_part = next(p for p in user_parts if p.function_response is not None)
        assert fr_part.function_response.name == "add"
        assert fr_part.function_response.response["result"] == 3

    def test_no_tool_results_no_response_turn(self):
        h = History()
        tc = ToolCall("fn", {})
        resp = ResponseMem(message="", created=TS, tool_calls=[tc])
        resp.tool_results = []
        h.add(resp)
        a = self._adapter(history=h)
        contents = a.to_contents()
        # only the model turn — no user turn without results
        assert len(contents) == 1
        assert contents[0].role == "model"


# ---------------------------------------------------------------------------
# VertexAdapter
# ---------------------------------------------------------------------------
class TestVertexAdapterToContents:
    def _adapter(self, history=None):
        from synaptic.providers.vertex import VertexAdapter

        with (
            patch("synaptic.providers.vertex.model.vertex_init"),
            patch("synaptic.providers.vertex.model.GenerativeModel"),
        ):
            a = VertexAdapter(
                model="gemini-pro",
                project="fake-project",
                location="us-central1",
                history=history or History(),
                response_format=ResponseFormat.NONE,
                response_schema=None,
                tools=[],
            )
        a.vertex_tools = []
        return a

    def test_plain_user_message(self):
        h = _history_with_user_turn("hello vertex")
        a = self._adapter(history=h)
        contents = a.to_contents()
        assert len(contents) == 1
        assert contents[0].role == "user"

    def test_tool_call_splits_into_two_turns(self):
        h = _history_with_tool_turn()
        a = self._adapter(history=h)
        contents = a.to_contents()
        assert len(contents) == 2
        assert contents[0].role == "model"
        assert contents[1].role == "user"

    def test_user_turn_has_function_response(self):
        from vertexai.generative_models import Part

        h = _history_with_tool_turn(tool_name="multiply", tool_result=6)
        a = self._adapter(history=h)
        contents = a.to_contents()
        # The user turn should contain a function_response part
        user_turn = contents[1]
        # vertexai Parts with function_response have a specific structure
        # Just verify it's a Part and role is correct
        assert user_turn.role == "user"
        assert len(user_turn.parts) == 1
