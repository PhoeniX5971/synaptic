"""Tool timing reaches observers and journals; provider JSON keeps non-ASCII literal."""

import asyncio
import json
import time

import pytest

import synaptic
from synaptic import Agent, EventBus, History, Model, Provider, ResponseMem, Tool, ToolCall
from synaptic.core import _json
from synaptic.core._tool_execution import run_tools_sync
from synaptic.providers._chat_history import history_messages
from synaptic.providers.claude.helpers import history_messages as claude_messages

ARABIC = "مرحبا بالعالم"


def slow(text: str = ""):
    time.sleep(0.02)
    return text or ARABIC


def agent_with(events, calls):
    tool = Tool("slow", {"name": "slow", "parameters": {}}, slow, add_to_registry=False)
    model = Model(Provider.OPENAI, "fixture", api_key="unused", tools=[tool])
    return Agent(model, tools=[tool], events=events), ResponseMem("", None, tool_calls=calls)


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_tool_end_and_journal_carry_executor_latency(mode):
    seen, bus = [], EventBus(errors="raise")
    bus.on("tool_start", lambda call: time.sleep(0.05))  # observer time is excluded
    bus.on("tool_end", lambda call, result: seen.append(result))
    agent, memory = agent_with(bus, [ToolCall("slow", {}, id="a")])
    if mode == "sync":
        results = agent._tool_results(memory.tool_calls, memory)
    else:
        results = asyncio.run(agent._atool_results(memory.tool_calls, memory))
    assert 15 <= seen[0]["latency_ms"] < 50
    assert seen[0]["result"] == ARABIC
    assert "latency_ms" not in results[0]  # the provider-visible receipt is unchanged
    assert memory.tool_outcomes["a"]["latency_ms"] == seen[0]["latency_ms"]


def test_denied_call_reports_zero_latency():
    seen, bus = [], EventBus(errors="raise")
    bus.on("tool_end", lambda call, result: seen.append(result))
    agent, memory = agent_with(bus, [ToolCall("slow", {}, id="a")])
    agent.permission = lambda name, args: False
    agent._tool_results(memory.tool_calls, memory)
    assert seen[0]["error"] == "Permission denied" and seen[0]["latency_ms"] == 0


def test_model_dispatch_journals_latency():
    tool = Tool("slow", {"name": "slow", "parameters": {}}, slow, add_to_registry=False)
    memory = ResponseMem("", None, tool_calls=[ToolCall("slow", {}, id="a")])
    run_tools_sync([tool], [], memory.tool_calls, memory=memory)
    assert memory.tool_outcomes["a"]["latency_ms"] >= 15
    assert "latency_ms" not in memory.tool_results[0]


def chat_payload():
    memory = ResponseMem("", None, tool_calls=[ToolCall("slow", {"text": ARABIC}, id="a")])
    memory.tool_results = [{"name": "slow", "call_id": "a", "result": ARABIC}]
    messages = history_messages(History([memory]), "", {"assistant": "assistant"})
    return messages[0]["tool_calls"][0]["function"]["arguments"], messages[1]["content"]


def test_chat_history_keeps_non_ascii_literal_by_default():
    arguments, content = chat_payload()
    assert ARABIC in arguments and ARABIC in content
    assert "\\u" not in arguments + content
    assert json.loads(arguments) == {"text": ARABIC}


def test_ascii_escaping_is_opt_in():
    synaptic.set_ensure_ascii(True)
    try:
        arguments, content = chat_payload()
        assert arguments.isascii() and content == ARABIC  # plain strings are not JSON text
        assert json.loads(arguments) == {"text": ARABIC}
        assert _json.dumps(ARABIC, ensure_ascii=False) == f'"{ARABIC}"'
    finally:
        synaptic.set_ensure_ascii(False)


def tool_turn(result):
    memory = ResponseMem("", None, tool_calls=[ToolCall("slow", {}, id="a")])
    memory.tool_results = [result]
    return History([memory])


@pytest.mark.parametrize(
    "receipt, text, failed",
    [
        ({"name": "slow", "call_id": "a", "result": {"ok": True, "v": None}}, None, False),
        ({"name": "slow", "call_id": "a", "result": ARABIC}, ARABIC, False),
        ({"name": "slow", "call_id": "a", "error": "Permission denied"}, None, True),
    ],
)
def test_tool_content_is_value_only_and_stays_correlated(receipt, text, failed):
    chat = history_messages(tool_turn(receipt), "", {"assistant": "assistant"})
    assert chat[0]["tool_calls"][0]["id"] == chat[1]["tool_call_id"] == "a"
    assert "call_id" not in chat[1]["content"]
    block = claude_messages(tool_turn(receipt))[1]["content"][0]
    assert block["tool_use_id"] == "a" and block.get("is_error", False) is failed
    from synaptic.providers.gemini.helpers import history_contents

    native = history_contents(tool_turn(receipt), {})[1].parts[0].function_response
    key = "error" if failed else "result"
    assert native.id == "a" and native.response == {key: receipt[key]}
    if failed:
        assert json.loads(chat[1]["content"]) == {"error": "Permission denied"}
        assert block["content"] == "Permission denied"
    elif text:
        assert chat[1]["content"] == block["content"] == text
    else:  # structured values are JSON for both, not a Python repr
        assert json.loads(chat[1]["content"]) == json.loads(block["content"]) == receipt["result"]


def test_sync_signal_reports_error_and_latency():
    seen, signals = [], EventBus(errors="raise")
    signals.on("ToolCallResult", seen.append)
    agent, memory = agent_with(EventBus(), [ToolCall("slow", {}, id="a")])
    agent.signals = signals
    agent._tool_results(memory.tool_calls, memory)
    agent.permission = lambda name, args: False
    denied = ResponseMem("", None, tool_calls=[ToolCall("slow", {}, id="b")])
    agent._tool_results(denied.tool_calls, denied)
    assert seen[0].error is None and seen[0].latency_ms >= 15
    assert seen[1].error == "Permission denied" and seen[1].call.id == "b"


def test_step_end_response_carries_model_latency():
    from tests.test_execution_recovery import make_model

    seen = []
    agent = Agent(make_model([], []), max_turns=1)
    agent.on("step_end", lambda response: seen.append(response.latency_ms))
    response = agent.run("go")
    assert isinstance(seen[0], int) and seen[0] >= 0
    restored = History.from_dict(json.loads(json.dumps(agent.session.history.to_dict())))
    assert restored.entries[-1].latency_ms is None and response is not None  # not persisted
    assert ResponseMem("", None).latency_ms is None
