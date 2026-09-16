import json
from datetime import datetime, timezone

import pytest

from synaptic import ContentBlock, History, ResponseMem, ToolCall, UserMem
from synaptic.core.base.memory import complete_tool_call
from synaptic.core._model_api import ModelAPI

NOW = datetime.now(timezone.utc)


def test_versioned_media_roundtrip_and_correlated_results():
    response = ResponseMem(
        "",
        NOW,
        tool_calls=[ToolCall("a", {}, id="native-a"), ToolCall("b", {}, id="native-b")],
        input_tokens=0,
        output_tokens=None,
        finish_reason="tool_calls",
    )
    history = History(
        [
            UserMem(
                "listen", NOW, content=[ContentBlock("audio", uri="data:audio/wav;base64,YQ==")]
            ),
            response,
        ]
    )
    api = ModelAPI()
    api.history = history
    api.supply_results(
        response, [{"call_id": "native-b", "result": 2}, {"call_id": "native-a", "result": 1}]
    )
    copied = History.from_dict(json.loads(json.dumps(history.to_dict())))
    assert copied.entries[0].content[0].uri == history.entries[0].content[0].uri
    assert [call.id for call in copied.entries[1].tool_calls] == ["native-a", "native-b"]
    assert copied.entries[1].tool_results[0]["result"] == 1
    assert copied.entries[1].input_tokens == 0 and copied.entries[1].output_tokens is None
    with pytest.raises(ValueError):
        complete_tool_call(response, {"native-a": 1})
    with pytest.raises(ValueError):
        History.from_dict({"version": 2})


def test_budget_truncates_complete_exchange_and_reports_loss():
    dropped = []
    exchange = ResponseMem(
        "call", NOW, tool_calls=[ToolCall("a", {})], tool_results=[{"result": 1}]
    )
    history = History(
        size=10, token_budget=2, token_counter=lambda memory: 2, on_truncate=dropped.extend
    )
    history.add(exchange)
    history.add(UserMem("next", NOW))
    assert dropped == [exchange] and history.entries[0].message == "next"
    assert dropped[0].tool_results == [{"result": 1}]


@pytest.mark.parametrize("provider", ["openai", "claude", "gemini", "xai"])
def test_provider_history_preserves_native_ids_and_supported_images(provider):
    history = History(
        [
            UserMem("see", NOW, content=[ContentBlock("image", uri="data:image/png;base64,YQ==")]),
            ResponseMem(
                "call",
                NOW,
                tool_calls=[ToolCall("echo", {}, id="native-id")],
                tool_results=[{"result": "ok"}],
            ),
        ]
    )
    if provider == "openai":
        from synaptic.providers.openai_.helpers import history_messages

        result = history_messages(history, "", {"user": "user", "assistant": "assistant"})
        assert result[1]["tool_calls"][0]["id"] == result[2]["tool_call_id"] == "native-id"
        assert result[0]["content"][1]["type"] == "image_url"
    elif provider == "claude":
        from synaptic.providers.claude.helpers import history_messages

        result = history_messages(history)
        assert (
            result[1]["content"][-1]["id"] == result[2]["content"][0]["tool_use_id"] == "native-id"
        )
        assert result[0]["content"][1]["type"] == "image"
    elif provider == "gemini":
        from synaptic.providers.gemini.helpers import history_contents

        result = history_contents(history, {"user": "user", "assistant": "model"})
        assert (
            result[1].parts[-1].function_call.id
            == result[2].parts[0].function_response.id
            == "native-id"
        )
        assert result[0].parts[1].inline_data.data == b"a"
    else:
        from synaptic.providers.xai.helpers import history_messages

        result = history_messages(history)
        assert result[1].tool_calls[0].id == result[2].tool_call_id == "native-id"
        assert len(result[0].content) == 2
