"""Real installed SDK parsing over fixture HTTP; no live provider qualification."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from openai import OpenAI, AsyncOpenAI

from synaptic import History, ResponseFormat, ToolRegistry
from synaptic.providers.openai_ import OpenAIAdapter
from synaptic.providers.deepseek import DeepSeekAdapter
from synaptic.providers.universal_openai import UniversalLLMAdapter


@pytest.mark.parametrize("adapter_type", [OpenAIAdapter, DeepSeekAdapter, UniversalLLMAdapter])
def test_chat_sdk_sync_stream_native_ids_usage_and_configuration(adapter_type):
    captured = []

    def handle(request):
        body = json.loads(request.content)
        captured.append(body)
        common = {"id": "response-id", "created": 0, "model": "fixture"}
        call = {
            "id": "native-call",
            "type": "function",
            "function": {"name": "echo", "arguments": '{"x":1}'},
        }
        if not body.get("stream"):
            return httpx.Response(
                200,
                json={
                    **common,
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "answer",
                                "tool_calls": [call],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
                },
            )
        chunks = [
            {
                **common,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "answer", "tool_calls": [dict(call, index=0)]},
                    }
                ],
            },
            {**common, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
            {
                **common,
                "choices": [],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
            },
        ]
        content = (
            "".join(
                "data: " + json.dumps(dict(chunk, object="chat.completion.chunk")) + "\n\n"
                for chunk in chunks
            )
            + "data: [DONE]\n\n"
        )
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=content)

    async def run():
        registry = ToolRegistry()
        adapter = adapter_type(
            model="fixture",
            history=History(),
            api_key="fixture",
            response_format=ResponseFormat.NONE,
            tools=[],
            max_tokens=99,
            tool_registry=registry,
        )
        await adapter.aclose()
        adapter._sync_closed = adapter._async_closed = False
        adapter.client = OpenAI(
            api_key="fixture", http_client=httpx.Client(transport=httpx.MockTransport(handle))
        )
        adapter.async_client = AsyncOpenAI(
            api_key="fixture", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle))
        )
        sync = adapter.invoke("prompt", temperature=0.25)
        chunks = [chunk async for chunk in adapter.astream("prompt", max_tokens=42)]
        assert (
            sync.tool_calls[0].id
            == next(c.function_call.id for c in chunks if c.function_call)
            == "native-call"
        )
        assert (
            (sync.input_tokens, sync.output_tokens)
            == (chunks[-1].input_tokens, chunks[-1].output_tokens)
            == (12, 3)
        )
        assert sync.finish_reason == chunks[-1].finish_reason == "tool_calls"
        assert captured[0]["max_tokens"] == 99 and captured[1]["max_tokens"] == 42
        assert captured[0]["temperature"] == 0.25 and captured[1]["stream_options"]["include_usage"]
        await adapter.aclose()
        assert (
            adapter.client.is_closed()
            and adapter.async_client.is_closed()
            and not registry._callbacks
        )

    asyncio.run(run())


def test_malformed_completed_tool_arguments_fail_without_proposal():
    from synaptic.providers._chat import arguments

    for value in ("{", "[]", "null"):
        with pytest.raises((ValueError, json.JSONDecodeError)):
            arguments(value)


def test_missing_capability_fails_before_sdk_call():
    from synaptic.core.capabilities import UnsupportedCapability

    adapter = DeepSeekAdapter(
        model="fixture", history=History(), api_key="fixture", response_format=ResponseFormat.NONE
    )
    with pytest.raises(UnsupportedCapability):
        adapter.invoke("prompt", images=["https://example.invalid/image"])
    asyncio.run(adapter.aclose())
