"""Provider-specific fixtures use installed SDK response types and captured requests."""

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest

from synaptic import History, ResponseFormat, ToolRegistry


def options():
    return dict(
        model="fixture",
        history=History(),
        api_key="fixture",
        tools=[],
        response_format=ResponseFormat.NONE,
        response_schema=None,
        tool_registry=ToolRegistry(),
        max_tokens=77,
    )


def test_gemini_native_response_and_limits():
    from google.genai import types
    from synaptic.providers.gemini import GeminiAdapter

    response = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                finish_reason="STOP",
                content=types.Content(
                    parts=[
                        types.Part(text="answer"),
                        types.Part(
                            function_call=types.FunctionCall(name="echo", args={}, id="native")
                        ),
                    ]
                ),
            )
        ],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=12, candidates_token_count=3
        ),
    )
    captured = []

    def generate(**kwargs):
        captured.append(kwargs)
        return response

    async def stream(**kwargs):
        captured.append(kwargs)

        async def source():
            yield response

        return source()

    client = NS(
        models=NS(generate_content=generate), aio=NS(models=NS(generate_content_stream=stream))
    )

    async def run():
        with patch("synaptic.providers.gemini.model.genai.Client", return_value=client):
            adapter = GeminiAdapter(**options())
        sync = adapter.invoke("prompt")
        chunks = [c async for c in adapter.astream("prompt", max_tokens=33)]
        assert (
            sync.tool_calls[0].id
            == next(c.function_call.id for c in chunks if c.function_call)
            == "native"
        )
        assert (
            (sync.input_tokens, sync.output_tokens)
            == (chunks[-1].input_tokens, chunks[-1].output_tokens)
            == (12, 3)
        )
        assert sync.finish_reason == chunks[-1].finish_reason
        assert (
            captured[0]["config"].max_output_tokens == 77
            and captured[1]["config"].max_output_tokens == 33
        )
        assert captured[1]["config"].automatic_function_calling.disable is True
        await adapter.aclose()

    asyncio.run(run())


def test_claude_native_events_and_configuration():
    from anthropic.types import (
        Message,
        RawMessageStartEvent,
        RawMessageDeltaEvent,
        MessageDeltaUsage,
    )
    from anthropic.types.raw_message_delta_event import Delta
    from anthropic.types import (
        RawContentBlockStartEvent,
        RawContentBlockDeltaEvent,
        RawContentBlockStopEvent,
        InputJSONDelta,
        ToolUseBlock,
    )
    from synaptic.providers.claude import ClaudeAdapter

    response = Message(
        id="resp",
        type="message",
        role="assistant",
        model="fixture",
        stop_reason="tool_use",
        content=[ToolUseBlock(type="tool_use", id="native", name="echo", input={})],
        usage={"input_tokens": 12, "output_tokens": 3},
    )
    captured = []
    events = [
        RawMessageStartEvent(type="message_start", message=response),
        RawContentBlockStartEvent(
            type="content_block_start", index=0, content_block=response.content[0]
        ),
        RawContentBlockDeltaEvent(
            type="content_block_delta",
            index=0,
            delta=InputJSONDelta(type="input_json_delta", partial_json="{}"),
        ),
        RawContentBlockStopEvent(type="content_block_stop", index=0),
        RawMessageDeltaEvent(
            type="message_delta",
            delta=Delta(stop_reason="tool_use"),
            usage=MessageDeltaUsage(output_tokens=3),
        ),
    ]

    class Manager:
        async def __aenter__(self):
            async def source():
                for event in events:
                    yield event

            return source()

        async def __aexit__(self, *args):
            captured.append("closed")

    def create(**kwargs):
        captured.append(kwargs)
        return response

    def stream(**kwargs):
        captured.append(kwargs)
        return Manager()

    async def run():
        with (
            patch(
                "synaptic.providers.claude.model.anthropic.Anthropic",
                return_value=NS(messages=NS(create=create)),
            ),
            patch(
                "synaptic.providers.claude.model.anthropic.AsyncAnthropic",
                return_value=NS(messages=NS(stream=stream)),
            ),
        ):
            adapter = ClaudeAdapter(**options())
        sync = adapter.invoke("prompt", temperature=0.2)
        chunks = [c async for c in adapter.astream("prompt", max_tokens=33)]
        assert (
            sync.tool_calls[0].id
            == next(c.function_call.id for c in chunks if c.function_call)
            == "native"
        )
        assert sync.finish_reason == chunks[-1].finish_reason == "tool_calls"
        assert (chunks[-1].input_tokens, chunks[-1].output_tokens) == (12, 3)
        assert (
            captured[0]["max_tokens"] == 77
            and captured[1]["max_tokens"] == 33
            and captured[0]["temperature"] == 0.2
        )
        assert captured[-1] == "closed"
        await adapter.aclose()

    asyncio.run(run())


def test_vertex_sdk_response_and_native_async_method():
    from vertexai.generative_models import GenerationResponse
    from synaptic.providers.vertex import VertexAdapter

    response = GenerationResponse.from_dict(
        {
            "candidates": [
                {
                    "finish_reason": "STOP",
                    "content": {"role": "model", "parts": [{"text": "answer"}]},
                }
            ],
            "usage_metadata": {"prompt_token_count": 12, "candidates_token_count": 3},
        }
    )
    captured = []

    def generate(messages, **kwargs):
        captured.append(kwargs)
        return response

    async def stream(messages, **kwargs):
        captured.append(kwargs)

        async def source():
            yield response

        return source()

    async def run():
        with (
            patch("synaptic.providers.vertex.model.vertex_init"),
            patch(
                "synaptic.providers.vertex.model.GenerativeModel",
                return_value=NS(generate_content=generate, generate_content_async=stream),
            ),
        ):
            adapter = VertexAdapter(**options(), project="fixture", location="fixture")
        sync = adapter.invoke("prompt")
        chunks = [c async for c in adapter.astream("prompt", max_tokens=33)]
        assert sync.message == chunks[-1].text == "answer"
        assert (
            (sync.input_tokens, sync.output_tokens)
            == (chunks[-1].input_tokens, chunks[-1].output_tokens)
            == (12, 3)
        )
        assert captured[0]["generation_config"].to_dict()["max_output_tokens"] == 77
        assert (
            captured[1]["generation_config"].to_dict()["max_output_tokens"] == 33
            and captured[1]["stream"] is True
        )
        await adapter.aclose()

    asyncio.run(run())


def test_xai_sdk_response_and_limits():
    from xai_sdk.chat import Response
    from xai_sdk.proto import chat_pb2 as p
    from synaptic.providers.xai import XAIAdapter
    proto = p.GetChatCompletionResponse(id='response', outputs=[{'index': 0, 'finish_reason': 'REASON_STOP',
        'message': {'role': 'ROLE_ASSISTANT', 'content': 'answer', 'tool_calls': [{'id': 'native',
        'function': {'name': 'echo', 'arguments': '{}'}}]}}], usage={'prompt_tokens': 12, 'completion_tokens': 3})
    response = Response(proto, 0)
    captured = []
    async def stream():
        yield response, NS(choices=[NS(content='answer')])
    def create(**kwargs):
        captured.append(kwargs)
        return NS(sample=lambda: response, stream=stream)
    client = NS(chat=NS(create=create))
    async def run():
        with patch('synaptic.providers.xai.model.SyncClient', return_value=client), patch('synaptic.providers.xai.model.AsyncClient', return_value=client):
            adapter = XAIAdapter(**options())
        sync = adapter.invoke('prompt')
        chunks = [c async for c in adapter.astream('prompt', max_tokens=33)]
        assert sync.tool_calls[0].id == next(c.function_call.id for c in chunks if c.function_call) == 'native'
        assert (sync.input_tokens, sync.output_tokens) == (chunks[-1].input_tokens, chunks[-1].output_tokens) == (12, 3)
        assert captured[0]['max_tokens'] == 77 and captured[1]['max_tokens'] == 33
        await adapter.aclose()
    asyncio.run(run())


def test_together_bounded_sdk_bridge_usage_and_limits():
    from openai.types.chat import ChatCompletion, ChatCompletionChunk
    from synaptic.providers.together import TogetherAdapter
    response = ChatCompletion(id='response', object='chat.completion', created=0, model='fixture',
        choices=[{'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': 'answer'}}],
        usage={'prompt_tokens': 12, 'completion_tokens': 3, 'total_tokens': 15})
    chunks = [ChatCompletionChunk(id='response', object='chat.completion.chunk', created=0, model='fixture',
        choices=[{'index': 0, 'finish_reason': 'stop', 'delta': {'content': 'answer'}}],
        usage={'prompt_tokens': 12, 'completion_tokens': 3, 'total_tokens': 15})]
    captured = []
    def create(**kwargs):
        captured.append(kwargs)
        return iter(chunks) if kwargs.get('stream') else response
    async def run():
        with patch('synaptic.providers.together.model.Together', return_value=NS(chat=NS(completions=NS(create=create)))):
            adapter = TogetherAdapter(**options())
        sync = adapter.invoke('prompt')
        output = [c async for c in adapter.astream('prompt', max_tokens=33)]
        assert sync.message == output[-1].text == 'answer'
        assert (output[-1].input_tokens, output[-1].output_tokens) == (12, 3)
        assert captured[0]['max_tokens'] == 77 and captured[1]['max_tokens'] == 33
        await adapter.aclose()
    asyncio.run(run())
