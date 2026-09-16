"""Shared Chat Completions parsing; only completed JSON objects become tool calls."""

import json
from datetime import datetime, timezone

from ..core.base import ResponseChunk, ResponseMem, ToolCallArgsDelta
from ..core.tool import ToolCall


def arguments(text):
    value = json.loads(text) if text else {}
    if not isinstance(value, dict):
        raise ValueError("Tool arguments must be a JSON object")
    return value


def parse_calls(message):
    return [
        ToolCall(
            call.function.name, arguments(call.function.arguments), id=getattr(call, "id", None)
        )
        for call in getattr(message, "tool_calls", None) or []
    ]


def response_memory(response):
    choice = response.choices[0] if response.choices else None
    usage = getattr(response, "usage", None)
    return ResponseMem(
        choice.message.content or "" if choice else "",
        datetime.now(timezone.utc),
        tool_calls=parse_calls(choice.message) if choice else [],
        input_tokens=getattr(usage, "prompt_tokens", None),
        output_tokens=getattr(usage, "completion_tokens", None),
        finish_reason=getattr(choice, "finish_reason", None),
        extensions={"response_id": getattr(response, "id", None)},
    )


async def assemble(source, abort=None):
    pending: dict[int, dict] = {}
    text, usage, finish, response_id = "", None, None, None
    try:
        async for chunk in source:
            if abort and abort.is_set():
                import asyncio

                raise asyncio.CancelledError()
            response_id = getattr(chunk, "id", None) or response_id
            usage = getattr(chunk, "usage", None) or usage
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            finish = getattr(choice, "finish_reason", None) or finish
            delta = choice.delta
            if delta.content:
                text += delta.content
                yield ResponseChunk(delta.content)
            for call in delta.tool_calls or []:
                current = pending.setdefault(call.index, {"id": None, "name": "", "args": ""})
                current["id"] = getattr(call, "id", None) or current["id"]
                if call.function:
                    current["name"] += call.function.name or ""
                    current["args"] += call.function.arguments or ""
                    if call.function.arguments and current["name"]:
                        yield ResponseChunk(
                            "",
                            tool_call_delta=ToolCallArgsDelta(
                                current["name"], call.function.arguments, current["args"]
                            ),
                        )
        for call in pending.values():
            yield ResponseChunk(
                "", function_call=ToolCall(call["name"], arguments(call["args"]), id=call["id"])
            )
        yield ResponseChunk(
            text,
            is_final=True,
            finish_reason=finish,
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            extensions={"response_id": response_id},
        )
    finally:
        close = getattr(source, "aclose", None) or getattr(source, "close", None)
        if close:
            import inspect

            result = close()
            if inspect.isawaitable(result):
                await result
