"""ClaudeStream implementation separated by runtime responsibility."""

import asyncio
import json
from typing import AsyncIterator, List, Optional
from ...core.base import ResponseChunk, ToolCallArgsDelta
from ...core.tool import ToolCall
from .helpers import safe_json

_STRUCTURED_OUTPUT_TOOL = "__structured_output__"


class ClaudeStream:
    async def astream(
        self,
        prompt: Optional[str],
        role: str = "user",
        audio: Optional[List[str]] = None,
        images=None,
        abort=None,
        **kwargs,
    ) -> AsyncIterator[ResponseChunk]:
        self.capabilities.check("invoke", images=images, audio=audio)
        role = "assistant" if role == "assistant" else "user"
        params = self._build_request(prompt, role, images=images, **kwargs)

        accumulated_message = ""
        tool_calls: List[ToolCall] = []
        active_tool_name: Optional[str] = None
        active_tool_json: str = ""
        is_structured_output_block = False
        input_tokens = output_tokens = None
        finish_reason = response_id = active_tool_id = None

        async with self.async_client.messages.stream(**params) as stream:
            async for event in stream:
                if abort and abort.is_set():
                    raise asyncio.CancelledError()

                event_type = getattr(event, "type", None)

                if event_type == "message_start":
                    response_id = getattr(event.message, "id", None)
                    u = getattr(getattr(event, "message", None), "usage", None)
                    if u:
                        input_tokens = getattr(u, "input_tokens", None)
                elif event_type == "message_delta":
                    finish_reason = (
                        getattr(getattr(event, "delta", None), "stop_reason", None) or finish_reason
                    )
                    u = getattr(event, "usage", None)
                    if u:
                        output_tokens = getattr(u, "output_tokens", None)
                elif event_type == "content_block_start":
                    block = getattr(event, "content_block", None)
                    if block and getattr(block, "type", None) == "tool_use":
                        active_tool_name = block.name
                        active_tool_id = block.id
                        active_tool_json = ""
                        is_structured_output_block = block.name == _STRUCTURED_OUTPUT_TOOL

                elif event_type == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    if delta is None:
                        continue
                    delta_type = getattr(delta, "type", None)
                    if delta_type == "text_delta":
                        accumulated_message += delta.text
                        yield ResponseChunk(text=delta.text, is_final=False, function_call=None)
                    elif delta_type == "input_json_delta":
                        active_tool_json += delta.partial_json
                        if not is_structured_output_block and active_tool_name:
                            yield ResponseChunk(
                                text="",
                                tool_call_delta=ToolCallArgsDelta(
                                    active_tool_name, delta.partial_json, active_tool_json
                                ),
                            )

                elif event_type == "content_block_stop":
                    if active_tool_name is not None:
                        args = safe_json(active_tool_json)
                        if is_structured_output_block:
                            json_str = json.dumps(args)
                            accumulated_message = json_str
                            yield ResponseChunk(text=json_str, is_final=False, function_call=None)
                        else:
                            tfc = ToolCall(name=active_tool_name, args=args, id=active_tool_id)
                            tool_calls.append(tfc)
                            yield ResponseChunk(text="", is_final=False, function_call=tfc)
                        active_tool_name = None
                        active_tool_json = ""
                        is_structured_output_block = False

        yield ResponseChunk(
            text=accumulated_message,
            is_final=True,
            function_call=None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason=finish_reason,
            extensions={"response_id": response_id},
        )
