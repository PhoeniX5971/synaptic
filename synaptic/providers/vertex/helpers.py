import mimetypes
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from vertexai.generative_models import Content, GenerationConfig, Part

from ...core.base import ResponseFormat, ResponseMem
from ...core.tool import ToolCall
from ...core.base.content import references, message_text


def history_contents(history, role_map) -> List[Content]:
    contents: list[Content] = []
    if history is None:
        return contents
    for mem in history.effective_mems():
        if isinstance(mem, ResponseMem) and mem.tool_calls:
            contents.append(
                Content(
                    role="model",
                    parts=[Part.from_text(message_text(mem) or "")]
                    + [
                        Part.from_dict(
                            {"function_call": {"name": call.name, "args": call.args}}
                        )
                        for call in mem.tool_calls
                    ],
                )
            )
            response_parts = []
            for call, result in zip(mem.tool_calls, mem.tool_results or []):
                resp = (
                    result.get("result", result.get("error", ""))
                    if isinstance(result, dict)
                    else str(result)
                )
                response_parts.append(
                    Part.from_function_response(
                        name=call.name,
                        response={"result": resp},
                    )
                )
            if response_parts:
                contents.append(Content(role="user", parts=response_parts))
        else:
            contents.append(
                Content(
                    role=role_map.get(mem.role, "user"),
                    parts=[Part.from_text(message_text(mem))]
                    + audio_parts(references(mem, "audio")),
                )
            )
    return contents


def audio_parts(audio: Optional[List[str]]) -> List[Part]:
    parts = []
    for src in audio or []:
        from .._media import read_media

        data, mime = read_media(src)
        parts.append(
            Part.from_data(
                data=data,
                mime_type=mime if mime and mime.startswith("audio/") else "audio/mpeg",
            )
        )
    return parts


def generation_config(
    response_format: ResponseFormat, temperature: float, max_tokens=1024, schema=None
) -> GenerationConfig:
    if hasattr(schema, "model_json_schema"):
        schema = schema.model_json_schema()
    mime = "application/json" if response_format == ResponseFormat.JSON else "text/plain"
    return GenerationConfig(
        temperature=temperature,
        response_mime_type=mime,
        max_output_tokens=max_tokens,
        response_schema=schema if response_format == ResponseFormat.JSON else None,
    )


def response_mem(response) -> ResponseMem:
    created = datetime.now().astimezone(timezone.utc)
    message = ""
    calls: List[ToolCall] = []
    if response.candidates:
        cand = response.candidates[0]
        for fc in cand.function_calls or []:
            calls.append(ToolCall(name=fc.name, args=dict(fc.args) or {}))
        if cand.content and cand.content.parts:
            for part in cand.content.parts:
                if part.text:
                    message += part.text
    um = getattr(response, "usage_metadata", None)
    return ResponseMem(
        message=message,
        created=created,
        tool_calls=calls,
        input_tokens=getattr(um, "prompt_token_count", None),
        output_tokens=getattr(um, "candidates_token_count", None),
        finish_reason=str(response.candidates[0].finish_reason)
        if response.candidates
        else None,
    )
