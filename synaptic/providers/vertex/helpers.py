import mimetypes
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from vertexai.generative_models import Content, GenerationConfig, Part

from ...core.base import ResponseFormat, ResponseMem
from ...core.tool import ToolCall


def history_contents(history, role_map) -> List[Content]:
    contents = []
    if history is None:
        return contents
    for mem in history.MemoryList:
        if isinstance(mem, ResponseMem) and mem.tool_calls:
            contents.append(Content(role="model", parts=[Part.from_text(mem.message or "")]))
            response_parts = []
            for call, result in zip(mem.tool_calls, mem.tool_results or []):
                resp = result.get("result", result.get("error", "")) if isinstance(result, dict) else str(result)
                response_parts.append(Part.from_function_response(
                    name=call.name,
                    response={"result": resp},
                ))
            if response_parts:
                contents.append(Content(role="user", parts=response_parts))
        else:
            contents.append(Content(
                role=role_map.get(mem.role, "user"),
                parts=[Part.from_text(mem.message)],
            ))
    return contents


def audio_parts(audio: Optional[List[str]]) -> List[Part]:
    parts = []
    for src in audio or []:
        path = Path(src)
        if path.exists():
            data = path.read_bytes()
            mime, _ = mimetypes.guess_type(src)
        else:
            with urllib.request.urlopen(src) as resp:
                data = resp.read()
                mime = resp.headers.get_content_type()
        parts.append(Part.from_data(data=data, mime_type=mime if mime and mime.startswith("audio/") else "audio/mpeg"))
    return parts


def generation_config(response_format: ResponseFormat, temperature: float) -> GenerationConfig:
    mime = "application/json" if response_format == ResponseFormat.JSON else "text/plain"
    return GenerationConfig(temperature=temperature, response_mime_type=mime)


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
        message=message, created=created, tool_calls=calls,
        input_tokens=getattr(um, "prompt_token_count", 0) or 0,
        output_tokens=getattr(um, "candidates_token_count", 0) or 0,
    )
