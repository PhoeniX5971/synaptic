import base64
import json
import mimetypes
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...core.base import ResponseFormat, ResponseMem
from ...core.tool import ToolCall


def image_url(src: str) -> str:
    path = Path(src)
    if not path.exists():
        return src
    mime, _ = mimetypes.guess_type(src)
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime or 'application/octet-stream'};base64,{data}"


def audio_block(src: str) -> Dict[str, Any]:
    path = Path(src)
    if path.exists():
        data = path.read_bytes()
        mime, _ = mimetypes.guess_type(src)
    else:
        with urllib.request.urlopen(src) as resp:
            data = resp.read()
            mime = resp.headers.get_content_type()
    fmt = (mime or "audio/mpeg").split("/")[-1].replace("mpeg", "mp3")
    return {"type": "input_audio", "input_audio": {"data": base64.b64encode(data).decode(), "format": fmt}}


def content(prompt: Optional[str], images: Optional[List[str]], audio: Optional[List[str]]) -> Any:
    if not images and not audio:
        return prompt or ""
    parts: List[Dict[str, Any]] = [{"type": "text", "text": prompt or ""}]
    parts.extend({"type": "image_url", "image_url": {"url": image_url(img)}} for img in images or [])
    parts.extend(audio_block(src) for src in audio or [])
    return parts


def messages(adapter, prompt: Optional[str], role: str, images=None, audio=None) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    system_message = adapter.instructions or ""
    if adapter.response_format == ResponseFormat.JSON:
        system_message += "\nYou must respond ONLY with valid JSON. Do not include explanations or markdown."
    if system_message:
        result.append({"role": "system", "content": system_message})
    result.extend(history_messages(adapter))
    if prompt is not None or images or audio:
        result.append({"role": adapter.role_map.get(role, "user"), "content": content(prompt, images, audio)})
    return result


def history_messages(adapter) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if adapter.history is None:
        return out
    for memory in adapter.history.effective_mems():
        if isinstance(memory, ResponseMem) and memory.tool_calls:
            out.append({
                "role": "assistant",
                "content": memory.message or "",
                "tool_calls": [
                    {
                        "id": f"call_{i}",
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.args) if tc.args else "{}"},
                    }
                    for i, tc in enumerate(memory.tool_calls)
                ],
            })
            for i, (tc, result) in enumerate(zip(memory.tool_calls, memory.tool_results or [])):
                out.append({
                    "role": "tool",
                    "tool_call_id": f"call_{i}",
                    "name": tc.name,
                    "content": json.dumps(result) if isinstance(result, dict) else str(result),
                })
        else:
            out.append({"role": adapter.role_map.get(memory.role, "user"), "content": memory.message})
    return out


def parse_calls(message: Any) -> List[ToolCall]:
    calls: List[ToolCall] = []
    for tc in getattr(message, "tool_calls", None) or []:
        if not hasattr(tc, "function"):
            continue
        try:
            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
        except json.JSONDecodeError:
            args = {}
        calls.append(ToolCall(name=tc.function.name, args=args))
    return calls


def pending_calls(pending: Dict[int, Dict[str, str]]) -> List[ToolCall]:
    calls = []
    for idx in sorted(pending):
        item = pending[idx]
        try:
            args = json.loads(item["args"]) if item["args"] else {}
        except json.JSONDecodeError:
            args = {}
        calls.append(ToolCall(name=item["name"], args=args))
    return calls
