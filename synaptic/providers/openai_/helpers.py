import base64
import json
import mimetypes
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...core.base import ResponseMem
from ...core.tool import ToolCall


def result_text(result: Any) -> str:
    try:
        if hasattr(result, "model_dump"):
            return json.dumps(result.model_dump())
        if hasattr(result, "__dict__"):
            return json.dumps(result.__dict__)
        if isinstance(result, (dict, list)):
            return json.dumps(result)
    except Exception:
        pass
    return str(result)


def history_messages(history, instructions: str, role_map: Dict[str, str]) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    if instructions:
        messages.append({"role": "system", "content": instructions})
    if history is None:
        return messages
    for memory in history.effective_mems():
        msg: Dict[str, Any] = {
            "role": role_map.get(memory.role, "user"),
            "content": memory.message,
        }
        if isinstance(memory, ResponseMem) and memory.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.args) if call.args else "{}",
                    },
                }
                for i, call in enumerate(memory.tool_calls)
            ]
        messages.append(msg)
        if isinstance(memory, ResponseMem) and memory.tool_results:
            for i, result in enumerate(memory.tool_results):
                name = memory.tool_calls[i].name if memory.tool_calls else f"tool_{i}"
                messages.append({
                    "role": "tool",
                    "name": name,
                    "content": result_text(result),
                    "tool_call_id": f"call_{i}",
                })
    return messages


def audio_blocks(audio: Optional[List[str]]) -> List[Dict[str, Any]]:
    blocks = []
    for src in audio or []:
        path = Path(src)
        if path.exists():
            data = path.read_bytes()
            mime, _ = mimetypes.guess_type(src)
        else:
            with urllib.request.urlopen(src) as resp:
                data = resp.read()
                mime = resp.headers.get_content_type()
        fmt = (mime or "audio/mpeg").split("/")[-1].replace("mpeg", "mp3")
        blocks.append({
            "type": "input_audio",
            "input_audio": {"data": base64.b64encode(data).decode(), "format": fmt},
        })
    return blocks


def add_prompt(messages: List[Dict[str, Any]], prompt: Optional[str], role: str, audio: Optional[List[str]]) -> None:
    blocks = audio_blocks(audio)
    if prompt is None and not blocks:
        return
    content: Any = ([{"type": "text", "text": prompt or ""}] + blocks) if blocks else (prompt or "")
    messages.append({"role": role, "content": content})


def parse_tool_calls(message: Any) -> List[ToolCall]:
    calls: List[ToolCall] = []
    for tool_call in getattr(message, "tool_calls", None) or []:
        if not hasattr(tool_call, "function"):
            continue
        try:
            args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
        except json.JSONDecodeError:
            args = {}
        calls.append(ToolCall(name=tool_call.function.name, args=args))
    return calls


def stream_tool_calls(pending: Dict[int, Dict[str, str]]) -> List[ToolCall]:
    calls = []
    for idx in sorted(pending):
        item = pending[idx]
        try:
            args = json.loads(item["args"]) if item["args"] else {}
        except json.JSONDecodeError:
            args = {}
        calls.append(ToolCall(name=item["name"], args=args))
    return calls
