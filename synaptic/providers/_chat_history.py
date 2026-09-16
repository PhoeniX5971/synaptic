import json
from typing import Any, Dict, List

from ..core.base import ResponseMem
from ..core.base.content import references, message_text
from ._media import chat_content


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
            "content": chat_content(
                message_text(memory), references(memory, "image"), references(memory, "audio")
            ),
        }
        if isinstance(memory, ResponseMem) and memory.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": call.id,
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
                messages.append(
                    {
                        "role": "tool",
                        "name": name,
                        "content": result_text(result),
                        "tool_call_id": memory.tool_calls[i].id,
                    }
                )
    return messages
