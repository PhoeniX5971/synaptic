import json
from typing import Any, Dict, List

from ...core.base import ResponseMem


def tool_id(turn_index: int, call_index: int) -> str:
    return f"toolu_{turn_index:03d}_{call_index:02d}"


def history_messages(history) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    if history is None:
        return messages
    for turn_idx, memory in enumerate(history.effective_mems()):
        if isinstance(memory, ResponseMem) and memory.tool_calls:
            content: List[Dict[str, Any]] = []
            if memory.message:
                content.append({"type": "text", "text": memory.message})
            for call_idx, call in enumerate(memory.tool_calls):
                content.append({
                    "type": "tool_use",
                    "id": tool_id(turn_idx, call_idx),
                    "name": call.name,
                    "input": call.args,
                })
            messages.append({"role": "assistant", "content": content})
            results = getattr(memory, "tool_results", None) or []
            result_content: List[Dict[str, Any]] = []
            for call_idx, call in enumerate(memory.tool_calls):
                result = results[call_idx] if call_idx < len(results) else {}
                resp = result.get("result", result.get("error", "")) if isinstance(result, dict) else str(result)
                result_content.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id(turn_idx, call_idx),
                    "content": str(resp),
                })
            if result_content:
                messages.append({"role": "user", "content": result_content})
        else:
            role = "assistant" if memory.role == "assistant" else "user"
            messages.append({"role": role, "content": memory.message})
    return messages


def schema_for(schema: Any) -> Any:
    if hasattr(schema, "model_json_schema"):
        return schema.model_json_schema()
    return schema


def safe_json(data: str) -> Dict[str, Any]:
    try:
        return json.loads(data) if data else {}
    except json.JSONDecodeError:
        return {}
