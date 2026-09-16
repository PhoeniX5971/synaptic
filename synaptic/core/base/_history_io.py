"""Versioned JSON history and call-result correlation."""

import copy
from datetime import datetime

from .content import ContentBlock
from ..tool import ToolCall


def serialize(history):
    entries = []
    for memory in history.entries:
        entry = {
            "message": memory.message,
            "role": memory.role,
            "created": memory.created.isoformat()
            if isinstance(memory.created, datetime)
            else memory.created,
            "content": [block.to_dict() for block in memory.content],
        }
        if hasattr(memory, "tool_calls"):
            entry.update(
                tool_calls=[
                    {"name": call.name, "args": copy.deepcopy(call.args), "id": call.id}
                    for call in memory.tool_calls
                ],
                tool_results=copy.deepcopy(memory.tool_results),
                input_tokens=memory.input_tokens,
                output_tokens=memory.output_tokens,
                finish_reason=memory.finish_reason,
                extensions=copy.deepcopy(memory.extensions),
            )
        entries.append(entry)
    return {"version": 1, "size": history.size, "entries": entries}


def deserialize(data):
    from .memory import History, Memory, ResponseMem, UserMem

    if data.get("version") != 1:
        raise ValueError("Unsupported history serialization version")
    entries: list[Memory] = []
    for raw in data["entries"]:
        entry = copy.deepcopy(raw)
        entry["content"] = tuple(ContentBlock(**block) for block in entry.get("content", []))
        if isinstance(entry.get("created"), str):
            entry["created"] = datetime.fromisoformat(entry["created"])
        if "tool_calls" in entry:
            entry["tool_calls"] = [ToolCall(**call) for call in entry["tool_calls"]]
            entries.append(ResponseMem(**entry))
        else:
            entries.append(UserMem(**entry))
    return History(entries, size=data.get("size", 10))


def correlate(memory, results):
    if isinstance(results, dict):
        indexed = results
    elif results and all(
        isinstance(result, dict) and "call_id" in result for result in results
    ):
        indexed = {result["call_id"]: result for result in results}
        if len(indexed) != len(results):
            raise ValueError("Duplicate external result call_id")
    else:
        if len(results) != len(memory.tool_calls):
            raise ValueError("Every tool call requires exactly one result")
        return copy.deepcopy(
            results
        )  # legacy ordered result lists remain explicit compatibility
    ids = [call.id for call in memory.tool_calls]
    if set(indexed) != set(ids) or len(ids) != len(set(ids)):
        raise ValueError("External results must match the complete set of call IDs")
    return [copy.deepcopy(indexed[call_id]) for call_id in ids]
