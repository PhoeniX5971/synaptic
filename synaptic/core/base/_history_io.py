"""Versioned context and independently retained recovery/attempt evidence."""

import copy
from datetime import datetime
from .content import ContentBlock
from ..tool import ToolCall
from ._results import correlate, indexed_results


def encode(memory):
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
    return entry


def serialize(history):
    return {
        "version": 1,
        "size": history.size,
        "entries": [encode(memory) for memory in history.entries],
        # Active context already encodes the same object; don't restore a stale second copy.
        "recovery": [
            encode(memory) for memory in history.recovery if memory not in history.entries
        ],
        "failed_attempts": [encode(memory) for memory in history.failed_attempts],
    }


def decode(raw):
    from .memory import ResponseMem, UserMem

    entry = copy.deepcopy(raw)
    entry["content"] = tuple(ContentBlock(**block) for block in entry.get("content", []))
    if isinstance(entry.get("created"), str):
        entry["created"] = datetime.fromisoformat(entry["created"])
    if "tool_calls" in entry:
        entry["tool_calls"] = [ToolCall(**call) for call in entry["tool_calls"]]
        memory = ResponseMem(**entry)
        known = indexed_results(memory, memory.tool_results, complete=False)
        memory.tool_results = [known[call.id] for call in memory.tool_calls if call.id in known]
        return memory
    return UserMem(**entry)


def deserialize(data):
    from .memory import History, ResponseMem

    if data.get("version") != 1:
        raise ValueError("Unsupported history serialization version")
    history = History([decode(raw) for raw in data["entries"]], size=data.get("size", 10))
    for raw in data.get("recovery", []):
        history._retain(decode(raw))
    for raw in data.get("failed_attempts", []):
        memory = decode(raw)
        if not isinstance(memory, ResponseMem):
            raise ValueError("Failed generation attempts must be response records")
        history.failed_attempts.append(memory)
    return history
