"""Credential-free public API example; also exercised against installed wheels."""

import json
from synaptic import History, ResponseMem, Tool, ToolCall, complete_tool_call


def add(a: int, b: int) -> int:
    return a + b


def main() -> None:
    tool = Tool(
        "add",
        {
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            }
        },
        add,
        add_to_registry=False,
    )
    call = ToolCall("add", {"a": 2, "b": 3}, id="offline-add")
    receipt = tool.prepare(call).execute()
    response = ResponseMem("", None, tool_calls=[call])
    history = History([response])
    complete_tool_call(response, {call.id: {"call_id": call.id, "result": receipt}}, history)
    restored = History.from_dict(json.loads(json.dumps(history.to_dict())))
    assert restored.recovery[0].tool_results[0]["result"] == 5
    print("Offline tool receipt survived history round trip: 5")


if __name__ == "__main__":
    main()
