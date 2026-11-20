"""Archive limits bound resolved evidence without ever dropping unresolved records."""

import warnings

import pytest

from synaptic import History, ResponseMem, ToolCall, UserMem


def exchange(index, resolved=True):
    memory = ResponseMem("", None, tool_calls=[ToolCall("t", {}, id=f"c{index}")])
    if resolved:
        memory.tool_results = [{"name": "t", "call_id": f"c{index}", "result": index}]
    else:
        memory.extensions["incomplete"] = True
    return memory


def filled(limit, unresolved=()):
    history = History(size=2, archive_limit=limit, on_truncate=lambda dropped: None)
    for index in range(6):
        history.add(exchange(index, index not in unresolved))
    history.failed_attempts.extend(ResponseMem(str(i), None) for i in range(5))
    history.enforce_budget()
    return history


def ids(history):
    return [memory.tool_calls[0].id for memory in history.recovery]


def test_unbounded_by_default():
    history = filled(None)
    assert len(history.recovery) == 6 and len(history.failed_attempts) == 5
    assert "archive_limit" not in history.to_dict()


def test_limit_drops_oldest_resolved_out_of_context_records():
    history = filled(3)
    assert ids(history) == ["c3", "c4", "c5"]
    assert [m.message for m in history.failed_attempts] == ["2", "3", "4"]
    assert [m.tool_calls[0].id for m in history.entries] == ["c4", "c5"]


def test_unresolved_and_in_context_records_are_never_dropped():
    history = filled(0, unresolved={0, 2})
    assert ids(history) == ["c0", "c2", "c4", "c5"] and history.failed_attempts == []
    restored = History.from_dict(history.to_dict())
    assert restored.archive_limit == 0 and sorted(ids(restored)) == ["c0", "c2", "c4", "c5"]


def test_negative_limit_is_rejected():
    with pytest.raises(ValueError, match="archive_limit"):
        History(archive_limit=-1)
