import copy
import json
from types import SimpleNamespace
import pytest
from synaptic import History, ResponseMem, ToolCall, complete_tool_call
from tests.test_public_integration import model_with

BAD = [
    {"a": {"call_id": "b", "result": 1}, "b": {"result": 2}},
    [{"call_id": "a", "result": 1}, {"result": 2}],
    [{"call_id": "a", "result": 1}, {"call_id": "a", "result": 2}],
    {"a": 1},
    {"a": 1, "b": 2, "c": 3},
    [{"call_id": None, "result": 1}, {"call_id": "b", "result": 2}],
]


@pytest.mark.parametrize("results", BAD)
@pytest.mark.parametrize("restored", [False, True])
def test_bad_ids_reject_without_mutation(results, restored):
    memory = ResponseMem(
        "", None, tool_calls=[ToolCall("one", {}, id="a"), ToolCall("two", {}, id="b")]
    )
    history = History([memory])
    if restored:
        history = History.from_dict(json.loads(json.dumps(history.to_dict())))
        memory = history.entries[0]
    model = model_with(SimpleNamespace(), history=history)
    before = copy.deepcopy(history.to_dict())
    with pytest.raises(ValueError):
        model.supply_results(memory, results)
    assert history.to_dict() == before


@pytest.mark.parametrize(
    "results",
    [
        {"b": {"result": 2}, "a": {"result": 1}},
        [{"call_id": "b", "result": 2}, {"call_id": "a", "result": 1}],
        [{"result": 1}, {"result": 2}],
        [1, 2],
    ],
)
def test_known_results_immutable_after_restoration(results):
    memory = ResponseMem(
        "", None, tool_calls=[ToolCall("one", {}, id="a"), ToolCall("two", {}, id="b")]
    )
    complete_tool_call(memory, results)
    history = History.from_dict(json.loads(json.dumps(History([memory]).to_dict())))
    memory = history.entries[0]
    before = history.to_dict()
    with pytest.raises(ValueError, match="already recorded"):
        complete_tool_call(memory, {"a": {"result": "changed"}, "b": {"result": 2}})
    assert history.to_dict() == before
    complete_tool_call(memory, memory.tool_results)
    assert history.to_dict() == before


def test_restored_malformed_receipt_ids_are_rejected():
    memory = ResponseMem("", None, tool_calls=[ToolCall("one", {}, id="a")])
    snapshot = History([memory]).to_dict()
    snapshot["entries"][0]["tool_results"] = [{"call_id": "wrong", "result": "original"}]
    with pytest.raises(ValueError, match="Unexpected"):
        History.from_dict(snapshot)


def test_restored_tagged_results_are_canonical_before_provider_projection():
    memory = ResponseMem(
        "", None, tool_calls=[ToolCall("one", {}, id="a"), ToolCall("two", {}, id="b")]
    )
    complete_tool_call(memory, [{"call_id": "b", "result": 2}, {"call_id": "a", "result": 1}])
    snapshot = History([memory]).to_dict()
    snapshot["entries"][0]["tool_results"].reverse()
    restored = History.from_dict(json.loads(json.dumps(snapshot)))
    result = restored.effective_mems()[0]
    assert [item["call_id"] for item in result.tool_results] == ["a", "b"]
    assert restored.recovery[0] is result
    with pytest.raises(ValueError, match="already recorded"):
        complete_tool_call(
            result,
            {"a": {"call_id": "a", "result": "changed"}, "b": {"call_id": "b", "result": 2}},
        )
