"""Todo lists: required items gate finishing, state persists, placement is cache-aware."""

import asyncio
import json

import pytest
from pydantic import BaseModel

from synaptic import Agent, FinalAnswer, Session, Todo, Todos, ToolCall
from synaptic.agent.session import BudgetExceeded
from synaptic.providers._chat_history import history_messages
from tests.test_tool_errors_and_guards import drive, scripted


class Draft(BaseModel):
    title: str


def update(todo_id, status, call_id=None, **extra):
    return ToolCall("todo_update", {"id": todo_id, "status": status, **extra}, id=call_id or f"u-{todo_id}-{status}")


def required():
    return [Todo("Check against the library", required=True), Todo("Optional polish")]


@pytest.mark.parametrize("mode", ["sync", "async", "stream"])
def test_finish_guard_refuses_until_required_items_are_done(mode):
    added, updated = [], []
    model = scripted([], "all done!", [update("r1", "done", note="checked")], "finished")
    agent = Agent(model, extensions=[Todos(required())])
    agent.on("todo_added", added.append)
    agent.on("todo_updated", updated.append)
    result = asyncio.run(drive(agent, mode))
    assert result.message == "finished" and agent.session.outcome == "completed"
    refusal = agent.session.history.entries[2].message
    assert "r1: Check against the library" in refusal and "r2" not in refusal
    assert updated == [
        {"text": "Check against the library", "required": True, "status": "done",
         "note": "checked", "skippable": False, "id": "r1"}
    ]
    assert added == []


def test_model_items_are_never_required_and_rules_are_enforced():
    events = []
    model = scripted(
        [],
        [ToolCall("todo_add", {"text": "Draft summary"}, id="a1")],
        [update("r1", "skipped"), update("zz", "done"), update("r3", "skipped")],
        [ToolCall("todo_update", {"id": "r1", "status": "later"}, id="bad")],
        [update("r1", "done"), ToolCall("todo_list", {}, id="list")],
        "finished",
    )
    todos = Todos([*required(), Todo("May be skipped", required=True, skippable=True)])
    agent = Agent(model, extensions=[todos])
    agent.on("todo_added", events.append)
    assert agent.run("go").message == "finished"
    history = agent.session.history.entries
    assert history[1].tool_results[0]["result"] == "Added as t1."
    errors = [r.get("error", r.get("result")) for r in history[2].tool_results]
    assert "cannot be skipped" in errors[0] and "No todo item" in errors[1] and errors[2] == "r3 is now skipped."
    assert "ValidationError" in history[3].tool_results[0]["error"]
    assert "[x] r1 (required): Check against the library" in history[4].tool_results[1]["result"]
    by_id = {item.id: item for item in todos.items}
    assert by_id["t1"].required is False and by_id["r3"].status == "skipped"
    assert events[0]["id"] == "t1" and todos.open_required() == []


def test_final_answer_is_refused_while_a_required_item_is_open():
    final = ToolCall("final_answer", {"title": "x"}, id="f1")
    model = scripted([], [final], [update("r1", "done")], [ToolCall("final_answer", {"title": "x"}, id="f2")])
    agent = Agent(model, extensions=[Todos(required()), FinalAnswer(Draft, required=True)])
    result = agent.run("go")
    assert "r1: Check against the library" in agent.session.history.entries[1].tool_results[0]["error"]
    assert result.output == Draft(title="x") and result.tool_calls[0].id == "f2"


def test_state_survives_serialization_and_is_not_duplicated():
    agent = Agent(scripted([], [update("r1", "in_progress")], [update("r1", "done")], "ok"), extensions=[Todos(required())])
    agent.run("go")
    saved = json.loads(json.dumps(agent.session.to_dict()))
    restored = Todos(required())
    Agent(scripted([], "x"), session=Session.from_dict(saved), extensions=[restored])
    assert [(item.id, item.status) for item in restored.items] == [("r1", "done"), ("r2", "open")]


def test_placement_keeps_the_system_prompt_stable_by_default():
    def sent(placement):
        model = scripted([], [update("r1", "done")], "ok")
        model.instructions = "Base."
        agent = Agent(model, extensions=[Todos(required(), placement=placement)])
        agent.run("go")
        payload = history_messages(agent.session.history, "", {})
        stored = [m.message for m in agent.session.history.entries]
        return model.instructions, payload[-1]["content"], stored

    instructions, last, stored = sent("message")
    assert instructions == "Base." and "[x] r1 (required)" in last
    assert not any("Todo list" in message for message in stored)
    instructions, last, stored = sent("instructions")
    assert "[x] r1 (required)" in instructions and last == "ok"
    with pytest.raises(ValueError, match="placement"):
        Todos(placement="history")


def test_budget_exhaustion_leaves_open_items_readable():
    todos = Todos(required())
    agent = Agent(scripted([], "done?", "done??", "done???"), max_turns=2, extensions=[todos])
    with pytest.raises(BudgetExceeded):
        agent.run("go")
    assert [item.id for item in todos.open_required()] == ["r1"]
    assert agent.session.extensions["todos"]["items"][0]["status"] == "open"
    assert agent.session.outcome == "exhausted"
