"""Task state travels with the session; context is compiled per request; runs resume."""

import json

import pytest

from synaptic import Agent, History, ResponseMem, Session, ToolCall, UserMem
from synaptic.providers._chat_history import history_messages
from tests.test_execution_recovery import make_model, make_tool


def restored(session):
    return Session.from_dict(json.loads(json.dumps(session.to_dict())))


def test_task_state_is_shared_with_tools_and_serialized():
    session = Session(metadata={"tenant": "a"})
    agent = None

    def mark():
        agent.task["done"] = agent.task.get("done", 0) + 1
        return "ok"

    tool = make_tool("mark", mark)
    agent = Agent(
        make_model([tool], [ToolCall("mark", {}, id="c1")]),
        tools=[tool],
        session=session,
        max_turns=1,
    )
    with pytest.raises(RuntimeError, match="max_turns"):
        agent.run("go")
    assert session.task == {"done": 1} and agent.task is session.task
    copy = restored(session)
    assert (copy.id, copy.task, copy.metadata) == (session.id, {"done": 1}, {"tenant": "a"})
    assert copy.state == "idle" and copy.outcome == "exhausted"
    assert [m.message for m in copy.history.entries] == ["go", ""]
    assert copy.history.entries[-1].tool_results[0]["result"] == "ok"
    with pytest.raises(ValueError, match="version"):
        Session.from_dict({"version": 2})


def test_compiler_projects_context_without_touching_history():
    history = History([UserMem("old", None), ResponseMem("reply", None), UserMem("new", None)])
    history.compiler = lambda mems: mems[-1:]
    assert [m.message for m in history.effective_mems()] == ["new"]
    assert [m["content"] for m in history_messages(history, "", {})] == ["new"]
    assert len(history.entries) == 3
    assert History.from_dict(history.to_dict()).compiler is None


def test_agent_context_receives_entries_and_task():
    seen = []

    def compile_context(entries, task):
        seen.append((len(entries), dict(task)))
        return [UserMem(f"goal: {task['goal']}", None), *entries[-1:]]

    agent = Agent(make_model([], []), context=compile_context)
    agent.task["goal"] = "ship"
    agent.run("first")
    agent.session.reset()
    agent.run("second")
    sent = [m["content"] for m in history_messages(agent.session.history, "", {})]
    assert sent[0] == "goal: ship" and len(sent) == 2
    assert seen[-1] == (4, {"goal": "ship"}) and len(agent.session.history.entries) == 4


def test_restored_session_resumes_without_a_new_prompt():
    tool = make_tool("effect", lambda: "ok")
    first = Agent(make_model([tool], [ToolCall("effect", {}, id="c1")]), tools=[tool], max_turns=1)
    with pytest.raises(RuntimeError, match="max_turns"):
        first.run("go")
    session = restored(first.session)
    final = Agent(make_model([], []), session=session).run(None)
    assert final.tool_calls == [] and session.outcome == "completed"
    assert [type(m).__name__ for m in session.history.entries] == [
        "UserMem",
        "ResponseMem",
        "ResponseMem",
    ]


def test_restored_unresolved_exchange_blocks_resume():
    def boom():
        raise LookupError("boom")

    tool = make_tool("effect", boom)
    first = Agent(make_model([tool], [ToolCall("effect", {}, id="c1")]), tools=[tool])
    with pytest.raises(LookupError):
        first.run("go")
    session = restored(first.session)
    with pytest.raises(ValueError, match="Reconcile"):
        Agent(make_model([], []), session=session).run(None)
