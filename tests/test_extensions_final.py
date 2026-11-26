"""The extension seam, runtime tool/instruction API and run-ending final answers."""

import asyncio
import json

import pytest
from pydantic import BaseModel

from synaptic import Agent, Extension, FinalAnswer, Session, ToolCall, UserMem
from synaptic.providers._chat_history import history_messages
from tests.test_execution_recovery import make_tool
from tests.test_tool_errors_and_guards import drive, scripted


class Draft(BaseModel):
    title: str
    score: int


def final(call_id="f1", name="final_answer", **arguments):
    return ToolCall(name, arguments or {"title": "ok", "score": 3}, id=call_id)


def calls_made(agent):
    return sum(type(m).__name__ == "ResponseMem" for m in agent.session.history.entries)


@pytest.mark.parametrize("mode", ["sync", "async", "stream"])
def test_valid_final_call_ends_the_run_without_another_model_call(mode):
    seen, chunks = [], []
    agent = Agent(scripted([], [final()], "never reached"), extensions=[FinalAnswer(Draft)])
    agent.on("final", lambda name, output: seen.append((name, output)))

    async def run():
        if mode != "stream":
            return await drive(agent, mode)
        async for chunk in agent.astream("go"):
            chunks.append(chunk)
        return agent.session.history.entries[-1]

    result = asyncio.run(run())
    assert calls_made(agent) == 1 and agent.session.outcome == "completed"
    assert seen == [("final_answer", Draft(title="ok", score=3))]
    assert result.tool_results[0]["result"] == "accepted" and "incomplete" not in result.extensions
    saved = json.loads(json.dumps(agent.session.to_dict()))
    assert saved["extensions"]["final"] == {"name": "final_answer", "output": {"title": "ok", "score": 3}}
    if mode == "stream":
        assert chunks[-1].is_final and chunks[-1].extensions["output"] == Draft(title="ok", score=3)
    else:
        assert result.output == Draft(title="ok", score=3) and result.output_name == "final_answer"
    restored = Session.from_dict(saved)
    follow = Agent(scripted([], "follow-up"), session=restored, extensions=[FinalAnswer(Draft)])
    assert follow.run("and then?").message == "follow-up" and follow.run is not None


def test_invalid_arguments_and_shared_responses_are_returned_to_the_model():
    lookup = make_tool("lookup", lambda: "data")
    model = scripted(
        [lookup],
        [final("bad", title="x", score="many")],
        [ToolCall("lookup", {}, id="l1"), final("early")],
        [final("good")],
    )
    agent = Agent(model, tools=[lookup], extensions=[FinalAnswer(Draft)])
    result = agent.run("go")
    errors = [m.tool_results for m in agent.session.history.entries[1:3]]
    assert "ValidationError" in errors[0][0]["error"]
    assert errors[1][0]["result"] == "data" and "alone" in errors[1][1]["error"]
    assert result.output.score == 3 and calls_made(agent) == 3
    assert [v["status"] for v in agent.session.history.entries[2].tool_outcomes.values()] == [
        "succeeded",
        "failed",
    ]


def test_required_final_refuses_plain_text_and_optional_allows_it():
    required = Agent(scripted([], "just text", [final()]), extensions=[FinalAnswer(Draft, required=True)])
    assert required.run("go").output_name == "final_answer"
    nudge = required.session.history.entries[2]
    assert isinstance(nudge, UserMem) and "final_answer" in nudge.message
    optional = Agent(scripted([], "just text"), extensions=[FinalAnswer(Draft)])
    result = optional.run("go")
    assert result.message == "just text" and result.output is None and result.output_name is None


def test_dict_schema_and_several_final_tools():
    schema = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}
    agent = Agent(
        scripted([], [ToolCall("final_count", {"n": 2}, id="c")]),
        extensions=[FinalAnswer(Draft), FinalAnswer(schema, name="final_count")],
    )
    result = agent.run("go")
    assert (result.output_name, result.output) == ("final_count", {"n": 2})
    assert agent.tool_names() == {"final_answer", "final_count"}


class Notes(Extension):
    name = "notes"

    def tools(self):
        def remember(text: str) -> str:
            self.state.setdefault("items", []).append(text)
            self.agent.notify("note_saved", text)
            return "saved"

        declaration = {"name": "remember", "parameters": {"type": "object"}}
        from synaptic import Tool

        return [Tool("remember", declaration, remember, add_to_registry=False)]

    def instructions(self):
        return "Notes so far: " + ", ".join(self.state.get("items", []))

    def context(self, entries):
        return [*entries, UserMem("reminder", None)]

    def before_finish(self, final=False):
        return None if self.state.get("items") else "Save one note first."


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_custom_extension_uses_every_hook(mode):
    events = []
    model = scripted([], "too early", [ToolCall("remember", {"text": "a"}, id="n1")], "done")
    model.instructions = "Base."
    agent = Agent(model, extensions=[Notes()])
    agent.on("note_saved", events.append)
    result = asyncio.run(drive(agent, mode))
    assert result.message == "done" and events == ["a"]
    assert model.instructions == "Base.\n\nNotes so far: a"
    assert agent.session.history.entries[2].message == "Save one note first."
    sent = [m["content"] for m in history_messages(agent.session.history, "", {})]
    assert sent[-1] == "reminder" and all(m.message != "reminder" for m in agent.session.history.entries)
    restored = Session.from_dict(json.loads(json.dumps(agent.session.to_dict())))
    assert restored.extensions == {"notes": {"items": ["a"]}}
    with pytest.raises(RuntimeError, match="one agent"):
        Agent(scripted([], "x"), extensions=agent.extensions)


def test_runtime_tools_and_instructions():
    first, second = make_tool("first", lambda: "1"), make_tool("second", lambda: "2")
    model = scripted([], [ToolCall("second", {}, id="s")], "done")
    model.instructions = "Old."
    agent = Agent(model, tools=[first])
    agent.add_tools(second)
    agent.remove_tools("first")
    agent.instructions = "New."
    assert agent.tool_names() == {"second"} and agent.instructions == "New."
    assert agent.run("go").message == "done" and model.instructions == "New."
    assert agent.session.history.entries[1].tool_results[0]["result"] == "2"


def test_agents_without_extensions_are_untouched():
    model = scripted([], "plain")
    model.instructions = "Set by the app."
    agent = Agent(model)
    model.instructions = "Changed directly."
    assert agent.run("go").message == "plain" and model.instructions == "Changed directly."
    assert "extensions" not in agent.session.to_dict() and agent.session.history.compiler is None


def test_unknown_tool_is_returned_only_when_errors_are_returned():
    def agent(mode):
        return Agent(scripted([], [ToolCall("missing", {}, id="m")], "recovered"), tool_errors=mode)

    returned = agent("return")
    assert returned.run("go").message == "recovered"
    assert "not registered" in returned.session.history.entries[1].tool_results[0]["error"]
    with pytest.raises(ValueError, match="not registered"):
        agent("raise").run("go")
