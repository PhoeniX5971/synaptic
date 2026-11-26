"""Skills load on demand, compose, persist with the session and can carry their own result."""

import asyncio
import json

import pytest
from pydantic import BaseModel

from synaptic import Agent, Model, Provider, Session, Skill, Skills, Todo, Todos, Tool, ToolCall
from tests.test_tool_errors_and_guards import drive, scripted


class Review(BaseModel):
    verdict: str


class Design(BaseModel):
    frames: int


def tool(name, result="ok"):
    declaration = {"name": name, "description": "", "parameters": {"type": "object"}}
    return Tool(name, declaration, lambda: result, add_to_registry=False)


def load(name, call_id=None):
    return ToolCall("load_skill", {"name": name}, id=call_id or f"load-{name}")


def catalog():
    return [
        Skill("review", "Review a contract", "Follow the checklist.", [tool("read_clause")],
              todos=[Todo("Check the library", required=True)], final=Review, final_required=True),
        Skill("figma", "Drive Figma", "Use frames.", [tool("draw")], final=Design),
        Skill("house_ui", "Our UI rules", "Use the house palette.", [tool("palette")]),
    ]


@pytest.mark.parametrize("mode", ["sync", "async", "stream"])
def test_loading_changes_tools_and_instructions_on_the_next_turn(mode):
    events = []
    model = scripted([], [load("figma")], [ToolCall("draw", {}, id="d1")], "done")
    model.instructions = "Base."
    agent = Agent(model, extensions=[Skills(catalog()), Todos()])
    agent.on("skill_loaded", events.append)
    assert agent.tool_names() == {"load_skill", "todo_add", "todo_update", "todo_list"}
    assert "- figma: Drive Figma" in agent._compose() and "Use frames." not in agent._compose()
    result = asyncio.run(drive(agent, mode))
    assert result.message == "done" and events == ["figma"]
    assert agent.session.history.entries[2].tool_results[0]["result"] == "ok"
    assert {"draw", "final_figma"} <= agent.tool_names() and "read_clause" not in agent.tool_names()
    assert model.instructions.startswith("Base.\n\nSkills you can load")
    assert model.instructions.endswith("## Skill: figma\nUse frames.")


def test_real_model_binding_follows_loaded_skills():
    skills = Skills(catalog())
    model = Model(Provider.OPENAI, "fixture", api_key="unused", instructions="Base.")
    agent = Agent(model, extensions=[skills, Todos()])
    agent._prepare_turn()
    before = {t["function"]["name"] for t in model.llm.openai_tools}
    skills.load("figma")
    agent._prepare_turn()
    after = {t["function"]["name"] for t in model.llm.openai_tools}
    assert "draw" not in before and after - before == {"draw", "final_figma"}
    assert model.llm.instructions.endswith("## Skill: figma\nUse frames.")


def test_several_skills_compose_in_load_order():
    model = scripted(
        [],
        [load("figma"), load("house_ui")],
        [ToolCall("palette", {}, id="p"), ToolCall("draw", {}, id="d")],
        [load("figma", "again"), load("nope", "bad")],
        "done",
    )
    skills = Skills(catalog())
    agent = Agent(model, extensions=[skills, Todos()])
    assert agent.run("go").message == "done" and skills.loaded == ["figma", "house_ui"]
    text = agent._compose()
    assert text.index("## Skill: figma") < text.index("## Skill: house_ui")
    assert {"draw", "palette", "final_figma"} <= agent.tool_names()
    history = agent.session.history.entries
    assert [r["result"] for r in history[2].tool_results] == ["ok", "ok"]
    assert "already loaded" in history[3].tool_results[0]["result"]
    assert "ValidationError" in history[3].tool_results[1]["error"]
    with pytest.raises(ValueError, match="Unknown skill"):
        skills.load("nope")


def test_tool_name_clash_is_reported_to_the_model():
    clashing = [Skill("a", "A", tools=[tool("shared")]), Skill("b", "B", tools=[tool("shared")])]
    agent = Agent(scripted([], [load("a"), load("b")], "done"), extensions=[Skills(clashing)])
    agent.run("go")
    error = agent.session.history.entries[1].tool_results[1]["error"]
    assert "tool names already in use: ['shared']" in error and agent.extensions[0].loaded == ["a"]
    with pytest.raises(ValueError, match="already in use"):
        Agent(scripted([], "x"), extensions=[Skills(clashing, preload_tools=True)])


def test_skill_todos_and_required_final_gate_the_run():
    done = ToolCall("todo_update", {"id": "review:1", "status": "done"}, id="u")
    model = scripted(
        [],
        [load("review")],
        "finished in prose",
        [ToolCall("final_review", {"verdict": "fine"}, id="f1")],
        [done],
        [ToolCall("final_review", {"verdict": "fine"}, id="f2")],
    )
    agent = Agent(model, extensions=[Skills(catalog()), Todos()])
    result = agent.run("go")
    history = agent.session.history.entries
    assert "final_review" in history[3].message and "review:1: Check the library" in history[3].message
    assert "review:1" in history[4].tool_results[0]["error"]
    assert (result.output_name, result.output) == ("final_review", Review(verdict="fine"))
    with pytest.raises(ValueError, match="add Todos"):
        Agent(scripted([], "x"), extensions=[Skills(catalog())])


def test_plain_text_is_allowed_when_no_loaded_skill_requires_a_final():
    agent = Agent(scripted([], [load("figma")], "just chatting"), extensions=[Skills(catalog()), Todos()])
    result = agent.run("go")
    assert result.message == "just chatting" and result.output is None


def test_first_valid_final_of_several_loaded_skills_ends_the_run():
    model = scripted([], [load("figma"), load("house_ui")], [ToolCall("final_figma", {"frames": 2}, id="f")])
    result = Agent(model, extensions=[Skills(catalog()), Todos()]).run("go")
    assert (result.output_name, result.output) == ("final_figma", Design(frames=2))


def test_loaded_skills_persist_with_the_session():
    first = Agent(scripted([], [load("figma")], "ok"), extensions=[Skills(catalog()), Todos()])
    first.run("go")
    saved = json.loads(json.dumps(first.session.to_dict()))
    assert saved["extensions"]["skills"] == {"loaded": ["figma"]}
    later = Agent(scripted([], "x"), session=Session.from_dict(saved), extensions=[Skills(catalog()), Todos()])
    assert "draw" in later.tool_names() and "## Skill: figma" in later._compose()
    with pytest.raises(ValueError, match="not provided"):
        Agent(scripted([], "x"), session=Session.from_dict(saved), extensions=[Skills(catalog()[:1]), Todos()])


def test_preloaded_tools_never_change_and_unloaded_finals_are_refused():
    model = scripted([], [ToolCall("final_figma", {"frames": 1}, id="early")], [load("figma")],
                     [ToolCall("final_figma", {"frames": 1}, id="ok")])
    agent = Agent(model, extensions=[Skills(catalog(), preload_tools=True), Todos()])
    before = agent.tool_names()
    result = agent.run("go")
    assert before == agent.tool_names() and {"draw", "palette", "read_clause", "final_review"} <= before
    assert "Load the figma skill first" in agent.session.history.entries[1].tool_results[0]["error"]
    assert result.output == Design(frames=1) and "Use frames." in agent._compose()
