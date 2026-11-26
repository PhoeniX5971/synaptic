# Core: Extensions, final answers, todos and skills

An extension adds behaviour to an `Agent` using only its public API. Synaptic ships
three (`FinalAnswer`, `Todos`, `Skills`); you can write your own the same way. Agents
without extensions behave exactly as before. Native tool calling only: none of this is
supported in text/signal mode.

```python
agent = Agent(model, extensions=[FinalAnswer(Report)])
```

## Changing tools and instructions at runtime

```python
agent.add_tools(search, fetch)      # offered from the next turn
agent.remove_tools("fetch")
agent.instructions = "Answer in French."   # the base instructions
```

Extensions append their own text to the base instructions and add their own tools at
every turn. If you use neither extensions nor `agent.instructions`, Synaptic never
touches `model.instructions`.

## Final answers

`FinalAnswer(schema)` adds a tool whose arguments are your schema. A valid call ends the
run immediately, with no further model call.

```python
class Report(BaseModel):
    title: str
    score: int

agent = Agent(model, extensions=[FinalAnswer(Report)])
result = await agent.arun("Review the document")
result.output        # Report(title=..., score=...)
result.output_name   # "final_answer"
```

- `schema` is a Pydantic-style class or a JSON-schema dict (the output is then a dict).
- Invalid arguments go back to the model as a tool error, so it can try again.
- The tool must be the only call in its response. Alongside other calls it is refused, so
  the model cannot finalise before seeing their results.
- By default the model may still end with plain text (`result.output` is then `None`).
  `FinalAnswer(schema, required=True)` refuses that and asks for the tool.
- Use several with different names (`FinalAnswer(Other, name="final_other")`). A run ends
  with exactly one result: the first valid final call.
- The result is also saved as `session.extensions["final"]`, fires a `final` event
  (`name, output`), and with `astream` arrives as one extra last chunk whose
  `extensions` hold `output` and `output_name`.
- The session stays valid afterwards, so follow-up runs on it work.

Any tool can end a run the same way by calling `agent.finish(output)`.

## Todo lists

`Todos` gives the agent a plan it maintains with three tools (`todo_add`, `todo_update`,
`todo_list`). Items you pass in can be required: the run cannot finish until they are done.

```python
todos = Todos([
    Todo("Check against the library", required=True),
    Todo("Confirm jurisdiction", required=True, skippable=True),
])
agent = Agent(model, extensions=[todos, FinalAnswer(Report)])
```

- Statuses are `open`, `in_progress`, `done` and `skipped`. A required item can only be
  skipped if it was created with `skippable=True`.
- Items the model adds itself are never required.
- If the model tries to finish with a required item open, whether by plain text or by a
  final-answer tool, it is told which items are open and the run continues. If
  `max_turns` runs out, `BudgetExceeded` is raised as usual and `todos.open_required()`
  (or `session.extensions["todos"]`) shows what was left.
- Events: `todo_added` and `todo_updated`, each with the full item as a dict.
- The list is saved with the session. Pass the same `Todos([...])` again when restoring;
  items already on the session keep their status and are not duplicated.

**Where the model sees the list.** By default (`placement="message"`) the current list
is added as the last message of every request. It is not stored in history and the
system prompt does not change, so provider prompt caching of a large system prompt keeps
working. `Todos(items, placement="instructions")` puts it at the end of the system
prompt instead, which changes that prompt whenever the list changes.

## Skills

A skill is a named bundle of instructions and tools that the agent loads only when a
task needs it. The model always sees a short catalog (name and description) and a
`load_skill` tool; a skill's instructions and tools appear from the turn after it loads.

```python
review = Skill(
    name="contract_review",
    description="Review a contract against a checklist",   # always visible
    instructions="...long how-to...",                        # visible once loaded
    tools=[read_clause, check_rule],
    todos=[Todo("Check against the library", required=True)],
    final=ReviewResult,            # adds a final_contract_review tool when loaded
    final_required=True,
)
agent = Agent(model, session=session, extensions=[Todos(), Skills([review, figma, house_ui])])
```

- **Skills compose.** Any number can be loaded in the same run; their instructions are
  joined in load order and all their tools are available together. There is no single
  "active" skill.
- Loading twice does nothing. An unknown name, or a skill whose tool names are already
  in use, returns an error to the model.
- A skill's `todos` are added to the todo list when it loads; this needs `Todos()` in
  the extensions.
- A skill can carry its own result schema. Its `final_<name>` tool works like
  `FinalAnswer`; with `final_required=True` the run cannot end in plain text once that
  skill is loaded. A run still ends with exactly one result: with several loaded skills,
  the first valid final call wins. Without a required final, plain text ends the run.
- **Loaded skills stay loaded for the life of the session**, including across
  `Session.to_dict()` / `from_dict()`. Only the names are saved, so pass the same `Skill`
  objects again when restoring. There is no unload.
- Event: `skill_loaded` with the skill's name.

**Prompt caching.** Loading a skill never rewrites existing instruction text: the
order is your base instructions, the fixed catalog, then each loaded skill appended at
the end. Put `Skills` last among extensions that add instructions to keep it that way.
Loading does add the skill's tools, and most providers treat the tool list as part of
the cached prefix, so expect one cache miss per load. `Skills(skills,
preload_tools=True)` declares every skill's tools from the first turn instead: the tool
list then never changes and only instructions load on demand, at the cost of more input
tokens up front.

## Events from inside a tool

`agent.events` is the public `EventBus`; subscribe with `agent.on(name, handler)`.

```python
async def show_result(html: str) -> str:
    await agent.events.aemit("show_result", html)   # delivered immediately
    return "shown"
```

A synchronous tool cannot await. Use `agent.notify("show_result", html)` instead: the
event is queued and delivered right after the current batch of tool calls. Events from
parallel tool calls interleave, so put your own identifier in the payload if the
receiver must tell them apart. A failing handler follows the bus's error policy
(a warning by default) and does not break the run.

## Writing an extension

```python
class Notes(Extension):
    name = "notes"                        # key of this extension's saved state

    def tools(self): ...                  # tools to offer, read every turn
    def instructions(self): ...           # text appended to the instructions, every turn
    def context(self, entries): ...       # what this request sees; not stored in history
    def before_finish(self, final=False): ...   # return a message to refuse finishing
```

`self.agent` is the agent and `self.state` is a JSON-compatible dict saved with the
session (`Session.to_dict()` / `from_dict()`). When `before_finish` returns a message
the run continues with that message shown to the model, within the normal `max_turns`
budget. One extension instance belongs to one agent.
