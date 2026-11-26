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
