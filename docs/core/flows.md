# Core: Flow And Pipeline

`Flow` and `Pipeline` are deterministic workflow helpers. They are useful when
your app needs fixed steps instead of an agent deciding what to do next.

## Flow

`Flow` routes between named nodes.

```python
from synaptic.run import Flow, NodeResult

flow = Flow()

@flow.node("draft")
def draft(topic):
    res = writer.invoke(f"Draft about {topic}")
    return NodeResult(output=res.message, next="review")

@flow.node("review")
def review(text):
    res = reviewer.invoke(text)
    return res.message

final = flow.run("Synaptic")
```

Use `await flow.arun(...)` when nodes are async.

## Pipeline

`Pipeline` runs steps in order.

```python
from synaptic.run import Pipeline

pipeline = Pipeline()
pipeline.step(lambda x: x.strip())
pipeline.step(lambda x: x.upper())

print(pipeline.run(" hello "))
```

Use `await pipeline.arun(...)` for async steps.

## When To Use These

- Use `Agent` for model-directed tool loops.
- Use `Flow` for branching app logic.
- Use `Pipeline` for simple ordered transformations.
