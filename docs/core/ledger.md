# Core: Run Ledger

A `Ledger` is an append-only, ordered record of what an agent run did. Use it to
inspect a run, store it, fork it, or replay its inputs against another model.

```python
from synaptic import Agent, Ledger

ledger = Ledger()
agent = Agent(model, tools=tools, ledger=ledger)
agent.run("Summarise the open tickets")

for entry in ledger.entries:
    print(entry["seq"], entry["kind"])
print(ledger.summary())
```

## Entries

Every entry has `seq`, `at` (UTC ISO time), `kind` and `session_id`.

| Kind | Extra fields |
| --- | --- |
| `run_start` | — |
| `input` | `prompt`, `role` |
| `model` | `provider`, `model`, `message`, `tool_calls`, token counts, `finish_reason`, `latency_ms` |
| `tool_start` | `call_id`, `name`, `args` |
| `tool_end` | `call_id`, `name`, `result` or `error`, `latency_ms` |
| `run_end` | `outcome` (`completed`, `failed`, `cancelled`, `exhausted`) |

Entries are copies: editing what `ledger.entries` returns does not change the record.
A tool that raises has a `tool_start` and no `tool_end`; the run ends `failed`.

## Storing

```python
with open("run.jsonl", "a") as out:
    ledger = Ledger(sink=lambda entry: out.write(json.dumps(entry) + "\n"))
```

`sink` receives each entry as it is appended. A failing sink raises a warning and never
interrupts the run. `ledger.to_list()` / `Ledger.from_list(...)` round-trip the record;
tool results must be JSON-compatible if you serialize it. Synaptic does not choose
where the record is stored.

## Fork, replay, compare

```python
from synaptic.run import replay, compare

branch = ledger.fork(seq=4)          # independent copy up to entry 4
fresh = replay(ledger, other_agent)  # same inputs, different agent or model
print(compare(ledger, fresh))        # summaries, tool-sequence match, numeric deltas
```

`replay` re-sends the recorded text inputs of each run (use `areplay` for async agents).
It really executes the other agent's tools, and recorded images or audio are not re-sent.
The ledger is an audit record, not the resume source: conversation state is restored
from `History`.

A model handoff adds a `handoff` entry with `source` and `target` model names.
