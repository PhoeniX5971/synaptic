# Retry policy and rationale

Retry wraps safe model generation. Autorun effects prohibit automatic retries; unknown
producers need retry_safe=True. Any yielded stream chunk is a replay barrier. A tool
failure is not a signal to repeat uncertain effects. Generic @retry remains available
for callers that explicitly own repetition safety.

`_retry_policy.py` classifies transport/timeouts, 408/429/5xx and known SDK transient
exceptions. Authentication, validation and programming errors do not match the default.
It validates finite timing, nonnegative retry count/delay, bounded exponential backoff
and jitter. Retry-after is a minimum: if it exceeds the maximum delay or remaining
deadline, do not retry early. Exceptions propagate with their original context.

Async calls/streams have a total deadline; cancelling during backoff propagates.
Sync deadlines bound retry scheduling but cannot preempt an already-running function.
Sources close in streaming finalization. The decorator's default broad filter retains
legacy caller responsibility; the model wrapper uses transient classification.

Validation: test_retry_contract, test_retry_boundaries and test_run pass 41 tests.
They cover timing/status classification, server delay, deadlines, zero/invalid counts,
pre-output retry, visible-output failure and the uncertain-effect barrier.

## Retrying inside an agent run

```python
from synaptic.run import RetryPolicy

agent = Agent(model, retry=RetryPolicy(retries=3, delay=0.5))
```

The policy applies to each model call of the run, in `run`, `arun` and `astream`. A call
is repeated only if it failed **before producing any output**; once a chunk has
arrived, the failure is raised and the partial response is kept in history, because
output is never replayed. Tool effects are never repeated by this. Each repeat is
recorded as a `retry` ledger entry and an `agent.retry` diagnostic event.

Pass the plain `Model` to the agent. A `Retry(model)` wrapper only takes effect for
synchronous `Agent.run`; `arun` and `astream` read the model's stream directly and
bypass it.

## What counts as temporary

`synaptic.run.transient(error)` is the default test: connection and timeout errors,
HTTP 408, 429 and 5xx, the SDKs' connection/timeout/rate-limit errors, and HTTP
transport errors (a connection or stream that breaks or times out while being read).

A status-less SDK `APIError` (an error event the server sends inside a stream) is **not**
treated as temporary, because it can also report a permanent failure. To retry it
anyway, supply your own test:

```python
RetryPolicy(retries=2, predicate=lambda e: transient(e) or type(e).__name__ == "APIError")
```
