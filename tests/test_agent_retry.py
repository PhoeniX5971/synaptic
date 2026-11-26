"""Model calls inside an agent run repeat only before any output, under a RetryPolicy."""

import asyncio

import pytest

from synaptic import Agent, Ledger, ResponseChunk, ResponseMem
from synaptic.run import RetryPolicy, transient
from tests.test_tool_errors_and_guards import drive, scripted


class TransportError(Exception):
    pass


class RemoteProtocolError(TransportError):
    pass


class APIError(Exception):
    pass


class APIConnectionError(APIError):
    pass


class Overloaded(Exception):
    status_code = 529


def test_transient_classification_by_type():
    assert transient(RemoteProtocolError("peer closed connection"))
    assert transient(APIConnectionError("reset")) and transient(Overloaded()) and transient(TimeoutError())
    assert not transient(APIError("An error occurred during streaming"))
    assert not transient(ValueError("bad request"))


def flaky(failures, error, after_chunk=False):
    """A model whose first `failures` calls raise, optionally after emitting a chunk."""
    model, state = scripted([], "unused"), {"calls": 0}

    async def stream(**kwargs):
        state["calls"] += 1
        if state["calls"] <= failures:
            if after_chunk:
                yield ResponseChunk("par")
            raise error
        yield ResponseChunk("answer")
        yield ResponseChunk("answer", is_final=True)

    def invoke(**kwargs):
        state["calls"] += 1
        if state["calls"] <= failures:
            raise error
        return ResponseMem("answer", None)

    model.llm.astream, model.llm.invoke, model.state = stream, invoke, state
    return model


@pytest.mark.parametrize("mode", ["sync", "async", "stream"])
def test_failure_before_output_is_retried(mode):
    ledger, model = Ledger(), flaky(2, RemoteProtocolError("dropped"))
    agent = Agent(model, retry=RetryPolicy(retries=3, delay=0), ledger=ledger)
    result = asyncio.run(drive(agent, mode))
    assert result.message == "answer" and model.state["calls"] == 3
    assert [type(m).__name__ for m in agent.session.history.entries] == ["UserMem", "ResponseMem"]
    assert [(e["attempt"], e["error"]) for e in ledger.of("retry")] == [
        (1, "RemoteProtocolError"),
        (2, "RemoteProtocolError"),
    ]


@pytest.mark.parametrize("mode", ["async", "stream"])
def test_failure_after_output_is_never_retried(mode):
    model = flaky(1, RemoteProtocolError("cut mid-reply"), after_chunk=True)
    agent = Agent(model, retry=RetryPolicy(retries=3, delay=0))
    with pytest.raises(RemoteProtocolError):
        asyncio.run(drive(agent, mode))
    partial = agent.session.history.entries[-1]
    assert model.state["calls"] == 1 and partial.message == "par" and partial.extensions["incomplete"]


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_permanent_errors_exhaustion_and_default_do_not_retry(mode):
    def attempts(error, **options):
        model = flaky(5, error)
        with pytest.raises(type(error)):
            asyncio.run(drive(Agent(model, **options), mode))
        return model.state["calls"]

    policy = RetryPolicy(retries=2, delay=0)
    assert attempts(ValueError("bad request"), retry=policy) == 1
    assert attempts(APIError("in-stream error"), retry=policy) == 1
    assert attempts(RemoteProtocolError("dropped"), retry=policy) == 3
    assert attempts(RemoteProtocolError("dropped")) == 1
    custom = RetryPolicy(retries=1, delay=0, predicate=lambda e: transient(e) or isinstance(e, APIError))
    assert attempts(APIError("in-stream error"), retry=custom) == 2
