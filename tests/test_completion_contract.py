import asyncio
from types import SimpleNamespace

import pytest

from synaptic import Agent, EventBus, Provider, ResponseChunk, ResponseMem
from synaptic.core import _runner
from synaptic.core.base import History


def fixture_model(chunks):
    async def stream(**kwargs):
        for chunk in chunks:
            yield chunk

    final = chunks[-1]
    llm = SimpleNamespace(
        astream=stream,
        invoke=lambda **kwargs: ResponseMem(
            final.text,
            None,
            input_tokens=final.input_tokens,
            output_tokens=final.output_tokens,
            finish_reason=final.finish_reason,
            extensions=final.extensions,
        ),
    )
    return SimpleNamespace(
        llm=llm,
        autorun=False,
        automem=True,
        history=History(),
        instructions="",
        provider=Provider.OPENAI,
        bind_tools=lambda tools: None,
    )


@pytest.mark.parametrize("tokens", [(12, 3), (0, 0), (None, None)])
@pytest.mark.parametrize("final_only", [False, True])
def test_completion_equivalent_across_invocation_and_agent(tokens, final_only):
    chunks = [
        ResponseChunk(
            "answer",
            is_final=True,
            input_tokens=tokens[0],
            output_tokens=tokens[1],
            finish_reason="stop",
            extensions={"request_id": "fixture"},
        )
    ]
    if not final_only:
        chunks.insert(0, ResponseChunk("answer"))

    async def run():
        model = fixture_model(chunks)
        sync = _runner.invoke(model, "prompt")
        async_mem = await _runner.ainvoke(model, "prompt")
        model.history = History()
        _ = [chunk async for chunk in _runner.astream(model, "prompt")]
        streamed = model.history.entries[-1]
        agent = Agent(fixture_model(chunks), events=EventBus())
        _ = [chunk async for chunk in agent.astream("prompt")]
        for mem in [sync, async_mem, streamed, agent.session.history.entries[-1]]:
            assert mem.message == "answer"
            assert (mem.input_tokens, mem.output_tokens) == tokens
            assert mem.finish_reason == "stop"
            assert mem.extensions == {"request_id": "fixture"}

    asyncio.run(run())
