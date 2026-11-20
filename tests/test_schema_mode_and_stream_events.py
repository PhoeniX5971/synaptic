"""Chat-family schema enforcement is selectable; stream diagnostics stand alone."""

import asyncio

import pytest

from synaptic import Model, Provider, ResponseChunk, ResponseFormat
from synaptic.core import _runner
from synaptic.logging import DiagnosticBus
from tests.test_completion_contract import fixture_model

SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}


class Draft:
    @staticmethod
    def model_json_schema():
        return SCHEMA


def response_format(provider, schema=SCHEMA, **options):
    model = Model(
        provider,
        "fixture",
        api_key="unused",
        base_url="http://localhost:8000/v1",
        response_format=ResponseFormat.JSON,
        response_schema=schema,
        **options,
    )
    return model.llm._request([], False, {})["response_format"]


@pytest.mark.parametrize(
    "provider, options, kind",
    [
        (Provider.OPENAI, {}, "json_schema"),
        (Provider.OPENAI, {"enforce_schema": False}, "json_object"),
        (Provider.UNIVERSAL_OPENAI, {}, "json_object"),
        (Provider.UNIVERSAL_OPENAI, {"enforce_schema": True}, "json_schema"),
        (Provider.DEEPSEEK, {}, "json_object"),
        (Provider.DEEPSEEK, {"enforce_schema": True}, "json_schema"),
    ],
)
def test_schema_enforcement_per_adapter(provider, options, kind):
    sent = response_format(provider, **options)
    assert sent["type"] == kind
    if kind == "json_schema":
        assert sent["json_schema"] == {"name": "response", "schema": SCHEMA}


def test_universal_enforces_a_pydantic_style_schema_when_asked():
    sent = response_format(Provider.UNIVERSAL_OPENAI, Draft, enforce_schema=True)
    assert sent["json_schema"]["schema"] == SCHEMA


def captured(payloads, drive):
    events, bus = [], DiagnosticBus(payloads=payloads)
    bus.subscribe(events.append)

    async def run():
        with bus.scope():
            await drive()

    asyncio.run(run())
    return events


@pytest.mark.parametrize("payloads", [True, False])
def test_stream_events_share_an_id_and_the_end_event_stands_alone(payloads):
    async def drive():
        async def one(prompt):
            model = fixture_model([ResponseChunk(prompt.upper()), ResponseChunk("", is_final=True)])
            model.model = "fixture"
            async for _ in _runner.astream(model, prompt, automem=False):
                await asyncio.sleep(0)

        await asyncio.gather(one("alpha"), one("beta"))

    events = captured(payloads, drive)
    ends = [event for event in events if event["type"] == "model.stream.end"]
    starts = {event["model_call_id"]: event for event in events if event["type"] == "model.stream.start"}
    assert len(ends) == 2 and len(starts) == 2
    for end in ends:
        assert end["role"] == "user" and isinstance(end["latency_ms"], int)
        if payloads:
            assert end["response"] == end["prompt"].upper()
            assert starts[end["model_call_id"]]["prompt"] == end["prompt"]
        else:
            assert "prompt" not in end and "response" not in end
            assert end["model_call_id"] in starts


def test_invoke_and_ainvoke_events_carry_a_call_id():
    async def drive():
        model = fixture_model([ResponseChunk("x", is_final=True)])
        model.blacklist = []
        model.model = "fixture"
        _runner.invoke(model, "sync", automem=False)
        await _runner.ainvoke(model, "async", automem=False)

    events = captured(True, drive)
    assert [event["type"] for event in events] == ["model.invoke", "model.ainvoke"]
    assert len({event["model_call_id"] for event in events}) == 2
    assert [event["prompt"] for event in events] == ["sync", "async"]
