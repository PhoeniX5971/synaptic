import asyncio
import json
import threading

from synaptic.logging import DiagnosticBus, QueuedSink
from synaptic.logging import _bus
from synaptic.logging._writer import JSONLWriter


def test_redaction_precedes_every_sink_and_payloads_are_explicit():
    events, failures = [], []
    bus = DiagnosticBus(secrets=["seed-secret"], on_error=failures.append)
    bus.subscribe(lambda item: (_ for _ in ()).throw(ValueError("seed-secret")))
    bus.subscribe(events.append)
    bus.emit(
        {
            "type": "request",
            "prompt": "private prompt",
            "input_tokens": 12,
            "endpoint": "https://name:password@example.org/path?key=seed-secret",
            "nested": {"authorization": "Bearer abc", "api_key": "seed-secret"},
        }
    )
    encoded = json.dumps(events)
    assert not any(
        value in encoded for value in ("seed-secret", "private prompt", "password", "Bearer abc")
    )
    assert events[0]["input_tokens"] == 12 and events[0]["endpoint"] == "https://example.org/path"
    assert bus.failures == 1 and failures == ["ValueError"]
    payload = DiagnosticBus(payloads=True, secrets=["seed-secret"])
    payload.subscribe(events.append)
    payload.emit({"type": "response", "response": "seed-secret visible"})
    assert events[-1]["response"] == "[redacted] visible"


def test_concurrent_scopes_and_subscriber_mutation_do_not_cross():
    async def run():
        one, two, output = DiagnosticBus(), DiagnosticBus(), []
        one.subscribe(lambda event: event.update(run_id="corrupted"))
        one.subscribe(output.append)
        two.subscribe(output.append)

        async def scoped(bus, name):
            with bus.scope(run_id=name):
                await asyncio.sleep(0)
                _bus.emit({"type": "request"})

        await asyncio.gather(scoped(one, "one"), scoped(two, "two"))
        assert {event["run_id"] for event in output} == {"one", "two"}

    asyncio.run(run())


def test_bounded_queue_reports_loss_and_drains():
    gate, started = threading.Event(), threading.Event()
    messages, errors = [], []

    def sink(event):
        started.set()
        gate.wait(2)
        messages.append(event)

    queued = QueuedSink(sink, capacity=1, on_error=errors.append)
    queued(1)
    assert started.wait(1)
    queued(2)
    queued(3)
    assert queued.dropped == 1 and errors
    gate.set()
    queued.close()
    assert messages == [1, 2]
    failed = QueuedSink(lambda event: 1 / 0, on_error=errors.append)
    failed({})
    failed.close()
    assert failed.failures == 1


def test_rotating_writer_and_event_size_limit(tmp_path):
    path = tmp_path / "events.jsonl"
    writer = JSONLWriter(str(path), max_bytes=100, backups=2)
    for i in range(20):
        writer.write({"type": "event", "i": i, "body": "x" * 40})
    writer.close()
    assert len(list(tmp_path.iterdir())) <= 3
    assert json.loads(path.read_text().splitlines()[-1])["i"] == 19
    output = []
    bus = DiagnosticBus(payloads=True, max_bytes=256)
    bus.subscribe(output.append)
    bus.emit({"type": "event", "response": "x" * 3000})
    assert output[0]["truncated"] is True


def test_oversized_unicode_correlation_stays_within_event_budget():
    output = []
    bus = DiagnosticBus(payloads=True, max_bytes=256)
    bus.subscribe(output.append)
    bus.emit(
        {
            "type": "🙂" * 100,
            "run_id": "🙂" * 100,
            "call_id": "🙂" * 100,
            "session_id": "🙂" * 100,
            "response": "large" * 100,
        }
    )
    assert len(json.dumps(output[0], ensure_ascii=False).encode()) <= 256
