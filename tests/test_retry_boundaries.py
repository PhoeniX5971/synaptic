import asyncio

import pytest

from synaptic.run import Retry, retry


@pytest.mark.parametrize(
    "options",
    [
        {"max_retries": -1},
        {"max_retries": 1.5},
        {"max_retries": True},
        {"delay": -1},
        {"delay": float("inf")},
        {"delay": float("nan")},
    ],
)
def test_retry_rejects_invalid_configuration(options):
    with pytest.raises(ValueError):
        Retry(object(), **options)
    with pytest.raises(ValueError):
        retry(**options)


@pytest.mark.parametrize("autorun", [True, None])
def test_hidden_effect_before_first_chunk_is_not_retried(autorun):
    class Producer:
        calls = 0

        async def astream(self, **kwargs):
            self.calls += 1
            raise ConnectionError("effect may have happened")
            yield

    async def run():
        producer = Producer()
        producer.autorun = autorun
        with pytest.raises(ConnectionError):
            _ = [c async for c in Retry(producer, delay=0).astream()]
        assert producer.calls == 1

    asyncio.run(run())


def test_cancel_backoff_propagates_and_stops_retrying():
    class Producer:
        autorun = False
        calls = 0

        async def astream(self):
            self.calls += 1
            started.set()
            raise ConnectionError("lost")
            yield

    async def run():
        nonlocal started
        started = asyncio.Event()
        producer = Producer()

        async def consume():
            return [c async for c in Retry(producer, delay=30).astream()]

        task = asyncio.create_task(consume())
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert producer.calls == 1

    started = None
    asyncio.run(run())


def test_validation_failures_are_not_retried_and_zero_means_one_attempt():
    class Producer:
        autorun = False
        calls = 0

        def invoke(self):
            self.calls += 1
            raise ValueError("invalid request")

    producer = Producer()
    with pytest.raises(ValueError):
        Retry(producer, delay=0).invoke()
    assert producer.calls == 1
    with pytest.raises(ValueError):
        Retry(producer, max_retries=0, delay=0, exceptions=(ValueError,)).invoke()
    assert producer.calls == 2
