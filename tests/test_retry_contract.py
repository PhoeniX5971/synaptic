import asyncio
from types import SimpleNamespace

import pytest

from synaptic.run import Retry
from synaptic.run._retry_policy import RetryPolicy, retry_after, transient


def test_classification_retry_after_and_timing_bounds():
    rate = SimpleNamespace(response=SimpleNamespace(status_code=429, headers={"Retry-After": "2"}))
    assert transient(rate) and retry_after(rate) == 2
    assert not transient(ValueError("malformed"))
    assert transient(ConnectionError())
    assert not transient(SimpleNamespace(status_code=401))
    policy = RetryPolicy(delay=1, jitter=0, max_delay=3)
    assert policy.pause(0, rate, 10) == 2
    assert policy.pause(2, ConnectionError(), 10) == 3
    assert policy.pause(0, rate, 1) is None
    rate.response.headers["Retry-After"] = "10"
    assert policy.pause(0, rate, 20) is None
    for _ in range(20):
        assert 0.8 <= RetryPolicy(delay=1).pause(0, ConnectionError(), 10) <= 1


def test_async_deadline_and_uncertain_effect_barrier():
    async def run():
        calls = []

        class Slow:
            autorun = False

            async def ainvoke(self):
                calls.append(1)
                await asyncio.sleep(1)

        with pytest.raises(TimeoutError):
            await Retry(Slow(), deadline=0.02).ainvoke()
        assert calls == [1]

        class Effectful:
            autorun = True

            async def ainvoke(self):
                calls.append(2)
                raise ConnectionError("effect may have happened")

        with pytest.raises(ConnectionError):
            await Retry(Effectful(), delay=0).ainvoke()
        assert calls == [1, 2]

    asyncio.run(run())
