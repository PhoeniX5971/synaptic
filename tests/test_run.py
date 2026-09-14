"""Tests for synaptic.run: Retry, @retry, Flow, Pipeline."""
import asyncio
import pytest

from synaptic.run import Flow, NodeResult, Pipeline, Retry, retry


# ---------------------------------------------------------------------------
# @retry decorator — sync
# ---------------------------------------------------------------------------
class TestRetryDecorator:
    def test_succeeds_first_try(self):
        calls = []

        @retry(max_retries=3, delay=0)
        def fn():
            calls.append(1)
            return "ok"

        assert fn() == "ok"
        assert len(calls) == 1

    def test_retries_on_failure_then_succeeds(self):
        calls = []

        @retry(max_retries=3, delay=0)
        def flaky():
            calls.append(1)
            if len(calls) < 3:
                raise ValueError("not yet")
            return "done"

        assert flaky() == "done"
        assert len(calls) == 3

    def test_raises_after_max_retries(self):
        @retry(max_retries=2, delay=0)
        def always_fails():
            raise RuntimeError("nope")

        with pytest.raises(RuntimeError, match="nope"):
            always_fails()

    def test_specific_exception_filter(self):
        calls = []

        @retry(max_retries=3, delay=0, exceptions=(ValueError,))
        def fn():
            calls.append(1)
            if len(calls) < 2:
                raise ValueError("retry me")
            return "ok"

        assert fn() == "ok"

    def test_non_matching_exception_not_retried(self):
        calls = []

        @retry(max_retries=3, delay=0, exceptions=(ValueError,))
        def fn():
            calls.append(1)
            raise TypeError("don't retry")

        with pytest.raises(TypeError):
            fn()
        assert len(calls) == 1  # no retries


# ---------------------------------------------------------------------------
# @retry decorator — async
# ---------------------------------------------------------------------------
class TestRetryDecoratorAsync:
    def test_async_succeeds(self):
        calls = []

        @retry(max_retries=3, delay=0)
        async def fn():
            calls.append(1)
            if len(calls) < 2:
                raise ValueError("wait")
            return "async ok"

        result = asyncio.run(fn())
        assert result == "async ok"
        assert len(calls) == 2

    def test_async_raises_after_max(self):
        @retry(max_retries=2, delay=0)
        async def fn():
            raise RuntimeError("async fail")

        with pytest.raises(RuntimeError):
            asyncio.run(fn())


# ---------------------------------------------------------------------------
# Retry wrapper class
# ---------------------------------------------------------------------------
class TestRetryWrapper:
    class FakeModel:
        autorun = False
        instructions = "original"
        _count = 0

        def invoke(self, prompt):
            self.__class__._count += 1
            if self.__class__._count < 3:
                raise ConnectionError("down")
            return f"response to: {prompt}"

        async def ainvoke(self, prompt):
            return self.invoke(prompt)

        async def astream(self, prompt, **kwargs):
            raise ConnectionError("stream down")
            yield  # make it a generator

    def setup_method(self):
        self.FakeModel._count = 0

    def test_invoke_retries(self):
        m = self.FakeModel()
        r = Retry(m, max_retries=3, delay=0)
        result = r.invoke("hello")
        assert "hello" in result

    def test_invoke_raises_after_max(self):
        m = self.FakeModel()
        r = Retry(m, max_retries=1, delay=0)
        with pytest.raises(ConnectionError):
            r.invoke("test")

    def test_ainvoke_retries(self):
        self.FakeModel._count = 0
        m = self.FakeModel()
        r = Retry(m, max_retries=3, delay=0)
        result = asyncio.run(r.ainvoke("hi"))
        assert "hi" in result

    def test_proxy_attribute_get(self):
        m = self.FakeModel()
        r = Retry(m, max_retries=1, delay=0)
        assert r.instructions == "original"

    def test_proxy_attribute_set(self):
        m = self.FakeModel()
        r = Retry(m, max_retries=1, delay=0)
        r.instructions = "updated"
        assert m.instructions == "updated"

    def test_astream_retries_on_failure(self):
        class StreamModel:
            autorun = False
            _attempts = 0

            async def astream(self, prompt, **kwargs):
                self.__class__._attempts += 1
                if self.__class__._attempts < 2:
                    raise ConnectionError("not ready")
                yield "chunk1"
                yield "chunk2"

        m = StreamModel()
        r = Retry(m, max_retries=3, delay=0)

        async def run():
            chunks = []
            async for c in r.astream("test"):
                chunks.append(c)
            return chunks

        result = asyncio.run(run())
        assert result == ["chunk1", "chunk2"]
        assert StreamModel._attempts == 2


# ---------------------------------------------------------------------------
# Flow
# ---------------------------------------------------------------------------
class TestFlow:
    def test_basic_two_node_flow(self):
        flow = Flow()

        @flow.node("a")
        def step_a(state):
            return NodeResult(output=state + " world", next="b")

        @flow.node("b")
        def step_b(state):
            return NodeResult(output=state.upper(), next=None)

        flow.entry("a")
        assert flow.run("hello") == "HELLO WORLD"

    def test_stops_when_next_is_none(self):
        flow = Flow()
        flow.add("only", lambda s: NodeResult(output=s * 2, next=None))
        flow.entry("only")
        assert flow.run(5) == 10

    def test_non_noderesult_return_terminates(self):
        flow = Flow()
        flow.add("x", lambda s: s + "!")
        flow.entry("x")
        assert flow.run("hi") == "hi!"

    def test_no_entry_raises(self):
        flow = Flow()
        flow.add("a", lambda s: NodeResult(output=s, next=None))
        with pytest.raises(ValueError, match="entry"):
            flow.run("test")

    def test_missing_node_raises(self):
        flow = Flow()
        flow.add("start", lambda s: NodeResult(output=s, next="ghost"))
        flow.entry("start")
        with pytest.raises(ValueError, match="ghost"):
            flow.run("x")

    def test_max_steps_exceeded_raises(self):
        flow = Flow(max_steps=3)
        # infinite loop: always routes back to self
        flow.add("loop", lambda s: NodeResult(output=s, next="loop"))
        flow.entry("loop")
        with pytest.raises(RuntimeError, match="max_steps"):
            flow.run("x")

    def test_conditional_routing(self):
        flow = Flow()
        flow.add("router", lambda s: NodeResult(output=s, next="yes" if s > 0 else "no"))
        flow.add("yes", lambda s: NodeResult(output="positive", next=None))
        flow.add("no", lambda s: NodeResult(output="negative", next=None))
        flow.entry("router")

        assert flow.run(1) == "positive"
        assert flow.run(-1) == "negative"

    def test_async_node_in_arun(self):
        flow = Flow()

        @flow.node("async_step")
        async def step(state):
            return NodeResult(output=state * 3, next=None)

        flow.entry("async_step")
        result = asyncio.run(flow.arun(4))
        assert result == 12

    def test_sync_node_in_arun(self):
        flow = Flow()
        flow.add("s", lambda x: NodeResult(output=x + 1, next=None))
        flow.entry("s")
        assert asyncio.run(flow.arun(10)) == 11

    def test_run_raises_on_async_node(self):
        flow = Flow()

        @flow.node("a")
        async def step(s):
            return NodeResult(output=s, next=None)

        flow.entry("a")
        with pytest.raises(RuntimeError, match="async"):
            flow.run("test")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class TestPipeline:
    def test_basic_chain(self):
        p = Pipeline([
            lambda s: s + " world",
            lambda s: s.upper(),
            lambda s: s + "!",
        ])
        assert p.run("hello") == "HELLO WORLD!"

    def test_single_step(self):
        p = Pipeline([lambda s: s * 2])
        assert p.run(5) == 10

    def test_empty_pipeline(self):
        p = Pipeline([])
        assert p.run("unchanged") == "unchanged"

    def test_async_arun(self):
        p = Pipeline([lambda s: s + 1, lambda s: s * 2])
        assert asyncio.run(p.arun(3)) == 8

    def test_async_step_in_arun(self):
        async def double(x):
            return x * 2

        p = Pipeline([double])
        assert asyncio.run(p.arun(5)) == 10

    def test_sync_raises_on_async_step(self):
        async def async_step(s):
            return s

        p = Pipeline([async_step])
        with pytest.raises(RuntimeError, match="arun"):
            p.run("test")
