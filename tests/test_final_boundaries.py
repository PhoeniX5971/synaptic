import asyncio
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest

from synaptic import Model, Provider, Tool, ToolCall, ToolRegistry, ResponseChunk, ResponseMem, History
from synaptic.core import _runner
from synaptic.mcp import MCPHub, StdioMCPServer
from synaptic.mcp.tools import make_tool
from synaptic.providers._lifecycle import AdapterLifecycle
from tests.test_completion_contract import fixture_model
from tests.test_mcp import FakeSession


def test_models_default_to_separate_registries_histories_and_clients():
    class Adapter:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.synaptic_tools, self.bound_tools = [], []
            self.subscription = self.tool_registry.on_change(self._invalidate_tools)
        def _invalidate_tools(self):
            self.synaptic_tools = list(self.tool_registry.all().values())
        def close(self):
            self.subscription.close()
    with patch('synaptic.core._factory.GeminiAdapter', Adapter):
        one, two = Model(Provider.GEMINI, 'one'), Model(Provider.GEMINI, 'two')
    one.tool_registry.register(Tool('only_one', {}, add_to_registry=False))
    assert one.history is not two.history and [t.name for t in one.tools] == ['only_one'] and two.tools == []
    one.close()
    two.close()
    assert not one.tool_registry._callbacks and not two.tool_registry._callbacks


def test_active_model_turn_executes_captured_definition():
    async def run():
        effects = []
        original = Tool('effect', {'parameters': {'type': 'object'}}, lambda: effects.append('original'), add_to_registry=False)
        replacement = Tool('effect', {'parameters': {'type': 'object'}}, lambda: effects.append('replacement'), add_to_registry=False)
        model = fixture_model([ResponseChunk('', is_final=True)])
        model.blacklist, model.autorun = [], True
        model.llm.synaptic_tools = [original]
        async def source(**kwargs):
            model.llm.synaptic_tools = [replacement]
            yield ResponseChunk('', function_call=ToolCall('effect', {}))
            yield ResponseChunk('', is_final=True)
        model.llm.astream = source
        await _runner.ainvoke(model, 'prompt')
        assert effects == ['original']
    asyncio.run(run())


def test_cleanup_attempts_all_clients_and_failed_close_can_retry():
    async def run():
        adapter = AdapterLifecycle()
        attempts = []
        async def bad():
            attempts.append('bad')
            if attempts.count('bad') == 1:
                raise RuntimeError('close failed')
        async def good():
            attempts.append('good')
        adapter.async_client, adapter._async_client = NS(close=bad), NS(close=good)
        adapter.client = NS(close=lambda: attempts.append('sync'))
        with pytest.raises(ExceptionGroup):
            await adapter.aclose()
        assert attempts == ['bad', 'good', 'sync']
        await adapter.aclose()
        await adapter.aclose()
        assert attempts == ['bad', 'good', 'sync', 'bad']
    asyncio.run(run())


def test_declared_mcp_output_schema_and_metadata_survive():
    async def run():
        async def call(name, arguments):
            return NS(content=[], structuredContent={'count': 'invalid'}, isError=False)
        definition = NS(name='count', inputSchema={'type': 'object'},
            outputSchema={'type': 'object', 'properties': {'count': {'type': 'integer'}}, 'required': ['count']},
            annotations={'readOnlyHint': True})
        tool = make_tool('server', definition, NS(call_tool=call))
        assert tool.metadata['outputSchema'] == definition.outputSchema
        assert tool.metadata['annotations']['readOnlyHint'] is True
        with pytest.raises(Exception):
            await tool.run()
    asyncio.run(run())


@pytest.mark.parametrize('action', ['close', 'unload'])
def test_catalog_in_flight_cannot_resurrect_after_close_or_unload(action):
    async def run():
        hub = MCPHub([StdioMCPServer('fixture', 'unused')])
        gate, started = asyncio.Event(), asyncio.Event()
        class Slow(FakeSession):
            async def list_tools(self, refresh=False):
                started.set()
                await gate.wait()
                return await super().list_tools(refresh)
        hub.sessions['fixture'] = Slow()
        load = asyncio.create_task(hub.load_server('fixture'))
        await started.wait()
        if action == 'unload':
            hub.unload_server('fixture')
            close = None
        else:
            close = asyncio.create_task(hub.close())
            await asyncio.sleep(0)
        gate.set()
        with pytest.raises(RuntimeError):
            await load
        if close:
            await close
        else:
            await hub.close()
        assert not hub.loaded and not hub.registry.all()
    asyncio.run(run())


def test_idle_eviction_and_backoff_are_enforced():
    async def run():
        from synaptic.mcp._hub_runtime import HubRuntime
        sessions = {'one': FakeSession(), 'two': FakeSession()}
        runtime = HubRuntime(sessions, idle_timeout=0.001, reconnect_backoff=1)
        async with runtime.operation('one'):
            pass
        await asyncio.sleep(0.01)
        async with runtime.operation('two'):
            assert sessions['one'].closed
        with pytest.raises(ConnectionError):
            async with runtime.operation('two'):
                raise ConnectionError('failed')
        with pytest.raises(ConnectionError, match='backoff'):
            async with runtime.operation('two'):
                pytest.fail('must not enter operation')
    asyncio.run(run())
