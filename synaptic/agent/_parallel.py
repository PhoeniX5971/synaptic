"""Opt-in concurrent dispatch for tools their author marked parallel-safe."""

import asyncio


def batches(agent, calls):
    """Group consecutive parallel-safe calls; every other call runs alone, in order."""
    limit = getattr(agent, "max_parallel_tools", 1)
    tools = getattr(agent, "_turn_tools", None) or agent._registry.all()
    group: list = []
    for call in calls:
        tool = tools.get(call.name)
        if limit > 1 and tool is not None and tool.metadata.get("parallel"):
            group.append(call)
            continue
        if group:
            yield group
            group = []
        yield [call]
    if group:
        yield group


async def gather(agent, batch, run):
    """Run one batch; siblings of a failed call finish so their receipts stay truthful."""
    if len(batch) == 1:
        return [await run(batch[0])]
    gate = asyncio.Semaphore(agent.max_parallel_tools)

    async def one(call):
        async with gate:
            return await run(call)

    return await asyncio.gather(*(one(call) for call in batch), return_exceptions=True)
