"""Re-run a ledger's recorded inputs against another agent and compare the records."""

from typing import Any

from .ledger import Ledger


def inputs(ledger: Ledger) -> list[dict[str, Any]]:
    """First input of every recorded run; a resumed run has prompt None."""
    runs: list[dict[str, Any]] = []
    for entry in ledger.entries:
        if entry["kind"] == "run_start":
            runs.append({"prompt": None, "role": "user", "open": True})
        elif entry["kind"] == "input" and runs and runs[-1].pop("open", False):
            media = {key: entry[key] for key in ("images", "audio") if key in entry}
            runs[-1].update(prompt=entry["prompt"], role=entry["role"], **media)
    for run in runs:
        run.pop("open", None)
    return runs


def _prepare(agent):
    if agent.session.state != "idle":
        agent.session.reset()


def replay(ledger: Ledger, agent) -> Ledger:
    """Synchronous replay of text and referenced (path/URL) media inputs."""
    fresh = Ledger()
    with fresh.attach(agent):
        for run in inputs(ledger):
            _prepare(agent)
            agent.run(**run)
    return fresh


async def areplay(ledger: Ledger, agent) -> Ledger:
    fresh = Ledger()
    with fresh.attach(agent):
        for run in inputs(ledger):
            _prepare(agent)
            await agent.arun(**run)
    return fresh


def compare(before: Ledger, after: Ledger) -> dict[str, Any]:
    """Side-by-side summaries plus numeric deltas (after minus before)."""
    a, b = before.summary(), after.summary()
    return {
        "before": a,
        "after": b,
        "same_tool_sequence": a["tool_calls"] == b["tool_calls"],
        "delta": {
            key: b[key] - a[key] for key in a if isinstance(a[key], int) and key != "runs"
        },
    }
