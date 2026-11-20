"""Check downstream types against an installed wheel, never checkout imports."""

import os
from pathlib import Path
import subprocess
import sys

VALID = """from typing import Any, assert_type, Literal
from synaptic import Model, Provider, History, ResponseMem, ResponseChunk, Agent, ContentBlock, ToolCall
from synaptic import ToolCallResult, set_ensure_ascii, Ledger, Session, Tool, subagent

history = History(size=20)
model = Model(Provider.OPENAI, "fixture", history=history, autorun=False, enforce_schema=True)
assert_type(model.invoke("hello"), ResponseMem)
assert_type(model.propose("plan"), ResponseMem)
assert_type(History.from_dict(history.to_dict()), History)
assert_type(ToolCall("tool", {}).id, str)
assert_type(ContentBlock("text", text="hello").kind, Literal["text", "image", "audio", "resource"])
set_ensure_ascii(False)
ledger, session = Ledger(), Session()
assert_type(ledger.fork(0), Ledger)
assert_type(ledger.append("note"), dict[str, Any])
assert_type(Session.from_dict(session.to_dict()), Session)
assert_type(History(archive_limit=5).archive_limit, int | None)
controlled = Agent(model, session=session, ledger=ledger, max_parallel_tools=2, tool_errors="return")
controlled.handoff(model)
assert_type(controlled.task, dict[str, Any])
assert_type(controlled.as_tool("worker", "Work"), Tool)
assert_type(subagent("worker", "Work", controlled), Tool)
assert_type(model.invoke("hello").latency_ms, int | None)
assert_type(ToolCallResult(ToolCall("tool", {}), None).latency_ms, int | None)
assert_type(ToolCallResult(ToolCall("tool", {}), None).error, str | None)

async def consume() -> None:
    assert_type(await model.ainvoke("hello"), ResponseMem)
    assert_type(await model.apropose("plan"), ResponseMem)
    async for chunk in model.astream("hello"):
        assert_type(chunk, ResponseChunk)
    agent = Agent(model)
    assert_type(agent.run("hello"), ResponseMem)
    assert_type(await agent.arun("hello"), ResponseMem)
    async for item in agent.astream("hello"):
        assert_type(item, ResponseChunk)
"""
# Each line must be rejected: accepting a single one can hide an Any-erased API.
INVALID = [
    "Model(Provider.OPENAI, 42)",
    'Model(Provider.OPENAI, "fixture", max_tokens="many")',
    "model.invoke(42)",
    'model.invoke("hello", images=42)',
    "await model.ainvoke(42)",
    "model.astream(42)",
    "model.propose(42)",
    'model.supply_results("not a response", [])',
    'History(size="many")',
    'ContentBlock("invalid-kind")',
    "agent.run(42)",
    'model.invoke("hello").missing_attribute',
    'set_ensure_ascii("yes")',
    'Ledger(sink=42)',
    'History(archive_limit="many")',
    'agent.as_tool(42, "Work")',
    'Session.from_dict(42)',
    'Agent(model, max_parallel_tools="all")',
    'Agent(model, tool_errors=42)',
    'Model(Provider.OPENAI, "fixture", enforce_schema="yes")',
    'model.invoke("hello").latency_ms + 1',
]


def check_types(python: Path, directory: Path) -> None:
    configuration = directory / "mypy.ini"
    configuration.write_text("[mypy]\nstrict = True\nincremental = False\n")
    valid = directory / "valid_consumer.py"
    valid.write_text(VALID)
    invalid = directory / "invalid_consumer.py"
    prefix = 'from synaptic import Model, Provider, History, ContentBlock, Agent, set_ensure_ascii, Ledger, Session\nmodel = Model(Provider.OPENAI, "fixture")\nagent = Agent(model)\nasync def invalid() -> None:\n'
    invalid.write_text(prefix + "".join("    " + line + "\n" for line in INVALID))
    env = {**os.environ, "MYPYPATH": "", "PYTHONPATH": ""}
    command = [
        sys.executable,
        "-m",
        "mypy",
        "--config-file",
        str(configuration),
        "--python-executable",
        str(python),
        "--no-incremental",
    ]
    positive = subprocess.run(
        [*command, str(valid)], cwd=directory, env=env, capture_output=True, text=True
    )
    if positive.returncode:
        raise AssertionError(positive.stdout + positive.stderr)
    negative = subprocess.run(
        [*command, str(invalid)], cwd=directory, env=env, capture_output=True, text=True
    )
    errors = {
        int(line.split(":")[1])
        for line in negative.stdout.splitlines()
        if line.split(":", 1)[0] in (str(invalid), invalid.name) and ": error:" in line
    }
    expected = set(range(5, 5 + len(INVALID)))
    if negative.returncode != 1 or errors != expected:
        raise AssertionError(
            f"Expected errors on {expected}, got {errors}:\n{negative.stdout}{negative.stderr}"
        )
    subprocess.run(
        [
            str(python),
            "-I",
            "-c",
            'from importlib.resources import files; assert files("synaptic").joinpath("py.typed").is_file()',
        ],
        cwd=directory,
        env=env,
        check=True,
    )
    print(
        f"Installed typing: valid API checks passed; {len(INVALID)} invalid lines rejected",
        flush=True,
    )
