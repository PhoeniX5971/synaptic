#!/usr/bin/env python3
"""CLI demo: multi-turn thinking blocks with live streaming."""
import asyncio
import sys
from dataclasses import dataclass

from synaptic import (
    Agent, BlockDelta, BlockDone, BlockStarted,
    Event, Model, Provider, SignalMode, Tool,
)
from synaptic.agent.events import EventBus
from synaptic.core.tool import ToolCall
from synaptic.signal.events import ToolCallDone, ToolCallStarted


# ── custom events ────────────────────────────────────────────────────────────

@dataclass
class ThinkingChunk:
    delta: str

@dataclass
class ThinkingDone:
    content: str

class AppEvent(Event):
    ThinkingChunk = "ThinkingChunk"
    ThinkingDone  = "ThinkingDone"


# ── thinking block handler ───────────────────────────────────────────────────

def thinking_handler(evt):
    if isinstance(evt, BlockStarted):
        return [ToolCallStarted(name="think")]
    if isinstance(evt, BlockDelta):
        return [ThinkingChunk(delta=evt.delta)]
    if isinstance(evt, BlockDone):
        return [
            ThinkingDone(content=evt.content),
            ToolCallDone(call=ToolCall(name="think", args={"content": evt.content})),
        ]
    return []


# ── hidden think tool (execution only, never shown to model as a tool) ───────

def _think(content: str) -> str:
    return "Thought noted. Keep thinking or give your final answer in <message>."

_THINK_TOOL = Tool(
    name="think",
    declaration={
        "name": "think",
        "description": "Internal reasoning step.",
        "parameters": {
            "type": "object",
            "properties": {"content": {"type": "string"}},
            "required": ["content"],
        },
    },
    function=_think,
    add_to_registry=False,
)


# ── model instructions ───────────────────────────────────────────────────────

_INSTRUCTIONS = """
## Thinking Protocol
You reason step by step using <thinking> blocks before answering.
Each thinking block lets you explore further — use as many as you need.

<thinking>
your reasoning here, as long as needed
</thinking>

When you are confident in your answer, write it in a <message> block:

<message>
your final answer here
</message>

Rules:
- Always start with at least one <thinking> block.
- Never put your final answer inside a thinking block.
- Never output bare text outside a block.
- More thinking leads to better answers.
"""


# ── terminal output ──────────────────────────────────────────────────────────

GRAY   = "\033[90m"
YELLOW = "\033[33m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

thought_n = [0]

def on_thinking_start(e):
    thought_n[0] += 1
    print(f"\n{GRAY}── thinking {thought_n[0]} ──{RESET}", flush=True)

def on_thinking_chunk(e):
    print(f"{GRAY}{e.delta}{RESET}", end="", flush=True)

def on_thinking_done(e):
    print(f"\n{GRAY}── /thinking ──{RESET}\n", flush=True)

def on_text_delta(e):
    print(e.text, end="", flush=True)

def on_turn_complete(e):
    label = "thinking turn" if e.tool_calls else "done"
    print(f"\n{GRAY}[{label} — {e.input_tokens}in {e.output_tokens}out]{RESET}\n", flush=True)


# ── main ─────────────────────────────────────────────────────────────────────

async def main():
    question = " ".join(sys.argv[1:]) or input("Question: ")

    model = Model(
        provider=Provider.UNIVERSAL_OPENAI,
        model="qwen2.5-coder:7b",
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        max_tokens=2048,
        temperature=0.3,
        instructions=_INSTRUCTIONS,
    )

    signals = EventBus()
    signals.on(Event.ToolCallStarted,   lambda e: on_thinking_start(e) if e.name == "think" else None)
    signals.on(AppEvent.ThinkingChunk,  on_thinking_chunk)
    signals.on(AppEvent.ThinkingDone,   on_thinking_done)
    signals.on(Event.TextDelta,         on_text_delta)
    signals.on(Event.TurnComplete,      on_turn_complete)

    # no tools passed → inject_instructions writes DSL rules with no tool list
    agent = Agent(
        model=model,
        tools=[],
        signals=signals,
        signal_mode=SignalMode.AUTO,
        max_turns=10,
    )

    # add think tool for execution only — model never sees it as a callable tool
    agent._registry.register(_THINK_TOOL)
    agent.register_block("thinking", thinking_handler)

    print(f"\n{BOLD}Q: {question}{RESET}\n")
    async for _ in agent.astream(question):
        pass


if __name__ == "__main__":
    asyncio.run(main())
