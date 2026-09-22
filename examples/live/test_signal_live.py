"""Live terminal demo: signal streaming for a file-write task."""
import asyncio
import os
import tempfile

from synaptic import Agent, Model, Provider, SignalMode, Tool
from synaptic.agent.events import EventBus


def write_file(path: str, content: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    return f"wrote {len(content)} bytes to {path}"


WRITE_TOOL = Tool(
    name="write_file",
    declaration={
        "name": "write_file",
        "description": "Write text content to a file at the given path.",
        "parameters": {
            "type": "object",
            "properties": {
                "path":    {"type": "string", "description": "file path to write"},
                "content": {"type": "string", "description": "text content to write"},
            },
            "required": ["path", "content"],
        },
    },
    function=write_file,
    add_to_registry=False,
)


async def main():
    model = Model(
        provider=Provider.UNIVERSAL_OPENAI,
        model="qwen2.5-coder:7b",
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        max_tokens=1024,
        temperature=0.0,
    )

    signals = EventBus()

    signals.on("BlockStarted",      lambda e: print(f"\n\033[90m[{e.type}{'  '+str(e.attrs) if e.attrs else ''}]\033[0m ", end="", flush=True))
    signals.on("TextDelta",         lambda e: print(e.text, end="", flush=True))
    signals.on("ToolCallStarted",   lambda e: print(f"\033[33m{e.name}\033[0m(", end="", flush=True))
    signals.on("ToolCallArgsDelta", lambda e: print(f"\033[36m{e.delta}\033[0m", end="", flush=True))
    signals.on("ToolCallDone",      lambda e: print(")", flush=True))
    signals.on("ToolCallResult",    lambda e: print(f"\n\033[32m→ {e.result}\033[0m", flush=True))
    signals.on("TurnComplete",      lambda e: print(f"\n\033[90m[{e.input_tokens}in {e.output_tokens}out]\033[0m\n", flush=True))

    agent = Agent(
        model=model,
        tools=[WRITE_TOOL],
        signals=signals,
        signal_mode=SignalMode.AUTO,
        max_turns=5,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "hello.py")
        print(f"\033[1mPrompt:\033[0m write a python hello world script to {path}\n")

        async for _ in agent.astream(
            f"Write a Python hello world script to {path}. "
            "Include a main() function and a few comments."
        ):
            pass

        if os.path.exists(path):
            print(f"\033[1mFile contents:\033[0m")
            print(open(path).read())
        else:
            print("\033[31mFile was not written.\033[0m")


if __name__ == "__main__":
    asyncio.run(main())
