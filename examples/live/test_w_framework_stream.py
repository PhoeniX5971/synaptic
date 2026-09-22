import os
import asyncio
from synaptic import Model, Provider, Tool, autotool
from dotenv import load_dotenv

# Load .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY", "")


@autotool(
    description="A tool that adds two numbers",
    param_descriptions={"a": "First integer", "b": "Second integer"},
    autobind=False,
)
def add_tool(a: int, b: int) -> int:
    return a + b


# Simple greet tool
def greet(name: str) -> str:
    return f"Hello, {name}!"


greet_declaration = {
    "name": "greet",
    "description": "Greet a person by name.",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Name of the person to greet"},
        },
        "required": ["name"],
    },
}

Tool(
    name="greet",
    declaration=greet_declaration,
    function=greet,
    default_params={"name": "User"},
)


async def test_streamed_response():
    model = Model(
        provider=Provider.GEMINI,
        model="gemini-2.5-flash",
        temperature=1,
        api_key=api_key,
        automem=True,
        autorun=True,
    )

    model.bind_tools([add_tool])

    prompt = "In roughly 2 sentences define how cool it is to have an online companion AI."

    # Collect streamed chunks
    accumulated_message = ""
    tool_calls = []

    async for chunk in model.astream(prompt, role="user"):
        if chunk.text:
            print(f"Chunk text: {chunk.text}")  # debug print
            accumulated_message += chunk.text
        if chunk.function_call:
            print(f"Tool call chunk: {chunk.function_call}")
            tool_calls.append(chunk.function_call)

    print("\n=== Final Results ===")
    print("Accumulated message:")
    print(accumulated_message)
    print("\nTool calls:")
    print(tool_calls)
    print("\nHistory:")
    print(model.history.MemoryList)

if __name__ == "__main__":
    asyncio.run(test_streamed_response())
