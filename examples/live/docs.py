from synaptic.core import Model, Provider, Tool
import os

api_key = os.getenv("GEMINI_API_KEY", "")


def add(a: int, b: int) -> int:
    return a + b


add_declaration = {
    "name": "add",
    "description": "Add two numbers together.",
    "parameters": {
        "type": "object",
        "properties": {
            "a": {"type": "integer", "description": "First number to add"},
            "b": {"type": "integer", "description": "Second number to add"},
        },
        "required": ["a", "b"],
    },
}

add_tool = Tool(
    name="add",
    declaration=add_declaration,
    function=add,
    default_params={"a": 0, "b": 0},
)

model = Model(
    provider=Provider.GEMINI,
    model="gemini-2.5-flash",
    api_key=api_key,
    automem=True,
    autorun=True,
    # tools=[add_tool],  # you can also use model.bind_tools([add_tool])
)
model.bind_tools([add_tool])

prompt = "Use the add function call to sum 12 and 30."

response = model.invoke(prompt)

print(response.message)
print("\n\n")
print(response.tool_calls)
print("\n\n")
print(response.tool_results)
print("\n\n")
print(model.history.MemoryList)
