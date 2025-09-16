from synaptic.core import Model, Provider, Tool, History


# Simple tool 1: add two numbers
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


# Simple tool 2: greet a person
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

history = History()


greet_tool = Tool(
    name="greet",
    declaration=greet_declaration,
    function=greet,
    default_params={"name": "User"},
)


model = Model(
    provider=Provider.GEMINI,
    model="gemini-2.5-pro",
    temperature=1,
    tools=[add_tool, greet_tool],
    automem=True,
    autorun=True,
)

prompt = (
    "You are a weeb, and a japanese pop fan."
    "Tell me what do you think about Ado, the japanese idol."
    "Call the add tool with a=5 and b=0, "
    "and call the greet tool with name='phoenix'."
)

memory = model.invoke(prompt)
print(
    memory.message
    + "\n\n\n"
    + str(memory.created)
    + "\n\n\n"
    + str(memory.tool_calls)
    + "\n\n\n"
    + str(memory.tool_results)
    + "\n\n\n"
    + str(model.history.MemoryList)
)

print("\n")

prompt = (
    "Do you remember what was our last conversation about?"
    "Also, list which tools you used and what their outputs were."
    "Keep your response concise."
)

memory = model.invoke(prompt)
print(
    memory.message
    + "\n\n\n"
    + str(memory.created)
    + "\n\n\n"
    + str(memory.tool_calls)
    + "\n\n\n"
    + str(memory.tool_results)
    + "\n\n\n"
    + str(model.history.MemoryList)
)
