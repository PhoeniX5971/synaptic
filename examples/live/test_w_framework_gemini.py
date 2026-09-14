from synaptic import Model, Provider, Tool, autotool
from dotenv import load_dotenv
import os

# load .env file
load_dotenv()

# fetch key from env
api_key = os.getenv("GEMINI_API_KEY", "")


@autotool(
    description="A tool that adds two numbers",
    param_descriptions={"a": "First integer", "b": "Second integer"},
    autobind=False,
)
def add_tool(a: int, b: int) -> int:
    return a + b


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


Tool(
    name="greet",
    declaration=greet_declaration,
    function=greet,
    default_params={"name": "User"},
)


model = Model(
    provider=Provider.GEMINI,
    model="gemini-2.5-flash",
    temperature=1,
    api_key=api_key,
    automem=True,
    autorun=True,
)

model.bind_tools([add_tool])

prompt = (
    # "You are a weeb, and a japanese pop fan."
    # "Tell me what do you think about Ado, the japanese idol."
    "Call the add tool with a=5 and b=0, "
    # "and call the greet tool with name='phoenix'."
)

memory = model.invoke(prompt, role="user")

memory = model.invoke(prompt, role="user")
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

# print("LIST TOOL_CALLS:")
# print(memory.list_tool_calls())
#
# print("\n\n\n")
#
# print("GET TOOL_CALL add:")
# print(memory.get_tool_call("add_tool"))
#
# print("\n\n\n")
#
# print("LIST ARGS of TOOL_CALL add:")
# print(memory.get_tool_call("add_tool").list_args())
#
# print("\n\n\n")
#
# print("GET ARG 'a' of TOOL_CALL add:")
# print(memory.get_tool_call("add_tool").get_arg("a"))

#
# prompt = (
#     "Do you remember what was our last conversation about?"
#     "Also, list which tools you used and what their outputs were."
#     "If not then tell me that you don't remember."
# )
#
# memory = model.invoke(prompt)
# print(
#     memory.message
#     + "\n\n\n"
#     + str(memory.created)
#     + "\n\n\n"
#     + str(memory.tool_calls)
#     + "\n\n\n"
#     + str(memory.tool_results)
#     + "\n\n\n"
#     + str(model.history.MemoryList)
# )
