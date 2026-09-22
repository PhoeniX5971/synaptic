from synaptic.core import Model, Provider, Tool
from synaptic.core.base import ResponseFormat
from dotenv import load_dotenv
import os
from pydantic import BaseModel
from typing import List

# load .env file
load_dotenv()

# fetch key from env
api_key = os.getenv("GEMINI_API_KEY", "")


class Response(BaseModel):
    message: str
    explanation: List[str]


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


model = Model(
    provider=Provider.GEMINI,
    model="gemini-2.5-flash",
    temperature=1,
    tools=[add_tool],
    api_key=api_key,
    automem=True,
    autorun=True,
    response_format=ResponseFormat.JSON,
    response_schema=Response,
)

prompt = (
    "You are a weeb, and a japanese pop fan."
    "Tell me what do you think about Ado, the japanese idol."
    # "Call the add tool with a=5 and b=0, "
    # "and call the greet tool with name='phoenix'."
)

memory = model.invoke(prompt, role="user")

# print(
#     """
#     === Model Tools ===\n
#     """
#     f"{model.tools}"
# )

print(
    memory.message
    # + "\n\n\n"
    # + str(memory.created)
    # + "\n\n\n"
    # + str(memory.tool_calls)
    # + "\n\n\n"
    # + str(memory.tool_results)
    # + "\n\n\n"
    # + str(model.history.MemoryList)
)

# print("LIST TOOL_CALLS:")
# print(memory.list_tool_calls())

print("\n\n\n")

# print("GET TOOL_CALL add:")
# print(memory.get_tool_call("add"))

print("\n\n\n")

# print("LIST ARGS of TOOL_CALL add:")
# print(memory.get_tool_call("add").list_args())

print("\n\n\n")

# print("GET ARG 'a' of TOOL_CALL add:")
# print(memory.get_tool_call("add").get_arg("a"))
