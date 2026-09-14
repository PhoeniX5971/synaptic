from google.genai import types
import google.genai as genai
from dotenv import load_dotenv

# Load API key
load_dotenv()


# Simple tool definition
tool_declaration = {
    "name": "get_time",
    "description": "Get the current time in UTC",
    "parameters": {
        "type": "object",
        "properties": {
            "timezone": {"type": "string", "description": "The timezone to return"},
        },
        "required": ["timezone"],
    },
}


def get_time():
    return "BREH IT WORKS"


contents = [
    types.Content(
        role="user",
        parts=[
            types.Part(
                text="tell me how was your day. I am talking to you MF, answer meeeeee!!! What's the time in UTC, use the tool."
            )
        ],
    )
]


client = genai.Client()
tools = types.Tool(function_declarations=[tool_declaration])  # type: ignore
config = types.GenerateContentConfig(tools=[tools])

# Generate response with tools enabled
response = client.models.generate_content(
    model="gemini-2.5-flash", contents=contents, config=config
)

for part in response.candidates[0].content.parts:  # type: ignore
    if part.text:
        print("Message:", part.text)
    if part.function_call:
        print("Tool:", part.function_call.name, "Args:", part.function_call.args)
    print(response.text)
    print(response.candidates[0].content.parts)  # type: ignore
