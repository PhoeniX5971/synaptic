import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------------
# Tools
# -------------------------
tools = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Adds two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "greet",
            "description": "Greets the user by name",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
]


# -------------------------
# Actual Python function implementations
# -------------------------
def add(a, b):
    return a + b


def greet(name):
    return f"Hello, {name}!"


# -------------------------
# Run
# -------------------------
def main():
    user_input = input("You: ")

    # first call
    first = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": user_input}],
        tools=tools,
        tool_choice="auto",
    )

    msg = first.choices[0].message

    if not msg.tool_calls:
        print("Assistant:", msg.content)
        return

    final_messages = [
        {"role": "user", "content": user_input},
        msg,  # the assistant tool-call message
    ]

    print(msg.tool_calls)

    # handle all tool calls
    for tc in msg.tool_calls:
        fn_name = tc.function.name
        fn_args = json.loads(tc.function.arguments)

        if fn_name == "add":
            result = add(**fn_args)
        elif fn_name == "greet":
            result = greet(**fn_args)
        else:
            result = "Unknown tool"

        # append tool result message
        final_messages.append(
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps({"result": result}),
            }
        )

    # second call: send tool results back to model
    second = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=final_messages,
    )

    print("Assistant:", second.choices[0].message.content)


if __name__ == "__main__":
    main()
