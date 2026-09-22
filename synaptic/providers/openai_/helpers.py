"""Legacy helper import path for the shared Chat conversion contract."""

from .._chat_history import history_messages as history_messages, result_text as result_text
from .._chat import parse_calls as parse_tool_calls, arguments
from .._media import chat_content
from ...core.tool import ToolCall


def audio_blocks(audio):
    return chat_content("", audio=audio)[1:] if audio else []


def add_prompt(messages, prompt, role, audio):
    if prompt is not None or audio:
        messages.append({"role": role, "content": chat_content(prompt, audio=audio)})


def stream_tool_calls(pending):
    return [
        ToolCall(item["name"], arguments(item["args"]), id=item.get("id"))
        for _, item in sorted(pending.items())
    ]
