"""Together message conversion shares the provider-neutral Chat wire shape."""

from .._chat_history import history_messages as convert_history
from .._media import chat_content as content, image_url
from .._chat import parse_calls
from ...core.base import ResponseFormat


def messages(adapter, prompt, role, images=None, audio=None):
    instructions = adapter.instructions or ""
    if adapter.response_format == ResponseFormat.JSON:
        instructions += "\nRespond only with valid JSON."
    result = convert_history(adapter.history, instructions, adapter.role_map)
    if prompt is not None or images or audio:
        result.append({"role": role, "content": content(prompt, images, audio)})
    return result


def history_messages(adapter):
    return convert_history(adapter.history, "", adapter.role_map)
