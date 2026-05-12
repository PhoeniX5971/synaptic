import base64
import json
import mimetypes
from io import BytesIO
from pathlib import Path
from typing import Any, List

from xai_sdk.chat import assistant, text, tool_result, user
from xai_sdk.proto import chat_pb2

from ...core.base import ResponseMem
from ...core.tool import ToolCall


def image_to_data_url(img: Any) -> str:
    try:
        from PIL import Image as PILImage  # type: ignore

        if isinstance(img, PILImage.Image):
            buf = BytesIO()
            fmt = img.format or "PNG"
            img.save(buf, format=fmt)
            data = base64.b64encode(buf.getvalue()).decode()
            return f"data:image/{fmt.lower()};base64,{data}"
    except ImportError:
        pass

    img_str = str(img)
    path = Path(img_str)
    if path.exists():
        mime, _ = mimetypes.guess_type(img_str)
        data = base64.b64encode(path.read_bytes()).decode()
        return f"data:{mime or 'application/octet-stream'};base64,{data}"
    return img_str


def history_messages(history) -> List[chat_pb2.Message]:
    messages: List[chat_pb2.Message] = []
    if history is None:
        return messages
    for idx, memory in enumerate(history.effective_mems()):
        if isinstance(memory, ResponseMem) and memory.tool_calls:
            calls = [
                chat_pb2.ToolCall(
                    id=f"call_{idx}_{i}",
                    function=chat_pb2.FunctionCall(
                        name=call.name,
                        arguments=json.dumps(call.args) if call.args else "{}",
                    ),
                )
                for i, call in enumerate(memory.tool_calls)
            ]
            messages.append(chat_pb2.Message(
                role=chat_pb2.MessageRole.ROLE_ASSISTANT,
                content=[text(memory.message or "")],
                tool_calls=calls,
            ))
            for i, (_, result) in enumerate(zip(memory.tool_calls, memory.tool_results or [])):
                value = json.dumps(result.get("result", result.get("error", result))) if isinstance(result, dict) else str(result)
                messages.append(tool_result(value, tool_call_id=f"call_{idx}_{i}"))
        elif memory.role == "assistant":
            messages.append(assistant(text(memory.message or "")))
        else:
            messages.append(user(text(memory.message or "")))
    return messages


def parse_calls(response: Any) -> List[ToolCall]:
    calls: List[ToolCall] = []
    for tc in response.tool_calls:
        try:
            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
        except json.JSONDecodeError:
            args = {}
        calls.append(ToolCall(name=tc.function.name, args=args))
    return calls
