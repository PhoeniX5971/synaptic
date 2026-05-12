from typing import Any


def normalize_result(result: Any) -> Any:
    if getattr(result, "isError", False) or getattr(result, "is_error", False):
        return {"error": _content_text(result) or "MCP tool failed"}

    structured = getattr(result, "structuredContent", None)
    structured = structured if structured is not None else getattr(result, "structured_content", None)
    if structured is not None:
        return structured

    text = _content_text(result)
    return text if text else result


def _content_text(result: Any) -> str:
    parts = []
    for item in getattr(result, "content", []) or []:
        text = getattr(item, "text", None)
        if text is not None:
            parts.append(text)
    return "\n".join(parts)
