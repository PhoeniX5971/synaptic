"""JSON-only MCP result envelope; text and legacy flattening are opt-in."""

import math
from types import SimpleNamespace
from typing import Any, TypedDict


class NormalizedResult(TypedDict):
    version: int
    content: list[dict[str, Any]]
    structured_content: Any
    has_structured_content: bool
    is_error: bool
    metadata: dict[str, Any]
    extensions: dict[str, Any]
    provenance: dict[str, str]


def json_value(value: Any) -> Any:
    """Copy JSON data and supported SDK models; reject opaque implementation objects."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("MCP data must contain finite JSON numbers")
        return value
    if isinstance(value, SimpleNamespace):
        value = vars(value)
    elif not isinstance(value, (dict, list)):
        from pydantic import BaseModel

        if isinstance(value, BaseModel):
            value = value.model_dump(mode="json", by_alias=True, exclude_unset=True)
        else:
            raise TypeError(f"Unsupported MCP data type: {type(value).__name__}")
    if isinstance(value, list):
        return [json_value(item) for item in value]
    if not all(isinstance(key, str) for key in value):
        raise TypeError("MCP JSON object keys must be strings")
    return {key: json_value(item) for key, item in value.items()}


def normalize_result(result: Any, *, provenance: dict[str, str] | None = None) -> NormalizedResult:
    """Preserve content, structured data, tool failure and protocol metadata together.

    Tool-reported errors remain results. Protocol and transport failures are raised
    separately by MCPSession; they are never converted to tool-reported failures.
    Unknown JSON extension fields and content modalities are preserved.
    """
    data = json_value(result)
    if not isinstance(data, dict):
        raise TypeError("MCP result must be an object")
    blocks = data.get("content", [])
    if not isinstance(blocks, list) or any(not isinstance(block, dict) for block in blocks):
        raise TypeError("MCP content must be a list of objects")
    structured_key = "structuredContent" if "structuredContent" in data else "structured_content"
    error = data.get("isError", data.get("is_error", False))
    if not isinstance(error, bool):
        raise TypeError("MCP error indicator must be boolean")
    metadata = data.get("_meta", data.get("meta", {})) or {}
    if not isinstance(metadata, dict):
        raise TypeError("MCP metadata must be an object")
    known = {
        "content",
        "structuredContent",
        "structured_content",
        "isError",
        "is_error",
        "_meta",
        "meta",
    }
    return NormalizedResult(
        version=1,
        content=blocks,
        structured_content=data.get(structured_key),
        has_structured_content=structured_key in data,
        is_error=error,
        metadata=metadata,
        extensions={key: value for key, value in data.items() if key not in known},
        provenance=json_value(provenance or {}),
    )


def extract_text(result: NormalizedResult) -> str:
    """Join text blocks only; media/resource blocks remain in the envelope."""
    return "\n".join(
        block["text"]
        for block in result["content"]
        if block.get("type") == "text" and isinstance(block.get("text"), str)
    )


def legacy_result(result: NormalizedResult) -> Any:
    """Opt into old lossy text/structured/error convenience behavior.

    Media-only output falls back to the JSON envelope, never an SDK object.
    """
    if result["is_error"]:
        return {"error": extract_text(result) or "MCP tool failed"}
    if result["structured_content"] is not None:
        return result["structured_content"]
    return extract_text(result) or result
