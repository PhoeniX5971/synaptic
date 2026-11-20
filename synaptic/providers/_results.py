"""One reading of Synaptic tool receipts; each adapter wraps the value natively."""

from typing import Any

from ..core._json import dumps


def tool_value(receipt: Any) -> tuple[Any, bool]:
    """Return a receipt's model-facing value and whether it is an error.

    Correlation belongs to the provider envelope's call ID, so the receipt's
    name and call_id are not repeated inside the content.
    """
    if not isinstance(receipt, dict):
        return receipt, False
    if "result" in receipt:
        return receipt["result"], False
    if "error" in receipt:
        return receipt["error"], True
    return receipt, False


def result_text(value: Any) -> str:
    """Strings pass through; structured values become JSON rather than a Python repr."""
    if isinstance(value, str):
        return value
    try:
        if hasattr(value, "model_dump"):
            return dumps(value.model_dump())
        if hasattr(value, "__dict__"):
            return dumps(value.__dict__)
        return dumps(value)
    except Exception:
        return str(value)


def tool_text(receipt: Any) -> str:
    """Content for text-only tool messages, which have no native error flag."""
    value, failed = tool_value(receipt)
    return dumps({"error": value}) if failed else result_text(value)


def tool_response(receipt: Any) -> dict:
    """Function-response mapping for SDKs that take a dict (Gemini, Vertex)."""
    value, failed = tool_value(receipt)
    return {"error": value} if failed else {"result": value}
