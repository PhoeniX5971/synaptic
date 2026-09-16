"""Portable finish categories with lossless native reasons in extensions."""


def finish_reason(value, extensions):
    if value is None:
        return None
    native = str(value)
    key = native.lower().rsplit(".", 1)[-1]
    aliases = {
        "stop": "stop",
        "reason_stop": "stop",
        "reason_max_len": "length",
        "reason_max_context": "length",
        "reason_tool_calls": "tool_calls",
        "end_turn": "stop",
        "stop_sequence": "stop",
        "1": "stop",
        "max_tokens": "length",
        "length": "length",
        "2": "length",
        "tool_use": "tool_calls",
        "tool_calls": "tool_calls",
        "safety": "content_filter",
        "recitation": "content_filter",
        "content_filter": "content_filter",
    }
    normalized = aliases.get(key, "unknown")
    if native != normalized:
        extensions.setdefault("native_finish_reason", native)
    return normalized
