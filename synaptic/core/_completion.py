"""Assemble provider chunks consistently across model and agent invocation."""

from typing import Any

from .base import ResponseMem


class Completion:
    def __init__(self) -> None:
        self.text = ""
        self.tool_calls: list[Any] = []
        self.input_tokens: int | None = None
        self.output_tokens: int | None = None
        self.finish_reason: str | None = None
        self.extensions: dict[str, Any] = {}

    def add(self, chunk) -> None:
        if not chunk.is_final:
            self.text += chunk.text or ""
        elif not self.text:
            # Final text is a snapshot, not a delta. Support final-only adapters.
            self.text = chunk.text or ""
        if chunk.function_call:
            self.tool_calls.append(chunk.function_call)
        for field in ("input_tokens", "output_tokens", "finish_reason"):
            value = getattr(chunk, field, None)
            if value is not None:
                setattr(self, field, value)
        self.extensions.update(getattr(chunk, "extensions", {}))

    def memory(self, created, tool_results=None) -> ResponseMem:
        return ResponseMem(
            message=self.text,
            created=created,
            tool_calls=self.tool_calls,
            tool_results=tool_results,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            finish_reason=self.finish_reason,
            extensions=dict(self.extensions),
        )
