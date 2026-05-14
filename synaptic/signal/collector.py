import enum
from typing import AsyncIterator, List, Optional, Set

from ..agent.events import EventBus
from ..core.base.base_model import ResponseChunk, ToolCallArgsDelta
from ..core.provider import Provider
from ..core.tool import ToolCall
from .events import TextDelta, ToolCallDone, ToolCallResult, ToolCallStarted, TurnComplete
from .dsl import DSLParser


class SignalMode(enum.Enum):
    NONE = "none"
    NATIVE = "native"    # arg deltas from providers that support it natively
    TEXT = "text"        # text DSL for all providers
    AUTO = "auto"        # native if supported, text DSL fallback


# NOTE: The following providers do not natively support streaming tool-call
# argument deltas. Agents using TEXT or AUTO mode on these providers fall back
# to a text-based DSL protocol (see dsl.py): the model emits tool calls as
# <tool name="...">...</tool> blocks in its text stream. Agent injects the
# required DSL instructions into the system prompt automatically.
_NON_NATIVE: Set[Provider] = {
    Provider.GEMINI,
    Provider.XAI,
    Provider.TOGETHER,
    Provider.VERTEX,
}
NATIVE_SIGNAL_PROVIDERS: Set[Provider] = {
    Provider.CLAUDE,
    Provider.OPENAI,
    Provider.DEEPSEEK,
    Provider.UNIVERSAL_OPENAI,
}


def needs_text_mode(provider: Provider, mode: SignalMode) -> bool:
    if mode == SignalMode.TEXT:
        return True
    if mode == SignalMode.AUTO:
        return provider in _NON_NATIVE
    return False


async def collect(
    source: AsyncIterator[ResponseChunk],
    bus: EventBus,
    text_mode: bool = False,
) -> AsyncIterator[ResponseChunk]:
    """Pass-through stream that fires SignalEvents on `bus` as side effects.

    Yields the original ResponseChunks unchanged (native mode) or yields
    derived chunks with DSL tool calls synthesised as function_call chunks
    (text mode). In both cases the Agent loop sees standard ResponseChunks.
    """
    announced: Set[str] = set()
    accumulated = ""
    tool_calls: List[ToolCall] = []
    dsl = DSLParser() if text_mode else None

    async def _fire(event_type: str, payload) -> None:
        await bus.aemit(event_type, payload)

    async for chunk in source:
        if text_mode and dsl is not None:
            if chunk.text and not chunk.is_final:
                for evt in dsl.push(chunk.text):
                    await _fire(type(evt).__name__, evt)
                    if isinstance(evt, ToolCallStarted):
                        announced.add(evt.name)
                    elif isinstance(evt, ToolCallDone):
                        tool_calls.append(evt.call)
                        yield ResponseChunk(text="", function_call=evt.call)
                    elif isinstance(evt, TextDelta):
                        accumulated += evt.text
                        yield ResponseChunk(text=evt.text)
                continue  # don't yield original chunk in text mode

            if chunk.is_final:
                for evt in dsl.flush():
                    await _fire(type(evt).__name__, evt)
                    if isinstance(evt, TextDelta):
                        accumulated += evt.text
                await _fire("TurnComplete", TurnComplete(
                    message=accumulated, tool_calls=tool_calls,
                    input_tokens=chunk.input_tokens, output_tokens=chunk.output_tokens,
                ))
                yield ResponseChunk(text=accumulated, is_final=True,
                                    input_tokens=chunk.input_tokens,
                                    output_tokens=chunk.output_tokens)
                return

        else:
            if chunk.tool_call_delta:
                td: ToolCallArgsDelta = chunk.tool_call_delta
                if td.name not in announced:
                    announced.add(td.name)
                    await _fire("ToolCallStarted", ToolCallStarted(name=td.name))
                await _fire("ToolCallArgsDelta", td)

            elif chunk.function_call:
                call = chunk.function_call
                if call.name not in announced:
                    await _fire("ToolCallStarted", ToolCallStarted(name=call.name))
                await _fire("ToolCallDone", ToolCallDone(call=call))
                tool_calls.append(call)

            elif chunk.text and not chunk.is_final:
                accumulated += chunk.text
                await _fire("TextDelta", TextDelta(text=chunk.text))

            if chunk.is_final:
                await _fire("TurnComplete", TurnComplete(
                    message=accumulated, tool_calls=tool_calls,
                    input_tokens=chunk.input_tokens, output_tokens=chunk.output_tokens,
                ))

        yield chunk
