import inspect
from typing import Callable, Optional

from ..core.tool import ToolCall

Permission = Callable[[str, dict], bool]


def allow(permission: Optional[Permission], call: ToolCall) -> bool:
    if permission is None:
        return True
    result = permission(call.name, call.args)
    if inspect.isawaitable(result):
        raise RuntimeError("Async permission requires arun()")
    return bool(result)


async def aallow(permission: Optional[Permission], call: ToolCall) -> bool:
    if permission is None:
        return True
    result = permission(call.name, call.args)
    if inspect.isawaitable(result):
        result = await result
    return bool(result)
