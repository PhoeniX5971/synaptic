import inspect
from ..core._awaitables import require_sync
from typing import Callable, Optional

from ..core.tool import ToolCall

Permission = Callable[[str, dict], bool]


def allow(permission: Optional[Permission], call: ToolCall) -> bool:
    if permission is None:
        return True
    result = permission(call.name, call.args)
    require_sync(result, "Async permission requires arun()")
    return bool(result)


async def aallow(permission: Optional[Permission], call: ToolCall) -> bool:
    if permission is None:
        return True
    result = permission(call.name, call.args)
    if inspect.isawaitable(result):
        result = await result
    return bool(result)
