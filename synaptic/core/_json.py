"""Provider-bound JSON text. Non-ASCII stays literal unless the consumer opts out."""

import json
from typing import Any

_ensure_ascii = False


def set_ensure_ascii(value: bool) -> None:
    """Choose whether JSON text sent to or echoed from providers escapes non-ASCII.

    The default is False: escaping turns each non-Latin character into a six-character
    ``\\uXXXX`` sequence that the model must read and write token by token.
    """
    global _ensure_ascii
    _ensure_ascii = bool(value)


def dumps(value: Any, **kwargs: Any) -> str:
    kwargs.setdefault("ensure_ascii", _ensure_ascii)
    return json.dumps(value, **kwargs)
