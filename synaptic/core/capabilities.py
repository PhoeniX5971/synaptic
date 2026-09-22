"""Implemented adapter capabilities; individual models may impose narrower limits."""

from dataclasses import dataclass


class UnsupportedCapability(NotImplementedError):
    pass


@dataclass(frozen=True)
class Capabilities:
    sync: bool = True
    async_invoke: bool = True
    streaming: bool = True
    tools: bool = True
    structured: bool = True
    images: bool = False
    audio: bool = False
    native_call_ids: bool = True
    async_transport: str = "native"

    def check(self, mode, *, images=None, audio=None, structured=False):
        field = {"invoke": "sync", "ainvoke": "async_invoke", "astream": "streaming"}[mode]
        for key, requested in [
            (field, True),
            ("images", images),
            ("audio", audio),
            ("structured", structured),
        ]:
            if requested and not getattr(self, key):
                raise UnsupportedCapability(f"This adapter does not implement {key}")
