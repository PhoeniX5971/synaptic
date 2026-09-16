"""Scoped atomic catalogs; observer errors are diagnostics, never rollback."""

import threading
import warnings
from types import MappingProxyType

from ._awaitables import require_sync


class Subscription:
    def __init__(self, dispose):
        self._dispose = dispose

    def close(self):
        if self._dispose:
            dispose, self._dispose = self._dispose, None
            dispose()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class ToolRegistry:
    """One event-loop/application scope; snapshots cannot mutate registration state."""

    def __init__(self, *, on_observer_error=None):
        self._tools, self._owners = {}, {}
        self._callbacks = []
        self._lock = threading.RLock()
        self._notifying = False
        self.version = 0
        self.on_observer_error = on_observer_error

    def _publish(self, candidate, owners, publish=None):
        if self._notifying:
            raise RuntimeError("Registry mutation from observers is not supported")
        # This section contains no awaits. Readers use the same lock.
        self._tools.clear()
        self._tools.update(candidate)
        self._owners = owners
        self.version += 1
        if publish:
            publish()
        self._notifying = True
        try:
            for callback in tuple(self._callbacks):
                try:
                    require_sync(callback(), "Registry observers must be synchronous")
                except Exception as exc:
                    try:
                        if self.on_observer_error:
                            require_sync(
                                self.on_observer_error(exc), "Error handlers must be synchronous"
                            )
                        else:
                            warnings.warn(
                                "Registry observer failed: " + type(exc).__name__, RuntimeWarning
                            )
                    except Exception:
                        # Warnings configured as errors must not undo a published transaction.
                        pass
        finally:
            self._notifying = False

    def register_many(self, tools):
        with self._lock:
            candidate = dict(self._tools)
            for tool in tools:
                if tool.name in self._owners:
                    raise ValueError("Tool alias is owned by another catalog: " + tool.name)
                candidate[tool.name] = tool
            if tools:
                self._publish(candidate, dict(self._owners))

    def register(self, tool):
        self.register_many([tool])

    def replace_owned(self, owner, tools, *, publish=None):
        """Replace exactly one owner's registrations; validate collisions before publish."""
        with self._lock:
            candidate = {
                name: tool for name, tool in self._tools.items() if self._owners.get(name) != owner
            }
            owners = {name: value for name, value in self._owners.items() if value != owner}
            for tool in tools:
                if tool.name in candidate:
                    raise ValueError("Tool alias collision: " + tool.name)
                candidate[tool.name], owners[tool.name] = tool, owner
            self._publish(candidate, owners, publish)

    def unregister(self, name):
        with self._lock:
            if name in self._owners:
                raise ValueError("Use the owning catalog to remove this tool")
            if name in self._tools:
                self._publish(
                    {k: v for k, v in self._tools.items() if k != name}, dict(self._owners)
                )

    def unregister_prefix(self, prefix):
        with self._lock:
            names = [name for name in self._tools if name.startswith(prefix)]
            if any(name in self._owners for name in names):
                raise ValueError("Prefix deletion cannot remove owned tools")
            if names:
                self._publish(
                    {k: v for k, v in self._tools.items() if k not in names}, dict(self._owners)
                )

    def on_change(self, fn):
        with self._lock:
            self._callbacks.append(fn)

        def dispose():
            with self._lock:
                if fn in self._callbacks:
                    self._callbacks.remove(fn)

        return Subscription(dispose)

    def all(self):
        with self._lock:
            return MappingProxyType(dict(self._tools))

    def autotool(self, description="", param_descriptions=None, default_params=None, **kwargs):
        from .tool import autotool

        return autotool(
            description=description,
            param_descriptions=param_descriptions,
            default_params=default_params,
            registry=self,
            **kwargs,
        )
