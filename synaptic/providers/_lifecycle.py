"""Dispose listeners and attempt every SDK cleanup, retaining failures for retry."""

import inspect

from ..core._awaitables import require_sync


class AdapterLifecycle:
    def _dispose_subscription(self):
        subscription = getattr(self, "_subscription", None)
        if subscription:
            subscription.close()
        for name in ("_scheduler", "_preparation_scheduler"):
            scheduler = getattr(self, name, None)
            if scheduler:
                scheduler.close()
        if not hasattr(self, "_closed_clients"):
            self._closed_clients = set()

    def close(self):
        self._dispose_subscription()
        errors = []
        for name in ("client", "_sync_client"):
            client = getattr(self, name, None)
            close = getattr(client, "close", None)
            if (
                close
                and not inspect.iscoroutinefunction(close)
                and id(client) not in self._closed_clients
            ):
                try:
                    require_sync(close(), "Use aclose for asynchronous client cleanup")
                    self._closed_clients.add(id(client))
                except Exception as error:
                    errors.append(error)
        if errors:
            raise ExceptionGroup("Provider client cleanup failed", errors)

    async def aclose(self):
        self._dispose_subscription()
        errors = []
        clients = [getattr(self, name, None) for name in ("async_client", "_async_client")]
        clients.append(getattr(getattr(self, "client", None), "aio", None))
        for client in clients:
            close = getattr(client, "aclose", None) or getattr(client, "close", None)
            if close and id(client) not in self._closed_clients:
                try:
                    result = close()
                    if inspect.isawaitable(result):
                        await result
                    self._closed_clients.add(id(client))
                except Exception as error:
                    errors.append(error)
        try:
            self.close()
        except Exception as error:
            errors.append(error)
        if errors:
            raise ExceptionGroup("Provider client cleanup failed", errors)
