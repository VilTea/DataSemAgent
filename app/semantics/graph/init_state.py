"""Graph initialization state — readiness + executor references + extra data."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.db.graph import GraphExecutor


class GraphInitState:
    def __init__(self):
        self._ready: dict[str, bool] = {}
        self._executors: dict[str, GraphExecutor] = {}
        self._extras: dict[str, dict] = {}

    def mark_ready(self, name: str) -> None:
        self._ready[name] = True

    def is_ready(self, name: str) -> bool:
        return self._ready.get(name, False)

    def set_executor(self, name: str, ex: GraphExecutor) -> None:
        self._executors[name] = ex

    def get_executor(self, name: str) -> GraphExecutor | None:
        return self._executors.get(name)

    def set_extra(self, name: str, key: str, value) -> None:
        self._extras.setdefault(name, {})[key] = value

    def get_extra(self, name: str, key: str):
        return self._extras.get(name, {}).get(key)

    def reset(self) -> None:
        self._ready.clear()
        self._executors.clear()
        self._extras.clear()


init_state = GraphInitState()
