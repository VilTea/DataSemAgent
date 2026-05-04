"""@hook decorator — auto-detects tool_name/node_name from self.name."""
from __future__ import annotations

from typing import Callable


def hook(
    point: str,
    *,
    priority: int = 100,
    on_error: str = "log",
    node_name: str | None = None,
    node_type: type | None = None,
    tool_name: str | None = None,
) -> Callable:
    # Usage:
    #   @hook(HookPoint.NODE_PREP_BEFORE, priority=50) runs before
    #   @hook(HookPoint.NODE_PREP_BEFORE, priority=200).
    #   Default is 100.
    def decorator(func: Callable) -> Callable:
        func._hook_config = {
            "point": point, "priority": priority, "on_error": on_error,
            "node_name": node_name, "node_type": node_type,
            "tool_name": tool_name,
        }
        return func
    return decorator
