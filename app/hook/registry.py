"""Hook registry — lifecycle interception for agent flow."""
from __future__ import annotations

import asyncio
import inspect
from typing import Callable

from app.logger import logger


class HookPoint:
    """Lifecycle hook points with documented callback parameters.

    Usage: ``@hook(HookPoint.NODE_PREP_BEFORE)`` — the constant name
    documents what params the callback receives.
    """
    NODE_INIT_BEFORE = "node.init.before"             # callback(ctx, node) — fires once, before first prep
    NODE_INIT_AFTER = "node.init.after"               # callback(ctx, node) — fires once, after first prep
    NODE_PREP_BEFORE = "node.prep_async.before"       # callback(ctx, node) — fires every prep
    NODE_PREP_AFTER = "node.prep_async.after"         # callback(ctx, node) — fires every prep
    NODE_EXEC_BEFORE = "node.exec_async.before"       # callback(ctx, node)
    NODE_EXEC_AFTER = "node.exec_async.after"         # callback(ctx, node, reason)
    TOOL_BEFORE = "tool.before"                       # callback(ctx, tool_call, tool)
    TOOL_AFTER = "tool.after"                         # callback(ctx, tool_call, tool, result)
    FLOW_START = "flow.start"                         # callback(ctx)
    FLOW_END = "flow.end"                             # callback(ctx)


class HookRegistry:
    def __init__(self):
        self._hooks: list[tuple[str, int, int, Callable, dict, str]] = []

    def on(
        self,
        point: str,
        callback: Callable,
        *,
        priority: int = 100,
        node_name: str | None = None,
        node_type: type | None = None,
        tool_name: str | None = None,
        on_error: str = "log",
    ) -> None:
        if on_error not in ("log", "raise"):
            raise ValueError(f"on_error must be 'log' or 'raise', got '{on_error}'")
        cb_id = id(callback)
        # Idempotent: skip if this exact callback is already registered at this point
        if any(h[0] == point and h[2] == cb_id for h in self._hooks):
            return
        self._hooks.append((point, priority, cb_id, callback, {
            "node_name": node_name, "node_type": node_type, "tool_name": tool_name,
        }, on_error))
        self._hooks.sort(key=lambda h: h[1])  # sort by priority

    async def emit(self, point: str, /, **context: any) -> None:
        for hook_point, _, _, callback, filters, on_error in self._hooks:
            if hook_point != point:
                continue
            if not self._match(filters, **context):
                continue
            try:
                sig = inspect.signature(callback)
                filtered = {k: v for k, v in context.items() if k in sig.parameters}
                ret = callback(**filtered)
                if asyncio.iscoroutine(ret):
                    await ret
            except Exception as e:
                if on_error == "raise":
                    raise
                logger.error(f"Hook '{point}' failed: {e}")

    @staticmethod
    def _match(filters: dict, **context: any) -> bool:
        node = context.get("node")
        if filters.get("node_name") and (node is None or node.name != filters["node_name"]):
            return False
        if filters.get("node_type") and (node is None or not isinstance(node, filters["node_type"])):
            return False
        tool_name = filters.get("tool_name")
        if tool_name:
            tool_call = context.get("tool_call")
            if tool_call is not None:
                if tool_call.function.name != tool_name:
                    return False
            elif node is not None:
                # Node-level hooks: check if tool is loaded on this node
                loaded = {t.name for t in getattr(node, "tools", [])}
                if tool_name not in loaded:
                    return False
        return True
