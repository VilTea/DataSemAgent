"""Hook registry — lifecycle interception for agent flow."""
from __future__ import annotations

import asyncio
import inspect
from typing import Callable

from app.logger import logger


def _callback_belongs_to(callback: Callable, obj: object) -> bool:
    """True if *callback* is a method bound to *obj* (or the object itself)."""
    if callback is obj:
        return True
    return getattr(callback, '__self__', None) is obj


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

OBSERVE_HOOK_POINTS = frozenset({
    HookPoint.FLOW_START, HookPoint.FLOW_END,
    HookPoint.NODE_EXEC_BEFORE, HookPoint.NODE_EXEC_AFTER,
    HookPoint.TOOL_BEFORE, HookPoint.TOOL_AFTER,
})


class HookRegistry:
    _debug: bool = False

    @classmethod
    def set_debug(cls, enabled: bool) -> None:
        cls._debug = enabled

    def __init__(self):
        self._hooks: list[tuple[str, int, int, Callable, dict, str]] = []

    def register(
        self, obj: Callable | type | object,
        *,
        whitelist: frozenset | None = None,
    ) -> list[str]:
        """Register all `@hook`-decorated members from *obj*.

        *obj* may be:
        - A decorated function → registers it directly
        - A class → scans methods for `_hook_config` (unbound)
        - An instance → scans methods for `_hook_config` (bound)

        When *whitelist* is provided, only hooks whose point is in the
        set are registered, and all registered hooks have ``on_error``
        forced to ``"log"``.

        Returns a list of hook point strings that were registered.
        """
        registered: list[str] = []
        force_log = whitelist is not None

        cfg = getattr(obj, "_hook_config", None)
        if cfg is not None:
            # Single decorated function or bound method
            if whitelist is not None and cfg["point"] not in whitelist:
                logger.debug(f"Skipping hook {cfg['point']}: not in whitelist")
                return registered
            self._on_from_config(cfg, obj, force_log=force_log)
            registered.append(cfg["point"])
            return registered

        # Class or instance — scan methods
        target_cls = obj if inspect.isclass(obj) else type(obj)
        for attr_name in dir(target_cls):
            attr = getattr(target_cls, attr_name, None)
            cfg = getattr(attr, "_hook_config", None)
            if cfg is None:
                continue
            if whitelist is not None and cfg["point"] not in whitelist:
                logger.debug(f"Skipping hook {cfg['point']}: not in whitelist")
                continue
            if not inspect.isclass(obj):
                attr = getattr(obj, attr_name)  # bind to instance
            self._on_from_config(cfg, attr, force_log=force_log)
            registered.append(cfg["point"])

        return registered

    def unregister(self, obj: object) -> None:
        """Remove all hooks whose callback belongs to *obj*.

        Safe to call even if *obj* was never registered.
        """
        self._hooks = [
            h for h in self._hooks
            if not _callback_belongs_to(h[3], obj)
        ]

    def _on_from_config(self, cfg: dict, callback: Callable, *, force_log: bool = False) -> None:
        self.on(
            cfg["point"], callback,
            priority=cfg.get("priority", 100),
            node_name=cfg.get("node_name"),
            node_type=cfg.get("node_type"),
            tool_name=cfg.get("tool_name"),
            on_error="log" if force_log else cfg.get("on_error", "log"),
        )

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
        # Use stable id: for bound methods, use the underlying function's id
        func = getattr(callback, '__func__', callback)
        cb_key = id(func)
        if any(h[0] == point and h[2] == cb_key for h in self._hooks):
            return
        self._hooks.append((point, priority, id(func), callback, {
            "node_name": node_name, "node_type": node_type, "tool_name": tool_name,
        }, on_error))
        self._hooks.sort(key=lambda h: h[1])  # sort by priority

    async def emit(self, point: str, /, **context: any) -> None:
        matched = False
        for hook_point, _, _, callback, filters, on_error in self._hooks:
            if hook_point != point:
                continue
            if not self._match(filters, **context):
                continue
            matched = True
            if self._debug or HookRegistry._debug:
                cb_name = getattr(callback, '__name__', str(callback))
                if hasattr(callback, '__self__'):
                    cb_name = f"{type(callback.__self__).__name__}.{cb_name}"
                logger.info(f"hook '{point}' → {cb_name}")
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
