"""Default Consumable implementation backed by asyncio.Queue.

QueuePipeline uses an internal queue to decouple producers (nodes calling
emit()) from consumers (dispatch loop). Consumers are dispatched concurrently
via asyncio.gather so a slow consumer does not block others.
"""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from app.pipeline.abc import Consumable, EventConsumer

if TYPE_CHECKING:
    from app.node.base import AgentContext
    from app.hook.registry import HookRegistry

_SENTINEL = object()


class QueuePipeline(Consumable):
    """Queue-driven pipeline with concurrent consumer dispatch.

    Args:
        max_queue_size: Maximum queue depth before ``emit()`` blocks
                        (backpressure). 0 = unlimited.

    Usage::

        pipeline = QueuePipeline()
        pipeline.register(cli_consumer)
        pipeline.register(sse_consumer)
        await pipeline.start()
        await pipeline.emit(some_event)
        await pipeline.stop()
    """

    def __init__(self, max_queue_size: int = 0):
        super().__init__()
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self._consumers: list[EventConsumer] = []
        self._task: asyncio.Task | None = None
        self._injected_hooks: HookRegistry | None = None

    # ------------------------------------------------------------------ #
    #  Lifecycle
    # ------------------------------------------------------------------ #

    async def start(self, ctx: AgentContext | None = None) -> None:
        from app.hook.registry import OBSERVE_HOOK_POINTS

        if ctx is not None:
            self._injected_hooks = ctx.hooks
            for c in self._consumers:
                self._injected_hooks.register(c, whitelist=OBSERVE_HOOK_POINTS)

        for consumer in self._consumers:
            try:
                await consumer.start()
            except Exception:
                pass  # isolated — one failure does not block others
        self._task = asyncio.create_task(self._dispatch_loop())

    async def stop(self) -> None:
        if self._task:
            # Send sentinel so the dispatch loop drains the queue and
            # exits cleanly — never cancel while a consumer is mid-write.
            await self._queue.put(_SENTINEL)
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        for consumer in self._consumers:
            try:
                await consumer.stop()
            except Exception:
                pass

        if self._injected_hooks is not None:
            for c in self._consumers:
                self._injected_hooks.unregister(c)
            self._injected_hooks = None

    # ------------------------------------------------------------------ #
    #  Dispatch
    # ------------------------------------------------------------------ #

    async def emit(self, event: Any) -> None:
        await self._queue.put(event)

    async def _dispatch_loop(self) -> None:
        while True:
            event = await self._queue.get()
            if event is _SENTINEL:
                return
            if not self._consumers:
                continue
            results = await asyncio.gather(
                *(self._safe_consume(c, event) for c in self._consumers),
                return_exceptions=True,
            )
            for consumer, result in zip(self._consumers, results):
                if isinstance(result, Exception):
                    pass  # consumer error isolated

    @staticmethod
    async def _safe_consume(consumer: EventConsumer, event: Any) -> None:
        try:
            await consumer.consume(event)
        except Exception:
            raise  # caught by gather(return_exceptions=True)

    # ------------------------------------------------------------------ #
    #  Registration
    # ------------------------------------------------------------------ #

    def register(self, consumer: EventConsumer) -> None:
        self._consumers.append(consumer)

    def unregister(self, consumer: EventConsumer) -> None:
        self._consumers.remove(consumer)
