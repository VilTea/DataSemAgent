"""Core abstractions for the pipeline consumption system.

These are interface-only — consumers and pipelines must implement these.
No message format restrictions are imposed. Each consumer interprets
events in its own way.
"""
from abc import ABC, abstractmethod
from typing import Any


class EventConsumer(ABC):
    """Receives events from a pipeline and interprets them.

    Lifecycle: ``start()`` is called before a run, ``stop()`` after it.
    Both are idempotent — they may be called multiple times across runs.

    Implementations must handle unknown event types gracefully
    (skip / ignore rather than crash), since the pipeline imposes
    no format restrictions on dispatched events.
    """

    @abstractmethod
    async def start(self, ctx: Any | None = None) -> None:
        """Called before the pipeline begins dispatching events.

        *ctx* is an optional AgentContext — implementations may use it
        to access the hook registry for registering consumer hooks.
        """
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Called after the pipeline stops dispatching events."""
        ...

    @abstractmethod
    async def consume(self, event: Any) -> None:
        """Process a single event.

        Args:
            event: An arbitrary object (AgentCompletion, Message, etc.).
                   The consumer is responsible for type-checking.
        """
        ...


class Consumable(ABC):
    """Manages a set of EventConsumers and dispatches events to them.

    Contract:
        - ``emit()`` MUST isolate consumer errors. A failed consumer
          must not propagate to the caller or affect other consumers.
        - ``start()`` and ``stop()`` MUST call all registered consumers,
          even if some fail during the process.

    Usage as context manager::

        pipeline = QueuePipeline()
        pipeline.register(foo)
        async with pipeline.bind(ctx):
            ...  # start/stop + hook lifecycle handled automatically
    """

    def __init__(self) -> None:
        self._bound_ctx: Any = None

    def bind(self, ctx: Any) -> "Consumable":
        """Bind to *ctx* and return self — ready for ``async with``."""
        self._bound_ctx = ctx
        return self

    async def __aenter__(self) -> "Consumable":
        await self.start(self._bound_ctx)
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.stop()
        self._bound_ctx = None

    # ------------------------------------------------------------------ #
    #  Abstract — subclasses MUST implement
    # ------------------------------------------------------------------ #

    @abstractmethod
    async def start(self, ctx: Any | None = None) -> None:
        """Start the pipeline and notify all registered consumers.

        *ctx* is an optional AgentContext — implementations may use it
        to access the hook registry for registering consumer hooks.
        """
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Stop the pipeline and notify all registered consumers."""
        ...

    @abstractmethod
    async def emit(self, event: Any) -> None:
        """Dispatch an event to all registered consumers.

        Consumer errors are isolated — they are logged but never raised.
        """
        ...

    @abstractmethod
    def register(self, consumer: EventConsumer) -> None:
        """Register a consumer to receive events."""
        ...

    @abstractmethod
    def unregister(self, consumer: EventConsumer) -> None:
        """Remove a previously registered consumer.

        Raises:
            ValueError: If the consumer is not registered.
        """
        ...
