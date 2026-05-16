from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING, TypeVar

from pocketflow import AsyncNode
from pydantic import BaseModel, Field, ConfigDict, PrivateAttr, model_validator

from app.hook import HookRegistry
from app.logger import logger
from app.schema import Memory, FinishReason
from app.tool.base import BaseTool, ToolContext

if TYPE_CHECKING:
    from app.pipeline.abc import Consumable

T = TypeVar('T')
_SINGLETON_KEY =  "__agent_context_singleton__"

class AgentContext(BaseModel):
    shared: dict[str, Any] = Field(default_factory=dict, description="Shared flow message bus")

    channel: asyncio.Queue = Field(default_factory=asyncio.Queue)
    memory: Memory = Field(default_factory=Memory)
    tools: list[BaseTool] = Field(default_factory=list)
    turns: int = Field(default=0)
    tool_context: ToolContext = Field(default_factory=ToolContext)
    hooks: HookRegistry = Field(default_factory=HookRegistry)

    _pipeline: Consumable | None = PrivateAttr(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def initialize_context(self) -> "AgentContext":
        if self.shared is None:
            raise RuntimeError("Shared data not initialized.")
        self.shared[_SINGLETON_KEY] = self
        return self

    def get_shared(self) -> dict[str, Any]:
        return self.shared

    async def publish(self, event: Any) -> None:
        """Publish an event.

        When a pipeline is set, dispatches to all registered consumers.
        Otherwise falls back to writing to ``channel`` for direct access.
        """
        if self._pipeline is not None:
            await self._pipeline.emit(event)
        else:
            await self.channel.put(event)


class BaseAgentNode(BaseModel, AsyncNode, ABC):
    logger_bind: dict = Field(default_factory=dict)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    """ PocketFlow """
    params: dict = Field(default_factory=dict)
    successors: dict = Field(default_factory=dict)

    max_retries: int = Field(default=1)
    wait: int = Field(default=0)

    def __init__(self, **kwargs):
        BaseModel.__init__(self,**kwargs)
        AsyncNode.__init__(self)
        self._pending_hooks = []
        for attr_name in dir(self.__class__):
            attr = getattr(self.__class__, attr_name, None)
            cfg = getattr(attr, "_hook_config", None)
            if cfg is None:
                continue
            self._pending_hooks.append({
                **cfg,
                "tool_name": cfg.get("tool_name") or self.name,
                "node_name": cfg.get("node_name") or self.name,
                "callback": attr,
            })

    async def prep_async(self, shared: dict[str, Any]) -> AgentContext:
        if _SINGLETON_KEY in shared:
            context = shared[_SINGLETON_KEY]
        else:
            context = AgentContext(shared=shared)
        for h in getattr(self, '_pending_hooks', []):
            context.hooks.on(
                h["point"], h["callback"].__get__(self),
                priority=h.get("priority", 100),
                node_name=h["node_name"], tool_name=h["tool_name"],
                on_error=h["on_error"],
            )
        return context

    @abstractmethod
    async def exec_async(self, context: AgentContext):
        pass

    async def post_async(self, shared, context: AgentContext, exec_res) -> str:
        if isinstance(exec_res, FinishReason):
            return str(exec_res) if exec_res else 'end'
        return exec_res

    @property
    def logger(self) -> "logger":
        return logger.bind(
            node_type=self.__class__.__name__,
            **self.logger_bind
        )