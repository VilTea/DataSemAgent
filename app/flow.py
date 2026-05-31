from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from pocketflow import AsyncFlow, AsyncNode

from app.hook import HookPoint
from app.node import AgentContext, AgentNode, ToolNode
from app.schema import Message, FinishReason

if TYPE_CHECKING:
    from app.pipeline.abc import Consumable


class AgentFlow(AsyncFlow):
    def __init__(self, pipeline: Consumable | None = None, **kwargs):
        super().__init__(**kwargs)
        self.context: AgentContext = AgentContext()
        self._pipeline = pipeline

    async def ask(self, prompt: str, context: AgentContext = None):
        if context:
            self.context = context

        await self.context.hooks.emit(HookPoint.FLOW_START, ctx=self.context)

        self.context.memory.add_message(Message.user_message(prompt))

        try:
            if self._pipeline is not None:
                self.context._pipeline = self._pipeline
                await self._pipeline.start(self.context)
            await asyncio.sleep(0)
            result = await self._run_async(self.context.get_shared())
            await asyncio.sleep(0)
            return result
        finally:
            if self._pipeline is not None:
                await self._pipeline.stop()
            self.context.turns += 1
            await self.context.hooks.emit(HookPoint.FLOW_END, ctx=self.context)


def react_flow(
    agent_node: AgentNode,
    pipeline: Consumable | None = None,
) -> AgentFlow:
    tool_node = ToolNode()
    output = AsyncNode()

    agent_node >> output
    agent_node - str(FinishReason.NONE) >> output
    agent_node - str(FinishReason.STOP) >> output
    agent_node - str(FinishReason.TOOL_CALLS) >> tool_node >> agent_node

    return AgentFlow(pipeline=pipeline, start=agent_node)