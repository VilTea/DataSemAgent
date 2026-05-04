import uuid
from typing import Any

from pydantic import Field, model_validator, ConfigDict

from app.hook import HookPoint
from app.llm import LLM, create_llm
from app.node.base import BaseAgentNode, AgentContext
from app.schema import Message, AgentCompletion, FinishReason, ToolChoice, Role
from app.tool.base import BaseTool


class AgentNode(BaseAgentNode):
    id: str = Field(default=str(uuid.uuid4()), description="唯一标识符")

    name: str = Field(..., description="智能体名称", min_length=1, max_length=100)
    description: str | None = Field(default=None, description="智能体描述")
    system_prompt: str | None = Field(default=None, description="系统提示词")

    tools: list[BaseTool] = Field(default_factory=list, description="工具列表")
    tool_choice: ToolChoice = Field(default=ToolChoice.AUTO)

    llm: LLM | None = Field(default=None, description="大模型请求实例")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def initialize_agent(self) -> "AgentNode":
        if not self.llm:
            self.llm = create_llm(config_name=self.name) # type: ignore[arg-type]

        self.logger_bind = {
            "agent_id": self.id,
            "agent_name": self.name,
            "agent_api_type": self.llm.__class__.__name__
        }

        self.tools = [t for t in self.tools if t.is_available()]

        return self

    async def prep_async(self, shared: dict[str, Any]) -> AgentContext:
        """
        准备阶段：从 shared 中获取数据

        Args:
            shared: 共享数据对象

        Returns:
            准备的数据，传递给 exec 方法
        """
        context = await super().prep_async(shared)
        self.logger.debug(f"Preparing agent: {self.name}")

        for tool in self.tools:
            tool.tool_context = context.tool_context
            for h in getattr(tool, "_pending_hooks", []):
                context.hooks.on(
                    h["point"], h["callback"].__get__(tool),
                    priority=h.get("priority", 100),
                    node_name=h["node_name"], tool_name=h["tool_name"],
                    on_error=h["on_error"],
                )

        first_time = not self._init_completed
        if first_time:
            await context.hooks.emit(HookPoint.NODE_INIT_BEFORE, ctx=context, node=self)

        await context.hooks.emit(HookPoint.NODE_PREP_BEFORE, ctx=context, node=self)
        if self.system_prompt:
            has_sys = context.memory.messages and context.memory.messages[0].role == Role.SYSTEM
            if not has_sys:
                context.memory.upsert_message(Message.system_message(self.system_prompt), 0, role=Role.SYSTEM)
        try:
            return context
        finally:
            await context.hooks.emit(HookPoint.NODE_PREP_AFTER, ctx=context, node=self)
            if first_time:
                self._init_completed = True
                await context.hooks.emit(HookPoint.NODE_INIT_AFTER, ctx=context, node=self)

    async def exec_async(self, context: AgentContext) -> FinishReason:
        await context.hooks.emit(HookPoint.NODE_EXEC_BEFORE, ctx=context, node=self)
        method = self.llm.ask_tool if self.tools else self.llm.ask
        kwargs = {"messages": context.memory.messages}
        if self.tools:
            kwargs.update(**{"tools": [tool.to_dict() for tool in self.tools], "tool_choice": self.tool_choice})

        msg_cur: AgentCompletion | None = None
        try:
            async for msg in method(**kwargs): # type: ignore[arg-type]
                if msg:
                    await context.publish(msg)
                    msg_cur = msg

            if msg_cur and msg_cur.finish_reason and msg_cur.finish_reason != FinishReason.NONE:
                message: Message
                if msg_cur.finish_reason == FinishReason.TOOL_CALLS:
                    message = Message.from_tool_calls(tool_calls=msg_cur.full_tool_calls, content=msg_cur.full_content, reasoning_content=msg_cur.full_reasoning_content)
                    context.tools.clear()
                    context.tools.extend(self.tools)
                else:
                    message = Message.assistant_message(content=msg_cur.full_content, reasoning_content=msg_cur.full_reasoning_content)
                context.memory.add_message(message)
                await context.hooks.emit(HookPoint.NODE_EXEC_AFTER, ctx=context, node=self, reason=msg_cur.finish_reason)
                return msg_cur.finish_reason

            await context.hooks.emit(HookPoint.NODE_EXEC_AFTER, ctx=context, node=self, reason=FinishReason.NONE)
            return FinishReason.NONE
        except Exception as e:
            self.logger.error(f"LLM execution failed: {e}")
            await context.hooks.emit(HookPoint.NODE_EXEC_AFTER, ctx=context, node=self, reason=FinishReason.ERROR)
            return FinishReason.ERROR

    async def post_async(self, shared, context: AgentContext, exec_res) -> str:
        context.turns += 1
        return await super().post_async(shared, context, exec_res)