from app.hook import HookPoint
from app.node.base import BaseAgentNode, AgentContext
from app.schema import Message
from app.tool.base import BaseTool, ToolResult


class ToolNode(BaseAgentNode):
    async def exec_async(self, context: AgentContext) -> list[ToolResult]:
        message = context.memory.messages[-1]
        tool_results = []
        if tool_calls := message.tool_calls:
            tools :dict[str, BaseTool] = {tool.name:tool for tool in context.tools}
            for tool_call in tool_calls:
                if tool := tools.get(tool_call.function.name):
                    await context.hooks.emit(HookPoint.TOOL_BEFORE, ctx=context, tool_call=tool_call, tool=tool)
                    result: ToolResult
                    try:
                        result : ToolResult = await tool.execute(tool_call)
                    except Exception as e:
                        result = ToolResult.failure_response(
                            tool_call.id,
                            tool_call.function.name,
                            f"Tool execution failed: {str(e)}")
                        self.logger.error(f"Tool execution error: {e}")
                    await context.hooks.emit(HookPoint.TOOL_AFTER, ctx=context, tool_call=tool_call, tool=tool, result=result)
                    tool_results.append(result)
                    msg = Message.tool_message(
                            content=result.content,
                            name=tool_call.function.name,
                            tool_call_id=tool_call.id,
                        )
                    context.memory.add_message(msg)
                    await context.publish(msg)
                else:
                    result: ToolResult = ToolResult.failure_response(
                        tool_call.id,
                        tool_call.function.name,
                        f"Permission Denied: {tool_call.function.name}"
                    )
                    tool_results.append(result)
                    msg = Message.tool_message(
                            content=result.content,
                            name=tool_call.function.name,
                            tool_call_id=tool_call.id,
                        )
                    context.memory.add_message(msg)
                    await context.publish(msg)
        return tool_results

    async def post_async(self, shared, context: AgentContext, exec_res) -> str:
        context.tools.clear()
        return "default"