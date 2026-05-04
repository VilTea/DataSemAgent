import json
from typing import Any

from pydantic import Field, ConfigDict, model_validator, create_model

from app.mcp import get_client_pool
from app.schema import ToolCall
from app.tool.base import BaseTool, ToolParameters, ToolResult


class MCPTool(BaseTool):
    server_id: str = Field(..., description="MCP服务器ID")
    mcp_tool_name: str = Field(..., description="MCP工具名称")
    timeout: int = Field(default=30, description="请求超时时间（秒）")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def initialize_mcp_tool(self) -> "MCPTool":
        if not self.mcp_tool_name:
            self.mcp_tool_name = self.name
        return self

    async def execute(self, tool_call: ToolCall, **kwargs: Any) -> ToolResult:
        try:
            arguments = self._parse_arguments(tool_call.function.arguments)

            client_pool = get_client_pool()
            client = client_pool.get_client(self.server_id)

            if not client:
                return ToolResult.failure_response(
                    tool_call.id,
                    self.name,
                    f"MCP server not registered: {self.server_id}",
                )

            result = await client.call_tool(self.mcp_tool_name, arguments)

            if result.isError:
                return ToolResult.failure_response(
                    tool_call.id,
                    self.name,
                    str(result.content),
                )

            content = result.content if isinstance(result.content, str) else str(result.content)

            return ToolResult.success_response(tool_call.id, self.name, content)

        except Exception as e:
            return ToolResult.failure_response(
                tool_call.id,
                self.name,
                f"Error executing MCP tool: {str(e)}",
            )

    @staticmethod
    def _parse_arguments(arguments: str | dict | None) -> dict[str, Any]:
        if arguments is None:
            return {}
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str):
            if not arguments.strip():
                return {}
            try:
                return json.loads(arguments)
            except json.JSONDecodeError:
                return {"_raw": arguments}
        return {}


def create_mcp_tool(
        name: str,
        description: str,
        server_id: str,
        mcp_tool_name: str | None = None,
        parameters: dict | None = None,
        strict: bool = False,
        timeout: int = 30,
) -> type[MCPTool]:
    # 处理参数
    tool_params = parameters or {}
    params_obj = ToolParameters(**tool_params) if tool_params else None
    base_fields = MCPTool.model_fields
    # 使用 Field 定义字段，保留文档字符串
    fields = {
        "name": (str, Field(default=name,
                            **{k: v for k, v in base_fields['name'].asdict()['attributes'].items() if k != "default"})),
        "description": (str, Field(description,
                                   **{k: v for k, v in base_fields['description'].asdict()['attributes'].items() if
                                      k != "default"})),
        "server_id": (str, Field(server_id,
                                 **{k: v for k, v in base_fields['server_id'].asdict()['attributes'].items() if
                                    k != "default"})),
        "mcp_tool_name": (str, Field(mcp_tool_name,
                                     **{k: v for k, v in base_fields['mcp_tool_name'].asdict()['attributes'].items() if
                                        k != "default"})),
        "parameters": (dict | ToolParameters | None, Field(
            params_obj,
            **{k: v for k, v in base_fields['parameters'].asdict()['attributes'].items() if k != "default"}
        )),
        "strict": (bool, Field(strict,
                               **{k: v for k, v in base_fields['strict'].asdict()['attributes'].items() if
                                  k != "default"})),
        "timeout": (int, Field(timeout,
                               **{k: v for k, v in base_fields['timeout'].asdict()['attributes'].items() if
                                  k != "default"})),
    }

    return create_model(
        name,
        __base__=(MCPTool,),
        __cls_kwargs__={
            "arbitrary_types_allowed": True,  # 保留基类的配置
        },
        **fields
    )
