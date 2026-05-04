from app.mcp.client import get_client_pool
from app.mcp.config import MCPServerConfig


class MCPToolInfo:
    def __init__(
        self,
        name: str,
        description: str = "",
        input_schema: dict | None = None,
        server_id: str | None = None,
    ):
        self.name = name
        self.description = description
        self.input_schema = input_schema or {}
        self.server_id = server_id

    def __repr__(self) -> str:
        return f"MCPToolInfo(name={self.name!r}, server_id={self.server_id!r})"


class MCPToolRegistry:
    def __init__(self):
        self._tools: dict[str, MCPToolInfo] = {}

    def register_tool(self, tool: MCPToolInfo) -> None:
        key = self._make_key(tool.name, tool.server_id)
        self._tools[key] = tool

    def unregister_tool(self, name: str, server_id: str | None = None) -> bool:
        key = self._make_key(name, server_id)
        return self._tools.pop(key, None) is not None

    def get_tool(self, name: str, server_id: str | None = None) -> MCPToolInfo | None:
        key = self._make_key(name, server_id)
        return self._tools.get(key)

    def list_tools(self) -> list[str]:
        return [tool.name for tool in self._tools.values()]

    def get_all_tools(self) -> list[MCPToolInfo]:
        return list(self._tools.values())

    def clear(self) -> None:
        self._tools.clear()

    @staticmethod
    def _make_key(name: str, server_id: str | None = None) -> str:
        if server_id:
            return f"{server_id}:{name}"
        return name

    async def auto_discover_tools(
        self,
        config: MCPServerConfig,
    ) -> list[MCPToolInfo]:
        client = get_client_pool().register_server(config)
        
        tools = await client.list_tools()
        
        discovered = []
        for tool in tools:
            tool_info = MCPToolInfo(
                name=tool.name,
                description=tool.description or "",
                input_schema=tool.inputSchema,
                server_id=config.server_id,
            )
            self.register_tool(tool_info)
            discovered.append(tool_info)
        
        return discovered


_registry = MCPToolRegistry()


def get_registry() -> MCPToolRegistry:
    return _registry
