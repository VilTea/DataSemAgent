import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from app.mcp.config import MCPServerConfig


class MCPClientManager:
    def __init__(self, config: MCPServerConfig):
        self.config = config
        self._lock = asyncio.Lock()

    @property
    def transport_type(self) -> str:
        return self.config.transport_type

    @property
    def needs_lock(self) -> bool:
        return self.transport_type in ["stdio", "python-module", "node-module"]

    @asynccontextmanager
    async def get_client(self) -> AsyncIterator:
        import fastmcp
        
        if self.needs_lock:
            async with self._lock:
                async with fastmcp.Client(
                    self.config.get_fastmcp_client_kwargs(),
                    timeout=self.config.timeout,
                ) as client:
                    yield client
        else:
            async with fastmcp.Client(
                self.config.get_fastmcp_client_kwargs(),
                timeout=self.config.timeout,
            ) as client:
                yield client

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> Any:
        async with self.get_client() as client:
            result = await client.call_tool(
                tool_name,
                arguments or {},
                timeout=self.config.timeout,
            )
            return result

    async def list_tools(self) -> list:
        async with self.get_client() as client:
            return await client.list_tools()

    async def list_resources(self) -> list:
        async with self.get_client() as client:
            return await client.list_resources()

    async def list_prompts(self) -> list:
        async with self.get_client() as client:
            return await client.list_prompts()

    async def ping(self) -> bool:
        try:
            async with self.get_client() as client:
                await client.ping()
                return True
        except Exception:
            return False


class MCPClientPool:
    def __init__(self):
        self._clients: dict[str, MCPClientManager] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def get_lock(self, server_id: str) -> asyncio.Lock:
        if server_id not in self._locks:
            self._locks[server_id] = asyncio.Lock()
        return self._locks[server_id]

    def register_server(self, config: MCPServerConfig) -> MCPClientManager:
        client = MCPClientManager(config)
        self._clients[config.server_id] = client
        return client

    def get_client(self, server_id: str) -> MCPClientManager | None:
        return self._clients.get(server_id)

    def unregister_server(self, server_id: str) -> None:
        self._clients.pop(server_id, None)
        self._locks.pop(server_id, None)

    def list_servers(self) -> list[str]:
        return list(self._clients.keys())


_client_pool = MCPClientPool()


def get_client_pool() -> MCPClientPool:
    return _client_pool
