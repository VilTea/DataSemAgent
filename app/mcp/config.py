from enum import Enum
from typing import Any

from app.config import MCPSettings as _MCPConfigSettings
from app.config import MCPServerToolSettings


class MCPAuthType(str, Enum):
    NONE = "none"
    API_KEY = "api_key"
    BEARER = "bearer"
    BASIC = "basic"


class MCPServerConfig(_MCPConfigSettings):
    server_id: str | None = None

    auth_type: MCPAuthType = MCPAuthType.NONE
    auth_value: str | None = None

    max_retry_attempts: int = 3
    retry_backoff_seconds: float = 1.0

    def __init__(self, server_id: str | None = None, **data):
        super().__init__(**data)
        self.server_id = server_id

    def get_fastmcp_client_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        
        if self.transport_type in ["streamable-http", "sse"]:
            kwargs["url"] = self.url
            if self.headers:
                kwargs["headers"] = self.headers
            if self.auth_type == MCPAuthType.BEARER and self.auth_value:
                kwargs.setdefault("headers", {})["Authorization"] = f"Bearer {self.auth_value}"
            elif self.auth_type == MCPAuthType.API_KEY and self.auth_value:
                kwargs.setdefault("headers", {})["X-API-Key"] = self.auth_value
        elif self.transport_type == "stdio":
            kwargs["command"] = self.command
            if self.args:
                kwargs["args"] = self.args
            if self.env:
                kwargs["env"] = self.env
            if self.cwd:
                kwargs["cwd"] = self.cwd
        elif self.transport_type in ["python-module", "node-module"]:
            kwargs["path"] = self.module_path

        return kwargs

    @classmethod
    def from_config(cls, server_id: str, config: _MCPConfigSettings) -> "MCPServerConfig":
        return cls(
            server_id=server_id,
            transport_type=config.transport_type,
            url=config.url,
            headers=config.headers,
            timeout=config.timeout,
            command=config.command,
            args=config.args,
            env=config.env,
            cwd=config.cwd,
            module_path=config.module_path,
            auto_discover=config.auto_discover,
            tools=config.tools,
        )


__all__ = [
    "MCPAuthType",
    "MCPServerConfig",
    "MCPServerToolSettings",
]
