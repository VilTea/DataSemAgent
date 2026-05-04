from abc import ABC, abstractmethod
from typing import Literal, Any

from pydantic import BaseModel, Field, ConfigDict, model_validator, PrivateAttr, field_serializer

from app.schema import ToolCall

class ToolContext(BaseModel):
    shared: dict[str, Any] = Field(default_factory=dict)

class ToolParameters(BaseModel):
    type: str = Field(description="object", default="The type of tool")
    properties: dict = Field(default={}, description="properties")
    required: list[str] = Field(default=True, description="required")


class ToolResult(BaseModel):
    tool_call_id: str = Field(..., description="tool id")
    name: str | None = Field(None, description="tool name")
    content: str | None = Field(..., description="tool content")

    _success: bool = PrivateAttr(default=True)

    def is_success(self):
        return self._success

    @staticmethod
    def success_response(tool_call_id: str, name: str, content: str):
        return ToolResult(tool_call_id=tool_call_id, name=name, content=content, _success=True)

    @staticmethod
    def failure_response(tool_call_id: str, name: str, content: str):
        return ToolResult(tool_call_id=tool_call_id, name=name,content=content, _success=False)


class BaseTool(BaseModel, ABC):
    # Metadata
    permission: Literal["global", "skills", "agent"] = Field(default="agent", description="permission")

    # Specification
    name: str = Field(..., description="The name of the tool.")
    description: str = Field(..., description="The description of the tool.")
    strict: bool = Field(default=False, description="Whether the tool should be strict or not.")
    parameters: dict | ToolParameters | None = Field(..., description="The parameters of the tool.")

    # shared
    tool_context: ToolContext | None = Field(default=None, description="The tool context.")

    # display
    _display_name: str | None = PrivateAttr(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def display_name(self):
        return self._display_name or self.name

    @display_name.setter
    def display_name(self, value: str):
        self._display_name = value

    @model_validator(mode="after")
    def initialize_tool(self) -> "BaseTool":
        if self.parameters and isinstance(self.parameters, dict):
            self.parameters = ToolParameters(**self.parameters)
        if not self.parameters:
            self.parameters = {}
        return self

    @model_validator(mode="after")
    def _scan_hooks(self) -> "BaseTool":
        self._pending_hooks = []
        for attr_name in dir(self.__class__):
            attr = getattr(self.__class__, attr_name, None)
            cfg = getattr(attr, "_hook_config", None)
            if cfg is None:
                continue
            self._pending_hooks.append({
                **cfg,
                "tool_name": cfg.get("tool_name") or self.name,
                "node_name": cfg.get("node_name"),
                "callback": attr,
            })
        return self

    @field_serializer('parameters')
    def serialize_parameters(self, parameters: dict | ToolParameters | None, _info):
        if parameters is None:
            return {}
        if isinstance(parameters, ToolParameters):
            return parameters.model_dump(exclude_none=True)
        return parameters

    def to_dict(self) -> dict:
        return {
            "type": "function",
            "function": self.model_dump(include={"name", "strict", "description", "parameters"})
        }

    def to_json(self) -> str:
        return f"""{{
            "type": "function",
            "function": {self.model_dump_json(include={"name", "strict", "description", "parameters"})}
        }}"""

    def tool_result(self, tool_call_id: str, content: str, success: bool = True) -> ToolResult:
        return ToolResult(
            tool_call_id=tool_call_id,
            name=self.name,
            content=content,
            _success=success
        )

    def is_available(self) -> bool:
        """Whether this tool is available for use.

        Override to gate tools behind initialization state.
        AgentNode filters out tools where this returns False.
        Default: always available.
        """
        return True

    async def __call__(self, **kwargs) -> ToolResult:
        return await self.execute(**kwargs)

    @abstractmethod
    async def execute(self, tool_call: ToolCall, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
