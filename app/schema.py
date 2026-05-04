# MIT License
#
# Copyright (c) 2025 manna_and_poem
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import json
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.util import MultiValueEnum


class FinishReason(MultiValueEnum):
    """ agent """
    STOP = "stop", "end_turn"
    TOOL_CALLS = "tool_calls", "tool_use"
    LENGTH = "length", "max_tokens"
    CONTENT_FILTER = "content_filter"
    STOP_SEQUENCE = "stop_sequence"
    """ flow """
    NONE = "none", None
    TERMINATE = "terminate"
    ERROR = "error"

    def __str__(self):
        return self.value

    def __bool__(self):
        return self != FinishReason.NONE

class Role(str, Enum):
    """Message role options"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    DEVELOPER = "developer"


ROLE_VALUES = tuple(role.value for role in Role)
ROLE_TYPE = Literal[ROLE_VALUES]  # type: ignore


class ToolChoice(str, Enum):
    """Tool choice options"""

    NONE = "none"
    AUTO = "auto"
    REQUIRED = "required"


TOOL_CHOICE_VALUES = tuple(choice.value for choice in ToolChoice)
TOOL_CHOICE_TYPE = Literal[TOOL_CHOICE_VALUES]  # type: ignore


class AgentState(str, Enum):
    """Agent execution states"""

    IDLE = "IDLE"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    ERROR = "ERROR"


class Function(BaseModel):
    name: str
    arguments: str

    @property
    def arguments_dict(self) -> dict[str, Any]:
        return json.loads(self.arguments)


class ToolCall(BaseModel):
    """Represents a tool/function call in a message"""

    id: str
    type: str = "function"
    function: Function

class AgentCompletion(BaseModel):
    """Represents a completion"""
    reasoning_content: str | None = None
    full_reasoning_content: str | None = None
    content: str | None = None
    full_content: str | None = None
    role: ROLE_TYPE
    tool_calls: list[ToolCall] | None = None
    full_tool_calls: list[ToolCall] | None = None
    finish_reason: FinishReason = FinishReason.NONE


class Message(BaseModel):
    """Represents a chat message in the conversation"""

    role: ROLE_TYPE = Field(...)  # type: ignore
    content: str | None  = Field(default=None)
    reasoning_content: str | None = Field(default=None)
    tool_calls: list[ToolCall] | None = Field(default=None)
    name: str | None  = Field(default=None)
    tool_call_id: str | None  = Field(default=None)

    def __add__(self, other) -> list["Message"]:
        """支持 Message + list 或 Message + Message 的操作"""
        if isinstance(other, list):
            return [self] + other
        elif isinstance(other, Message):
            return [self, other]
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: '{type(self).__name__}' and '{type(other).__name__}'"
            )

    def __radd__(self, other) -> list["Message"]:
        """支持 list + Message 的操作"""
        if isinstance(other, list):
            return other + [self]
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: '{type(other).__name__}' and '{type(self).__name__}'"
            )

    def to_dict(self) -> dict:
        """Convert message to dictionary format"""
        message = {"role": self.role}
        if self.reasoning_content is not None:
            message["reasoning_content"] = self.reasoning_content
        if self.content is not None:
            message["content"] = self.content
        if self.tool_calls is not None:
            message["tool_calls"] = [tool_call.model_dump() for tool_call in self.tool_calls]
        if self.name is not None:
            message["name"] = self.name
        if self.tool_call_id is not None:
            message["tool_call_id"] = self.tool_call_id
        return message

    @classmethod
    def user_message(
        cls, content: str
    ) -> "Message":
        """Create a user message"""
        return cls(role=Role.USER, content=content)

    @classmethod
    def system_message(cls, content: str) -> "Message":
        """Create a system message"""
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def assistant_message(
        cls, content: str | None = None, reasoning_content: str | None = None
    ) -> "Message":
        """Create an assistant message"""
        return cls(role=Role.ASSISTANT, content=content, reasoning_content=reasoning_content)

    @classmethod
    def tool_message(
        cls, content: str, name, tool_call_id: str
    ) -> "Message":
        """Create a tool message"""
        return cls(
            role=Role.TOOL,
            content=content,
            name=name,
            tool_call_id=tool_call_id,
        )

    @classmethod
    def from_tool_calls(
        cls,
        tool_calls: list[Any],
        content: str  | list[str] = "",
        reasoning_content: str | None = None,
        **kwargs,
    ):
        """Create ToolCallsMessage from raw tool calls.

        Args:
            tool_calls: Raw tool calls from LLM
            content: Optional message content
        """
        formatted_calls = [
            ToolCall(id = call.id, function = call.function.model_dump(), type = "function")
            for call in tool_calls
        ]
        return cls(
            role=Role.ASSISTANT,
            content=content,
            tool_calls=formatted_calls,
            reasoning_content=reasoning_content,
            **kwargs,
        )


class Memory(BaseModel):
    messages: list[Message] = Field(default_factory=list)
    max_messages: int = Field(default=100)

    def upsert_message(self, message: Message, index: int, role: ROLE_TYPE = None) -> None:
        if self.messages:
            if len(self.messages) > index:
                if role and self.messages[index].role != role:
                    self.messages.insert(index, message)
                else:
                    self.messages[index] = message
            elif len(self.messages) == index:
                self.messages.append(message)
            else:
                raise IndexError(f"index {index} out of range")
        elif index == 0:
            self.messages.append(message)

    def add_message(self, message: Message) -> None:
        """Add a message to memory"""
        self.messages.append(message)
        # Optional: Implement message limit
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def add_messages(self, messages: list[Message]) -> None:
        """Add multiple messages to memory"""
        self.messages.extend(messages)
        # Optional: Implement message limit
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def clear(self) -> None:
        """Clear all messages"""
        self.messages.clear()

    def get_recent_messages(self, n: int) -> list[Message]:
        """Get n most recent messages"""
        return self.messages[-n:]

    def to_dict_list(self) -> list[dict]:
        """Convert messages to list of dicts"""
        return [msg.to_dict() for msg in self.messages]