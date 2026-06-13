import json
import time
from typing import Protocol, runtime_checkable, AsyncGenerator, ClassVar

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI, OpenAIError
from pydantic import BaseModel, Field, ConfigDict
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type

from app.config import LLMSettings, config
from app.schema import Message, ToolCall, TOOL_CHOICE_TYPE, ToolChoice, Role, Function, AgentCompletion, FinishReason


@runtime_checkable
class LLM(Protocol):
    type: ClassVar[str]

    async def ask(self, messages: list[Message], stream: bool = True,
                  temperature: float | None = None) \
            -> AsyncGenerator[AgentCompletion | None]: ...
    async def ask_tool(self, messages: list[Message], stream: bool = True,
                       tools: list[dict] | None = None, temperature: float | None = None,
                       tool_choice: TOOL_CHOICE_TYPE | None = None)\
            -> AsyncGenerator[AgentCompletion | None]: ...

_LLM_REGISTRY: dict[str, type] = {}

def register_llm(cls: type) -> type:
    """Decorator: register an LLM implementation by its type string."""
    _LLM_REGISTRY[cls.type] = cls
    return cls

def create_llm(config_name: str = 'default', llm_setting: LLMSettings | None = None) -> LLM:
    """Factory: create the correct LLM instance based on config type."""
    if not llm_setting:
        if config_name not in config.llm:
            config_name = 'default'
        llm_setting = config.llm[config_name]
    cls = _LLM_REGISTRY.get(llm_setting.type)
    if cls is None:
        raise ValueError(f"Unknown LLM type '{llm_setting.type}'. Known: {list(_LLM_REGISTRY)}")
    return cls(config_name=config_name, llm_setting=llm_setting)

@register_llm
class OpenAILLM(BaseModel):
    config: LLMSettings
    client: AsyncOpenAI = Field(exclude=True)

    type: ClassVar[str] = "openai"
    _instance: ClassVar[dict[str, "OpenAILLM"]] = {}

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __new__(cls, config_name: str = 'default', llm_setting: LLMSettings | None = None):
        if config_name not in cls._instance:
            instance = super().__new__(cls)
            cls._instance[config_name] = instance
        return cls._instance[config_name]

    def __init__(self, config_name: str = 'default', llm_setting: LLMSettings | None = None):
        if not hasattr(self, 'client'):
            if not llm_setting:
                if config_name not in config.llm:
                    config_name = 'default'
                llm_setting = config.llm[config_name]
            super().__init__(config=llm_setting,
                             client=AsyncOpenAI(base_url=llm_setting.base_url,api_key=llm_setting.api_key))

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(
            (OpenAIError, Exception, ValueError)
        ),  # Don't retry TokenLimitExceeded
    )
    async def ask(self, messages: list[Message], stream: bool = True, temperature: float | None = None) -> AsyncGenerator[AgentCompletion | None]:
        """Send a chat completion request and yield responses."""
        request_params = {
            "model": self.config.model,
            "messages": [msg.model_dump() for msg in messages],
            "temperature": temperature or self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": stream,
            **self.config.args
        }
        if stream:
            request_params["stream_options"] = {"include_usage": True}
        completion = await self.client.chat.completions.create(**request_params)
        if stream:
            content = ''
            _usage_obj = None
            async for chunk in completion:
                if hasattr(chunk, 'usage') and chunk.usage:
                    _usage_obj = chunk.usage
                if chunk.choices and (delta := chunk.choices[0].delta):
                    content += delta.content or ''
                    response = AgentCompletion(
                        role=Role.ASSISTANT,
                        content=delta.content,
                        full_content=content,
                        finish_reason=FinishReason(chunk.choices[0].finish_reason)
                    )
                    if _usage_obj:
                        response.usage_input_tokens = _usage_obj.prompt_tokens
                        response.usage_output_tokens = _usage_obj.completion_tokens
                    yield response
        else:
            if completion.choices and (message := completion.choices[0].message):
                response = AgentCompletion(
                    role=Role.ASSISTANT,
                    content=message.content,
                    full_content=message.content,
                    finish_reason=FinishReason(completion.choices[0].finish_reason)
                )
                if hasattr(completion, 'usage') and completion.usage:
                    response.usage_input_tokens = completion.usage.prompt_tokens
                    response.usage_output_tokens = completion.usage.completion_tokens
                yield response

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(
            (OpenAIError, Exception, ValueError)
        ),  # Don't retry TokenLimitExceeded
    )
    async def ask_tool(self, messages: list[Message], stream: bool = True,
                       tools: list[dict] | None = None, temperature: float | None = None,
                       tool_choice: TOOL_CHOICE_TYPE | None = ToolChoice.AUTO) \
            -> AsyncGenerator[AgentCompletion | None]:
        """
        Send a chat completion request with tool calls support.

        Args:
            messages: List of conversation messages
            stream: Whether to stream the response
            tools: Optional list of tools available to the model
            temperature: Optional temperature override
            tool_choice: Optional tool choice override

        Yields:
            For streaming: Content chunks and tool call deltas
            For non-streaming: Complete content string and list of tool calls
        """
        request_params = {
            "model": self.config.model,
            "messages": [msg.model_dump() for msg in messages],
            "temperature": temperature or self.config.temperature,
            "max_tokens": self.config.max_tokens,
            # "max_completion_tokens": self.config.max_tokens,
            "stream": stream,
            **self.config.args
        }
        if tools:
            request_params["tools"] = tools
            request_params["tool_choice"] = tool_choice or ToolChoice.AUTO
        if stream:
            request_params["stream_options"] = {"include_usage": True}
        t_start = time.time_ns()
        first_chunk = True
        completion = await self.client.chat.completions.create(**request_params)

        if stream:
            role = None
            tool_calls = {}
            content = ''
            reasoning_content = ''
            _usage_obj = None  # captured from usage chunk (empty choices)
            _ttft_ns: int | None = None
            async for chunk in completion:
                choice = chunk.choices[0] if chunk.choices else None
                delta = chunk.choices[0].delta if chunk.choices else None
                # Capture usage from this chunk (may arrive on final or dedicated chunk)
                if hasattr(chunk, 'usage') and chunk.usage:
                    _usage_obj = chunk.usage
                # Pure usage chunk (no choices, no delta) — skip
                if not choice and not delta:
                    continue
                if not role and delta.role:
                    role = Role(delta.role)
                response = AgentCompletion(role=role)
                if first_chunk:
                    _ttft_ns = time.time_ns() - t_start
                    first_chunk = False
                response.time_to_first_chunk_ns = _ttft_ns
                reasoning_content += delta.model_extra.get('reasoning_content') or ''
                response.reasoning_content = delta.model_extra.get('reasoning_content') or ''
                if delta and delta.content:
                    content += delta.content or ''
                    response.content = delta.content
                if delta and (_tool_calls := delta.tool_calls):
                    response.tool_calls = _tool_calls
                    for tool_call in _tool_calls:
                        if not tool_calls.get(tool_call.index):
                            tool_calls[tool_call.index] = ToolCall(
                                id=tool_call.id,
                                function=Function(
                                    name = tool_call.function.name,
                                    arguments = tool_call.function.arguments or '',
                                )
                            )
                        else:
                            tool_calls[tool_call.index].function.arguments += tool_call.function.arguments or ''
                elif delta and (function_call := delta.function_call):
                    response.tool_calls.append(ToolCall(
                            id = '-1',
                            type = 'function',
                            function=Function(
                                name = function_call.name,
                                arguments = function_call.arguments or '',
                            ),
                        ))
                    if not tool_calls.get(-1):
                        tool_calls[-1] = ToolCall(
                            id = '-1',
                            type = 'function',
                            function=Function(
                                name = function_call.name,
                                arguments = function_call.arguments or '',
                            )
                        )
                    else:
                        tool_calls[-1].function.arguments += function_call.function.arguments or ''
                response.full_reasoning_content = reasoning_content
                response.full_content = content
                response.full_tool_calls = list(tool_calls.values())
                response.finish_reason = FinishReason(choice.finish_reason)
                # Attach usage on the last response
                if _usage_obj:
                    response.usage_input_tokens = _usage_obj.prompt_tokens
                    response.usage_output_tokens = _usage_obj.completion_tokens
                yield response
                if choice.finish_reason:
                    break
        else:
            if completion.choices:
                message = completion.choices[0].message
                response = AgentCompletion(**message.model_dump(
                        exclude={'refusal', 'annotations', 'audio', 'function_call'}))
                if not message.tool_calls and (function_call := message.function_call):
                    response.tool_calls = [ToolCall(
                        id='-1',
                        type='function',
                        function=Function(**function_call.model_dump(exclude=set('index')))
                    )]
                response.finish_reason = FinishReason(completion.choices[0].finish_reason)
                response.full_reasoning_content = response.reasoning_content
                response.full_content = response.content
                response.full_tool_calls = response.tool_calls
                if hasattr(completion, 'usage') and completion.usage:
                    response.usage_input_tokens = completion.usage.prompt_tokens
                    response.usage_output_tokens = completion.usage.completion_tokens
                yield response
            else:
                yield None


@register_llm
class AnthropicLLM(BaseModel):
    config: LLMSettings
    client: AsyncAnthropic = Field(exclude=True)

    type: ClassVar[str] = "anthropic"
    _instance: ClassVar[dict[str, "AnthropicLLM"]] = {}

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __new__(cls, config_name: str = 'default', llm_setting: LLMSettings | None = None):
        if config_name not in cls._instance:
            instance = super().__new__(cls)
            cls._instance[config_name] = instance
        return cls._instance[config_name]

    def __init__(self, config_name: str = 'default', llm_setting: LLMSettings | None = None):
        if not hasattr(self, 'client'):
            from anthropic import AsyncAnthropic
            if not llm_setting:
                if config_name not in config.llm:
                    config_name = 'default'
                llm_setting = config.llm[config_name]
            super().__init__(config=llm_setting,
                             client=AsyncAnthropic(base_url=llm_setting.base_url, api_key=llm_setting.api_key))

    @staticmethod
    def _parse_messages(messages: list[Message]) -> tuple[str | None, list[dict]]:
        """Convert OpenAI-format messages to Anthropic format.

        Returns (system_text, anthropic_messages).
        System messages are extracted and concatenated separately.
        """
        system_parts = []
        anthropic_messages: list[dict] = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_parts.append(msg.content)
            elif msg.role == Role.USER:
                anthropic_messages.append({"role": "user", "content": msg.content or ""})
            elif msg.role == Role.ASSISTANT:
                if msg.tool_calls:
                    blocks: list[dict] = []
                    if msg.content:
                        blocks.append({"type": "text", "text": msg.content})
                    for tc in msg.tool_calls:
                        try:
                            inp = json.loads(tc.function.arguments)
                        except (json.JSONDecodeError, TypeError):
                            inp = {}
                        blocks.append({
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.function.name,
                            "input": inp,
                        })
                    anthropic_messages.append({"role": "assistant", "content": blocks})
                else:
                    anthropic_messages.append({"role": "assistant", "content": msg.content or ""})
            elif msg.role == Role.TOOL:
                anthropic_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": msg.content or "",
                    }]
                })

        system = "\n\n".join(system_parts) if system_parts else None
        return system, anthropic_messages

    @staticmethod
    def _convert_tools(tools: list[dict]) -> list[dict]:
        """Convert OpenAI tool definitions to Anthropic format."""
        result = []
        for tool in tools:
            if tool.get("type") == "function":
                result.append({
                    "name": tool["function"]["name"],
                    "description": tool["function"].get("description", ""),
                    "input_schema": tool["function"].get("parameters", {"type": "object", "properties": {}}),
                })
        return result

    @staticmethod
    def _convert_tool_choice(tool_choice: TOOL_CHOICE_TYPE | None) -> dict | None:
        if tool_choice is None or tool_choice == ToolChoice.AUTO:
            return {"type": "auto"}
        elif tool_choice == ToolChoice.REQUIRED:
            return {"type": "any"}
        elif tool_choice == ToolChoice.NONE:
            return None
        return {"type": str(tool_choice)}

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type((Exception, ValueError)),
    )
    async def ask(
        self, messages: list[Message], stream: bool = True, temperature: float | None = None
    ) -> AsyncGenerator[AgentCompletion | None]:
        from anthropic import APIError
        system, anthropic_messages = self._parse_messages(messages)

        kwargs: dict = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": anthropic_messages,
            **self.config.args
        }
        if system:
            kwargs["system"] = system
        if temperature is not None:
            kwargs["temperature"] = temperature
        elif self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature

        if stream:
            _usage_input: int | None = None
            _usage_output: int | None = None
            async with self.client.messages.stream(**kwargs) as stream_ctx:
                content = ''
                async for event in stream_ctx:
                    if event.type == "message_start":
                        if hasattr(event, 'message') and event.message.usage:
                            _usage_input = event.message.usage.input_tokens
                    elif event.type == "text_delta":
                        content += event.text
                        yield AgentCompletion(
                            role=Role.ASSISTANT,
                            content=event.text,
                            full_content=content,
                        )
                    elif event.type == "message_delta":
                        if hasattr(event, 'usage') and event.usage:
                            _usage_output = event.usage.output_tokens
                        yield AgentCompletion(
                            role=Role.ASSISTANT,
                            full_content=content,
                            finish_reason=FinishReason(event.delta.stop_reason),
                            usage_input_tokens=_usage_input,
                            usage_output_tokens=_usage_output,
                        )
        else:
            message = await self.client.messages.create(**kwargs)
            text = "".join(block.text for block in message.content if block.type == "text")
            yield AgentCompletion(
                role=Role.ASSISTANT,
                content=text,
                full_content=text,
                finish_reason=FinishReason(message.stop_reason),
                usage_input_tokens=message.usage.input_tokens if message.usage else None,
                usage_output_tokens=message.usage.output_tokens if message.usage else None,
            )

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type((Exception, ValueError)),
    )
    async def ask_tool(
        self, messages: list[Message], stream: bool = True,
        tools: list[dict] | None = None, temperature: float | None = None,
        tool_choice: TOOL_CHOICE_TYPE | None = ToolChoice.AUTO
    ) -> AsyncGenerator[AgentCompletion | None]:
        system, anthropic_messages = self._parse_messages(messages)

        kwargs: dict = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": anthropic_messages,
            **self.config.args
        }
        if system:
            kwargs["system"] = system
        if temperature is not None:
            kwargs["temperature"] = temperature
        elif self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature
        if tools:
            kwargs["tools"] = self._convert_tools(tools)
            tc = self._convert_tool_choice(tool_choice)
            if tc is not None:
                kwargs["tool_choice"] = tc

        if stream:
            t_start = time.time_ns()
            first_chunk = True
            _usage_input: int | None = None
            _usage_output: int | None = None
            _ttft_ns: int | None = None
            async with self.client.messages.stream(**kwargs) as stream_ctx:
                content = ''
                tool_calls: dict[int, ToolCall] = {}
                current_tool_index = -1

                async for event in stream_ctx:
                    if event.type == "message_start":
                        # Anthropic sends usage.input_tokens on message_start
                        if hasattr(event, 'message') and event.message.usage:
                            _usage_input = event.message.usage.input_tokens
                    elif event.type == "text_delta":
                        content += event.text
                        response = AgentCompletion(
                            role=Role.ASSISTANT,
                            content=event.text,
                            full_content=content,
                            full_tool_calls=list(tool_calls.values()) if tool_calls else None,
                        )
                        if first_chunk:
                            _ttft_ns = time.time_ns() - t_start
                            first_chunk = False
                        response.time_to_first_chunk_ns = _ttft_ns
                        yield response
                    elif event.type == "content_block_start":
                        if event.content_block.type == "tool_use":
                            current_tool_index += 1
                            cb = event.content_block
                            tc = ToolCall(
                                id=cb.id,
                                function=Function(name=cb.name, arguments=""),
                            )
                            tool_calls[current_tool_index] = tc
                            response = AgentCompletion(
                                role=Role.ASSISTANT,
                                full_content=content,
                                tool_calls=[tc],
                                full_tool_calls=list(tool_calls.values()),
                            )
                            if first_chunk:
                                _ttft_ns = time.time_ns() - t_start
                                first_chunk = False
                            response.time_to_first_chunk_ns = _ttft_ns
                            yield response
                    elif event.type == "content_block_delta":
                        if event.delta.type == "input_json_delta":
                            tc = tool_calls.get(current_tool_index)
                            if tc:
                                tc.function.arguments += event.delta.partial_json
                                yield AgentCompletion(
                                    role=Role.ASSISTANT,
                                    full_content=content,
                                    tool_calls=[tc],
                                    full_tool_calls=list(tool_calls.values()),
                                )
                    elif event.type == "message_delta":
                        # message_delta carries final usage.output_tokens
                        if hasattr(event, 'usage') and event.usage:
                            _usage_output = event.usage.output_tokens
                        yield AgentCompletion(
                            role=Role.ASSISTANT,
                            full_content=content,
                            full_tool_calls=list(tool_calls.values()) if tool_calls else None,
                            finish_reason=FinishReason(event.delta.stop_reason),
                            usage_input_tokens=_usage_input,
                            usage_output_tokens=_usage_output,
                        )
        else:
            message = await self.client.messages.create(**kwargs)
            text = "".join(block.text for block in message.content if block.type == "text")
            tc_list = []
            for block in message.content:
                if block.type == "tool_use":
                    tc_list.append(ToolCall(
                        id=block.id,
                        function=Function(name=block.name, arguments=json.dumps(block.input)),
                    ))
            yield AgentCompletion(
                role=Role.ASSISTANT,
                content=text or None,
                full_content=text or None,
                tool_calls=tc_list if tc_list else None,
                full_tool_calls=tc_list if tc_list else None,
                finish_reason=FinishReason(message.stop_reason),
                usage_input_tokens=message.usage.input_tokens if message.usage else None,
                usage_output_tokens=message.usage.output_tokens if message.usage else None,
            )