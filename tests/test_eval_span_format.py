"""Tests for Phase 1 eval trace enrichment — span IDs, timestamps, OTel attrs."""
import asyncio
from unittest.mock import MagicMock

from app.eval.collector import EvalCollector
from app.schema import Message, FinishReason


class _MockLLM:
    type = "openai"
    config = MagicMock()
    config.model = "gpt-4"
    config.temperature = 0.7
    config.max_tokens = 4096


class _MockCtx:
    memory = MagicMock()
    hooks = MagicMock()


class _MockToolCall:
    def __init__(self, id_, name, args):
        self.id = id_
        self.function = MagicMock()
        self.function.name = name
        self.function.arguments_dict = args


class _MockTool:
    name = "mock_tool"


class _MockToolResult:
    def model_dump(self, mode, exclude_none):
        return {"content": "result_data"}
    def is_success(self):
        return True


class TestEvalSpanFields:
    def test_session_start_has_span_id_and_timestamp(self):
        collector = EvalCollector()
        collector._enabled = True
        collector._output_dir = "data/eval"

        ctx = _MockCtx()
        ctx.memory.messages = [Message.system_message("You are helpful.")]

        node = MagicMock()
        node.system_prompt = "You are helpful."
        node.llm = _MockLLM()

        collector._session_id = "a1b2c3d4e5f6"
        collector._session_span_id = collector._session_id
        event = collector._build_session_start(ctx, node)

        assert "_timestamp" in event
        assert isinstance(event["_timestamp"], int)
        assert event["span_id"] == collector._session_span_id
        assert event["parent_span_id"] is None
        assert event["session_id"] == collector._session_id

    def test_turn_start_has_span_hierarchy(self):
        collector = EvalCollector()
        collector._enabled = True
        collector._output_dir = "data/eval"

        ctx = _MockCtx()
        ctx.memory.messages = [
            Message.system_message("sys"),
            Message.user_message("hello"),
        ]
        node = MagicMock()
        node.llm = _MockLLM()

        async def _run():
            await collector._on_flow_start(ctx)
            collector._session_span_id = collector._session_id
            collector._msg_snapshot = 0
            collector._turn_started = False
            await collector._on_llm_before(ctx, node)

        asyncio.run(_run())

        events = collector._events
        turn = [e for e in events if e["type"] == "turn_start"][0]
        llm_in = [e for e in events if e["type"] == "llm_input"][0]

        assert "_timestamp" in turn
        assert turn["span_id"] is not None
        assert turn["parent_span_id"] == collector._session_span_id

        assert "_timestamp" in llm_in
        assert llm_in["span_id"] is not None
        assert llm_in["parent_span_id"] == turn["span_id"]

    def test_llm_input_has_otel_attrs(self):
        collector = EvalCollector()
        collector._enabled = True
        collector._output_dir = "data/eval"

        ctx = _MockCtx()
        ctx.memory.messages = [
            Message.system_message("sys"),
            Message.user_message("hello"),
        ]
        node = MagicMock()
        node.llm = _MockLLM()

        async def _run():
            await collector._on_flow_start(ctx)
            collector._session_span_id = collector._session_id
            collector._msg_snapshot = 0
            collector._turn_started = False
            await collector._on_llm_before(ctx, node)

        asyncio.run(_run())

        llm_in = [e for e in collector._events if e["type"] == "llm_input"][0]
        assert llm_in["gen_ai.provider.name"] == "openai"
        assert llm_in["gen_ai.request.model"] == "gpt-4"
        assert llm_in["gen_ai.request.temperature"] == 0.7
        assert llm_in["gen_ai.request.max_tokens"] == 4096
        assert llm_in["gen_ai.request.stream"] is True

    def test_llm_output_has_usage_and_ttft(self):
        collector = EvalCollector()
        collector._enabled = True
        collector._output_dir = "data/eval"

        ctx = _MockCtx()
        # Last message in memory is the assistant response
        ctx.memory.messages = [
            Message.system_message("sys"),
            Message.user_message("hello"),
            Message.assistant_message(content="The answer is 42."),
        ]

        node = MagicMock()
        node._eval_ttft_ns = 850_000_000
        node._eval_usage_input = 420
        node._eval_usage_output = 25
        node._eval_usage_reasoning = 100

        async def _run():
            await collector._on_flow_start(ctx)
            collector._session_span_id = collector._session_id
            collector._turn_span_id = collector._new_span_id()
            collector._turn_count = 0
            await collector._on_llm_after(ctx, node, FinishReason.STOP)

        asyncio.run(_run())

        llm_out = [e for e in collector._events if e["type"] == "llm_output"][0]
        assert "_timestamp" in llm_out
        assert llm_out["span_id"] == collector._inference_span_id
        assert llm_out["parent_span_id"] == collector._turn_span_id
        assert llm_out["gen_ai.response.time_to_first_chunk_ns"] == 850_000_000
        assert llm_out["gen_ai.usage.input_tokens"] == 420
        assert llm_out["gen_ai.usage.output_tokens"] == 25
        assert llm_out["gen_ai.usage.reasoning.output_tokens"] == 100
        assert llm_out["finish_reason"] == "stop"

    def test_tool_call_has_span_hierarchy(self):
        collector = EvalCollector()
        collector._enabled = True
        collector._output_dir = "data/eval"

        ctx = _MockCtx()
        ctx.memory.messages = []

        async def _run():
            await collector._on_flow_start(ctx)
            collector._session_span_id = collector._session_id
            collector._turn_span_id = collector._new_span_id()
            collector._inference_span_id = collector._new_span_id()
            collector._turn_count = 0

            tool_call = _MockToolCall("call_123", "sql_exec", {"sql": "SELECT 1"})
            tool = _MockTool()
            result = _MockToolResult()

            await collector._on_tool_before(ctx, tool_call, tool)
            await collector._on_tool_after(ctx, tool_call, tool, result)

        asyncio.run(_run())

        tool_ev = [e for e in collector._events if e["type"] == "tool_call"][0]
        assert "_timestamp" in tool_ev
        assert tool_ev["span_id"] is not None
        assert tool_ev["parent_span_id"] == collector._inference_span_id
        assert tool_ev["name"] == "sql_exec"
        assert tool_ev["arguments"] == {"sql": "SELECT 1"}

    def test_new_span_id_is_unique_and_12_hex(self):
        collector = EvalCollector()
        ids = {collector._new_span_id() for _ in range(100)}
        assert len(ids) == 100
        for sid in ids:
            assert len(sid) == 12
            assert all(c in "0123456789abcdef" for c in sid)

    def test_disabled_collector_skips_events(self):
        collector = EvalCollector()
        collector._enabled = False

        ctx = _MockCtx()
        node = MagicMock()

        async def _run():
            await collector._on_flow_start(ctx)
            await collector._on_llm_before(ctx, node)

        asyncio.run(_run())
        assert collector._events == []

    def test_hybrid_token_counting_fills_null_usage(self):
        collector = EvalCollector()
        collector._enabled = True
        collector._output_dir = "data/eval"
        collector._token_counting = "hybrid"

        ctx = _MockCtx()
        ctx.memory.messages = [
            Message.system_message("You are helpful."),
            Message.user_message("What is 2+2?"),
            Message.assistant_message(content="The answer is 4."),
        ]

        node = MagicMock()
        node._eval_ttft_ns = None
        node._eval_usage_input = None   # API didn't return usage
        node._eval_usage_output = None
        node._eval_usage_reasoning = None

        async def _run():
            await collector._on_flow_start(ctx)
            collector._session_span_id = collector._session_id
            collector._turn_span_id = collector._new_span_id()
            collector._turn_count = 0
            await collector._on_llm_after(ctx, node, FinishReason.STOP)

        asyncio.run(_run())

        llm_out = [e for e in collector._events if e["type"] == "llm_output"][0]
        # tiktoken fallback should fill in estimates
        assert llm_out["gen_ai.usage.input_tokens"] > 0
        assert llm_out["gen_ai.usage.output_tokens"] > 0
        assert isinstance(llm_out["gen_ai.usage.input_tokens"], int)
        assert isinstance(llm_out["gen_ai.usage.output_tokens"], int)

    def test_api_token_counting_preserves_null(self):
        collector = EvalCollector()
        collector._enabled = True
        collector._output_dir = "data/eval"
        collector._token_counting = "api"

        ctx = _MockCtx()
        ctx.memory.messages = [
            Message.system_message("sys"),
            Message.user_message("hello"),
            Message.assistant_message(content="ok"),
        ]

        node = MagicMock()
        node._eval_usage_input = None
        node._eval_usage_output = None
        node._eval_usage_reasoning = None

        async def _run():
            await collector._on_flow_start(ctx)
            collector._session_span_id = collector._session_id
            collector._turn_span_id = collector._new_span_id()
            collector._turn_count = 0
            await collector._on_llm_after(ctx, node, FinishReason.STOP)

        asyncio.run(_run())

        llm_out = [e for e in collector._events if e["type"] == "llm_output"][0]
        assert llm_out["gen_ai.usage.input_tokens"] is None
        assert llm_out["gen_ai.usage.output_tokens"] is None
