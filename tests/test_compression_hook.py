import asyncio
import pytest
from unittest.mock import MagicMock
from app.memory.hook import CompressionHook
from app.schema import FinishReason


class _MockLLMSettings:
    context_window = 131072


class _MockAgentSettings:
    compression_threshold = 0.8
    compression_keep_recent_turns = 3
    compression_summary_prompt = ""


class TestCompressionHook:
    def test_constructor_probes_tiktoken(self):
        hook = CompressionHook(_MockLLMSettings(), _MockAgentSettings())
        assert not hook._disabled

    def test_last_finish_stop_updated(self):
        hook = CompressionHook(_MockLLMSettings(), _MockAgentSettings())
        assert hook._last_finish_was_stop is True

        async def _run():
            await hook._on_after_exec(None, None, FinishReason.STOP)
        asyncio.run(_run())
        assert hook._last_finish_was_stop is True

        async def _run2():
            await hook._on_after_exec(None, None, FinishReason.TOOL_CALLS)
        asyncio.run(_run2())
        assert hook._last_finish_was_stop is False
