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


class TestInstallCompressionHook:
    """Regression tests for the wiring bugs found in post-landing review."""

    def test_install_registers_on_flow_context_hooks(self, monkeypatch):
        """install_compression_hook must call flow.context.hooks.register,
        not flow.hooks (which does not exist on AgentFlow)."""
        from app.memory.hook import install_compression_hook
        from app.config import config

        captured = {}

        class _FakeRegistry:
            def register(self, obj, **kw):
                captured["obj"] = obj
                return ["node.exec_async.before"]

        class _FakeContext:
            def __init__(self):
                self.hooks = _FakeRegistry()

        class _FakeFlow:
            def __init__(self):
                self.context = _FakeContext()

        monkeypatch.setattr(config, "llm", {"default": _MockLLMSettings()})
        monkeypatch.setattr(config, "agent", {"default": _MockAgentSettings()})

        flow = _FakeFlow()
        hook_obj = install_compression_hook(flow)
        assert "obj" in captured, "register must be called on flow.context.hooks"
        assert captured["obj"] is hook_obj
