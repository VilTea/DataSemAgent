"""Tests for the hook system."""
import asyncio
import pytest

from app.hook import HookRegistry


class TestHookRegistry:
    def test_register_and_emit(self):
        registry = HookRegistry()
        results = []
        registry.on("test.point", lambda ctx: results.append(ctx["x"]))
        asyncio.run(registry.emit("test.point", ctx={"x": 42}))
        assert results == [42]

    def test_filter_by_node_name(self):
        registry = HookRegistry()
        results = []
        registry.on("point", lambda ctx, node: results.append(node.name), node_name="agent-a")
        class FakeNode:
            name = "agent-a"
        asyncio.run(registry.emit("point", ctx={}, node=FakeNode()))
        assert results == ["agent-a"]

    def test_filter_mismatch_skips(self):
        registry = HookRegistry()
        results = []
        registry.on("point", lambda ctx, node: results.append(node.name), node_name="agent-a")
        class FakeNode:
            name = "agent-b"
        asyncio.run(registry.emit("point", ctx={}, node=FakeNode()))
        assert results == []

    def test_on_error_raise(self):
        registry = HookRegistry()
        registry.on("point", lambda ctx: 1/0, on_error="raise")
        with pytest.raises(ZeroDivisionError):
            asyncio.run(registry.emit("point", ctx={}))

    def test_on_error_log(self):
        registry = HookRegistry()
        registry.on("point", lambda ctx: 1/0, on_error="log")
        asyncio.run(registry.emit("point", ctx={}))

    def test_async_callback(self):
        registry = HookRegistry()
        results = []
        async def callback(ctx):
            await asyncio.sleep(0)
            results.append(ctx["x"])
        registry.on("point", callback)
        asyncio.run(registry.emit("point", ctx={"x": 7}))
        assert results == [7]

    def test_multiple_callbacks(self):
        registry = HookRegistry()
        results = []
        registry.on("point", lambda ctx: results.append("a"))
        registry.on("point", lambda ctx: results.append("b"))
        asyncio.run(registry.emit("point", ctx={}))
        assert results == ["a", "b"]


class TestHookDecorator:
    def test_decorator_stores_config(self):
        from app.hook.decorator import hook
        @hook("tool.before", tool_name="test", on_error="raise")
        def my_hook(ctx, tool_call, tool):
            pass
        assert my_hook._hook_config == {
            "point": "tool.before", "priority": 100, "node_name": None,
            "node_type": None, "tool_name": "test", "on_error": "raise",
        }

    def test_auto_detect_tool_name(self):
        from app.tool.base import BaseTool
        from app.hook.decorator import hook
        class MyTool(BaseTool):
            name: str = 'my_tool'
            description: str = 'test'
            parameters: dict = {}
            permission: str = 'agent'
            async def execute(self, tc): pass
            @hook('tool.before')
            async def guard(self, ctx, tool_call, tool): pass
        t = MyTool()
        hooks = [h for h in getattr(t, '_pending_hooks', [])]
        assert len(hooks) == 1
        assert hooks[0]["tool_name"] == "my_tool"
