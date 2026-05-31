"""Tests for HookRegistry whitelist and unregister."""
from app.hook.registry import HookRegistry, HookPoint, OBSERVE_HOOK_POINTS
from app.hook.decorator import hook


class ConsumerWithValidHooks:
    @hook(HookPoint.TOOL_AFTER)
    async def _on_tool(self, ctx, tool_call, tool, result):
        pass


class ConsumerWithForbiddenHook:
    @hook(HookPoint.NODE_INIT_BEFORE)
    async def _on_init(self, ctx, node):
        pass


class ConsumerWithMixedHooks:
    @hook(HookPoint.TOOL_AFTER)
    async def _on_tool(self, ctx, tool_call, tool, result):
        pass

    @hook(HookPoint.NODE_INIT_BEFORE)
    async def _on_init(self, ctx, node):
        pass


class TestHookRegistryWhitelist:
    def test_valid_hook_registered(self):
        reg = HookRegistry()
        consumer = ConsumerWithValidHooks()
        registered = reg.register(consumer, whitelist=OBSERVE_HOOK_POINTS)
        assert HookPoint.TOOL_AFTER in registered

    def test_forbidden_hook_skipped(self):
        reg = HookRegistry()
        consumer = ConsumerWithForbiddenHook()
        registered = reg.register(consumer, whitelist=OBSERVE_HOOK_POINTS)
        assert len(registered) == 0

    def test_mixed_hooks_filtered(self):
        reg = HookRegistry()
        consumer = ConsumerWithMixedHooks()
        registered = reg.register(consumer, whitelist=OBSERVE_HOOK_POINTS)
        assert HookPoint.TOOL_AFTER in registered
        assert HookPoint.NODE_INIT_BEFORE not in registered

    def test_no_whitelist_registers_all(self):
        reg = HookRegistry()
        consumer = ConsumerWithForbiddenHook()
        registered = reg.register(consumer)
        assert HookPoint.NODE_INIT_BEFORE in registered


class TestHookRegistryUnregister:
    def test_unregister_removes_hooks(self):
        reg = HookRegistry()
        consumer = ConsumerWithValidHooks()
        reg.register(consumer, whitelist=OBSERVE_HOOK_POINTS)
        assert len(reg._hooks) == 1
        reg.unregister(consumer)
        assert len(reg._hooks) == 0

    def test_unregister_non_registered_is_safe(self):
        reg = HookRegistry()
        consumer = ConsumerWithValidHooks()
        reg.unregister(consumer)  # should not raise
