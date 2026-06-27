from app.config import config
from app.hook import HookPoint, hook
from app.memory.compressor import ContextCompressor
from app.memory.token_counter import TokenCounter
from app.logger import logger
from app.schema import FinishReason


class CompressionHook:
    def __init__(self, llm_settings, agent_settings):
        self._llm_settings = llm_settings
        self._agent_settings = agent_settings
        self._disabled = False
        self._last_finish_was_stop = True

        try:
            tc = TokenCounter()
            tc.count("probe")
            self._compressor = ContextCompressor(tc)
        except Exception:
            self._disabled = True
            logger.warning("Compression disabled: tiktoken unavailable")

    @hook(HookPoint.NODE_EXEC_BEFORE, priority=50)
    async def _on_before_exec(self, ctx, node):
        if self._disabled:
            return
        cw = getattr(self._llm_settings, 'context_window', 0) or 0
        th = getattr(self._agent_settings, 'compression_threshold', 0) or 0.0
        if not cw or not th:
            return

        compressed = await self._compressor.compress(
            ctx.memory.messages,
            context_window=cw,
            threshold=th,
            keep_recent_turns=getattr(self._agent_settings, 'compression_keep_recent_turns', 3),
            is_turn_boundary=self._last_finish_was_stop,
            llm=node.llm,
            summary_system_prompt=getattr(self._agent_settings, 'compression_summary_prompt', None) or None,
        )
        if compressed is not ctx.memory.messages:
            ctx.memory.messages = compressed
            ctx.memory._token_total = self._compressor._token_counter.count(
                [m.to_dict() for m in compressed]
            )
            self._last_finish_was_stop = True
            await ctx.hooks.emit(HookPoint.CONTEXT_COMPRESSED, ctx=ctx)

    @hook(HookPoint.NODE_EXEC_AFTER, priority=50)
    async def _on_after_exec(self, ctx, node, reason):
        self._last_finish_was_stop = (reason == FinishReason.STOP)


def install_compression_hook(flow) -> CompressionHook:
    llm_cfg = config.llm.get("default")
    agent_cfg = config.agent.get("default")
    hook_obj = CompressionHook(llm_cfg, agent_cfg)
    flow.context.hooks.register(hook_obj)
    return hook_obj
