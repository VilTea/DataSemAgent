# app/semantics/graph/reasoning/flow.py
from app.flow import react_flow
from app.node.agent import AgentNode
from app.semantics.graph.reasoning.reflector import ReasoningReflector
from app.semantics.graph.reasoning.tools import (
    EmitReasoningTool, ReadReasoningTool, DeleteReasoningPathTool,
)
from app.semantics.graph.reasoning.contract import ReasoningGraphDoc
from app.semantics.graph.reasoning.builder import ReasoningGraphBuilder
from app.semantics.graph.reasoning.merger import SynonymMerger
import asyncio
from datetime import datetime
from pathlib import Path
from app.logger import logger


class _ReflectionLogger:
    """Captures the full reflection sub-flow output to a local log file."""

    def __init__(self):
        log_dir = Path("logs") / "reflection"
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._path = log_dir / f"reflect_{ts}.log"
        self._f = open(str(self._path), "w", encoding="utf-8")

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        self._f.flush()

    async def consume(self, event) -> None:
        from app.schema import AgentCompletion, Message, Role
        if isinstance(event, AgentCompletion):
            if event.reasoning_content:
                self._f.write(f"[think] {event.reasoning_content}")
            if event.content:
                self._f.write(event.content)
            if event.full_tool_calls:
                for tc in event.full_tool_calls:
                    if tc.function.name and tc.function.arguments:
                        self._f.write(f"\n\n--- TOOL: {tc.function.name} ---\n")
                        self._f.write(tc.function.arguments)
                        self._f.write("\n")
            if event.finish_reason and event.finish_reason.value not in (None, "none"):
                self._f.write(f"\n[finish: {event.finish_reason.value}]\n")
        elif isinstance(event, Message) and event.role == Role.TOOL:
            self._f.write(f"\n--- RESULT ---\n{event.content or ''}\n")
        self._f.flush()

    @property
    def path(self) -> str:
        return str(self._path)


async def _reflection_callback(ctx) -> None:
    """FLOW_END hook: trigger reflection every N turns (configurable)."""
    if ctx.turns <= 0:
        return
    from app.config import config
    interval = config.agent["default"].reflection_interval
    if interval == 0 or ctx.turns % interval != 0:
        return
    await _do_reflection(ctx.memory)


async def _do_reflection(memory) -> None:
    from app.semantics.graph.init_state import init_state
    from app.semantics.graph.loader import create_graph_loader
    loader = init_state.get_executor("reasoning_graph")
    if loader is None:
        loader = create_graph_loader("reasoning")
    doc = await run_reasoning_reflection(memory, loader)
    if doc is not None:
        init_state.set_executor("reasoning_graph", loader)
        init_state.mark_ready("reasoning_graph")


def install_reflection_hook(flow) -> None:
    """Register the reflection hook on an AgentFlow instance."""
    from app.hook import HookPoint
    flow.context.hooks.on(HookPoint.FLOW_END, _reflection_callback, priority=200)


async def run_reasoning_reflection(memory, loader) -> ReasoningGraphDoc | None:
    """Run the reasoning reflection sub-React loop. Returns the final graph doc."""
    log = _ReflectionLogger()
    print(f"\n  [reflection] log: {log.path}")
    try:
        emit_tool = EmitReasoningTool()
        read_tool = ReadReasoningTool(emit_tool)
        delete_tool = DeleteReasoningPathTool(emit_tool)

        existing_summary = _get_existing_summary(loader)
        prompt = ReasoningReflector.build_prompt(memory, existing_summary)

        # Write the full prompt to the log
        log._f.write(f"=== REFLECTION PROMPT ({len(prompt)} chars) ===\n{prompt}\n\n=== LLM OUTPUT ===\n\n")
        print(f"  [reflection] prompt: {len(prompt)} chars, existing: {existing_summary[:60]}")

        from app.pipeline import QueuePipeline
        pipeline = QueuePipeline()
        pipeline.register(log)

        agent = AgentNode(
            name="reasoning-reflector",
            system_prompt=prompt,
            tools=[emit_tool, read_tool, delete_tool],
        )
        flow = react_flow(agent_node=agent, pipeline=pipeline)
        await pipeline.start()
        await flow._run_async(flow.context.get_shared())
        await pipeline.stop()

        doc = emit_tool.accumulated
        nf = len(doc.facts) if doc else 0
        ns = len(doc.steps) if doc else 0
        ne = len(doc.edges) if doc else 0

        # Write final summary
        log._f.write(f"\n=== RESULT ===\n{f'facts: {nf}, steps: {ns}, edges: {ne}'}\n")
        print(f"  [reflection] done: {nf}F / {ns}S / {ne}E")

        if not doc.facts and not doc.steps:
            return None

        merger = SynonymMerger()
        merged = merger.merge(doc)
        if merged:
            log._f.write(f"merged: {merged} facts\n")

        builder = ReasoningGraphBuilder()
        gdoc = builder.build(doc)
        loader.load(gdoc)
        return doc

    except Exception as e:
        logger.error(f"Reasoning reflection failed: {e}", exc_info=True)
        log._f.write(f"\n=== ERROR ===\n{e}\n")
        return None
    finally:
        log._f.close()


def _get_existing_summary(loader) -> str:
    try:
        r = loader.execute("MATCH (f:Fact) RETURN f.content LIMIT 20")
        facts = []
        while r.has_next():
            row = r.get_next()
            facts.append(str(row[0])[:200])
        if facts:
            return "Existing facts:\n" + "\n".join(f"- {f}" for f in facts)
    except Exception:
        pass
    return "(no existing reasoning graph)"
