"""Entity graph build flow — PocketFlow topology + init + console progress."""
from __future__ import annotations

import asyncio
from typing import Any

from pocketflow import AsyncNode, AsyncFlow

from app.pipeline.abc import EventConsumer
from app.semantics.graph.entity.nodes import (
    K_EXECUTOR,
    K_MODEL,
    K_PIPELINE,
    K_VALIDATION_RETRIES,
    _MAX_RETRIES,
    _STAGE,
    CompilerNode,
    SamplerNode,
    SchemaAgentNode,
    SchemaValidatorNode,
    ValidatorNode,
)
from app.semantics.models import SemanticModel


class CliProgressConsumer(EventConsumer):
    """Prints entity graph build progress to stdout."""

    async def start(self) -> None: pass
    async def stop(self) -> None: pass

    _step_labels = {
        "sampler": "Sampler",
        "schema_agent": "Schema agent",
        "schema_validator": "Schema validator",
        "mapping_agent": "Mapping agent",
        "validator": "Mapping validator",
        "compiler": "Compiler",
        "metric_graph": "Metric graph",
    }

    async def consume(self, event: Any) -> None:
        if not isinstance(event, dict) or event.get("stage") != _STAGE:
            return
        step = event.get("step", "")
        status = event.get("status", "")
        label = self._step_labels.get(step, step)
        error = event.get("error", "")

        if status == "error":
            print(f"  [{label}] ERROR: {error}")
        elif status == "running":
            retry = " (retry)" if event.get("retry") else ""
            print(f"  [{label}] running{retry}...")
        elif status == "done":
            if step == "sampler":
                for t, info in event.get("result", {}).items():
                    print(f"  {t}: {info['cols']} cols x {info['rows']} rows sampled")
            elif step == "schema_agent":
                schema = event.get("result", {})
                print("  Entities:")
                for e in schema.get("entities", []):
                    tags = ""
                    if e.get("is_weak"): tags += f" [weak→{', '.join(e.get('strong_parents', []))}]"
                    if e.get("is_event"): tags += " [event]"
                    print(f"    {e['label']}{tags}: {e.get('description', '')}")
                print("  Relations:")
                for r in schema.get("relations", []):
                    src = r.get("from", r.get("from_", "?"))
                    print(f"    {src} -[{r['label']}]-> {r['to']}")
            elif step == "schema_validator":
                errors = event.get("errors", [])
                retries = event.get("retries", 0)
                if errors:
                    for e in errors:
                        print(f"  [R] {e}")
                    if retries >= event.get("max", _MAX_RETRIES):
                        print(f"  [ABORT] Max retries ({retries}) exceeded")
                    else:
                        print(f"  Retry {retries}/{event.get('max', _MAX_RETRIES)}")
                else:
                    print("  0 errors")
            elif step == "mapping_agent":
                result = event.get("result", {})
                print("  Entities:")
                for e in result.get("entities", []):
                    if isinstance(e, dict):
                        props = e.get('properties', {})
                        prop_str = f" props={props}" if props else " (no properties!)"
                        print(f"    {e['name']} ← {e['table']}.{e['key_columns']}{prop_str}")
                    else:
                        print(f"    {e}")
                print("  Edges:")
                for e in result.get("edges", []):
                    if isinstance(e, dict):
                        print(f"    {e['label']}: {e['from']} → {e['to']} (table={e['table']}, fk={e['fk_column']})")
                    else:
                        print(f"    {e}")
            elif step == "validator":
                errors = event.get("errors", [])
                retries = event.get("retries", 0)
                if errors:
                    for e in errors:
                        print(f"  [R] {e}")
                    if retries >= event.get("max", _MAX_RETRIES):
                        print(f"  [ABORT] Max retries ({retries}) exceeded")
                    else:
                        print(f"  Retry {retries}/{event.get('max', _MAX_RETRIES)}")
                else:
                    print("  0 errors")
            elif step == "compiler":
                print("  Graph built successfully")


def build_entity_flow() -> AsyncFlow:
    from app.semantics.graph.entity.nodes import MappingFlowNode

    sampler = SamplerNode(max_retries=1)
    schema_agent = SchemaAgentNode()
    schema_validator = SchemaValidatorNode(max_retries=1)
    mapping_agent = MappingFlowNode(max_retries=3)
    validator = ValidatorNode(max_retries=1)
    compiler = CompilerNode(max_retries=1)
    abort = AsyncNode()

    sampler >> schema_agent
    schema_agent - "ok" >> schema_validator
    schema_agent - "error" >> schema_agent
    schema_agent - "retry_schema" >> schema_agent
    schema_validator - "pass" >> mapping_agent
    schema_validator - "retry_schema" >> schema_agent
    schema_validator - "abort" >> abort
    mapping_agent - "ok" >> validator
    mapping_agent - "error" >> mapping_agent
    mapping_agent - "abort" >> abort
    mapping_agent - "retry_mapping" >> mapping_agent
    validator - "retry" >> mapping_agent
    validator - "retry_schema" >> schema_agent
    validator - "retry_mapping" >> mapping_agent
    validator - "pass" >> compiler
    validator - "abort" >> abort
    compiler - "retry_mapping" >> mapping_agent
    compiler - "default" >> abort
    compiler - "abort" >> abort
    schema_agent - "abort" >> abort

    return AsyncFlow(start=sampler)


async def init_entity_graph(
    model: SemanticModel,
    executor,
    loader=None,
    consumers: list[EventConsumer] | None = None,
) -> None:
    from app.pipeline import QueuePipeline

    if loader is None:
        from app.semantics.graph.loader import create_graph_loader
        loader = create_graph_loader("entity")

    from app.semantics.graph.init_state import init_state
    init_state.set_executor("entity_graph", loader)

    pipeline = QueuePipeline()
    for c in (consumers or []):
        pipeline.register(c)

    await pipeline.start()
    try:
        shared: dict[str, Any] = {
            K_MODEL: model,
            K_EXECUTOR: executor,
            "entity_loader": loader,
            K_PIPELINE: pipeline,
            K_VALIDATION_RETRIES: 0,
        }
        flow = build_entity_flow()
        await flow._run_async(shared)
        await asyncio.sleep(0.2)
    finally:
        await pipeline.stop()
