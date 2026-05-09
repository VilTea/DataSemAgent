"""Entity graph flow nodes — agent nodes, validators, and compiler wrapper."""
from __future__ import annotations

import json
from typing import Any

from pocketflow import AsyncNode

from app.logger import logger
from app.node.agent import AgentNode
from app.node.base import AgentContext
from app.pipeline.abc import Consumable
from app.schema import FinishReason, Message, ToolCall, TOOL_CHOICE_TYPE, ToolChoice
from app.semantics.graph.entity.compiler import GraphCompiler
from app.semantics.graph.entity.mapping import DataMapping
from app.semantics.graph.entity.sampler import DataSampler
from app.semantics.graph.entity.schema import EntityGraphSchema
from app.semantics.graph.entity.validator import MappingValidator
from app.semantics.graph.init_state import init_state
from app.semantics.models import SemanticModel
from app.tool.base import BaseTool, ToolResult


# ── Shared context keys ──
K_MODEL = "entity_model"
K_SAMPLES = "entity_samples"
K_EXECUTOR = "entity_executor"
K_SCHEMA = "entity_schema"
K_MAPPING = "entity_mapping"
K_VALIDATION_ERRORS = "entity_validation_errors"
K_VALIDATION_RETRIES = "entity_validation_retries"
K_PIPELINE = "entity_pipeline"

_MAX_RETRIES = 3
_STAGE = "entity_graph_init"


# ── Structured output tools ──

class EmitEntitySchemaTool(BaseTool):
    permission: str = "agent"
    name: str = "emit_entity_schema"
    description: str = "Emit the entity graph schema. Call with entities and relations arrays."
    strict: bool = True
    parameters: dict = {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "description": {"type": "string"},
                        "strong_parents": {"type": "array", "items": {"type": "string"}},
                        "is_weak": {"type": "boolean"},
                        "is_event": {"type": "boolean"},
                    },
                    "required": ["label", "description", "is_weak", "is_event"],
                },
            },
            "relations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "from": {"type": "string"},
                        "to": {"type": "string"},
                        "role": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["label", "from", "to"],
                },
            },
        },
        "required": ["entities", "relations"],
    }

    async def execute(self, tool_call: ToolCall, **kwargs) -> ToolResult:
        return ToolResult.success_response(tool_call.id, self.name, "Schema captured.")


class EmitDataMappingTool(BaseTool):
    permission: str = "agent"
    name: str = "emit_data_mapping"
    description: str = (
        "Emit the data mapping. Each entity needs: entity (matching a schema label), "
        "node_source (type: table|union|join with table/key_columns), properties (dict). "
        "Each edge needs: label, from (entity+key_column), to (entity+key_column, "
        "must be a column in the FROM entity's source table)."
    )
    strict: bool = False
    parameters: dict = {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "entity": {"type": "string", "description": "Entity label from schema"},
                        "node_source": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string", "enum": ["table", "union", "join"]},
                                "table": {"type": "string"},
                                "key_columns": {
                                    "type": "array", "items": {"type": "string"},
                                    "description": "Column(s) for the entity key. Use multiple for composite PKs.",
                                },
                            },
                            "required": ["type", "table", "key_columns"],
                        },
                        "properties": {
                            "type": "object",
                            "description": "Property name → column reference in the source table. Every entity should include at least the columns visible in the sample data.",
                        },
                        "strong_parents": {
                            "type": "object",
                            "description": "Parent entity label → FK column in this entity's source table. For weak entities that depend on multiple parents, include one entry per parent."
                        },
                    },
                    "required": ["entity", "node_source", "properties"],
                },
            },
            "edges": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "from": {
                            "type": "object",
                            "properties": {"entity": {"type": "string"}, "key_column": {"type": "string"}},
                            "required": ["entity", "key_column"],
                        },
                        "to": {
                            "type": "object",
                            "properties": {
                                "entity": {"type": "string"},
                                "key_column": {
                                    "type": "string",
                                    "description": "FK column in the FROM entity's source table (e.g. ss_customer_sk in store_sales)",
                                },
                            },
                            "required": ["entity", "key_column"],
                        },
                    },
                    "required": ["label", "from", "to"],
                },
            },
        },
        "required": ["entities", "edges"],
    }

    async def execute(self, tool_call: ToolCall, **kwargs) -> ToolResult:
        return ToolResult.success_response(tool_call.id, self.name, "Mapping captured.")


# ── Helpers ──

def _abort_on_max(shared: dict, node: str, fallback: str) -> str:
    retries = shared.get(K_VALIDATION_RETRIES, 0) + 1
    shared[K_VALIDATION_RETRIES] = retries
    if retries >= _MAX_RETRIES:
        logger.error(f"{node}: max retries ({_MAX_RETRIES}) exceeded, aborting")
        return "abort"
    return fallback


def _inject_memory_error(memory, tool_name: str, errors: list[str]) -> None:
    error_text = "Validation failed. Fix these issues and try again:\n" + "\n".join(f"- {e}" for e in errors)
    for msg in reversed(memory.messages):
        if msg.role == "assistant" and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.function.name == tool_name:
                    memory.add_message(Message.tool_message(
                        content=error_text, tool_call_id=tc.id, name=tool_name,
                    ))
                    return


def _deep_merge(base: dict, other: dict) -> dict:
    """Recursively merge *other* into *base*. Lists are concatenated."""
    for key, value in other.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        elif key in base and isinstance(base[key], list) and isinstance(value, list):
            base[key].extend(value)
        else:
            base[key] = value
    return base


def _extract_tool_args(context: AgentContext, tool_name: str) -> tuple[dict | None, str | None]:
    if not context.memory.messages:
        return None, "No messages in agent context"
    last_msg = context.memory.messages[-1]
    if not last_msg.tool_calls:
        return None, f"LLM did not call any tool. Response: {(last_msg.content or '')[:200]}"
    for tc in last_msg.tool_calls:
        if tc.function.name == tool_name:
            raw = tc.function.arguments or ""
            try:
                return json.loads(raw), None
            except json.JSONDecodeError:
                pass
            # LLM may emit multiple concatenated JSON objects, e.g.:
            #   {"entities": [...]}{"edges": [...]}
            # Parse them all and deeply merge into a single dict.
            # If the last object is truncated, salvage what we already parsed.
            try:
                decoder = json.JSONDecoder()
                merged: dict = {}
                idx = 0
                while idx < len(raw):
                    raw_slice = raw[idx:].strip()
                    if not raw_slice:
                        break
                    try:
                        obj, end = decoder.raw_decode(raw_slice)
                        merged = _deep_merge(merged, obj)
                        idx += end
                    except json.JSONDecodeError:
                        if merged:
                            logger.warning(
                                f"Salvaged partial {tool_name} args ({len(merged)} keys), "
                                f"dropping {len(raw) - idx} trailing chars"
                            )
                            return merged, None
                        raise  # nothing parsed yet → genuine failure
                if merged:
                    logger.info(f"Merged {len(raw)} chars of {tool_name} args into unified dict")
                    return merged, None
            except json.JSONDecodeError:
                pass
            snippet = raw[:300] + ("..." if len(raw) > 300 else "")
            err = f"Failed to parse {tool_name} arguments. Preview: {snippet}"
            logger.warning(err)
            return None, err
    return None, f"LLM called {[tc.function.name for tc in last_msg.tool_calls]}, but not {tool_name}"


def _persist_schema(loader, schema) -> None:
    """Persist entity graph schema alongside the Kùzu database."""
    import json
    from pathlib import Path
    if schema is None:
        return
    path = getattr(loader, "_path", None)
    if not path or path == ":memory:":
        return
    schema_file = Path(str(path) + ".schema.json")
    try:
        if hasattr(schema, "model_dump"):
            data = schema.model_dump()
        else:
            data = schema
        schema_file.write_text(json.dumps(data, ensure_ascii=False, default=str), "utf-8")
    except Exception:
        pass


def load_entity_schema(loader) -> object | None:
    """Load persisted entity graph schema from disk. Returns None if not found."""
    import json
    from pathlib import Path
    from app.semantics.graph.entity.schema import EntityGraphSchema
    path = getattr(loader, "_path", None)
    if not path or path == ":memory:":
        return None
    schema_file = Path(str(path) + ".schema.json")
    if not schema_file.exists():
        return None
    try:
        data = json.loads(schema_file.read_text("utf-8"))
        return EntityGraphSchema(**data)
    except Exception:
        return None


async def _emit(shared: dict[str, Any], event: dict[str, Any]) -> None:
    pipeline: Consumable | None = shared.get(K_PIPELINE)
    if pipeline is not None:
        await pipeline.emit(event)


# ── Prompt builders ──

def _build_schema_prompt(model: SemanticModel, samples: dict) -> str:
    sample_text = DataSampler.format_for_prompt(samples)
    ds_info = "\n".join(
        f"  {ds.name} (source: {ds.source}, pk: {ds.primary_key})"
        for ds in model.datasets
    )
    return (
        "Analyze this database and design an entity-relationship graph schema.\n"
        "Call emit_entity_schema with your result. Do NOT output raw JSON.\n\n"
        f"## Datasets\n{ds_info}\n\n"
        f"## Relationships\n"
        + "\n".join(
            f"  {r.from_dataset}.{r.from_columns} -> {r.to_dataset}.{r.to_columns}"
            for r in (model.relationships or [])
        )
        + "\n\n"
        "IMPORTANT: Edge labels must be DISTINCT from entity labels.\n"
        "If you have an entity 'customer', do NOT name an edge 'customer' —\n"
        "use descriptive names like 'purchased_by', 'sold_on', 'lives_in'.\n"
        + f"\n\n{sample_text}"
    )


def _build_mapping_prompt(model: SemanticModel, schema: EntityGraphSchema, samples: dict) -> str:
    sample_text = DataSampler.format_for_prompt(samples)
    schema_json = schema.model_dump_json(indent=2)
    return (
        "Given this entity graph schema, produce a data mapping.\n"
        "Call emit_data_mapping with your result. Do NOT output raw YAML.\n\n"
        "For each entity, include a 'properties' dict mapping logical property names\n"
        "to column references in the source table. Include all meaningful columns from\n"
        "the sample data (e.g. {\"name\": \"c_first_name\", \"email\": \"c_email_address\"}).\n"
        "At minimum include the descriptive columns visible in the samples.\n\n"
        "CRITICAL for edges: edges read rows from the FROM entity's source table.\n"
        "Both from.key_column and to.key_column are columns IN THAT SOURCE TABLE.\n"
        "to.key_column must be the FK column in the source table (e.g. ss_customer_sk,\n"
        "NOT c_customer_sk which is the target entity's key).\n\n"
        f"## Graph Schema\n{schema_json}\n\n"
        f"{sample_text}"
    )


# ── Nodes ──

class _EntityFlowNode(AsyncNode):
    async def prep_async(self, shared: dict[str, Any]) -> dict[str, Any]:
        self._shared = shared
        return shared

    async def post_async(self, shared, prep_res, exec_res):
        return exec_res

    async def exec_fallback_async(self, prep_res, exc):
        logger.error(f"{type(self).__name__} crashed: {exc}", exc_info=True)
        return "error"


class SamplerNode(_EntityFlowNode):
    async def exec_async(self, prep_res: dict[str, Any]) -> str:
        shared = prep_res
        model: SemanticModel = shared[K_MODEL]
        executor = shared[K_EXECUTOR]
        table_names = [ds.source for ds in model.datasets]

        await _emit(shared, {"stage": _STAGE, "step": "sampler", "status": "running",
                             "tables": table_names})
        sampler = DataSampler(executor, sample_size=3)
        samples = await sampler.sample(table_names)
        shared[K_SAMPLES] = samples
        await _emit(shared, {"stage": _STAGE, "step": "sampler", "status": "done",
                             "result": {t: {"cols": len(d["columns"]), "rows": len(d["rows"])}
                                       for t, d in samples.items()}})
        return "default"


class SchemaAgentNode(AgentNode):
    name: str = "entity-schema-agent"
    description: str = "Infers entity graph schema from OSI model and sample data"
    system_prompt: str = ""
    tools: list = [EmitEntitySchemaTool()]

    async def exec_fallback_async(self, prep_res, exc):
        logger.error(f"SchemaAgentNode crashed: {exc}", exc_info=True)
        await _emit(self._shared, {"stage": _STAGE, "step": "schema_agent", "status": "error",
                                   "error": f"Node crashed: {exc}"})
        return _abort_on_max(self._shared, "schema_agent", "error")

    async def prep_async(self, shared: dict[str, Any]) -> AgentContext:
        self._shared = shared
        model: SemanticModel = shared[K_MODEL]
        samples: dict = shared.get(K_SAMPLES, {})
        prev_errors: list[str] = shared.get(K_VALIDATION_ERRORS, [])
        self.system_prompt = _build_schema_prompt(model, samples)

        if not hasattr(self, '_own_memory'):
            from app.schema import Memory
            self._own_memory = Memory()

        if prev_errors:
            _inject_memory_error(self._own_memory, "emit_entity_schema", prev_errors)

        await _emit(shared, {"stage": _STAGE, "step": "schema_agent", "status": "running",
                             "retry": bool(prev_errors), "prev_errors": prev_errors})

        ctx = await super().prep_async(shared)
        # Copy the system prompt that AgentNode added to the new context's memory
        for msg in ctx.memory.messages:
            if msg.role == "system":
                self._own_memory.upsert_message(msg, 0)
        object.__setattr__(ctx, '_memory', self._own_memory)
        return ctx

    async def post_async(self, shared, context, exec_res) -> str:
        schema_data, extract_err = _extract_tool_args(context, "emit_entity_schema")
        if schema_data:
            try:
                schema = EntityGraphSchema.model_validate(schema_data)
                shared[K_SCHEMA] = schema
                await _emit(shared, {"stage": _STAGE, "step": "schema_agent", "status": "done",
                                     "result": schema.model_dump(by_alias=True)})
                return "ok"
            except Exception as e:
                logger.error(f"Schema validation failed: {e}", exc_info=True)
                shared[K_VALIDATION_ERRORS] = [f"Schema validation: {e}"]
                await _emit(shared, {"stage": _STAGE, "step": "schema_agent", "status": "error",
                                     "error": str(e)})
                return _abort_on_max(shared, "schema_agent", "error")

        err = extract_err or "No structured output from LLM"
        logger.error(err)
        await _emit(shared, {"stage": _STAGE, "step": "schema_agent", "status": "error", "error": err})
        return _abort_on_max(shared, "schema_agent", "error")


class MappingAgentNode(AgentNode):
    name: str = "entity-mapping-agent"
    description: str = "Produces data mapping from entity graph schema and database"
    system_prompt: str = ""
    tools: list = [EmitDataMappingTool()]

    async def exec_fallback_async(self, prep_res, exc):
        logger.error(f"MappingAgentNode crashed: {exc}", exc_info=True)
        await _emit(self._shared, {"stage": _STAGE, "step": "mapping_agent", "status": "error",
                                   "error": f"Node crashed: {exc}"})
        return _abort_on_max(self._shared, "mapping_agent", "error")

    async def prep_async(self, shared: dict[str, Any]) -> AgentContext:
        self._shared = shared
        model: SemanticModel = shared[K_MODEL]
        schema: EntityGraphSchema | None = shared.get(K_SCHEMA)
        samples: dict = shared.get(K_SAMPLES, {})
        prev_errors: list[str] = shared.get(K_VALIDATION_ERRORS, [])

        if schema is None:
            self.system_prompt = "Error: no schema available"
        else:
            self.system_prompt = _build_mapping_prompt(model, schema, samples)

        if not hasattr(self, '_own_memory'):
            from app.schema import Memory
            self._own_memory = Memory()

        if prev_errors:
            _inject_memory_error(self._own_memory, "emit_data_mapping", prev_errors)

        await _emit(shared, {"stage": _STAGE, "step": "mapping_agent", "status": "running",
                             "retry": bool(prev_errors), "prev_errors": prev_errors})

        ctx = await super().prep_async(shared)
        for msg in ctx.memory.messages:
            if msg.role == "system":
                self._own_memory.upsert_message(msg, 0)
        object.__setattr__(ctx, '_memory', self._own_memory)
        return ctx

    async def post_async(self, shared, context, exec_res) -> str:
        mapping_data, extract_err = _extract_tool_args(context, "emit_data_mapping")
        if mapping_data:
            try:
                mapping = DataMapping.model_validate(mapping_data)
                shared[K_MAPPING] = mapping
                shared[K_VALIDATION_ERRORS] = []
                await _emit(shared, {"stage": _STAGE, "step": "mapping_agent", "status": "done",
                                     "result": {
                                         "entities": [
                                             {"name": e.entity, "table": _ns_table(e.node_source),
                                              "key_columns": _ns_key_cols(e.node_source),
                                              "properties": e.properties}
                                             for e in mapping.entities
                                         ],
                                         "edges": [
                                             {"label": e.label, "from": e.from_.entity, "to": e.to.entity,
                                              "table": _edge_table(mapping, e),
                                              "fk_column": e.to.key_column}
                                             for e in mapping.edges
                                         ]}})
                return "ok"
            except Exception as e:
                logger.error(f"Mapping validation failed: {e}", exc_info=True)
                shared[K_VALIDATION_ERRORS] = [str(e)]
                await _emit(shared, {"stage": _STAGE, "step": "mapping_agent", "status": "error",
                                     "error": str(e)})
                return _abort_on_max(shared, "mapping_agent", "error")

        err = extract_err or "No structured output from LLM"
        logger.error(err)
        # Dump full memory for diagnostics
        for i, msg in enumerate(context.memory.messages[-4:]):
            tc_names = [tc.function.name for tc in (msg.tool_calls or [])]
            logger.error(
                f"  memory[{i - len(context.memory.messages)}] "
                f"role={msg.role} content={msg.content!r:.200} "
                f"tool_calls={tc_names}"
            )
        logger.error(f"  finish_reason={exec_res}")
        await _emit(shared, {"stage": _STAGE, "step": "mapping_agent", "status": "error", "error": err})
        return _abort_on_max(shared, "mapping_agent", "error")


class SchemaValidatorNode(_EntityFlowNode):
    async def exec_async(self, prep_res: dict[str, Any]) -> str:
        shared = prep_res
        schema: EntityGraphSchema | None = shared.get(K_SCHEMA)
        await _emit(shared, {"stage": _STAGE, "step": "schema_validator", "status": "running"})
        if schema is None:
            return "abort"
        errors = _validate_schema(schema)
        shared[K_VALIDATION_ERRORS] = errors
        if errors:
            retries = shared.get(K_VALIDATION_RETRIES, 0) + 1
            shared[K_VALIDATION_RETRIES] = retries
            await _emit(shared, {"stage": _STAGE, "step": "schema_validator", "status": "done",
                                 "errors": errors, "retries": retries, "max": _MAX_RETRIES})
            if retries >= _MAX_RETRIES:
                return "abort"
            return "retry_schema"
        await _emit(shared, {"stage": _STAGE, "step": "schema_validator", "status": "done", "errors": []})
        return "pass"


def _validate_schema(schema: EntityGraphSchema) -> list[str]:
    errors: list[str] = []
    labels = {e.label for e in schema.entities}
    for entity in schema.entities:
        if not entity.label.replace("_", "").isalnum() or " " in entity.label:
            errors.append(f"Entity label '{entity.label}' is invalid for Kùzu. Use only letters, numbers, underscores.")
        if entity.is_weak:
            if not entity.strong_parents:
                errors.append(f"Weak entity '{entity.label}' has no strong_parents set")
            else:
                for parent in entity.strong_parents:
                    if parent not in labels:
                        errors.append(f"Weak entity '{entity.label}' references unknown parent '{parent}'")
    for rel in schema.relations:
        if rel.from_ not in labels:
            errors.append(f"Relation '{rel.label}': source '{rel.from_}' not in entities")
        if rel.to not in labels:
            errors.append(f"Relation '{rel.label}': target '{rel.to}' not in entities")
        if rel.label in labels:
            errors.append(f"Edge label '{rel.label}' conflicts with entity label of the same name.")
        if " " in rel.label or not rel.label.replace("_", "").isalnum():
            errors.append(f"Relation label '{rel.label}' is invalid for Kùzu.")
    if len(labels) != len(schema.entities):
        errors.append("Duplicate entity labels detected")
    return errors


class ValidatorNode(_EntityFlowNode):
    async def exec_fallback_async(self, prep_res, exc):
        logger.error(f"ValidatorNode crashed: {exc}", exc_info=True)
        err = str(exc)
        self._shared[K_VALIDATION_ERRORS] = [f"Validator crash: {err}"]
        await _emit(self._shared, {"stage": _STAGE, "step": "validator", "status": "error", "error": err})
        retries = self._shared.get(K_VALIDATION_RETRIES, 0) + 1
        self._shared[K_VALIDATION_RETRIES] = retries
        if retries >= _MAX_RETRIES:
            return "abort"
        return "retry_mapping"

    async def exec_async(self, prep_res: dict[str, Any]) -> str:
        shared = prep_res
        model: SemanticModel = shared[K_MODEL]
        schema = shared.get(K_SCHEMA)
        mapping = shared.get(K_MAPPING)
        await _emit(shared, {"stage": _STAGE, "step": "validator", "status": "running"})
        if schema is None or mapping is None:
            return "abort"
        try:
            errors = MappingValidator(schema, mapping, model).validate()
        except Exception as e:
            logger.error(f"Validator crashed: {e}", exc_info=True)
            errors = [f"Validator crashed: {e}"]
        shared[K_VALIDATION_ERRORS] = errors
        if errors:
            retries = shared.get(K_VALIDATION_RETRIES, 0) + 1
            shared[K_VALIDATION_RETRIES] = retries
            await _emit(shared, {"stage": _STAGE, "step": "validator", "status": "done",
                                 "errors": errors, "retries": retries, "max": _MAX_RETRIES})
            if retries >= _MAX_RETRIES:
                return "abort"
            if any("strong_parents" in e or "parent" in e or "weak" in e.lower() for e in errors):
                return "retry_schema"
            return "retry_mapping"
        await _emit(shared, {"stage": _STAGE, "step": "validator", "status": "done", "errors": []})
        return "pass"


class CompilerNode(_EntityFlowNode):
    async def exec_fallback_async(self, prep_res, exc):
        logger.error(f"CompilerNode crashed: {exc}", exc_info=True)
        err = str(exc)
        self._shared[K_VALIDATION_ERRORS] = [f"Compiler error: {err}"]
        await _emit(self._shared, {"stage": _STAGE, "step": "compiler", "status": "error", "error": err})
        self._shared[K_VALIDATION_RETRIES] = self._shared.get(K_VALIDATION_RETRIES, 0) + 1
        if self._shared[K_VALIDATION_RETRIES] >= _MAX_RETRIES:
            return "abort"
        return "retry_mapping"

    async def exec_async(self, prep_res: dict[str, Any]) -> str:
        shared = prep_res
        mapping: DataMapping = shared[K_MAPPING]
        executor = shared[K_EXECUTOR]
        loader = shared.get("entity_loader")
        await _emit(shared, {"stage": _STAGE, "step": "compiler", "status": "running"})
        compiler = GraphCompiler(mapping, executor)
        doc = await compiler.build()
        if loader is not None:
            loader.load(doc)
        schema = shared.get(K_SCHEMA)
        init_state.set_extra("entity_graph", "schema", schema)
        _persist_schema(loader, schema)
        init_state.mark_ready("entity_graph")
        await _emit(shared, {"stage": _STAGE, "step": "compiler", "status": "done"})
        return "default"


# ── Mapping helpers for events ──

def _ns_table(ns) -> str:
    if hasattr(ns, "table"): return ns.table
    if hasattr(ns, "base_table"): return ns.base_table
    if hasattr(ns, "sources") and ns.sources: return ns.sources[0].table
    return "?"

def _ns_key_cols(ns) -> list[str]:
    if hasattr(ns, "get_key_columns"): return ns.get_key_columns()
    return []

def _edge_table(mapping, edge) -> str:
    for e in mapping.entities:
        if e.entity == edge.from_.entity:
            return _ns_table(e.node_source)
    return "?"
