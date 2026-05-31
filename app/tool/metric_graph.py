"""Metric lineage query tool."""
from typing import Literal

from pydantic import PrivateAttr, model_validator

from app.db.graph import GraphExecutor
from app.hook import HookPoint, hook
from app.schema import ToolCall
from app.semantics.graph.init_state import init_state
from app.tool.base import BaseTool, ToolResult


class MetricGraphTool(BaseTool):
    permission: Literal["global", "skills", "agent"] = "agent"
    name: str = "metric_lineage"
    description: str = (
        "Query the metric lineage graph (Cypher). "
        "Schema (metrics, dimensions, relationships) is in the system prompt "
        "under <metric_lineage_schema> — use ONLY those names."
    )
    strict: bool = True
    parameters: dict = {
        "type": "object", "properties": {
            "query": {"type": "string"},
        }, "required": ["query"],
    }

    _executor: GraphExecutor | None = PrivateAttr(default=None)

    def is_available(self) -> bool:
        return init_state.is_ready("metric_graph")

    @model_validator(mode="after")
    def _init(self) -> "MetricGraphTool":
        if init_state.is_ready("metric_graph"):
            self._executor = init_state.get_executor("metric_graph")
        return self

    @hook(HookPoint.NODE_INIT_BEFORE)
    def _inject_schema(self, ctx, node):
        if not node.system_prompt or self._executor is None:
            return
        text = _build_schema(self._executor)
        node.system_prompt = (
            node.system_prompt
            + "\n\n<metric_lineage_schema>\n"
            + text
            + "\n</metric_lineage_schema>"
        )

    async def execute(self, tool_call: ToolCall, **kwargs) -> ToolResult:
        query = tool_call.function.arguments_dict.get("query", "").strip()
        if not query:
            return ToolResult.failure_response(tool_call.id, self.name, "No query provided.")
        try:
            ex = self._executor or init_state.get_executor("metric_graph")
            if ex is None:
                return ToolResult.failure_response(tool_call.id, self.name, "Graph not initialized.")
            result = ex.execute(query)
            headers = result.get_column_names()
            rows = _fetch_all(result)
            return ToolResult.success_response(tool_call.id, self.name, _fmt(query, headers, rows))
        except Exception as e:
            return ToolResult.failure_response(tool_call.id, self.name, f"Query failed: {e}")


def _build_schema(ex) -> str:
    compat_note = ""
    if getattr(ex, 'graph_type', '') == 'kuzu':
        compat_note = " KuzuDB: no OPTIONAL MATCH, no collect(), no type()/labels()."

    r = ex.execute("MATCH (m:Metric) RETURN m.name, m.description ORDER BY m.name")
    metrics = _fetch_all(r)
    r = ex.execute("MATCH (d:Dimension) RETURN d.name, d.dataset, d.is_time ORDER BY d.name")
    dims = _fetch_all(r)
    r = ex.execute("MATCH ()-[e]->() RETURN DISTINCT label(e) AS rel")
    edges = _fetch_all(r)

    node_schemas = {}
    for label in ("Metric", "Dimension", "PhysicalField"):
        try:
            rows = _fetch_all(ex.execute(f"CALL table_info('{label}') RETURN name"))
            props = [str(r2[0]) for r2 in rows if str(r2[0]) not in ("id",)]
            if props:
                node_schemas[label] = props
        except Exception:
            pass

    lines = [f"Metric lineage graph. Use ONLY exact labels & properties.{compat_note}"]
    for label, props in sorted(node_schemas.items()):
        lines.append(f"  {label}: {', '.join(props)}")
    lines.append("\nMetrics:")
    for name, desc in metrics:
        lines.append(f"  {name}: {desc or ''}")
    lines.append("\nDimensions:")
    for name, ds, is_time in dims:
        t = " (time)" if is_time == "True" or is_time is True else ""
        lines.append(f"  {name} [{ds}]{t}")
    lines.append("\nRelationships:")
    for e in edges:
        lines.append(f"  {e[0]}")
    if metrics:
        m0 = metrics[0][0]
        lines.append(f"\nExample: MATCH (m:Metric {{name:'{m0}'}})-[:AGGREGATES_FROM]->(f:PhysicalField) RETURN f.name, f.dataset")
    return "\n".join(lines)


def _fetch_all(result):
    rows = []
    while result.has_next():
        rows.append(result.get_next())
    return rows


def _fmt(query, headers, rows):
    if not rows:
        return f"**No results.**\n\n```cypher\n{query}\n```"
    lines = [f"**{len(rows)} row(s)**", "", "```cypher", query, "```", ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)
