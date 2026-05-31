"""Entity graph query tool."""
from typing import Literal

from pydantic import PrivateAttr, model_validator

from app.db.graph import GraphExecutor
from app.hook import HookPoint, hook
from app.schema import ToolCall
from app.semantics.graph.init_state import init_state
from app.tool.base import BaseTool, ToolResult


class EntityGraphTool(BaseTool):
    permission: Literal["global", "skills", "agent"] = "agent"
    name: str = "entity_graph"
    description: str = (
        "Query the business entity graph (Cypher). "
        "Schema (exact labels, properties, relationships) is in the system "
        "prompt under <entity_graph_schema> — use ONLY those names."
    )
    strict: bool = True
    parameters: dict = {
        "type": "object", "properties": {
            "query": {"type": "string"},
        }, "required": ["query"],
    }

    _executor: GraphExecutor | None = PrivateAttr(default=None)

    def is_available(self) -> bool:
        return init_state.is_ready("entity_graph")

    @model_validator(mode="after")
    def _init(self) -> "EntityGraphTool":
        if init_state.is_ready("entity_graph"):
            self._executor = init_state.get_executor("entity_graph")
        return self

    @hook(HookPoint.NODE_INIT_BEFORE)
    def _inject_schema(self, ctx, node):
        if not node.system_prompt or self._executor is None:
            return
        text = _build_schema(self._executor)
        node.system_prompt = (
            node.system_prompt
            + "\n\n<entity_graph_schema>\n"
            + text
            + "\n</entity_graph_schema>"
        )

    async def execute(self, tool_call: ToolCall, **kwargs) -> ToolResult:
        query = tool_call.function.arguments_dict.get("query", "").strip()
        if not query:
            return ToolResult.failure_response(tool_call.id, self.name, "No query provided.")
        try:
            ex = self._executor or init_state.get_executor("entity_graph")
            if ex is None:
                return ToolResult.failure_response(tool_call.id, self.name, "Graph not initialized.")
            result = ex.execute(query)
            headers = result.get_column_names()
            rows = _fetch_all(result)
            return ToolResult.success_response(tool_call.id, self.name, _fmt(query, headers, rows))
        except Exception as e:
            return ToolResult.failure_response(tool_call.id, self.name, f"Query failed: {e}")


def _build_schema(ex) -> str:
    total_nodes = _fetch_all(ex.execute("MATCH (n) RETURN COUNT(*) AS cnt"))[0][0]
    total_edges = _fetch_all(ex.execute("MATCH ()-[e]->() RETURN COUNT(*) AS cnt"))[0][0]

    compat = ""
    if getattr(ex, 'graph_type', '') == 'kuzu':
        compat = (
            "\nKuzuDB: no type(), no labels(), no OPTIONAL MATCH, no collect(). "
            "Use only the exact labels and properties below."
        )

    lines = [
        f"Business entity graph ({total_nodes} nodes, {total_edges} edges). "
        f"Every DB row is a node, every FK is an edge. "
        f"Use ONLY these EXACT names — case-sensitive.{compat}",
    ]

    # ── node labels & properties (from KuzuDB directly) ──
    node_labels = []
    try:
        rows = _fetch_all(ex.execute("CALL show_tables() RETURN name, type"))
        for name, ttype in rows:
            if ttype == "NODE":
                node_labels.append((name, _fetch_properties(ex, name)))
    except Exception:
        pass

    if node_labels:
        lines.append("\nNode labels & properties:")
        for label, props in node_labels:
            cnt = _fetch_all(ex.execute(f"MATCH (n:{label}) RETURN COUNT(*) AS cnt"))[0][0]
            prop_str = ", ".join(props) if props else "id only"
            lines.append(f"  {label} ({cnt} nodes)  properties: {prop_str}")
            # Show 1 sample row so the agent sees actual values
            try:
                sample = _fetch_all(ex.execute(f"MATCH (n:{label}) RETURN n LIMIT 1"))
                if sample:
                    lines.append(f"    example: {_truncate_repr(str(sample[0][0]), 200)}")
            except Exception:
                pass

    # ── edge labels & directions (from KuzuDB directly) ──
    edge_labels = []
    try:
        rows = _fetch_all(ex.execute("CALL show_tables() RETURN name, type"))
        for name, ttype in rows:
            if ttype == "REL":
                edge_labels.append(name)
    except Exception:
        pass

    if edge_labels:
        lines.append("\nRelationships:")
        for label in edge_labels:
            cnt = _fetch_all(ex.execute(f"MATCH ()-[e:{label}]->() RETURN COUNT(*) AS cnt"))[0][0]
            # Discover which node types this edge connects
            endpoints = _discover_edge_endpoints(ex, label)
            direction = f"({endpoints[0]})-[{label}]->({endpoints[1]})" if len(endpoints) == 2 else f"???-[{label}]->???"
            edge_props = _fetch_properties(ex, label)
            prop_str = f" properties: {', '.join(edge_props)}" if edge_props else ""
            lines.append(f"  {direction} ({cnt} edges){prop_str}")

    # ── explicit example queries ──
    if node_labels:
        lines.append("\nExample queries (use EXACTLY these labels):")
        lines.append(f"  MATCH (n:{node_labels[0][0]}) RETURN n LIMIT 5")
        if len(node_labels) > 1 and edge_labels:
            lines.append(
                f"  MATCH (a:{node_labels[0][0]})-[e:{edge_labels[0]}]->(b:{node_labels[1][0]}) "
                f"RETURN a, e, b LIMIT 5"
            )

    return "\n".join(lines)


def _discover_edge_endpoints(ex, label: str) -> list[str]:
    """Find which node types an edge connects."""
    try:
        rows = _fetch_all(ex.execute(
            f"MATCH (a)-[e:{label}]->(b) RETURN a, b LIMIT 1"
        ))
        if rows:
            a = str(rows[0][0])
            b = str(rows[0][1])
            src = _extract_label(a)
            dst = _extract_label(b)
            return [src, dst]
    except Exception:
        pass
    return []


def _extract_label(node_repr: str) -> str:
    """Extract _label from KuzuDB node repr."""
    import re
    m = re.search(r"'_label':\s*'([^']+)'", node_repr)
    return m.group(1) if m else "?"


def _truncate_repr(text: str, max_len: int) -> str:
    return text if len(text) <= max_len else text[:max_len - 3] + "..."


def _fetch_properties(ex, label: str) -> list[str]:
    try:
        rows = _fetch_all(ex.execute(f"CALL table_info('{label}') RETURN name"))
        return sorted(
            str(r[0]) for r in rows
            if str(r[0]) not in ("id", "from", "to")
        )
    except Exception:
        return []


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
