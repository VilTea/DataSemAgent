"""Entity graph query tool."""
from typing import Literal

from pydantic import PrivateAttr, model_validator

from app.db.graph import GraphExecutor
from app.schema import ToolCall
from app.semantics.graph.init_state import init_state
from app.tool.base import BaseTool, ToolResult


class EntityGraphTool(BaseTool):
    permission: Literal["global", "skills", "agent"] = "agent"
    name: str = "entity_graph"
    description: str = "Query the business entity knowledge graph. Input is a Cypher query."
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
            ex = init_state.get_executor("entity_graph")
            self._executor = ex
            self.description = _build_description(ex)
        return self

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


def _build_description(ex) -> str:
    from app.semantics.graph.entity.schema import EntityGraphSchema
    schema = init_state.get_extra("entity_graph", "schema")
    nodes = _fetch_all(ex.execute("MATCH (n) RETURN COUNT(*) AS cnt"))[0][0]
    edges = _fetch_all(ex.execute("MATCH ()-[e]->() RETURN COUNT(*) AS cnt"))[0][0]

    lines = [
        f"Query the business entity graph ({nodes} nodes, {edges} edges). "
        f"Built from the OSI semantic model — every row from the relational database "
        f"is a node, every foreign key is an edge. "
        f"The same data accessible via sql_exec lives here as connected entities. "
        f"Input is a Cypher query.\n",
    ]
    if schema and isinstance(schema, EntityGraphSchema):
        lines.append("## Entity types")
        for e in schema.entities:
            cnt_row = _fetch_all(ex.execute(f"MATCH (n:{e.label}) RETURN COUNT(*) AS cnt"))
            cnt = cnt_row[0][0] if cnt_row else 0
            props = _fetch_properties(ex, e.label)
            tags = []
            if e.is_event: tags.append("event")
            if e.is_weak: tags.append(f"weak→{', '.join(e.strong_parents)}")
            tag = f" ({', '.join(tags)})" if tags else ""
            lines.append(f"- **{e.label}**{tag} ({cnt} nodes): {e.description}")
            if props:
                lines.append(f"  Properties: {', '.join(props)}")
        lines.append("\n## Relationships")
        for r in schema.relations:
            cnt_row = _fetch_all(ex.execute(f"MATCH ()-[e:{r.label}]->() RETURN COUNT(*) AS cnt"))
            cnt = cnt_row[0][0] if cnt_row else 0
            lines.append(f"- **{r.label}** ({cnt} edges): {r.from_} → {r.to}")
    lines.append("\n## Example queries")
    if schema and schema.entities:
        e0 = schema.entities[0]
        lines.append(f"MATCH (n:{e0.label}) RETURN n LIMIT 5")
    if schema and schema.relations:
        r0 = schema.relations[0]
        lines.append(f"MATCH ()-[e:{r0.label}]->() RETURN e LIMIT 5")
    return "\n".join(lines)


def _fetch_properties(ex, label: str) -> list[str]:
    """Return sorted property names for a node label, excluding internal keys."""
    try:
        r = ex.execute(f"MATCH (n:{label}) RETURN n LIMIT 1")
        rows = _fetch_all(r)
        if not rows:
            return []
        raw = str(rows[0][0])
        # Extract quoted keys from the dict-like repr: 'key': value
        import re
        keys = re.findall(r"'([^']+)'\s*:", raw)
        return sorted(k for k in keys if not k.startswith("_"))
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
