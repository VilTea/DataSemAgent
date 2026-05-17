# app/tool/reasoning_graph.py
from typing import Literal
from pydantic import PrivateAttr, model_validator
from app.db.graph import GraphExecutor
from app.schema import ToolCall
from app.semantics.graph.init_state import init_state
from app.tool.base import BaseTool, ToolResult


class ReasoningGraphTool(BaseTool):
    permission: Literal["global", "skills", "agent"] = "agent"
    name: str = "reasoning_graph"
    description: str = "Query the reasoning chain graph (Cypher). Input is a Cypher query."
    strict: bool = True
    parameters: dict = {
        "type": "object", "properties": {
            "query": {"type": "string"},
        }, "required": ["query"],
    }

    _executor: GraphExecutor | None = PrivateAttr(default=None)

    def is_available(self) -> bool:
        return True  # always show; return "not ready" when graph not yet built

    @model_validator(mode="after")
    def _init(self) -> "ReasoningGraphTool":
        if init_state.is_ready("reasoning_graph"):
            self._executor = init_state.get_executor("reasoning_graph")
            self.description = self._build_desc()
        else:
            self.description = (
                "Query the reasoning chain graph (Cypher). "
                "The graph has not been built yet — it is created automatically "
                "after several conversation rounds when the reflection mechanism runs. "
                "All content is stored in English."
            )
        return self

    async def execute(self, tool_call: ToolCall, **kwargs) -> ToolResult:
        query = tool_call.function.arguments_dict.get("query", "").strip()
        if not query:
            return ToolResult.failure_response(tool_call.id, self.name, "No query provided.")
        ex = self._executor or init_state.get_executor("reasoning_graph")
        if ex is None:
            return ToolResult.failure_response(
                tool_call.id, self.name,
                "Reasoning graph has not been built yet. "
                "It is created automatically after several conversation rounds."
            )
        try:
            result = ex.execute(query)
            headers = result.get_column_names()
            rows = self._fetch_all(result)
            return ToolResult.success_response(tool_call.id, self.name, self._fmt(query, headers, rows))
        except Exception as e:
            return ToolResult.failure_response(tool_call.id, self.name, f"Query failed: {e}")

    def _build_desc(self) -> str:
        ex = self._executor or init_state.get_executor("reasoning_graph")
        if ex is None:
            return "Query the reasoning chain graph."
        n = self._fetch_all(ex.execute("MATCH (n) RETURN COUNT(*) AS cnt"))[0][0]
        e = self._fetch_all(ex.execute("MATCH ()-[e]->() RETURN COUNT(*) AS cnt"))[0][0]
        ont = self._fetch_all(
            ex.execute("MATCH (f:Fact) WHERE f.is_ontology = True RETURN COUNT(*) AS cnt")
        )[0][0]

        # ── node schema (CALL table_info gives property names) ──
        node_schemas: list[str] = []
        for label in ("Fact", "ReasoningStep", "OSIRef", "Source"):
            try:
                rows = self._fetch_all(
                    ex.execute(f"CALL table_info('{label}') RETURN name")
                )
                props = [str(r[0]) for r in rows if str(r[0]) != "id"]
                if props:
                    node_schemas.append(f"{label}{{{', '.join(props)}}}")
            except Exception:
                pass

        # ── edge schema ──
        edge_schemas: list[str] = []
        for label in ("input_to", "outputs", "references", "sourced_from", "equivalent_to"):
            try:
                rows = self._fetch_all(
                    ex.execute(f"CALL table_info('{label}') RETURN name")
                )
                props = [str(r[0]) for r in rows if str(r[0]) not in ("id", "from", "to")]
                if props:
                    edge_schemas.append(f"{label}{{{', '.join(props)}}}")
                else:
                    edge_schemas.append(label)
            except Exception:
                pass

        if not edge_schemas:
            edge_schemas.append("(no edges yet)")

        return (
            f"Reasoning chain graph ({n} nodes, {e} edges, {ont} ontology concepts). "
            f"ALL content is stored in English. "
            f"This graph stores reusable fact reasoning patterns extracted from past "
            f"conversation rounds — not specific data values, but general analytical "
            f"approaches and inference chains that remain valuable across sessions. "
            f"When facing a complex analytical problem, ALWAYS check this graph first "
            f"to see if there are relevant reusable reasoning patterns before starting "
            f"from scratch.\n\n"
            f"Schema — ONLY these tables & columns exist; do NOT invent others:\n"
            f"  Nodes: {', '.join(node_schemas) or '(none)'}\n"
            f"  Edges: {', '.join(edge_schemas)}\n"
            f"Input is a Cypher query."
        )

    @staticmethod
    def _fetch_all(result):
        rows = []
        while result.has_next():
            rows.append(result.get_next())
        return rows

    @staticmethod
    def _fmt(query, headers, rows):
        if not rows:
            return f"**No results.**\n\n```cypher\n{query}\n```"
        lines = [f"**{len(rows)} row(s)**", "", "```cypher", query, "```", ""]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join("---" for _ in headers) + " |")
        for row in rows:
            lines.append("| " + " | ".join(str(v) for v in row) + " |")
        return "\n".join(lines)
