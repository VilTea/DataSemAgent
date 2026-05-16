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

        # Discover which edge types actually exist (avoid misleading the agent).
        all_edge_labels = ("input_to", "outputs", "references", "sourced_from", "equivalent_to")
        existing_edges: list[str] = []
        for label in all_edge_labels:
            try:
                cnt = self._fetch_all(
                    ex.execute(f"MATCH ()-[e:{label}]->() RETURN COUNT(*) AS cnt")
                )[0][0]
                if cnt > 0:
                    existing_edges.append(label)
            except Exception:
                pass

        # Build edge type list with property hints for types that have them
        edge_parts = []
        for label in existing_edges:
            if label == "input_to":
                edge_parts.append("input_to(dependency: necessary/sufficient/contributing)")
            elif label == "equivalent_to":
                edge_parts.append("equivalent_to(merged=True for synonyms)")
            else:
                edge_parts.append(label)
        edge_desc = ", ".join(edge_parts) if edge_parts else "(no edges yet)"

        return (
            f"Reasoning chain graph ({n} nodes, {e} edges, {ont} ontology concepts). "
            f"ALL content is stored in English. "
            f"This graph stores reusable fact reasoning patterns extracted from past "
            f"conversation rounds — not specific data values, but general analytical "
            f"approaches and inference chains that remain valuable across sessions. "
            f"When facing a complex analytical problem, ALWAYS check this graph first "
            f"to see if there are relevant reusable reasoning patterns before starting "
            f"from scratch.\n\n"
            f"Existing edge types: {edge_desc}.\n"
            f"Node types: Fact (is_ontology=True, parent_id for hierarchy; children "
            f"inherit parent chains), ReasoningStep (method: deduction/induction/analogy/"
            f"abduction), OSIRef, Source.\n"
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
