# app/tool/reasoning_graph.py
from typing import Literal
from pydantic import PrivateAttr, model_validator
from app.db.graph import GraphExecutor
from app.hook import HookPoint, hook
from app.schema import ToolCall
from app.semantics.graph.init_state import init_state
from app.tool.base import BaseTool, ToolResult


class ReasoningGraphTool(BaseTool):
    permission: Literal["global", "skills", "agent"] = "agent"
    name: str = "reasoning_graph"
    description: str = (
        "Query the reasoning chain graph (Cypher). "
        "Schema (node/edge types, properties) is in the system prompt under "
        "<reasoning_graph_schema>. If the graph has not been built yet, it is "
        "created automatically after several conversation rounds. "
        "All content is stored in English."
    )
    strict: bool = True
    parameters: dict = {
        "type": "object", "properties": {
            "query": {"type": "string"},
        }, "required": ["query"],
    }

    _executor: GraphExecutor | None = PrivateAttr(default=None)

    def is_available(self) -> bool:
        return True

    @model_validator(mode="after")
    def _init(self) -> "ReasoningGraphTool":
        if init_state.is_ready("reasoning_graph"):
            self._executor = init_state.get_executor("reasoning_graph")
        return self

    @hook(HookPoint.NODE_INIT_BEFORE)
    def _inject_schema(self, ctx, node):
        if not node.system_prompt:
            return
        ex = self._executor or init_state.get_executor("reasoning_graph")
        if ex is None:
            return
        text = self._build_schema(ex)
        node.system_prompt = (
            node.system_prompt
            + "\n\n<reasoning_graph_schema>\n"
            + text
            + "\n</reasoning_graph_schema>"
        )

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

    def _build_schema(self, ex) -> str:
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
            f"ALL content is in English. "
            f"ONLY these tables & columns exist:\n"
            f"  Nodes: {', '.join(node_schemas) or '(none)'}\n"
            f"  Edges: {', '.join(edge_schemas)}"
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
