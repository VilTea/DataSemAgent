# app/semantics/graph/reasoning/tools.py
import json
from pydantic import PrivateAttr
from app.schema import ToolCall
from app.tool.base import BaseTool, ToolResult
from app.semantics.graph.reasoning.contract import ReasoningGraphDoc


class EmitReasoningTool(BaseTool):
    permission: str = "agent"
    name: str = "emit_reasoning"
    description: str = (
        "Emit reasoning chains incrementally. Facts, steps, and edges merge "
        "by id — later calls override earlier ones."
    )
    strict: bool = False
    parameters: dict = {
        "type": "object",
        "properties": {
            "facts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "content": {"type": "string"},
                        "confidence": {"type": "number"},
                        "reflection_notes": {"type": "string"},
                        "is_ontology": {"type": "boolean", "description": "Organising concept — children inherit its chains"},
                        "parent_id": {"type": "string", "description": "Parent fact id for ontology hierarchy"},
                    },
                    "required": ["id", "content"],
                },
            },
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "description": {"type": "string"},
                        "method": {"type": "string", "enum": ["deduction", "induction", "analogy", "abduction"]},
                        "confidence": {"type": "number"},
                    },
                    "required": ["id", "description"],
                },
            },
            "edges": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "from": {"type": "string"},
                        "to": {"type": "string"},
                        "label": {"type": "string", "enum": ["input_to", "outputs", "references", "sourced_from", "equivalent_to"]},
                        "dependency": {"type": "string", "enum": ["necessary", "sufficient", "contributing"]},
                        "merged": {"type": "boolean"},
                    },
                    "required": ["from", "to", "label"],
                },
            },
        },
        "required": ["facts", "steps", "edges"],
    }

    _accumulated: ReasoningGraphDoc = PrivateAttr(default_factory=ReasoningGraphDoc)

    @property
    def accumulated(self) -> ReasoningGraphDoc:
        return self._accumulated

    async def execute(self, tool_call: ToolCall, **kwargs) -> ToolResult:
        try:
            data = tool_call.function.arguments_dict
            delta = ReasoningGraphDoc.model_validate(data)
        except Exception as e:
            return ToolResult.failure_response(tool_call.id, self.name, f"Invalid: {e}")

        self._accumulated.merge(delta)
        nf = len(self._accumulated.facts)
        ns = len(self._accumulated.steps)
        ne = len(self._accumulated.edges)
        ont_count = sum(1 for f in self._accumulated.facts if f.is_ontology)
        return ToolResult.success_response(
            tool_call.id, self.name,
            f"Merged ({nf} facts ({ont_count} ontologies), {ns} steps, {ne} edges)."
        )


class ReadReasoningTool(BaseTool):
    permission: str = "agent"
    name: str = "read_reasoning"
    description: str = "Read current accumulated reasoning graph."
    strict: bool = False
    parameters: dict = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
        "required": [],
    }
    _emit_tool: EmitReasoningTool = PrivateAttr()

    def __init__(self, emit_tool: EmitReasoningTool, **kwargs):
        super().__init__(**kwargs)
        self._emit_tool = emit_tool

    async def execute(self, tool_call: ToolCall, **kwargs) -> ToolResult:
        doc = self._emit_tool.accumulated
        return ToolResult.success_response(
            tool_call.id, self.name, doc.model_dump_json(indent=2)
        )


class DeleteReasoningPathTool(BaseTool):
    permission: str = "agent"
    name: str = "delete_reasoning_path"
    description: str = "Delete a fact or step by id."
    strict: bool = False
    parameters: dict = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "Node id to delete"},
        },
        "required": ["id"],
    }
    _emit_tool: EmitReasoningTool = PrivateAttr()

    def __init__(self, emit_tool: EmitReasoningTool, **kwargs):
        super().__init__(**kwargs)
        self._emit_tool = emit_tool

    async def execute(self, tool_call: ToolCall, **kwargs) -> ToolResult:
        target = tool_call.function.arguments_dict.get("id", "")
        doc = self._emit_tool.accumulated
        for lst in [doc.facts, doc.steps]:
            for i, n in enumerate(lst):
                if n.id == target:
                    lst.pop(i)
                    return ToolResult.success_response(
                        tool_call.id, self.name, f"Removed '{target}'"
                    )
        return ToolResult.failure_response(tool_call.id, self.name, f"Not found: {target}")
