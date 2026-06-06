"""SubmitAnswerTool — constrained answer output for benchmark scoring."""
from typing import Literal

from app.schema import ToolCall
from app.tool.base import BaseTool, ToolResult


class SubmitAnswerTool(BaseTool):
    permission: Literal["global", "skills", "agent"] = "agent"
    name: str = "submit_answer"
    description: str = (
        "Submit the final answer for the task. Call ONCE when analysis is complete. "
        "answer_type: 'number' for numeric, 'list' for comma-separated values, "
        "'grouped_list' for group:value pairs, 'text' for free-text (yes/no etc), "
        "'not_applicable' when no answer applies."
    )
    strict: bool = True
    parameters: dict = {
        "type": "object",
        "properties": {
            "answer_type": {
                "type": "string",
                "enum": ["number", "list", "grouped_list", "text", "not_applicable"],
                "description": "Expected answer format per the task guidelines",
            },
            "number_value": {
                "type": "number",
                "description": "Required when answer_type='number'.",
            },
            "list_value": {
                "type": "array", "items": {"type": "string"},
                "description": "Required when answer_type='list'.",
            },
            "grouped_value": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "group": {"type": "string"},
                        "value": {"type": "number"},
                    },
                    "required": ["group", "value"],
                },
                "description": "Required when answer_type='grouped_list'.",
            },
            "text_value": {
                "type": "string",
                "description": "Required when answer_type='text'. Free-text (yes/no, name, etc).",
            },
        },
        "required": ["answer_type"],
    }

    async def execute(self, tool_call: ToolCall, **kwargs) -> ToolResult:
        args = tool_call.function.arguments_dict
        atype = args.get("answer_type", "")

        if atype == "number":
            if "number_value" not in args or args["number_value"] is None:
                return ToolResult.failure_response(
                    tool_call.id, self.name,
                    "answer_type='number' but number_value is missing."
                )
            return ToolResult.success_response(tool_call.id, self.name, str(args["number_value"]))

        if atype == "list":
            val = args.get("list_value")
            if not val:
                return ToolResult.failure_response(
                    tool_call.id, self.name,
                    "answer_type='list' but list_value is missing or empty."
                )
            return ToolResult.success_response(tool_call.id, self.name, ", ".join(str(v) for v in val))

        if atype == "grouped_list":
            val = args.get("grouped_value")
            if not val:
                return ToolResult.failure_response(
                    tool_call.id, self.name,
                    "answer_type='grouped_list' but grouped_value is missing or empty."
                )
            lines = [f"{g['group']}: {g['value']}" for g in val]
            return ToolResult.success_response(tool_call.id, self.name, "; ".join(lines))

        if atype == "text":
            val = args.get("text_value", "")
            if not val:
                return ToolResult.failure_response(
                    tool_call.id, self.name,
                    "answer_type='text' but text_value is missing or empty."
                )
            return ToolResult.success_response(tool_call.id, self.name, val)

        if atype == "not_applicable":
            return ToolResult.success_response(tool_call.id, self.name, "Not Applicable")

        return ToolResult.failure_response(
            tool_call.id, self.name,
            f"Unknown answer_type '{atype}'. Must be: number, list, grouped_list, text, not_applicable."
        )
