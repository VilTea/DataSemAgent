# tests/test_reasoning_tools.py
import asyncio
from unittest.mock import MagicMock


class TestEmitReasoningTool:
    def test_accumulates_facts(self):
        from app.semantics.graph.reasoning.tools import EmitReasoningTool
        from app.schema import ToolCall, Function

        tool = EmitReasoningTool()
        tc = ToolCall(id="1", function=Function(name="emit_reasoning", arguments='''{
            "facts": [{"id": "f1", "content": "Revenue grew 20%"}],
            "steps": [{"id": "s1", "description": "Query monthly sales", "method": "deduction"}],
            "edges": [{"from": "f1", "to": "s1", "label": "input_to", "dependency": "necessary"}]
        }'''))
        result = asyncio.run(tool.execute(tc))
        assert "Merged" in result.content
        doc = tool.accumulated
        assert len(doc.facts) == 1
        assert len(doc.edges) == 1

    def test_merge_overrides_by_id(self):
        from app.semantics.graph.reasoning.tools import EmitReasoningTool
        from app.schema import ToolCall, Function

        tool = EmitReasoningTool()
        tc1 = ToolCall(id="1", function=Function(name="emit_reasoning", arguments='''{
            "facts": [{"id": "f1", "content": "Revenue grew 20%"}],
            "steps": [],
            "edges": []
        }'''))
        asyncio.run(tool.execute(tc1))

        tc2 = ToolCall(id="2", function=Function(name="emit_reasoning", arguments='''{
            "facts": [{"id": "f1", "content": "Revenue grew 25% (corrected)"}],
            "steps": [],
            "edges": []
        }'''))
        asyncio.run(tool.execute(tc2))

        doc = tool.accumulated
        assert len(doc.facts) == 1
        assert doc.facts[0].content == "Revenue grew 25% (corrected)"
