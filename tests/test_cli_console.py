import asyncio
import pytest
from unittest.mock import MagicMock

from app.schema import AgentCompletion, FinishReason, Role, Message, ToolCall, Function
from app.pipeline.abc import EventConsumer


class TestRichConsumer:
    @pytest.fixture
    def mock_console(self):
        return MagicMock()

    def test_is_event_consumer(self, mock_console):
        from app.cli.console import RichConsumer
        consumer = RichConsumer(mock_console)
        assert isinstance(consumer, EventConsumer)

    def test_consume_content_stream(self, mock_console):
        from app.cli.console import RichConsumer
        consumer = RichConsumer(mock_console)
        event = AgentCompletion(role=Role.ASSISTANT, content="Hello", full_content="Hello")
        asyncio.run(consumer.consume(event))
        mock_console.print.assert_any_call("Hello", end="")

    def test_consume_tool_calls(self, mock_console):
        from app.cli.console import RichConsumer
        consumer = RichConsumer(mock_console)
        tc = ToolCall(id="1", function=Function(name="sql_exec", arguments='{"sql": "SELECT 1"}'))
        event = AgentCompletion(role=Role.ASSISTANT, finish_reason=FinishReason.TOOL_CALLS,
                                full_tool_calls=[tc])
        asyncio.run(consumer.consume(event))
        assert mock_console.print.call_count >= 1

    def test_consume_finish_reason(self, mock_console):
        from app.cli.console import RichConsumer
        consumer = RichConsumer(mock_console)
        event = AgentCompletion(
            role=Role.ASSISTANT, finish_reason=FinishReason.STOP, full_content=""
        )
        asyncio.run(consumer.consume(event))
        mock_console.print.assert_called()

    def test_consume_tool_message(self, mock_console):
        from app.cli.console import RichConsumer
        consumer = RichConsumer(mock_console)
        msg = Message(role=Role.TOOL, content="**10 row(s)**\n| col |", tool_call_id="1", name="entity_graph")
        asyncio.run(consumer.consume(msg))
        mock_console.print.assert_called()
