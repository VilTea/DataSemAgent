import pytest
from app.schema import Message, Role


class TestMessageInjected:
    def test_injected_default_role_user(self):
        msg = Message.injected(content="summary text")
        assert msg.role == Role.USER
        assert msg.content == "summary text"
        assert msg.injected is True

    def test_injected_explicit_role(self):
        msg = Message.injected(content="sys summary", role=Role.SYSTEM)
        assert msg.role == Role.SYSTEM
        assert msg.content == "sys summary"
        assert msg.injected is True

    def test_injected_no_tool_fields(self):
        msg = Message.injected(content="hello")
        assert msg.tool_calls is None
        assert msg.tool_call_id is None
        assert msg.name is None
