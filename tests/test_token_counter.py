import pytest
from app.memory.token_counter import TokenCounter


class TestTokenCounter:
    def test_count_string(self):
        tc = TokenCounter()
        result = tc.count("hello world")
        assert isinstance(result, int)
        assert result > 0

    def test_count_empty_string(self):
        tc = TokenCounter()
        assert tc.count("") == 0

    def test_count_message_dicts(self):
        tc = TokenCounter()
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
        ]
        result = tc.count(msgs)
        assert isinstance(result, int)
        assert result > 0

    def test_count_message_without_content(self):
        tc = TokenCounter()
        msgs = [{"role": "assistant"}]
        result = tc.count(msgs)
        assert isinstance(result, int)

    def test_count_consistent(self):
        tc = TokenCounter()
        a = tc.count("hello world")
        b = tc.count("hello world")
        assert a == b
