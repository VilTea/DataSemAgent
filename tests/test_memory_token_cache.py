from app.memory.token_counter import TokenCounter
from app.schema import Message, Memory


class TestMemoryTokenCache:
    def test_add_message_increments_cache(self):
        mem = Memory()
        mem._token_counter = TokenCounter()
        mem._token_total = 0
        msg = Message.user_message("hello world")
        mem.add_message(msg)
        assert mem._token_total > 0

    def test_clear_resets_cache(self):
        mem = Memory()
        mem._token_counter = TokenCounter()
        msg = Message.user_message("hello world")
        mem.add_message(msg)
        assert mem._token_total > 0
        mem.clear()
        assert mem._token_total == 0
        assert mem.messages == []
