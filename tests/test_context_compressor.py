import asyncio
import pytest
from unittest.mock import MagicMock
from app.memory.token_counter import TokenCounter
from app.memory.compressor import ContextCompressor
from app.schema import Message, Role


def _make_msg(role, content="", name=None, injected=False):
    return Message(role=role, content=content, name=name, injected=injected)


class TestContextCompressor:
    @pytest.fixture
    def compressor(self):
        return ContextCompressor(TokenCounter())

    @pytest.fixture
    def sample_messages(self):
        return [
            _make_msg(Role.SYSTEM, "You are helpful."),
            _make_msg(Role.USER, "Q1"),
            _make_msg(Role.ASSISTANT, "A1"),
            _make_msg(Role.TOOL, "tool result 1", name="sql_exec"),
            _make_msg(Role.USER, "Q2"),
            _make_msg(Role.ASSISTANT, "A2"),
            _make_msg(Role.TOOL, "tool result 2", name="entity_graph"),
            _make_msg(Role.USER, "Q3"),
        ]

    def test_no_compression_below_threshold(self, compressor, sample_messages):
        async def _run():
            return await compressor.compress(
                sample_messages,
                context_window=1000000, threshold=0.8,
                keep_recent_turns=1, is_turn_boundary=True, llm=None,
            )
        result = asyncio.run(_run())
        assert result is sample_messages

    def test_tier_1_prunes_old_tool_results(self, compressor, sample_messages):
        async def _run():
            return await compressor.compress(
                list(sample_messages),  # copy
                context_window=40, threshold=0.5,
                keep_recent_turns=1, is_turn_boundary=True, llm=None,
            )
        result = asyncio.run(_run())
        for m in result:
            if m.role == Role.TOOL and hasattr(m, 'name') and m.name in ("sql_exec", "entity_graph"):
                assert "[cleared:" in (m.content or "")

    def test_find_keep_zone_counts_turns(self, compressor, sample_messages):
        idx = compressor._find_keep_zone(sample_messages, keep_turns=2)
        assert idx == 4  # Q2 index

    def test_find_keep_zone_skips_injected(self, compressor):
        msgs = [
            _make_msg(Role.SYSTEM, "sys"),
            _make_msg(Role.USER, "real Q1"),
            _make_msg(Role.ASSISTANT, "A1"),
            _make_msg(Role.USER, "summary", injected=True),
            _make_msg(Role.USER, "real Q2"),
        ]
        idx = compressor._find_keep_zone(msgs, keep_turns=1)
        assert idx == 4

    def test_serialize_for_summary(self, compressor, sample_messages):
        text = compressor._serialize_for_summary(sample_messages[:4])
        assert "[system]" in text
        assert "[user]" in text
        assert "[tool result from sql_exec]" in text

    def test_tier_2_gated_by_is_turn_boundary(self, compressor, sample_messages):
        async def _run():
            return await compressor.compress(
                list(sample_messages),
                context_window=40, threshold=0.5,
                keep_recent_turns=1, is_turn_boundary=False, llm=MagicMock(),
            )
        result = asyncio.run(_run())
        assert not any("Conversation Summary" in (m.content or "") for m in result)

    def test_force_truncate_preserves_system_prompt(self, compressor):
        msgs = [
            _make_msg(Role.SYSTEM, "sys prompt"),
            _make_msg(Role.USER, "Q1"),
            _make_msg(Role.ASSISTANT, "A1"),
            _make_msg(Role.USER, "Q2"),
            _make_msg(Role.ASSISTANT, "A2"),
        ]
        result = compressor._force_truncate(msgs, limit=30)
        assert result[0].role == Role.SYSTEM
        assert result[0].content == "sys prompt"
