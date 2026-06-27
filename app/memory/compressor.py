from app.memory.token_counter import TokenCounter
from app.schema import Message


DEFAULT_SUMMARY_PROMPT = """Summarize the conversation above concisely. Keep:
- Key analytical findings and numerical results
- Metric definitions and SQL patterns used
- Important reasoning patterns and pitfalls encountered
- The user's original question and any clarifications
Discard:
- Verbose tool outputs (replace with a one-line description)
- Dead ends, retries, and failed queries
- Repetitive exploration steps"""


class ContextCompressor:
    def __init__(self, token_counter: TokenCounter):
        self._token_counter = token_counter

    async def compress(
        self,
        messages: list[Message],
        *,
        context_window: int,
        threshold: float,
        keep_recent_turns: int,
        is_turn_boundary: bool,
        llm,
        summary_system_prompt: str | None = None,
    ) -> list[Message]:
        limit = int(context_window * threshold)
        tok = self._token_counter.count([m.to_dict() for m in messages])
        if tok <= limit:
            return messages

        # Operate on a copy — never mutate the caller's list
        messages = list(messages)
        keep_start = self._find_keep_zone(messages, keep_recent_turns)

        # ---- Tier 1: prune old tool results ----
        for i in range(keep_start):
            if messages[i].role == "tool":
                messages[i].content = f"[cleared: {messages[i].name}]"
        tok = self._token_counter.count([m.to_dict() for m in messages])
        if tok <= limit:
            return messages

        # ---- Tier 2: LLM summarization (turn boundaries only) ----
        if not is_turn_boundary:
            # Still over limit but mid-turn: force-truncate as fallback
            return self._force_truncate(messages, limit)

        transcript = self._serialize_for_summary(messages[:keep_start])
        prompt = summary_system_prompt or DEFAULT_SUMMARY_PROMPT

        try:
            summary = await self._call_summarizer(llm, prompt, transcript)
        except Exception:
            return messages

        compressed = [
            messages[0],
            Message.injected(f"[Conversation Summary]\n\n{summary}"),
        ]
        compressed.extend(messages[keep_start:])

        tok = self._token_counter.count([m.to_dict() for m in compressed])
        if tok > limit:
            compressed = self._force_truncate(compressed, limit)

        return compressed

    def _find_keep_zone(self, messages: list[Message], keep_turns: int) -> int:
        turns = 0
        for i in range(len(messages) - 1, -1, -1):
            m = messages[i]
            if m.role == "user" and not m.injected:
                turns += 1
                if turns >= keep_turns:
                    return i
        return 0

    def _serialize_for_summary(self, messages: list[Message]) -> str:
        lines = []
        for m in messages:
            role = m.role or "unknown"
            content = m.content or ""
            if m.role == "tool":
                content = f"[tool result from {m.name}]"
            lines.append(f"[{role}]: {content}")
        return "\n".join(lines)

    async def _call_summarizer(self, llm, system_prompt: str, transcript: str) -> str:
        msgs = [
            Message.system_message(system_prompt),
            Message.user_message(transcript),
        ]
        async for completion in llm.ask_tool(messages=msgs, stream=False):
            if completion and completion.content:
                return completion.content
        return ""

    def _force_truncate(self, messages: list[Message], limit: int) -> list[Message]:
        result = list(messages)
        while self._token_counter.count([m.to_dict() for m in result]) > limit and len(result) > 2:
            cut = 1
            while cut < len(result) and result[cut].role != "user":
                cut += 1
            if cut >= len(result) or result[cut].role != "user":
                break
            next_user = cut + 1
            while next_user < len(result) and result[next_user].role != "user":
                next_user += 1
            result = [result[0]] + result[next_user:]
        return result
