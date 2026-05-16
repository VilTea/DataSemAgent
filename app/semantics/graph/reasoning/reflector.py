# app/semantics/graph/reasoning/reflector.py
from app.logger import logger
from app.schema import Memory


class ReasoningReflector:
    LAST_N_ROUNDS = 10

    @staticmethod
    def build_prompt(memory: Memory, existing_summary: str) -> str:
        rounds = ReasoningReflector._extract_last_n_rounds(memory)
        return (
            "Analyze this conversation and extract **reusable reasoning patterns** — "
            "not specific data values. Data changes over time, but reasoning patterns "
            "remain valuable across sessions.\n\n"
            "**What to extract:**\n"
            "- General analytical approaches: how dimensional breakdowns expose trends, "
            "how period-over-period comparisons work, which metrics correlate with which dimensions\n"
            "- Multi-step reasoning chains: the sequence of inferences that led to a conclusion\n"
            "- Common pitfalls or dead ends in the analysis\n"
            "- Ontology concepts that organise knowledge (e.g. 'Customer Segmentation', "
            "'Seasonality Analysis', 'Profitability Decomposition')\n\n"
            "**What to IGNORE:**\n"
            "- Specific numbers (e.g. 'profit was $1.2M' — instead note 'profit magnitude is relevant')\n"
            "- Entity identifiers (e.g. 'customer_id=12345' — instead note 'individual customer granularity')\n"
            "- Time-bound facts that won't hold after the next data refresh\n\n"
            "ONTOLOGIES: Use is_ontology=True to mark organising concepts. "
            "Set parent_id on child facts to build hierarchy — children inherit all parent chains.\n\n"
            "Call emit_reasoning to add patterns incrementally. Stop when done.\n\n"
            "For each chain:\n"
            "- Facts: reusable conclusions with confidence (0-1)\n"
            "- Steps: inference method (deduction/induction/analogy/abduction)\n"
            "- dependency: necessary / sufficient / contributing\n"
            "- OSI refs: which metric/dimension/dataset definitions were used\n"
            "- Sources: which conversation rounds produced these facts\n\n"
            f"## Existing reasoning graph (to avoid duplicates)\n{existing_summary}\n\n"
            f"## Last {ReasoningReflector.LAST_N_ROUNDS} rounds\n{rounds}"
        )

    @staticmethod
    def _extract_last_n_rounds(memory: Memory) -> str:
        msgs = memory.messages
        user_indices = [
            i for i, m in enumerate(msgs)
            if m.role == "user" and not m.injected
        ]
        if not user_indices:
            return "(no user messages)"
        start_idx = user_indices[-min(len(user_indices), ReasoningReflector.LAST_N_ROUNDS)]
        last_msgs = msgs[start_idx:]
        lines = []
        for m in last_msgs:
            if m.injected:
                continue
            role = m.role
            content = (m.content or "")[:500]
            if m.tool_calls:
                names = [tc.function.name for tc in m.tool_calls]
                content = f"[tool_calls: {', '.join(names)}] {content}"
            lines.append(f"[{role}] {content}")
        return "\n".join(lines)
