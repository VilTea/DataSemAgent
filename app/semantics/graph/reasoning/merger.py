# app/semantics/graph/reasoning/merger.py
from app.semantics.graph.reasoning.contract import ReasoningGraphDoc, EquivalentEdge


class SynonymMerger:
    def merge(self, doc: ReasoningGraphDoc) -> int:
        """Merge equivalent facts. Returns count of merged facts."""
        eq_map: dict[str, str] = {}  # redundant_id → canonical_id
        for e in doc.edges:
            if isinstance(e, EquivalentEdge) and e.merged:
                eq_map[e.to] = e.from_

        merged = 0
        for fact in doc.facts:
            if fact.id in eq_map:
                fact.merged_into = eq_map[fact.id]
                merged += 1
        return merged
