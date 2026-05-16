# tests/test_reasoning_merger.py
from app.semantics.graph.reasoning.contract import (
    ReasoningGraphDoc, FactNode, EquivalentEdge,
)
from app.semantics.graph.reasoning.merger import SynonymMerger

def test_merger():
    doc = ReasoningGraphDoc()
    doc.add_node(FactNode(id="f1", content="Revenue grew"))
    doc.add_node(FactNode(id="f2", content="Revenue increased"))
    doc.add_edge(EquivalentEdge(from_="f1", to="f2", merged=True))

    m = SynonymMerger()
    count = m.merge(doc)
    assert count == 1
    assert doc.facts[1].merged_into == "f1"

def test_no_merge_when_not_merged_flag():
    doc = ReasoningGraphDoc()
    doc.add_node(FactNode(id="f1", content="A"))
    doc.add_node(FactNode(id="f2", content="B"))
    doc.add_edge(EquivalentEdge(from_="f1", to="f2", merged=False))

    m = SynonymMerger()
    count = m.merge(doc)
    assert count == 0
    assert doc.facts[1].merged_into is None
