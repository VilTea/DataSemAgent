# tests/test_reasoning_contract.py
from app.semantics.graph.reasoning.contract import (
    FactNode, StepNode, OSIRefNode, SourceNode,
    ReasoningGraphDoc, InputEdge, OutputEdge, ReferenceEdge,
    SourcedFromEdge, EquivalentEdge,
)

class TestReasoningContract:
    def test_fact_node(self):
        f = FactNode(id="f1", content="Sales increased 20%", confidence=0.85,
                     reflection_notes="Derived from SQL query", timestamp=1715000000)
        assert f.content == "Sales increased 20%"
        assert f.confidence == 0.85
        assert f.merged_into is None

    def test_step_node(self):
        s = StepNode(id="s1", description="Aggregate monthly sales", method="induction",
                     confidence=0.9, timestamp=1715000000)
        assert s.method == "induction"

    def test_graph_doc_merge(self):
        doc = ReasoningGraphDoc()
        doc.add_node(FactNode(id="f1", content="A"))
        doc.add_node(FactNode(id="f2", content="B"))
        doc.add_edge(InputEdge(from_="f1", to="s1", dependency="necessary"))
        doc.add_edge(OutputEdge(from_="s1", to="f2"))
        assert len(doc.facts) == 2
        assert len(doc.edges) == 2

        doc2 = ReasoningGraphDoc()
        doc2.add_node(FactNode(id="f1", content="A (updated)"))
        doc2.add_node(FactNode(id="f3", content="C"))
        doc.merge(doc2)
        assert len(doc.facts) == 3
        assert doc.facts[0].content == "A (updated)"
