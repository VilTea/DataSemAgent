# tests/test_reasoning_builder.py
from app.semantics.graph.reasoning.contract import (
    ReasoningGraphDoc, FactNode, StepNode, InputEdge, OutputEdge,
)
from app.semantics.graph.reasoning.builder import ReasoningGraphBuilder

def test_build_graph_document():
    doc = ReasoningGraphDoc()
    doc.add_node(FactNode(id="f1", content="A"))
    doc.add_node(FactNode(id="f2", content="B"))
    doc.add_node(StepNode(id="s1", description="Step", method="deduction"))
    doc.add_edge(InputEdge(from_="f1", to="s1", dependency="necessary"))
    doc.add_edge(OutputEdge(from_="s1", to="f2"))

    builder = ReasoningGraphBuilder()
    gdoc = builder.build(doc)
    assert len(gdoc.nodes) > 0
    assert len(gdoc.edges) > 0
    # Check that node labels are correct
    labels = {n.label for n in gdoc.nodes}
    assert "Fact" in labels
    assert "ReasoningStep" in labels
