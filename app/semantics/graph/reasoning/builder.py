# app/semantics/graph/reasoning/builder.py
import time
from app.semantics.graph.contract import GraphDocument, GraphNode, GraphEdge
from app.semantics.graph.reasoning.contract import ReasoningGraphDoc


class ReasoningGraphBuilder:
    def build(self, doc: ReasoningGraphDoc) -> GraphDocument:
        g = GraphDocument()
        ts = int(time.time())

        for f in doc.facts:
            g.add_node(GraphNode(
                id=f.id, label="Fact",
                properties={
                    "content": f.content,
                    "confidence": f.confidence,
                    "reflection_notes": f.reflection_notes,
                    "timestamp": f.timestamp or ts,
                    "merged_into": f.merged_into or "",
                    "is_ontology": f.is_ontology,
                    "parent_id": f.parent_id or "",
                },
            ))
        for s in doc.steps:
            g.add_node(GraphNode(
                id=s.id, label="ReasoningStep",
                properties={
                    "description": s.description,
                    "method": s.method,
                    "confidence": s.confidence,
                    "timestamp": s.timestamp or ts,
                },
            ))
        for r in doc.osi_refs:
            g.add_node(GraphNode(
                id=r.id, label="OSIRef",
                properties={
                    "osi_type": r.osi_type,
                    "osi_name": r.osi_name,
                    "definition": r.definition,
                },
            ))
        for s in doc.sources:
            g.add_node(GraphNode(
                id=s.id, label="Source",
                properties={
                    "source_type": s.source_type,
                    "turn_number": s.turn_number,
                    "query_sql": s.query_sql,
                },
            ))

        for e in doc.edges:
            props = {}
            if hasattr(e, 'dependency'):
                props['dependency'] = e.dependency
            if hasattr(e, 'merged'):
                props['merged'] = e.merged
            g.add_edge(GraphEdge(
                from_=e.from_,
                to=e.to,
                label=e.label,
                properties=props,
            ))

        return g
