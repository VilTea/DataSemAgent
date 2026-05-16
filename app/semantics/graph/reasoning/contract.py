# app/semantics/graph/reasoning/contract.py
from __future__ import annotations
from pydantic import BaseModel, Field


class FactNode(BaseModel):
    id: str
    content: str
    confidence: float = 1.0
    reflection_notes: str = ""
    timestamp: int = 0
    merged_into: str | None = None
    is_ontology: bool = False    # organising concept; children inherit its chains
    parent_id: str | None = None  # parent Fact id for ontology hierarchy


class StepNode(BaseModel):
    id: str
    description: str
    method: str = "deduction"
    confidence: float = 1.0
    timestamp: int = 0


class OSIRefNode(BaseModel):
    id: str
    osi_type: str
    osi_name: str
    definition: str = ""


class SourceNode(BaseModel):
    id: str
    source_type: str
    turn_number: int = 0
    query_sql: str = ""


class InputEdge(BaseModel):
    model_config = {"populate_by_name": True}
    from_: str = Field(..., alias="from")
    to: str
    label: str = "input_to"
    dependency: str = "contributing"


class OutputEdge(BaseModel):
    model_config = {"populate_by_name": True}
    from_: str = Field(..., alias="from")
    to: str
    label: str = "outputs"


class ReferenceEdge(BaseModel):
    model_config = {"populate_by_name": True}
    from_: str = Field(..., alias="from")
    to: str
    label: str = "references"


class SourcedFromEdge(BaseModel):
    model_config = {"populate_by_name": True}
    from_: str = Field(..., alias="from")
    to: str
    label: str = "sourced_from"


class EquivalentEdge(BaseModel):
    model_config = {"populate_by_name": True}
    from_: str = Field(..., alias="from")
    to: str
    label: str = "equivalent_to"
    merged: bool = False


EdgeItem = InputEdge | OutputEdge | ReferenceEdge | SourcedFromEdge | EquivalentEdge


class ReasoningGraphDoc(BaseModel):
    facts: list[FactNode] = Field(default_factory=list)
    steps: list[StepNode] = Field(default_factory=list)
    osi_refs: list[OSIRefNode] = Field(default_factory=list)
    sources: list[SourceNode] = Field(default_factory=list)
    edges: list[EdgeItem] = Field(default_factory=list)

    def add_node(self, node):
        if isinstance(node, FactNode):
            self.facts.append(node)
        elif isinstance(node, StepNode):
            self.steps.append(node)
        elif isinstance(node, OSIRefNode):
            self.osi_refs.append(node)
        elif isinstance(node, SourceNode):
            self.sources.append(node)

    def add_edge(self, edge):
        self.edges.append(edge)

    def merge(self, other: "ReasoningGraphDoc"):
        by_id = {}
        for lst in [self.facts, self.steps, self.osi_refs, self.sources]:
            for n in lst:
                by_id[n.id] = n
        for lst in [other.facts, other.steps, other.osi_refs, other.sources]:
            for n in lst:
                by_id[n.id] = n
        self.facts = [n for n in by_id.values() if isinstance(n, FactNode)]
        self.steps = [n for n in by_id.values() if isinstance(n, StepNode)]
        self.osi_refs = [n for n in by_id.values() if isinstance(n, OSIRefNode)]
        self.sources = [n for n in by_id.values() if isinstance(n, SourceNode)]
        self.edges.extend(other.edges)
