"""Graph document contract — Pydantic models for the intermediate graph format."""
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class GraphNode(BaseModel):
    id: str = Field(..., description="Unique node ID, e.g. 'metric:total_sales'")
    label: str = Field(..., description="Node label, e.g. 'Metric', 'Dimension'")
    properties: dict[str, Any] = Field(
        default_factory=dict,
        description="Key-value properties for the node",
    )


class GraphEdge(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    from_: str = Field(..., alias="from", description="Source node ID")
    to: str = Field(..., description="Target node ID")
    label: str = Field(..., description="Edge label, e.g. 'AGGREGATES_FROM'")
    properties: dict[str, Any] = Field(
        default_factory=dict,
        description="Key-value properties for the edge",
    )


class GraphMeta(BaseModel):
    model: str = Field(default="", description="Semantic model name")
    version: str = Field(default="", description="OSI spec version")
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 timestamp",
    )


class GraphDocument(BaseModel):
    meta: GraphMeta = Field(default_factory=GraphMeta)
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)

    def node_ids(self) -> set[str]:
        return {n.id for n in self.nodes}

    def add_node(self, id: str, label: str, **properties) -> GraphNode:
        node = GraphNode(id=id, label=label, properties=properties)
        self.nodes.append(node)
        return node

    def add_edge(self, from_id: str, to_id: str, label: str, **properties) -> GraphEdge:
        edge = GraphEdge(from_=from_id, to=to_id, label=label, properties=properties)
        self.edges.append(edge)
        return edge
