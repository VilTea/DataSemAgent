"""Entity graph schema — entity types and relationship types inferred by agent."""
from pydantic import BaseModel, ConfigDict, Field


class EntityDef(BaseModel):
    label: str = Field(..., description="Unique entity label, e.g. 'BankCard'")
    description: str = Field(default="", description="Human-readable description")
    strong_parents: list[str] = Field(default_factory=list, description="Parent entities for weak entities")
    is_weak: bool = Field(default=False, description="Depends on strong_parents for existence")
    is_event: bool = Field(default=False, description="Row decomposes into multiple edges")


class RelationDef(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    label: str = Field(..., description="Unique relation label, e.g. 'TRANSFERRED_FROM'")
    from_: str = Field(..., alias="from", description="Source entity label")
    to: str = Field(..., description="Target entity label")
    role: str | None = Field(default=None, description="Semantic role, e.g. 'source', 'target'")
    description: str = Field(default="")


class EntityGraphSchema(BaseModel):
    entities: list[EntityDef] = Field(default_factory=list)
    relations: list[RelationDef] = Field(default_factory=list)

    def entity_labels(self) -> set[str]:
        return {e.label for e in self.entities}

    def relation_labels(self) -> set[str]:
        return {r.label for r in self.relations}
