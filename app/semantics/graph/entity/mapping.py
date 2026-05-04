"""Data mapping contract — how each entity/edge is derived from database tables."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


# ── Node source types ──

class TableSource(BaseModel):
    type: Literal["table"] = "table"
    table: str = Field(..., description="Source table name")
    key_columns: list[str] = Field(
        default_factory=list,
        description="Column(s) forming the entity's key. Use multiple for composite PKs.",
    )
    # Backward compat: accept single string
    key_column: str | None = Field(default=None, exclude=True)

    def get_key_columns(self) -> list[str]:
        if self.key_columns:
            return self.key_columns
        if self.key_column:
            return [self.key_column]
        return []

    def get_key(self, row: dict) -> str:
        cols = self.get_key_columns()
        return "|".join(str(row[c]) for c in cols)

class JoinSource(BaseModel):
    type_: Literal["join"] = Field(default="join", alias="type")
    base_table: str
    joins: list[dict] = Field(default_factory=list, description="[{table, on}] join clauses")
    key_columns: list[str] = Field(default_factory=list)
    key_column: str | None = Field(default=None, exclude=True)

    def get_key_columns(self) -> list[str]:
        if self.key_columns:
            return self.key_columns
        if self.key_column:
            return [self.key_column]
        return []

    def get_key(self, row: dict) -> str:
        cols = self.get_key_columns()
        return "|".join(str(row[c]) for c in cols)

class UnionSourceEntry(BaseModel):
    table: str
    key_columns: list[str] = Field(default_factory=list)
    key_column: str | None = Field(default=None, exclude=True)

    def get_key_columns(self) -> list[str]:
        if self.key_columns:
            return self.key_columns
        if self.key_column:
            return [self.key_column]
        return []

    def get_key(self, row: dict) -> str:
        cols = self.get_key_columns()
        return "|".join(str(row[c]) for c in cols)

class UnionSource(BaseModel):
    type_: Literal["union"] = Field(default="union", alias="type")
    sources: list[UnionSourceEntry] = Field(default_factory=list)


# ── Entity mapping ──

class EntityMapping(BaseModel):
    entity: str = Field(..., description="Matches EntityDef.label")
    node_source: TableSource | JoinSource | UnionSource
    properties: dict[str, str] = Field(
        default_factory=dict,
        description="Property name → column reference (or '${key}' for key_column value)"
    )
    strong_parents: dict[str, str] = Field(
        default_factory=dict,
        description="Parent entity label → FK column in the child's source table"
    )


# ── Edge mapping ──

class EdgeEndpoint(BaseModel):
    entity: str
    key_column: str | None = Field(default=None, description="Column providing the entity key")

class EdgeRoute(BaseModel):
    label: str
    direction: str = Field(..., description="'A -> B'")
    bind_column: str | None = None
    role: str | None = None

class EdgeMapping(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    label: str = Field(..., description="Matches RelationDef.label")
    from_: EdgeEndpoint = Field(..., alias="from")
    to: EdgeEndpoint
    condition: str | None = Field(default=None, description="e.g. '${amount} > 10000'")
    routes: list[EdgeRoute] | None = Field(default=None, description="Multi-route edges")
    type: str | None = Field(default=None, description="'event_decompose' for one-row→multi-edge")
    role_column: str | None = Field(default=None, description="Column determining edge label variant")
    key_column: str | None = Field(default=None, description="Column providing the target key")


# ── Top-level mapping ──

class DataMapping(BaseModel):
    version: str = Field(default="0.1.0")
    model: str = Field(default="", description="Semantic model name")
    entities: list[EntityMapping] = Field(default_factory=list)
    edges: list[EdgeMapping] = Field(default_factory=list)

    def entity_keys(self) -> set[str]:
        return {e.entity for e in self.entities}

    def collect_table_columns(self) -> dict[str, set[str]]:
        """Collect all table→columns references for validation."""
        result: dict[str, set[str]] = {}
        for em in self.entities:
            ns = em.node_source
            if isinstance(ns, TableSource):
                result.setdefault(ns.table, set()).add(ns.key_column)
            elif isinstance(ns, UnionSource):
                for s in ns.sources:
                    result.setdefault(s.table, set()).add(s.key_column)
            elif isinstance(ns, JoinSource):
                result.setdefault(ns.base_table, set()).add(ns.key_column)
                for j in ns.joins:
                    result.setdefault(j["table"], set())
            for prop_col in em.properties.values():
                if "." in prop_col and not prop_col.startswith("$"):
                    t, c = prop_col.split(".", 1)
                    result.setdefault(t, set()).add(c)
        return result
