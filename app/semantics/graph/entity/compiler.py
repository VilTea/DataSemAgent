"""Graph compiler — reads relational data via SqlExecutor, produces GraphDocument."""
from app.db.base import SqlExecutor
from app.semantics.graph.contract import GraphDocument
from app.semantics.graph.entity.mapping import (
    DataMapping,
    EdgeMapping,
    EntityMapping,
    JoinSource,
    TableSource,
    UnionSource,
)
from app.semantics.models import SemanticModel

_UNLIMITED = 999999


class GraphCompiler:
    """Compiles entity/edge mappings into a GraphDocument by reading from a database.

    Database-agnostic — uses SqlExecutor for reads, produces GraphDocument.
    The caller is responsible for loading the document via a GraphLoader.
    """

    def __init__(self, mapping: DataMapping, executor: SqlExecutor,
                 model: SemanticModel | None = None):
        self._mapping = mapping
        self._executor = executor
        # OSI logical name → physical column reference map.
        self._logical_to_phys: dict[str, str] = {}
        if model is not None:
            for ds in model.datasets:
                for f in (ds.fields or []):
                    phys = f.expression.dialects[0].expression if f.expression.dialects else f.name
                    self._logical_to_phys[f.name] = phys

    async def build(self) -> GraphDocument:
        doc = GraphDocument()
        pending_weak = await self._build_nodes(doc)
        await self._resolve_orphans(doc, pending_weak)
        await self._build_edges(doc)
        return doc

    # ------------------------------------------------------------------ #
    #  Phase 1 — nodes
    # ------------------------------------------------------------------ #

    @staticmethod
    def _node_id(entity: str, key_val: str) -> str:
        return f"{entity}:{key_val}"

    async def _build_nodes(self, doc: GraphDocument) -> set[tuple[str, str, str]]:
        ordered = sorted(
            self._mapping.entities,
            key=lambda e: (1 if e.strong_parents else 0, e.entity),
        )
        existing_keys: dict[str, set[str]] = {e.entity: set() for e in self._mapping.entities}
        pending: set[tuple[str, str, str]] = set()

        for em in ordered:
            ns = em.node_source
            if isinstance(ns, TableSource):
                rows, _ = await self._executor.execute(
                    f"SELECT * FROM {ns.table}", limit=_UNLIMITED
                )
                for row in rows:
                    key_val = ns.get_key(row)
                    if key_val in existing_keys[em.entity]:
                        raise RuntimeError(
                            f"Duplicate key '{key_val}' for entity '{em.entity}' "
                            f"from table '{ns.table}' columns {ns.get_key_columns()}. "
                            f"Choose unique key column(s)."
                        )
                    existing_keys[em.entity].add(key_val)
                    node_id = self._node_id(em.entity, key_val)
                    doc.add_node(node_id, em.entity, **self._build_props(em, row, key_val))

                    for parent_label, fk_col in em.strong_parents.items():
                        parent_key_val = str(row.get(fk_col, ""))
                        if parent_key_val and parent_key_val not in existing_keys.get(parent_label, set()):
                            pending.add((em.entity, parent_label, parent_key_val))
            elif isinstance(ns, UnionSource):
                for src in ns.sources:
                    rows, _ = await self._executor.execute(
                        f"SELECT * FROM {src.table}", limit=_UNLIMITED
                    )
                    for row in rows:
                        key_val = src.get_key(row)
                        if key_val in existing_keys[em.entity]:
                            raise RuntimeError(
                                f"Duplicate key '{key_val}' for entity '{em.entity}' "
                                f"from table '{src.table}' columns {src.get_key_columns()}."
                            )
                        existing_keys[em.entity].add(key_val)
                        node_id = self._node_id(em.entity, key_val)
                        doc.add_node(node_id, em.entity, **self._build_props(em, row, key_val))
        return pending

    # ------------------------------------------------------------------ #
    #  Phase 2a — orphans
    # ------------------------------------------------------------------ #

    async def _resolve_orphans(self, doc: GraphDocument, pending: set[tuple[str, str, str]]) -> None:
        existing = {n.id for n in doc.nodes}
        seen_unknown: set[tuple[str, str]] = set()
        for child_entity, parent_label, parent_key_val in pending:
            parent_id = self._node_id(parent_label, parent_key_val)
            if parent_id not in existing:
                key = (parent_label, parent_key_val)
                if key not in seen_unknown:
                    doc.add_node(parent_id, parent_label, name=f"Unknown_{parent_label}")
                    seen_unknown.add(key)

    # ------------------------------------------------------------------ #
    #  Phase 2b — edges
    # ------------------------------------------------------------------ #

    async def _build_edges(self, doc: GraphDocument) -> None:
        for em in self._mapping.edges:
            table = self._find_edge_table(em)
            if table is None:
                continue
            from_entity_def = None
            for e in self._mapping.entities:
                if e.entity == em.from_.entity:
                    from_entity_def = e
                    break
            if from_entity_def is None:
                continue
            rows, _ = await self._executor.execute(
                f"SELECT * FROM {table}", limit=_UNLIMITED
            )
            for row in rows:
                if em.condition and not self._eval_condition(em.condition, row):
                    continue
                raw_from_key = from_entity_def.node_source.get_key(row)
                raw_to_key = str(row.get(em.to.key_column or "", ""))
                if raw_from_key and raw_to_key:
                    doc.add_edge(
                        self._node_id(em.from_.entity, raw_from_key),
                        self._node_id(em.to.entity, raw_to_key),
                        em.label,
                    )

    def _find_edge_table(self, edge: EdgeMapping) -> str | None:
        for em in self._mapping.entities:
            if em.entity == edge.from_.entity:
                ns = em.node_source
                if isinstance(ns, TableSource):
                    return ns.table
                if isinstance(ns, JoinSource):
                    return ns.base_table
                if isinstance(ns, UnionSource) and ns.sources:
                    return ns.sources[0].table
        return None

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _build_props(self, em: EntityMapping, row: dict, key_val: str) -> dict:
        props = {}
        for prop_name, col_ref in em.properties.items():
            if col_ref == "${key}":
                props[prop_name] = key_val
                continue
            # Resolve OSI logical name → physical column name, then
            # physical → row value (row keys are physical from SELECT *).
            phys_col = self._logical_to_phys.get(col_ref, col_ref)
            if phys_col in row:
                props[prop_name] = row[phys_col]
                continue
            if "." in col_ref:
                _, col = col_ref.split(".", 1)
                phys_col2 = self._logical_to_phys.get(col, col)
                if phys_col2 in row:
                    props[prop_name] = row[phys_col2]
        return props

    @staticmethod
    def _eval_condition(condition: str, row: dict) -> bool:
        expr = condition
        for k, v in row.items():
            expr = expr.replace("${" + k + "}", str(v))
        try:
            return eval(expr)
        except Exception:
            return True
