"""Graph database executor — protocol + Kùzu implementation.

Same architectural level as ``SqlExecutor`` in ``app/db/base.py``.
Tools depend on the protocol, config drives the implementation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable, TYPE_CHECKING

import kuzu

from app.config import config, GraphDatabaseType, GraphDatabaseSettings

if TYPE_CHECKING:
    from app.semantics.graph.contract import GraphDocument


@runtime_checkable
class GraphExecutor(Protocol):
    graph_type: ClassVar[str]
    driver: ClassVar[str]

    def load(self, doc: GraphDocument) -> None:
        """Load a graph document, replacing all existing data."""
        ...

    def execute(self, query: str) -> Any:
        """Execute a query and return an iterable of rows."""
        ...

    def close(self) -> None:
        """Release resources."""
        ...


class KuzuExecutor:
    """Kùzu graph database — singleton connection per path, cleared on reload."""

    graph_type: ClassVar[str] = "kuzu"
    driver: ClassVar[str] = "kuzu"

    def __init__(self, path: str = "data/entity_graph"):
        self._path = path
        self._db: kuzu.Database | None = None
        self._conn: kuzu.Connection | None = None

    @property
    def connection(self) -> kuzu.Connection | None:
        return self._conn

    # ── Public API ──

    def load(self, doc: GraphDocument) -> None:
        self._ensure_open()
        self._clear_all()
        self._create_schema(doc)
        self._insert_nodes(doc)
        self._insert_edges(doc)

    def execute(self, query: str) -> Any:
        self._ensure_open()
        return self._conn.execute(query)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        if self._db is not None:
            self._db.close()
            self._db = None

    # ── Connection ──

    def _ensure_open(self) -> None:
        if self._conn is not None:
            return
        if self._path in ("", ":memory:"):
            self._db = kuzu.Database()
        else:
            p = Path(self._path).resolve()
            p.parent.mkdir(parents=True, exist_ok=True)
            self._db = kuzu.Database(str(p))
        self._conn = kuzu.Connection(self._db)

    # ── Schema rebuild ──

    def _clear_all(self) -> None:
        r = self._conn.execute("CALL show_tables() RETURN *")
        tables = []
        while r.has_next():
            row = r.get_next()
            tables.append((row[1], row[2]))
        for name, ttype in tables:
            if ttype == "REL":
                self._conn.execute(f"DROP TABLE {name}")
        for name, ttype in tables:
            if ttype == "NODE":
                self._conn.execute(f"DROP TABLE {name}")

    def _create_schema(self, doc: GraphDocument) -> None:
        node_props: dict[str, dict[str, type]] = {}
        for n in doc.nodes:
            props = node_props.setdefault(n.label, {})
            for k, v in n.properties.items():
                if k == "id":
                    continue
                if v is None:
                    continue
                existing = props.get(k)
                if existing is str:
                    continue  # str is the fallback, anything more specific wins
                vt = type(v)
                if existing is None or (existing is int and vt is float):
                    props[k] = vt
        for label, prop_types in node_props.items():
            cols = ["id STRING"] + [
                f"{k} {self._python_to_kuzu_type(t)}" for k, t in sorted(prop_types.items())
            ]
            self._conn.execute(
                f"CREATE NODE TABLE {label} ({', '.join(cols)}, PRIMARY KEY(id))"
            )
        edge_rels: dict[str, set[tuple[str, str]]] = {}
        edge_props: dict[str, dict[str, type]] = {}
        for e in doc.edges:
            rels = edge_rels.setdefault(e.label, set())
            from_node = next((n for n in doc.nodes if n.id == e.from_), None)
            to_node = next((n for n in doc.nodes if n.id == e.to), None)
            if from_node and to_node:
                rels.add((from_node.label, to_node.label))
            # Collect edge property types
            for k, v in e.properties.items():
                if v is None:
                    continue
                vt = type(v)
                ep = edge_props.setdefault(e.label, {})
                existing = ep.get(k)
                if existing is str:
                    continue
                if existing is None or (existing is int and vt is float):
                    ep[k] = vt
        for label, pairs in edge_rels.items():
            for i, (from_lbl, to_lbl) in enumerate(pairs):
                if i == 0:
                    cols = [f"FROM {from_lbl} TO {to_lbl}"]
                    for k, t in sorted(edge_props.get(label, {}).items()):
                        cols.append(f"{k} {self._python_to_kuzu_type(t)}")
                    self._conn.execute(f"CREATE REL TABLE {label} ({', '.join(cols)})")

    @staticmethod
    def _python_to_kuzu_type(t: type) -> str:
        if t is int:
            return "INT64"
        if t is float:
            return "DOUBLE"
        if t is bool:
            return "BOOL"
        if t is list:
            return "STRING[]"
        return "STRING"

    def _insert_nodes(self, doc: GraphDocument) -> None:
        for node in doc.nodes:
            fields = [f"id: '{self._esc(node.id)}'"]
            for k, v in node.properties.items():
                if k == "id":
                    continue
                if isinstance(v, str):
                    fields.append(f"{k}: '{self._esc(v)}'")
                elif isinstance(v, bool):
                    fields.append(f"{k}: {str(v).lower()}")
                elif isinstance(v, list):
                    items = ", ".join(f"'{self._esc(x)}'" if isinstance(x, str) else str(x) for x in v)
                    fields.append(f"{k}: [{items}]")
                else:
                    fields.append(f"{k}: {v}")
            self._conn.execute(f"CREATE (:{node.label} {{{', '.join(fields)}}})")

    def _insert_edges(self, doc: GraphDocument) -> None:
        for edge in doc.edges:
            if edge.properties:
                prop_strs = []
                for k, v in edge.properties.items():
                    if isinstance(v, str):
                        prop_strs.append(f"{k}: '{self._esc(v)}'")
                    elif isinstance(v, bool):
                        prop_strs.append(f"{k}: {str(v).lower()}")
                    else:
                        prop_strs.append(f"{k}: {v}")
                prop_clause = " {" + ", ".join(prop_strs) + "}"
            else:
                prop_clause = ""
            self._conn.execute(
                f"MATCH (a), (b) "
                f"WHERE a.id = '{self._esc(edge.from_)}' AND b.id = '{self._esc(edge.to)}' "
                f"CREATE (a)-[:{edge.label}{prop_clause}]->(b)"
            )

    @staticmethod
    def _esc(val: str) -> str:
        return val.replace("\\", "\\\\").replace("'", "\\'")


def create_graph_executor(db_config_key: str = "entity") -> GraphExecutor:
    db_cfg: GraphDatabaseSettings = config.graph_database[db_config_key]
    if db_cfg.type == GraphDatabaseType.KUZU and db_cfg.driver == "kuzu":
        return KuzuExecutor(path=str(db_cfg.specific.path))
    raise ValueError(f"Unsupported graph database: {db_cfg.type}/{db_cfg.driver}")
