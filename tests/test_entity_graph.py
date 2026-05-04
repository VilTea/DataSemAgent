"""Integration tests for entity graph pipeline (compiler)."""
import asyncio
import sqlite3

import pytest

from app.semantics.graph.entity.mapping import (
    DataMapping, EdgeEndpoint, EdgeMapping, EntityMapping, TableSource,
)
from app.db.base import AioSqlLiteExecutor
from app.semantics.graph.entity.compiler import GraphCompiler
from app.semantics.graph.loader import KuzuLoader


@pytest.fixture
def sample_db(tmp_path):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE users (id INTEGER, name TEXT);
        CREATE TABLE cards (card_id TEXT, owner_id INTEGER);
        INSERT INTO users VALUES (1, 'Alice'), (2, 'Bob');
        INSERT INTO cards VALUES ('C001', 1), ('C002', 2), ('C003', 999);
    """)
    conn.commit()
    conn.close()
    return db_path


class TestGraphCompiler:
    def test_compile_nodes_and_edges(self, sample_db, tmp_path):
        import uuid
        mapping = DataMapping(
            entities=[
                EntityMapping(entity="User", node_source=TableSource(table="users", key_column="id"),
                              properties={"name": "name"}),
                EntityMapping(entity="BankCard", node_source=TableSource(table="cards", key_column="card_id"),
                              properties={"card_number": "${key}"}, strong_parents={"User": "owner_id"}),
            ],
            edges=[
                EdgeMapping(label="OWNS", from_=EdgeEndpoint(entity="User", key_column="owner_id"),
                            to=EdgeEndpoint(entity="BankCard", key_column="card_id")),
            ],
        )

        executor = AioSqlLiteExecutor(path=str(sample_db))
        compiler = GraphCompiler(mapping, executor)
        doc = asyncio.run(compiler.build())

        ex = KuzuLoader(path=str(tmp_path / "eg" / uuid.uuid4().hex[:8]))
        ex.load(doc)

        r = ex.execute("MATCH (u:User) RETURN COUNT(u) AS cnt")
        assert _one(r) == 3
        r = ex.execute("MATCH (b:BankCard) RETURN COUNT(b) AS cnt")
        assert _one(r) == 3
        r = ex.execute("MATCH (u:User) WHERE u.name = 'Unknown_User' RETURN u.id, u.name")
        rows = _all(r)
        assert len(rows) >= 1
        assert any("Unknown_User" in str(r) for r in rows)
        ex.close()


def _one(result):
    return result.get_next()[0]


def _all(result):
    rows = []
    while result.has_next():
        rows.append(result.get_next())
    return rows
