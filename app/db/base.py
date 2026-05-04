import re
from pathlib import Path
from typing import runtime_checkable, Protocol, Any, ClassVar

import aiosqlite
import sqlglot
import sqlglot.expressions as exp
from pydantic import BaseModel, ConfigDict, Field
from sqlglot.dialects.dialect import DialectType
from sqlglot.dialects.sqlite import SQLite

from app.exceptions import QueryNotAllowedError


@runtime_checkable
class SqlExecutor(Protocol):
    """Protocol for SQL database executors.

    Callers use the ``dialect`` attribute to determine which sqlglot dialect
    the target database speaks, so SQL can be translated to the correct dialect
    **before** being passed to ``execute()``.

    Usage::

        translated = sqlglot.transpile(sql, read="ansi", write=executor.dialect)[0]
        rows, total = await executor.execute(translated)
    """

    db_type: ClassVar[str]
    driver: ClassVar[str]
    dialect: ClassVar[DialectType]

    async def execute(self, sql: str, limit: int = 100) -> tuple[list[dict[str, Any]], int]: ...
    async def get_table_info(self, table_name: str) -> dict[str, str]: ...


class AioSqlLiteExecutor(BaseModel):
    """SQLite executor backed by ``aiosqlite``.

    The ``dialect`` is set to ``sqlite`` so callers can translate ANSI SQL
    into SQLite-compatible syntax before execution.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    db_type: ClassVar[str] = "sqlite"
    driver: ClassVar[str] = "aiosqlite"
    dialect: ClassVar[DialectType] = SQLite

    path: str | Path = Field(default="data/app.db", description="Database file path")
    timeout: float = Field(default=30.0, description="Connection timeout in seconds")
    check_same_thread: bool = Field(default=True, description="Whether to check same thread")

    async def execute(self, sql: str, limit: int = 100) -> tuple[list[dict[str, Any]], int]:
        parsed = sqlglot.parse_one(sql)
        if not isinstance(parsed, exp.Query):
            raise QueryNotAllowedError(
                f"Only query statements are allowed, got: {type(parsed).__name__}"
            )

        async with aiosqlite.connect(
                str(self.path),
                timeout=self.timeout,
                check_same_thread=self.check_same_thread,
        ) as conn:
            conn.row_factory = aiosqlite.Row

            total = 0
            if not parsed.find(exp.Limit):
                total = await self.count_total(conn, parsed)

            async with conn.execute(sql) as cursor:
                if limit <= 0:
                    rows = await cursor.fetchall()
                else:
                    rows = await cursor.fetchmany(size=limit)
                result = [dict(row) for row in rows]
                total = total or len(result)

            return result, total

    async def get_table_info(self, table_name: str) -> dict[str, str]:
        """Return {column_name: sqlite_type} for *table_name* via PRAGMA."""
        async with aiosqlite.connect(
                str(self.path),
                timeout=self.timeout,
                check_same_thread=self.check_same_thread,
        ) as conn:
            conn.row_factory = aiosqlite.Row
            async with conn.execute(f"PRAGMA table_info('{table_name}')") as cursor:
                rows = await cursor.fetchall()
                return {row["name"]: row["type"].upper() for row in rows}

    async def count_total(self, conn: aiosqlite.Connection, parsed: exp.Query) -> int:
        """Return total row count for *sql* by wrapping it as a subquery."""
        try:
            count_ast = exp.select("COUNT(*)").from_(parsed.subquery("_t"))
            count_sql = count_ast.sql(dialect=self.dialect)
        except Exception:
            # Fallback: naive COUNT(*) replacement
            count_sql = re.sub(
                r"^SELECT\s+.*?\s+FROM\s+",
                "SELECT COUNT(*) FROM ",
                parsed.sql(dialect=self.dialect),
                count=1,
                flags=re.IGNORECASE,
            )

        async with conn.execute(count_sql) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else 0


def create_sql_executor(db_config_key: str = "default") -> SqlExecutor:
    """Factory: create a SqlExecutor from config."""
    from app.config import config
    db_cfg = config.database[db_config_key]
    if db_cfg.type == "sqlite":
        return AioSqlLiteExecutor(path=str(db_cfg.specific.path))
    raise ValueError(f"Unsupported database type: {db_cfg.type}")
