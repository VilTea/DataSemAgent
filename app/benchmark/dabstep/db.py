"""One-time ETL: DABstep context files → SQLite database."""
import csv
import json
import sqlite3
from pathlib import Path
from typing import Any


_DABSTEP_DATA = Path(__file__).resolve().parent.parent.parent.parent / "data" / "dabstep"
_DB_PATH = _DABSTEP_DATA / "benchmark.db"


def ensure_tables(context_dir: str) -> str:
    """Create SQLite tables from DABstep context files.  Idempotent.

    Returns the path to the SQLite database file.
    """
    ctx = Path(context_dir)
    if not ctx.exists():
        raise FileNotFoundError(f"DABstep context directory not found: {context_dir}")

    _DABSTEP_DATA.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH))

    _load_csv(conn, ctx / "payments.csv", "payments")
    _load_json(conn, ctx / "fees.json", "fees")
    _load_json(conn, ctx / "merchant_data.json", "merchants")
    _load_csv(conn, ctx / "merchant_category_codes.csv", "mcc_codes")
    _load_csv(conn, ctx / "acquirer_countries.csv", "acquirer_countries")

    conn.commit()
    conn.close()
    return str(_DB_PATH)


def _infer_type(values: list) -> str:
    for v in values:
        if v is None or v == "":
            continue
        if isinstance(v, (list, dict)):
            return "TEXT"
        if isinstance(v, bool):
            continue
        try:
            int(v)
            continue
        except (ValueError, TypeError):
            pass
        try:
            float(v)
            return "REAL"
        except (ValueError, TypeError):
            return "TEXT"
    return "INTEGER"


def _load_csv(conn: sqlite3.Connection, path: Path, table: str) -> None:
    if not path.exists():
        return
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not rows:
            return
        columns, col_types = _schema_from_rows(rows)
        cols_ddl = ", ".join(f'"{c}" {t}' for c, t in zip(columns, col_types))
        conn.execute(f'CREATE TABLE IF NOT EXISTS "{table}" ({cols_ddl})')
        conn.execute(f'DELETE FROM "{table}"')
        placeholders = ", ".join("?" for _ in columns)
        conn.executemany(
            f'INSERT INTO "{table}" ({", ".join(f"\"{c}\"" for c in columns)}) VALUES ({placeholders})',
            [tuple(_coerce(r.get(c, ""), t) for c, t in zip(columns, col_types)) for r in rows],
        )


def _load_json(conn: sqlite3.Connection, path: Path, table: str) -> None:
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    if not data:
        return
    columns, col_types = _schema_from_rows(data)
    cols_ddl = ", ".join(f'"{c}" {t}' for c, t in zip(columns, col_types))
    conn.execute(f'CREATE TABLE IF NOT EXISTS "{table}" ({cols_ddl})')
    conn.execute(f'DELETE FROM "{table}"')
    placeholders = ", ".join("?" for _ in columns)
    conn.executemany(
        f'INSERT INTO "{table}" ({", ".join(f"\"{c}\"" for c in columns)}) VALUES ({placeholders})',
        [tuple(_coerce(r.get(c, ""), t) for c, t in zip(columns, col_types)) for r in data],
    )


def _schema_from_rows(rows: list[dict]) -> tuple[list[str], list[str]]:
    columns = list(rows[0].keys())
    col_types = []
    for c in columns:
        samples = [r.get(c) for r in rows[:100]]
        col_types.append(_infer_type(samples))
    return columns, col_types


def _coerce(value: Any, sql_type: str) -> Any:
    if value is None or value == "":
        return None
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    if isinstance(value, bool):
        return int(value)
    if sql_type == "INTEGER":
        try:
            return int(value)
        except (ValueError, TypeError):
            return value
    if sql_type == "REAL":
        try:
            return float(value)
        except (ValueError, TypeError):
            return value
    return str(value)
