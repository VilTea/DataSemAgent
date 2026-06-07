"""BIRD Mini-Dev task loading — HuggingFace dataset + database files."""
from __future__ import annotations

import json
import os
import sqlite3
import zipfile
from pathlib import Path
from typing import Any

import requests

_BIRD_DATA = Path(__file__).resolve().parent.parent.parent.parent / "data" / "bird"
_DB_DIR = _BIRD_DATA / "dev_databases"
_DB_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"


def load_tasks(
    difficulty: str | None = None,
    db_ids: list[str] | None = None,
    max_tasks: int | None = None,
) -> list[dict[str, Any]]:
    """Load BIRD Mini-Dev tasks from HuggingFace.

    Returns task dicts with keys: question_id, db_id, question, SQL, evidence,
    difficulty.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets library required. pip install datasets")

    ds = load_dataset("birdsql/bird_mini_dev", "default", split="mini_dev_sqlite")
    tasks: list[dict[str, Any]] = []
    for row in ds:
        d = row.get("difficulty", "")
        db = row.get("db_id", "")
        if difficulty and d != difficulty:
            continue
        if db_ids and db not in db_ids:
            continue
        tasks.append({
            "question_id": int(row["question_id"]),
            "db_id": db,
            "question": row["question"],
            "SQL": row["SQL"],
            "evidence": row.get("evidence", ""),
            "difficulty": d,
        })
    if max_tasks:
        tasks = tasks[:max_tasks]
    return tasks


def ensure_databases(db_ids: set[str] | None = None) -> dict[str, str]:
    """Download & extract BIRD databases.  Returns {db_id: sqlite_path}.

    Only downloads databases present in *db_ids* (or all 11 if None).
    Cached — skips databases already on disk.
    """
    _DB_DIR.mkdir(parents=True, exist_ok=True)

    # Discover which databases are already available
    available: dict[str, str] = {}
    for entry in _DB_DIR.iterdir():
        if entry.is_dir():
            db_name = entry.name
            sqlite_path = entry / f"{db_name}.sqlite"
            if sqlite_path.exists():
                available[db_name] = str(sqlite_path)

    needed = db_ids - set(available.keys()) if db_ids else None

    if needed and not _zip_downloaded():
        _download_dev_zip()

    if needed:
        _extract_databases(needed)

    # Re-scan after extraction
    for entry in _DB_DIR.iterdir():
        if entry.is_dir():
            db_name = entry.name
            if db_name not in available:
                sqlite_path = entry / f"{db_name}.sqlite"
                if sqlite_path.exists():
                    available[db_name] = str(sqlite_path)

    return available


def _zip_downloaded() -> bool:
    return (_BIRD_DATA / "dev.zip").exists()


def _download_dev_zip() -> None:
    """Download dev.zip (~2 GB) from Alibaba OSS."""
    dest = _BIRD_DATA / "dev.zip"
    print(f"Downloading {_DB_URL} ...")
    with requests.get(_DB_URL, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 // total
                    print(f"\r  {pct}% ({downloaded}/{total})", end="", flush=True)
        print()


def _extract_databases(db_ids: set[str]) -> None:
    """Extract specific databases from the nested dev_databases.zip inside dev.zip."""
    # BIRD dev.zip wraps the real database archive: dev_*/dev_databases.zip
    with zipfile.ZipFile(_BIRD_DATA / "dev.zip", "r") as outer:
        inner_name = None
        for name in outer.namelist():
            if name.endswith("dev_databases.zip"):
                inner_name = name
                break
        if not inner_name:
            print("  dev_databases.zip not found inside dev.zip")
            return

        # Extract inner zip to memory, then extract individual databases
        inner_data = outer.read(inner_name)
        import io
        with zipfile.ZipFile(io.BytesIO(inner_data), "r") as inner:
            for name in inner.namelist():
                # Paths like: debit_card_specializing/database_description/...
                #             debit_card_specializing/debit_card_specializing.sqlite
                parts = name.split("/")
                if len(parts) < 3:
                    continue
                db_name = parts[1]  # dev_databases/db_name/...
                if db_name not in db_ids:
                    continue
                # Strip "dev_databases/" prefix from the zip path
                rel = name[len("dev_databases/"):] if name.startswith("dev_databases/") else name
                target = _DB_DIR / rel
                if target.exists():
                    continue
                if name.endswith(".sqlite"):
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with open(target, "wb") as dst:
                        dst.write(inner.read(name))
                    print(f"  Extracted: {db_name}/{os.path.basename(name)}")


def get_db_info(db_path: str) -> dict[str, Any]:
    """Read schema info from a BIRD SQLite database.

    Returns a dict with table names as keys, each containing column info.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    tables = [row[0] for row in cursor.fetchall()]

    schema: dict[str, list[dict]] = {}
    for table in tables:
        cursor.execute(f"PRAGMA table_info(\"{table}\")")
        columns = []
        for row in cursor.fetchall():
            columns.append({
                "name": row[1],
                "type": row[2],
                "nullable": not row[3],
                "pk": bool(row[5]),
            })
        schema[table] = columns

    conn.close()
    return schema
