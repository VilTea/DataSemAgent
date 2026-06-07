"""BIRD scoring — Execution Accuracy (EX) via result-set comparison."""
from __future__ import annotations

import sqlite3


def score(predicted_sql: str, gold_sql: str, db_path: str) -> tuple[bool, str]:
    """Execute both SQL statements and compare result sets.

    Returns (passed, details) where details explains mismatches.
    Both queries run on the same database; result sets are compared
    value-by-value, order-independent.
    """
    try:
        conn = sqlite3.connect(db_path)
        gold_rows, gold_cols = _execute(conn, gold_sql)
        pred_rows, pred_cols = _execute(conn, predicted_sql)
        conn.close()

        # Compare column count
        if len(gold_cols) != len(pred_cols):
            return False, (
                f"Column count mismatch: gold={len(gold_cols)} ({gold_cols}), "
                f"pred={len(pred_cols)} ({pred_cols})"
            )

        # Compare row count
        if len(gold_rows) != len(pred_rows):
            return False, (
                f"Row count mismatch: gold={len(gold_rows)}, pred={len(pred_rows)}"
            )

        # Normalize values for comparison
        gold_set = _normalize_rows(gold_rows)
        pred_set = _normalize_rows(pred_rows)

        if gold_set != pred_set:
            # Find differences
            only_gold = gold_set - pred_set
            only_pred = pred_set - gold_set
            detail = f"Result mismatch"
            if only_gold:
                detail += f" | only in gold: {list(only_gold)[:3]}"
            if only_pred:
                detail += f" | only in pred: {list(only_pred)[:3]}"
            return False, detail

        return True, ""

    except Exception as e:
        return False, f"Execution error: {e}"


def _execute(conn: sqlite3.Connection, sql: str) -> tuple[list[tuple], list[str]]:
    cursor = conn.execute(sql)
    cols = [d[0] for d in cursor.description] if cursor.description else []
    rows = cursor.fetchall()
    return rows, cols


def _normalize_rows(rows: list[tuple]) -> set[tuple]:
    """Normalize row values for comparison — order rows for deterministic sets."""
    normalized = []
    for row in rows:
        norm = tuple(
            round(float(v), 6) if isinstance(v, float) else v
            for v in row
        )
        normalized.append(norm)
    # Sort for deterministic comparison (result order may differ)
    try:
        return set(normalized)
    except TypeError:
        # Unhashable types — compare as sorted list
        return set(str(r) for r in normalized)
