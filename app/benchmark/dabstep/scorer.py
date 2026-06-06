"""DABstep answer scoring — normalization + exact match."""
from __future__ import annotations

import re


def normalize_answer(raw: str, answer_type: str) -> str:
    """Normalize a raw answer string for comparison."""
    raw = raw.strip()
    if raw.lower() == "not applicable":
        return "not_applicable"
    if answer_type == "number":
        return _normalize_number(raw)
    if answer_type == "list":
        return _normalize_list(raw)
    if answer_type == "grouped_list":
        return _normalize_grouped(raw)
    if answer_type == "text":
        return _normalize_text(raw)
    return raw


def _normalize_text(raw: str) -> str:
    return raw.strip().lower()


def score(predicted: str, expected: str, answer_type: str) -> bool:
    pred = normalize_answer(predicted, answer_type)
    exp = normalize_answer(expected, answer_type)
    if pred == exp:
        return True
    if answer_type == "number":
        return _fuzzy_number_match(pred, exp)
    return False


def _fuzzy_number_match(pred: str, exp: str, rel_tol: float = 1e-5) -> bool:
    """Fallback: compare as floats with relative tolerance.

    Some tasks say 'round to 6 decimals' but the ground truth has full
    precision; others have minor floating-point divergence.  This lets
    values that differ only in the 6+th significant digit still match.
    """
    try:
        p = float(pred)
        e = float(exp)
    except (ValueError, TypeError):
        return False
    if e == 0:
        return abs(p) < rel_tol
    return abs(p - e) / abs(e) < rel_tol


def _normalize_number(raw: str) -> str:
    # Strip trailing % sign (LLM sometimes omits it)
    raw = raw.strip().rstrip("%")
    nums = re.findall(r"[-+]?\d*\.?\d+", raw)
    if not nums:
        return raw
    val = float(nums[-1])
    if val == int(val):
        return str(int(val))
    # Strip trailing zeros: 2.50 -> 2.5, 73.150 -> 73.15
    s = f"{val:.10f}".rstrip("0").rstrip(".")
    return s


def _normalize_list(raw: str) -> str:
    items = re.split(r"[,;\s]+", raw.strip())
    items = [i.strip() for i in items if i.strip()]
    return ", ".join(sorted(items))


def _normalize_grouped(raw: str) -> str:
    pairs = []
    for part in re.split(r"[;]", raw):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"([^:]+):\s*([-+]?\d*\.?\d+)", part)
        if m:
            pairs.append((m.group(1).strip(), float(m.group(2))))
    pairs.sort(key=lambda x: (x[1], x[0]))
    return ",".join(f"{g}:{v}" for g, v in pairs)
