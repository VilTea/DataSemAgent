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
    return raw


def score(predicted: str, expected: str, answer_type: str) -> bool:
    pred = normalize_answer(predicted, answer_type)
    exp = normalize_answer(expected, answer_type)
    return pred == exp


def _normalize_number(raw: str) -> str:
    nums = re.findall(r"[-+]?\d*\.?\d+", raw)
    if not nums:
        return raw
    return nums[-1]


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
