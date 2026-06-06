"""DABstep task loading via Hugging Face datasets."""
from __future__ import annotations

from typing import Any

_TASK_FIELDS = ("task_id", "question", "answer", "guidelines", "level")


def load_tasks(
    split: str = "default",
    level: str | None = None,
    task_ids: list[int] | None = None,
    max_tasks: int | None = None,
) -> list[dict[str, Any]]:
    """Load DABstep tasks from Hugging Face datasets.

    Each task dict includes an *answer_type* key inferred from guidelines.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets library required. pip install datasets")

    ds = load_dataset("adyen/DABstep", name="tasks", split=split)
    tasks: list[dict[str, Any]] = []
    for row in ds:
        t = {f: row.get(f) for f in _TASK_FIELDS}
        t["task_id"] = int(t["task_id"])
        t["answer_type"] = _infer_answer_type(str(row.get("guidelines", "")))
        if level and t.get("level") != level:
            continue
        if task_ids and t["task_id"] not in task_ids:
            continue
        tasks.append(t)
    if max_tasks:
        tasks = tasks[:max_tasks]
    return tasks


def load_ground_truth(
    task_ids: set[int] | None = None,
    max_scan: int = 200000,
) -> dict[int, str]:
    """Load ground-truth answers by scanning task_scores stream.

    DABstep keeps ground truth private; we find correct answers by locating
    rows with ``score=True`` for each *task_id*.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        return {}

    target = task_ids or set()
    if not target:
        return {}

    try:
        ds = load_dataset("adyen/DABstep", name="task_scores",
                          split="default", streaming=True)
        gt: dict[int, str] = {}
        for i, row in enumerate(ds):
            if i >= max_scan:
                break
            tid = int(row.get("task_id", 0))
            if tid not in target or tid in gt:
                continue
            score_val = row.get("score")
            if str(score_val).lower() == "true":
                answer = str(row.get("agent_answer", ""))
                if answer:
                    gt[tid] = answer
                    if len(gt) >= len(target):
                        break
        return gt
    except Exception:
        return {}


def _infer_answer_type(guidelines: str) -> str:
    """Infer the expected answer_type from DABstep guidelines text."""
    g = guidelines.lower()
    if "group" in g and ("broken down" in g or "grouping" in g):
        return "grouped_list"
    if "list" in g or "comma separated" in g:
        return "list"
    if "number" in g or "decimal" in g:
        return "number"
    if "yes or no" in g or "yes/no" in g:
        return "text"
    return "not_applicable"
