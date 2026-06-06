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


def load_ground_truth(split: str = "default") -> dict[int, str]:
    """Load ground-truth answers, returning {task_id: answer_string}."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets library required. pip install datasets")

    try:
        ds = load_dataset("adyen/DABstep", name="task_scores", split=split)
        gt: dict[int, str] = {}
        for row in ds:
            tid = int(row.get("task_id", 0))
            answer = row.get("answer", "")
            if tid and answer:
                gt[tid] = str(answer)
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
    return "not_applicable"
