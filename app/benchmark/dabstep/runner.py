"""BenchmarkRunner — orchestrates DABstep tasks through the agent."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.benchmark.base import BenchmarkReport
from app.hook import HookPoint
from app.tool.sql_exec import SqlExecTool

from .db import ensure_tables
from .model import build_model
from .scorer import score
from .task_loader import load_tasks
from .tool import SubmitAnswerTool

_DABSTEP_SYSTEM_PROMPT = """\
You are a financial data analyst. Answer the question using the available data.

Use sql_exec to query the database. The database contains:
- payments: transaction records
- fees: fee schedules
- merchants: merchant metadata
- mcc_codes: merchant category codes
- acquirer_countries: country codes

When you have the final answer, call submit_answer with the appropriate
answer_type. Read the task guidelines carefully to choose the right format.
"""

_REPORT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "dabstep"


@dataclass
class TaskResult:
    task_id: int
    question: str
    level: str
    answer_type: str = ""
    guidelines: str = ""
    predicted: str = ""
    expected: str = ""
    passed: bool = False
    duration_ms: float = 0
    error: str = ""
    trace_file: str = ""


class BenchmarkRunner:
    """Runs DABstep tasks through the DataSemAgent and scores results."""

    def __init__(self, context_dir: str, llm_config: str = "default"):
        self._context_dir = context_dir
        self._llm_config = llm_config

    async def run(
        self,
        task_ids: list[int] | None = None,
        level: str | None = None,
        max_tasks: int | None = None,
    ) -> BenchmarkReport:
        db_path = ensure_tables(self._context_dir)
        model = build_model()
        tasks = load_tasks(task_ids=task_ids, level=level, max_tasks=max_tasks)
        ground_truth: dict[int, str] = {}

        report = BenchmarkReport()
        t0 = time.perf_counter()

        for task in tasks:
            result = await self._run_single(task, model, ground_truth)
            report.results.append(result)

        report.total_duration_ms = (time.perf_counter() - t0) * 1000
        self._save_report(report)
        return report

    def _save_report(self, report: BenchmarkReport) -> None:
        _REPORT_DIR.mkdir(parents=True, exist_ok=True)
        path = _REPORT_DIR / "report.json"
        data = {
            "accuracy": report.accuracy,
            "passed": report.passed,
            "total": report.total,
            "duration_s": report.total_duration_ms / 1000,
            "tasks": [
                {
                    "task_id": r.task_id,
                    "level": r.level,
                    "answer_type": r.answer_type,
                    "question": r.question,
                    "guidelines": r.guidelines,
                    "predicted": r.predicted,
                    "expected": r.expected,
                    "passed": r.passed,
                    "duration_s": r.duration_ms / 1000,
                    "error": r.error,
                    "trace_file": r.trace_file,
                }
                for r in report.results
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        print(f"Report saved to {path}")

    async def _run_single(
        self, task: dict, model, ground_truth: dict[int, str],
    ) -> TaskResult:
        result = TaskResult(
            task_id=task["task_id"],
            question=task["question"],
            level=task.get("level", ""),
            answer_type=task.get("answer_type", ""),
            guidelines=task.get("guidelines", ""),
            expected=ground_truth.get(task["task_id"], ""),
        )
        t0 = time.perf_counter()

        try:
            from app.flow import react_flow
            from app.node.agent import AgentNode
            from app.pipeline import QueuePipeline
            from app.eval.collector import EvalCollector

            sql_tool = SqlExecTool(model_source=model, db_config_key="dabstep")
            answer_tool = SubmitAnswerTool()

            agent = AgentNode(
                name=self._llm_config,
                system_prompt=_DABSTEP_SYSTEM_PROMPT,
                tools=[sql_tool, answer_tool],
            )

            captured: dict = {}

            def _on_answer(ctx, tool_call, tool, res):
                captured["raw"] = res.content
                captured["session_id"] = getattr(
                    ctx.memory, "_eval_session_id", ""
                ) if hasattr(ctx, 'memory') else ""

            collector = EvalCollector()
            pipeline = QueuePipeline()
            pipeline.register(collector)

            flow = react_flow(agent_node=agent, pipeline=pipeline)
            async with pipeline.bind(flow.context):
                flow.context.hooks.on(
                    HookPoint.TOOL_AFTER, _on_answer,
                    tool_name="submit_answer", on_error="log",
                )
                await flow.context.hooks.emit(HookPoint.FLOW_START, ctx=flow.context)
                await flow._run_async(flow.context.get_shared())

            result.predicted = captured.get("raw", "")
            result.trace_file = getattr(collector, "_session_id", "")
            if result.predicted and result.expected:
                result.passed = score(
                    result.predicted, result.expected,
                    task.get("answer_type", "number"),
                )

        except Exception as e:
            result.error = str(e)

        result.duration_ms = (time.perf_counter() - t0) * 1000
        return result
