"""BenchmarkRunner — orchestrates BIRD Mini-Dev tasks through the agent."""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.benchmark.base import BenchmarkReport
from app.hook import HookPoint
from app.tool.sql_exec import SqlExecTool

from .model_builder import build_model
from .scorer import score
from .task_loader import ensure_databases, get_db_info, load_tasks

_BIRD_SYSTEM_PROMPT = """\
You are a data analyst. Answer the question using SQL queries against the database.

Use sql_exec to query the database. Use ONLY the EXACT table and column names from
<sql_schema> in the system prompt — raw column names from the database will be
REJECTED by the translator.

When you have the final answer, call submit_answer with the appropriate
answer_type. Read the task guidelines carefully to choose the right format.

The evidence field (if present) contains hints about how to answer the question.
"""

_REPORT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "bird"


def _normalize_val(val: str) -> str:
    """Normalize a string value for comparison — strip whitespace, unify number format."""
    v = val.strip().rstrip("%")
    try:
        f = float(v)
        if f == int(f):
            return str(int(f))
        # Round to 6 decimal places for consistency
        return f"{f:.6f}".rstrip("0").rstrip(".")
    except ValueError:
        return v.lower()


@dataclass
class TaskResult:
    question_id: int
    db_id: str
    question: str
    difficulty: str = ""
    predicted: str = ""
    expected: str = ""
    passed: bool = False
    duration_ms: float = 0
    error: str = ""
    trace_file: str = ""


class BenchmarkRunner:
    """Runs BIRD Mini-Dev tasks through the DataSemAgent."""

    def __init__(self, llm_config: str = "default", concurrency: int = 4):
        self._llm_config = llm_config
        self._concurrency = concurrency
        self._report_lock = asyncio.Lock()

    async def run(
        self,
        question_ids: list[int] | None = None,
        difficulty: str | None = None,
        db_ids: list[str] | None = None,
        max_tasks: int | None = None,
    ) -> BenchmarkReport:
        tasks = load_tasks(difficulty=difficulty, db_ids=db_ids, max_tasks=max_tasks)
        if question_ids:
            idset = set(question_ids)
            tasks = [t for t in tasks if t["question_id"] in idset]

        # Ensure database files are downloaded
        target_dbs = {t["db_id"] for t in tasks}
        db_paths = ensure_databases(target_dbs)

        report = BenchmarkReport()
        t0 = time.perf_counter()
        sem = asyncio.Semaphore(self._concurrency)

        async def _run_one(task: dict) -> TaskResult:
            async with sem:
                result = await self._run_single(task, db_paths)
                async with self._report_lock:
                    report.results.append(result)
                    report.total_duration_ms = (time.perf_counter() - t0) * 1000
                    self._save_report(report)
                return result

        results = await asyncio.gather(*(_run_one(t) for t in tasks))
        report.results.sort(key=lambda r: r.question_id)
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
                    "question_id": r.question_id,
                    "db_id": r.db_id,
                    "difficulty": r.difficulty,
                    "question": r.question,
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

    async def _run_single(
        self, task: dict, db_paths: dict[str, str],
    ) -> TaskResult:
        result = TaskResult(
            question_id=task["question_id"],
            db_id=task["db_id"],
            question=task["question"],
            difficulty=task.get("difficulty", ""),
            expected=task.get("SQL", ""),  # ground-truth SQL
        )
        t0 = time.perf_counter()

        try:
            from app.flow import react_flow
            from app.node.agent import AgentNode
            from app.pipeline import QueuePipeline
            from app.eval.collector import EvalCollector
            from app.benchmark.dabstep.tool import SubmitAnswerTool

            db_name = task["db_id"]
            db_path = db_paths.get(db_name, "")
            if not db_path:
                result.error = f"Database not found: {db_name}"
                result.duration_ms = (time.perf_counter() - t0) * 1000
                return result

            # Build OSI model from DB schema
            schema = get_db_info(db_path)
            model = build_model(db_name, schema, db_path)

            db_config_key = f"bird_{db_name}"
            sql_tool = SqlExecTool(model_source=model, db_config_key=db_config_key)
            answer_tool = SubmitAnswerTool()

            agent = AgentNode(
                name=self._llm_config,
                system_prompt=_BIRD_SYSTEM_PROMPT,
                tools=[sql_tool, answer_tool],
            )

            captured: dict = {}
            def _on_answer(ctx, tool_call, tool, result):
                captured["raw"] = result.content
                # Also capture the translated SQL from sql_exec results
                # by scanning the context memory for the last sql_exec call

            collector = EvalCollector()
            collector.set_metadata({
                "question_id": task["question_id"],
                "db_id": task["db_id"],
                "difficulty": task.get("difficulty", ""),
                "question": task["question"],
            })
            pipeline = QueuePipeline()
            pipeline.register(collector)

            flow = react_flow(agent_node=agent, pipeline=pipeline)
            flow.context.hooks.on(
                HookPoint.TOOL_AFTER, _on_answer,
                tool_name="submit_answer", on_error="log",
            )

            # Build the prompt with evidence
            question = task["question"]
            if task.get("evidence"):
                question += f"\n\n[Evidence: {task['evidence']}]"

            await flow.ask(question)

            # Capture the textual answer from submit_answer
            predicted_answer = captured.get("raw", "")
            result.predicted = predicted_answer if predicted_answer else "(no answer)"
            result.trace_file = getattr(collector, "_session_id", "")

            # Score by comparing execution results of LLM SQL vs ground-truth SQL
            from app.semantics.sql.translator import SQLTranslator
            from app.semantics.sql.parser import OSIModelParser

            msgs = flow.context.memory.messages if hasattr(flow.context, 'memory') else []
            translator = SQLTranslator(OSIModelParser(model), strict=False)
            for msg in msgs:
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls:
                        if tc.function.name != "sql_exec":
                            continue
                        try:
                            args = json.loads(tc.function.arguments)
                            llm_sql = args.get("sql", "")
                            if not llm_sql:
                                continue
                            physical = translator.translate(llm_sql)
                            import sqlglot
                            dialect_sql = sqlglot.transpile(
                                physical.physical_sql, write="sqlite")[0]
                            passed, err = score(dialect_sql, task["SQL"], db_path)
                            if passed:
                                result.passed = True
                                break
                            # Keep last error for diagnostics
                            if err:
                                result.error = err
                        except Exception as e:
                            result.error = f"Scoring error: {e}"
                    if result.passed:
                        break

            # Fallback: compare submitted text answer against gold SQL result
            if not result.passed and predicted_answer and predicted_answer != "(no answer)":
                try:
                    import sqlite3 as _sqlite3
                    _conn = _sqlite3.connect(db_path)
                    _gold_rows = _conn.execute(task["SQL"]).fetchall()
                    _conn.close()
                    if _gold_rows and len(_gold_rows) == 1 and len(_gold_rows[0]) == 1:
                        gold_val = _gold_rows[0][0]
                        pred_norm = _normalize_val(predicted_answer)
                        gold_norm = _normalize_val(str(gold_val))
                        if pred_norm == gold_norm:
                            result.passed = True
                            result.error = ""
                        elif not result.error:
                            result.error = f"Text mismatch: pred={pred_norm} gold={gold_norm}"
                except Exception as _e:
                    if not result.error:
                        result.error = f"Fallback score error: {_e}"

        except Exception as e:
            result.error = str(e)

        result.duration_ms = (time.perf_counter() - t0) * 1000
        return result
