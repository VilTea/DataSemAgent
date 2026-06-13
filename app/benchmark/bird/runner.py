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
You are a data analyst. Write a SINGLE SQL query that directly answers the question.

Rules:
- Use ONLY field names from <sql_schema>. Raw column names are REJECTED.
- Write ONE query whose result IS the answer. Do NOT explore the schema first
  unless you genuinely don't understand the table structure.
- The query result will be compared against a reference — same columns, same rows.
- If the question asks for a single value, your query must return exactly one row
  with one column.
- If the question asks for a list, return one column with multiple rows.

When submitting the final answer via submit_answer:
- For numbers: submit just the number (e.g. '42' or '3.14'), no extra text.
- For names/text: submit just the value (e.g. 'CZE'), no extra text.
- For lists: submit as a comma-separated list (e.g. 'A, B, C').
- If the question cannot be answered, submit 'Not Applicable'.

The evidence field (if present) contains hints about the expected SQL logic.
"""

_REPORT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "bird"


def _values_match(a: str, b: str) -> bool:
    """Compare two string values — try numeric, then case-insensitive text."""
    a, b = a.strip().rstrip("%"), b.strip().rstrip("%")
    if a == b:
        return True
    # Numeric comparison
    try:
        fa, fb = float(a.replace(",", "")), float(b.replace(",", ""))
        if fa == fb:
            return True
        if fb != 0 and abs(fa - fb) / abs(fb) < 1e-4:
            return True  # close enough (0.01% tolerance)
    except (ValueError, ZeroDivisionError):
        pass
    return a.lower() == b.lower()


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

    def __init__(self, llm_config: str = "default", concurrency: int = 4,
                 task_timeout: float = 300):
        self._llm_config = llm_config
        self._concurrency = concurrency
        self._task_timeout = task_timeout
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

            await asyncio.wait_for(flow.ask(question), timeout=self._task_timeout)

            # Capture the textual answer from submit_answer
            predicted_answer = captured.get("raw", "")
            result.predicted = predicted_answer if predicted_answer else "(no answer)"
            result.trace_file = getattr(collector, "_session_id", "")

            # Score: (1) last SQL result-set vs gold SQL, (2) text answer vs gold result
            from app.semantics.sql.translator import SQLTranslator
            from app.semantics.sql.parser import OSIModelParser
            import sqlglot
            translator = SQLTranslator(OSIModelParser(model), strict=False)

            # Find the last sql_exec call
            msgs = flow.context.memory.messages if hasattr(flow.context, 'memory') else []
            last_sql = ""
            for msg in reversed(msgs):
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in reversed(msg.tool_calls):
                        if tc.function.name == "sql_exec":
                            try:
                                args = json.loads(tc.function.arguments)
                                last_sql = args.get("sql", "")
                            except Exception:
                                pass
                            break
                    if last_sql:
                        break

            if last_sql:
                try:
                    physical = translator.translate(last_sql)
                    dialect_sql = sqlglot.transpile(
                        physical.physical_sql, write="sqlite")[0]
                    passed, err = score(dialect_sql, task["SQL"], db_path)
                    if passed:
                        result.passed = True
                    elif err:
                        result.error = err
                except Exception as e:
                    result.error = f"Scoring error: {e}"

            # Text answer vs gold result (handles multi-step reasoning)
            if not result.passed and predicted_answer and predicted_answer != "(no answer)":
                try:
                    import sqlite3 as _sqlite3
                    _conn = _sqlite3.connect(db_path)
                    _gold_rows = _conn.execute(task["SQL"]).fetchall()
                    _conn.close()
                    if _gold_rows and len(_gold_rows) == 1 and len(_gold_rows[0]) == 1:
                        gold_str = str(_gold_rows[0][0]).strip()
                        if _values_match(predicted_answer.strip(), gold_str):
                            result.passed = True
                            result.error = ""
                        elif not result.error:
                            result.error = f"Text mismatch: pred={predicted_answer[:60]} gold={gold_str[:60]}"
                except Exception as _e:
                    if not result.error:
                        result.error = f"Text compare error: {_e}"

        except asyncio.TimeoutError:
            result.error = f"Task timed out after {self._task_timeout}s"
        except Exception as e:
            result.error = str(e)

        result.duration_ms = (time.perf_counter() - t0) * 1000
        return result
