"""DABstep benchmark CLI — uv run python -m app.benchmark.dabstep"""
import argparse
import asyncio
import os

from app.benchmark.config import BenchmarkConfig
from app.benchmark.dabstep.runner import BenchmarkRunner


def main():
    parser = argparse.ArgumentParser(description="DABstep Benchmark Runner")
    parser.add_argument("--context-dir", required=True,
                        help="Path to DABstep data/context/ directory")
    parser.add_argument("--level", choices=["easy", "hard"],
                        help="Filter tasks by difficulty")
    parser.add_argument("--max-tasks", type=int,
                        help="Maximum number of tasks to run")
    parser.add_argument("--task-ids", type=int, nargs="*",
                        help="Specific task IDs to run")
    parser.add_argument("--llm-config", default="default",
                        help="LLM config key to use")
    args = parser.parse_args()

    os.environ["DABSTEP_CONTEXT_DIR"] = args.context_dir

    async def _run():
        with BenchmarkConfig():
            runner = BenchmarkRunner(
                context_dir=args.context_dir,
                llm_config=args.llm_config,
            )
            report = await runner.run(
                task_ids=args.task_ids,
                level=args.level,
                max_tasks=args.max_tasks,
            )

        print(f"\nDABstep Benchmark")
        print(f"Tasks: {report.total}  Passed: {report.passed}  "
              f"Accuracy: {report.accuracy:.1%}  "
              f"Duration: {report.total_duration_ms / 1000:.1f}s\n")

        for r in report.results:
            status = "PASS" if r.passed else "FAIL"
            if r.error:
                print(f"  [{status}] Task {r.task_id}: ERROR — {r.error[:100]}")
            else:
                print(f"  [{status}] Task {r.task_id}: predicted={r.predicted!r} expected={r.expected!r}")

    asyncio.run(_run())


if __name__ == "__main__":
    main()
