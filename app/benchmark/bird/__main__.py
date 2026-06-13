"""CLI entry point for BIRD benchmark."""
import argparse
import asyncio

from app.benchmark.config import BenchmarkConfig


def main():
    parser = argparse.ArgumentParser(description="BIRD Mini-Dev Benchmark Runner")
    parser.add_argument("--difficulty", choices=["simple", "moderate", "challenging"],
                        help="Filter by difficulty level")
    parser.add_argument("--max-tasks", type=int, help="Maximum number of tasks to run")
    parser.add_argument("--question-ids", type=int, nargs="*",
                        help="Specific question IDs to run")
    parser.add_argument("--db-ids", nargs="*",
                        help="Specific database IDs to run")
    parser.add_argument("--llm-config", default="default",
                        help="LLM configuration name")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Number of concurrent tasks")
    parser.add_argument("--task-timeout", type=int, default=300,
                        help="Per-task timeout in seconds")
    args = parser.parse_args()

    async def _run():
        from .runner import BenchmarkRunner
        runner = BenchmarkRunner(
            llm_config=args.llm_config,
            concurrency=args.concurrency,
            task_timeout=args.task_timeout,
        )
        return await runner.run(
            question_ids=args.question_ids,
            difficulty=args.difficulty,
            db_ids=args.db_ids,
            max_tasks=args.max_tasks,
        )

    with BenchmarkConfig():
        report = asyncio.run(_run())
        print(f"\nBIRD Mini-Dev Benchmark")
        print(f"Tasks: {report.total}  Passed: {report.passed}  "
              f"Accuracy: {report.accuracy:.1%}  "
              f"Duration: {report.total_duration_ms / 1000:.1f}s")
        for r in report.results:
            status = "PASS" if r.passed else "FAIL"
            print(f"  [{status}] Q{r.question_id}: "
                  f"predicted={r.predicted[:60] if r.predicted else 'N/A'} "
                  f"| db={r.db_id}")


if __name__ == "__main__":
    main()
