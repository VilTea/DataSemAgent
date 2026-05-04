"""Data sampler — reads first N rows from each table via a database executor."""
from typing import Any

from app.db.base import SqlExecutor


class DataSampler:
    def __init__(self, executor: SqlExecutor, sample_size: int = 3):
        self._executor = executor
        self._sample_size = sample_size

    async def sample(self, table_names: list[str]) -> dict[str, dict]:
        result = {}
        for table in table_names:
            try:
                rows, _ = await self._executor.execute(
                    f"SELECT * FROM {table}", limit=self._sample_size
                )
                if rows:
                    result[table] = {
                        "columns": list(rows[0].keys()),
                        "rows": rows,
                    }
            except Exception:
                result[table] = {"columns": [], "rows": []}
        return result

    @staticmethod
    def format_for_prompt(samples: dict[str, dict]) -> str:
        lines = ["## Database Sample Data", ""]
        for table, data in samples.items():
            lines.append(f"### {table}")
            if data["columns"]:
                lines.append(f"Columns: {', '.join(data['columns'])}")
                lines.append(f"Sample rows ({len(data['rows'])}):")
                lines.append("```")
                for row in data["rows"]:
                    lines.append(str(row))
                lines.append("```")
            else:
                lines.append("(table not accessible or empty)")
            lines.append("")
        return "\n".join(lines)
