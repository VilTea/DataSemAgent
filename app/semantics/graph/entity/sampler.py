"""Data sampler — reads first N rows from each table via a database executor."""
from __future__ import annotations

from typing import TYPE_CHECKING

from app.db.base import SqlExecutor

if TYPE_CHECKING:
    from app.semantics.models import SemanticModel


class DataSampler:
    def __init__(self, executor: SqlExecutor, model: SemanticModel | None = None, sample_size: int = 3):
        self._executor = executor
        self._sample_size = sample_size
        # Build physical → OSI logical name map per dataset source.
        self._phys_to_logical: dict[str, dict[str, str]] = {}
        if model is not None:
            for ds in model.datasets:
                src_map: dict[str, str] = {}
                for f in (ds.fields or []):
                    phys = f.expression.dialects[0].expression if f.expression.dialects else f.name
                    src_map[phys] = f.name
                self._phys_to_logical[ds.source] = src_map

    async def sample(self, table_names: list[str]) -> dict[str, dict]:
        result = {}
        for table in table_names:
            try:
                rows, _ = await self._executor.execute(
                    f"SELECT * FROM {table}", limit=self._sample_size
                )
                if rows:
                    remap = self._phys_to_logical.get(table, {})
                    result[table] = {
                        "columns": [remap.get(c, c) for c in rows[0].keys()],
                        "rows": [{remap.get(k, k): v for k, v in r.items()} for r in rows],
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
