import sqlglot
from pathlib import Path
from typing import Literal

from pydantic import Field, PrivateAttr, model_validator

from app.config import config
from app.db.base import SqlExecutor
from app.exceptions import QueryNotAllowedError
from app.schema import ToolCall
from app.semantics.models import SemanticModel, OSISpecification
from app.semantics.sql.ddl_generator import DDLGenerator
from app.semantics.sql.exceptions import (
    DatasetNotFoundError,
    FieldNotFoundError,
    MetricNotFoundError,
)
from app.semantics.sql.parser import OSIModelParser
from app.semantics.sql.translator import SQLTranslator
from app.semantics.sql.validator import SQLValidator, SQLValidationError
from app.hook import HookPoint, hook
from app.tool.base import BaseTool, ToolResult


class SqlExecTool(BaseTool):
    permission: Literal["global", "skills", "agent"] = "agent"
    name: str = "sql_exec"
    description: str = (
        "Execute logical SQL queries against the semantic data model. "
        "Schema (datasets, fields, metrics, validation rules) is in the "
        "system prompt under <sql_schema>."
    )
    strict: bool = True
    parameters: dict = {
        "type": "object",
        "properties": {
            "sql": {
                "type": "string",
                "description": "Logical SQL — use OSI field names, not physical column names.",
            }
        },
        "required": ["sql"],
    }

    model_source: SemanticModel | str | None = Field(
        default=None,
        description="OSI semantic model instance or path to YAML file. Uses config default if not set.",
        exclude=True,
    )
    db_config_key: str = Field(
        default="default",
        description="Key of the database configuration in config.database to use.",
        exclude=True,
    )

    _parser: OSIModelParser = PrivateAttr(default=None)
    _validator: SQLValidator = PrivateAttr(default=None)
    _translator: SQLTranslator = PrivateAttr(default=None)
    _executor: SqlExecutor = PrivateAttr(default=None)
    _physical_to_logical: dict[str, str] = PrivateAttr(default_factory=dict)
    _ddl_prompt: str = PrivateAttr(default="")

    @model_validator(mode="after")
    def initialize_sql_components(self) -> "SqlExecTool":
        model = self._resolve_model(self.model_source)
        self._parser = OSIModelParser(model)
        self._validator = SQLValidator(model)
        self._translator = SQLTranslator(self._parser)
        self._executor = self._create_executor(self.db_config_key)
        self._physical_to_logical = self._build_reverse_mappings(model)
        ddl_generator = DDLGenerator(model)
        self._ddl_prompt = ddl_generator.prompt
        return self

    @hook(HookPoint.NODE_INIT_BEFORE)
    def _inject_schema(self, ctx, node):
        if not node.system_prompt or not self._ddl_prompt:
            return
        node.system_prompt = (
            node.system_prompt
            + "\n\n<sql_schema>\n"
            + self._ddl_prompt
            + "\n</sql_schema>"
        )

    async def execute(self, tool_call: ToolCall, **kwargs) -> ToolResult:
        sql = tool_call.function.arguments_dict.get("sql", "").strip()
        if not sql:
            return ToolResult.failure_response(
                tool_call.id, self.name, "No SQL provided"
            )

        try:
            self._validator.validate_strict(sql)
        except SQLValidationError as e:
            return ToolResult.failure_response(
                tool_call.id, self.name, f"SQL validation failed: {e}"
            )
        except Exception as e:
            return ToolResult.failure_response(
                tool_call.id, self.name,
                f"SQL parse error: {e}"
            )

        try:
            result = self._translator.translate(sql)
            physical_sql = result.physical_sql
            dialect_sql = sqlglot.transpile(
                physical_sql, write=self._executor.dialect
            )[0]
        except FieldNotFoundError as e:
            return ToolResult.failure_response(
                tool_call.id, self.name,
                f"Translation failed: unknown field '{e}'. "
                f"Use field names as defined in the OSI model. "
                f"Available fields per dataset are listed in CREATE TABLE comments."
            )
        except MetricNotFoundError as e:
            return ToolResult.failure_response(
                tool_call.id, self.name,
                f"Translation failed: unknown metric '{e}'. Available metrics: {', '.join(self._parser.list_metrics())}."
            )
        except DatasetNotFoundError as e:
            return ToolResult.failure_response(
                tool_call.id, self.name,
                f"Translation failed: unknown dataset '{e}'. Available datasets: {', '.join(self._parser.list_datasets())}."
            )
        except Exception as e:
            return ToolResult.failure_response(
                tool_call.id, self.name,
                f"Translation failed: {type(e).__name__}: {e}"
            )

        try:
            rows, total = await self._executor.execute(dialect_sql)
        except QueryNotAllowedError as e:
            return ToolResult.failure_response(
                tool_call.id, self.name, f"Invalid SQL: {e}"
            )
        except Exception as e:
            humanized = self._humanize_error(str(e))
            return ToolResult.failure_response(
                tool_call.id, self.name,
                f"Execution error: {humanized}\n"
                f"Executed SQL: {dialect_sql}"
            )

        output = self._format_output(rows, total)
        return ToolResult.success_response(tool_call.id, self.name, output)

    # ------------------------------------------------------------------ #
    #  Error humanization: physical → logical names
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_reverse_mappings(model: SemanticModel) -> dict[str, str]:
        """Build physical-name → logical-name mapping for error translation.

        Covers table names (source → dataset name) and column names
        (physical expression → logical field name).
        Longer keys are prioritized when multiple matches could occur.
        """
        mapping: dict[str, str] = {}

        for ds in model.datasets:
            # Physical table → logical dataset
            mapping[ds.source] = ds.name
            # Also handle SQLite-stripped names (tpcds.public.store_sales → store_sales)
            stripped = ds.source.rsplit(".", 1)[-1]
            if stripped != ds.source:
                mapping[stripped] = ds.name

            # Physical column → logical field
            for field in ds.fields or []:
                expr = field.expression.dialects[0].expression
                logical = field.name
                # Map both the simple column name and the expression
                if " " not in expr and expr == field.name:
                    mapping[expr] = logical
                mapping[expr] = logical

        # Metric expressions → logical metric name (first-wins for duplicates)
        for metric in model.metrics or []:
            expr = metric.expression.dialects[0].expression
            logical = metric.name
            if expr not in mapping:
                mapping[expr] = logical

        # Sort by key length descending so longer matches take priority
        return dict(sorted(mapping.items(), key=lambda x: -len(x[0])))

    def _humanize_error(self, error_message: str) -> str:
        """Translate physical identifiers in an error message to logical names."""
        import re
        msg = error_message
        for physical, logical in self._physical_to_logical.items():
            if physical == logical:
                continue  # skip identity mappings that cause cascade bugs
            if physical not in msg:
                continue
            # Use word-boundary replacement to avoid cascading:
            # 'description' in 'mc.mcc_description' → 'mc.mcc_mcc_description'
            msg = re.sub(rf'\b{re.escape(physical)}\b', logical, msg)
        return msg

    @staticmethod
    def _resolve_model(model_source: SemanticModel | str | None) -> SemanticModel:
        if isinstance(model_source, SemanticModel):
            return model_source
        if isinstance(model_source, str):
            spec = OSISpecification.load_from_yaml(model_source)
            return spec.semantic_model[0]

        semantics_dir = config.main_config.paths["semantics"]
        yaml_files = sorted(Path(semantics_dir).glob("*.yaml"))
        if not yaml_files:
            raise FileNotFoundError(
                f"No YAML semantic model found in {semantics_dir}"
            )
        spec = OSISpecification.load_from_yaml(str(yaml_files[0]))
        return spec.semantic_model[0]

    @staticmethod
    def _create_executor(db_config_key: str) -> SqlExecutor:
        from app.db.base import create_sql_executor
        return create_sql_executor(db_config_key)

    @staticmethod
    def _format_output(rows: list[dict], total: int) -> str:
        limit = len(rows)
        lines = [
            "## Query Results",
            "",
            f"**Total rows**: {total} | **Returned**: {limit}",
            "",
        ]

        if rows:
            headers = list(rows[0].keys())
            lines.append("")
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join("---" for _ in headers) + " |")
            for row in rows:
                lines.append(
                    "| "
                    + " | ".join(str(row.get(h, "")) for h in headers)
                    + " |"
                )
        else:
            lines.append("")
            lines.append("*No rows returned*")

        return "\n".join(lines)
