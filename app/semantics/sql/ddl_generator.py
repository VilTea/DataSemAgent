from typing import Optional

from app.semantics.models import (
    SemanticModel,
    Dataset,
    OSIField,
    Metric,
    Dialect,
    AIContext,
    Relationship,
)

# OSI SQL rules (embedded in DDL prompt)
OSI_SQL_RULES = """-- [OSI SQL Rules]
-- DIMENSION columns: use in GROUP BY / WHERE / ORDER BY.
-- METRIC columns: pre-computed aggregates (SUM/COUNT/AVG).  Use them directly
--   as SELECT expressions — do NOT wrap them in SUM(), AVG(), etc.
--   To get per-group metric values, add dimensions to GROUP BY:
--     SELECT merchant_name, fraud_rate FROM payments GROUP BY merchant_name
--   The metric is automatically computed per group.  This is NOT re-aggregation.
--   Use HAVING (not WHERE) to filter on metric values.
-- TIME_DIMENSION columns: support time-series filtering and grouping.
-- JOIN requirements: if a metric comment says "requires JOIN with X",
--   you MUST add that JOIN to the FROM clause.
-- **Violating any rule is forbidden!!**

-- [[After writing SQL, check each rule — the SQL MUST fully comply!]]"""


class DDLGenerator:
    def __init__(self, semantic_model: SemanticModel, target_dialect: Dialect = Dialect.ANSI_SQL):
        self.semantic_model = semantic_model
        self.target_dialect = target_dialect
        self._relationship_index: dict[str, list[Relationship]] = {}
        self._build_relationship_index()

    def _build_relationship_index(self):
        """Build index from dataset to its related relationships."""
        for rel in self.semantic_model.relationships or []:
            self._relationship_index.setdefault(rel.from_dataset, []).append(rel)
            self._relationship_index.setdefault(rel.to_dataset, []).append(rel)

    def _get_dataset_relationships(self, dataset_name: str) -> list[Relationship]:
        """Get all relationships associated with a dataset."""
        return self._relationship_index.get(dataset_name, [])

    def _get_relationships_as_from(self, dataset_name: str) -> list[Relationship]:
        """Get relationships where this dataset is from_dataset (many-side)."""
        return [r for r in self._relationship_index.get(dataset_name, []) if r.from_dataset == dataset_name]

    def _get_relationships_as_to(self, dataset_name: str) -> list[Relationship]:
        """Get relationships where this dataset is to_dataset (one-side)."""
        return [r for r in self._relationship_index.get(dataset_name, []) if r.to_dataset == dataset_name]

    def generate(self, dataset_name: str, include_rules: bool = True) -> str:
        dataset = self._get_dataset(dataset_name)
        if not dataset:
            raise ValueError(f"Dataset '{dataset_name}' not found")

        ddl = self._generate_create_table(dataset)
        if include_rules:
            ddl = OSI_SQL_RULES + "\n\n" + ddl
        return ddl

    def generate_for_datasets(self, dataset_names: list[str], include_rules: bool = True) -> dict[str, str]:
        """Generate DDL for a list of dataset names."""
        result = {}
        for name in dataset_names:
            try:
                result[name] = self.generate(name, include_rules=include_rules)
            except ValueError:
                result[name] = f"-- ERROR: Dataset '{name}' not found"
        return result

    def _get_dataset(self, dataset_name: str) -> Optional[Dataset]:
        for dataset in self.semantic_model.datasets:
            if dataset.name == dataset_name:
                return dataset
        return None

    def _is_dimension_field(self, field: OSIField) -> bool:
        return field.dimension is not None

    def _is_time_dimension(self, field: OSIField) -> bool:
        return field.dimension is not None and field.dimension.is_time

    def _is_dimension_by_name(self, col_name: str, dataset: Dataset) -> bool:
        for field in (dataset.fields or []):
            if field.name == col_name:
                return self._is_dimension_field(field)
        return False

    def _get_prefixed_name(self, field: OSIField) -> str:
        return field.name

    def _get_related_metrics(self, dataset_name: str) -> list[tuple[Metric, list[str]]]:
        """Get metrics related to a dataset and their additional dependencies.

        Returns:
            list of (metric, other_dependencies) tuples
        """
        if not self.semantic_model.metrics:
            return []

        related = []
        for metric in self.semantic_model.metrics:
            if metric.expression.dialects:
                expr = metric.expression.dialects[0].expression

                # Check if this metric references the current dataset (exact match tablename. pattern)
                if f'{dataset_name}.' in expr:
                    # Find other referenced datasets
                    other_deps = []
                    for ds in self.semantic_model.datasets:
                        if ds.name != dataset_name and f'{ds.name}.' in expr:
                            other_deps.append(ds.name)

                    related.append((metric, other_deps))

        return related

    def _build_field_comment(self, field: OSIField, dataset: Dataset) -> str:
        parts = []

        if self._is_dimension_field(field):
            if self._is_time_dimension(field):
                parts.append("[TIME_DIMENSION]")
            else:
                parts.append("[DIMENSION]")

        if field.description:
            parts.append(field.description)
        elif field.label:
            parts.append(field.label)

        # Relationship annotation if this field is a foreign key column
        rel_comment = self._build_field_relationship_comment(field.name, dataset)
        if rel_comment:
            parts.append(rel_comment)

        parts.extend(self._build_ai_context_parts(field.ai_context))

        return " | ".join(parts)

    def _build_ai_context_parts(self, ai_ctx: str | AIContext | None) -> list[str]:
        """Parse AIContext (str or object) into comment fragments."""
        if not ai_ctx:
            return []
        if isinstance(ai_ctx, str):
            return [ai_ctx]
        parts = []
        if ai_ctx.synonyms:
            parts.append(f"synonyms: {', '.join(ai_ctx.synonyms)}")
        if ai_ctx.instructions:
            parts.append(f"instructions: {ai_ctx.instructions}")
        if ai_ctx.examples:
            parts.append(f"examples: {'; '.join(ai_ctx.examples)}")
        return parts

    def _get_prefixed_col_name(self, col_name: str, dataset_name: str) -> str:
        """Return column name as-is (no prefix)."""
        return col_name

    def _build_field_relationship_comment(self, field_name: str, dataset: Dataset) -> str | None:
        """Build relationship comment for a field (FK direction only, no AI context)."""
        rel_parts = []
        for rel in self._get_dataset_relationships(dataset.name):
            # Field is in from_columns (current table is many-side)
            if rel.from_dataset == dataset.name and field_name in rel.from_columns:
                idx = rel.from_columns.index(field_name)
                to_col = self._get_prefixed_col_name(rel.to_columns[idx], rel.to_dataset)
                to_table = rel.to_dataset
                rel_parts.append(f"FK -> {to_table}.{to_col}")
            # Field is in to_columns (current table is one-side)
            elif rel.to_dataset == dataset.name and field_name in rel.to_columns:
                idx = rel.to_columns.index(field_name)
                from_col = self._get_prefixed_col_name(rel.from_columns[idx], rel.from_dataset)
                from_table = rel.from_dataset
                rel_parts.append(f"<- {from_table}.{from_col}")

        return f"Relationship: {'; '.join(rel_parts)}" if rel_parts else None

    def _build_relationship_ai_context(self, rel: Relationship) -> str | None:
        """Extract AI context comment from a relationship."""
        parts = self._build_ai_context_parts(rel.ai_context)
        return " | ".join(parts) if parts else None

    def _build_relationship_ai_comments(self, dataset: Dataset) -> list[str]:
        """Generate separate AI context comments for relationships associated with a dataset."""
        comments = []
        for rel in self._get_dataset_relationships(dataset.name):
            ai_ctx = self._build_relationship_ai_context(rel)
            if ai_ctx:
                # Determine relationship direction (using logical names)
                if rel.from_dataset == dataset.name:
                    comments.append(f"Relationship [{rel.name}]: {dataset.name} -> {rel.to_dataset} | {ai_ctx}")
                elif rel.to_dataset == dataset.name:
                    comments.append(f"Relationship [{rel.name}]: {rel.from_dataset} -> {dataset.name} | {ai_ctx}")
        return comments

    def _build_metric_comment(self, metric: Metric, deps: list[str] | None = None) -> str:
        parts = ["[METRIC] Already aggregated — do NOT re-aggregate"]

        if deps:
            parts.append(f"**This metric requires JOIN with: {', '.join(deps)}**")

        if metric.description:
            parts.append(metric.description)

        parts.extend(self._build_ai_context_parts(metric.ai_context))

        return " | ".join(parts)

    def _generate_create_table(self, dataset: Dataset) -> str:
        table_name = dataset.name
        columns = []

        # 1. Fields
        for field in (dataset.fields or []):
            col_name = self._get_prefixed_name(field)

            col_def = col_name

            comment = self._build_field_comment(field, dataset)
            if comment:
                col_def += self._generate_column_comment(comment)

            columns.append(col_def)

        # 2. Metrics (virtual columns)
        for metric, deps in self._get_related_metrics(dataset.name):
            col_name = metric.name
            col_def = col_name

            comment = self._build_metric_comment(metric, deps)
            if comment:
                col_def += self._generate_column_comment(comment)

            columns.append(col_def)

        # 3. Constraints
        if dataset.primary_key:
            pk_cols = ", ".join(dataset.primary_key)
            columns.append(f"PRIMARY KEY ({pk_cols})")

        if dataset.unique_keys:
            for idx, uk in enumerate(dataset.unique_keys):
                uk_cols = ", ".join(uk)
                columns.append(f"UNIQUE KEY uk_{idx} ({uk_cols})")

        # 4. Relationship comments as table-level comments
        rel_comment = self._build_table_relationship_comment(dataset)
        rel_ai_comments = self._build_relationship_ai_comments(dataset)

        # Build DDL
        ddl = f"CREATE TABLE {table_name} (\n    " + ",\n    ".join(columns) + "\n);"

        # Relationship AI context comments (separate from table comment)
        if rel_ai_comments:
            for comment in rel_ai_comments:
                ddl += f"\n-- {comment}"

        # Table comment
        table_comment_parts = []
        if dataset.description:
            table_comment_parts.append(dataset.description)
        table_comment_parts.extend(self._build_ai_context_parts(dataset.ai_context))
        if rel_comment:
            table_comment_parts.append(rel_comment)

        if table_comment_parts:
            table_comment = " | ".join(table_comment_parts)
            ddl += f"\nCOMMENT ON TABLE {table_name} IS '{table_comment}';"

        return ddl

    def _build_table_relationship_comment(self, dataset: Dataset) -> str | None:
        """Build relationship comment for a table (FK direction only, no AI context)."""
        rels = self._get_dataset_relationships(dataset.name)
        if not rels:
            return None

        parts = []
        # As from_dataset (many-side → one-side)
        for rel in self._get_relationships_as_from(dataset.name):
            to_table = rel.to_dataset
            cols = ", ".join([self._get_prefixed_col_name(c, dataset.name) for c in rel.from_columns])
            to_cols = ", ".join([self._get_prefixed_col_name(c, rel.to_dataset) for c in rel.to_columns])
            parts.append(f"{cols} → {to_table}({to_cols})")

        # As to_dataset (one-side ← many-side)
        for rel in self._get_relationships_as_to(dataset.name):
            from_table = rel.from_dataset
            cols = ", ".join([self._get_prefixed_col_name(c, dataset.name) for c in rel.to_columns])
            from_cols = ", ".join([self._get_prefixed_col_name(c, rel.from_dataset) for c in rel.from_columns])
            parts.append(f"{cols} ← {from_table}({from_cols})")

        return f"Relationship: {'; '.join(parts)}" if parts else None

    def _map_datatype(self, field: OSIField) -> str:
        if self._is_time_dimension(field):
            return "DATE"

        expr = field.expression.dialects[0].expression.lower() if field.expression.dialects else ""

        if "date" in expr or "time" in expr:
            return "DATE"
        elif "amount" in expr or "price" in expr or "cost" in expr:
            return "DECIMAL"
        elif "quantity" in expr or "count" in expr or "number" in expr:
            return "INTEGER"
        elif expr.endswith("_sk") or expr.endswith("_id"):
            return "VARCHAR(50)"

        return "VARCHAR"

    def _generate_column_comment(self, comment: str) -> str:
        escaped = comment.replace("'", "''")
        return f" COMMENT '{escaped}'"

    def generate_all(self) -> dict[str, str]:
        return {ds.name: self.generate(ds.name) for ds in self.semantic_model.datasets}

    @property
    def prompt(self) -> str:
        """Generate complete DDL prompt including model-level AI context."""
        parts = []
        # Model-level AI context
        model_ctx = self._build_ai_context_parts(self.semantic_model.ai_context)
        if model_ctx:
            parts.append(f"-- SemanticModel: {self.semantic_model.name} | {' | '.join(model_ctx)}")
            parts.append('')

        for idx, ds in enumerate(self.semantic_model.datasets):
            parts.append(self.generate(ds.name, idx == 0))

        return '\n\n'.join(parts)

