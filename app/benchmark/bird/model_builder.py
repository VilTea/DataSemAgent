"""Auto-generate OSI semantic model from BIRD database schema."""
from __future__ import annotations

from app.semantics.models import (
    Dataset,
    Dialect,
    DialectExpression,
    Dimension,
    Expression,
    Metric,
    OSIField,
    Relationship,
    SemanticModel,
)


# Column types that are likely dimensions (categorical, not aggregatable)
_DIMENSION_TYPES = {"TEXT", "VARCHAR", "CHAR", "NVARCHAR", "NCHAR", "DATE",
                    "DATETIME", "TIMESTAMP", "BOOLEAN", "BOOL"}
# Column types that are likely numeric / aggregatable
_METRIC_TYPES = {"INTEGER", "INT", "REAL", "FLOAT", "DOUBLE", "DECIMAL",
                 "NUMERIC", "BIGINT", "SMALLINT", "TINYINT", "MONEY"}


def build_model(
    db_name: str,
    schema: dict[str, list[dict]],
    db_path: str | None = None,
) -> SemanticModel:
    """Generate an OSI SemanticModel from a BIRD database schema.

    Heuristics:
    - Text/date columns → DIMENSION
    - Numeric columns → plain field (usable in aggregations)
    - Primary key columns → plain field
    - Foreign keys detected by naming convention (``*_id`` → references parent table)

    No metrics are auto-generated — the LLM must write its own aggregations.
    """
    datasets: list[Dataset] = []
    relationships: list[Relationship] = []
    table_names = list(schema.keys())

    for table, columns in schema.items():
        fields: list[OSIField] = []
        for col in columns:
            col_name = col["name"]
            col_type = (col["type"] or "").upper()
            is_pk = col.get("pk", False)

            # Classify: dimension vs plain (numeric)
            is_dimension = any(t in col_type for t in _DIMENSION_TYPES)
            dimension = Dimension() if is_dimension and not is_pk else None

            fields.append(OSIField(
                name=col_name,
                expression=Expression(dialects=[
                    DialectExpression(dialect=Dialect.ANSI_SQL, expression=col_name),
                ]),
                description=f"{col_type}" + (" PK" if is_pk else ""),
                dimension=dimension,
            ))

        datasets.append(Dataset(
            name=table,
            source=table,
            description=f"Table {table}",
            fields=fields,
        ))

    # Detect relationships by FK naming convention: column_name → parent_table
    for table, columns in schema.items():
        for col in columns:
            col_name = col["name"]
            if col_name.endswith("_id") and col_name != "id":
                # Guess parent table: customer_id → customer(s)
                base = col_name[:-3]  # strip "_id"
                for parent in table_names:
                    if parent == table:
                        continue
                    if parent == base or parent == f"{base}s" or parent.startswith(base):
                        relationships.append(Relationship(
                            name=f"{table}_to_{parent}",
                            from_=table,
                            to=parent,
                            from_columns=[col_name],
                            to_columns=["id"],
                        ))
                        break

    return SemanticModel(
        name=f"bird_{db_name}",
        description=f"Auto-generated model for BIRD database: {db_name}",
        datasets=datasets,
        relationships=relationships,
        metrics=[],
        dimensions=[],
    )
