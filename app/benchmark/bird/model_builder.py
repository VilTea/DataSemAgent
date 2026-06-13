"""Load OSI semantic model — handwritten YAML preferred, auto-gen as fallback."""
from __future__ import annotations

from pathlib import Path

from app.semantics.models import (
    Dataset,
    Dialect,
    DialectExpression,
    Dimension,
    Expression,
    Metric,
    OSIField,
    OSISpecification,
    Relationship,
    SemanticModel,
)

_MODEL_DIR = Path(__file__).resolve().parent.parent.parent.parent / "config" / "benchmark" / "bird_models"


def build_model(db_name: str, schema: dict[str, list[dict]] | None = None,
                db_path: str | None = None) -> SemanticModel:
    """Load a handwritten OSI model for *db_name*, or auto-generate one.

    Handwritten models live in ``config/benchmark/bird_models/{db_name}.yaml``.
    If no YAML exists, falls back to auto-generation from the database schema.
    """
    yaml_path = _MODEL_DIR / f"{db_name}.yaml"
    if yaml_path.exists():
        spec = OSISpecification.load_from_yaml(str(yaml_path))
        return spec.semantic_model[0]

    # Fallback: auto-generate from schema
    if schema is None:
        raise ValueError(f"No schema provided for {db_name} and no YAML model found")
    return _auto_generate(db_name, schema)


def _auto_generate(db_name: str, schema: dict[str, list[dict]]) -> SemanticModel:
    """Auto-generate a 1:1 OSI model from database schema (fallback)."""
    _DIMENSION_TYPES = {"TEXT", "VARCHAR", "CHAR", "DATE", "DATETIME",
                        "TIMESTAMP", "BOOLEAN", "BOOL"}
    datasets: list[Dataset] = []
    relationships: list[Relationship] = []
    table_names = list(schema.keys())

    for table, columns in schema.items():
        fields: list[OSIField] = []
        for col in columns:
            col_name = col["name"]
            col_type = (col["type"] or "").upper()
            is_pk = col.get("pk", False)
            is_dim = any(t in col_type for t in _DIMENSION_TYPES)
            dimension = Dimension() if is_dim and not is_pk else None
            fields.append(OSIField(
                name=col_name,
                expression=Expression(dialects=[
                    DialectExpression(dialect=Dialect.ANSI_SQL, expression=col_name),
                ]),
                description=col_type + (" PK" if is_pk else ""),
                dimension=dimension,
            ))
        datasets.append(Dataset(name=table, source=table,
                                description=f"Table {table}", fields=fields))

    for table, columns in schema.items():
        for col in columns:
            col_name = col["name"]
            if col_name.endswith("_id") and col_name != "id":
                base = col_name[:-3]
                for parent in table_names:
                    if parent == table:
                        continue
                    if parent == base or parent == f"{base}s":
                        relationships.append(Relationship(
                            name=f"{table}_to_{parent}",
                            from_=table, to=parent,
                            from_columns=[col_name], to_columns=["id"],
                        ))
                        break

    return SemanticModel(
        name=f"bird_{db_name}",
        description=f"Auto-generated model for {db_name}",
        datasets=datasets, relationships=relationships,
        metrics=[], dimensions=[],
    )
