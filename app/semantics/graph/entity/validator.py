"""Mapping validator — cross-stage consistency and reference integrity."""
from app.semantics.graph.entity.schema import EntityGraphSchema
from app.semantics.graph.entity.mapping import DataMapping, TableSource, UnionSource
from app.semantics.models import SemanticModel


class MappingValidator:
    def __init__(self, schema: EntityGraphSchema, mapping: DataMapping, model: SemanticModel):
        self._schema = schema
        self._mapping = mapping
        self._model = model
        self._pk_map: dict[str, set[str]] = {}
        for ds in model.datasets:
            self._pk_map[ds.source] = set(ds.primary_key or [])

    def validate(self) -> list[str]:
        errors: list[str] = []
        self._check_schema_coverage(errors)
        self._check_table_column_refs(errors)
        self._check_key_uniqueness(errors)
        self._check_weak_entity_parents(errors)
        return errors

    def _check_key_uniqueness(self, errors: list[str]) -> None:
        """Verify each entity's key columns match the primary key of its source table."""
        for em in self._mapping.entities:
            ns = em.node_source
            table = None
            key_cols = []
            if isinstance(ns, TableSource):
                table = ns.table
                key_cols = ns.get_key_columns()
            elif isinstance(ns, UnionSource) and ns.sources:
                table = ns.sources[0].table
                key_cols = ns.sources[0].get_key_columns()

            if table and key_cols and table in self._pk_map:
                pks = self._pk_map[table]
                if pks and set(key_cols) != pks:
                    errors.append(
                        f"Entity '{em.entity}': key_columns {key_cols} do not match "
                        f"primary key of table '{table}' (PK: {sorted(pks)}). "
                        f"Use the table's primary key columns exactly."
                    )

    def _check_schema_coverage(self, errors: list[str]) -> None:
        schema_entities = self._schema.entity_labels()
        mapping_entities = self._mapping.entity_keys()
        for e in mapping_entities - schema_entities:
            errors.append(f"Mapping entity '{e}' not in graph schema")
        for e in schema_entities - mapping_entities:
            errors.append(f"Schema entity '{e}' has no mapping")

        schema_rels = self._schema.relation_labels()
        mapping_rels = {e.label for e in self._mapping.edges}
        for r in mapping_rels - schema_rels:
            errors.append(f"Mapping edge '{r}' not in graph schema")

    def _check_table_column_refs(self, errors: list[str]) -> None:
        known_tables = {ds.source for ds in self._model.datasets}
        for em in self._mapping.entities:
            ns = em.node_source
            if hasattr(ns, "table") and ns.table not in known_tables:
                errors.append(f"Entity '{em.entity}' references unknown table '{ns.table}'")

        for em in self._mapping.edges:
            if em.from_.entity not in self._mapping.entity_keys():
                errors.append(f"Edge '{em.label}' from unknown entity '{em.from_.entity}'")
            if em.to.entity not in self._mapping.entity_keys():
                errors.append(f"Edge '{em.label}' to unknown entity '{em.to.entity}'")

    def _check_weak_entity_parents(self, errors: list[str]) -> None:
        for entity in self._schema.entities:
            if entity.is_weak:
                if not entity.strong_parents:
                    errors.append(f"Weak entity '{entity.label}' has no strong_parents")
                else:
                    for parent in entity.strong_parents:
                        if parent not in self._schema.entity_labels():
                            errors.append(
                                f"Weak entity '{entity.label}' parent "
                                f"'{parent}' not found in schema"
                            )

    def validate_strict(self) -> None:
        errors = self.validate()
        if errors:
            raise MappingValidationError(errors)


class MappingValidationError(Exception):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("\n".join(errors))
