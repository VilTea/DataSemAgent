"""Deterministic graph skeleton generator — parses OSI model into GraphDocument."""
import sqlglot
import sqlglot.expressions as exp

from app.semantics.graph.contract import GraphDocument
from app.semantics.models import SemanticModel


class GraphSkeletonEngine:
    """Generate graph skeleton from a SemanticModel.

    Creates nodes for Metrics, Dimensions, LogicalDatasets, PhysicalFields
    and edges for MAPS_TO, DERIVED_FROM, AGGREGATES_FROM, SLICES_BY, JOINS_TO.
    """

    def __init__(self, model: SemanticModel):
        self._model = model

    def build(self) -> GraphDocument:
        doc = GraphDocument(
            meta={"model": self._model.name, "version": "0.1.1"}
        )
        self._add_dataset_nodes(doc)
        self._add_dimension_nodes(doc)
        self._add_physical_field_nodes(doc)
        self._add_maps_to_edges(doc)
        self._add_joins_to_edges(doc)
        self._add_metric_nodes(doc)
        return doc

    def _add_dataset_nodes(self, doc: GraphDocument) -> None:
        for ds in self._model.datasets:
            doc.add_node(
                f"dataset:{ds.name}",
                "LogicalDataset",
                name=ds.name,
                source=ds.source,
                primary_key=ds.primary_key or [],
            )

    def _add_dimension_nodes(self, doc: GraphDocument) -> None:
        for ds in self._model.datasets:
            for field in (ds.fields or []):
                if field.dimension is None:
                    continue
                logical = field.name
                dim_name = logical
                dim_id = f"dim:{dim_name}"
                doc.add_node(
                    dim_id,
                    "Dimension",
                    name=dim_name,
                    is_time=field.dimension.is_time,
                    dataset=ds.name,
                )
                # Connect dimension to its parent dataset
                doc.add_edge(
                    f"dataset:{ds.name}",
                    dim_id,
                    "MAPS_TO",
                    logical_name=field.name,
                    is_dimension=True,
                )

    def _add_physical_field_nodes(self, doc: GraphDocument) -> None:
        for ds in self._model.datasets:
            for field in (ds.fields or []):
                expr = field.expression.dialects[0].expression
                col_name = expr.split(".")[-1] if "." in expr else expr
                doc.add_node(
                    f"field:{ds.source}.{col_name}",
                    "PhysicalField",
                    name=col_name,
                    dataset=ds.name,
                )

    def _add_maps_to_edges(self, doc: GraphDocument) -> None:
        for ds in self._model.datasets:
            for field in (ds.fields or []):
                expr = field.expression.dialects[0].expression
                col_name = expr.split(".")[-1] if "." in expr else expr
                doc.add_edge(
                    f"dataset:{ds.name}",
                    f"field:{ds.source}.{col_name}",
                    "MAPS_TO",
                    logical_name=field.name,
                )

    def _add_joins_to_edges(self, doc: GraphDocument) -> None:
        for rel in (self._model.relationships or []):
            on_parts = [
                f"{f}={t}"
                for f, t in zip(rel.from_columns, rel.to_columns)
            ]
            doc.add_edge(
                f"dataset:{rel.from_dataset}",
                f"dataset:{rel.to_dataset}",
                "JOINS_TO",
                on=", ".join(on_parts),
                relationship=rel.name,
            )

    def _add_metric_nodes(self, doc: GraphDocument) -> None:
        if not self._model.metrics:
            return

        field_lookup: dict[str, str] = {}
        for node in doc.nodes:
            if node.label == "PhysicalField":
                parts = node.id.split(":", 1)[1].split(".", 1)
                if len(parts) == 2:
                    dataset_source = parts[0]
                    col_name = node.properties.get("name", "")
                    field_lookup[f"{dataset_source}.{col_name}"] = node.id
                    field_lookup[col_name] = node.id

        for metric in self._model.metrics:
            expr_text = metric.expression.dialects[0].expression
            doc.add_node(
                f"metric:{metric.name}",
                "Metric",
                name=metric.name,
                expression=expr_text,
                description=metric.description or "",
            )

            parsed = sqlglot.parse_one(expr_text)
            referenced_datasets: set[str] = set()

            for col_node in parsed.find_all(exp.Column):
                col_name = col_node.name
                qualifier = col_node.table

                if qualifier:
                    ref_key = f"{qualifier}.{col_name}"
                    field_id = field_lookup.get(ref_key)
                    if field_id:
                        doc.add_edge(
                            f"metric:{metric.name}",
                            field_id,
                            "AGGREGATES_FROM",
                        )
                    for ds in self._model.datasets:
                        if ds.name == qualifier or ds.source == qualifier:
                            referenced_datasets.add(ds.name)
                            break
                else:
                    field_id = field_lookup.get(col_name)
                    if field_id:
                        doc.add_edge(
                            f"metric:{metric.name}",
                            field_id,
                            "AGGREGATES_FROM",
                        )

            for ds_name in referenced_datasets:
                doc.add_edge(
                    f"metric:{metric.name}",
                    f"dataset:{ds_name}",
                    "DERIVED_FROM",
                )

            for ds_name in referenced_datasets:
                for ds in self._model.datasets:
                    if ds.name == ds_name:
                        for field in (ds.fields or []):
                            if field.dimension is not None:
                                logical = field.name
                                dim_name = logical
                                doc.add_edge(
                                    f"metric:{metric.name}",
                                    f"dim:{dim_name}",
                                    "SLICES_BY",
                                )
                        break
