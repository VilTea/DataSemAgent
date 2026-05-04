"""Graph document validator — checks reference integrity and completeness."""
from app.semantics.graph.contract import GraphDocument
from app.semantics.models import SemanticModel


class GraphValidator:
    """Validate a GraphDocument against its source SemanticModel.

    Returns a list of error strings. Empty list = valid.
    """

    def __init__(self, doc: GraphDocument, model: SemanticModel):
        self._doc = doc
        self._model = model

    def validate(self) -> list[str]:
        errors: list[str] = []
        self._check_edge_refs(errors)
        self._check_metric_fields(errors)
        self._check_orphan_nodes(errors)
        return errors

    def _check_edge_refs(self, errors: list[str]) -> None:
        ids = self._doc.node_ids()
        for edge in self._doc.edges:
            if edge.from_ not in ids:
                errors.append(f"Edge '{edge.label}' references unknown source node '{edge.from_}'")
            if edge.to not in ids:
                errors.append(f"Edge '{edge.label}' references unknown target node '{edge.to}'")

    def _check_metric_fields(self, errors: list[str]) -> None:
        for node in self._doc.nodes:
            if node.label != "Metric":
                continue
            has_agg = any(
                e.from_ == node.id and e.label == "AGGREGATES_FROM"
                for e in self._doc.edges
            )
            if not has_agg:
                errors.append(f"Metric '{node.id}' has no AGGREGATES_FROM edges")

    def _check_orphan_nodes(self, errors: list[str]) -> None:
        referenced: set[str] = set()
        for edge in self._doc.edges:
            referenced.add(edge.from_)
            referenced.add(edge.to)
        for node in self._doc.nodes:
            if node.id not in referenced:
                errors.append(f"Orphan node '{node.id}' has no edges")

    def validate_strict(self) -> None:
        errors = self.validate()
        if errors:
            raise GraphValidationError(errors)


class GraphValidationError(Exception):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("\n".join(errors))
