"""Pipeline orchestrator — skeleton → validate → load."""
from app.semantics.graph.contract import GraphDocument
from app.semantics.graph.loader import GraphLoader
from app.semantics.graph.metric.skeleton import GraphSkeletonEngine
from app.semantics.graph.metric.validator import GraphValidationError, GraphValidator
from app.semantics.models import SemanticModel


async def build_metric_graph(
    model: SemanticModel,
    loader: GraphLoader | None = None,
) -> GraphDocument:
    doc = GraphSkeletonEngine(model).build()
    errors = GraphValidator(doc, model).validate()
    if errors:
        raise GraphValidationError(errors)
    if loader is None:
        from app.semantics.graph.loader import create_graph_loader
        loader = create_graph_loader("metric")
    loader.load(doc)
    return doc
