from app.semantics.models import (
    Dataset,
    Dialect as OSIDialect,
    Metric,
    OSIField,
    SemanticModel,
)
from app.semantics.sql.exceptions import (
    DatasetNotFoundError,
    FieldNotFoundError,
    MetricNotFoundError,
)
from app.semantics.sql.models import FieldMapping, TranslationContext

class OSIModelParser:
    def __init__(self, model: SemanticModel):
        self._model = model
        self._logical_to_physical: dict[str, FieldMapping] = {}
        self._build_indexes()

    def _build_indexes(self):
        for ds in self._model.datasets:
            if ds.fields:
                for field in ds.fields:
                    physical_expr = self._get_physical_expression(field)
                    logical_name = field.name
                    mapping = FieldMapping(
                        logical_name=logical_name,
                        physical_expression=physical_expr,
                        dataset_name=ds.name,
                        is_dimension=field.dimension is not None,
                        is_metric=False,
                    )
                    self._logical_to_physical[logical_name] = mapping

        if self._model.metrics:
            for metric in self._model.metrics:
                logical_name = metric.name
                physical_expr = self._get_metric_expression(metric)
                self._logical_to_physical[logical_name] = FieldMapping(
                    logical_name=logical_name,
                    physical_expression=physical_expr,
                    dataset_name=None,
                    is_dimension=False,
                    is_metric=True,
                )

    @property
    def model(self) -> SemanticModel:
        return self._model

    def list_datasets(self) -> list[str]:
        return [ds.name for ds in self._model.datasets]

    def list_metrics(self) -> list[str]:
        if not self._model.metrics:
            return []
        return [m.name for m in self._model.metrics]

    def is_metric(self, name: str) -> bool:
        return name in self._logical_to_physical and self._logical_to_physical[name].is_metric

    def is_dimension(self, name: str) -> bool:
        return name in self._logical_to_physical and self._logical_to_physical[name].is_dimension

    def is_field(self, name: str) -> bool:
        return name in self._logical_to_physical

    def get_dataset(self, name: str) -> Dataset:
        for ds in self._model.datasets:
            if ds.name == name:
                return ds
        raise DatasetNotFoundError(name)

    def resolve_field(self, field_name: str, dataset_name: str | None = None) -> FieldMapping:
        if field_name in self._logical_to_physical:
            return self._logical_to_physical[field_name]
        raise FieldNotFoundError(field_name, dataset_name)

    def _get_physical_expression(self, field) -> str:
        for dialect_expr in field.expression.dialects:
            if dialect_expr.dialect == OSIDialect.ANSI_SQL:
                return dialect_expr.expression
        
        if field.expression.dialects:
            return field.expression.dialects[0].expression
        
        return field.name

    def resolve_metric(self, metric_name: str) -> str:
        if metric_name in self._logical_to_physical:
            return self._logical_to_physical[metric_name].physical_expression
        raise MetricNotFoundError(metric_name)

    def _get_metric_expression(self, metric: Metric) -> str:
        for dialect_expr in metric.expression.dialects:
            if dialect_expr.dialect == OSIDialect.ANSI_SQL:
                return dialect_expr.expression
        
        if metric.expression.dialects:
            return metric.expression.dialects[0].expression
        
        return metric.name

    def get_dataset_source(self, dataset_name: str) -> str:
        ds = self.get_dataset(dataset_name)
        return ds.source

    def resolve_join_paths(self, from_dataset: str, to_dataset: str) -> list:
        if from_dataset == to_dataset:
            return []
        
        paths = []
        if self._model.relationships:
            for rel in self._model.relationships:
                if rel.from_dataset == from_dataset and rel.to_dataset == to_dataset:
                    paths.append(rel)
                elif rel.from_dataset == to_dataset and rel.to_dataset == from_dataset:
                    paths.append(rel)
        
        return paths

    def create_translation_context(self) -> TranslationContext:
        ctx = TranslationContext(parser=self)
        
        return ctx
