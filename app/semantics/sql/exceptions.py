from app.exceptions import DataSemAgentError


class FieldNotFoundError(DataSemAgentError):
    def __init__(self, field_name: str, dataset_name: str | None = None):
        self.field_name = field_name
        self.dataset_name = dataset_name
        if dataset_name:
            super().__init__(f"Field '{field_name}' not found in dataset '{dataset_name}'")
        else:
            super().__init__(f"Field '{field_name}' not found")


class MetricNotFoundError(DataSemAgentError):
    def __init__(self, metric_name: str):
        self.metric_name = metric_name
        super().__init__(f"Metric '{metric_name}' not found")


class DatasetNotFoundError(DataSemAgentError):
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        super().__init__(f"Dataset '{dataset_name}' not found")
