import json
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

from app.config import config

SqlFragment = str  # SQL 表达式片段，非验证仅标记意图
JsonString = str  # JSON 序列化字符串

# Enums for standardization
class Dialect(str, Enum):
    """Supported SQL and expression language dialects"""
    ANSI_SQL = "ANSI_SQL"
    SNOWFLAKE = "SNOWFLAKE"
    MDX = "MDX"
    TABLEAU = "TABLEAU"
    DATABRICKS = "DATABRICKS"


class Vendor(str, Enum):
    """Supported vendors for custom extensions"""
    COMMON = "COMMON"
    SNOWFLAKE = "SNOWFLAKE"
    SALESFORCE = "SALESFORCE"
    DBT = "DBT"
    DATABRICKS = "DATABRICKS"


# Core models
class AIContext(BaseModel):
    """Additional context for AI tools"""
    instructions: str | None = Field(default=None, description="Instructions for AI on how to use this entity")
    synonyms: list[str] | None = Field(default=None, description="Alternative names and terms")
    examples: list[str] | None = Field(default=None, description="Sample questions or use cases")

    model_config = ConfigDict(extra="allow")
    # Allow additional fields for extensibility


class CustomExtension(BaseModel):
    """Vendor-specific attributes for extensibility"""
    vendor_name: Vendor = Field(..., description="Vendor identifier")
    data: JsonString = Field(..., description="JSON string containing vendor-specific data")

    @classmethod
    @field_validator('data')
    def validate_json(cls, v):
        """Validate that data is valid JSON"""
        try:
            json.loads(v)
            return v
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in custom extension data: {e}")


class DialectExpression(BaseModel):
    """Expression in a specific dialect"""
    dialect: Dialect = Field(Dialect.ANSI_SQL, description="SQL or dialect-specific expression")
    expression: SqlFragment = Field(..., description="SQL or dialect-specific expression")


class Expression(BaseModel):
    """Expression definition with multi-dialect support"""
    dialects: list[DialectExpression] = Field(..., min_length=1, description="Expressions in different dialects")

    @staticmethod
    def from_dict(expressions: dict[Dialect, str]) -> "Expression":
        """Create an Expression object from a dictionary of dialect -> expression strings"""
        dialects = [DialectExpression(dialect=dialect, expression=expr)
                    for dialect, expr in expressions.items()]
        return Expression(dialects=dialects)


class Dimension(BaseModel):
    """Dimension metadata"""
    is_time: bool = Field(False, description="Indicates if this is a time-based dimension for temporal filtering")


class OSIField(BaseModel):
    """Row-level attribute for grouping, filtering, and metric expressions"""
    name: str = Field(..., description="Unique identifier for the field within the dataset")
    expression: Expression = Field(..., description="Expression definition with dialect support")
    dimension: Dimension | None = Field(default=None, description="Dimension metadata")
    label: str | None = Field(default=None, description="Label for categorization")
    description: str | None = Field(default=None, description="Human-readable description")
    ai_context: str | AIContext | None = Field(default=None, description="Additional context for AI tools")
    custom_extensions: list[CustomExtension] | None = Field(default=None, description="Vendor-specific attributes")

    @staticmethod
    def time_dimension(name: str, column_name: str, description: str = None) -> "OSIField":
        """Create a time dimension field"""
        return OSIField(
            name=name,
            expression=Expression.from_dict({Dialect.ANSI_SQL: column_name}),
            dimension=Dimension(is_time=True),
            description=description
        )

    @staticmethod
    def computed_field(name: str, sql_expression: SqlFragment, description: str = None) -> "OSIField":
        """Create a computed field with a SQL expression"""
        return OSIField(
            name=name,
            expression=Expression.from_dict({Dialect.ANSI_SQL: sql_expression}),
            description=description
        )

    @staticmethod
    def simple_field(name: str, column_name: str, description: str = None) -> "OSIField":
        """Create a simple field that directly references a column"""
        return OSIField(
            name=name,
            expression=Expression.from_dict({Dialect.ANSI_SQL: column_name}),
            description=description
        )


class Dataset(BaseModel):
    """Logical dataset representing a business entity (fact or dimension table)"""
    name: str = Field(..., description="Unique identifier for the dataset")
    source: str = Field(..., description="Reference to underlying physical table/view or query")
    primary_key: list[str] | None = Field(default=None, description="Primary key columns (single or composite)")
    unique_keys: list[list[str]] | None = Field(default=None, description="Array of unique key definitions")
    description: str | None = Field(default=None, description="Human-readable description")
    ai_context: str | AIContext | None= Field(default=None, description="Additional context for AI tools")
    fields: list[OSIField] | None = Field(default=None, description="Row-level attributes")
    custom_extensions: list[CustomExtension] | None = Field(default=None, description="Vendor-specific attributes")

    @classmethod
    @field_validator('unique_keys')
    def validate_unique_keys(cls, v):
        """Validate that unique_keys are properly formatted"""
        if v:
            for key in v:
                if not key:
                    raise ValueError("Unique key cannot be empty")
        return v

    @classmethod
    @field_validator('fields')
    def validate_field_names_unique(cls, v):
        """Validate that field names are unique within the dataset"""
        if v:
            field_names = [field.name for field in v]
            duplicates = [name for name in field_names if field_names.count(name) > 1]
            if duplicates:
                raise ValueError(f"Duplicate field names found: {set(duplicates)}")
        return v


class Relationship(BaseModel):
    """Foreign key relationship between datasets"""
    name: str = Field(..., description="Unique identifier for the relationship")
    from_dataset: str = Field(..., alias="from", description="Dataset on the many side of the relationship")
    to_dataset: str = Field(..., alias="to", description="Dataset on the one side of the relationship")
    from_columns: list[str] = Field(..., min_length=1, description="Foreign key columns in the 'from' dataset")
    to_columns: list[str] = Field(..., min_length=1, description="Primary/unique key columns in the 'to' dataset")
    ai_context: str | AIContext | None= Field(default=None, description="Additional context for AI tools")
    custom_extensions: list[CustomExtension] | None = Field(default=None, description="Vendor-specific attributes")

    @model_validator(mode="after")
    def validate_columns_match(self):
        """Validate that from_columns and to_columns have the same length"""
        if self.from_columns and self.to_columns and len(self.from_columns) != len(self.to_columns):
            raise ValueError(
                f"from_columns ({len(self.from_columns)}) and to_columns ({len(self.to_columns)}) must have the same length")
        return self

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Metric(BaseModel):
    """Quantitative measure defined on business data"""
    name: str = Field(..., description="Unique identifier for the metric")
    expression: Expression = Field(..., description="Expression definition with dialect support")
    description: str | None = Field(default=None, description="Human-readable description of what the metric measures")
    ai_context: str | AIContext | None = Field(default=None, description="Additional context for AI tools")
    custom_extensions: list[CustomExtension] | None = Field(default=None, description="Vendor-specific attributes")

    @staticmethod
    def simple_metric(name: str, sql_expression: SqlFragment, description: str = None) -> "Metric":
        """Create a simple metric with a SQL aggregation expression"""
        return Metric(
            name=name,
            expression=Expression.from_dict({Dialect.ANSI_SQL: sql_expression}),
            description=description
        )


class SemanticModel(BaseModel):
    """Top-level container representing a complete semantic model"""
    name: str = Field(..., description="Unique identifier for the semantic model")
    description: str | None = Field(default=None, description="Human-readable description")
    ai_context: str | AIContext | None= Field(default=None, description="Additional context for AI tools")
    datasets: list[Dataset] = Field(..., min_length=1, description="Collection of logical datasets")
    relationships: list[Relationship] | None = Field(default=None, description="Defines how datasets are connected")
    metrics: list[Metric] | None = Field(default=None, description="Quantifiable measures spanning datasets")
    custom_extensions: list[CustomExtension] | None = Field(default=None, description="Vendor-specific attributes")

    @classmethod
    @field_validator('datasets')
    def validate_dataset_names_unique(cls, v):
        """Validate that dataset names are unique within the semantic model"""
        dataset_names = [dataset.name for dataset in v]
        duplicates = [name for name in dataset_names if dataset_names.count(name) > 1]
        if duplicates:
            raise ValueError(f"Duplicate dataset names found: {set(duplicates)}")
        return v

    @classmethod
    @field_validator('metrics')
    def validate_metric_names_unique(cls, v):
        """Validate that metric names are unique within the semantic model"""
        if v:
            metric_names = [metric.name for metric in v]
            duplicates = [name for name in metric_names if metric_names.count(name) > 1]
            if duplicates:
                raise ValueError(f"Duplicate metric names found: {set(duplicates)}")
        return v

    @classmethod
    @field_validator('relationships')
    def validate_relationship_names_unique(cls, v):
        """Validate that relationship names are unique within the semantic model"""
        if v:
            rel_names = [rel.name for rel in v]
            duplicates = [name for name in rel_names if rel_names.count(name) > 1]
            if duplicates:
                raise ValueError(f"Duplicate relationship names found: {set(duplicates)}")
        return v

    @model_validator(mode='after')
    def validate_relationship_references(self):
        """Validate that relationships reference existing datasets"""
        if self.relationships and self.datasets:
            dataset_names = {dataset.name for dataset in self.datasets}
            for rel in self.relationships:
                if rel.from_dataset not in dataset_names:
                    raise ValueError(f"Relationship '{rel.name}' references unknown dataset '{rel.from_dataset}'")
                if rel.to_dataset not in dataset_names:
                    raise ValueError(f"Relationship '{rel.name}' references unknown dataset '{rel.to_dataset}'")
        return self

# Root model for the complete OSI specification
class OSISpecification(BaseModel):
    """Root model for OSI Core Metadata Specification"""
    version: Literal["0.1.1"] = Field(..., description="OSI specification version")
    dialects: list[Dialect] | None = Field(default=None,
                                              description="Supported expression language dialects (enumeration definition)")
    vendors: list[Vendor] | None = Field(default=None,
                                            description="Supported vendors for custom extensions (enumeration definition)")
    semantic_model: list[SemanticModel] = Field(..., description="Collection of semantic model definitions")

    model_config = ConfigDict(extra="forbid")
    # No additional properties allowed at root level

    def save_to_yaml(self, yaml_path: str):
        """Save OSI specification to a YAML file"""
        import yaml
        with open(yaml_path, 'w') as f:
            yaml.dump(self.model_dump(by_alias=True, exclude_none=True), f, sort_keys=False)

    def save_to_json(self, json_path: str):
        """Save OSI specification to a JSON file"""
        import json
        with open(json_path, 'w') as f:
            json.dump(self.model_dump(by_alias=True, exclude_none=True), f, indent=2)

    # Serialization utilities
    @staticmethod
    def load_from_yaml(yaml_path: str | Path) -> "OSISpecification":
        """Load and validate OSI specification from a YAML file"""
        import yaml
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return OSISpecification(**data)

    @staticmethod
    def load_from_json(json_path: str) -> "OSISpecification":
        """Load and validate OSI specification from a JSON file"""
        import json
        with open(json_path, 'r') as f:
            data = json.load(f)
        return OSISpecification(**data)