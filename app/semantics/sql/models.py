from dataclasses import dataclass, field
from typing import Any


@dataclass
class ColumnInfo:
    physical_expr: str
    logical_name: str
    dataset_name: str | None = None
    is_dimension: bool = False
    is_metric: bool = False


@dataclass
class FieldMapping:
    logical_name: str
    physical_expression: str
    dataset_name: str | None = None
    is_dimension: bool = False
    is_metric: bool = False


@dataclass
class Scope:
    columns: dict[str, ColumnInfo] = field(default_factory=dict)
    tables: dict[str, str] = field(default_factory=dict)
    outer_tables: set[str] = field(default_factory=set)
    
    def __setitem__(self, key: str, value: ColumnInfo) -> None:
        self.columns[key] = value
    
    def __getitem__(self, key: str) -> ColumnInfo | None:
        return self.columns.get(key)
    
    def __contains__(self, key: str) -> bool:
        return key in self.columns or key in self.tables


@dataclass
class TranslationContext:
    scope_stack: list[Scope] = field(default_factory=list)
    cte_mappings: dict[str, str] = field(default_factory=dict)
    table_alias_map: dict[str, str] = field(default_factory=dict)
    parser: Any = field(default=None, repr=False)
    outer_table_refs: set[str] = field(default_factory=set)

    def push_scope(self, columns: dict[str, ColumnInfo] | None = None, tables: dict[str, str] | None = None, outer_tables: set[str] | None = None) -> None:
        scope = Scope(
            columns=columns or {},
            tables=tables or {},
            outer_tables=outer_tables or set()
        )
        self.scope_stack.append(scope)

    def pop_scope(self) -> Scope:
        return self.scope_stack.pop() if self.scope_stack else Scope()

    def get_current_scope(self) -> Scope:
        return self.scope_stack[-1] if self.scope_stack else Scope()

    def resolve_column(self, name: str) -> ColumnInfo | None:
        for scope in reversed(self.scope_stack):
            if name in scope.columns:
                return scope.columns[name]
        return None

    def get_table_alias(self, alias: str) -> str | None:
        for scope in reversed(self.scope_stack):
            if alias in scope.tables:
                return scope.tables[alias]
        return self.table_alias_map.get(alias)

    def set_table_alias(self, alias: str, physical_name: str) -> None:
        self.table_alias_map[alias] = physical_name

    def get_current_table(self) -> str | None:
        if self.table_alias_map:
            return next(iter(self.table_alias_map.values()))
        return None
    
    def set_current_source(self, source: str) -> None:
        resolved = self.table_alias_map.get(source, source)
        self._current_source = resolved
    
    def get_current_source(self) -> str | None:
        return getattr(self, '_current_source', None)
