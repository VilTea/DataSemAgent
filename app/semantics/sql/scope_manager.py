from app.semantics.sql.exceptions import DatasetNotFoundError
from app.semantics.sql.parser import OSIModelParser


class ScopeManager:
    """Scoped-aware context manager for SQL translation.

    Wraps TranslationContext to provide clean scope management,
    table resolution, and outer reference tracking.
    """

    def __init__(self, parser: OSIModelParser):
        self._parser = parser

    # ------------------------------------------------------------------ #
    # Scope stack management
    # ------------------------------------------------------------------ #

    def push_scope(self, ctx, columns=None, tables=None):
        return ctx.push_scope(columns=columns, tables=tables)

    def pop_scope(self, ctx):
        return ctx.pop_scope()

    def current_scope(self, ctx):
        return ctx.get_current_scope()

    def resolve_column(self, ctx, name):
        return ctx.resolve_column(name)

    # ------------------------------------------------------------------ #
    # Current source tracking
    # ------------------------------------------------------------------ #

    def set_current_source(self, ctx, source: str):
        ctx.set_current_source(source)

    def get_current_source(self, ctx):
        return getattr(ctx, '_current_source', None)

    # ------------------------------------------------------------------ #
    # Table alias management (scoped lookup + global fallback)
    # ------------------------------------------------------------------ #

    def register_table(self, ctx, alias: str, physical_name: str):
        """Register a table alias mapping in the current scope and globally."""
        ctx.set_table_alias(alias, physical_name)
        scope = ctx.get_current_scope()
        scope.tables[alias] = physical_name

    def resolve_table_alias(self, ctx, alias: str) -> str | None:
        """Resolve a table alias to its physical name (scoped-first)."""
        for scope in reversed(ctx.scope_stack):
            if alias in scope.tables:
                return scope.tables[alias]
        return ctx.table_alias_map.get(alias)

    def resolve_physical_table(self, ctx, logical_name: str) -> str | None:
        """Resolve a logical table name to its physical name."""
        # Check scope stack first
        for scope in reversed(ctx.scope_stack):
            if logical_name in scope.tables:
                return scope.tables[logical_name]
        # Check global map
        if logical_name in ctx.table_alias_map:
            return ctx.table_alias_map[logical_name]
        # Fall back to parser
        try:
            return self._parser.get_dataset_source(logical_name)
        except DatasetNotFoundError:
            return None

    def is_table_in_current_scope(self, ctx, table_alias: str) -> bool:
        """Check if a table alias is registered in any scope on the stack."""
        for scope in ctx.scope_stack:
            if table_alias in scope.tables:
                return True
        return False

    def get_known_datasets(self):
        return {ds.name for ds in self._parser._model.datasets}

    def get_dataset_source(self, dataset_name: str) -> str:
        return self._parser.get_dataset_source(dataset_name)

    # ------------------------------------------------------------------ #
    # Outer reference tracking
    # ------------------------------------------------------------------ #

    def is_outer_ref(self, ctx, table_name: str) -> bool:
        return table_name in ctx.outer_table_refs

    def add_outer_ref(self, ctx, table_name: str):
        ctx.outer_table_refs.add(table_name)

    def remove_outer_ref(self, ctx, table_name: str):
        ctx.outer_table_refs.discard(table_name)
