import sqlglot
import sqlglot.expressions as exp

from app.semantics.sql.exceptions import DatasetNotFoundError, FieldNotFoundError


class FieldExpander:
    """Expands logical field references to physical SQL expressions.

    Builds physical SQL AST nodes from logical metric, dimension, and column
    references using the OSI semantic model's resolution capabilities.
    """

    def __init__(self, parser, scope_mgr, classifier, col_transformer):
        self._parser = parser
        self._scope_mgr = scope_mgr
        self._classifier = classifier
        self._col_transformer = col_transformer

    # ------------------------------------------------------------------ #
    # Expression builders
    # ------------------------------------------------------------------ #

    def build_metric_expression(self, alias: str, physical) -> exp.Expression:
        if isinstance(physical, str):
            parsed = sqlglot.parse_one(physical)
            return exp.alias_(parsed, alias)
        return exp.alias_(physical, alias)

    def build_dimension_expression(self, alias: str, physical: str) -> exp.Expression:
        parts = physical.split('.')
        if len(parts) == 2:
            table, col = parts
            return exp.alias_(exp.column(col, table), alias)
        return exp.alias_(exp.column(physical), alias)

    def build_column_expression(self, alias: str, physical) -> exp.Expression:
        if isinstance(physical, str):
            parts = physical.split('.')
            if len(parts) == 2:
                table, col = parts
                return exp.alias_(exp.column(col, table), alias)
            return exp.alias_(exp.column(physical), alias)
        return exp.alias_(physical, alias)

    # ------------------------------------------------------------------ #
    # Field expansion
    # ------------------------------------------------------------------ #

    def expand_metric(self, item, alias: str, ctx) -> exp.Expression:
        name = self._classifier.get_item_name(item)
        metric_name = name or alias
        metric_expr = self._parser.resolve_metric(metric_name)
        parsed = sqlglot.parse_one(metric_expr)

        # Replace dataset-name table qualifiers with their SQL aliases
        known = {ds.name for ds in self._parser._model.datasets}
        for col in parsed.find_all(exp.Column):
            if col.table and col.table in known:
                scope_alias = self._col_transformer._scope_alias_for(ctx, col.table)
                if scope_alias:
                    col.set("table", scope_alias)

        return parsed

    def _is_cte_ref(self, table_ref: str, ctx) -> bool:
        """Return True if table_ref is a CTE / derived table, not a dataset or alias."""
        if not table_ref:
            return False
        known_sources = {ds.source for ds in self._parser._model.datasets}
        known_names = {ds.name for ds in self._parser._model.datasets}
        if table_ref in known_sources or table_ref in known_names:
            return False
        # Check if table_ref maps to a known source (table alias) vs self-mapping (CTE)
        resolved = ctx.table_alias_map.get(table_ref)
        if resolved and (resolved in known_sources or resolved in known_names):
            return False
        for scope in ctx.scope_stack:
            phys = scope.tables.get(table_ref)
            if phys and (phys in known_sources or phys in known_names):
                return False
        return True

    def expand_dimension(self, item, alias: str, ctx):
        name = self._classifier.get_item_name(item)
        field_name = name

        try:
            mapping = self._parser.resolve_field(field_name)
            physical_col = mapping.physical_expression

            table_ref = self.get_table_reference(item, ctx)

            if not table_ref and mapping.dataset_name:
                table_ref = ctx.get_table_alias(mapping.dataset_name)

            if not table_ref:
                current = self._scope_mgr.get_current_source(ctx)
                if current and current != 'cte':
                    table_ref = current

            if not table_ref:
                for src_alias, phys_name in ctx.table_alias_map.items():
                    try:
                        src = self._parser.get_dataset_source(src_alias)
                        if src and mapping.dataset_name:
                            for ds in self._parser._model.datasets:
                                if ds.name == mapping.dataset_name and ds.source == src:
                                    table_ref = phys_name
                                    break
                    except Exception:
                        pass
                    if table_ref:
                        break

            if any(op in physical_col for op in ['+', '-', '*', '/']):
                parsed_expr = sqlglot.parse_one(physical_col, dialect=None)
                for col in parsed_expr.find_all(exp.Column):
                    self._col_transformer.transform_column(col, ctx)
                return parsed_expr.sql()

            if table_ref:
                if self._is_cte_ref(table_ref, ctx):
                    # CTE columns are already logical aliases — pass through unchanged
                    return f"{table_ref}.{field_name}"
                return f"{table_ref}.{physical_col}"
            return physical_col

        except FieldNotFoundError:
            table_ref = self.get_table_reference(item, ctx)
            return f"{table_ref}.{field_name}" if table_ref else field_name

    def expand_column(self, item, alias: str, ctx):
        name = self._classifier.get_item_name(item)
        table_ref = self.get_table_reference(item, ctx)

        try:
            mapping = self._parser.resolve_field(name)
            physical_expr = mapping.physical_expression

            if mapping.is_dimension:
                table_for_col = table_ref or self.find_table_for_field(
                    mapping.dataset_name, ctx
                )
                if table_for_col:
                    return f"{table_for_col}.{physical_expr}"
                return physical_expr

            parsed = sqlglot.parse_one(physical_expr)
            table_for_col = table_ref or self.find_table_for_field(
                mapping.dataset_name, ctx
            )
            if table_for_col:
                self._add_table_prefix_to_expression(parsed, table_for_col)

            if physical_expr != name:
                return exp.alias_(parsed, alias)
            return parsed

        except FieldNotFoundError:
            if table_ref:
                return f"{table_ref}.{name}"
            return name

    # ------------------------------------------------------------------ #
    # Table reference helpers
    # ------------------------------------------------------------------ #

    def get_table_reference(self, item, ctx) -> str | None:
        # Unwrap Aliases to reach the inner Column
        inner = item
        if isinstance(item, exp.Alias):
            inner = item.this
        if isinstance(inner, exp.Column):
            table = inner.table
            if table:
                resolved = self._scope_mgr.resolve_table_alias(ctx, table)
                if resolved:
                    if resolved != table:
                        return table  # preserve alias
                    return resolved
                try:
                    return self._parser.get_dataset_source(table)
                except DatasetNotFoundError:
                    return table
        return None

    def find_table_for_field(self, dataset_name: str | None, ctx) -> str | None:
        if dataset_name:
            resolved = self._scope_mgr.resolve_table_alias(ctx, dataset_name)
            return resolved or dataset_name

        for alias in ctx.table_alias_map:
            return ctx.table_alias_map[alias]
        return None

    def _add_table_prefix_to_expression(self, expr, table: str):
        for node in expr.walk(bfs=False):
            if isinstance(node, exp.Column):
                if not node.table:
                    node.set("table", table)
