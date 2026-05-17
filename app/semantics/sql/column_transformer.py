import sqlglot
import sqlglot.expressions as exp

from app.semantics.sql.exceptions import FieldNotFoundError, MetricNotFoundError


class ColumnTransformer:
    """Handles in-place AST transformations of Column nodes.

    Responsible for rewriting logical column references (dimension, metric, plain)
    to their physical equivalents in the sqlglot AST.
    """

    def __init__(self, parser, scope_mgr, classifier):
        self._parser = parser
        self._scope_mgr = scope_mgr
        self._classifier = classifier
        self._known_datasets = {ds.name for ds in parser._model.datasets}

    def _scope_alias_for(self, ctx, dataset_name: str) -> str | None:
        """Return the SQL alias (e.g. ``"ss"`` for ``FROM store_sales AS ss``).

        *dataset_name* is a logical dataset name (e.g. ``"store_sales"``).
        We resolve it to the physical source name first, then scan scope tables
        for a user-defined alias that maps to that same physical source.
        """
        physical_source = self._parser.get_dataset_source(dataset_name)
        known_names = self._known_datasets
        for scope in reversed(ctx.scope_stack):
            for alias, phys in scope.tables.items():
                if (
                    phys in (physical_source, dataset_name)
                    and alias != dataset_name
                    and alias not in known_names
                ):
                    return alias
        return None

    def transform_column(self, col: exp.Column, ctx) -> None:
        """Transform a column reference in-place to its physical representation."""
        alias = col.name

        if col.table:
            table_alias = ctx.get_table_alias(col.table)
            if table_alias:
                is_outer_ref = col.table in ctx.outer_table_refs
                is_logical_name = col.table in self._known_datasets

                is_dimension_field = self._parser.is_dimension(alias)
                is_metric_field = self._parser.is_metric(alias)

                if is_dimension_field:
                    try:
                        mapping = self._parser.resolve_field(alias)
                        pexpr = mapping.physical_expression
                        if pexpr != alias:
                            parsed = sqlglot.parse_one(pexpr, dialect=None)
                            ref = col.table or table_alias
                            for inner_col in parsed.find_all(exp.Column):
                                if ref and not inner_col.table:
                                    inner_col.set("table", ref)
                            self._replace_column_with_expression(col, parsed)
                        else:
                            col.set("this", exp.to_identifier(pexpr))
                    except FieldNotFoundError:
                        pass
                    if not is_outer_ref and is_logical_name:
                        col.set("table", table_alias)
                    return
                elif is_metric_field:
                    try:
                        mapping = self._parser.resolve_field(alias)
                        pexpr = mapping.physical_expression
                        if pexpr != alias:
                            parsed = sqlglot.parse_one(pexpr, dialect=None)
                            ref = col.table or table_alias
                            for inner_col in parsed.find_all(exp.Column):
                                if ref and not inner_col.table:
                                    inner_col.set("table", ref)
                            self._replace_column_with_expression(col, parsed)
                        else:
                            col.set("this", exp.to_identifier(pexpr))
                    except FieldNotFoundError:
                        pass

                if not is_outer_ref and is_logical_name:
                    col.set("table", table_alias)
                return

        outer_ref = ctx.resolve_column(alias)
        if outer_ref:
            # Don't mangle metric expressions — they're SQL fragments, not column names
            if outer_ref.is_metric:
                col.set("this", exp.to_identifier(alias))
                return
            col_name = outer_ref.physical_expr
            parsed = sqlglot.parse_one(col_name, dialect=None)
            if isinstance(parsed, exp.Column):
                col_name = parsed.name
            col.set("this", exp.to_identifier(col_name))
            return

        if self._parser.is_dimension(alias):
            # When the current source is a CTE / subquery, the column comes
            # from the derived table's output — do NOT expand OSI expressions.
            if not col.table and self._is_current_source_cte(ctx):
                return
            try:
                mapping = self._parser.resolve_field(alias)
                physical_col = mapping.physical_expression
                table_ref = None
                if col.table and self._is_scoped_alias(col.table, ctx):
                    table_ref = col.table
                elif mapping.dataset_name:
                    table_ref = ctx.get_table_alias(mapping.dataset_name)
                    # Prefer SQL alias (e.g. 'ss') over physical table name
                    scope_alias = self._scope_alias_for(ctx, mapping.dataset_name)
                    if scope_alias:
                        table_ref = scope_alias
                    if not table_ref:
                        # If the current source's dataset also defines this field,
                        # prefer it (handles duplicate field names across datasets).
                        # Otherwise use the field's actual dataset source — BUT only
                        # when the current source is a base table (not a CTE/subquery).
                        if not self._current_source_has_field_for(ctx, alias):
                            if not self._is_current_source_cte(ctx):
                                try:
                                    table_ref = self._parser.get_dataset_source(mapping.dataset_name)
                                except Exception:
                                    pass
                if not table_ref:
                    table_ref = ctx.get_current_source() or ctx.get_current_table()
                    if table_ref:
                        # Prefer SQL alias over physical name (e.g. 'o' over 'stg_orders'),
                        # but NOT dataset names that happen to map to the same physical source.
                        for alias, phys in ctx.table_alias_map.items():
                            if (phys == table_ref and alias != table_ref
                                    and alias not in self._known_datasets):
                                table_ref = alias
                                break
                if physical_col != alias:
                    parsed = sqlglot.parse_one(physical_col, dialect=None)
                    for inner_col in parsed.find_all(exp.Column):
                        if table_ref and not inner_col.table:
                            inner_col.set("table", table_ref)
                    self._replace_column_with_expression(col, parsed)
                else:
                    if table_ref:
                        col.set("table", table_ref)
                    col.set("this", exp.to_identifier(physical_col))
            except FieldNotFoundError:
                pass
        elif not self._parser.is_metric(alias):
            # Only replace the table when col.table is a bare dataset NAME
            # (e.g. 'store_sales') — preserve user aliases and CTE names.
            if col.table and col.table in self._known_datasets:
                table_ref = ctx.get_table_alias(col.table)
                if table_ref:
                    scope_alias = self._scope_alias_for(ctx, col.table)
                    col.set("table", scope_alias or table_ref)
            elif not col.table:
                try:
                    mapping = self._parser.resolve_field(alias)
                    physical_col = mapping.physical_expression
                    table_for_col = self._find_table_for_field(mapping.dataset_name, ctx)
                    if table_for_col:
                        col.set("table", table_for_col)
                    col.set("this", exp.to_identifier(physical_col))
                except FieldNotFoundError:
                    pass

    def transform_column_in_condition(self, col: exp.Column, ctx) -> None:
        """Transform a column reference found in a WHERE/JOIN condition."""
        alias = col.name

        if self._parser.is_metric(alias):
            self.transform_metric_column(col, ctx)
            return

        if not col.table:
            self.transform_column(col, ctx)
            return

        if col.table in ctx.outer_table_refs:
            return

        if not self._scope_mgr.is_table_in_current_scope(ctx, col.table):
            return

        self.transform_column(col, ctx)

    def transform_outer_ref_column(self, col: exp.Column, ctx) -> None:
        """Transform a column reference from an outer scope (correlated subquery)."""
        alias = col.name
        if self._parser.is_dimension(alias):
            try:
                mapping = self._parser.resolve_field(alias)
                col.set("this", exp.to_identifier(mapping.physical_expression))
                if col.table:
                    if col.table in self._known_datasets:
                        table_alias = ctx.get_table_alias(col.table)
                        if table_alias:
                            col.set("table", table_alias)
            except FieldNotFoundError:
                pass

    def transform_metric_column(self, col: exp.Column, ctx) -> None:
        """Replace a metric column reference with its expanded expression in-place."""
        alias = col.name
        if not self._parser.is_metric(alias):
            return

        try:
            metric_expr = self._parser.resolve_metric(alias)
            parsed = sqlglot.parse_one(metric_expr)

            for node in parsed.walk(bfs=False):
                if isinstance(node, exp.Column):
                    if node.table:
                        scope_alias = self._scope_alias_for(ctx, node.table)
                        if scope_alias:
                            node.set("table", scope_alias)
                    else:
                        table_ref = ctx.get_current_table()
                        if table_ref:
                            node.set("table", table_ref)

            self._replace_column_with_expression(col, parsed)
        except MetricNotFoundError:
            pass

    def _replace_column_with_expression(self, col: exp.Column, replacement: exp.Expression):
        """Replace a column AST node with an arbitrary expression in its parent."""
        parent = col.parent
        if parent is None:
            return

        if isinstance(parent, exp.Expression):
            for key in parent.args:
                value = parent.args.get(key)
                if value is col:
                    parent.set(key, replacement)
                    return
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        if item is col:
                            value[i] = replacement
                            return

        for key in parent.args:
            self._replace_in_node(parent.args[key], col, replacement)

    def _replace_in_node(self, node, target: exp.Column, replacement: exp.Expression):
        """Recursively search and replace a target column node within a subtree."""
        if node is target:
            return

        if isinstance(node, exp.Expression):
            for key in node.args:
                value = node.args[key]
                if value is target:
                    node.set(key, replacement)
                    return
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        if item is target:
                            value[i] = replacement
                            return
                elif isinstance(value, exp.Expression):
                    self._replace_in_node(value, target, replacement)

    @staticmethod
    def _is_scoped_alias(table: str, ctx) -> bool:
        """True if *table* is a known alias (JOIN/CTE/subquery), not a bare dataset name."""
        return table is not None and table in ctx.table_alias_map

    def _is_current_source_cte(self, ctx) -> bool:
        """True when the current source is a CTE / derived table, not a base dataset."""
        current = ctx.get_current_source()
        if not current:
            return False
        known_sources = {ds.source for ds in self._parser._model.datasets}
        known_names = self._known_datasets
        if current in known_sources or current in known_names:
            return False
        resolved = ctx.table_alias_map.get(current)
        if resolved and (resolved in known_sources or resolved in known_names):
            return False
        for scope in ctx.scope_stack:
            phys = scope.tables.get(current)
            if phys and (phys in known_sources or phys in known_names):
                return False
        return True

    def _current_source_has_field_for(self, ctx, field_name: str) -> bool:
        """Return True if *field_name* is defined on the current source's dataset."""
        current = ctx.get_current_source()
        if not current:
            return False
        # Resolve alias to physical/dataset name (e.g. 'o' → 'stg_orders' / 'orders')
        resolved = ctx.table_alias_map.get(current, current)
        for ds in self._parser._model.datasets:
            if ds.source == resolved or ds.name == resolved:
                if ds.fields:
                    for f in ds.fields:
                        if f.name == field_name:
                            return True
                break
        return False

    def _find_table_for_field(self, dataset_name: str | None, ctx) -> str | None:
        """Find the best table reference for *dataset_name* — prefers SQL alias."""
        if dataset_name:
            scope_alias = self._scope_alias_for(ctx, dataset_name)
            if scope_alias:
                return scope_alias
            resolved = self._scope_mgr.resolve_table_alias(ctx, dataset_name)
            return resolved or dataset_name

        for alias in ctx.table_alias_map:
            return ctx.table_alias_map[alias]
        return None
