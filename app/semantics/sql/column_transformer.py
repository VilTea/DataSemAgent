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
                        col.set("this", exp.to_identifier(mapping.physical_expression))
                    except FieldNotFoundError:
                        pass
                    if not is_outer_ref and is_logical_name:
                        col.set("table", table_alias)
                    return
                elif is_metric_field:
                    try:
                        mapping = self._parser.resolve_field(alias)
                        col.set("this", exp.to_identifier(mapping.physical_expression))
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
            col.set("this", exp.to_identifier(
                outer_ref.physical_expr.split('.')[-1] if '.' in outer_ref.physical_expr else outer_ref.physical_expr
            ))
            return

        if self._parser.is_dimension(alias):
            try:
                mapping = self._parser.resolve_field(alias)
                physical_col = mapping.physical_expression
                table_ref = None
                if mapping.dataset_name:
                    table_ref = ctx.get_table_alias(mapping.dataset_name)
                if not table_ref:
                    table_ref = ctx.get_current_table()
                if table_ref:
                    col.set("table", table_ref)
                col.set("this", exp.to_identifier(physical_col))
            except FieldNotFoundError:
                pass
        elif not self._parser.is_metric(alias):
            table_ref = ctx.get_table_alias(col.table) if col.table else None
            if table_ref:
                col.set("table", table_ref)
            else:
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

    def _find_table_for_field(self, dataset_name: str | None, ctx) -> str | None:
        """Find the physical table alias for a given dataset name."""
        if dataset_name:
            resolved = self._scope_mgr.resolve_table_alias(ctx, dataset_name)
            return resolved or dataset_name

        for alias in ctx.table_alias_map:
            return ctx.table_alias_map[alias]
        return None
