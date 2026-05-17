from typing import Any

import sqlglot.expressions as exp

from app.semantics.sql.exceptions import FieldNotFoundError
from app.semantics.sql.models import TranslationContext
from app.semantics.sql.parser import OSIModelParser


class ClauseTransformer:
    """Transforms individual SQL clauses for logical-to-physical mapping.

    Handles FROM/JOIN, WHERE, GROUP BY, HAVING, ORDER BY, LIMIT, and
    expression-level transformations including function calls.
    """

    def __init__(self, parser: OSIModelParser, col_transformer, subquery_handler):
        self._parser = parser
        self._col_transformer = col_transformer
        self._subquery_handler = subquery_handler

    # ------------------------------------------------------------------ #
    # FROM / JOIN transformation
    # ------------------------------------------------------------------ #

    def transform_from(self, select: exp.Select, ctx: TranslationContext):
        from_clause = select.args.get("from_")
        if from_clause and from_clause.this:
            self._subquery_handler.transform_source(from_clause.this, ctx)

    def transform_joins(self, select: exp.Select, ctx: TranslationContext):
        if select.args.get("joins"):
            for join in select.args["joins"]:
                self.transform_join(join, ctx)

    def transform_join(self, join: exp.Join, ctx: TranslationContext):
        if join.this:
            self._subquery_handler.transform_source(join.this, ctx)

        join_on = join.args.get("on")
        if join_on:
            self.transform_condition(join_on, ctx)

    # ------------------------------------------------------------------ #
    # WHERE / condition transformation
    # ------------------------------------------------------------------ #

    def transform_where(self, select: exp.Select, ctx: TranslationContext):
        where_clause = select.args.get("where")
        if where_clause:
            self.transform_condition(where_clause, ctx)

    def transform_condition(self, condition: Any, ctx: TranslationContext):
        subqueries = []
        subquery_nodes = set()

        for node in condition.walk(bfs=False):
            if isinstance(node, (exp.Subquery, exp.Exists)):
                subqueries.append(node)
                for n in node.walk():
                    subquery_nodes.add(id(n))

        for col_node in condition.walk(bfs=False):
            if isinstance(col_node, exp.Column) and id(col_node) not in subquery_nodes:
                if col_node.table in ctx.outer_table_refs:
                    self._col_transformer.transform_outer_ref_column(col_node, ctx)
                else:
                    self._col_transformer.transform_column_in_condition(col_node, ctx)

        for subquery in subqueries:
            self._subquery_handler.transform_subquery(subquery, ctx)

    # ------------------------------------------------------------------ #
    # GROUP BY transformation
    # ------------------------------------------------------------------ #

    def transform_group_by(self, select: exp.Select, ctx: TranslationContext):
        group = select.args.get("group")
        if group:
            for expr in group.expressions:
                self._transform_group_expr(expr, ctx)

    def _transform_group_expr(self, expr: Any, ctx: TranslationContext):
        if isinstance(expr, exp.Column):
            self._col_transformer.transform_column(expr, ctx)
        elif isinstance(expr, exp.Paren):
            for node in expr.walk(bfs=False):
                if isinstance(node, exp.Column):
                    self._col_transformer.transform_column(node, ctx)

    # ------------------------------------------------------------------ #
    # HAVING transformation
    # ------------------------------------------------------------------ #

    def transform_having(self, select: exp.Select, ctx: TranslationContext):
        having_clause = select.args.get("having")
        if having_clause:
            for node in having_clause.walk(bfs=False):
                if isinstance(node, exp.Column):
                    if self._parser.is_metric(node.name):
                        self._col_transformer.transform_metric_column(node, ctx)
                    else:
                        self._col_transformer.transform_column(node, ctx)

    # ------------------------------------------------------------------ #
    # ORDER BY transformation
    # ------------------------------------------------------------------ #

    def transform_order_by(self, select: exp.Select, ctx: TranslationContext):
        order = select.args.get("order")
        if order:
            for item in order.expressions:
                if isinstance(item.this, exp.Column):
                    if self._parser.is_metric(item.this.name):
                        self._col_transformer.transform_metric_column(item.this, ctx)
                    else:
                        self._col_transformer.transform_column(item.this, ctx)

    # ------------------------------------------------------------------ #
    # LIMIT transformation (no-op)
    # ------------------------------------------------------------------ #

    @staticmethod
    def transform_limit(select: exp.Select, ctx: TranslationContext):
        pass

    # ------------------------------------------------------------------ #
    # Expression transformation
    # ------------------------------------------------------------------ #

    def transform_expression(self, expr: Any, ctx: TranslationContext) -> None:
        for node in expr.walk(bfs=False):
            if isinstance(node, exp.Column):
                self._col_transformer.transform_column(node, ctx)
        # Expand metric references inside expressions (e.g. ROUND(total_profit / total_sales * 100, 2))
        for node in expr.walk(bfs=False):
            if isinstance(node, exp.Column) and self._parser.is_metric(node.name):
                self._col_transformer.transform_metric_column(node, ctx)

    def transform_function(
        self, func: Any, alias: str, ctx: TranslationContext
    ) -> exp.Expression:
        if hasattr(func, "this") and isinstance(func.this, exp.Column):
            col = func.this
            name = col.name

            table_ref = None
            # Only resolve the table when col.table is a bare dataset NAME
            # (e.g. 'store_sales') — preserve user aliases and CTE names.
            if col.table and col.table in self._col_transformer._known_datasets:
                table_ref = ctx.get_table_alias(col.table)
                scope_alias = self._col_transformer._scope_alias_for(ctx, col.table)
                if scope_alias:
                    table_ref = scope_alias

            if not table_ref:
                try:
                    mapping = self._parser.resolve_field(name)
                    if mapping.dataset_name:
                        table_ref = ctx.table_alias_map.get(mapping.dataset_name)
                except FieldNotFoundError:
                    pass

            if table_ref:
                col.set("table", table_ref)

            if self._parser.is_dimension(name):
                try:
                    mapping = self._parser.resolve_field(name)
                    col.set("this", exp.to_identifier(mapping.physical_expression))
                except FieldNotFoundError:
                    pass
            elif not self._parser.is_metric(name):
                try:
                    mapping = self._parser.resolve_field(name)
                    col.set("this", exp.to_identifier(mapping.physical_expression))
                except FieldNotFoundError:
                    pass

        if alias and not func.alias:
            return exp.alias_(func, alias)
        return func
