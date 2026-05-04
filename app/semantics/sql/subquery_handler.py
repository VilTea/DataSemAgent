from typing import Any, Callable

import sqlglot.expressions as exp

from app.semantics.sql.exceptions import DatasetNotFoundError
from app.semantics.sql.models import ColumnInfo, TranslationContext
from app.semantics.sql.parser import OSIModelParser
from app.semantics.sql.scope_manager import ScopeManager


class SubqueryHandler:
    """Handles transformation of subqueries, CTEs, set operations, and table sources.

    Manages scope isolation, outer reference tracking, and table alias
    save/restore patterns required for correct subquery translation.
    """

    def __init__(
        self,
        parser: OSIModelParser,
        scope_mgr: ScopeManager,
        strict: bool,
        on_transform_select: Callable,
        on_transform_node: Callable,
    ):
        self._parser = parser
        self._scope_mgr = scope_mgr
        self._strict = strict
        self._on_transform_select = on_transform_select
        self._on_transform_node = on_transform_node

    # ------------------------------------------------------------------ #
    # Source / table resolution
    # ------------------------------------------------------------------ #

    def resolve_sources(self, select: exp.Select, ctx: TranslationContext):
        current_scope = ctx.get_current_scope()

        from_clause = select.args.get("from_")
        if from_clause and from_clause.this:
            source = from_clause.this
            if isinstance(source, exp.Table):
                self._resolve_table(source, ctx, current_scope)

        if select.args.get("joins"):
            for join in select.args["joins"]:
                if join.this and isinstance(join.this, exp.Table):
                    self._resolve_table(join.this, ctx, current_scope)

    def _resolve_table(
        self, source: exp.Table, ctx: TranslationContext, current_scope
    ):
        table_name = source.name
        alias = source.alias or table_name

        # Check if table_name is already a physical name (reverse lookup)
        physical_by_reverse = None
        for log_name, phys_name in ctx.table_alias_map.items():
            if phys_name == table_name:
                physical_by_reverse = log_name
                break

        if physical_by_reverse:
            physical_name = table_name
            current_scope.tables[alias] = physical_name
            current_scope.tables[table_name] = physical_name
            source.set("this", exp.to_identifier(physical_name))
            return

        # Check if table_name or alias is already registered
        if table_name in ctx.table_alias_map or alias in ctx.table_alias_map:
            physical_name = (
                ctx.table_alias_map.get(table_name)
                or ctx.table_alias_map.get(alias)
                or table_name
            )
            current_scope.tables[alias] = physical_name
            current_scope.tables[table_name] = physical_name
            source.set("this", exp.to_identifier(physical_name))
            return

        # Resolve via parser and register
        try:
            physical_name = self._parser.get_dataset_source(table_name)
            self._scope_mgr.register_table(ctx, alias, physical_name)
            self._scope_mgr.register_table(ctx, table_name, physical_name)
            current_scope.tables[alias] = physical_name
            current_scope.tables[table_name] = physical_name
            source.set("this", exp.to_identifier(physical_name))
        except DatasetNotFoundError:
            if self._strict:
                raise
            self._scope_mgr.register_table(ctx, alias, alias)
            self._scope_mgr.register_table(ctx, table_name, alias)
            current_scope.tables[alias] = alias
            current_scope.tables[table_name] = alias

    # ------------------------------------------------------------------ #
    # Source (FROM / JOIN) transformation
    # ------------------------------------------------------------------ #

    def transform_source(self, source: Any, ctx: TranslationContext):
        if isinstance(source, exp.Subquery):
            if source.alias:
                ctx.push_scope(
                    {
                        source.alias: ColumnInfo(
                            physical_expr=source.alias,
                            logical_name=source.alias,
                        )
                    }
                )
            subquery = source.this
            ctx.push_scope()
            self._on_transform_select(subquery, ctx)
            ctx.pop_scope()
        elif isinstance(source, exp.Table):
            alias = source.alias or source.name
            if alias not in ctx.table_alias_map:
                try:
                    physical = self._parser.get_dataset_source(source.name)
                    ctx.set_table_alias(alias, physical)
                except DatasetNotFoundError:
                    pass

    # ------------------------------------------------------------------ #
    # Subquery transformation (scope isolation)
    # ------------------------------------------------------------------ #

    def transform_subquery(
        self, subquery: exp.Subquery, ctx: TranslationContext
    ) -> exp.Subquery:
        saved_table_alias_map = dict(ctx.table_alias_map)
        saved_outer_refs = ctx.outer_table_refs.copy()

        outer_tables = set(ctx.table_alias_map.keys())

        inner_select = subquery.this

        self.resolve_sources(inner_select, ctx)
        new_tables = {
            k: v
            for k, v in ctx.table_alias_map.items()
            if k not in outer_tables
        }

        ctx.table_alias_map = saved_table_alias_map
        for t in outer_tables:
            ctx.outer_table_refs.add(t)

        ctx.push_scope(tables=dict(ctx.table_alias_map))
        ctx.table_alias_map.update(new_tables)

        self._on_transform_select(inner_select, ctx)
        ctx.pop_scope()

        ctx.outer_table_refs = saved_outer_refs
        ctx.table_alias_map = saved_table_alias_map

        if subquery.alias:
            ctx.push_scope(
                {
                    subquery.alias: ColumnInfo(
                        physical_expr=subquery.alias,
                        logical_name=subquery.alias,
                    )
                }
            )

        return subquery

    # ------------------------------------------------------------------ #
    # CTE / WITH transformation
    # ------------------------------------------------------------------ #

    def transform_with(
        self, with_expr: exp.With, ctx: TranslationContext
    ) -> exp.With:
        new_expressions = []
        for cte in with_expr.expressions:
            if isinstance(cte, exp.CTE):
                cte_alias = cte.alias
                new_cte = cte.copy()
                new_cte.args["this"] = self._on_transform_node(cte.this, ctx)
                new_expressions.append(new_cte)

                if cte_alias:
                    ctx.set_table_alias(cte_alias, cte_alias)
                    ctx.push_scope(
                        {
                            cte_alias: ColumnInfo(
                                physical_expr=cte_alias,
                                logical_name=cte_alias,
                            )
                        }
                    )
            else:
                new_expressions.append(cte)

        with_expr.set("expressions", new_expressions)
        return with_expr

    @staticmethod
    def transform_cte(cte: exp.CTE) -> exp.CTE:
        return cte

    # ------------------------------------------------------------------ #
    # Set operation (UNION / INTERSECT / EXCEPT) transformation
    # ------------------------------------------------------------------ #

    def transform_set_operation(self, node: Any, ctx: TranslationContext):
        if node.left:
            left = node.left
            if isinstance(left, exp.Select):
                ctx.push_scope()
                self._on_transform_select(left, ctx)
                ctx.pop_scope()

        if hasattr(node, "right") and node.right:
            right = node.right
            if isinstance(right, exp.Select):
                ctx.push_scope()
                self._on_transform_select(right, ctx)
                ctx.pop_scope()

        return node
