from dataclasses import dataclass
from typing import Any

import sqlglot
import sqlglot.expressions as exp

ANSI_DIALECT = None

from app.semantics.sql.classifier import FieldClassifier
from app.semantics.sql.column_transformer import ColumnTransformer
from app.semantics.sql.exceptions import (
    FieldNotFoundError,
)
from app.semantics.sql.models import ColumnInfo, TranslationContext
from app.semantics.sql.parser import OSIModelParser
from app.semantics.sql.scope_manager import ScopeManager
from app.semantics.sql.expander import FieldExpander
from app.semantics.sql.subquery_handler import SubqueryHandler
from app.semantics.sql.clause_transformer import ClauseTransformer


@dataclass
class TranslationResult:
    physical_sql: str
    original_sql: str


class SQLTranslator:
    def __init__(self, parser: OSIModelParser, strict: bool = True):
        self._parser = parser
        self._strict = strict
        self._known_datasets = {ds.name for ds in parser._model.datasets}
        self._scope_mgr = ScopeManager(parser)
        self._classifier = FieldClassifier(parser, strict=strict)
        self._col_transformer = ColumnTransformer(parser, self._scope_mgr, self._classifier)
        self._expander = FieldExpander(parser, self._scope_mgr, self._classifier, self._col_transformer)
        self._subquery_handler = SubqueryHandler(
            parser, self._scope_mgr, strict,
            on_transform_select=self._transform_select,
            on_transform_node=self._transform_node,
        )
        self._clause_transformer = ClauseTransformer(parser, self._col_transformer, self._subquery_handler)

    def translate(self, logical_sql: str) -> TranslationResult:
        parsed = sqlglot.parse_one(logical_sql, dialect=ANSI_DIALECT)
        ctx = self._parser.create_translation_context()
        
        transformed = self._transform_node(parsed, ctx)
        physical_sql = transformed.sql(dialect=ANSI_DIALECT)
        
        return TranslationResult(
            physical_sql=physical_sql,
            original_sql=logical_sql,
        )

    def _transform_node(self, node: Any, ctx: TranslationContext):
        if isinstance(node, exp.Select):
            return self._transform_select(node, ctx)
        elif isinstance(node, (exp.Union, exp.Intersect, exp.Except)):
            return self._subquery_handler.transform_set_operation(node, ctx)
        elif isinstance(node, exp.Subquery):
            return self._subquery_handler.transform_subquery(node, ctx)
        elif isinstance(node, exp.With):
            return self._subquery_handler.transform_with(node, ctx)
        elif isinstance(node, exp.CTE):
            return self._subquery_handler.transform_cte(node)
        else:
            return self._transform_children(node, ctx)

    def _transform_select(self, select: exp.Select, ctx: TranslationContext):
        with_expr = select.args.get("with_")
        if with_expr:
            select.set('with_', self._subquery_handler.transform_with(with_expr, ctx))
        
        ctx.push_scope()
        
        self._subquery_handler.resolve_sources(select, ctx)

        self._clause_transformer.transform_from(select, ctx)
        self._clause_transformer.transform_joins(select, ctx)

        from_clause = select.args.get("from_")
        if from_clause and from_clause.this:
            table = from_clause.this
            for alias, physical in ctx.table_alias_map.items():
                if physical == table.name:
                    self._scope_mgr.set_current_source(ctx, alias)
                    break
            else:
                table_name = table.alias or table.name
                self._scope_mgr.set_current_source(ctx, table_name)

        self._transform_select_list(select, ctx)
        self._clause_transformer.transform_where(select, ctx)
        self._clause_transformer.transform_group_by(select, ctx)
        self._clause_transformer.transform_having(select, ctx)
        self._clause_transformer.transform_order_by(select, ctx)
        self._clause_transformer.transform_limit(select, ctx)

        ctx.pop_scope()

        return select

    def _transform_select_list(self, select: exp.Select, ctx: TranslationContext):
        new_expressions = []
        current_scope = ctx.get_current_scope()
        
        for item in select.expressions:
            if isinstance(item, exp.Star):
                new_expressions.append(item)
                continue
            
            if isinstance(item, exp.Subquery):
                transformed_subquery = self._subquery_handler.transform_subquery(item, ctx)
                alias = item.alias or ""
                if alias:
                    new_expr = exp.alias_(transformed_subquery, alias)
                else:
                    new_expr = transformed_subquery
                new_expressions.append(new_expr)
                if alias:
                    current_scope[alias] = ColumnInfo(
                        physical_expr=alias,
                        logical_name=alias,
                    )
                continue
            
            if isinstance(item, exp.Alias) and isinstance(item.this, exp.Subquery):
                inner_subquery = item.this
                transformed_subquery = self._subquery_handler.transform_subquery(inner_subquery, ctx)
                new_expr = exp.alias_(transformed_subquery, item.alias)
                new_expressions.append(new_expr)
                if item.alias:
                    current_scope[item.alias] = ColumnInfo(
                        physical_expr=item.alias,
                        logical_name=item.alias,
                    )
                continue
            
            if isinstance(item, exp.Alias) and isinstance(item.this, exp.Window):
                window = item.this
                partition = window.args.get("partition_by")
                if partition:
                    for expr in partition:
                        for node in expr.walk(bfs=False):
                            if isinstance(node, exp.Column):
                                self._col_transformer.transform_column(node, ctx)
                order = window.args.get("order")
                if order:
                    for node in order.walk(bfs=False):
                        if isinstance(node, exp.Column):
                            self._col_transformer.transform_column(node, ctx)
                alias = item.alias or ""
                new_expr = exp.alias_(window, alias)
                new_expressions.append(new_expr)
                if alias:
                    current_scope[alias] = ColumnInfo(
                        physical_expr=alias,
                        logical_name=alias,
                    )
                continue
            
            if isinstance(item, exp.Window):
                new_expressions.append(item)
                current_scope[item.name] = ColumnInfo(
                    physical_expr=f"{item.table}.{item.name}",
                    logical_name=item.name,
                )
                continue
            
            alias = self._get_alias(item)
            col_type, col_info = self._classifier.classify(item, alias, ctx, self._scope_mgr)
            
            if col_type == "metric":
                physical = self._expander.expand_metric(item, alias, ctx)
                physical_str = physical.sql() if hasattr(physical, 'sql') else str(physical)
                new_expr = self._expander.build_metric_expression(alias, physical)
            elif col_type == "dimension":
                physical = self._expander.expand_dimension(item, alias, ctx)
                physical_str = physical if isinstance(physical, str) else str(physical)
                new_expr = self._expander.build_dimension_expression(alias, physical_str)
            elif col_type == "outer_ref":
                if isinstance(item, exp.Column) and item.table:
                    known_datasets = {ds.name for ds in self._parser._model.datasets}
                    if item.table in known_datasets:
                        physical_table = ctx.get_table_alias(item.table)
                        if physical_table:
                            new_expr = exp.column(item.name, physical_table)
                        else:
                            new_expr = exp.column(item.name, item.table)
                    elif item.table in ctx.table_alias_map:
                        physical_table = ctx.table_alias_map.get(item.table)
                        if physical_table and physical_table != item.table:
                            new_expr = exp.column(item.name, physical_table)
                        else:
                            new_expr = exp.column(item.name, item.table)
                    else:
                        new_expr = exp.column(item.name, item.table)
                else:
                    new_expr = exp.column(alias)
                physical_str = new_expr.sql() if hasattr(new_expr, 'sql') else str(new_expr)
                new_expr.set("alias", alias)
            else:
                # CTE / subquery alias columns — pass through without OSI resolution.
                # Dataset aliases (like 'o' for 'orders') map to known physical sources;
                # true CTE/subquery aliases map to themselves or unknown values.
                if (isinstance(item, exp.Column) and item.table
                        and item.table in ctx.table_alias_map
                        and item.table not in self._known_datasets
                        and (ctx.table_alias_map[item.table] == item.table
                             or ctx.table_alias_map[item.table] not in
                             {ds.source for ds in self._parser._model.datasets})):
                    self._clause_transformer.transform_expression(item, ctx)
                    physical_str = item.sql() if hasattr(item, 'sql') else str(item)
                    new_expr = item
                elif hasattr(item, 'this') and isinstance(item.this, exp.Column):
                    new_expr = self._clause_transformer.transform_function(item, alias, ctx)
                    physical_str = new_expr.sql() if hasattr(new_expr, 'sql') else str(new_expr)
                else:
                    field_name = alias if alias else self._classifier.get_item_name(item)
                    try:
                        mapping = self._parser.resolve_field(field_name)
                        physical_expr = mapping.physical_expression
                        
                        parsed_expr = sqlglot.parse_one(physical_expr, dialect=None)
                        if not isinstance(parsed_expr, exp.Column):
                            table_ref = self._expander.get_table_reference(item, ctx)
                            if not table_ref:
                                table_ref = self._scope_mgr.get_current_source(ctx)
                            if not table_ref and mapping.dataset_name:
                                table_ref = ctx.get_table_alias(mapping.dataset_name)
                            
                            for col in parsed_expr.find_all(exp.Column):
                                if table_ref:
                                    col.set('table', table_ref)
                            
                            physical_str = parsed_expr.sql()
                            new_expr = exp.alias_(parsed_expr, alias) if alias else parsed_expr
                        else:
                            self._clause_transformer.transform_expression(item, ctx)
                            physical_str = item.sql() if hasattr(item, 'sql') else str(item)
                            new_expr = item
                    except FieldNotFoundError:
                        if self._strict:
                            plain_col = (isinstance(item, exp.Column)
                                         or (isinstance(item, exp.Alias)
                                             and isinstance(item.this, exp.Column)))
                            known_sources = {ds.source for ds in self._parser._model.datasets}
                            in_cte_scope = any(
                                a not in self._known_datasets and a not in known_sources
                                for a in ctx.table_alias_map
                            )
                            if plain_col and not in_cte_scope:
                                if field_name in self._parser.list_metrics():
                                    from app.semantics.sql.exceptions import MetricNotFoundError
                                    raise MetricNotFoundError(field_name)
                                raise
                        self._clause_transformer.transform_expression(item, ctx)
                        physical_str = item.sql() if hasattr(item, 'sql') else str(item)
                        new_expr = item
                        self._clause_transformer.transform_expression(item, ctx)
                        physical_str = item.sql() if hasattr(item, 'sql') else str(item)
                        new_expr = item
            
            new_expressions.append(new_expr)
            
            current_scope[alias] = ColumnInfo(
                physical_expr=physical_str,
                logical_name=alias,
                is_dimension=(col_type == "dimension"),
                is_metric=(col_type == "metric"),
            )
        
        if new_expressions:
            select.set("expressions", new_expressions)

    def _get_alias(self, item: Any) -> str:
        if hasattr(item, "alias") and item.alias:
            return item.alias
        if hasattr(item, "this") and isinstance(item.this, exp.Alias):
            return item.this.alias
        if hasattr(item, "name"):
            return item.name or ""
        return str(item)


    def _transform_children(self, node: Any, ctx: TranslationContext):
        for key, value in node.args.items():
            if isinstance(value, list):
                new_list = []
                for item in value:
                    if isinstance(item, sqlglot.exp.Expression):
                        new_list.append(self._transform_node(item, ctx))
                    else:
                        new_list.append(item)
                node.set(key, new_list)
            elif isinstance(value, sqlglot.exp.Expression):
                node.set(key, self._transform_node(value, ctx))
        return node
