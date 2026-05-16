"""
OSI SQL Validator

校验逻辑 SQL 是否违反规约，不做下推。
每条 Rule 类负责自身校验和错误信息格式化。
"""

from __future__ import annotations

from abc import abstractmethod, ABC

import sqlglot
import sqlglot.expressions as exp
from sqlglot.optimizer.scope import build_scope, Scope

from app.exceptions import DataSemAgentError
from app.semantics.models import SemanticModel


class SQLValidationError(DataSemAgentError):
    """SQL 违反规约时抛出"""

    def __init__(self, message: str, rule: str = "", sql: str = ""):
        self.rule = rule
        self.sql = sql
        super().__init__(message)


# =========================================================================
# Rule base class
# =========================================================================

class Rule(ABC):
    """校验规则基类。每条规则负责自身校验和错误信息格式化。"""
    id: str = ""
    name: str = ""
    description: str = ""

    @abstractmethod
    def check(self, select: exp.Select, scope: Scope, validator: "SQLValidator") -> list[str]:
        """执行校验，返回错误消息列表。空列表表示通过。"""
        raise NotImplementedError

    def _error(self, msg: str) -> str:
        """格式化错误信息"""
        return f"[{self.id}] {msg}"


# =========================================================================
# Rule implementations
# =========================================================================

class NoReAggregationRule(Rule):
    """R1: 指标列不可再聚合（含窗口函数）"""
    id = "R1"
    name = "指标列不可再聚合"
    description = "指标列已是聚合结果，禁止再套聚合函数（含窗口函数）"

    def check(self, select: exp.Select, scope: Scope, validator: "SQLValidator") -> list[str]:
        for item in select.expressions:
            for node in item.find_all(exp.AggFunc, bfs=False):
                for arg in node.walk(bfs=False):
                    if isinstance(arg, exp.Column) and validator.is_metric(arg.name):
                        in_window = False
                        curr = arg.parent
                        while curr:
                            if isinstance(curr, exp.Window):
                                in_window = True
                                break
                            curr = curr.parent
                        suffix = "(窗口函数)" if in_window else ""
                        return [
                            self._error(f"指标列 {arg.name} 不可再被聚合函数 {type(node).__name__}{suffix} 包裹")
                        ]
        return []


class MetricGroupByRule(Rule):
    """R2: 指标列必须有维度列 GROUP BY"""
    id = "R2"
    name = "指标列必须有维度列 GROUP BY"
    description = "SELECT 中出现指标列时，所有维度列必须在 GROUP BY 中"

    def check(self, select: exp.Select, scope: Scope, validator: "SQLValidator") -> list[str]:
        selected_dims, alias_map, has_metric = validator.analyze_select_list(select)
        if not has_metric:
            return []

        # If all metric references are through subquery/CTE aliases, aggregation
        # already happened in the subquery — no GROUP BY needed.
        if validator._all_metrics_from_subquery(select, scope):
            return []

        group_cols = validator.get_group_by_columns(select, alias_map)
        missing = selected_dims - group_cols
        if missing:
            missing_str = ", ".join(sorted(missing))
            return [
                self._error(f"SELECT 中包含指标列，但维度列 [{missing_str}] 未在 GROUP BY 中")
            ]
        return []


class CrossDatasetMetricRule(Rule):
    """R3: 跨数据集指标的表必须在 FROM 里"""
    id = "R3"
    name = "跨数据集指标的表必须在 FROM 里"
    description = "指标列涉及多表时，所有涉及的表必须在当前作用域的 FROM/JOIN 中"

    def check(self, select: exp.Select, scope: Scope, validator: "SQLValidator") -> list[str]:
        from_tables = validator._collect_from_tables(scope)
        errors = []

        for item in select.expressions:
            for col in item.walk(bfs=False):
                if not isinstance(col, exp.Column) or not validator.is_metric(col.name):
                    continue
                # Metric accessed through a subquery/CTE alias (e.g. curr.total_profit)
                # — the subquery already handles the metric's table dependencies.
                if col.table and validator._is_scope_source(scope, col.table):
                    continue
                metric_name = col.name
                required_tables = validator.get_metric_tables(metric_name)
                if not required_tables:
                    continue
                missing = required_tables - from_tables
                if missing:
                    missing_str = ", ".join(sorted(missing))
                    errors.append(
                        self._error(f"指标 {metric_name} 需要表 [{missing_str}]，但它们不在当前作用域的 FROM/JOIN 中")
                    )
        return errors


class NoMetricInWhereRule(Rule):
    """R4: 指标列不能在 WHERE 中使用"""
    id = "R4"
    name = "指标列不能在 WHERE 中使用"
    description = "指标列是聚合结果，应在 HAVING 中过滤；例外：FROM 包含 CTE/子查询时合法"

    def check(self, select: exp.Select, scope: Scope, validator: "SQLValidator") -> list[str]:
        # FROM 有 CTE 或派生表时，指标列已计算完成，WHERE 中使用合法
        for source in scope.sources.values():
            if isinstance(source, Scope):
                return []

        where_clause = select.args.get("where")
        if not where_clause:
            return []

        for col in where_clause.walk(bfs=False):
            if isinstance(col, exp.Column) and validator.is_metric(col.name):
                return [
                    self._error(f"指标列 {col.name} 不能在 WHERE 中使用，应使用 HAVING 过滤聚合结果")
                ]
        return []


class NoMetricInGroupByRule(Rule):
    """R5: 指标列不能在 GROUP BY 中使用"""
    id = "R5"
    name = "指标列不能在 GROUP BY 中使用"
    description = "指标列是聚合结果，GROUP BY 只能使用维度列"

    def check(self, select: exp.Select, scope: Scope, validator: "SQLValidator") -> list[str]:
        group = select.args.get("group")
        if not group:
            return []

        for expr in group.expressions:
            for col in expr.find_all(exp.Column, bfs=False):
                if validator.is_metric(col.name):
                    return [
                        self._error(f"指标列 {col.name} 不能在 GROUP BY 中使用，只能对维度列分组")
                    ]
        return []


# =========================================================================
# Validator
# =========================================================================

# 规约注册表 - 新增规则只需在此添加
RULES: list[Rule] = [
    NoReAggregationRule(),
    MetricGroupByRule(),
    CrossDatasetMetricRule(),
    NoMetricInWhereRule(),
    NoMetricInGroupByRule(),
]


class SQLValidator:
    """基于 OSI 规约校验逻辑 SQL"""

    def __init__(self, model: SemanticModel, rules: list[Rule] | None = None):
        self._model = model
        self._metric_names: set[str] = set()
        self._metric_tables: dict[str, set[str]] = {}
        self._dimension_names: set[str] = set()
        self._dataset_names = {ds.name for ds in model.datasets}
        self._rules = rules or RULES
        self._build_indexes()

    def _build_indexes(self):
        for metric in self._model.metrics or []:
            self._metric_names.add(metric.name)

            if metric.expression.dialects:
                expr = metric.expression.dialects[0].expression
                tables = set()
                for ds in self._model.datasets:
                    if f"{ds.name}." in expr:
                        tables.add(ds.name)
                self._metric_tables[metric.name] = tables

        for ds in self._model.datasets:
            for field in ds.fields or []:
                if field.dimension is not None:
                    self._dimension_names.add(field.name)

    # =========================================================================
    # Public API
    # =========================================================================

    def validate(self, logical_sql: str) -> list[str]:
        """校验逻辑 SQL，返回错误列表。空列表表示合规。"""
        errors: list[str] = []
        parsed = sqlglot.parse_one(logical_sql)
        root = build_scope(parsed)
        if root is None:
            return errors
        self._validate_scope(root, errors)
        return errors

    def validate_strict(self, logical_sql: str) -> None:
        """严格校验，遇到第一个违规即抛出异常。"""
        errors = self.validate(logical_sql)
        if errors:
            raise SQLValidationError(
                message=errors[0],
                rule="OSI SQL 规约",
                sql=logical_sql,
            )

    # =========================================================================
    # Scope traversal
    # =========================================================================

    def _validate_scope(self, scope: Scope, errors: list[str]) -> None:
        """遍历所有作用域，逐一应用规则"""
        for child_scope in scope.traverse():
            if isinstance(child_scope.expression, exp.Select):
                for rule in self._rules:
                    rule_errors = rule.check(child_scope.expression, child_scope, self)
                    errors.extend(rule_errors)

    # =========================================================================
    # Helpers
    # =========================================================================

    def is_metric(self, name: str) -> bool:
        return name in self._metric_names

    def is_dimension(self, name: str) -> bool:
        return name in self._dimension_names

    def get_metric_tables(self, metric_name: str) -> set[str]:
        return self._metric_tables.get(metric_name, set())

    @staticmethod
    def _is_scope_source(scope: Scope, alias: str) -> bool:
        """True if *alias* in scope.sources is a subquery/CTE (Scope), not a base table."""
        source = scope.sources.get(alias)
        return isinstance(source, Scope)

    @staticmethod
    def _collect_from_tables(scope: Scope) -> set[str]:
        """收集当前作用域 FROM/JOIN 中的逻辑表名"""
        tables = set()
        for name, source in scope.sources.items():
            if isinstance(source, exp.Table):
                tables.add(source.name)
            elif isinstance(source, Scope) and name:
                tables.add(name)
        return tables

    def analyze_select_list(self, select: exp.Select) -> tuple[set[str], dict[str, str], bool]:
        """分析 SELECT 列表，返回 (维度列名集合, alias→列名映射, 是否有指标)"""
        dims: set[str] = set()
        alias_map: dict[str, str] = {}  # SELECT alias → dimension column name
        has_metric = False

        for item in select.expressions:
            if isinstance(item, exp.Star):
                continue

            alias = item.alias if hasattr(item, 'alias') and item.alias else None

            if isinstance(item, exp.Column):
                name = item.name
                if self.is_metric(name):
                    has_metric = True
                elif self.is_dimension(name):
                    dims.add(name)
                    if alias:
                        alias_map[alias] = name
            elif isinstance(item, exp.Alias):
                inner = item.this
                if isinstance(inner, exp.Window):
                    continue
                self._analyze_expression(inner, dims, has_metric_ref=[has_metric])
                if isinstance(inner, exp.Column) and self.is_dimension(inner.name):
                    dims.add(inner.name)
                    alias_map[alias] = inner.name
            elif isinstance(item, exp.Window):
                continue
            elif isinstance(item, exp.AggFunc):
                pass
            else:
                has_metric_ref = [has_metric]
                self._analyze_expression(item, dims, has_metric_ref)
                has_metric = has_metric_ref[0]

        return dims, alias_map, has_metric

    def _analyze_expression(self, expr: exp.Expression, dims: set, has_metric_ref: list) -> None:
        """分析表达式中的维度列和指标列"""
        for col in expr.find_all(exp.Column, bfs=False):
            name = col.name
            if self.is_metric(name):
                has_metric_ref[0] = True
            elif self.is_dimension(name):
                dims.add(name)

    def get_group_by_columns(self, select: exp.Select, alias_map: dict[str, str] | None = None) -> set[str]:
        """获取 GROUP BY 中的列名。将别名还原为实际的列名。"""
        cols: set[str] = set()
        alias_map = alias_map or {}
        group = select.args.get("group")
        if group:
            for expr in group.expressions:
                if isinstance(expr, exp.Column):
                    name = expr.name
                    cols.add(alias_map.get(name, name))
                elif isinstance(expr, exp.Alias):
                    cols.add(alias_map.get(expr.alias, expr.alias))
                elif hasattr(expr, 'name'):
                    name = getattr(expr, 'name')
                    cols.add(alias_map.get(name, name))

        if select.args.get("distinct"):
            for item in select.expressions:
                if isinstance(item, exp.Column) and self.is_dimension(item.name):
                    cols.add(item.name)
                elif isinstance(item, exp.Alias) and isinstance(item.this, exp.Column):
                    if self.is_dimension(item.this.name):
                        cols.add(item.alias or item.this.name)

        return cols

    def _all_metrics_from_subquery(self, select: exp.Select, scope: Scope) -> bool:
        """True if every metric in the SELECT is accessed through a subquery/CTE alias."""
        metrics_found = False
        for item in select.expressions:
            for col in item.walk(bfs=False):
                if isinstance(col, exp.Column) and self.is_metric(col.name):
                    metrics_found = True
                    if not col.table or not self._is_scope_source(scope, col.table):
                        return False
        return metrics_found

    def _find_metrics_in_select(self, select: exp.Select) -> set[str]:
        """找出 SELECT 列表中使用的指标名"""
        metrics = set()
        for item in select.expressions:
            for col in item.walk(bfs=False):
                if isinstance(col, exp.Column) and self.is_metric(col.name):
                    metrics.add(col.name)
        return metrics
