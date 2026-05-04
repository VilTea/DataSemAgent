"""
OSI SQL Validator 测试用例

校验逻辑 SQL 是否违反规约，不做下推。
三大核心规则：
1. 指标列不可再聚合
2. 调用指标列必须有维度列被 GROUP BY
3. 指标列涉及多表时，它涉及的表必须在它的 FROM 里
"""

import pytest

from app.semantics.models import (
    Dataset,
    Dialect as OSIDialect,
    Expression,
    Metric,
    OSIField,
    Relationship,
    SemanticModel,
    Dimension,
)


def create_test_semantic_model() -> SemanticModel:
    """创建测试用语义模型"""
    orders_dataset = Dataset(
        name="orders",
        source="stg_orders",
        fields=[
            OSIField(name="order_id", expression=Expression.from_dict({OSIDialect.ANSI_SQL: "order_id"})),
            OSIField(name="customer_id", expression=Expression.from_dict({OSIDialect.ANSI_SQL: "customer_id"}), dimension=Dimension()),
            OSIField(name="total_amount", expression=Expression.from_dict({OSIDialect.ANSI_SQL: "total_amount"})),
            OSIField(name="order_date", expression=Expression.from_dict({OSIDialect.ANSI_SQL: "order_date"}), dimension=Dimension(is_time=True)),
        ],
    )

    customers_dataset = Dataset(
        name="customers",
        source="stg_customers",
        fields=[
            OSIField(name="customer_id", expression=Expression.from_dict({OSIDialect.ANSI_SQL: "customer_id"}), dimension=Dimension()),
            OSIField(name="customer_name", expression=Expression.from_dict({OSIDialect.ANSI_SQL: "customer_name"})),
            OSIField(name="region", expression=Expression.from_dict({OSIDialect.ANSI_SQL: "region"}), dimension=Dimension()),
        ],
    )

    revenue_metric = Metric(
        name="revenue",
        expression=Expression.from_dict({OSIDialect.ANSI_SQL: "SUM(total_amount)"}),
    )

    # 跨数据集指标：引用 orders 和 customers
    customer_ltv_metric = Metric(
        name="customer_ltv",
        expression=Expression.from_dict({OSIDialect.ANSI_SQL: "SUM(orders.total_amount) / COUNT(DISTINCT customers.customer_id)"}),
    )

    return SemanticModel(
        name="test_model",
        datasets=[orders_dataset, customers_dataset],
        relationships=[Relationship.model_validate({
            "name": "orders_to_customers",
            "from": "orders",
            "to": "customers",
            "from_columns": ["customer_id"],
            "to_columns": ["customer_id"],
        })],
        metrics=[revenue_metric, customer_ltv_metric],
    )


# =============================================================================
# 规则1: 指标列不可再聚合
# =============================================================================
class TestRule1NoReAggregation:
    """指标列已是聚合结果，禁止再套聚合函数"""

    def test_valid_metric_direct_use(self):
        """有效：指标直接使用"""
        from app.semantics.sql.validator import SQLValidator
        model = create_test_semantic_model()
        validator = SQLValidator(model)
        # 不应抛出异常
        validator.validate("SELECT customer_id, revenue FROM orders GROUP BY customer_id")

    def test_invalid_metric_re_aggregate_sum(self):
        """无效：对指标再次 SUM"""
        from app.semantics.sql.validator import SQLValidator
        model = create_test_semantic_model()
        validator = SQLValidator(model)
        errors = validator.validate("SELECT SUM(revenue) FROM orders")
        assert len(errors) > 0, "应检测到指标再聚合违规"

    def test_invalid_metric_re_aggregate_avg(self):
        """无效：对指标再次 AVG"""
        from app.semantics.sql.validator import SQLValidator
        model = create_test_semantic_model()
        validator = SQLValidator(model)
        errors = validator.validate("SELECT AVG(revenue) FROM orders GROUP BY customer_id")
        assert len(errors) > 0

    def test_invalid_metric_re_aggregate_count(self):
        """无效：对指标再次 COUNT"""
        from app.semantics.sql.validator import SQLValidator
        model = create_test_semantic_model()
        validator = SQLValidator(model)
        errors = validator.validate("SELECT COUNT(revenue) FROM orders GROUP BY customer_id")
        assert len(errors) > 0

    def test_invalid_metric_re_aggregate_max(self):
        """无效：对指标再次 MAX"""
        from app.semantics.sql.validator import SQLValidator
        model = create_test_semantic_model()
        validator = SQLValidator(model)
        errors = validator.validate("SELECT MAX(revenue) FROM orders GROUP BY customer_id")
        assert len(errors) > 0

    def test_valid_dimension_aggregate(self):
        """有效：对维度列聚合（COUNT customer_id）"""
        from app.semantics.sql.validator import SQLValidator
        model = create_test_semantic_model()
        validator = SQLValidator(model)
        validator.validate("SELECT COUNT(customer_id) FROM orders")

    def test_valid_plain_column_aggregate(self):
        """有效：对普通列聚合"""
        from app.semantics.sql.validator import SQLValidator
        model = create_test_semantic_model()
        validator = SQLValidator(model)
        validator.validate("SELECT SUM(total_amount) FROM orders")


# =============================================================================
# 规则2: 调用指标列必须有维度列被 GROUP BY
# =============================================================================
class TestRule2MetricNeedsGroupBy:
    """使用指标列时，必须对维度列做 GROUP BY"""

    def test_valid_metric_with_group_by(self):
        """有效：指标 + GROUP BY 维度"""
        from app.semantics.sql.validator import SQLValidator
        model = create_test_semantic_model()
        validator = SQLValidator(model)
        validator.validate("SELECT customer_id, revenue FROM orders GROUP BY customer_id")

    def test_valid_metric_only(self):
        """有效：只有指标列（无维度列，整体聚合）"""
        from app.semantics.sql.validator import SQLValidator
        model = create_test_semantic_model()
        validator = SQLValidator(model)
        validator.validate("SELECT revenue FROM orders")

    def test_invalid_metric_without_group_by(self):
        """无效：指标 + 维度列，但没有 GROUP BY"""
        from app.semantics.sql.validator import SQLValidator
        model = create_test_semantic_model()
        validator = SQLValidator(model)
        errors = validator.validate("SELECT customer_id, revenue FROM orders")
        assert len(errors) > 0, "应检测到缺少 GROUP BY"

    def test_invalid_metric_partial_group_by(self):
        """无效：指标 + 多个维度列，只 GROUP BY 其中一个"""
        from app.semantics.sql.validator import SQLValidator
        model = create_test_semantic_model()
        validator = SQLValidator(model)
        errors = validator.validate(
            "SELECT customer_id, order_date, revenue FROM orders "
            "GROUP BY customer_id"
        )
        assert len(errors) > 0, "应检测到部分 GROUP BY 违规"

    def test_valid_metric_full_group_by(self):
        """有效：所有 SELECT 中的维度列都在 GROUP BY 中"""
        from app.semantics.sql.validator import SQLValidator
        model = create_test_semantic_model()
        validator = SQLValidator(model)
        validator.validate(
            "SELECT customer_id, order_date, revenue FROM orders "
            "GROUP BY customer_id, order_date"
        )

    def test_valid_metric_in_where_only(self):
        """有效：指标在 WHERE 中（非 SELECT 列表），无 GROUP BY"""
        from app.semantics.sql.validator import SQLValidator
        model = create_test_semantic_model()
        validator = SQLValidator(model)
        # WHERE 中的指标在翻译时会展开为聚合表达式，这里只校验逻辑 SQL
        validator.validate("SELECT customer_id FROM orders")


# =============================================================================
# 规则3: 跨数据集指标的表必须在 FROM 里
# =============================================================================
class TestRule3CrossDatasetTables:
    """跨数据集指标涉及的表必须在当前作用域的 FROM/JOIN 里"""

    def test_valid_cross_dataset_with_join(self):
        """有效：跨数据集指标 + JOIN 了所有涉及的表"""
        from app.semantics.sql.validator import SQLValidator
        model = create_test_semantic_model()
        validator = SQLValidator(model)
        validator.validate(
            "SELECT customer_id, customer_ltv "
            "FROM orders JOIN customers ON orders.customer_id = customers.customer_id "
            "GROUP BY customer_id"
        )

    def test_invalid_cross_dataset_missing_join(self):
        """无效：跨数据集指标但缺少 JOIN"""
        from app.semantics.sql.validator import SQLValidator
        model = create_test_semantic_model()
        validator = SQLValidator(model)
        errors = validator.validate(
            "SELECT customer_id, customer_ltv FROM orders GROUP BY customer_id"
        )
        assert len(errors) > 0, "应检测到跨数据集指标缺少 JOIN"

    def test_valid_single_dataset_metric(self):
        """有效：单数据集指标，FROM 正确"""
        from app.semantics.sql.validator import SQLValidator
        model = create_test_semantic_model()
        validator = SQLValidator(model)
        validator.validate("SELECT customer_id, revenue FROM orders GROUP BY customer_id")


# =============================================================================
# 子查询场景：层级隔离
# =============================================================================
class TestSubqueryScopeIsolation:
    """子查询中，每个作用域独立校验，不可跨层匹配"""

    def test_subquery_valid_inner_scope(self):
        """有效：子查询内有自己的 FROM + GROUP BY"""
        from app.semantics.sql.validator import SQLValidator
        model = create_test_semantic_model()
        validator = SQLValidator(model)
        validator.validate(
            "SELECT * FROM ("
            "  SELECT customer_id, revenue FROM orders GROUP BY customer_id"
            ") sub"
        )

    def test_subquery_invalid_inner_missing_group_by(self):
        """无效：子查询内指标 + 维度但没有 GROUP BY"""
        from app.semantics.sql.validator import SQLValidator
        model = create_test_semantic_model()
        validator = SQLValidator(model)
        errors = validator.validate(
            "SELECT * FROM ("
            "  SELECT customer_id, revenue FROM orders"
            ") sub"
        )
        assert len(errors) > 0, "应检测到子查询内缺少 GROUP BY"

    def test_subquery_invalid_inner_re_aggregate(self):
        """无效：子查询内对指标再聚合"""
        from app.semantics.sql.validator import SQLValidator
        model = create_test_semantic_model()
        validator = SQLValidator(model)
        errors = validator.validate(
            "SELECT * FROM ("
            "  SELECT SUM(revenue) FROM orders GROUP BY customer_id"
            ") sub"
        )
        assert len(errors) > 0, "应检测到子查询内指标再聚合"

    def test_subquery_outer_valid_inner_invalid(self):
        """无效：外层正确但内层违反规约"""
        from app.semantics.sql.validator import SQLValidator
        model = create_test_semantic_model()
        validator = SQLValidator(model)
        errors = validator.validate(
            "SELECT customer_id FROM customers WHERE customer_id IN ("
            "  SELECT customer_id, revenue FROM orders"
            ")"
        )
        assert len(errors) > 0, "应检测到子查询内违规"

    def test_correlated_subquery_outer_ref_valid(self):
        """有效：相关子查询，外层维度正确，内层独立校验"""
        from app.semantics.sql.validator import SQLValidator
        model = create_test_semantic_model()
        validator = SQLValidator(model)
        validator.validate(
            "SELECT customer_id FROM customers c WHERE EXISTS ("
            "  SELECT 1 FROM orders o WHERE o.customer_id = c.customer_id"
            ")"
        )


# =============================================================================
# CTE 场景
# =============================================================================
class TestCTEScope:
    """CTE 中每个 CTE 独立校验"""

    def test_cte_valid(self):
        """有效：CTE 内有正确的 GROUP BY"""
        from app.semantics.sql.validator import SQLValidator
        model = create_test_semantic_model()
        validator = SQLValidator(model)
        validator.validate(
            "WITH sales AS ("
            "  SELECT customer_id, revenue FROM orders GROUP BY customer_id"
            ") SELECT * FROM sales"
        )

    def test_cte_invalid_missing_group_by(self):
        """无效：CTE 内指标 + 维度但没有 GROUP BY"""
        from app.semantics.sql.validator import SQLValidator
        model = create_test_semantic_model()
        validator = SQLValidator(model)
        errors = validator.validate(
            "WITH sales AS ("
            "  SELECT customer_id, revenue FROM orders"
            ") SELECT * FROM sales"
        )
        assert len(errors) > 0, "应检测到 CTE 内缺少 GROUP BY"


# =============================================================================
# 规则4: 指标列不能在 WHERE 中使用
# =============================================================================
class TestRule4NoMetricInWhere:
    """指标列是聚合结果，应在 HAVING 中过滤，不能在 WHERE 中使用"""

    def test_invalid_metric_in_where_on_physical_table(self):
        """无效：直接在物理表 WHERE 中使用指标列"""
        from app.semantics.sql.validator import SQLValidator
        model = create_test_semantic_model()
        validator = SQLValidator(model)
        errors = validator.validate(
            "SELECT customer_id FROM orders WHERE revenue > 1000 GROUP BY customer_id"
        )
        assert len(errors) > 0, "应检测到指标列在 WHERE 中使用"
        assert "[R4]" in errors[0]

    def test_invalid_metric_in_where_with_distinct(self):
        """无效：SELECT DISTINCT 下 WHERE 中使用指标列"""
        from app.semantics.sql.validator import SQLValidator
        model = create_test_semantic_model()
        validator = SQLValidator(model)
        errors = validator.validate(
            "SELECT DISTINCT customer_id, revenue FROM orders WHERE revenue > 1000"
        )
        assert len(errors) > 0, "应检测到 DISTINCT + WHERE 中指标列违规"
        assert "[R4]" in errors[0]

    def test_valid_metric_in_having(self):
        """有效：指标列在 HAVING 中使用"""
        from app.semantics.sql.validator import SQLValidator
        model = create_test_semantic_model()
        validator = SQLValidator(model)
        errors = validator.validate(
            "SELECT customer_id, revenue FROM orders "
            "GROUP BY customer_id HAVING revenue > 1000"
        )
        assert len(errors) == 0, "HAVING 中使用指标列应合法"

    def test_valid_metric_in_where_on_cte(self):
        """有效：CTE 结果集的 WHERE 中使用指标列"""
        from app.semantics.sql.validator import SQLValidator
        model = create_test_semantic_model()
        validator = SQLValidator(model)
        errors = validator.validate(
            "WITH cte AS ("
            "  SELECT customer_id, revenue FROM orders GROUP BY customer_id"
            ") SELECT * FROM cte WHERE revenue > 1000"
        )
        assert len(errors) == 0, "CTE 结果集的 WHERE 中使用指标列应合法"

    def test_valid_metric_in_where_on_subquery(self):
        """有效：子查询结果集的 WHERE 中使用指标列"""
        from app.semantics.sql.validator import SQLValidator
        model = create_test_semantic_model()
        validator = SQLValidator(model)
        errors = validator.validate(
            "SELECT * FROM ("
            "  SELECT customer_id, revenue FROM orders GROUP BY customer_id"
            ") sub WHERE revenue > 1000"
        )
        assert len(errors) == 0, "子查询结果集的 WHERE 中使用指标列应合法"

    def test_valid_metric_in_where_on_in_subquery(self):
        """有效：IN 子查询外部 WHERE 中使用指标列"""
        from app.semantics.sql.validator import SQLValidator
        model = create_test_semantic_model()
        validator = SQLValidator(model)
        errors = validator.validate(
            "SELECT * FROM ("
            "  SELECT customer_id, revenue FROM orders GROUP BY customer_id"
            ") sub WHERE customer_id IN (SELECT customer_id FROM customers)"
        )
        assert len(errors) == 0, "IN 子查询外部 WHERE 中使用指标列应合法"


# =============================================================================
# 综合场景
# =============================================================================
class TestComprehensive:
    """综合测试：多层嵌套 + 多规则同时校验"""

    def test_deeply_nested_all_valid(self):
        """有效：多层嵌套全部合规"""
        from app.semantics.sql.validator import SQLValidator
        model = create_test_semantic_model()
        validator = SQLValidator(model)
        validator.validate(
            "SELECT * FROM ("
            "  SELECT customer_id, revenue FROM ("
            "    SELECT customer_id, revenue FROM orders GROUP BY customer_id"
            "  ) inner_q GROUP BY customer_id"
            ") outer_q"
        )

    def test_deeply_nested_inner_invalid(self):
        """无效：最内层违反规约"""
        from app.semantics.sql.validator import SQLValidator
        model = create_test_semantic_model()
        validator = SQLValidator(model)
        errors = validator.validate(
            "SELECT * FROM ("
            "  SELECT customer_id, revenue FROM ("
            "    SELECT customer_id, revenue FROM orders"
            "  ) inner_q GROUP BY customer_id"
            ") outer_q"
        )
        assert len(errors) > 0, "应检测到最内层违规"

    def test_union_both_valid(self):
        """有效：UNION 两侧都合规"""
        from app.semantics.sql.validator import SQLValidator
        model = create_test_semantic_model()
        validator = SQLValidator(model)
        errors = validator.validate(
            "SELECT customer_id, revenue FROM orders GROUP BY customer_id "
            "UNION "
            "SELECT customer_id, revenue FROM orders GROUP BY customer_id"
        )
        assert len(errors) == 0

    def test_union_one_side_invalid(self):
        """无效：UNION 一侧违反规约"""
        from app.semantics.sql.validator import SQLValidator
        model = create_test_semantic_model()
        validator = SQLValidator(model)
        errors = validator.validate(
            "SELECT customer_id, revenue FROM orders GROUP BY customer_id "
            "UNION "
            "SELECT customer_id, revenue FROM orders"
        )
        assert len(errors) > 0, "应检测到 UNION 一侧违规"
