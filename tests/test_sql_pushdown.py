"""
OSI SQL Pushdown 测试用例

核心测试场景：
1. 字段分类：维度、度量、普通字段（名称与 OSI 模型一致）
2. 别名传递：子查询内层保持逻辑名，外层引用别名
3. 表名映射：逻辑表名 → 物理表名
4. 表达式下推：WHERE、GROUP BY、JOIN 等子句的字段处理
5. 子查询处理：标量、IN/EXISTS、相关子查询
6. CTE 处理：WITH 子句的别名映射
"""

import pytest
from sqlglot.dialects import Dialect

from app.semantics.models import (
    Dataset,
    Dialect as OSIDialect,
    Expression,
    OSIField,
    Relationship,
    SemanticModel,
    Metric,
    Dimension,
)
from app.semantics.sql import (
    OSIModelParser,
    SQLTranslator,
    FieldNotFoundError,
    DatasetNotFoundError,
)


def create_test_semantic_model() -> SemanticModel:
    """创建测试用语义模型"""
    orders_dataset = Dataset(
        name="orders",
        source="stg_orders",
        fields=[
            OSIField(
                name="order_id",
                expression=Expression.from_dict({OSIDialect.ANSI_SQL: "order_id"}),
            ),
            OSIField(
                name="customer_id",
                expression=Expression.from_dict({OSIDialect.ANSI_SQL: "customer_id"}),
                dimension=Dimension(),
            ),
            OSIField(
                name="total_amount",
                expression=Expression.from_dict({OSIDialect.ANSI_SQL: "total_amount"}),
            ),
            OSIField(
                name="order_date",
                expression=Expression.from_dict({OSIDialect.ANSI_SQL: "order_date"}),
                dimension=Dimension(is_time=True),
            ),
            OSIField(
                name="profit",
                expression=Expression.from_dict({OSIDialect.ANSI_SQL: "amount - cost"}),
            ),
        ],
    )

    customers_dataset = Dataset(
        name="customers",
        source="stg_customers",
        fields=[
            OSIField(
                name="customer_id",
                expression=Expression.from_dict({OSIDialect.ANSI_SQL: "customer_id"}),
                dimension=Dimension(),
            ),
            OSIField(
                name="customer_name",
                expression=Expression.from_dict({OSIDialect.ANSI_SQL: "customer_name"}),
            ),
            OSIField(
                name="region",
                expression=Expression.from_dict({OSIDialect.ANSI_SQL: "region"}),
                dimension=Dimension(),
            ),
            OSIField(
                name="status",
                expression=Expression.from_dict({OSIDialect.ANSI_SQL: "state"}),
                dimension=Dimension(),
            ),
        ],
    )

    orders_to_customers = Relationship.model_validate({
        "name": "orders_to_customers",
        "from": "orders",
        "to": "customers",
        "from_columns": ["customer_id"],
        "to_columns": ["customer_id"],
    })

    revenue_metric = Metric(
        name="revenue",
        expression=Expression.from_dict({OSIDialect.ANSI_SQL: "SUM(total_amount)"}),
    )

    return SemanticModel(
        name="test_model",
        datasets=[orders_dataset, customers_dataset],
        relationships=[orders_to_customers],
        metrics=[revenue_metric],
    )


# =============================================================================
# 场景1: 字段分类与基本映射
# =============================================================================
class TestFieldClassification:
    """测试字段分类：维度、度量、普通字段"""

    def test_dimension_field_mapping(self):
        """维度字段：添加表前缀，保持逻辑名"""
        model = create_test_semantic_model()
        parser = OSIModelParser(model)
        translator = SQLTranslator(parser)

        result = translator.translate("SELECT customer_id FROM orders")
        assert result.physical_sql == "SELECT stg_orders.customer_id AS customer_id FROM stg_orders"

    def test_dimension_field_with_alias(self):
        """维度字段带别名"""
        model = create_test_semantic_model()
        parser = OSIModelParser(model)
        translator = SQLTranslator(parser)

        result = translator.translate("SELECT customer_id AS cid FROM orders")
        assert result.physical_sql == "SELECT stg_orders.customer_id AS cid FROM stg_orders"

    def test_metric_field_expansion(self):
        """度量字段：展开为物理表达式"""
        model = create_test_semantic_model()
        parser = OSIModelParser(model)
        translator = SQLTranslator(parser)

        result = translator.translate("SELECT revenue FROM orders")
        assert result.physical_sql == "SELECT SUM(total_amount) AS revenue FROM stg_orders"

    def test_metric_field_with_alias(self):
        """度量字段带别名"""
        model = create_test_semantic_model()
        parser = OSIModelParser(model)
        translator = SQLTranslator(parser)

        result = translator.translate("SELECT revenue AS mre FROM orders")
        assert result.physical_sql == "SELECT SUM(total_amount) AS mre FROM stg_orders"

    def test_plain_column_mapping(self):
        """普通字段：无前缀 → 添加表前缀"""
        model = create_test_semantic_model()
        parser = OSIModelParser(model)
        translator = SQLTranslator(parser)

        result = translator.translate("SELECT order_id FROM orders")
        assert result.physical_sql == "SELECT stg_orders.order_id FROM stg_orders"

    def test_computed_field(self):
        """计算字段：映射到物理表达式"""
        model = create_test_semantic_model()
        parser = OSIModelParser(model)
        translator = SQLTranslator(parser)

        result = translator.translate("SELECT profit FROM orders")
        assert result.physical_sql == "SELECT stg_orders.amount - stg_orders.cost AS profit FROM stg_orders"

    def test_time_dimension(self):
        """时间维度字段"""
        model = create_test_semantic_model()
        parser = OSIModelParser(model)
        translator = SQLTranslator(parser)

        result = translator.translate("SELECT order_date FROM orders")
        assert "stg_orders.order_date" in result.physical_sql


# =============================================================================
# 场景2: 表名映射
# =============================================================================
class TestTableMapping:
    """测试逻辑表名到物理表名的映射"""

    def test_single_table_mapping(self):
        """单表：orders → stg_orders"""
        model = create_test_semantic_model()
        parser = OSIModelParser(model)
        translator = SQLTranslator(parser)

        result = translator.translate("SELECT order_id FROM orders")
        assert "stg_orders" in result.physical_sql

    def test_join_table_mapping(self):
        """JOIN：多表映射"""
        model = create_test_semantic_model()
        parser = OSIModelParser(model)
        translator = SQLTranslator(parser)

        result = translator.translate(
            "SELECT customer_id FROM orders JOIN customers ON orders.customer_id = customers.customer_id"
        )
        assert "stg_orders" in result.physical_sql
        assert "stg_customers" in result.physical_sql

    def test_table_alias_mapping(self):
        """表别名：o → stg_orders"""
        model = create_test_semantic_model()
        parser = OSIModelParser(model)
        translator = SQLTranslator(parser)

        result = translator.translate("SELECT o.order_id FROM orders AS o")
        assert result.physical_sql == "SELECT o.order_id FROM stg_orders AS o"


# =============================================================================
# 场景3: 别名传递（核心场景）
# =============================================================================
class TestAliasPropagation:
    """测试子查询内层保持逻辑名别名，外层引用别名"""

    def test_simple_subquery_alias_preservation(self):
        """
        子查询内层 SELECT 列表的维度字段需要添加逻辑名别名
        
        逻辑 SQL:
            SELECT * FROM (SELECT customer_id FROM orders) AS subq
        
        物理 SQL:
            SELECT * FROM (SELECT stg_orders.customer_id AS customer_id FROM stg_orders) AS subq
        """
        model = create_test_semantic_model()
        parser = OSIModelParser(model)
        translator = SQLTranslator(parser)

        result = translator.translate(
            "SELECT subq.customer_id FROM (SELECT customer_id FROM orders) AS subq"
        )
        assert result.physical_sql == "SELECT subq.customer_id FROM (SELECT stg_orders.customer_id AS customer_id FROM stg_orders) AS subq"

    def test_subquery_multiple_columns(self):
        """子查询多列的别名传递"""
        model = create_test_semantic_model()
        parser = OSIModelParser(model)
        translator = SQLTranslator(parser)

        result = translator.translate(
            "SELECT subq.customer_id, subq.total_amount FROM "
            "(SELECT customer_id, total_amount FROM orders) AS subq"
        )
        assert result.physical_sql == (
            "SELECT subq.customer_id, subq.total_amount FROM "
            "(SELECT stg_orders.customer_id AS customer_id, stg_orders.total_amount FROM stg_orders) AS subq"
        )

    def test_nested_subquery_alias(self):
        """多层嵌套子查询的别名传递"""
        model = create_test_semantic_model()
        parser = OSIModelParser(model)
        translator = SQLTranslator(parser)

        result = translator.translate(
            "SELECT * FROM ("
            "SELECT * FROM (SELECT customer_id FROM orders) AS inner_subq"
            ") AS outer_subq"
        )
        assert "AS customer_id" in result.physical_sql

    def test_subquery_alias_rename(self):
        """子查询内层字段别名重命名"""
        model = create_test_semantic_model()
        parser = OSIModelParser(model)
        translator = SQLTranslator(parser)

        result = translator.translate(
            "SELECT subq.cid FROM "
            "(SELECT customer_id AS cid FROM orders) AS subq"
        )
        assert result.physical_sql == "SELECT subq.cid FROM (SELECT stg_orders.customer_id AS cid FROM stg_orders) AS subq"


# =============================================================================
# 场景4: CTE 别名传递
# =============================================================================
class TestCTEAliasPropagation:
    """测试 WITH 子句的别名传递"""

    def test_cte_basic(self):
        """CTE 基本映射"""
        model = create_test_semantic_model()
        parser = OSIModelParser(model)
        translator = SQLTranslator(parser)

        result = translator.translate(
            "WITH cte AS (SELECT order_id FROM orders) SELECT * FROM cte"
        )
        assert result.physical_sql == "WITH cte AS (SELECT stg_orders.order_id FROM stg_orders) SELECT * FROM cte"

    def test_cte_with_dimension_alias(self):
        """CTE 内层维度字段保持逻辑名"""
        model = create_test_semantic_model()
        parser = OSIModelParser(model)
        translator = SQLTranslator(parser)

        result = translator.translate(
            "WITH cte AS (SELECT customer_id FROM orders) SELECT cte.customer_id FROM cte"
        )
        assert "stg_orders.customer_id AS customer_id" in result.physical_sql
        assert "cte.customer_id" in result.physical_sql

    def test_cte_with_column_alias(self):
        """CTE 内层字段重命名"""
        model = create_test_semantic_model()
        parser = OSIModelParser(model)
        translator = SQLTranslator(parser)

        result = translator.translate(
            "WITH cte AS (SELECT order_id AS oid FROM orders) SELECT cte.oid FROM cte"
        )
        assert "stg_orders.order_id AS oid" in result.physical_sql
        assert "cte.oid" in result.physical_sql

    def test_cte_join(self):
        """CTE 与表 JOIN"""
        model = create_test_semantic_model()
        parser = OSIModelParser(model)
        translator = SQLTranslator(parser)

        result = translator.translate(
            "WITH cte AS (SELECT order_id FROM orders) "
            "SELECT cte.order_id, status FROM cte JOIN customers "
            "ON cte.customer_id = customers.customer_id"
        )
        assert "stg_customers.state AS status" in result.physical_sql


# =============================================================================
# 场景5: 子查询类型
# =============================================================================
class TestSubqueryTypes:
    """测试不同类型的子查询"""

    def test_scalar_subquery(self):
        """标量子查询：内层使用物理列"""
        model = create_test_semantic_model()
        parser = OSIModelParser(model)
        translator = SQLTranslator(parser)

        result = translator.translate(
            "SELECT order_id FROM orders WHERE total_amount > "
            "(SELECT AVG(total_amount) FROM orders)"
        )
        assert result.physical_sql == (
            "SELECT stg_orders.order_id FROM stg_orders "
            "WHERE stg_orders.total_amount > (SELECT AVG(stg_orders.total_amount) FROM stg_orders)"
        )

    def test_in_subquery(self):
        """IN 子查询：内层保持逻辑名"""
        model = create_test_semantic_model()
        parser = OSIModelParser(model)
        translator = SQLTranslator(parser)

        result = translator.translate(
            "SELECT order_id FROM orders WHERE customer_id IN "
            "(SELECT customer_id FROM customers WHERE region = 'North')"
        )
        assert "stg_customers.customer_id AS customer_id" in result.physical_sql

    def test_exists_subquery(self):
        """EXISTS 子查询"""
        model = create_test_semantic_model()
        parser = OSIModelParser(model)
        translator = SQLTranslator(parser)

        result = translator.translate(
            "SELECT order_id FROM orders o WHERE EXISTS "
            "(SELECT 1 FROM customers c WHERE c.customer_id = o.customer_id)"
        )
        assert "c.customer_id" in result.physical_sql
        assert "o.customer_id" in result.physical_sql

    def test_correlated_subquery(self):
        """相关子查询"""
        model = create_test_semantic_model()
        parser = OSIModelParser(model)
        translator = SQLTranslator(parser)

        result = translator.translate(
            "SELECT o1.order_id FROM orders o1 WHERE o1.total_amount > "
            "(SELECT AVG(o2.total_amount) FROM orders o2 WHERE o2.customer_id = o1.customer_id)"
        )
        assert "stg_orders.total_amount" in result.physical_sql
        assert "o1.customer_id" in result.physical_sql

    def test_subquery_in_select_list(self):
        """SELECT 列表中的子查询"""
        model = create_test_semantic_model()
        parser = OSIModelParser(model)
        translator = SQLTranslator(parser)

        result = translator.translate(
            "SELECT order_id, "
            "(SELECT customer_name FROM customers WHERE customer_id = orders.customer_id) AS cust_name "
            "FROM orders"
        )
        assert "stg_customers.customer_name" in result.physical_sql
        assert "stg_orders.customer_id" in result.physical_sql


# =============================================================================
# 场景6: JOIN 处理
# =============================================================================
class TestJoinHandling:
    """测试 JOIN 条件的处理"""

    def test_join_condition_mapping(self):
        """JOIN 条件中的字段映射"""
        model = create_test_semantic_model()
        parser = OSIModelParser(model)
        translator = SQLTranslator(parser)

        result = translator.translate(
            "SELECT customer_id, total_amount FROM orders "
            "JOIN customers ON orders.customer_id = customers.customer_id"
        )
        assert "stg_orders.customer_id" in result.physical_sql
        assert "stg_customers.customer_id" in result.physical_sql

    def test_join_with_group_by(self):
        """带 GROUP BY 的 JOIN"""
        model = create_test_semantic_model()
        parser = OSIModelParser(model)
        translator = SQLTranslator(parser)

        result = translator.translate(
            "SELECT region, SUM(total_amount) FROM orders "
            "JOIN customers ON orders.customer_id = customers.customer_id "
            "GROUP BY region"
        )
        assert "stg_customers.region" in result.physical_sql
        assert "stg_orders.total_amount" in result.physical_sql


# =============================================================================
# 场景7: SQL 子句处理
# =============================================================================
class TestSQLClauses:
    """测试各 SQL 子句的处理"""

    def test_where_clause(self):
        """WHERE 子句"""
        model = create_test_semantic_model()
        parser = OSIModelParser(model)
        translator = SQLTranslator(parser)

        result = translator.translate(
            "SELECT order_id FROM orders WHERE customer_id = 1 AND total_amount > 100"
        )
        assert "stg_orders.customer_id = 1" in result.physical_sql
        assert "stg_orders.total_amount > 100" in result.physical_sql

    def test_group_by_clause(self):
        """GROUP BY 子句"""
        model = create_test_semantic_model()
        parser = OSIModelParser(model)
        translator = SQLTranslator(parser)

        result = translator.translate(
            "SELECT customer_id, SUM(total_amount) FROM orders GROUP BY customer_id"
        )
        assert "stg_orders.customer_id" in result.physical_sql

    def test_order_by_clause(self):
        """ORDER BY 子句"""
        model = create_test_semantic_model()
        parser = OSIModelParser(model)
        translator = SQLTranslator(parser)

        result = translator.translate(
            "SELECT customer_id FROM orders ORDER BY customer_id DESC"
        )
        assert "ORDER BY" in result.physical_sql

    def test_having_clause(self):
        """HAVING 子句"""
        model = create_test_semantic_model()
        parser = OSIModelParser(model)
        translator = SQLTranslator(parser)

        result = translator.translate(
            "SELECT customer_id, SUM(total_amount) FROM orders "
            "GROUP BY customer_id HAVING SUM(total_amount) > 1000"
        )
        assert "HAVING" in result.physical_sql

    def test_window_function(self):
        """窗口函数"""
        model = create_test_semantic_model()
        parser = OSIModelParser(model)
        translator = SQLTranslator(parser)

        result = translator.translate(
            "SELECT order_id, ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date) AS rn "
            "FROM orders"
        )
        assert "PARTITION BY stg_orders.customer_id" in result.physical_sql
        assert "ORDER BY stg_orders.order_date" in result.physical_sql


# =============================================================================
# 场景8: 表达式处理
# =============================================================================
class TestExpressionHandling:
    """测试表达式中的字段处理"""

    def test_case_when_expression(self):
        """CASE WHEN 表达式"""
        model = create_test_semantic_model()
        parser = OSIModelParser(model)
        translator = SQLTranslator(parser)

        result = translator.translate(
            "SELECT CASE WHEN total_amount > 1000 THEN 'high' ELSE 'low' END AS level FROM orders"
        )
        assert "stg_orders.total_amount" in result.physical_sql

    def test_arithmetic_expression(self):
        """算术表达式"""
        model = create_test_semantic_model()
        parser = OSIModelParser(model)
        translator = SQLTranslator(parser)

        result = translator.translate(
            "SELECT total_amount * 1.1 AS amount_with_tax FROM orders"
        )
        assert "stg_orders.total_amount * 1.1" in result.physical_sql

    def test_function_call(self):
        """函数调用"""
        model = create_test_semantic_model()
        parser = OSIModelParser(model)
        translator = SQLTranslator(parser)

        result = translator.translate(
            "SELECT COALESCE(total_amount, 0) AS safe_amount FROM orders"
        )
        assert "COALESCE(stg_orders.total_amount, 0)" in result.physical_sql

    def test_distinct_aggregation(self):
        """聚合函数中的 DISTINCT"""
        model = create_test_semantic_model()
        parser = OSIModelParser(model)
        translator = SQLTranslator(parser)

        result = translator.translate(
            "SELECT COUNT(DISTINCT customer_id) FROM orders"
        )
        assert "COUNT(DISTINCT stg_orders.customer_id)" in result.physical_sql


# =============================================================================
# 场景9: 指标展开
# =============================================================================
class TestMetricExpansion:
    """测试指标在各种子句中的展开"""

    def test_metric_in_select(self):
        """指标在 SELECT 列表中展开"""
        model = create_test_semantic_model()
        parser = OSIModelParser(model)
        translator = SQLTranslator(parser)

        result = translator.translate("SELECT revenue FROM orders")
        assert result.physical_sql == "SELECT SUM(total_amount) AS revenue FROM stg_orders"

    def test_metric_with_alias(self):
        """指标带别名"""
        model = create_test_semantic_model()
        parser = OSIModelParser(model)
        translator = SQLTranslator(parser)

        result = translator.translate("SELECT revenue AS total_rev FROM orders")
        assert "SUM(total_amount)" in result.physical_sql
        assert "AS total_rev" in result.physical_sql

    def test_metric_with_table_prefix(self):
        """指标带表前缀"""
        model = create_test_semantic_model()
        parser = OSIModelParser(model)
        translator = SQLTranslator(parser)

        result = translator.translate("SELECT orders.revenue FROM orders")
        assert "SUM(total_amount) AS revenue" in result.physical_sql

    def test_metric_in_where(self):
        """指标在 WHERE 子句中展开"""
        model = create_test_semantic_model()
        parser = OSIModelParser(model)
        translator = SQLTranslator(parser)

        result = translator.translate(
            "SELECT customer_id FROM orders WHERE revenue > 1000"
        )
        assert "WHERE SUM(stg_orders.total_amount) > 1000" in result.physical_sql

    def test_metric_in_having(self):
        """指标在 HAVING 子句中展开"""
        model = create_test_semantic_model()
        parser = OSIModelParser(model)
        translator = SQLTranslator(parser)

        result = translator.translate(
            "SELECT customer_id, SUM(total_amount) FROM orders "
            "GROUP BY customer_id HAVING revenue > 1000"
        )
        assert "HAVING SUM(stg_orders.total_amount) > 1000" in result.physical_sql

    def test_metric_in_order_by(self):
        """指标在 ORDER BY 子句中展开"""
        model = create_test_semantic_model()
        parser = OSIModelParser(model)
        translator = SQLTranslator(parser)

        result = translator.translate(
            "SELECT customer_id FROM orders ORDER BY revenue DESC"
        )
        assert "ORDER BY SUM(stg_orders.total_amount) DESC" in result.physical_sql

    def test_metric_in_subquery_select(self):
        """指标在子查询 SELECT 中展开"""
        model = create_test_semantic_model()
        parser = OSIModelParser(model)
        translator = SQLTranslator(parser)

        result = translator.translate(
            "SELECT * FROM (SELECT revenue FROM orders) AS subq"
        )
        assert "SUM(total_amount) AS revenue" in result.physical_sql

    def test_metric_not_found(self):
        """不存在的指标"""
        model = create_test_semantic_model()
        parser = OSIModelParser(model)
        translator = SQLTranslator(parser)

        with pytest.raises(FieldNotFoundError):
            translator.translate("SELECT nonexistent FROM orders")


# =============================================================================
# 场景10: 集合操作
# =============================================================================
class TestSetOperations:
    """测试集合操作"""

    def test_union(self):
        """UNION"""
        model = create_test_semantic_model()
        parser = OSIModelParser(model)
        translator = SQLTranslator(parser)

        result = translator.translate(
            "SELECT order_id FROM orders UNION SELECT order_id FROM orders WHERE total_amount > 100"
        )
        assert "stg_orders" in result.physical_sql
        assert "UNION" in result.physical_sql

    def test_intersect(self):
        """INTERSECT"""
        model = create_test_semantic_model()
        parser = OSIModelParser(model)
        translator = SQLTranslator(parser)

        result = translator.translate(
            "SELECT order_id FROM orders INTERSECT SELECT order_id FROM orders WHERE total_amount > 100"
        )
        assert "INTERSECT" in result.physical_sql


# =============================================================================
# 场景10: 错误处理
# =============================================================================
class TestErrorHandling:
    """测试错误处理"""

    def test_field_not_found(self):
        """字段不存在"""
        model = create_test_semantic_model()
        parser = OSIModelParser(model)
        translator = SQLTranslator(parser)

        with pytest.raises(FieldNotFoundError):
            translator.translate("SELECT nonexistent FROM orders")

    def test_dataset_not_found(self):
        """数据集不存在"""
        model = create_test_semantic_model()
        parser = OSIModelParser(model)
        translator = SQLTranslator(parser)

        with pytest.raises(DatasetNotFoundError):
            translator.translate("SELECT order_id FROM nonexistent")

    def test_metric_not_found(self):
        """度量不存在"""
        model = create_test_semantic_model()
        parser = OSIModelParser(model)
        translator = SQLTranslator(parser)

        with pytest.raises(FieldNotFoundError):
            translator.translate("SELECT nonexistent FROM orders")


# =============================================================================
# 场景11: OSIModelParser 单元测试
# =============================================================================
class TestOSIModelParser:
    """OSIModelParser 单元测试"""

    def test_list_datasets(self):
        model = create_test_semantic_model()
        parser = OSIModelParser(model)
        assert parser.list_datasets() == ["orders", "customers"]

    def test_list_metrics(self):
        model = create_test_semantic_model()
        parser = OSIModelParser(model)
        assert parser.list_metrics() == ["revenue"]

    def test_get_dataset(self):
        model = create_test_semantic_model()
        parser = OSIModelParser(model)
        orders = parser.get_dataset("orders")
        assert orders.name == "orders"
        assert orders.source == "stg_orders"

    def test_get_dataset_not_found(self):
        model = create_test_semantic_model()
        parser = OSIModelParser(model)
        with pytest.raises(DatasetNotFoundError):
            parser.get_dataset("nonexistent")

    def test_resolve_field(self):
        model = create_test_semantic_model()
        parser = OSIModelParser(model)
        mapping = parser.resolve_field("order_id", "orders")
        assert mapping.logical_name == "order_id"
        assert mapping.physical_expression == "order_id"

    def test_resolve_computed_field(self):
        model = create_test_semantic_model()
        parser = OSIModelParser(model)
        mapping = parser.resolve_field("profit", "orders")
        assert mapping.physical_expression == "amount - cost"

    def test_resolve_metric(self):
        model = create_test_semantic_model()
        parser = OSIModelParser(model)
        metric_expr = parser.resolve_metric("revenue")
        assert metric_expr == "SUM(total_amount)"
