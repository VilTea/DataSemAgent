---
name: osi-sql-writer
description: 基于 Open Semantic Interchange (OSI) 规范编写逻辑 SQL 的技能。当需要基于语义模型编写查询时激活，指导正确使用维度列、指标列、时间维度等。
version: 1.0.0
license: Apache-2.0
compatibility: 需要 OSI 语义模型定义
---

# OSI SQL Writer

基于 OSI v0.1.1 规范编写逻辑 SQL 的完整指南。

## ⚠️ 重要声明

**编写的 SQL 不可违反以下规约，否则将产生错误结果！**

## 触发条件

当需要基于 OSI 语义模型编写 SQL 查询时激活此技能。

## 核心规则

### 字段类型

| 类型 | 规则 |
|------|------|
| 维度列 | GROUP BY / WHERE / ORDER BY |
| 指标列 | **已是聚合结果，禁止再聚合！**用 HAVING 过滤，使用时必须按需对维度列做 GROUP BY |
| 普通列 | 任意用途 |
| 时间维度 | `is_time=true`，支持时间序列 |

### SQL 规则（不可违反）

1. **指标已是聚合结果** - 已包含 SUM/COUNT/AVG 等聚合，**禁止再套聚合函数**
2. **指标必须分组** - 使用时，必须按需对维度列做 GROUP BY
3. **指标过滤用 HAVING** - 不可用 WHERE 过滤指标
4. **维度可自由使用** - `可用于任意子句
5. **跨数据集指标需 JOIN** - 注释中标注 `需JOIN: 表名` 的指标，**必须遵循**，必须关联对应表

### ❌ 常见错误（违反规约）

```sql
-- 错误：对指标列再次聚合（违反规约1）
SELECT SUM(total_revenue) FROM orders

-- 错误：在 WHERE 中使用指标（违反规约3）
SELECT customer_id FROM orders WHERE total_revenue > 1000

-- 错误：使用指标列但未做 GROUP BY（违反规约2）
SELECT customer_id, total_revenue FROM orders

-- 错误：使用跨数据集指标但未 JOIN（违反规约5，**必须遵循**）
-- m_customer_lifetime_value 标注 "需JOIN: customer"
SELECT ss_customer_sk, customer_lifetime_value FROM store_sales GROUP BY ss_customer_sk
```

### ✅ 正确写法

```sql
-- 正确：指标直接使用，无需聚合，配合 GROUP BY
SELECT customer_id, total_revenue FROM orders GROUP BY customer_id

-- 正确：用 HAVING 过滤指标
SELECT customer_id, total_revenue FROM orders 
GROUP BY customer_id
HAVING total_revenue > 1000

-- 正确：跨数据集指标必须 JOIN 对应表
SELECT ss_customer_sk, customer_lifetime_value 
FROM store_sales ss
JOIN customer c ON ss.ss_customer_sk = c.c_customer_sk
GROUP BY ss_customer_sk
```

## SQL 模式

### 基础聚合
```sql
SELECT [维度], [指标] FROM [表] GROUP BY [维度] HAVING [指标] > [值]
```

### 时间序列
```sql
SELECT d.year, [指标] FROM [表] t JOIN date_dim d ON t.date = d.date GROUP BY d.year
```

### 多维分析
```sql
SELECT [维度1], [维度2], [指标] FROM [表] GROUP BY [维度1], [维度2]
```

### 跨数据集指标
```sql
SELECT [维度], [指标] FROM [表1] JOIN [表2] ON [关联条件] GROUP BY [维度]
```

## 检查清单

- [ ] 指标列是否**未被聚合函数包裹**？
- [ ] 指标列是否与 GROUP BY 配合？
- [ ] 指标过滤是否使用 HAVING？
- [ ] 跨数据集指标是否**遵循需JOIN要求**？
- [ ] 时间维度是否正确关联？
