# 🤖 DataSemAgent

> [English Version](README.md)

基于 [OSI 规范](https://github.com/open-semantic-interchange/OSI) 的**语义数据分析智能体**。
将业务问题转化为三条管线协同的可执行数据查询。

---

## 🧭 工作流程

1. 📐 **建模** — 在 OSI 语义模型中定义业务术语（指标、维度）
2. 🔍 **索引** — 从数据库构建实体图谱与指标血缘
3. ⚡ **提问** — 用自然语言提问；智能体自动编写逻辑 SQL、校验、翻译并执行

---

## ⛓️ 三条管线

### 🧱 语义 SQL

智能体用业务术语编写逻辑 SQL，校验后翻译为物理 SQL。

```
逻辑 SQL                              物理 SQL
──────────────────────────            ──────────────────────────────
SELECT customer_id, revenue           SELECT stg_orders.customer_id,
FROM orders                                  SUM(stg_orders.total_amount)
GROUP BY customer_id                  FROM stg_orders
                                      GROUP BY stg_orders.customer_id
```

**✅ 校验规则：**
- 指标不可再聚合
- 维度必须在 `GROUP BY` 中
- 跨数据集指标需 `JOIN`
- 指标过滤用 `HAVING`，不用 `WHERE`
- 指标不能出现在 `GROUP BY`

### 🕸️ 实体图谱

LLM 驱动的流水线，将数据库每一行转为图节点，外键转为边。

```
┌──────────┐  purchased_by  ┌──────────┐
│ store_   │ ──────────────►│ customer │
│ sales    │                └──────────┘
│          │  includes       ┌──────────┐
│ ss_item  │ ──────────────►│ item     │
│ _sk: 6   │                │ i_brand  │
│ ss_price │  occurred_at   │ :BrandA  │
│ : 303.0  │ ──────────────►└──────────┘
└──────────┘                ┌──────────┐
                            │ store    │
                            └──────────┘
```

**🔁 流水线：** 采样器 → Schema 智能体 → 校验器 → 映射智能体（增量 React 循环）→ 校验器 → 编译器  
映射阶段逐步构建，每次调用实时校验。校验失败自动重试。

### 🌳 指标血缘

由 OSI 模型确定性构建——无需 LLM 调用。

```
┌──────────┐  AGGREGATES_FROM  ┌────────────────┐
│ 指标     │ ─────────────────►│ 物理字段        │
│ revenue  │                   │ ss_ext_sales    │
│          │  SLICES_BY        │ _price          │
│          │ ─────────────────►│                 │
│          │                   │ 维度            │
│          │                   │ customer_id     │
└──────────┘                   └────────────────┘
```

💡 在编写 SQL 之前，发现可用指标、物理来源和合法维度切片。

---

## 🚀 快速开始

```bash
git clone https://github.com/VilTea/DataSemAgent.git && cd DataSemAgent
uv sync
cp config/llm/config.toml.demo config/llm/config.toml   # 填入 API Key
uv run python run.py --lang zh
```

| 命令 | 说明 |
|------|------|
| `[1]` **init** | 构建实体图谱 + 指标血缘图谱 |
| `[2]` **ask** | 多轮智能问答 |
| `[3]` **exit** | 退出 |
| `[9]` **build-db** | 从 OSI 模型生成测试数据库 |

### 构建测试数据库

从 OSI 语义模型生成 SQLite 测试数据库：

```bash
# CLI 菜单选择 [9] build-db（显示模型/数据库路径确认）
# 或直接运行：
uv run python tests/build_tpcds_test_data.py --db data/test.db
```

创建物理表并填充逼真样本数据——3 年日期、10 个客户/商品、5 个门店、500 条交易。

---

## ⚙️ 配置

| 组件 | 文件 |
|------|------|
| 🧠 LLM（`openai` / `anthropic`） | `config/llm/config.toml` |
| 🗄️ SQL 数据库 | `config/database.toml` |
| 🔗 图数据库 | `config/graph_database.toml` |
| 📦 OSI 模型 | `config/config.toml` → `[paths]` |
| 🔌 MCP 服务器 | `config/mcp/servers.yaml` |

---

## 📁 项目结构

```
  run.py
  app/
  ├── semantics/sql/        # SQL 翻译
  ├── semantics/graph/      # 实体 + 指标图谱
  ├── node/                 # 智能体编排
  ├── tool/                 # sql_exec, entity_graph, metric_lineage
  ├── llm.py                # OpenAI / Anthropic
  ├── hook/                 # 生命周期钩子
  ├── pipeline/             # 流式输出
  └── cli/                  # 终端界面
  config/                   # 配置 + 国际化 (zh/en)
  tests/
```

---

📄 Apache 2.0 · Copyright 2026 [VILTEA](https://github.com/VILTEA)
