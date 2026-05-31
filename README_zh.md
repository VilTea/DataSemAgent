# 🤖 DataSemAgent

> [English Version](README.md)

基于 [OSI 规范](https://github.com/open-semantic-interchange/OSI) 的**语义数据分析智能体**。
将业务问题转化为可执行查询——*不是靠猜表名，而是在一个受控的语义框架内推理。*

---

## 💡 为什么要做这个

LLM 能写 SQL，但写出来的 SQL 太脆了。让它"查一下各客户的营收"，它完全不知道你的数据分散在 `orders_old`（字段叫 `amt`）和 `orders_new`（字段叫 `total_amount`）两张表里，旁边还躺着一张只保留近 90 天数据的汇总表。它不在"营收"这个业务概念里思考——它在猜列名。

解决思路：让业务人员定义一次"营收"——指向哪张表、哪个字段、用什么聚合函数——然后所有人（包括 LLM）都用这个定义来写 SQL。写出来的东西先过规则校验，再翻译成物理 SQL 执行。

这个思路不新鲜（Looker 的 LookML、dbt 的 Semantic Layer 都是成熟方案），但 [OSI 规范](https://github.com/open-semantic-interchange/OSI) 是开源社区对语义层标准化的一个尝试。DataSemAgent 是我对这个规范的探索——一个能跑的原型，用来看看这条路能走多远。

---

## 🧭 工作流程

1. 📐 **建模** — 在 OSI 语义模型中定义业务术语（指标、维度）
2. 🔍 **索引** — 从数据库构建实体图谱、指标血缘与推理链
3. ⚡ **提问** — 用自然语言提问；智能体自动编写逻辑 SQL、校验、翻译并执行

---

## ⛓️ 四条管线

### 🧱 语义 SQL

智能体用业务术语编写*逻辑 SQL*，校验后翻译为物理 SQL。

```
逻辑 SQL                              物理 SQL
──────────────────────────            ──────────────────────────────
SELECT customer_id, revenue           SELECT stg_orders.customer_id,
FROM orders                                  SUM(stg_orders.total_amount)
GROUP BY customer_id                  FROM stg_orders
                                      GROUP BY stg_orders.customer_id
```

**✅ 校验规则**（每条都是独立的 `Rule` 类，基于 `sqlglot` 的 AST + Scope 分析实现）：

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
映射阶段逐步构建，每次调用实时校验。校验失败自动重试，错误信息作为结构化反馈注入下一次 LLM 调用。

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

### 🧠 推理链图谱

后台反思从对话中提取**可复用的推理模式**——不存"利润 120 万"这种具体数字，而是存储"同比环比分析能揭示季节性波动"这类通用方法。本体层级组织概念，子节点继承父节点的推理链。

反思每 N 轮自动触发（通过 `config/agent.toml` 配置）。问答模式中使用 `/reflect` 手动触发。智能体在面对复杂分析问题时会被提示**优先查询推理图谱**——过去的经验可以节省时间，避免重复错误。

**🔁 流水线：** 反射器（从最近 N 轮对话构建提示词）→ 智能体（增量 EmitReasoningTool）→ 同义词合并器 → 构建器 → KuzuDB

---

## 🚀 快速开始

```bash
git clone https://github.com/VilTea/DataSemAgent.git && cd DataSemAgent
uv sync
cp config/llm/config.toml.demo config/llm/config.toml   # 填入 API Key
uv run python run.py --lang zh
```

### CLI 菜单

| 命令 | 说明 |
|------|------|
| `[1]` **init** | 构建实体图谱 + 指标血缘图谱 |
| `[2]` **ask** | 多轮智能问答 |
| `[3]` **exit** | 退出 |
| `[8]` **inspect** | 交互式推理图谱浏览器 |
| `[9]` **build-db** | 从 OSI 模型生成测试数据库 |

### CLI 选项

```bash
uv run python run.py --lang zh          # 中文界面
uv run python run.py --lang en          # 英文界面
uv run python run.py --debug            # 详细 hook 触发日志
uv run python run.py --model path/to/model.yaml  # 自定义 OSI 模型路径
```

### 问答模式命令

| 命令 | 说明 |
|------|------|
| `/reflect` | 手动触发推理链反思 |
| `/exit`, `/q`, `/quit` | 返回主菜单 |

### 构建测试数据库

从 OSI 语义模型生成 SQLite 测试数据库：

```bash
# CLI 菜单选择 [9] build-db（显示模型/数据库路径确认）
# 或直接运行：
uv run python tests/build_tpcds_test_data.py --db data/test.db
```

创建物理表并填充逼真样本数据——3 年日期、10 个客户/商品、5 个门店、500 条交易。

---

## 🧰 智能体工具

| 工具 | 说明 |
|------|------|
| `sql_exec` | 带预聚合指标的逻辑 SQL 执行 |
| `entity_graph` | 实体属性图 Cypher 查询 |
| `metric_lineage` | 发现可用指标、维度及其来源 |
| `reasoning_graph` | 查询累积的可复用推理模式 |
| `todo_write` | 跟踪多步骤任务进度 |

---

## ⚙️ 配置

| 组件 | 文件 |
|------|------|
| 🧠 LLM（`openai` / `anthropic`） | `config/llm/config.toml` |
| 🤖 智能体行为 | `config/agent.toml` |
| 🗄️ SQL 数据库 | `config/database.toml` |
| 🔗 图数据库 | `config/graph_database.toml` |
| 📦 OSI 模型 | `config/config.toml` → `[paths]` |
| 🔌 MCP 服务器 | `config/mcp/servers.yaml` |

### 智能体设置（`config/agent.toml`）

```toml
[default]
reflection_interval = 5   # 每 N 轮对话触发一次推理反思
```

---

## 📁 项目结构

```
  run.py
  app/
  ├── semantics/sql/          # SQL 翻译（分类器、展开器、翻译器、校验器）
  ├── semantics/graph/        # 实体 + 指标 + 推理图谱
  │   ├── entity/             #   LLM 驱动的实体图谱流水线
  │   ├── metric/             #   确定性指标血缘
  │   └── reasoning/          #   后台反思推理链图谱
  ├── node/                   # 智能体编排（PocketFlow: AgentNode、ToolNode）
  ├── tool/                   # sql_exec, entity_graph, metric_graph, reasoning_graph, todo_write
  ├── llm.py                  # OpenAI / Anthropic 客户端
  ├── hook/                   # 生命周期钩子（节点/工具/流程粒度，优先级排序）
  ├── pipeline/               # 流式输出（EventConsumer / Consumable 抽象）
  └── cli/                    # Typer + Rich 终端界面
  config/                     # 配置 + 国际化 (zh/en)
  tests/
```

---

## 📢 近期更新

**评估管线**——无侵入式的对话轨迹收集，用于模型效果评估。`EvalCollector` 通过消费者钩子捕获完整的 LLM 输入输出、工具调用及结果，输出为 JSONL 格式，支持可配置的敏感数据脱敏。配置见 `config/eval.toml`。

**工具 Schema 注入**——四个工具（`sql_exec`、`entity_graph`、`metric_lineage`、`reasoning_graph`）现在通过 `<tag>` 标签将 schema 定义注入系统提示词。智能体不再需要反复探测图谱来发现标签、属性和关系——所有信息在对话开始时即可用。

**消费者钩子系统**——`EventConsumer` 实现可以通过 `@hook` 注解观察智能体生命周期事件（工具调用、LLM 执行、流程边界）。钩子仅观察（强制 `on_error="log"`，优先级 200），通过 `async with pipeline.bind(ctx)` 自动注册。

**SQL 别名下推修复**——修复了 CTE、子查询、纯字段、维度和指标场景下的多处别名优先级问题。SQL 别名现在能一致保留，不再被物理表名替换。

---

## 📍 现状

这是一个**个人验证阶段**的项目，不是产品：

- 只在 TPC-DS 测试数据集上验证过，没有经过真实业务场景的覆盖
- 实体图谱的质量依赖 LLM 本身能力，复杂 Schema 下的识别效果还没有充分评估
- 没有性能优化、没有生产部署方案、错误处理的边界还很粗糙

它是一个能跑的原型，展示了基于 OSI 的语义分析可能是什么样子——不是开箱即用的工具。

---

## 为什么开源

- **想听到反馈。** 个人项目最大的问题是闭门造车。如果你对语义层、Text-to-SQL、Agent 架构感兴趣，任何意见都在 GitHub 上提。
- **OSI 规范值得探索。** 语义层标准化如果落地，对可复用的数据分析组件会有很大帮助。这个项目是我对规范的实践验证。
- **代码本身也许有帮助。** PocketFlow 编排、结构化重试反馈、sqlglot AST 分析、Hook 机制——拆出来也许能省一些调研时间。

---

📄 Apache 2.0 · Copyright 2026 [VILTEA](https://github.com/VILTEA)
