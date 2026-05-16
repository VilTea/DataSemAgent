# 🤖 DataSemAgent

> [中文版本](README_zh.md)

**Semantic data analysis agent** built on the [OSI specification](https://github.com/open-semantic-interchange/OSI).
Translates business questions into executable queries — *not by guessing table schemas, but by reasoning within a controlled semantic framework.*

---

## 💡 Why

LLMs can write SQL, but the SQL they write is brittle. Ask "revenue by customer" and the model has no idea that your data spans `orders_old` (column `amt`) and `orders_new` (column `total_amount`), or that a performance summary table only covers the last 90 days. It does not think in business concepts — it guesses column names.

The idea: define "revenue" once — which tables, which columns, which aggregation — and let everyone (LLM included) use that definition. Queries pass through rule-based validation first, then translate to physical SQL. No more guessing.

This approach isn't new (Looker's LookML, dbt's Semantic Layer), but the [OSI spec](https://github.com/open-semantic-interchange/OSI) is an open-source attempt at standardizing it. DataSemAgent is my exploration of that spec — a working prototype built to see how far the idea can go.

---

## 🧭 How it works

1. 📐 **Model** — Define business terms (metrics, dimensions) in an OSI semantic model
2. 🔍 **Index** — Build entity graphs, metric lineage, and reasoning chains from your database
3. ⚡ **Ask** — Ask questions in natural language; the agent writes logical SQL, validates, translates, and executes it

---

## ⛓️ Pipelines

### 🧱 Semantic SQL

The agent writes *logical SQL* using business terms, then validates and translates it to physical SQL.

```
Logical                               Physical
──────────────────────────            ──────────────────────────────
SELECT customer_id, revenue           SELECT stg_orders.customer_id,
FROM orders                                  SUM(stg_orders.total_amount)
GROUP BY customer_id                  FROM stg_orders
                                      GROUP BY stg_orders.customer_id
```

**✅ Validation rules** (each an independent `Rule` class on top of `sqlglot`'s AST + Scope analysis):

- Metrics cannot be re-aggregated
- Dimensions must appear in `GROUP BY`
- Cross-dataset metrics require proper `JOIN`s
- Filter metrics in `HAVING`, not `WHERE`
- Metrics cannot appear in `GROUP BY`

### 🕸️ Entity Graph

LLM-driven pipeline turns every database row into a graph node and every foreign key into an edge.

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

**🔁 Pipeline:** Sampler → Schema Agent → Validator → Mapping Agent (incremental React loop) → Validator → Compiler  
Mapping builds step by step with inline validation on each call. Failed validation injects structured feedback into the next LLM call for automatic retry.

### 🌳 Metric Lineage

Built deterministically from the OSI model — no LLM calls needed.

```
┌──────────┐  AGGREGATES_FROM  ┌────────────────┐
│ Metric   │ ─────────────────►│ PhysicalField   │
│ revenue  │                   │ ss_ext_sales    │
│          │  SLICES_BY        │ _price          │
│          │ ─────────────────►│                 │
│          │                   │ Dimension       │
│          │                   │ customer_id     │
└──────────┘                   └────────────────┘
```

💡 Discover available metrics, their physical sources, and valid dimension slices — before writing SQL.

### 🧠 Reasoning Chain Graph

Background reflection extracts **reusable reasoning patterns** from conversations — not specific numbers ("profit was $1.2M"), but general approaches ("period-over-period comparison reveals seasonal volatility"). An ontology hierarchy organizes concepts; child facts inherit parent chains.

Reflection fires automatically every N turns (configurable via `config/agent.toml`). Use `/reflect` in ask mode to trigger manually. The agent is prompted to **check the reasoning graph first** when facing complex analytical problems — past experience saves time.

**🔁 Pipeline:** Reflector (builds prompt from last N rounds) → Agent (incremental EmitReasoningTool) → Synonym Merger → Builder → KuzuDB

---

## 🚀 Quick Start

```bash
git clone https://github.com/VilTea/DataSemAgent.git && cd DataSemAgent
uv sync
cp config/llm/config.toml.demo config/llm/config.toml   # add your API key
uv run python run.py --lang zh
```

### CLI Menu

| Command | Description |
|---------|-------------|
| `[1]` **init** | Build entity graph & metric lineage |
| `[2]` **ask** | Multi-turn agent Q&A |
| `[3]` **exit** | Quit |
| `[8]` **inspect** | Interactive reasoning graph browser |
| `[9]` **build-db** | Generate test database from OSI model |

### CLI Options

```bash
uv run python run.py --lang zh          # Chinese UI
uv run python run.py --lang en          # English UI
uv run python run.py --debug            # Verbose hook trigger logging
uv run python run.py --model path/to/model.yaml  # Custom OSI model path
```

### In-Ask Commands

| Command | Description |
|---------|-------------|
| `/reflect` | Manually trigger reasoning chain reflection |
| `/exit`, `/q`, `/quit` | Return to main menu |

### Build Test Database

Build a SQLite database from the OSI semantic model for development and testing:

```bash
# Via CLI menu: select [9] build-db (shows model/DB path confirmation)
# Or directly:
uv run python tests/build_tpcds_test_data.py --db data/test.db
```

Creates physical tables and populates them with realistic sample data — 3 years of dates, 10 customers/items, 5 stores, 500 transactions.

---

## 🧰 Agent Tools

| Tool | Description |
|------|-------------|
| `sql_exec` | Logical SQL with pre-aggregated metrics |
| `entity_graph` | Cypher queries over the entity property graph |
| `metric_lineage` | Discover metrics, dimensions, and their sources |
| `reasoning_graph` | Query accumulated reusable reasoning patterns |
| `todo_write` | Track multi-step task progress |

---

## ⚙️ Configuration

| Component | File |
|-----------|------|
| 🧠 LLM (`openai` / `anthropic`) | `config/llm/config.toml` |
| 🤖 Agent behavior | `config/agent.toml` |
| 🗄️ SQL Database | `config/database.toml` |
| 🔗 Graph Database | `config/graph_database.toml` |
| 📦 OSI Model | `config/config.toml` → `[paths]` |
| 🔌 MCP Servers | `config/mcp/servers.yaml` |

### Agent Settings (`config/agent.toml`)

```toml
[default]
reflection_interval = 5   # trigger reasoning reflection every N turns
```

---

## 📁 Project Structure

```
  run.py
  app/
  ├── semantics/sql/          # SQL translation (classifier, expander, translator, validator)
  ├── semantics/graph/        # Entity + metric + reasoning graphs
  │   ├── entity/             #   LLM-driven entity graph pipeline
  │   ├── metric/             #   Deterministic metric lineage
  │   └── reasoning/          #   Background reflection chain graph
  ├── node/                   # Agent orchestration (PocketFlow: AgentNode, ToolNode)
  ├── tool/                   # sql_exec, entity_graph, metric_graph, reasoning_graph, todo_write
  ├── llm.py                  # OpenAI / Anthropic client
  ├── hook/                   # Lifecycle hooks (node/tool/flow granularity, priority ordering)
  ├── pipeline/               # Streaming output (EventConsumer / Consumable abstraction)
  └── cli/                    # Typer + Rich terminal UI
  config/                     # Config + i18n (zh/en)
  tests/
```

---

## 📍 Status

This is a **personal validation-stage project**, not a product:

- Tested on TPC-DS datasets only; real-world schema coverage is unknown
- Entity graph quality depends on the underlying LLM; complex schemas haven't been thoroughly evaluated
- No performance optimization, no production deployment plan, rough error handling

It is a working prototype that explores what OSI-based semantic analysis can look like — not a turnkey tool.

---

## Why open source

- **Feedback.** Solo projects drift. If you work on semantic layers, Text-to-SQL, or LLM agents, I'd love to hear what you think.
- **OSI spec exploration.** Semantic layer standardization matters for reusable data components. This project is my way of stress-testing the spec.
- **Code that might help.** The PocketFlow orchestration, structured retry feedback, sqlglot AST patterns, and hook system could save someone some research time.

---

📄 Apache 2.0 · Copyright 2026 [VILTEA](https://github.com/VILTEA)
