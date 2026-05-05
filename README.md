# 🤖 DataSemAgent

> [中文版本](README_zh.md)

**Semantic data analysis agent** built on the [OSI specification](https://github.com/open-semantic-interchange/OSI).
Translates business questions into executable queries across three integrated pipelines.

---

## 🧭 How it works

1. 📐 **Model** — Define business terms (metrics, dimensions) in an OSI semantic model
2. 🔍 **Index** — Build entity graphs and metric lineage from your database
3. ⚡ **Ask** — Ask questions in natural language; the agent writes logical SQL, validates, translates, and executes it

---

## ⛓️ Pipelines

### 🧱 Semantic SQL

The agent writes logical SQL using business terms, then validates and translates it to physical SQL.

```
Logical                               Physical
──────────────────────────            ──────────────────────────────
SELECT customer_id, revenue           SELECT stg_orders.customer_id,
FROM orders                                  SUM(stg_orders.total_amount)
GROUP BY customer_id                  FROM stg_orders
                                      GROUP BY stg_orders.customer_id
```

**✅ Validation rules:**
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

**🔁 Pipeline:** Sampler → Schema Agent → Validator → Mapping Agent → Validator → Compiler
Auto-retries on failure with structured feedback to the LLM.

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

---

## 🚀 Quick Start

```bash
git clone https://github.com/VilTea/DataSemAgent.git && cd DataSemAgent
uv sync
cp config/llm/config.toml.demo config/llm/config.toml   # add your API key
uv run python run.py --lang zh
```

| Command | Description |
|---------|-------------|
| `[1]` **init** | Build entity graph & metric lineage |
| `[2]` **ask** | Multi-turn agent Q&A |
| `[3]` **exit** | Quit |

---

## ⚙️ Configuration

| Component | File |
|-----------|------|
| 🧠 LLM (`openai` / `anthropic`) | `config/llm/config.toml` |
| 🗄️ SQL Database | `config/database.toml` |
| 🔗 Graph Database | `config/graph_database.toml` |
| 📦 OSI Model | `config/config.toml` → `[paths]` |
| 🔌 MCP Servers | `config/mcp/servers.yaml` |

---

## 📁 Project Structure

```
  run.py
  app/
  ├── semantics/sql/        # SQL translation
  ├── semantics/graph/      # Entity + metric graphs
  ├── node/                 # Agent orchestration
  ├── tool/                 # sql_exec, entity_graph, metric_lineage
  ├── llm.py                # OpenAI / Anthropic
  ├── hook/                 # Lifecycle hooks
  ├── pipeline/             # Streaming output
  └── cli/                  # Terminal UI
  config/                   # Config + i18n (zh/en)
  tests/
```

---

📄 Apache 2.0 · Copyright 2026 [VILTEA](https://github.com/VILTEA)
