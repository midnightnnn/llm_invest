<p align="center">
  <h1 align="center">🏟️ LLM Arena</h1>
  <p align="center">
    <b>Multi-LLM Autonomous Investment Arena</b><br>
    3 AI agents compete with real money, real tools, and zero human intervention.
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/python-3.12+-3776ab?style=flat-square&logo=python&logoColor=white" alt="Python 3.12+">
    <img src="https://img.shields.io/badge/license-MIT-22c55e?style=flat-square" alt="MIT License">
    <img src="https://img.shields.io/badge/framework-Google_ADK-4285f4?style=flat-square&logo=google&logoColor=white" alt="Google ADK">
    <img src="https://img.shields.io/badge/tests-600+-f59e0b?style=flat-square" alt="Tests">
  </p>
  <p align="center">
    <a href="#quick-start">Quick Start</a> ·
    <a href="#architecture">Architecture</a> ·
    <a href="#how-it-works">How It Works</a> ·
    <a href="#admin-ui">Admin UI</a> ·
    <a href="#cli-reference">CLI</a>
  </p>
</p>

---

> **GPT-5.2** vs **Gemini 3 Flash** vs **Claude Sonnet 4.6** — each agent independently discovers stocks,
> builds conviction, and executes real trades across **US + Korean markets**.
> No hardcoded strategies. No pre-filtered stock lists. **The LLM *is* the strategy.**

---

## Quick Start

### Prerequisites

- Python 3.12+
- GCP project with BigQuery + Firestore
- At least one LLM API key (OpenAI, Google AI, or Anthropic)

### 1. Clone & Install

```bash
git clone https://github.com/your-username/LLm_arena.git
cd LLm_arena
pip install -e .[dev]
```

> **Optional** — install forecasting models (PyTorch, NeuralForecast, Chronos, TimesFM):
> ```bash
> pip install -e .[dev,forecasting]
> ```

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
# GCP (required)
GOOGLE_CLOUD_PROJECT=your-gcp-project
BQ_DATASET=llm_arena

# LLM keys — at least one
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AI...
ANTHROPIC_API_KEY=sk-ant-...

# Live trading — optional, paper trading works without these
KIS_API_KEY=...
KIS_API_SECRET=...
KIS_ACCOUNT_NO=...
```

### 3. Initialize & Run

```bash
# Create database tables
llm-arena init-bq

# Run a trading cycle (paper trading by default)
llm-arena run-pipeline --market us      # NASDAQ + NYSE
llm-arena run-pipeline --market kospi   # KOSPI + KOSDAQ

# Launch Admin UI
llm-arena serve-ui                      # → http://localhost:8080
```

That's it — agents will analyze the market, draft strategies, review each other's picks, and execute trades autonomously.

---

## Architecture

```mermaid
flowchart TB
    %% ─── Entry Points ───
    subgraph ENTRY[" 🎯 Entry Points "]
        direction LR
        CLI(["🖥️ CLI<br>run-pipeline --market us|kospi"])
        SCHED(["⏰ Cloud Scheduler<br>US 15:00 ET · KR 14:30 KST"])
        ADMIN(["🎛️ Admin UI<br>Prompts · Risk · Tools · Memory"])
    end

    %% ─── Orchestrator ───
    ORCH{{"⚙️ Orchestrator"}}

    %% ─── Pre-Trade Pipeline ───
    subgraph PIPELINE[" 🔄 Pre-Trade Pipeline "]
        direction LR
        SYNC["📡 Sync<br><i>quotes · account<br>trades · cash</i>"]
        RECON["🔍 Reconcile<br><i>auto-recovery</i>"]
        FCAST["📈 Forecast<br><i>neural + foundation<br>model stacking</i>"]
        RSRCH["🔬 Research<br><i>holdings · movers</i>"]
        SYNC --> RECON --> FCAST --> RSRCH
    end

    %% ─── Agent Arena ───
    subgraph ARENA[" 🏟️ Agent Arena · Draft → Peer Review → Execute "]
        direction LR
        GPT["🟢 <b>GPT-5.2</b><br>OpenAI"]
        GEM["🔵 <b>Gemini 3 Flash</b><br>Google AI"]
        CLD["🟣 <b>Claude Sonnet 4.6</b><br>Anthropic"]
    end

    %% ─── Tool Layer ───
    subgraph TOOLS[" 🧰 18 Autonomous Tools + MCP "]
        direction LR
        TQ["📊 Quant<br><i>screen · optimize<br>forecast · technicals</i>"]
        TS["📰 Sentiment<br><i>reddit · SEC<br>earnings · F&G</i>"]
        TM["🌐 Macro<br><i>FRED · ECOS<br>indices</i>"]
        TC["🧠 Memory<br><i>vector search<br>peer lessons</i>"]
        TMCP["🔌 MCP<br><i>custom servers</i>"]
    end

    %% ─── Risk & Execution ───
    RISK{{"🛡️ Risk Engine<br><i>limits · buffers · cooldowns</i>"}}
    GW(["⚡ Execution Gateway"])

    %% ─── Storage Layer ───
    subgraph STORE[" 💾 Storage "]
        direction LR
        BQ[("BigQuery<br><i>Event Store</i>")]
        FS[("Firestore<br><i>Vector DB</i>")]
        KIS["🏦 KIS Broker API"]
    end

    %% ─── Flow ───
    ENTRY --> ORCH --> PIPELINE --> ARENA
    ARENA <-->|"tool calls"| TOOLS
    ARENA -->|"OrderIntent"| RISK
    RISK -->|"approved"| GW --> KIS
    TOOLS <--> BQ & FS
    ADMIN -.->|"live config"| BQ
    GW -.->|"trade log"| BQ

    %% ─── Styles ───
    classDef entry fill:#dbeafe,stroke:#3b82f6,stroke-width:2px,color:#1e40af
    classDef orch fill:#d1fae5,stroke:#059669,stroke-width:2.5px,color:#065f46
    classDef pipe fill:#e0e7ff,stroke:#6366f1,stroke-width:1.5px,color:#312e81
    classDef agent fill:#ede9fe,stroke:#8b5cf6,stroke-width:2.5px,color:#4c1d95
    classDef tool fill:#ecfdf5,stroke:#10b981,stroke-width:1.5px,color:#064e3b
    classDef risk fill:#fee2e2,stroke:#ef4444,stroke-width:2.5px,color:#991b1b
    classDef gw fill:#fef3c7,stroke:#d97706,stroke-width:2px,color:#92400e
    classDef store fill:#fff7ed,stroke:#f97316,stroke-width:2px,color:#9a3412

    class CLI,SCHED,ADMIN entry
    class ORCH orch
    class SYNC,RECON,FCAST,RSRCH pipe
    class GPT,GEM,CLD agent
    class TQ,TS,TM,TC,TMCP tool
    class RISK risk
    class GW gw
    class BQ,FS,KIS store
```

<details>
<summary><b>Project Structure</b></summary>

```
arena/
  agents/          # ADK ReAct agents + Research + Memory Compaction
  memory/          # Long-term memory (store, vector, policy, query, cleanup)
  ui/              # Admin UI (FastAPI + Jinja2 + HTMX)
  tools/           # Tool registry (quant, sentiment, macro, context)
  data/            # BigQuery repositories + schema
  broker/          # Paper / Live (KIS) broker adapters
  execution/       # Central order gateway
  open_trading/    # KIS client + account sync
  forecasting/     # Multi-model stacking forecasts
  security/        # Secret Manager integration
  config.py        # Settings + runtime overrides
  context.py       # Context builder + memory reranking
  orchestrator.py  # Cycle orchestration
  risk.py          # Risk engine
tests/             # 600+ test cases (pytest)
scripts/           # Deploy scripts + migrations
```

</details>

---

## How It Works

```mermaid
flowchart TD
    START(["⏰ Scheduler Trigger<br>US 15:00 ET · KR 14:30 KST · Mon–Fri"])
    START --> HOL{"🗓️ Holiday?"}
    HOL -->|"Yes"| SKIP(["Skip"])
    HOL -->|"No"| H["1 · Hydrate tenant runtime<br><i>secrets + config</i>"]
    H --> S["2 · Sync market data + broker<br><i>quotes · account · trades · cash · dividends</i>"]
    S --> R["3 · Reconciliation + auto-recovery"]
    R --> F["4 · Build forecasts<br><i>neural + foundation models</i>"]
    F --> RS["5 · Research Agent<br><i>holdings analysis + movers</i>"]
    RS --> DRAFT["6 · Draft Round<br><i>3 agents in parallel · analysis only</i>"]
    DRAFT --> EXEC["7 · Execution Round<br><i>3 agents in parallel · real trades</i><br>OrderIntent → RiskEngine → Broker → KIS API"]
    EXEC --> NAV["8 · Record official NAV + compress memories"]

    classDef trigger fill:#dbeafe,stroke:#3b82f6,stroke-width:2px,color:#1e40af
    classDef decision fill:#fef3c7,stroke:#d97706,stroke-width:2px,color:#92400e
    classDef step fill:#f0fdf4,stroke:#22c55e,stroke-width:1.5px,color:#14532d
    classDef skip fill:#fee2e2,stroke:#ef4444,stroke-width:1.5px,color:#991b1b

    class START trigger
    class HOL decision
    class H,S,R,F,RS,DRAFT,EXEC,NAV step
    class SKIP skip
```

### The Agents

| Agent | Model | Provider |
|-------|-------|----------|
| GPT | GPT-5.2 | OpenAI |
| Gemini | Gemini 3 Flash | Google AI / Vertex AI |
| Claude | Claude Sonnet 4.6 | Anthropic / Vertex AI |

Each agent runs on [Google ADK](https://github.com/google/adk-python) with ReAct reasoning and gets an independent virtual portfolio tracked against a single brokerage account.

---

## Tools

Agents autonomously choose which tools to call at each reasoning step.

| Tool | Category | Description |
|------|----------|-------------|
| `get_research_briefing` | Context | Research via Gemini Google Search Grounding |
| `search_past_experiences` | Context | Semantic search over past memories |
| `search_peer_lessons` | Context | Lessons learned by other agents |
| `portfolio_diagnosis` | Context | Holdings diagnosis + HRP rebalance plan |
| `save_memory` | Context | Persist a manual memory note |
| `screen_market` | Quant | Universe screening with filters |
| `optimize_portfolio` | Quant | Portfolio optimization + rebalance orders |
| `forecast_returns` | Quant | Neural + foundation model stacking forecasts |
| `technical_signals` | Quant | RSI / MACD / Bollinger / SMA |
| `correlation_matrix` | Quant | Correlation analysis |
| `sector_summary` | Quant | Per-sector return & volatility |
| `get_fundamentals` | Quant | Valuation metrics (PER / PBR / ROE) |
| `index_snapshot` | Macro | Major index quotes (auto-routed by market) |
| `macro_snapshot` | Macro | Macro indicators (US: FRED, KR: ECOS) |
| `fear_greed_index` | Macro | VIX-based fear/greed gauge |
| `earnings_calendar` | Macro | Earnings schedule |
| `fetch_reddit_sentiment` | Sentiment | Social sentiment |
| `fetch_sec_filings` | Sentiment | SEC EDGAR filings |

> **+ MCP** — Add custom tool servers via Admin UI (SSE / Streamable HTTP).

---

## Admin UI

All settings live in BigQuery and take effect on the next cycle — **no redeploy needed**.

| Page | Description |
|------|-------------|
| **Prompt** | System prompt that guides agent behavior |
| **Agents** | Add/remove agents, swap models, per-agent overrides |
| **Risk** | Position limits, cash buffers, cooldowns, turnover caps |
| **Sleeve** | Per-agent target capital allocation |
| **Tools** | Toggle built-in tools on/off per cycle |
| **MCP** | Register custom tool servers |
| **Memory** | 3D neural graph visualization of memory policy |

---

## Memory System

```mermaid
flowchart LR
    ST["💾 Store<br><i>BigQuery + Firestore</i>"]
    RT["🔍 Retrieve + Rerank<br><i>vector search → scoring<br>type · freshness · ticker · perf</i>"]
    IN["💉 Inject<br><i>cycle context +<br>mid-REACT tool calls</i>"]
    CO["🗜️ Compress<br><i>MemoryCompactionAgent<br>→ strategic lessons</i>"]
    CL["🧹 Cleanup<br><i>forgetting curves ·<br>policy-driven expiry</i>"]

    ST --> RT --> IN --> CO --> CL
    CL -.->|"feedback loop"| ST

    classDef store fill:#dbeafe,stroke:#3b82f6,stroke-width:2px,color:#1e40af
    classDef retrieve fill:#ecfdf5,stroke:#10b981,stroke-width:2px,color:#064e3b
    classDef inject fill:#fef3c7,stroke:#d97706,stroke-width:2px,color:#92400e
    classDef compress fill:#ede9fe,stroke:#8b5cf6,stroke-width:2px,color:#4c1d95
    classDef cleanup fill:#fee2e2,stroke:#ef4444,stroke-width:2px,color:#991b1b

    class ST store
    class RT retrieve
    class IN inject
    class CO compress
    class CL cleanup
```

10 policy groups — Storage, Event Types, Hierarchy, Tagging, Forgetting, Graph, Compaction, Retrieval, REACT Injection, Cleanup — all editable through the 3D Memory Graph in Admin UI.

---

## Multi-Tenant

| Feature | Description |
|---------|-------------|
| Auto-provisioning | New users get a `simulated_only` tenant on first login |
| Public demo | Optionally expose an operator-funded tenant as read-only |
| Paper trading | Activates when KIS demo credentials are saved |
| Live trading | Requires explicit backend approval (`promote-tenant-live`) |
| Data isolation | Trades, portfolios, memory, config — fully isolated per tenant |
| BYOK | Each tenant brings their own LLM API keys |

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `init-bq` | Create BigQuery tables |
| `run-pipeline --market us\|kospi` | Full pipeline: sync → forecast → trade |
| `run-shared-prep --market us` | Shared sync + forecast, then dispatch agents |
| `run-agent-cycle --market us` | Agent trading cycle only |
| `serve-ui` | Launch Admin UI (port 8080) |
| `recover-sleeves` | Checkpoint rebuild + re-reconcile |
| `promote-tenant-live --tenant <id>` | Promote tenant to live trading |
| `set-tenant-simulated --tenant <id>` | Reset tenant to simulated mode |

Add `--live` for live trading mode. Add `--all-tenants` for multi-tenant batch.

<details>
<summary>All sync & utility commands</summary>

| Command | Description |
|---------|-------------|
| `sync-market` | Sync market features |
| `sync-market-quotes` | Sync latest quotes |
| `sync-account` | Sync broker account snapshot |
| `sync-broker-trades` | Sync broker trade history |
| `sync-broker-cash` | Sync broker cash events |
| `sync-dividends` | Sync dividend records |
| `build-forecasts` | Generate return forecasts |

</details>

---

## Deployment

```bash
# Dual market jobs (US + KOSPI on separate schedules)
DUAL_MARKET=true bash scripts/deploy_cloud_run_job.sh

# Admin UI
bash scripts/deploy_cloud_run_ui.sh
```

| Component | Schedule |
|-----------|----------|
| US Job | 15:00 ET, Mon–Fri |
| KOSPI Job | 14:30 KST, Mon–Fri |
| Admin UI | Always-on |

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Agent Framework | [Google ADK](https://github.com/google/adk-python) (ReAct) |
| LLM Providers | OpenAI, Google AI / Vertex AI, Anthropic |
| Database | BigQuery (event store) + Firestore (vector search) |
| Broker | [KIS Open Trading API](https://apiportal.koreainvestment.com/) |
| Forecasting | NeuralForecast, Chronos, TimesFM, Lag-Llama |
| UI | FastAPI + Jinja2 + HTMX |
| Infra | GCP Cloud Run Jobs + Cloud Run Service |

---

## Development

```bash
pip install -e .[dev]
pytest                        # 600+ tests
pytest tests/test_risk.py -v  # specific module
```

---

## License

[MIT](LICENSE) — Copyright (c) 2026 midnightnnn
