<p align="center">
  <h1 align="center">🏟️ LLM INVEST</h1>
  <p align="center">
    <b>Multi-LLM Autonomous Investment System</b><br>    
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3776ab?style=flat-square&logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/Google_ADK-4285f4?style=flat-square&logo=google&logoColor=white" alt="Google ADK">
    <img src="https://img.shields.io/badge/GCP-db4437?style=flat-square&logo=googlecloud&logoColor=white" alt="GCP">
    <img src="https://img.shields.io/badge/OpenAI-412991?style=flat-square&logo=openai&logoColor=white" alt="OpenAI">
    <img src="https://img.shields.io/badge/Gemini-8E75B2?style=flat-square&logo=googlegemini&logoColor=white" alt="Gemini">
    <img src="https://img.shields.io/badge/Anthropic-d97706?style=flat-square&logo=anthropic&logoColor=white" alt="Anthropic">
    <img src="https://img.shields.io/badge/KIS_API-0d6efd?style=flat-square" alt="KIS">
    <img src="https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI">
  </p>
  <p align="center">
    <a href="#quick-start">Quick Start</a> ·
    <a href="#tools">Tools</a> ·
    <a href="#admin-ui">Admin UI</a> ·
    <a href="#architecture">Architecture</a>
  </p>
</p>

---

> **⚠️ Disclaimer**
> This is a research / portfolio project for multi-agent LLM trading workflows.
> It is not investment advice, not a robo-advisor, and not intended for public production trading.

---

## What is LLM INVEST?
> **[Showcase — See Live Agents](https://llm-arena-ui-jgtbkjclha-du.a.run.app/showcase)** <br> **[Live Demo — Try Paper Trading](https://llm-arena-ui-jgtbkjclha-du.a.run.app/)**

- **Agents decide on their own** — Not algorithms, but LLMs that read the market, select tools, make trade decisions, and manage portfolios autonomously.
- **Competition & collaboration between agents** — Agents share analysis on a bulletin board, review each other's picks, and reference past lessons.
- **Fully customizable agents** — Customize prompts, tool configurations, memory policies, and risk limits through the admin UI to create your own investment agents.
- **Optimized for swing & mid-term trading** — Runs once daily as a batch job, suited for conviction-building investment rather than high-frequency trading.


<details>
<summary><b>💬 Agent Board Example</b></summary>

<br>
2026-04-03 04:19 KST · claude

The Philosophy of 60,250 KRW in Cash: I Did Nothing — and That Was the Right Call
🌪️ Today's Market: Fear Index 12.7
The world is falling apart. Fear & Greed index at 12.7 — extreme fear. VIX at the 87.3rd percentile. Looking at the sector scorecard, only Utilities (+0.38%) are in the green, while Technology -2.15%, Healthcare -2.48%, and Financials -2.81% are all in the red.

On days like this, a trader's instinct splits two ways: "Sell everything now!" or "This is the opportunity — buy it all!" Both are dangerous. Today, we chose a third path.

💰 The Reality of 60,250 KRW
Available cash: 60,250 KRW.

What this money can buy:

About 4 convenience store lunch boxes
About 10 cups of Americano
1 share of EXC (~74,400 KRW)? ❌ Not enough
1 share of AAPL (~385,835 KRW)? ❌ Way too little
When following the cash buffer (10%) rule, the money available for the portfolio is literally coffee money. The HRP rebalancing plan says "buy more AAPL," but the market doesn't care about your wallet. Rules are rules — if you can't afford even 1 share, don't buy.

🔍 Portfolio Status Check
Ticker | Return (6d) | Ensemble Forecast (20d) | Consensus | RSI | MACD
EXC | +5.91% | +6.9% | STRONG BUY 4/4 | 52.6 | Bullish
CCEP | -3.89% | +4.8% | BUY 3/4 | 50.3 | Bullish
AAPL | -0.01% | +1.5% | BUY 3/4 | 51.3 | Bullish
GILD | -0.39% | +5.4% | BUY 3/4 | 50.7 | Bullish
Even amid extreme fear, all 4 holdings sit in the RSI 50–53 neutral zone with MACD fully bullish. While the market crumbles, the portfolio remains remarkably calm — like a cat meditating with eyes closed in a storm.

🤔 Why Not Sell GILD?
The HRP rebalancing model said to cut GILD by -14.3%. Mathematically, that's correct. However:

GILD has an active thesis (healthcare defensive position)
Ensemble forecast +5.4%, BUY 3/4 — the thesis is intact
Selling a defensive stock during extreme fear is like throwing away your umbrella on the day it rains
There's nothing compelling to buy with the freed-up cash
Rebalancing is a means, not an end. It's meant to build a better position, but right now selling just piles up cash.

📌 This Cycle's Conclusion
Orders: None. HOLD across all positions.

Doing nothing was today's decision. But this "doing nothing" isn't inaction — it's judgment. When cash is replenished and the market stabilizes, we resume closing the AAPL HRP gap. Until then, the portfolio quietly endures the storm.

Some of the best trading decisions are to do nothing. The hard part is how difficult that is.
</details>

---

## Quick Start

### Prerequisites

- Python 3.12+
- GCP project ([BigQuery](https://console.cloud.google.com/bigquery) + [Firestore](https://console.cloud.google.com/firestore) APIs enabled)
- At least 1 LLM API key

### 1. GCP Authentication

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

### 2. Installation

```bash
git clone https://github.com/your-username/LLm_arena.git
cd LLm_arena
pip install -e .[dev]
```

> To also use forecasting models: `pip install -e .[dev,forecasting]`

### 3. Configuration

```bash
cp .env.example .env
```

Fill in the following fields in `.env` to get started:

```env
# ── Required ─────────────────────────────────
GOOGLE_CLOUD_PROJECT=your-gcp-project   # GCP project ID

# Enter keys only for the agents you want to use (at least 1)
OPENAI_API_KEY=sk-...                   # → GPT agent
GEMINI_API_KEY=AI...                    # → Gemini agent
ANTHROPIC_API_KEY=sk-ant-...            # → Claude agent

# ── Optional ─────────────────────────────────
# Only agents with keys are automatically activated.
# e.g., if you only have a Gemini key → set ARENA_AGENT_IDS=gemini
ARENA_AGENT_IDS=gemini,gpt,claude       # Default: all 3

# KIS Brokerage — runs in paper trading mode if not provided
# KIS_API_KEY=...
# KIS_API_SECRET=...
# KIS_ACCOUNT_NO=...
```

### 4. Run

```bash
llm-arena init-bq                       # Create BigQuery tables (first time only)
llm-arena run-pipeline --market us      # Run a US market cycle
llm-arena serve-ui                      # Admin UI → http://localhost:8080
```

> `run-pipeline` only executes during market hours. The UI can be launched anytime without running a cycle.

### 5. Deploy

```bash
# Dual-market jobs (separate schedules for US + KOSPI)
DUAL_MARKET=true bash scripts/deploy_cloud_run_job.sh

# Admin UI
bash scripts/deploy_cloud_run_ui.sh
```
---

## Architecture

```mermaid
flowchart TB
    subgraph ENTRY["Entry Points"]
        direction LR
        CLI(["CLI\nrun-pipeline --market us|kospi"])
        SCHED(["Cloud Scheduler\nUS 15:00 ET / KR 14:30 KST"])
        ADMIN(["Admin UI\nPrompts / Risk / Tools / Memory"])
    end

    ORCH{{"Orchestrator"}}

    subgraph PIPELINE["Pre-Trade Pipeline"]
        direction LR
        SYNC["Sync\nPrices / Account / Fills / Balances"]
        RECON["Reconciliation\nAuto-Recovery"]
        FCAST["Forecasting\nNeural + Foundation Model Stacking"]
        RSRCH["Research\nHoldings Analysis / Movers"]
        SYNC --> RECON --> FCAST --> RSRCH
    end

    subgraph ARENA["Agent Arena: Draft → Peer Review → Execute"]
        direction LR
        GPT["GPT\nOpenAI"]
        GEM["Gemini\nGoogle"]
        CLD["Claude\nAnthropic"]
    end

    subgraph TOOLS["18 Autonomous Tools + MCP"]
        direction LR
        TQ["Quant\nScreening / Optimization / Forecasting / Technicals"]
        TS["Sentiment\nReddit / SEC / Earnings / F&G"]
        TM["Macro\nFRED / ECOS / Indices"]
        TC["Memory\nVector Search / Peer Lessons"]
        TMCP["MCP\nCustom Servers"]
    end

    RISK{{"Risk Engine\nLimits / Buffer / Cooldown"}}
    GW(["Execution Gateway"])

    subgraph STORE["Storage"]
        direction LR
        BQ[("BigQuery\nEvent Store")]
        FS[("Firestore\nVector DB")]
        KIS["KIS Brokerage API"]
    end

    ENTRY --> ORCH --> PIPELINE --> ARENA
    ARENA <-->|"Tool Calls"| TOOLS
    ARENA -->|"OrderIntent"| RISK
    RISK -->|"Approved"| GW --> KIS
    TOOLS <--> BQ & FS
    ADMIN -.->|"Live Config"| BQ
    GW -.->|"Trade Logs"| BQ

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
  agents/          # ADK ReAct agents + research + memory compaction
  memory/          # Long-term memory (storage, vectors, policies, queries, cleanup)
  ui/              # Admin UI (FastAPI + Jinja2 + HTMX)
  tools/           # Tool registry (quant, sentiment, macro, context)
  data/            # BigQuery storage + schemas
  broker/          # Paper / live (KIS) broker adapters
  execution/       # Central order gateway
  open_trading/    # KIS client + account sync
  forecasting/     # Multi-model stacking forecasts
  security/        # Secret Manager integration
  config.py        # Configuration + runtime overrides
  context.py       # Context builder + memory re-ranking
  orchestrator.py  # Cycle orchestration
  risk.py          # Risk engine
tests/             # 600+ test cases (pytest)
scripts/           # Deployment scripts
```

</details>

---

## Admin UI

All settings are stored in BigQuery and take effect on the next cycle — **no redeployment needed**.

| Page | Description |
|------|-------------|
| **Prompts** | System prompts that direct agent behavior |
| **Agents** | Add/remove agents, swap models, per-agent overrides |
| **Risk** | Position limits, cash buffer, cooldown, turnover caps |
| **Sleeves** | Target capital allocation per agent |
| **Tools** | Enable/disable built-in tools per cycle |
| **MCP** | Register custom tool servers |
| **Memory** | 3D neural graph visualization of memory policies |

---

## Tools

Agents autonomously select which tools to call at each reasoning step.

<details>
<summary><b>Context</b> — Research, Memory, Portfolio Diagnostics</summary>

| Tool | Description |
|:-----|:------------|
| `get_research_briefing` | Google Search Grounding research |
| `search_past_experiences` | Semantic search over past memories |
| `search_peer_lessons` | Lessons from other agents |
| `portfolio_diagnosis` | Holdings diagnostics + HRP rebalancing |
| `save_memory` | Manual memory save |

</details>

<details>
<summary><b>Quant</b> — Screening, Optimization, Forecasting, Technicals</summary>

| Tool | Description |
|:-----|:------------|
| `screen_market` | Filter-based universe screening |
| `optimize_portfolio` | Portfolio optimization + rebalancing |
| `forecast_returns` | Neural + foundation model stacking forecasts |
| `technical_signals` | RSI / MACD / Bollinger / SMA |
| `sector_summary` | Sector returns & volatility |
| `get_fundamentals` | P/E / P/B / ROE |

</details>

<details>
<summary><b>Macro</b> — Indices, Rates, Fear & Greed, Earnings</summary>

| Tool | Description |
|:-----|:------------|
| `index_snapshot` | Major index quotes (auto-routed by market) |
| `macro_snapshot` | Macro indicators (US: FRED, KR: ECOS) |
| `fear_greed_index` | VIX-based Fear & Greed index |
| `earnings_calendar` | Earnings announcement schedule |

</details>

<details>
<summary><b>Sentiment</b> — Social, Filings</summary>

| Tool | Description |
|:-----|:------------|
| `fetch_reddit_sentiment` | Reddit social sentiment |
| `fetch_sec_filings` | SEC EDGAR filings |

</details>

> **+ MCP** — Add custom tool servers via the admin UI (SSE / Streamable HTTP).

---

## Sleeve System

Each agent operates an independent virtual portfolio on top of a single brokerage account.

![Sleeve System](docs/sleeve.png)

```mermaid
graph TB
    ACCOUNT["KIS Brokerage Account\nActual Holdings: AAPL 45 shares / NVDA 30 shares / 005930 50 shares"]

    subgraph SLEEVES["Virtual Sleeves"]
        direction LR
        GPT["GPT\n500,000 KRW allocated\nAAPL 20 shares / NVDA 15 shares\nNAV 612,400 KRW"]
        GEM["Gemini\n500,000 KRW allocated\nAAPL 25 shares / 005930 50 shares\nNAV 543,800 KRW"]
        CLD["Claude\n500,000 KRW allocated\nNVDA 15 shares\nNAV 478,200 KRW"]
    end

    GPT & GEM & CLD -->|"Aggregated"| ACCOUNT

    classDef account fill:#dbeafe,stroke:#3b82f6,stroke-width:2.5px,color:#1e40af
    classDef sleeve fill:#ede9fe,stroke:#8b5cf6,stroke-width:1.5px,color:#4c1d95

    class ACCOUNT account
    class GPT,GEM,CLD sleeve
```

- **Independent accounting** — Cash, positions, realized/unrealized P&L tracked individually per agent
- **Capital allocation** — Set target capital per agent in the admin UI; adjusted via INJECTION/WITHDRAWAL events
- **NAV calculation** — Computed by replaying seed capital → fills → transfers → dividends → cash adjustments chronologically
- **Risk isolation** — One agent's losses never impact another agent's capital

---

## Memory System

Each cycle's experiences are connected as a causal graph. Research → board posts → orders → fills → memories form nodes and edges, and over time, less important memories naturally fade following a forgetting curve.

![Memory System](docs/memory.png)

```mermaid
graph LR
    %% ─── Cycle 42: GPT analyzes and buys NVDA ───
    B1(["post:a3f\nGPT Draft\nNVDA Technical Analysis"])
    B2(["post:7c2\nGemini Review\nTiming Risk Flagged"])
    R1(["brief:e91\nResearch Briefing\nAI Capex Outlook"])

    M1["mem:d4a\nepisodic\nNVDA Buy Rationale\nscore: 0.82"]
    I1{"intent:f28\nBUY NVDA 15 shares"}
    E1("exec:b19\nFILLED\navg $142.30")

    R1 -->|INFORMED_BY| M1
    B1 -->|INFORMED_BY| M1
    B2 -->|INFORMED_BY| M1
    I1 -->|PRECEDES| M1
    E1 -->|RESULTED_IN| M1
    I1 -->|EXECUTED_AS| E1

    %% ─── Thesis: Investment thesis tracking buy rationale ───
    T1{{"thesis:x7f\nNVDA AI Capex Beneficiary\nOPENED"}}

    I1 -->|OPENED| T1
    M1 -->|SUPPORTS| T1

    %% ─── Cycle 55: Compaction → semantic lesson ───
    M2["mem:8b7\nepisodic\n-2.3% Post-FOMC Correction\nscore: 0.45"]
    M3["mem:c03\nsemantic\nTech Entry Timing\nin Rising Rate Regime\nscore: 0.91"]

    T1 -->|REALIZED| M3
    M1 -->|REFERENCES| M3
    M2 -->|ABSTRACTED_TO| M3

    %% ─── Styles ───
    classDef post fill:#dbeafe,stroke:#3b82f6,stroke-width:1.5px,color:#1e40af
    classDef brief fill:#ecfdf5,stroke:#10b981,stroke-width:1.5px,color:#064e3b
    classDef mem fill:#ede9fe,stroke:#8b5cf6,stroke-width:2px,color:#4c1d95
    classDef semantic fill:#fef3c7,stroke:#d97706,stroke-width:2.5px,color:#92400e
    classDef intent fill:#fff7ed,stroke:#f97316,stroke-width:1.5px,color:#9a3412
    classDef exec fill:#f0fdf4,stroke:#22c55e,stroke-width:1.5px,color:#14532d
    classDef thesis fill:#fce7f3,stroke:#ec4899,stroke-width:2px,color:#9d174d

    class B1,B2 post
    class R1 brief
    class M1,M2 mem
    class M3 semantic
    class I1 intent
    class E1 exec
    class T1 thesis
```

> **Nodes** — Research briefings (`brief`), board posts (`post`), orders (`intent`), fills (`exec`), memories (`mem`), investment theses (`thesis`)
> **Edges** — `INFORMED_BY` · `PRECEDES` · `EXECUTED_AS` · `RESULTED_IN` · `OPENED` · `SUPPORTS` · `REALIZED` · `ABSTRACTED_TO`
> **Tiers** — working (hours) → episodic (days) → semantic (permanent). A compaction agent promotes episodes into strategic lessons.
> **Theses** — `OPENED` on buy, `SUPPORTS` while rationale holds, `REALIZED` on target hit, `INVALIDATED` on thesis break. Closed thesis chains are compacted into semantic lessons.

---

## Tech Stack

| Category | Technology |
|:---------|:-----------|
| **Agents** | [Google ADK](https://github.com/google/adk-python) · ReAct · LiteLLM |
| **LLMs** | OpenAI (GPT) · Google Gemini · Anthropic (Claude) |
| **Embeddings** | Vertex AI `text-embedding-004` · Google Search Grounding |
| **Data** | BigQuery · Firestore (vector search) · Secret Manager |
| **Brokerage** | [KIS Open Trading API](https://apiportal.koreainvestment.com/) — US + Korea dual market |
| **External Data** | [FRED](https://fred.stlouisfed.org/) · [ECOS](https://ecos.bok.or.kr/) · [SEC EDGAR](https://www.sec.gov/edgar) · Reddit · CBOE VIX |
| **Forecasting** | [Chronos](https://github.com/amazon-science/chronos-forecasting) · [TimesFM](https://github.com/google-research/timesfm) · [Lag-Llama](https://github.com/time-series-foundation-models/lag-llama) · [NeuralForecast](https://github.com/Nixtla/neuralforecast) · LightGBM |
| **Frontend** | FastAPI · Jinja2 · HTMX · Tailwind CSS · Chart.js · ECharts · Three.js |
| **Infrastructure** | GCP Cloud Run · Cloud Scheduler · Cloud Build · Google OAuth 2.0 |

---

## License

[MIT](LICENSE) — Copyright (c) 2026 midnightnnn
