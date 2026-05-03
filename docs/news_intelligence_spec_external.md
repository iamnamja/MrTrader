# News Intelligence Layer — Consolidated Implementation Spec

**Project:** TradingBot — Automated Swing + Intraday Equity Trading System
**Component:** News Ingestion, Event Interpretation, and Decision Integration Layer
**Document type:** External design review (Claude + ChatGPT synthesis) — preserved as reference
**Date:** April 27, 2026
**Status:** Reference only. See `phases_58_onwards_plan.md` for what we're actually building and when.

> **How to use this document:**
> This is the full external spec produced by consulting two LLMs on the news intelligence
> architecture question. It is intentionally ambitious. `phases_58_onwards_plan.md` contains
> the honest assessment of what's premature vs. what to build, and maps each section of this
> spec to the appropriate phase (58–66).
>
> Key sections to reference when implementing:
> - **SQL schemas** → Section 3 (use verbatim for Phase 58/61 tables)
> - **Python dataclasses** → Section 4 (use for Phase 60 `NewsSignal`)
> - **Prompt template** → Section 8.5 (use for Phase 60 Haiku scorer)
> - **Decision policy rules** → Section 11 (use for Phase 63 scoring overlay)
> - **Config YAML** → Section 17 (use for Phase 60+ threshold management)

---

## 0. How to Read This Document

This is a complete specification for replacing the current boolean news gate with a structured **News Intelligence Service** (NIS). It is written so that an implementing agent can build it phase-by-phase without ambiguity.

**Conventions:**
- All thresholds are starting values. They MUST be configurable in YAML.
- All schemas include `schema_version` strings — never break compatibility silently.
- All LLM prompts are versioned (`prompt_version`) and persisted with each output for reproducibility.
- Every decision the system makes (block, size-down, boost) MUST be logged with a stable signal_id so it can be audited and back-tested later.
- "MUST" / "SHOULD" / "MAY" follow RFC 2119 conventions.

**What is NOT being changed in this spec:**
- The XGBoost models themselves (training pipeline, walk-forward methodology, feature engineering for price/volume/technicals)
- The Alpaca execution path
- The PM → RM → Trader agent pattern
- The 09:45 swing scan and 60/90/120-minute intraday scan windows
- The Claude Haiku integration for the existing 3 audit/narrative tasks

This spec only adds a new upstream service (NIS) and modifies how PM and RM consume its output. The existing XGBoost model continues to score on price/volume/technical features. News integrates initially as a **runtime overlay** (sizing/veto adjustments to model output), and only becomes a model input feature in Phase 4 after sufficient point-in-time history has been collected.

---

## 1. Core Design Principles

These principles are non-negotiable and govern every later decision in this spec.

### 1.1 The LLM is an interpreter, not a trader
The LLM produces structured event interpretations (event_type, materiality, direction, confidence, novelty, already_priced_in). Deterministic Python code maps those interpretations to trading policy. The LLM never sees the words "buy," "sell," "long," "short," or "trade" in its input. This:
- Prevents the LLM from being too conservative or too aggressive based on its training priors
- Keeps trading policy auditable, deterministic, and testable
- Makes the LLM swappable (Claude, GPT, Llama) without changing trading behavior

### 1.2 Separate the gate from the feature
The current system collapses news into a single fuzzy mechanism. The new system has two distinct consumption paths:
- **Risk overlay path** — narrow, conservative gates for catastrophic, discrete, model-invalidating events: earnings within holding window, trading halt, M&A pending, FDA action, hard regulatory event. Binary or near-binary.
- **Continuous score path** — materiality-weighted, confidence-weighted scores that adjust position size and conviction. Continuous.

These do different jobs and MUST not be conflated.

### 1.3 Point-in-time everything
Every news event, signal, and decision is stored with three timestamps: `published_at`, `ingested_at`, `evaluated_at`. Backtests use only data available at the simulated decision time. Revised summaries and updated articles do not retroactively rewrite history. This is the precondition for honest backtesting of any news-aware model.

### 1.4 Fail open, but degrade observably
News API outages, LLM outages, or schema errors MUST NOT block trading. The system falls back to the next-cheapest interpretation tier (LLM → deterministic rules → cached signal with `stale=true` → no signal). Every fallback is logged so operators can see when the news layer is degraded.

### 1.5 Cluster-level signals, not article-level
The same story syndicated across 8 outlets is one event, not eight. Clustering happens before scoring. Source count and source diversity within a cluster become **inputs to confidence**, not duplicate signals.

### 1.6 Information decays
A news event is most impactful in the first hour, declines through the day, and is largely priced in by the next session unless it is a slow-burning narrative. Materiality scores in rolling features apply exponential decay with horizon-appropriate half-lives.

### 1.7 Versioning is non-optional
- `prompt_version` is stored with every LLM call output
- `schema_version` is stored with every signal
- `model_provider` and `model_name` are stored with every LLM call
- Without this, no honest backtest is possible.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                  Source Adapters (pluggable)                    │
│   Polygon · Alpaca · SEC EDGAR · RSS · Earnings · Macro · Halts │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              Layer 1 – Ingestion & Normalization                │
│       Common schema, content-hash dedupe, entity match          │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              Layer 2 – Clustering                               │
│    Group syndicated/repeat stories into a single cluster        │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              Layer 3 – Deterministic Pre-Scoring                │
│  Source reliability, regex event detectors, cheap discardable   │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              Layer 4 – LLM Materiality Scorer                   │
│      Haiku primary · Sonnet escalation · structured JSON        │
│            Cached by content hash + prompt version              │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              Layer 5 – Point-in-Time Event Store (SQLite)       │
│   news_raw · news_cluster · news_signal · calendar_event        │
│             decision_audit · llm_call_log                       │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌──────────────────────────┴──────────────────────────────────────┐
│  Feature Builder            │      Risk Overlay Gates           │
│  (rolling, decayed)         │  (earnings, halts, macro, hard)   │
└────────────┬────────────────┴─────────────────┬────────────────-┘
             ▼                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│   Portfolio Manager (consumes NewsSignal + MarketContext)       │
│        → Risk Manager (applies veto + size adjustments)         │
│              → Trader (executes via Alpaca)                     │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│       Decision Audit + Outcome Tracking + Retraining Set        │
└─────────────────────────────────────────────────────────────────┘
```

### 2.1 Module/file layout

```
tradingbot/
  news/
    __init__.py
    sources/
      base.py                  # NewsSource ABC
      polygon_source.py
      alpaca_source.py
      edgar_source.py
      rss_source.py
    ingestion.py               # Normalization, dedupe, entity matching
    clustering.py              # Story clustering across sources
    deterministic_scorer.py    # Regex + reliability + event taxonomy
    llm_scorer.py              # Haiku/Sonnet calls, JSON validation, caching
    llm_providers/
      base.py                  # LLMScorer ABC
      claude_provider.py
      openai_provider.py       # Failover only
    feature_builder.py         # Rolling, decayed features for ML & PM
    signal.py                  # NewsSignal, NewsEvent, MarketContext dataclasses
    intelligence_service.py    # Public NewsIntelligenceService API
    storage/
      schema.sql
      repository.py
    config.py
  calendars/
    earnings.py
    macro.py
    halts.py
  decision/
    policy.py                  # PM/RM rule engine
    audit.py
config/
  news_intel.yaml
  prompts/
    materiality_v1.txt
    materiality_v2.txt
```

---

## 3. Data Model

All tables are SQLite. All timestamps are ISO 8601 UTC with `Z` suffix. All scores are floats in `[0.0, 1.0]` unless explicitly signed (direction_score is `[-1.0, 1.0]`).

### 3.1 `news_raw`
```sql
CREATE TABLE news_raw (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    provider_article_id TEXT,
    published_at TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    title TEXT NOT NULL,
    summary TEXT,
    body TEXT,
    url TEXT,
    primary_symbols TEXT NOT NULL,       -- JSON array
    raw_payload_hash TEXT NOT NULL,      -- SHA256 of normalized (title + summary)
    source_reliability REAL NOT NULL,
    UNIQUE(raw_payload_hash, source)
);
CREATE INDEX idx_news_raw_published ON news_raw(published_at);
CREATE INDEX idx_news_raw_symbols ON news_raw(primary_symbols);
```

### 3.2 `news_cluster`
```sql
CREATE TABLE news_cluster (
    cluster_id TEXT PRIMARY KEY,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    primary_symbol TEXT NOT NULL,
    related_symbols TEXT,                -- JSON array
    canonical_title TEXT NOT NULL,
    article_ids TEXT NOT NULL,           -- JSON array of news_raw.id
    source_count INTEGER NOT NULL,
    source_diversity REAL NOT NULL,
    duplicate_count INTEGER NOT NULL DEFAULT 1,
    cluster_age_minutes INTEGER NOT NULL
);
CREATE INDEX idx_cluster_symbol ON news_cluster(primary_symbol);
CREATE INDEX idx_cluster_first_seen ON news_cluster(first_seen_at);
```

### 3.3 `news_signal`
```sql
CREATE TABLE news_signal (
    signal_id TEXT PRIMARY KEY,
    cluster_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    evaluated_at TEXT NOT NULL,
    horizon TEXT NOT NULL,               -- 'intraday_4h' | 'swing_3d' | 'swing_7d'
    schema_version TEXT NOT NULL,
    prompt_version TEXT,
    scorer_tier TEXT NOT NULL,           -- 'rules' | 'haiku' | 'sonnet' | 'cached'
    model_provider TEXT,
    model_name TEXT,
    event_type TEXT NOT NULL,
    event_scope TEXT NOT NULL,           -- 'company' | 'sector' | 'market' | 'macro'
    direction_score REAL NOT NULL,       -- [-1.0, 1.0]
    materiality_score REAL NOT NULL,
    downside_risk_score REAL NOT NULL,
    upside_catalyst_score REAL NOT NULL,
    confidence REAL NOT NULL,
    novelty_score REAL NOT NULL,
    already_priced_in_score REAL NOT NULL,
    decay_half_life_minutes INTEGER NOT NULL,
    rationale TEXT,
    raw_llm_response TEXT,
    FOREIGN KEY (cluster_id) REFERENCES news_cluster(cluster_id)
);
CREATE INDEX idx_signal_symbol_eval ON news_signal(symbol, evaluated_at);
```

### 3.4 `calendar_event`
```sql
CREATE TABLE calendar_event (
    id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,            -- 'earnings' | 'cpi' | 'fomc' | 'nfp' | etc.
    symbol TEXT,                         -- null for macro events
    event_time TEXT NOT NULL,
    importance TEXT NOT NULL,            -- 'low' | 'medium' | 'high'
    source TEXT NOT NULL,
    confirmed INTEGER NOT NULL DEFAULT 0,
    payload TEXT,                        -- JSON: prior, consensus, etc.
    fetched_at TEXT NOT NULL
);
CREATE INDEX idx_calendar_time ON calendar_event(event_time);
CREATE INDEX idx_calendar_symbol ON calendar_event(symbol);
```

### 3.5 `decision_audit`
```sql
CREATE TABLE decision_audit (
    decision_id TEXT PRIMARY KEY,
    decided_at TEXT NOT NULL,
    symbol TEXT NOT NULL,
    strategy TEXT NOT NULL,              -- 'swing' | 'intraday'
    horizon TEXT NOT NULL,
    model_score REAL,
    news_signal_id TEXT,
    calendar_risk_payload TEXT,          -- JSON snapshot
    market_context_payload TEXT,         -- JSON snapshot
    final_decision TEXT NOT NULL,        -- 'enter'|'block'|'size_down'|'boost'|'exit_review'|'hold'
    size_multiplier REAL NOT NULL,
    decision_path TEXT NOT NULL,         -- which rule fired
    block_reason TEXT,
    realized_outcome_4h REAL,
    realized_outcome_1d REAL,
    realized_outcome_3d REAL,
    counterfactual_outcome TEXT
);
CREATE INDEX idx_audit_symbol_time ON decision_audit(symbol, decided_at);
```

### 3.6 `llm_call_log`
```sql
CREATE TABLE llm_call_log (
    call_id TEXT PRIMARY KEY,
    called_at TEXT NOT NULL,
    cluster_id TEXT,
    symbol TEXT,
    provider TEXT NOT NULL,
    model_name TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    latency_ms INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    cache_hit INTEGER NOT NULL DEFAULT 0,
    escalated_from TEXT,
    error TEXT
);
CREATE INDEX idx_llm_called_at ON llm_call_log(called_at);
```

---

## 4. Python Dataclasses (signal.py)

```python
from dataclasses import dataclass, field
from typing import Literal, Optional
from datetime import datetime

EventType = Literal[
    "earnings", "guidance", "analyst_action", "m_and_a",
    "legal_regulatory", "product", "management", "supply_chain",
    "macro", "geopolitical", "sector", "halt", "filing_8k", "other",
]

EventScope = Literal["company", "sector", "market", "macro"]
ScorerTier = Literal["rules", "haiku", "sonnet", "cached", "fallback"]
ActionPolicy = Literal[
    "ignore", "watch",
    "size_down_light",     # 0.75x
    "size_down_heavy",     # 0.50x
    "size_up_light",       # 1.15x — only when conviction is exceptional
    "block_entry",
    "exit_review",
    "manual_review",
]

@dataclass
class NewsEvent:
    cluster_id: str
    title: str
    published_at: datetime
    source: str
    source_count_in_cluster: int
    event_type: EventType
    materiality_score: float
    confidence: float
    rationale: str

@dataclass
class NewsSignal:
    symbol: str
    horizon: Literal["intraday_4h", "swing_3d", "swing_7d"]
    evaluated_at: datetime
    direction_score: float           # [-1.0, 1.0]
    materiality_score: float
    downside_risk_score: float
    upside_catalyst_score: float
    confidence: float
    novelty_score: float
    already_priced_in_score: float
    action_policy: ActionPolicy
    reason: str
    top_events: list
    scorer_tier: ScorerTier
    schema_version: str = "v1"
    stale: bool = False
    fallback_used: bool = False

@dataclass
class CalendarRisk:
    symbol: str
    earnings_today: bool
    earnings_tomorrow: bool
    earnings_proximity_days: Optional[int]
    holds_through_earnings: bool
    macro_event_today: bool
    macro_event_window_now: bool
    next_macro_event: Optional[str]
    next_macro_event_time: Optional[datetime]

@dataclass
class MarketContext:
    as_of: datetime
    regime: Literal["risk_on", "risk_off", "neutral", "macro_event_pending", "macro_event_window"]
    vix_level: float
    vix_regime: Literal["low", "normal", "elevated", "high"]
    spy_premarket_return: float
    macro_high_impact_today: bool
    sectors_under_stress: list
    schema_version: str = "v1"
```

---

## 5. Source Adapters

### 5.1 Source reliability weights (from `news_intel.yaml`)
```yaml
source_reliability:
  edgar: 1.00
  reuters: 0.95
  bloomberg: 0.95
  wsj: 0.90
  marketwatch: 0.80
  polygon: 0.75
  alpaca: 0.70
  benzinga: 0.70
  rss_other: 0.50
  stocktwits: 0.30
  social_media: 0.20
```

### 5.2 Source priority by phase
| Source | Phase | Cost | Notes |
|---|---|---|---|
| Polygon | Now (existing) | Paid | Keep, stop trusting article sentiment blindly |
| Alpaca News | Now (existing) | Free | Keep |
| Earnings calendar | Phase 58 | Free (Polygon or Finnhub) | **Highest priority** |
| Macro calendar | Phase 59 | Free (FRED + BLS) | FOMC, CPI, NFP, PPI |
| SEC EDGAR (8-K) | Phase 65 | Free | Official disclosures, high signal |
| Trading halts | Phase 65 | Free (NASDAQ RSS) | Hard gate, no LLM needed |
| Reuters/MarketWatch RSS | Phase 65 | Free | Source diversity |
| Benzinga | Phase 66+ | ~$10-25/mo | Fast tape, analyst actions |

---

## 6. Ingestion, Dedupe, and Clustering

### 6.1 Deduplication
- `raw_payload_hash = sha256(normalize(title + summary))` where normalize = lowercase, collapse whitespace, strip punctuation
- `UNIQUE(raw_payload_hash, source)` constraint in DB handles duplicates at insert time

### 6.2 Clustering (Phase 65 — defer until then)
Two articles in the same cluster if:
- Exact `raw_payload_hash` match (cross-source duplicate), OR
- Share primary symbol AND TF-IDF cosine similarity ≥ 0.65 AND published_at within 4 hours

For Phase 60: use simpler heuristic — same symbol + same hour + same source = same cluster.

---

## 7. Deterministic Pre-Scoring

### 7.1 Hard event regex matchers
```python
HARD_EVENT_PATTERNS = {
    "halt":             [r"\btrading halt(ed)?\b", r"\bhalted\b"],
    "filing_8k":        [r"\bForm 8[- ]K\b", r"\b8-K filed\b"],
    "earnings_release": [r"\breports? Q[1-4] (results|earnings)\b",
                         r"\bearnings (beat|miss)\b"],
    "ma_announcement":  [r"\bto acquire\b", r"\bagrees? to be acquired\b",
                         r"\bmerger agreement\b"],
    "fda_action":       [r"\bFDA (approves|rejects|clears)\b",
                         r"\bComplete Response Letter\b"],
}
```

### 7.2 Cheap discards (skip LLM scoring)
- All articles older than `max_article_age_hours` (24 for swing, 4 for intraday)
- Max source_reliability < 0.4
- Title length < 5 words AND no body
- Watch-list/recap article: `\b(top|best|worst) \d+ stocks\b`

### 7.3 Adversarial test cases (must not produce false positives)
- "tax loss harvesting"
- "no loss of life"
- "downgrade their estimate of the cost"
- Ticker "ALL", "IT", "ON" used as English words, not tickers

---

## 8. LLM Materiality Scorer

### 8.1 Tiering policy
```
ALL clusters that survive Section 7 → Haiku
  ↓
Escalate to Sonnet if (Phase 66+):
  - Haiku confidence < 0.60, OR
  - Haiku materiality_score >= 0.70, OR
  - event_type in {m_and_a, legal_regulatory, fda_action, halt}
```

### 8.2 Caching
- Cache key: `sha256(prompt_version + headline + summary + symbol)`
- TTL: 1 hour for fresh articles, 24 hours for articles > 4h old
- Cache hit: `cost_usd=0`, `cache_hit=1` in llm_call_log

### 8.3 Token budget
- Input cap: 3000 tokens per cluster
- Output cap: 400 tokens
- Daily budget cap (config): $5/day. When exceeded → deterministic-only mode.

### 8.4 Prompt template v1 (`config/prompts/materiality_v1.txt`)

```
You are a financial event interpretation system. Your role is to classify
whether a news event is material to a specific stock over a specific horizon.

You DO NOT recommend trades, sizes, or directions of trading. You classify
events. A separate deterministic system maps your classifications to actions.

Use only the information provided. Do not invent facts. If the article is
vague, lower the confidence score. If symbol relevance is unclear, set
relevant_to_symbol=false.

=== INPUT ===
Symbol: {symbol}
Company: {company_name}
Sector: {sector}
Horizon: {horizon}

Article cluster ({source_count} sources):
- Canonical title: {canonical_title}
- Summary: {summary}
- Published at: {published_at}
- Cluster age: {cluster_age_minutes} minutes

Current price context for {symbol}:
- Premarket return today: {premarket_return}%
- Intraday return so far: {intraday_return}%
- Return since cluster first seen: {return_since_first_seen}%
- Relative volume vs 20d avg: {relative_volume}x

Market context:
- SPY return today: {spy_return}%
- VIX level: {vix_level} ({vix_regime})
- High-impact macro event today: {macro_today}

=== OUTPUT ===
Return STRICT JSON ONLY. No prose before or after.

{
  "relevant_to_symbol": <bool>,
  "event_type": <earnings|guidance|analyst_action|m_and_a|legal_regulatory|
                 product|management|supply_chain|macro|geopolitical|
                 sector|halt|filing_8k|other>,
  "event_scope": <company|sector|market|macro>,
  "direction_score": <float [-1.0, 1.0], negative=bearish for {symbol}>,
  "materiality_score": <float [0.0, 1.0]>,
  "downside_risk_score": <float [0.0, 1.0]>,
  "upside_catalyst_score": <float [0.0, 1.0]>,
  "confidence": <float [0.0, 1.0]>,
  "novelty_score": <float [0.0, 1.0], 1.0=first time markets hearing this>,
  "already_priced_in_score": <float [0.0, 1.0]>,
  "rationale": "<one sentence, max 150 chars>"
}

already_priced_in_score guidance:
- Price moved sharply in news direction before article: score 0.6-0.9
- Price has not yet reacted: score 0.0-0.3
- Unclear: score 0.5

novelty_score guidance:
- Story reported repeatedly over prior days: score 0.2-0.4
- Fresh disclosure or first-time event: score 0.7-1.0
```

### 8.5 Decay function (applied at read time)
```python
def decayed_materiality(stored: float, age_minutes: int, half_life: int) -> float:
    return stored * (0.5 ** (age_minutes / half_life))
```
Default half-lives:
- `intraday_4h`: 60 minutes
- `swing_3d`: 360 minutes (6 hours)
- `swing_7d`: 1440 minutes (1 day)
- Slow-decay events (earnings, halt, m_and_a): half-life × 3

---

## 9. Calendar Layer

### 9.1 Earnings rules (Phase 58)
```yaml
earnings_rules:
  swing:
    block_new_entry_days_before: 2
    block_new_entry_days_after: 1
    force_exit_review_if_holding_through: true
  intraday:
    block_entry_if_earnings_today: true
    block_entry_if_earnings_tomorrow_amclose: true
```

### 9.2 Macro event rules (Phase 59)
```yaml
macro_window:
  pre_event_pause_minutes: 15
  post_event_pause_minutes: 15
  fomc_press_conference_pause_minutes: 60
  high_vix_extension_factor: 2.0
  vix_threshold_for_extension: 25.0

macro_high_impact_events:
  - cpi
  - ppi
  - fomc_decision
  - fomc_press_conference
  - nonfarm_payrolls
  - unemployment_rate
  - retail_sales
  - core_pce
```

---

## 10. Feature Builder (Phase 64 — defer until 60 days of history)

```python
@dataclass
class SymbolNewsFeatures:
    symbol: str
    as_of: datetime
    materiality_decayed_30m: float
    materiality_decayed_4h: float
    materiality_decayed_1d: float
    materiality_decayed_3d: float
    materiality_decayed_7d: float
    direction_decayed_4h: float
    direction_decayed_3d: float
    downside_risk_max_4h: float
    downside_risk_max_3d: float
    upside_catalyst_max_3d: float
    article_count_4h: int
    cluster_count_4h: int
    novelty_max_3d: float
    confidence_weighted_avg_3d: float
    earnings_proximity_days: Optional[int]
    is_earnings_today: bool
    is_holding_through_earnings: bool
    has_active_halt: bool
    has_active_8k: bool
    had_news_in_window: bool     # distinguish "no news" from "low-materiality news"
    schema_version: str = "v1"
```

---

## 11. Decision Policy (Phase 63 — defer until audit data exists)

### 11.1 PM scoring overlay
```python
news_overlay = (
    cfg.upside_weight     * signal.upside_catalyst_score   # default +0.10
  - cfg.downside_weight   * signal.downside_risk_score     # default -0.15
  - cfg.earnings_weight   * earnings_risk_value            # default -0.10
  - cfg.macro_weight      * macro_risk_value               # default -0.10
) * signal.confidence * (1.0 - 0.5 * signal.already_priced_in_score)

final_score = model_score + news_overlay
```

### 11.2 RM hard veto rules
```python
if calendar_risk.earnings_today: veto("earnings_today")
if calendar_risk.holds_through_earnings: veto("would_hold_through_earnings")
if features.has_active_halt: veto("active_halt")
if market_context.regime == "macro_event_window": veto("macro_event_window")
if (signal.downside_risk_score >= 0.75
    and signal.confidence >= 0.70
    and signal.already_priced_in_score < 0.50):
    veto("material_downside_news")
```

### 11.3 RM size adjustments
```python
size_mult = 1.0
if signal.downside_risk_score >= 0.50 and signal.confidence >= 0.60:
    size_mult *= 0.50
if calendar_risk.earnings_proximity_days is not None and calendar_risk.earnings_proximity_days <= 5:
    size_mult *= 0.75
if market_context.vix_regime == "elevated":
    size_mult *= 0.85
elif market_context.vix_regime == "high":
    size_mult *= 0.65
if (signal.upside_catalyst_score >= 0.70 and signal.confidence >= 0.70
        and final_score >= base_threshold + 0.05):
    size_mult = min(size_mult * 1.15, 1.15)
size_mult = max(0.0, min(1.25, size_mult))
```

---

## 12. Operational Workflow

### 12.1 Pre-market (08:00 ET) — Phase 62
1. Fetch overnight news for full universe
2. Refresh earnings calendar for next 10 trading days
3. Fetch macro events for today
4. Cluster, dedupe, deterministic pre-score
5. LLM-score (Haiku) all clusters that pass filters
6. Generate `MarketContext` snapshot
7. Build pre-market briefing (seeds existing Haiku audit task #3)

### 12.2 At PM scan (09:45 ET swing, 60/90/120 min intraday) — Phase 60
1. PM requests cached `NewsSignal` per candidate symbol
2. PM applies scoring overlay (Section 11.1)
3. RM applies veto + size logic (Section 11.2-11.3)
4. Trader executes; `decision_audit` written (Phase 61)

### 12.3 Intraday monitor (every 5 min) — Phase 53 (already built)
- Current `NewsMonitor` handles this
- Upgrade to emit `NewsSignal` in Phase 60

### 12.4 EOD (16:30 ET) — Phase 61
1. Fill `decision_audit.realized_outcome_*` for elapsed horizons
2. Roll daily metrics

---

## 13. Failure Modes and Fallbacks

| Failure | Fallback |
|---|---|
| Polygon API down | Use cached news, mark `stale=true` |
| Alpaca news down | Same |
| Earnings calendar fetch failed | Use yesterday's snapshot; if >24h stale, treat unknown as risk and block swing entries |
| LLM primary down | Fall back to deterministic-only scoring; confidence floored to 0.4 |
| LLM returns unparseable JSON | Deterministic floor; log raw response |
| Daily LLM budget exceeded | Deterministic only until reset |
| SQLite write failure | Retry 3× then fail open (no signal, PM proceeds without news) |

**Critical invariant:** A degraded news layer never blocks trading via false positives.

---

## 14. Key Metrics

### 14.1 The honest metric (Phase 63+)
```
news_layer_alpha = realized_pnl_with_news_overlay - realized_pnl_without_news_overlay
```
Computed by replaying `decision_audit` without the news overlay. If negative over 30+ days, the news layer is costing money.

### 14.2 Daily metrics
- Articles ingested per source
- LLM cost USD (Haiku / Sonnet / cached)
- LLM avg latency + error rate
- Decisions: enter / block / size_down / boost / exit_review counts
- False-positive block rate (blocked, but realized return was positive)

---

## 15. Configuration Reference (`config/news_intel.yaml`)

```yaml
schema_version: "v1"

source_reliability:
  edgar: 1.00
  reuters: 0.95
  polygon: 0.75
  alpaca: 0.70

clustering:
  window_hours: 6
  title_similarity_threshold: 0.65
  publish_proximity_hours: 4

deterministic_filters:
  max_article_age_hours_swing: 24
  max_article_age_hours_intraday: 4
  min_source_reliability: 0.40

llm:
  primary_provider: claude
  primary_model: claude-haiku-4-5-20251001
  escalation_model: claude-sonnet-4-6
  daily_budget_usd: 5.00
  max_input_tokens_per_article: 3000
  max_output_tokens: 400
  cache_ttl_hours_old: 24
  cache_ttl_hours_fresh: 1

escalation:
  haiku_min_confidence: 0.60
  haiku_materiality_threshold: 0.70
  always_escalate_event_types: [m_and_a, legal_regulatory, fda_action, halt]

decay_minutes:
  intraday_4h: 60
  swing_3d: 360
  swing_7d: 1440
  slow_decay_event_types: [earnings, halt, m_and_a, fda_action]
  slow_decay_multiplier: 3.0

policy_weights:
  upside: 0.10
  downside: 0.15
  earnings: 0.10
  macro: 0.10
  already_priced_in_attenuation: 0.50

veto_thresholds:
  downside_risk: 0.75
  downside_confidence: 0.70
  already_priced_in_max: 0.50

size_thresholds:
  size_down_heavy_downside: 0.50
  size_down_heavy_confidence: 0.60
  size_down_heavy_multiplier: 0.50
  earnings_proximity_days: 5
  earnings_proximity_multiplier: 0.75
  vix_elevated_multiplier: 0.85
  vix_high_multiplier: 0.65
  upside_boost_threshold: 0.70
  upside_boost_confidence: 0.70
  upside_boost_multiplier: 1.15
  size_floor: 0.0
  size_ceiling: 1.25

earnings:
  swing_block_days_before: 2
  swing_block_days_after: 1
  swing_force_exit_review_through_earnings: true
  intraday_block_if_today: true
  intraday_block_if_tomorrow_amclose: true

macro:
  pre_event_pause_minutes: 15
  post_event_pause_minutes: 15
  fomc_press_pause_minutes: 60
  high_vix_extension_factor: 2.0
  vix_threshold_for_extension: 25.0

vix_regime:
  low_max: 14
  normal_max: 20
  elevated_max: 28

monitoring:
  alert_llm_cost_pct: 0.80
  alert_llm_error_rate_1h: 0.05
  alert_source_down_minutes: 60
  alert_manual_review_per_day: 10
```

---

## 16. Testing Strategy

### 16.1 Unit tests
- `test_ingestion.py`: dedup logic on golden fixtures
- `test_clustering.py`: 5 sources publishing same story → 1 cluster
- `test_deterministic_scorer.py`: regex matchers, adversarial false-positive cases
- `test_llm_scorer.py`: mocked LLM provider; validation, fallback all tested
- `test_feature_builder.py`: decay function, silent stocks distinction
- `test_policy.py`: every decision rule has positive and negative test case

### 16.2 Adversarial tests
- Article with prompt injection in title → LLM treats as untrusted text, JSON contract holds
- Ticker collision: "ALL", "IT", "ON" as English words
- Article published 6 hours ago appearing for first time (source latency) → decay logic unaffected
- Same title, different published_at by 4 hours → not a duplicate
