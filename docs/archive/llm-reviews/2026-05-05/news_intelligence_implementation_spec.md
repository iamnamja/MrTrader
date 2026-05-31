# News Intelligence Layer — Consolidated Implementation Spec

**Project:** TradingBot — Automated Swing + Intraday Equity Trading System
**Component:** News Ingestion, Event Interpretation, and Decision Integration Layer
**Document type:** Engineering specification for implementation by Claude Code
**Author:** Synthesis of two external design reviews + quant-systems best-practices overlay
**Date:** April 27, 2026
**Status:** Approved for implementation — to be reviewed by repo-resident Claude Code before final commit

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

```text
┌─────────────────────────────────────────────────────────────────┐
│                  Source Adapters (pluggable)                    │
│   Polygon · Alpaca · SEC EDGAR · RSS · Earnings · Macro · Halts │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              Layer 1 — Ingestion & Normalization                │
│       Common schema, content-hash dedupe, entity match          │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              Layer 2 — Clustering                               │
│    Group syndicated/repeat stories into a single cluster        │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              Layer 3 — Deterministic Pre-Scoring                │
│  Source reliability, regex event detectors, cheap discardable   │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              Layer 4 — LLM Materiality Scorer                   │
│      Haiku primary · Sonnet escalation · structured JSON        │
│            Cached by content hash + prompt version              │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              Layer 5 — Point-in-Time Event Store (SQLite)       │
│   news_raw · news_cluster · news_signal · calendar_event        │
│             decision_audit · llm_call_log                       │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌──────────────────────────────┴──────────────────────────────────┐
│  Feature Builder            │      Risk Overlay Gates           │
│  (rolling, decayed)         │  (earnings, halts, macro, hard)   │
└────────────┬────────────────┴──────────────────┬────────────────┘
             ▼                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│   Portfolio Manager (consumes NewsSignal + MarketContext)       │
│        → Risk Manager (applies veto + size adjustments)         │
│              → Trader (executes via Alpaca)                     │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│       Decision Audit + Outcome Tracking + Retraining Set        │
└─────────────────────────────────────────────────────────────────┘
```

### 2.1 Module/file layout (under existing `src/` or `tradingbot/`)

```text
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
      base.py                  # LLMScorer ABC (Claude/GPT/Llama interchangeable)
      claude_provider.py
      openai_provider.py
      ollama_provider.py       # Local Llama fallback
    feature_builder.py         # Rolling, decayed features for ML & PM
    signal.py                  # NewsSignal, NewsEvent, MarketContext dataclasses
    intelligence_service.py    # Public NewsIntelligenceService API
    storage/
      schema.sql
      repository.py            # All DB read/write
    config.py                  # Loads thresholds/weights from YAML
  calendars/
    earnings.py
    macro.py
    halts.py
  decision/
    policy.py                  # The PM/RM rule engine that consumes NewsSignal
    audit.py                   # Decision logging
config/
  news_intel.yaml              # Thresholds, weights, decay rates, source weights
  prompts/
    materiality_v1.txt
    materiality_v2.txt
tests/
  news/
    test_ingestion.py
    test_clustering.py
    test_llm_scorer.py
    test_feature_builder.py
    test_policy.py
    fixtures/
      golden_articles.json
      golden_signals.json
```

---

## 3. Data Model

All tables are SQLite. All timestamps are ISO 8601 UTC with `Z` suffix (e.g. `2026-04-27T13:30:00Z`). All scores are floats in `[0.0, 1.0]` unless explicitly signed (direction_score is `[-1.0, 1.0]`).

### 3.1 `news_raw`
```sql
CREATE TABLE news_raw (
    id TEXT PRIMARY KEY,                  -- UUID4 generated at ingest
    source TEXT NOT NULL,                 -- 'polygon', 'alpaca', 'edgar', 'rss:reuters'
    provider_article_id TEXT,             -- vendor's ID, nullable
    published_at TEXT NOT NULL,           -- vendor-reported timestamp
    ingested_at TEXT NOT NULL,            -- when WE first saw it
    title TEXT NOT NULL,
    summary TEXT,
    body TEXT,                            -- full body if available, else null
    url TEXT,
    primary_symbols TEXT NOT NULL,        -- JSON array of tickers
    raw_payload_hash TEXT NOT NULL,       -- SHA256 of (title + summary) normalized
    source_reliability REAL NOT NULL,     -- 0.0-1.0 from config
    UNIQUE(raw_payload_hash, source)
);
CREATE INDEX idx_news_raw_published ON news_raw(published_at);
CREATE INDEX idx_news_raw_symbols ON news_raw(primary_symbols);
```

### 3.2 `news_cluster`
```sql
CREATE TABLE news_cluster (
    cluster_id TEXT PRIMARY KEY,          -- UUID4
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    primary_symbol TEXT NOT NULL,
    related_symbols TEXT,                 -- JSON array
    canonical_title TEXT NOT NULL,
    article_ids TEXT NOT NULL,            -- JSON array of news_raw.id
    source_count INTEGER NOT NULL,        -- distinct sources
    source_diversity REAL NOT NULL,       -- normalized 0-1
    duplicate_count INTEGER NOT NULL DEFAULT 1,
    cluster_age_minutes INTEGER NOT NULL  -- last_seen - first_seen
);
CREATE INDEX idx_cluster_symbol ON news_cluster(primary_symbol);
CREATE INDEX idx_cluster_first_seen ON news_cluster(first_seen_at);
```

### 3.3 `news_signal`
```sql
CREATE TABLE news_signal (
    signal_id TEXT PRIMARY KEY,           -- UUID4
    cluster_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    evaluated_at TEXT NOT NULL,
    horizon TEXT NOT NULL,                -- 'intraday_4h' | 'swing_3d' | 'swing_7d'
    schema_version TEXT NOT NULL,         -- e.g. 'v1'
    prompt_version TEXT,                  -- null if deterministic-only
    scorer_tier TEXT NOT NULL,            -- 'rules' | 'haiku' | 'sonnet' | 'cached'
    model_provider TEXT,
    model_name TEXT,
    event_type TEXT NOT NULL,
    event_scope TEXT NOT NULL,            -- 'company' | 'sector' | 'market' | 'macro'
    direction_score REAL NOT NULL,        -- [-1.0, 1.0]
    materiality_score REAL NOT NULL,      -- [0.0, 1.0]
    downside_risk_score REAL NOT NULL,
    upside_catalyst_score REAL NOT NULL,
    confidence REAL NOT NULL,
    novelty_score REAL NOT NULL,
    already_priced_in_score REAL NOT NULL,
    decay_half_life_minutes INTEGER NOT NULL,
    rationale TEXT,
    raw_llm_response TEXT,                -- JSON, full response for audit
    FOREIGN KEY (cluster_id) REFERENCES news_cluster(cluster_id)
);
CREATE INDEX idx_signal_symbol_eval ON news_signal(symbol, evaluated_at);
CREATE INDEX idx_signal_cluster ON news_signal(cluster_id);
```

### 3.4 `calendar_event`
```sql
CREATE TABLE calendar_event (
    id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,             -- 'earnings' | 'cpi' | 'fomc' | 'nfp' | 'ppi' | 'unemp' | 'retail_sales'
    symbol TEXT,                          -- null for macro events
    event_time TEXT NOT NULL,             -- scheduled UTC time
    importance TEXT NOT NULL,             -- 'low' | 'medium' | 'high'
    source TEXT NOT NULL,
    confirmed BOOLEAN NOT NULL DEFAULT 0,
    payload TEXT                          -- JSON: prior, consensus, etc.
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
    horizon TEXT NOT NULL,
    model_score REAL,                     -- XGBoost output
    news_signal_id TEXT,                  -- nullable
    calendar_risk_payload TEXT,           -- JSON
    market_context_payload TEXT,          -- JSON
    final_decision TEXT NOT NULL,         -- 'enter' | 'block' | 'size_down' | 'boost' | 'exit_review' | 'hold'
    size_multiplier REAL NOT NULL,
    decision_path TEXT NOT NULL,          -- which rule fired
    expected_reason TEXT,
    -- Outcome tracking (filled in later by post-trade evaluator):
    realized_outcome_15m REAL,
    realized_outcome_4h REAL,
    realized_outcome_1d REAL,
    realized_outcome_3d REAL,
    counterfactual_outcome TEXT           -- 'blocked but would have profited?' etc.
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
    cache_hit BOOLEAN NOT NULL DEFAULT 0,
    escalated_from TEXT,                  -- 'haiku' if this is a sonnet escalation
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
    "ignore",
    "watch",
    "size_down_light",     # 0.75x
    "size_down_heavy",     # 0.50x
    "size_up_light",       # 1.15x — only when conviction is exceptional
    "block_entry",
    "exit_review",
    "manual_review",
]

@dataclass
class NewsEvent:
    """A single article/cluster contributing to a NewsSignal."""
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
    """The structured output PM/RM consume for a single symbol+horizon."""
    symbol: str
    horizon: Literal["intraday_4h", "swing_3d", "swing_7d"]
    evaluated_at: datetime

    # Continuous scores (decayed when read for current time)
    direction_score: float           # [-1.0, 1.0]
    materiality_score: float         # [0.0, 1.0]
    downside_risk_score: float
    upside_catalyst_score: float
    confidence: float
    novelty_score: float
    already_priced_in_score: float

    # Discrete recommendation (derived, advisory only)
    action_policy: ActionPolicy
    reason: str

    # Provenance
    top_events: list[NewsEvent]
    scorer_tier: ScorerTier
    schema_version: str = "v1"
    stale: bool = False
    fallback_used: bool = False

@dataclass
class CalendarRisk:
    symbol: str
    earnings_today: bool
    earnings_tomorrow: bool
    earnings_proximity_days: Optional[int]    # None if no upcoming earnings within window
    holds_through_earnings: bool              # True if planned hold crosses the date
    macro_event_today: bool
    macro_event_window_now: bool              # within ±15 min of release
    next_macro_event: Optional[str]
    next_macro_event_time: Optional[datetime]

@dataclass
class MarketContext:
    """Market-level state used by PM/RM in addition to per-symbol signals."""
    as_of: datetime
    regime: Literal["risk_on", "risk_off", "neutral", "macro_event_pending", "macro_event_window"]
    vix_level: float
    vix_regime: Literal["low", "normal", "elevated", "high"]
    spy_premarket_return: float
    breadth_indicator: Optional[float]        # advance-decline or similar
    macro_high_impact_today: bool
    sectors_under_stress: list[str]           # tickers/sectors with cluster activity
    schema_version: str = "v1"
```

---

## 5. Source Adapters

### 5.1 Common interface
```python
from abc import ABC, abstractmethod
from typing import Iterable
from datetime import datetime

class NewsSource(ABC):
    name: str
    reliability: float                         # 0.0-1.0 from config

    @abstractmethod
    def fetch_since(self, since: datetime, symbols: list[str] | None = None
                    ) -> Iterable[dict]:
        """Yield raw articles. Each must include at minimum:
        title, summary, published_at, url, symbols (list), provider_article_id"""
```

### 5.2 Required sources (Phase 1-3)

| Source | Tier | Cost | Phase | Notes |
|---|---|---|---|---|
| Polygon | Existing | Paid | 1 | Keep, but stop trusting article-level sentiment |
| Alpaca News | Existing | Free | 1 | Keep |
| SEC EDGAR (8-K filings) | New | Free | 1 | Highest single-source signal — official disclosures |
| Earnings calendar | New | Free (Polygon) or Finnhub free tier | 1 | **Highest priority Phase 1 deliverable** |
| Macro calendar | New | Free (FRED + ForexFactory or BLS) | 1 | CPI, FOMC, NFP, PPI, unemp, retail sales |
| Trading halts feed | New | Free (NASDAQ Trader RSS) | 2 | Separate ingestion path — hard gate |
| Reuters/MarketWatch RSS | New | Free | 2 | Source diversity for clustering |
| Benzinga | Optional | ~$10-25/mo | 3+ | Fast tape, analyst actions |
| StockTwits | Optional | Free tier | 4+ | Crowd attention, low priority |
| Reddit/Twitter | Skip | Free | — | Defer indefinitely for top-50 SPX universe |

### 5.3 Source reliability weights (initial values, in `news_intel.yaml`)
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
These weights feed into cluster confidence calculations. They MUST be tunable without code changes.

---

## 6. Ingestion, Dedupe, and Clustering

### 6.1 Ingestion (`ingestion.py`)
For each source adapter:
1. Fetch raw articles since last cursor
2. Normalize to common schema
3. Compute `raw_payload_hash = sha256(normalize(title + summary))` where normalize = lowercase, collapse whitespace, strip punctuation
4. Resolve tickers (entity match): vendor-supplied tickers MUST be validated against `company_name` to avoid false matches on common-word tickers (e.g., "ALL", "IT", "ON")
5. Insert into `news_raw`; if `(raw_payload_hash, source)` already exists, skip

### 6.2 Clustering (`clustering.py`)
Cluster articles within a 6-hour rolling window:
1. Two articles are in the same cluster if EITHER:
   - exact `raw_payload_hash` match (cross-source duplicate), OR
   - share primary symbol AND TF-IDF cosine similarity of titles ≥ 0.65 AND published_at within 4 hours of each other
2. Update `news_cluster.source_count`, `source_diversity = unique_sources / max(unique_sources_seen_today, 1)`
3. The cluster's "canonical title" is the most-recent title from the highest-reliability source

### 6.3 Why clustering matters
- **Confidence boost**: a cluster with 5 distinct sources in 30 min is a real event, not noise
- **Cost control**: we LLM-score the cluster once, not 5 times
- **Decision logic input**: source_count and source_diversity become inputs to confidence

---

## 7. Deterministic Pre-Scoring

Before any LLM call, every cluster goes through deterministic scoring. This:
- Catches obvious cases cheaply
- Rejects junk (recipe articles, unrelated companies sharing a ticker)
- Provides the LLM with a richer context (when escalated)

### 7.1 Hard event regex matchers
Each rule maps to a confident event_type and skips the LLM entirely:

```python
HARD_EVENT_PATTERNS = {
    "halt":          [r"\btrading halt(ed)?\b", r"\bhalted\b"],
    "filing_8k":     [r"\bForm 8[- ]K\b", r"\b8-K filed\b"],
    "earnings_release": [r"\breports? Q[1-4] (results|earnings)\b",
                         r"\bearnings (beat|miss)\b"],
    "ma_announcement":  [r"\bto acquire\b", r"\bagrees? to be acquired\b",
                         r"\bmerger agreement\b"],
    "fda_action":    [r"\bFDA (approves|rejects|clears)\b",
                      r"\bComplete Response Letter\b"],
}
```
A regex match sets `event_type` and `materiality_score >= 0.85`, but `direction_score` and `confidence` still need either rules + price context OR an LLM call (the regex doesn't know if it's a beat or a miss).

### 7.2 Cheap discards
A cluster is discarded (not LLM-scored) if ANY of:
- All articles older than `max_article_age_hours` (default 24 for swing, 4 for intraday)
- Maximum source_reliability across articles < 0.4
- Title length < 5 words AND no body
- Symbol mention is in a watch-list/recap article (regex: `\b(top|best|worst) \d+ stocks\b`)

### 7.3 Fast-path direction inference
For analyst actions, regex can infer direction reliably:
```python
ANALYST_DIRECTION_PATTERNS = {
    +0.4: [r"\bupgrade", r"\braise(s|d) (price target|PT)", r"\bbuy rating"],
    -0.4: [r"\bdowngrade", r"\bcut (price target|PT)", r"\bsell rating"],
}
```
These provide a cheap deterministic floor. The LLM still gets called for materiality and confidence, but cost can be reduced by sometimes skipping LLM entirely on small analyst pieces (configurable).

---

## 8. LLM Materiality Scorer

### 8.1 Provider abstraction
```python
class LLMScorer(ABC):
    @abstractmethod
    def score_cluster(
        self,
        cluster: NewsCluster,
        symbol: str,
        market_context: MarketContext,
        price_context: PriceContext,
        prompt_version: str,
    ) -> dict:  # parsed structured output
        ...
```
Implementations: `ClaudeProvider` (Haiku/Sonnet), `OpenAIProvider`, `OllamaProvider` (local Llama for offline fallback).

### 8.2 Tiering policy
```text
ALL clusters that survive Section 7 → Haiku
  ↓
Escalate to Sonnet if:
  - Haiku returned confidence < 0.60, OR
  - Haiku returned materiality_score >= 0.70 (high-stakes — get a second opinion), OR
  - Haiku returned event_type in {m_and_a, legal_regulatory, fda_action, halt}
```
Self-consistency check (Phase 2+):
- For any signal that triggers `block_entry` or `exit_review`, run Sonnet a second time with article order reversed in the prompt
- If event_type or direction sign disagrees, downgrade confidence by 0.20 and route to `manual_review`

### 8.3 Caching
- Cache key: `sha256(prompt_version + cluster_canonical_title + cluster_summary + symbol + horizon)`
- Cache value: full LLM response JSON
- TTL: 24 hours for clusters older than 4h, 1 hour for fresher clusters (allows re-evaluation as price action develops)
- Cache hit logged in `llm_call_log` with `cache_hit=1, cost_usd=0`

### 8.4 Token budget
- Cap input at 3000 tokens per article (truncate body, keep title + summary + first N paragraphs)
- Cap output at 400 tokens
- Daily LLM spend cap (config): default $5/day. Hard stop on Haiku, then on Sonnet escalation. Fall back to deterministic-only when exceeded; alert.

### 8.5 Prompt template (`prompts/materiality_v1.txt`)

```text
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
Horizon: {horizon}                       # "intraday_4h" or "swing_3d" or "swing_7d"

Article cluster (deduplicated across {source_count} sources):
- Canonical title: {canonical_title}
- Summary: {summary}
- Published at: {published_at}
- Cluster age: {cluster_age_minutes} minutes
- Sources: {source_list}

Current price context for {symbol}:
- Premarket return today: {premarket_return}%
- Intraday return so far: {intraday_return}%
- Return since cluster first seen: {return_since_first_seen}%
- Relative volume vs 20d avg: {relative_volume}x
- 5-day return: {return_5d}%

Market context:
- SPY return today: {spy_return}%
- Sector ETF return today: {sector_return}%
- VIX level: {vix_level} (regime: {vix_regime})
- High-impact macro event today: {macro_today}

=== OUTPUT ===
Return STRICT JSON ONLY. No prose before or after. Schema:

{
  "relevant_to_symbol": <bool>,
  "event_type": <one of: earnings, guidance, analyst_action, m_and_a,
                 legal_regulatory, product, management, supply_chain,
                 macro, geopolitical, sector, halt, filing_8k, other>,
  "event_scope": <one of: company, sector, market, macro>,
  "direction_score": <float in [-1.0, 1.0], negative = bearish for {symbol}>,
  "materiality_score": <float in [0.0, 1.0], how much should this move price>,
  "downside_risk_score": <float in [0.0, 1.0]>,
  "upside_catalyst_score": <float in [0.0, 1.0]>,
  "confidence": <float in [0.0, 1.0], your confidence in this classification>,
  "novelty_score": <float in [0.0, 1.0], 1.0 = first time markets are hearing this>,
  "already_priced_in_score": <float in [0.0, 1.0], based on price context above>,
  "rationale": "<one sentence, max 200 chars>"
}

Guidance for already_priced_in_score:
- If price has moved sharply in the direction implied by the news before
  this article cluster appeared, score higher (0.6-0.9).
- If price has not yet reacted, score lower (0.0-0.3).
- If unclear, score 0.5.

Guidance for novelty_score:
- If the cluster references events that have been reported repeatedly over
  prior days, score lower (0.2-0.4).
- If this is a fresh disclosure or first-time event, score higher (0.7-1.0).

DO NOT include any text outside the JSON object.
```

### 8.6 Output validation
1. Parse JSON. On parse failure: log error, fall back to deterministic-only signal with `confidence=0.3`, set `fallback_used=true`
2. Validate types and ranges. On validation failure: same fallback
3. Persist `raw_llm_response` to `news_signal.raw_llm_response` for audit even when validation succeeds

### 8.7 Decay function applied at read time
When the policy engine reads a `NewsSignal`, scores are decayed based on age:

```python
def decayed_materiality(stored: float, age_minutes: int, half_life: int) -> float:
    return stored * (0.5 ** (age_minutes / half_life))
```
Default half-lives:
- intraday_4h horizon: 60 minutes
- swing_3d horizon: 360 minutes (6 hours)
- swing_7d horizon: 1440 minutes (1 day)
For event_type=earnings, halt, m_and_a: half-life × 3 (these decay slower).

---

## 9. Calendar Layer

### 9.1 Earnings calendar (Phase 1, **highest priority**)
- Daily fetch from primary source (Polygon or Finnhub free) at 04:00 ET
- Store all confirmed earnings dates for universe (top 50 SPX + open positions)
- Update `CalendarRisk` per symbol on each PM scan

Initial rules (configurable):
```yaml
earnings_rules:
  swing:
    block_new_entry_days_before: 2        # no new entries 2 trading days before
    block_new_entry_days_after: 1         # no new entries day after (volatility)
    force_exit_review_if_holding_through: true
  intraday:
    block_entry_if_earnings_today: true
    block_entry_if_earnings_tomorrow_amclose: true   # don't hold overnight into earnings
```

### 9.2 Macro calendar (Phase 1)
High-impact events:
```yaml
macro_high_impact:
  - cpi
  - ppi
  - fomc_decision
  - fomc_press_conference
  - nonfarm_payrolls
  - unemployment_rate
  - retail_sales
  - core_pce
```
Window rules:
```yaml
macro_window:
  pre_event_pause_minutes: 15
  post_event_pause_minutes: 15
  fomc_press_conference_pause_minutes: 60     # entire press conference window
  high_vix_extension_factor: 2.0              # if VIX > 25, double the pause windows
```

### 9.3 Halts feed (Phase 2)
- Poll NASDAQ Trader halt RSS every 60 seconds during market hours
- On any halt for a held position: immediate `exit_review` flag (NOT auto-exit; PM logs decision)
- On any halt for a watchlist symbol: hard `block_entry` until halt resolves AND post-halt price discovery completes (10 min minimum)

---

## 10. Feature Builder

### 10.1 Per-symbol features (used by PM and, in Phase 4, the model)
At each scan window, compute for every symbol in scope:

```python
@dataclass
class SymbolNewsFeatures:
    symbol: str
    as_of: datetime

    # Decayed materiality-weighted aggregates
    materiality_decayed_30m: float
    materiality_decayed_4h: float
    materiality_decayed_1d: float
    materiality_decayed_3d: float
    materiality_decayed_7d: float

    direction_decayed_4h: float       # weighted by materiality * confidence
    direction_decayed_3d: float

    downside_risk_max_4h: float       # max single-cluster downside in window
    downside_risk_max_3d: float
    upside_catalyst_max_3d: float

    article_count_4h: int
    cluster_count_4h: int
    source_diversity_4h: float

    novelty_max_3d: float
    confidence_weighted_avg_3d: float

    earnings_proximity_days: Optional[int]
    is_earnings_today: bool
    is_holding_through_earnings: bool

    has_active_halt: bool
    has_active_8k: bool

    schema_version: str = "v1"
```

### 10.2 Storage
Features are computed on demand (not stored) in Phase 1-3. In Phase 4 (when feeding the model), they MUST be backfilled across history using the point-in-time event store and stored in `symbol_news_features` for training.

### 10.3 Silent stocks
A symbol with NO news clusters in the look-back window is NOT the same as a symbol with neutral-scored news. Distinguish:
- `materiality_decayed_*` = 0.0 with `cluster_count_* > 0` → news was scored as low materiality
- `materiality_decayed_*` = 0.0 with `cluster_count_* == 0` → no news at all
The model and PM logic must treat these differently. Add a boolean feature `had_news_in_window`.

---

## 11. Decision Policy (PM/RM Rule Engine)

This is where the LLM's interpretation becomes a trading decision. All thresholds in `news_intel.yaml`.

### 11.1 PM scoring overlay
The model continues to produce its base score. PM applies an additive overlay:

```python
news_overlay = (
    cfg.upside_weight     * signal.upside_catalyst_score   # +0.10 default
  - cfg.downside_weight   * signal.downside_risk_score     # -0.15 default
  - cfg.earnings_weight   * earnings_risk_value            # -0.10 default
  - cfg.macro_weight      * macro_risk_value               # -0.10 default
)

# Confidence-scaled: low-confidence news has weaker effect
news_overlay *= signal.confidence

# Already-priced-in attenuation: if it's priced in, don't double-count
news_overlay *= (1.0 - 0.5 * signal.already_priced_in_score)

final_score = model_score + news_overlay
```

### 11.2 RM hard veto rules (in priority order)
```python
# 1. Hard discrete events
if calendar_risk.earnings_today: veto("earnings_today")
if calendar_risk.holds_through_earnings: veto("would_hold_through_earnings")
if features.has_active_halt: veto("active_halt")
if market_context.regime == "macro_event_window": veto("macro_event_window")

# 2. High-confidence material downside
if (signal.downside_risk_score >= 0.75
    and signal.confidence >= 0.70
    and signal.already_priced_in_score < 0.50):
    veto("material_downside_news")

# 3. Self-consistency disagreement
if signal.action_policy == "manual_review":
    veto("manual_review_required")
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

if signal.upside_catalyst_score >= 0.70 and signal.confidence >= 0.70 and final_score >= base_threshold + 0.05:
    size_mult = min(size_mult * 1.15, 1.15)   # cap upside boost

# Floor and ceiling
size_mult = max(0.0, min(1.25, size_mult))
```

### 11.4 Decision logging
EVERY decision (enter, block, size_down, boost, exit_review, hold) is written to `decision_audit` with:
- The `news_signal_id` consulted
- The `calendar_risk_payload` snapshot
- The `market_context_payload` snapshot
- The `model_score`
- The `final_decision` and `decision_path` (which rule fired)

---

## 12. Operational Workflow

### 12.1 Pre-market (08:00 ET)
1. Fetch overnight news for full universe (top 50 SPX + held positions)
2. Fetch and refresh earnings calendar for next 10 trading days
3. Fetch macro events for today
4. Cluster, dedupe, deterministic pre-score
5. LLM-score (Haiku) all clusters that pass filters
6. Generate `MarketContext` snapshot
7. Build pre-market briefing (Haiku synthesis from existing audit task #3) — now grounded in structured signals

### 12.2 At PM scan (09:45 ET swing, 60/90/120 min intraday)
1. PM requests `NewsSignal` per candidate symbol from NIS (cached, sub-100ms)
2. PM applies scoring overlay (Section 11.1)
3. RM applies veto + size logic (Section 11.2-11.3)
4. Trader executes; decision audit written

### 12.3 Intraday news monitor (every 5 minutes during market hours)
1. Fetch new articles since last cursor
2. Cluster against last 6h
3. New clusters: deterministic pre-score → LLM if material
4. Updated clusters (new sources joined): re-score if confidence delta would change action_policy
5. For symbols with held positions: any new cluster with downside_risk_score ≥ 0.60 AND confidence ≥ 0.60 triggers `exit_review` flag for next PM consultation (NOT auto-exit)

### 12.4 EOD (16:30 ET)
1. Compute realized outcomes (15m, 4h, 1d) for today's decisions where horizon has elapsed
2. Update `decision_audit.realized_outcome_*`
3. Compute counterfactual for blocked trades (would they have been profitable?)
4. Roll daily metrics (Section 14)

---

## 13. Failure Modes and Fallbacks

| Failure | Fallback | Observable signal |
|---|---|---|
| Polygon API down | Use cached news older than 5min, mark `stale=true` | Source health metric |
| Alpaca news API down | Same | Source health metric |
| EDGAR scraper down | No fallback (free, unreliable) — proceed without | Log error, alert if down >1h |
| Earnings calendar fetch failed | Use yesterday's snapshot, mark stale; if >24h stale, treat unknown earnings as risk and block swing entries | Calendar health metric |
| LLM provider primary down (Anthropic) | Fail over to OpenAI provider | LLM call log |
| LLM provider all down | Fall back to deterministic-only scoring (Section 7); confidence floored to 0.4 | LLM call log |
| LLM returns unparseable JSON | Use deterministic floor; log raw response | LLM call log |
| Daily LLM budget exceeded | Stop LLM calls; deterministic only until reset | Cost metric |
| SQLite locked / write failure | Retry 3× then alert and fail open (no signal returned, PM proceeds without news input) | Storage health |

The critical invariant: **a degraded news layer never blocks trading via false positives**. If the system can't be confident, it abstains from gating rather than over-gating.

---

## 14. Metrics and Monitoring

### 14.1 Daily metrics (rolled at EOD into `metrics_daily` table)
- Articles ingested per source
- Clusters created
- Clusters LLM-scored (Haiku / Sonnet / cached)
- LLM cost USD (total, by tier)
- LLM avg latency ms
- LLM error rate
- Decisions made: enter / block / size_down / boost / exit_review counts
- Block reason distribution
- **Hypothetical PnL of blocked trades** (would they have profited?)
- **Realized PnL of size-down trades** vs hypothetical full-size
- **Realized PnL of boosted trades** vs hypothetical base-size
- False-positive block rate (blocked, but realized 4h/1d return positive)
- News-driven exit_review rate

### 14.2 The honest metric
```text
news_layer_alpha = realized_pnl_with_news_overlay - realized_pnl_without_news_overlay
```
This is computed from the decision_audit table by replaying decisions WITHOUT the news overlay and comparing. If `news_layer_alpha < 0` over a meaningful window, the news layer is costing money and needs revision.

### 14.3 Alerts
- LLM daily cost > 80% of budget
- LLM error rate > 5% over 1 hour
- Source down > 1 hour during market hours
- Hypothetical PnL of blocked trades > realized PnL over rolling 20 days (over-gating)
- Decisions per day with `manual_review` policy > 10 (LLM uncertainty too high)

---

## 15. Implementation Phases

### Phase 1: Foundation (Week 1-2) — **highest priority**
**Goal:** Replace boolean gate with structured signal; add the highest-value calendar gate.

Deliverables:
1. Database schema migrated; all tables in Section 3 created
2. `NewsSource` ABC and adapters for Polygon, Alpaca, EDGAR
3. `ingestion.py` + `clustering.py` working end-to-end (no LLM yet)
4. `deterministic_scorer.py` producing `NewsSignal` for every cluster (rules-only)
5. **Earnings calendar** integrated with Polygon or Finnhub free tier
6. **Macro calendar** integrated (FRED + ForexFactory)
7. `CalendarRisk` and `MarketContext` dataclasses produced
8. PM and RM updated to consume `NewsSignal` + `CalendarRisk` (Section 11 rules, but using deterministic scorer only)
9. `decision_audit` populated for every PM decision
10. Old `has_negative_news()` deprecated but not removed (kept for one release as rollback safety)

**Acceptance criteria:**
- 100% of clusters dedup correctly across Polygon + Alpaca for golden test fixtures
- Earnings gate prevents at least one historically-bad scenario in replay (specify a known earnings-related drawdown in your backtest history)
- No PM decision proceeds without a written `decision_audit` row
- Old boolean gate path produces same outputs as new path for last 30 days of history (regression test)

### Phase 2: LLM Layer (Week 3-4)
**Goal:** Replace deterministic-only scoring with Haiku + Sonnet escalation; add halts.

Deliverables:
1. `LLMScorer` ABC + `ClaudeProvider` with Haiku + Sonnet escalation (Section 8.2)
2. Prompt template `materiality_v1` deployed
3. Caching layer (Section 8.3)
4. `llm_call_log` populated
5. Self-consistency check on high-stakes decisions (Section 8.2)
6. Halt feed integration
7. Daily cost tracking + budget enforcement
8. Provider redundancy: `OpenAIProvider` implemented and tested as failover

**Acceptance criteria:**
- LLM cost under $5/day for 30 consecutive days at full universe coverage
- p95 LLM latency under 2 seconds (Haiku) and under 6 seconds (Sonnet)
- 95%+ JSON parse success rate
- Failover from Claude to OpenAI works in <30 seconds when Anthropic returns 5xx for 60 seconds

### Phase 3: Operational Maturity (Week 5-6)
**Goal:** Full metrics, alerts, and post-trade analytics.

Deliverables:
1. EOD outcome tracking populated for all decisions
2. Daily metrics rolled to `metrics_daily`
3. Alerting on Section 14.3 conditions
4. Hypothetical PnL of blocked trades reported
5. Pre-market briefing rewritten to consume structured signals (Section 12.1)
6. Backtest harness for the news layer itself (replay 90 days of news through the pipeline; verify decisions match what the live system would have made)

**Acceptance criteria:**
- News-layer-alpha metric (Section 14.2) computed daily
- Backtest harness produces deterministic results (same input → same output)
- Operations runbook documented for every fallback in Section 13

### Phase 4: News as Model Feature (Week 7+)
**Goal:** Promote news from runtime overlay to model input — but only after sufficient point-in-time history exists.

**Precondition:** ≥ 60 trading days of point-in-time `news_signal` history.

Deliverables:
1. Backfill: re-score historical news through current LLM prompt to create training dataset (use cached responses where available)
2. Add `SymbolNewsFeatures` to feature pipeline
3. Train candidate XGBoost models with news features; walk-forward validate
4. Compare candidate vs baseline on out-of-sample performance net of transaction costs
5. Gate promotion on:
   - Sharpe improvement > 0.10
   - Max drawdown not worse
   - Stable feature importance across folds
   - No look-ahead leakage detected (specifically: train/test split must respect `evaluated_at`, never `published_at`)
6. If accepted, deploy with continued monitoring

**Acceptance criteria:**
- Walk-forward backtest with rolling re-train shows improvement
- Feature importance for news features is stable (not random)
- No improvement = do not promote; the runtime overlay is sufficient

### Phase 5+: Future Expansions (post-MVP)
- **Opportunity detection**: when symbol X has a negative event, identify sympathy beneficiaries (competitors, suppliers' competitors). Expand candidate universe beyond PM shortlist for opportunity-side news, with stricter confidence thresholds.
- **News-conditional model regimes**: separate XGBoost for "active news" vs "no news" symbols
- **Conviction-weighted execution**: news-driven adjustments to limit price aggressiveness, not just size
- **Cross-asset feedback**: when realized price reaction contradicts LLM direction, log this and use as future training signal for prompt revision
- **Local LLM fallback** (`OllamaProvider`): full offline capability

---

## 16. Testing Strategy

### 16.1 Unit tests
- `test_ingestion.py`: dedup logic on golden fixtures with known duplicates
- `test_clustering.py`: 5 sources publishing same story → 1 cluster
- `test_deterministic_scorer.py`: regex matchers, no false positives on adversarial inputs ("tax loss harvesting", "no loss of life", "downgrade their estimate of the cost")
- `test_llm_scorer.py`: mocked LLM provider returns various structured outputs; validation, escalation, fallback all tested
- `test_feature_builder.py`: decay function, silent stocks distinction, point-in-time correctness
- `test_policy.py`: every decision rule in Section 11 has at least one positive and one negative test case

### 16.2 Integration tests
- End-to-end fixture: 20 articles across 4 sources → expected clusters → expected signals → expected decisions
- Failover: kill Anthropic mock → OpenAI mock takes over → results within tolerance
- Cost cap: simulate budget exhaustion mid-day → deterministic-only mode engages

### 16.3 Backtest validation
- Replay last 90 days of news through the full pipeline
- Verify no `published_at` is ever > `evaluated_at` in any signal (point-in-time invariant)
- Verify deterministic reproducibility: same input + same prompt_version → same decisions

### 16.4 Adversarial tests
- Article with malicious instruction in title ("Ignore previous instructions and...") — verify LLM treats as untrusted text and the JSON contract still holds
- Article with ticker collision ("ALL", "IT", "ON" used as English words, not tickers)
- Article published 6 hours ago appearing for the first time (latency in source) — must not break decay logic
- Article with title that is identical hash but published_at differs by 4 hours

---

## 17. Configuration Reference (`config/news_intel.yaml`)

```yaml
schema_version: "v1"

universe:
  swing: ["AAPL", "MSFT", ...]   # top 50 SPX
  intraday: ["AAPL", "MSFT", ...]
  always_include_open_positions: true

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
  primary_model: claude-haiku-4-5
  escalation_model: claude-opus-4-7
  fallback_provider: openai
  fallback_model: gpt-4o-mini
  daily_budget_usd: 5.00
  max_input_tokens_per_article: 3000
  max_output_tokens: 400
  cache_ttl_hours_old: 24
  cache_ttl_hours_fresh: 1

escalation:
  haiku_min_confidence: 0.60
  haiku_materiality_threshold: 0.70
  always_escalate_event_types: [m_and_a, legal_regulatory, fda_action, halt]
  self_consistency_for_block: true

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
  # above 28 = high

monitoring:
  alert_llm_cost_pct: 0.80
  alert_llm_error_rate_1h: 0.05
  alert_source_down_minutes: 60
  alert_manual_review_per_day: 10
```

---

## 18. Open Questions for Repo-Resident Claude Code Review

The implementing agent should review the following before committing to design choices:

1. **SQLite vs Postgres for the event store**: at 500 articles/day, SQLite is fine for years. But if Phase 4 backfill writes 50k+ historical signals, is SQLite still the right choice for the `news_signal` table? Validate against actual `runner.py` performance.

2. **Existing `NewsMonitor` class**: where does the new `NewsIntelligenceService` slot in relative to it? Is `NewsMonitor` deleted, or kept as a thin adapter that delegates to NIS?

3. **Existing Polygon sentiment features in the swing model**: do we deprecate `news_sentiment_3d/7d/momentum` immediately and retrain without them, or keep them in the model until Phase 4 produces validated replacements?

4. **Test fixtures**: do we have a corpus of historical articles + price reactions to use as golden test data, or do we need to capture one over the next 2 weeks of paper trading?

5. **Existing `GitGuardrails` and `run_task.py` workflow**: NIS is a multi-week build with cross-cutting changes. Should this be one large branch, or a sequence of guarded numbered tasks per Phase 1 deliverable? Recommend the latter.

6. **Existing pre-market briefing (audit task #3)**: should the new structured signals replace the briefing's free-form narrative, or augment it? Recommend augmentation: keep the narrative summary, but seed it from structured signals so it's grounded.

7. **Universe expansion for opportunity detection (Phase 5)**: do we expand beyond top-50 SPX, and if so, what universe? Russell 1000? S&P 500? Sector ETFs only?

---

## 19. Quick Reference: What Changes Day One

If this spec is accepted and Phase 1 begins immediately:

**Removed/deprecated:**
- `has_negative_news(symbol, window=30min) -> bool` (kept for one release as fallback)
- Keyword-based negative news classification

**Added:**
- `NewsIntelligenceService.get_signal(symbol, horizon) -> NewsSignal`
- `CalendarService.get_risk(symbol, now) -> CalendarRisk`
- `MarketContextService.snapshot() -> MarketContext`
- New tables: `news_raw`, `news_cluster`, `news_signal`, `calendar_event`, `decision_audit`, `llm_call_log`

**Modified:**
- PM scoring: model_score + news_overlay (Section 11.1)
- RM logic: veto and size rules (Section 11.2-11.3)
- Pre-market briefing: now grounded in structured signals
- All trade decisions: write `decision_audit` row

**Unchanged:**
- XGBoost model architecture and training (Phase 1-3)
- Alpaca execution
- Existing 3 Haiku audit tasks (1, 2 unchanged; 3 augmented)
- Walk-forward validation methodology
- Scan timing (09:45 swing, 60/90/120 intraday)

---

## 20. Bottom Line

The current system treats news as a single fuzzy boolean. The new system treats news as structured event interpretation that feeds two distinct paths:
- A narrow, conservative **risk overlay** of hard gates for catastrophic events
- A continuous, confidence-weighted **scoring overlay** for sizing and conviction

The LLM is positioned as an interpretation engine, not a trader. Trading policy stays in deterministic Python code, auditable and testable. Every decision is logged with full provenance so backtesting and post-trade analysis can answer the only honest question: **is the news layer adding alpha, or just adding the appearance of intelligence?**

Phase 1 (earnings calendar + structured signal + decision audit) is the highest-leverage 2 weeks of work in the entire build. Get that right and everything downstream is incremental.
