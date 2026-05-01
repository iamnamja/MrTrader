# News & LLM Integration Design Review

**Project:** Automated swing + intraday equity trading system  
**Focus:** Morning news feed, real-world context ingestion, LLM-based event interpretation, and actionable trading workflow  
**Prepared for:** Min Kim  

---

## Executive Summary

Your current news setup is a solid version-one safety layer, but it is not yet a true news intelligence layer. Today, news mostly functions as a binary gate: if negative news appears, block a new entry or consider exiting. That is useful, but too blunt for a trading workflow that needs to understand event type, materiality, time horizon, confidence, novelty, and whether the information is already priced in.

The main recommendation is: **do not make the LLM the trader.** Use the LLM as an **event interpretation engine** that converts messy real-world information into structured signals that your Portfolio Manager, Risk Manager, and ML models can consume.

The right direction is not:

```text
News headline -> LLM says buy/sell -> trade
```

The right direction is:

```text
News headline -> event classification -> materiality scoring -> structured signal -> PM/RM decision policy -> trade or no trade
```

---

## Core Architecture Recommendation

Build a reusable **News Intelligence Service** that sits upstream of the Portfolio Manager and Risk Manager.

```text
Raw News / Filings / Calendars / Macro Events
        ↓
Normalize + Deduplicate + Entity Match
        ↓
Event Classifier
        ↓
LLM Materiality Scorer
        ↓
Point-in-Time Event Store
        ↓
Feature Builder
        ↓
PM Candidate Ranking
        ↓
Risk Manager Position Sizing / Veto / Pause Rules
        ↓
Trader Execution
        ↓
Post-Trade Outcome Logging
        ↓
Retraining Dataset
```

The LLM should not make final trading decisions. It should provide structured interpretation such as:

```json
{
  "symbol": "AAPL",
  "event_type": "supplier_issue",
  "event_scope": "company",
  "direction_score": -0.6,
  "materiality_score": 0.72,
  "downside_risk_4h": 0.68,
  "swing_impact_3d": -0.45,
  "confidence": 0.81,
  "novelty": 0.74,
  "already_priced_in": 0.52,
  "recommended_policy": "size_down_or_block_if_model_edge_weak",
  "reason": "Supplier disruption may pressure near-term sentiment; not confirmed by company."
}
```

---

## Critique of the Current Design

### 1. The binary negative-news gate is too blunt

Current design:

```python
has_negative_news(symbol, window=30min) -> boolean
```

This loses too much information. A downgrade, lawsuit, earnings miss, product recall, macro shock, and competitor issue may all be labeled negative, but they should not trigger the same action.

Replace it with:

```python
get_news_context(symbol, horizon="intraday") -> NewsSignal
```

Where `NewsSignal` includes materiality, direction, confidence, event type, horizon, and policy recommendation.

### 2. You only watch PM-shortlisted symbols

This means news can help you avoid bad trades, but it cannot help discover new opportunities. For example, if one semiconductor company has a major supply issue, competitors may benefit. Your system should eventually support both:

- **Risk detection:** avoid or reduce exposure to damaged names.
- **Opportunity detection:** identify beneficiaries or sympathy moves.

### 3. Sentiment alone is not enough

Vendor sentiment and keyword sentiment are weak proxies. The system needs event classification.

Examples:

- Analyst downgrade
- Earnings miss
- Guidance cut
- Regulatory action
- Lawsuit
- Product launch
- M&A rumor
- SEC filing
- Management change
- Macro/Fed event
- Sector shock

Each has different behavior over different horizons.

### 4. Training/inference mismatch risk

You currently use Polygon rolling sentiment in swing training, but intraday has no news features. If you add live LLM scores directly into inference without historical point-in-time LLM scores, you create a mismatch between training and live operation.

Use LLM signals as runtime overlays first. Only add them as ML features after you have logged enough point-in-time history.

### 5. No earnings and macro calendar layer

This is one of the highest-value missing pieces. Earnings and high-impact macro events can dominate technical signals. Add this before building complex LLM logic.

---

## Proposed End-State Workflow

### Morning Workflow: 08:00-09:45 ET

1. Pull overnight and premarket news for the full trading universe.
2. Pull earnings calendar and high-impact macro calendar.
3. Normalize and deduplicate articles.
4. Entity-match articles to symbols.
5. Apply cheap deterministic filters.
6. Send only likely material articles to the LLM.
7. Store structured `NewsSignal` records.
8. Build a market context object.
9. Feed PM with symbol-level news signals and market-level context.
10. Let PM rank candidates and RM apply sizing/veto rules.

### Intraday Workflow

1. Poll or stream news during market hours.
2. Process only new, relevant, non-duplicate events.
3. LLM-score only likely material items.
4. Update the cached signal for affected symbols.
5. PM/RM consume cached signal during scan windows.
6. Existing positions get exit-review flags only when the event is material and relevant to the holding horizon.

---

## Recommended Data Model

Create point-in-time tables so that future backtests know exactly what the bot knew at the time.

### `news_raw`

```sql
CREATE TABLE news_raw (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    provider_article_id TEXT,
    published_at TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    title TEXT NOT NULL,
    summary TEXT,
    url TEXT,
    symbols TEXT,
    source_name TEXT,
    raw_payload_hash TEXT
);
```

### `news_cluster`

```sql
CREATE TABLE news_cluster (
    cluster_id TEXT PRIMARY KEY,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    primary_symbol TEXT,
    related_symbols TEXT,
    canonical_title TEXT,
    article_ids TEXT,
    duplicate_count INTEGER DEFAULT 1
);
```

### `news_signal`

```sql
CREATE TABLE news_signal (
    signal_id TEXT PRIMARY KEY,
    cluster_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    evaluated_at TEXT NOT NULL,
    model_name TEXT NOT NULL,
    horizon TEXT NOT NULL,
    event_type TEXT,
    event_scope TEXT,
    direction_score REAL,
    materiality_score REAL,
    downside_risk_score REAL,
    upside_catalyst_score REAL,
    confidence REAL,
    novelty_score REAL,
    already_priced_in_score REAL,
    action_policy TEXT,
    rationale TEXT,
    schema_version TEXT NOT NULL
);
```

### `calendar_event`

```sql
CREATE TABLE calendar_event (
    id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    symbol TEXT,
    event_time TEXT NOT NULL,
    importance TEXT,
    source TEXT,
    payload TEXT
);
```

---

## Replace `has_negative_news()` with `NewsSignal`

Create a typed object that the PM and RM can consume.

```python
from dataclasses import dataclass
from typing import Literal

Policy = Literal[
    "ignore",
    "watch",
    "size_down",
    "block_entry",
    "exit_review",
    "manual_review",
]

@dataclass
class NewsEvent:
    source: str
    title: str
    published_at: str
    event_type: str
    materiality_score: float
    confidence: float
    rationale: str

@dataclass
class NewsSignal:
    symbol: str
    horizon: str
    direction_score: float
    materiality_score: float
    downside_risk_score: float
    upside_catalyst_score: float
    confidence: float
    novelty_score: float
    already_priced_in_score: float
    action_policy: Policy
    top_events: list[NewsEvent]
    stale: bool = False
```

PM usage:

```python
news_context = news_intel.build_symbol_signal(symbol, horizon="intraday")

if news_context.action_policy == "block_entry":
    reject_candidate(symbol, reason="Material downside news")

elif news_context.action_policy == "size_down":
    candidate.size_multiplier *= 0.50

elif news_context.upside_catalyst_score >= 0.70 and model_score >= threshold:
    candidate.conviction += 0.05
```

---

## LLM Usage Pattern

### Do not send every article to the LLM

Use a 3-stage router:

```text
Stage 1: Cheap filter
- Is symbol mentioned?
- Is source trusted?
- Is article new?
- Is this duplicate/rewrite?
- Does it contain material event keywords?

Stage 2: Lightweight deterministic scoring
- Vendor sentiment
- Keyword category
- Source reliability
- Recency
- Article count surge

Stage 3: LLM only for likely material items
- Earnings surprise
- Guidance change
- Analyst downgrade/upgrade
- M&A
- SEC filing
- Regulatory/legal issue
- Product launch/failure
- Management change
- Major macro/geopolitical shock
```

This keeps LLM cost and latency controlled.

### Prompt the LLM for event interpretation, not trade advice

Use a focused prompt:

```text
You are a financial news event classifier for an automated equity trading system.

Your job is not to recommend trades. Your job is to classify whether the attached news is material to the specified stock over the specified horizon.

Evaluate only the information provided. Do not invent facts. If the article is vague, lower confidence.

Symbol: {symbol}
Company: {company_name}
Sector: {sector}
Trading horizon: {intraday_4h_or_swing_3d}
Current position intent: {considering_long_entry | currently_long | no_position}
Article title: {title}
Article summary: {summary}
Published at: {published_at}

Current market context:
- Premarket move: {premarket_return}
- Relative volume: {relative_volume}
- SPY move: {spy_return}
- Sector ETF move: {sector_return}

Return strict JSON only.
```

Expected response:

```json
{
  "relevant_to_symbol": true,
  "event_type": "earnings|guidance|analyst_action|m_and_a|legal_regulatory|product|management|macro|geopolitical|sector|other",
  "event_scope": "company|sector|market|macro",
  "direction_score": -0.4,
  "materiality_score": 0.7,
  "downside_risk_score": 0.6,
  "upside_catalyst_score": 0.1,
  "time_horizon": "hours",
  "confidence": 0.8,
  "novelty_score": 0.7,
  "already_priced_in_score": 0.4,
  "recommended_policy": "size_down",
  "rationale": "Analyst downgrade may pressure near-term sentiment, but no fundamental company event was reported."
}
```

---

## Prevent the LLM from Being Too Conservative

Do not let the LLM decide final policy alone. Let it provide scores, then apply deterministic rules.

Example:

```python
if signal.materiality_score >= 0.80 and signal.downside_risk_score >= 0.70:
    block_entry = True

elif signal.materiality_score >= 0.60 and signal.downside_risk_score >= 0.50:
    size_multiplier = 0.50

elif signal.upside_catalyst_score >= 0.70 and model_score >= threshold:
    size_multiplier = min(1.25, base_size_multiplier)

else:
    size_multiplier = 1.0
```

Also track LLM impact over time:

- Trades blocked by news
- Trades size-reduced by news
- Trades boosted by positive catalyst
- Return of blocked trades if they had been entered
- False-positive news blocks
- False-negative missed news events

This is how you tune conservatism without guessing.

---

## News as ML Feature vs Runtime Gate

### Recommended phase approach

```text
Phase 1:
Use LLM signal as runtime overlay only.
Log every signal and every trade outcome.

Phase 2:
After enough data, build point-in-time historical features.

Phase 3:
Retrain XGBoost with LLM-derived features.

Phase 4:
Compare model with and without news features.
Only promote if walk-forward results improve after transaction costs.
```

### Swing features to add later

```text
news_materiality_1d
news_materiality_3d
news_direction_1d
news_direction_3d
news_downside_risk_1d
news_article_count_3d
news_novelty_max_3d
news_confidence_weighted_score_3d
earnings_proximity_days
post_earnings_drift_flag
macro_high_impact_day
```

### Intraday features to add later

```text
news_materiality_30m
news_direction_30m
news_downside_risk_30m
news_count_60m
source_count_60m
novelty_max_60m
macro_release_window_flag
earnings_today_flag
premarket_news_score
```

---

## Earnings and Macro Risk Layer

Add this before complex LLM features.

### Earnings rules

For swing trades:

```text
No new swing long entries:
- Day before earnings
- Day of earnings
- Morning after earnings, unless strategy explicitly supports post-earnings drift

Existing positions:
- Reduce or exit before earnings unless the model has been trained and validated on holding through earnings
```

For intraday trades:

```text
Allow only if the intraday model has a specific earnings-day strategy.
Otherwise, block or heavily reduce risk on earnings day.
```

### Macro rules

High-impact macro events:

```text
- CPI
- PPI
- FOMC decision
- Fed press conference
- Nonfarm payrolls
- Unemployment rate
- Retail sales
- Major Treasury auction events, if rate-sensitive names are in scope
```

Initial policy:

```text
Before release:
- Pause new entries 10-15 minutes before event

After release:
- Pause 5-15 minutes after event
- Resume only after spread/volatility normalizes
- Reduce risk budget for high-beta names

For FOMC:
- Pause new entries before decision
- Be extra careful during the press conference window
```

---

## Recommended Data Sources

You already have Polygon and Alpaca. Keep both.

Add these in priority order:

### 1. Earnings calendar

Use a low-cost provider such as Alpha Vantage, Financial Modeling Prep, Nasdaq earnings calendar, or another provider you can reliably automate. The important thing is not which source you pick first; it is that earnings proximity becomes a first-class risk feature.

### 2. Macro calendar

Use BLS, FRED, and/or other public economic calendars for CPI, payrolls, unemployment, PPI, and other high-impact releases. Store expected release timestamps and importance.

### 3. SEC EDGAR

Use EDGAR for company filings, especially 8-Ks. This adds a primary-source signal that is different from article sentiment.

### 4. Alpha Vantage News Sentiment

Useful as a cheap second opinion and additional sentiment feed.

### 5. RSS feeds from trusted sources

Useful later for broader market context, but less structured and more difficult to normalize.

### 6. Reddit/social

Do not prioritize yet. Treat it as an attention/rumor signal only, not a trade trigger. It has high false-positive risk.

---

## Revised Agent Workflow

Current:

```text
PM proposes -> RM approves/vetoes -> Trader executes
```

Recommended:

```text
Market Context Agent
    ↓
News/Event Agent
    ↓
PM proposes trades
    ↓
RM adjusts sizing/vetoes
    ↓
Trader executes
    ↓
Post-trade evaluator logs outcome
```

The News/Event Agent should output structured context:

```json
{
  "market_regime": "risk_on|risk_off|neutral|macro_event_pending",
  "symbol_event_risks": {},
  "symbol_catalysts": {},
  "calendar_risks": {},
  "recommended_constraints": {}
}
```

The PM decides whether the trade still makes sense. The RM decides whether the risk is acceptable.

---

## Two-Week Implementation Plan

### Priority 1: Replace boolean news gate with structured `NewsSignal`

This is the highest-impact change.

Build:

```python
class NewsIntelligenceService:
    def ingest_sources(self): ...
    def dedupe_and_cluster(self): ...
    def score_with_rules(self): ...
    def score_with_llm_if_needed(self): ...
    def build_symbol_signal(self, symbol, horizon): ...
    def build_market_context(self): ...
    def explain_signal(self, symbol): ...
```

Expose:

```python
news_context = news_intel.build_symbol_signal(symbol, horizon="intraday")
calendar_context = calendar_service.get_risk(symbol, now)
market_context = news_intel.build_market_context()
```

### Priority 2: Add earnings and macro calendar gates

Add:

```text
earnings_proximity_days
earnings_today
earnings_tomorrow
macro_high_impact_today
macro_event_window_now
```

Simple first rules:

```text
Swing:
- Avoid new entries within 1 trading day before earnings
- Avoid holding through earnings unless explicitly allowed

Intraday:
- Pause new entries around CPI, FOMC, and NFP
- Resume after volatility normalizes
```

### Priority 3: Add LLM materiality scoring for filtered events only

Do this in the morning and intraday, but always cache results.

```text
Premarket:
- Pull overnight articles for universe
- Deduplicate
- Select top material candidates
- LLM scores only those
- Save structured scores
- PM reads scores at 09:45

Intraday:
- Stream or poll news
- If article passes materiality prefilter, call LLM
- Update signal cache
- PM/RM uses cached signal
```

Do not call the LLM synchronously inside every PM decision.

---

## Example PM and RM Decision Policy

PM score adjustment:

```python
final_score = (
    model_score
    + 0.10 * news_context.upside_catalyst_score
    - 0.15 * news_context.downside_risk_score
    - 0.10 * calendar_context.earnings_risk
    - 0.10 * market_context.macro_risk
)
```

RM veto/size logic:

```python
if calendar_context.earnings_today:
    veto("Earnings today")

if market_context.macro_event_window_now:
    veto("High-impact macro release window")

if news_context.downside_risk_score > 0.75 and news_context.confidence > 0.70:
    veto("Material downside news")

if news_context.downside_risk_score > 0.50:
    size *= 0.50
```

---

## Backtesting and Feedback Loop

To avoid look-ahead bias:

1. Store `published_at`, `ingested_at`, and `evaluated_at` for every article and LLM signal.
2. During backtests, only use articles and signals available before the simulated decision timestamp.
3. Do not use revised summaries or updated articles unless they were available at the time.
4. Version your LLM prompt and schema.
5. Version your model/provider.
6. Store failed LLM calls and fallback behavior.

Track these metrics:

```text
- PnL impact of news blocks
- PnL impact of size reductions
- Return of blocked trades if entered
- False-positive block rate
- False-negative event miss rate
- Average LLM cost per day
- Average LLM latency
- Number of articles scored per day
- Number of trades affected by news layer
```

This lets you answer the important question: did the news layer improve real trading outcomes, or did it just make the system sound smarter?

---

## Key Risks and Edge Cases

### 1. Duplicate news amplification

The same story may be syndicated across many sources. Without clustering, the system may think one event is several independent events.

### 2. Ticker ambiguity

Some tickers are common words. Entity matching must validate company name, exchange, and context.

### 3. Already-priced-in events

A negative headline after the stock is already down 8% may not mean avoid. It may mean the risk has already expressed itself. Include price context in the LLM prompt.

### 4. Source quality

Treat SEC filings, company releases, and trusted financial news differently from blogs or social posts.

### 5. Macro days

On CPI/FOMC/NFP days, single-name technicals may be overwhelmed by index-level movement.

### 6. Over-blocking

If the system blocks too often, it may avoid both bad trades and good opportunities. Measure the hypothetical return of blocked trades.

### 7. Latency and cost

Do not LLM-score inside the hot path. Precompute, cache, and fail open.

---

## Bottom Line

Your current approach is not fundamentally flawed. It is a good start. But it is currently more of a **news safety gate** than a **news intelligence system**.

The next step should be:

```text
Keyword/news sentiment gate -> structured event intelligence layer
```

The system should evolve from:

```text
“News is negative, block trade.”
```

to:

```text
“This is a company-specific regulatory event, high materiality, high downside risk, likely relevant over 1-3 days, confidence 0.82, and not fully priced in. Reduce size or block entry unless model edge is very strong.”
```

The highest-impact next actions are:

1. Replace the boolean negative-news gate with a structured `NewsSignal`.
2. Add earnings and macro calendar gates immediately.
3. Add LLM materiality scoring only for filtered, deduped, high-relevance events.

This keeps the system flexible, extensible, testable, and realistic for live trading.

