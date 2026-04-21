# MrTrader — Phases 17–22 Specification

Last updated: 2026-04-21

---

## Context

Phases 1–16 delivered: ML entry signals, basic stop/target/trail exits, sector-level risk rules,
regime detection, mean-reversion strategy, circuit breaker, kill switch, audit log, dashboard,
portfolio-level risk, and backtesting infrastructure.

Phases 17–22 transform MrTrader from an entry-centric system into a fully dynamic, position-aware
trading bot modelled on real trading desk practices.

---

## Phase 17 — Core Workflow Overhaul

**Goal:** PM becomes continuous. Agents communicate bidirectionally. Hold periods are dynamic.
Execution is conviction-weighted.

### 17.1 PM Continuous Review Loop

PM currently fires once at market open. New behaviour: PM runs a **30-minute review cycle**
during market hours (9:30 AM – 4:00 PM ET) doing three things per cycle:

1. **Re-score open positions** — fetch latest bars, run `model.predict()` on fresh features
   - Score < EXIT_THRESHOLD (0.35) → send exit signal to `trader_exit_requests` queue
   - Score still strong (≥ 0.55) → no action, let Trader's price-action logic handle
   - Earnings within 3 days → send exit regardless of score

2. **Scan universe for new entries** — if gross exposure < 80% and budget allows
   - Swing: score SP-500 universe
   - Intraday: score Russell 1000 universe
   - Top-ranked new opportunities → `trade_proposals` queue → RM → Trader (existing flow)
   - Filter out: already held, already queued, recently RM-rejected symbols

3. **Target adjustment on profitable positions** — if score has increased since entry
   - Score increase ≥ 0.10 → extend target by 0.5 × ATR
   - Write updated `target_price` to `trades` DB table (durable, survives restart)

### 17.2 New Redis Queues

| Queue | Direction | Purpose |
|---|---|---|
| `trader_exit_requests` | PM → Trader | PM-driven exits (thesis broken, earnings, score degraded) |
| `pm_reeval_requests` | Trader → PM | Trader requests re-evaluation when technicals weaken |

Exit requests bypass RM — exits are always valid, no approval needed.

PM processes `pm_reeval_requests` in its 30-min cycle and responds to `trader_exit_requests`
with one of: `EXIT`, `HOLD`, or `EXTEND_TARGET`.

### 17.3 Trader — Technical Exit Signals (triggers for PM reeval requests)

Trader currently exits on stop/target/trail. Add technical weakness signals that trigger
a PM reeval request rather than an immediate exit:

- RSI diverging against position (RSI making lower highs while price makes higher highs, or vice versa)
- MACD histogram crossing against the position direction
- Volume fading: current bar volume < 50% of 20-bar average on a directional bar
- Price approaching stop but not hit: within 0.5 × ATR of stop → request reeval

On receiving PM's response:
- `EXIT` → `_execute_exit()` with reason `PM_THESIS_BROKEN`
- `HOLD` → continue monitoring
- `EXTEND_TARGET` → update `target_price` in position state and DB

### 17.4 Dynamic Hold Period

Remove hardcoded 10-day swing / 2-hour intraday limits.

**Swing:** hold as long as technical trend intact AND PM score ≥ 0.35 AND no exit signal.
Maximum configurable cap (default 20 days) as a safety net.

**Intraday:** hold as long as momentum indicators positive AND no reversal signal.
Only hard gate: must close by 3:45 PM EOD. No hardcoded 2-hour limit.

### 17.5 Partial Exits

When a position hits 1 × ATR profit:
- Exit `PARTIAL_EXIT_PCT` of shares (default 50%, configurable in DB agent config)
- Move remaining stop to breakeven (entry price × 0.999)
- Track `partial_exited: True` flag on position so logic fires only once
- Remaining shares continue with existing trailing stop logic

### 17.6 Conviction-Based Position Sizing

Wire ML confidence score into `size_position()`:

| ML Score | Size Multiplier |
|---|---|
| 0.55 – 0.65 | 0.75× standard |
| 0.65 – 0.75 | 1.0× standard |
| > 0.75 | 1.25× standard |

Still bounded by RM hard limits — conviction multiplier cannot breach position size cap.

### 17.7 Strategy-Level Circuit Breakers

Separate circuit breakers for swing and intraday:

- **Intraday CB**: if win rate drops below 40% over last 20 intraday trades → pause intraday
  entries only; swing continues unaffected
- **Swing CB**: if swing underperforms SPY by > 10% over 30 days → flag for manual review
  (do not auto-pause — swing moves too slowly for automated intervention to be reliable)
- Global circuit breaker remains as last resort

### 17.8 Implementation Order (within Phase 17)

1. `trader_exit_requests` queue + Trader listener
2. PM 30-min review loop — position re-scoring and exit signals first
3. Partial exit logic in Trader
4. Conviction-based position sizing
5. `pm_reeval_requests` queue + PM response handler
6. Strategy-level circuit breakers
7. Dynamic hold period (replace hardcoded limits)

---

## Phase 18 — Execution Quality

**Goal:** Better average fill prices. Liquidity protection. Slippage visibility.

### 18.1 Limit Orders for Swing Entries

Replace market orders with limit orders at entry − 0.3% for swing trades.
- If unfilled by session end → cancel; symbol stays in `approved_symbols` for next day
- Intraday entries remain market orders (speed matters more than price for intraday)
- Track fill rate in DB to monitor if limits are reducing entry rate meaningfully

### 18.2 Bid-Ask Spread Check (pre-trade)

In RM pre-approval:
- Fetch latest quote from Alpaca before approving
- If spread > 0.15% of mid-price → reject with reason `spread_too_wide`
- Prevents paying excessive transaction costs on illiquid names

### 18.3 ADTV Liquidity Check in RM

Compute 20-day average daily trading volume in dollars (ADTV) for each proposed trade.
- If proposed trade cost > 1% of ADTV → reject or reduce size proportionally
- Prevents market impact from own orders, critical as account scales

### 18.4 Slippage Tracking

- Record `intended_price` (price at signal generation) vs `filled_price` (actual fill) on every trade
- Slippage = (filled − intended) / intended × 100 bps
- Aggregate in performance dashboard: avg slippage by symbol, by time of day, by strategy
- Alert if rolling average slippage > 20 bps

---

## Phase 19 — Risk Intelligence

**Goal:** Upgrade RM from position-by-position checks to portfolio-level risk awareness.

### 19.1 Correlation Risk

Before approving a new entry, compute 60-day return correlation between proposed symbol
and each open position.
- If correlation > 0.75 with any existing position → reject or reduce size by 50%
- Prevents holding effectively duplicate positions under different tickers

### 19.2 Beta-Adjusted Exposure

Track portfolio beta = Σ(position_value × beta) / account_value.
- New RM rule: if portfolio beta > 1.3 → block new entries with beta > 1.2
- Beta sourced from 252-day regression vs SPY (already computed in swing features)
- Prevents a "diversified" 10-stock portfolio from being 1.5× levered to SPY

### 19.3 Factor Overexposure Detection

Tag each position by dominant factor: momentum, value, growth, defensive.
Initial implementation uses sector as a proxy.
- If > 60% of portfolio capital is in the same factor → reduce new entries in that factor
- Refine factor tagging in a later iteration

### 19.4 Revised RM Rule Execution Order

1. PDT compliance (from Phase 21 — priority elevated if account < $25k)
2. ADTV liquidity
3. Bid-ask spread
4. Gross exposure cap
5. Strategy budget cap
6. Buying power
7. Position size
8. Correlation check *(Phase 19)*
9. Beta exposure *(Phase 19)*
10. Sector concentration
11. Factor concentration *(Phase 19)*
12. Daily loss limit
13. Account drawdown
14. Open position count
15. Portfolio heat

---

## Phase 20 — Pre-market Intelligence & Event Monitoring

**Goal:** Situational awareness before first trade of day and throughout market hours.

### 20.1 Pre-market Routine (9:00–9:25 AM ET daily)

New `PremarketAgent` or job in orchestrator:

- **Economic calendar**: fetch today's macro events (FOMC, CPI, NFP, GDP) from a free API
  - FOMC announcement days: reduce all position sizing 50%, no new intraday entries
  - NFP Fridays: no intraday entries before 10:00 AM ET
- **Overnight gap analysis**: for each open position, compute open vs prior close
  - Gap adverse > 3% → immediate PM re-evaluation at open (don't wait for 30-min cycle)
  - Gap adverse > 5% → auto-exit at open
- **Futures context**: SPY pre-market % feeds intraday sizing
  - SPY futures down > 1.5% → halve intraday position sizes
  - SPY futures down > 2.5% → no new intraday entries that session

### 20.2 Real-Time Event Monitoring

- **Polygon.io news webhook**: stream news for all held symbols and approved watchlist
  - Material news on a held symbol → bypass 30-min cycle, trigger immediate PM reeval
- **SEC EDGAR 8-K monitor**: poll for 8-K filings on held symbols every 15 minutes
  - Material item types (1.01 material agreements, 2.02 results of operations, 5.02 executive changes)
    → trigger immediate reeval
- **Analyst changes**: FMP API (already integrated) — poll pre-market daily
  - Downgrade on held position → PM reeval request
  - Upgrade on approved-but-not-entered symbol → bump priority in queue

### 20.3 Confirmed Earnings Calendar

Replace estimated earnings dates (last 10-Q + 90 days) with confirmed dates from FMP.
- Auto-exit swing positions 3 days before confirmed earnings (not 3-day estimate window)
- Block new entries for any symbol with earnings within 5 calendar days

---

## Phase 21 — Compliance & Regulatory Guardrails

**Goal:** Protect the account from regulatory restrictions and tax surprises.

### 21.1 PDT Rule Tracking

Pattern Day Trader rule: if account equity < $25,000, max 3 round-trip day trades
per rolling 5-business-day window. Violation → 90-day restriction.

- Track day trade count in DB (buy + sell same symbol same session = 1 round trip)
- If equity < $25k and day trade count reaches 2 → block new intraday entries for remainder
  of the 5-day window; display warning on dashboard
- If equity ≥ $25k → PDT check skipped (PDT designation allows unlimited day trades)

### 21.2 Wash Sale Awareness

- Track all closed positions that realized a loss
- If PM proposes re-entering same symbol within 30 days of a loss close → RM flags warning
  in trade record (does not block — let trader decide — but makes it visible)
- Dashboard shows active wash sale windows

### 21.3 Settlement Date Tracking (Reg T)

Equities settle T+1 (since May 2024).
- Track settled vs unsettled cash separately in account state
- RM buying power check uses settled cash only to prevent free-riding violations
- Unsettled cash displayed separately on dashboard

### 21.4 Enhanced Kill Switch

Tiered halt levels (currently only global):

| Level | Scope | Trigger |
|---|---|---|
| Symbol-level | Block single ticker | Bad fill, data anomaly, manual override |
| Strategy-level | Pause intraday or swing | Poor recent performance, model degradation |
| Account-level | Full halt | Circuit breaker, manual emergency |

Symbol-level and strategy-level halts configurable from dashboard without restart.

---

## Phase 22 — Performance Intelligence & Continuous Improvement

**Goal:** Close the feedback loop so the system learns from what is and isn't working.

### 22.1 Live Signal Quality Monitoring

- Track win rate, avg P&L, avg hold time broken down by signal type
  (ML_RANK, EMA_CROSSOVER, RSI_DIP, MEAN_REVERSION)
- Rolling 30-trade window per signal type
- If any signal type win rate drops below 45% → downweight or disable via DB config (no restart)
- Dashboard panel: live signal quality by type

### 22.2 Benchmark Comparison

- Fetch SPY daily return each session
- Track: strategy cumulative return, SPY cumulative return, alpha, Sharpe vs SPY Sharpe
- Dashboard shows all four metrics on the main performance panel
- If strategy underperforms SPY by > 15% over 60 days → automated review flag (not auto-halt)

### 22.3 Model Health Monitoring

After each PM 30-min cycle, log:
- Count of EXIT / HOLD / EXTEND_TARGET decisions
- Average score of held positions
- Prediction score distribution vs training distribution (concept drift detection)

Alerts:
- Average held position score drops below 0.45 → model may be stale → flag for retraining
- Score distribution diverges significantly from training → model extrapolating outside domain

### 22.4 Weekly Performance Report

Auto-generated every Friday at market close (extends Phase 15 analytics):
- Trades taken (count, win rate, avg P&L)
- Win rate by strategy and signal type
- Benchmark comparison (strategy vs SPY)
- Signal quality summary
- Factor/sector exposure summary
- Top 3 winners and losers with PM reasoning
- Model health status

---

## Swing Universe Expansion

**Current:** SP-100 (~81 symbols)
**Planned:** S&P 500 (~500 symbols)

**Rationale:** S&P 500 is the natural middle ground. All 500 names are:
- Liquid (ADTV typically > $50M)
- Covered by SEC EDGAR XBRL (fundamentals)
- Available on Alpaca and Polygon
- Well within FINRA Reg SHO coverage (short interest)

**What this requires:**
- Update `SECTOR_MAP` in `app/utils/constants.py` to cover all ~500 SP-500 symbols
- Retrain swing model on SP-500 universe (3-5 years daily bars)
- Feature engineering unchanged — same 126 features, larger symbol set
- Training time increases ~6× (manageable on modern hardware)
- Clear feature store before retrain to avoid mixed-universe cache entries

**When:** After Phase 17 is stable. Expanding universe mid-refactor adds noise.

---

## Intraday Universe

**Unchanged:** Russell 1000 (~1000 symbols)

---

## Summary Table

| Phase | Name | Primary Benefit | Estimated Effort |
|---|---|---|---|
| 17 | Core Workflow Overhaul | Continuous PM, bidirectional agents, dynamic holds, partial exits, conviction sizing | 2–3 weeks |
| 18 | Execution Quality | Better fills, liquidity protection, slippage tracking | 3–5 days |
| 19 | Risk Intelligence | Correlation, beta, factor risk in RM | 1 week |
| 20 | Pre-market & Event Intelligence | Economic calendar, real-time news, earnings accuracy | 1–2 weeks |
| 21 | Compliance & Guardrails | PDT, wash sale, settlement, tiered kill switch | 3–4 days |
| 22 | Performance Intelligence | Live signal quality, benchmark comparison, model health | 1 week |
