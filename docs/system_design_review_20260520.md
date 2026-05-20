# MrTrader — Comprehensive System Design Review
**Date:** 2026-05-20  
**Purpose:** Full architecture review for LLM analysis — PM / RM / Trader pipeline, data flows, L/S model design, NIS integration, and design rationale.

---

## 1. EXECUTIVE SUMMARY

MrTrader is a Python-based algorithmic trading system targeting a $20k paper account on Alpaca. The architecture follows a three-agent pipeline: Portfolio Manager (PM) generates scored proposals; Risk Manager (RM) applies 11-point risk validation; Trader Agent executes entries and manages exits. Agents communicate via Redis queues (async).

**Current live configuration:**
- Strategy selector: `pead_quality_short`
- Shorts enabled: `pm.pead_enable_shorts = true`
- Net long target: 40% (`pm.ls_net_exposure_pct = 0.40`)
- Long universe: top 20 by composite score
- Short universe: top 15 by negative signal
- Max open positions: 7 (configurable, DB-backed)
- Per-position risk: 2% of account NAV

---

## 2. HIGH-LEVEL ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────┐
│                   PORTFOLIO MANAGER                          │
│  08:00 ET: Swing pre-market analysis                        │
│  09:45/10:45/13:00 ET: Intraday scans                       │
│                                                             │
│  Swing Selector (DB config: pm.swing_selector)              │
│  ├─ "pead_quality_short"  ← LIVE                           │
│  ├─ "pead"                                                  │
│  ├─ "quality_short"                                         │
│  ├─ "factor_portfolio"                                      │
│  └─ default: XGBoost ML model                              │
│                                                             │
│  NIS Overlay (Tier 1 macro + Tier 2 per-symbol)            │
│  Regime sizing multiplier (0.3× – 1.0×)                   │
│  Vol-targeting quantity adjustment                          │
│  ProposalLog persistence (UUID-tracked)                    │
└──────────────────────────┬──────────────────────────────────┘
                           │ Redis: trade_proposals queue
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    RISK MANAGER                              │
│  RiskLimits: 14 params, DB-configurable, 60s TTL cache     │
│  11-point sequential validation (FAIL-FAST)                 │
│  Direction-aware rules (shorts reduce net exposure)         │
│  Compliance integration (PDT, wash sale, settlement)        │
│  ProposalLog writeback (APPROVED / REJECTED + reason)      │
└──────────────────────────┬──────────────────────────────────┘
                           │ Redis: trader_approved_trades queue
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    TRADER AGENT                              │
│  Startup: Alpaca ↔ DB reconciliation                       │
│  Entry: limit (swing) or market (intraday)                 │
│  PENDING_FILL persistence (survives restarts)               │
│  Active positions: in-memory dict + DB Trade record         │
│  Exit loop: stop / target / trailing / time / PM / news    │
│  Partial exit: 50% at 1×ATR profit → breakeven stop        │
│  P&L: Order table as authoritative ledger (recompute_partial_pnl) │
└──────────────────────────┬──────────────────────────────────┘
                           │ Alpaca API (broker)
                           ▼
              ┌────────────────────────┐
              │   Alpaca Paper Account │
              │   Shorting enabled ✓   │
              │   $20k capital         │
              └────────────────────────┘
```

---

## 3. PORTFOLIO MANAGER — DETAILED DESIGN

### 3.1 Daily Schedule (Eastern Time)

| Time | Task | Output |
|------|------|--------|
| 06:00 | Earnings calendar prefetch | Cached for all gate checks today |
| 08:00–09:40 | Swing pre-market analysis | swing_proposals cached in memory |
| 09:00 | Premarket intelligence + NIS morning digest | Macro Tier 1 signal + per-symbol Tier 2 cache warm |
| 09:00 (async loop) | Post-event NIS refresh check | Re-fetch macro 3 min after each calendar event release |
| 09:45 | Intraday scan window 1 | Full Russell 1000 → top 150 by volume → intraday model scores → morning candidates list |
| 09:50–11:00 | Send swing proposals to RM | Queued to Redis |
| 10:45 | Intraday scan window 2 | Re-scores morning candidates |
| 13:00 | Intraday scan window 3 | Re-scores morning candidates |
| Throughout | Adaptive re-scan trigger | If SPY moves ≥1.5% from last scan baseline, re-rank (1h cooldown, no lunch 11:15–13:00) |
| 15:55 | EOD cleanup | Cancel unfilled PENDING_FILL limit orders |
| 17:00+ | Model retraining | XGBoost HPO on fresh daily bars |

### 3.2 Swing Selector Routing

```python
selector = get_agent_config(db, "pm.swing_selector")
# Routes to one of:
#   _analyze_swing_pead_quality_short()  ← LIVE (pead_quality_short)
#   _analyze_swing_pead()
#   _analyze_swing_quality_short()
#   _analyze_swing_factor_portfolio()
#   (default) XGBoost ML model daily scoring
```

### 3.3 Active Strategy: PEAD + Quality Short (`pead_quality_short`)

Combines two complementary signal sources:

**PEAD Scorer** (`app/ml/pead_scorer.py`):
- Source: FMP `get_earnings_features_at(symbol, as_of)` — point-in-time safe
- Long signal: EPS surprise > +5%, report ≤ 3 calendar days old
- Short signal: EPS surprise < −5%, report ≤ 3 calendar days old
- Confidence scaling: 5% surprise → 0.65 conf; ≥20% → 0.90 conf (clamped)
- Max hold: 5 trading days (annotated on proposal)
- Rationale: Post-earnings anomaly drift is well-documented in academic literature; FMP already returns `epsActual`/`epsEstimated` with PIT filtering — zero incremental data cost

**QualityShortScorer** (`app/ml/factor_scorer.py` or equivalent):
- Short candidates based on fundamental deterioration signals
- Low/negative profit margin, declining revenue growth, high PE inverse signal
- Returns `direction = "SELL_SHORT"` with negative confidence

**CombinedLSScorer**: Merges both, deduplicates symbols, respects `pm.ls_top_n_long` and `pm.ls_top_n_short` caps.

### 3.4 Factor Portfolio Strategy (available, not currently live)

Composite score (daily cross-sectional, z-scored ±3σ):

| Factor | Weight | IC / Rationale |
|--------|--------|----------------|
| `momentum_252d_ex1m` | 2.0× | IR=1.99 (dominant); 12-month ex 1-month momentum; well-documented in literature |
| `price_to_52w_high` | 1.0× | Breakout proximity; strength indicator |
| `profit_margin` | 1.0× | Quality filter; fundamentally sound longs |
| `operating_margin` | 1.0× | Quality filter |
| `pe_ratio` (inverse) | 1.0× | Value tilt |
| `price_to_52w_low` | 0.5× | Tier 2 (optional) |
| `volume_trend` | 0.5× | Tier 2 (optional); 20d/60d vol ratio |
| `range_expansion` | 0.5× | Tier 2 (optional); 5d ATR / 20d ATR |
| `gross_margin` | 0.5× | Tier 2 (optional) |
| `revenue_growth_yoy` | 0.5× | Tier 2 (optional) |

Regime gate: SPY > SMA(200) AND VIX < 30. If fails → suppress all entries.

### 3.5 Proposal Construction Pipeline

For each scored symbol:

```
1. Entry quality gates (_check_swing_entry_gates):
   - Earnings within 3 days? → reject
   - Symbol already held? → reject
   - Daily rejection count ≥ 3? → reject

2. Position sizing:
   - Base: account_value × position_risk_pct (2%) × confidence_scalar
   - confidence_scalar: linear 0.5×–2.0× over [min_conf, 1.0]
   - Regime multiplier: 0.3×–1.0× (from premarket_intel regime detector)
   - Vol-targeting: scale inversely to ATR normalization

3. NIS Tier 2 overlay per symbol:
   - sig = nis.get_stock_signal(symbol, sector, 24h, macro_ctx)
   - action_policy == "block_entry" → skip symbol entirely
   - sizing_multiplier ≠ 1.0 → adjust quantity

4. Decision audit (write_decision()):
   - Logs: model_score, size_multiplier, news_signal, macro_context,
           regime_sizing_mult, vol_targeting_mult, top_features

5. ProposalLog record:
   - Status: "PENDING" → "SENT" (at 09:50)
   - UUID-keyed for end-to-end lineage

6. Proposal dict:
   {
     "symbol":         str,
     "direction":      "BUY" | "SELL_SHORT",
     "quantity":       int,
     "entry_price":    float,
     "confidence":     float [0.55, 0.95],
     "stop_loss":      float,   # e.g. entry×0.98 for long, entry×1.02 for short
     "profit_target":  float,   # e.g. entry×1.05 for long, entry×0.95 for short
     "trade_type":     "swing" | "intraday",
     "proposal_uuid":  str,     # UUIDv4, links Proposal→Trade→Order
     "news_signal":    dict,    # optional
     "macro_sizing_factor": float,  # optional
   }
```

### 3.6 Intraday Multi-Window Regime Logic

```
SPY intraday move:
  ≤ −1.5%    → HARD_STOP: no new intraday entries all session
  −1.5%–−0.75% → CAUTION: min_conf=0.65, size=50%
  ≥ +2.0%   → CHASE: min_conf=0.65, gap-up caution

Opportunity score gate (Phase 88):
  opp_score < 0.35   → suppress all intraday entries
  0.35–0.64          → cap at 2 candidates
  ≥ 0.65             → normal (TOP_N_INTRADAY=5)

Macro calendar gate (Phase 59):
  block_new_entries=true → suppress entries during FOMC/NFP/CPI
  sizing_factor < 1.0   → reduce all intraday sizes
```

---

## 4. NEWS INTELLIGENCE SERVICE (NIS)

### Design Philosophy

NIS is a two-tier overlay, not a standalone signal. It does not replace ML scoring; it adjusts position sizing and blocks individual entries when news sentiment contradicts the ML signal or when macro risk is elevated.

### 4.1 Tier 1: Macro Context (Day-level)

```
Source:    Finnhub economic calendar → Claude Haiku LLM classification
Cadence:   Once per day (09:00 ET) + forced refresh 3 min after each event release
Cache:     MacroSignalCache (DB); TTL = 1 day
Outputs:
  - overall_risk:         'LOW' | 'MEDIUM' | 'HIGH'
  - global_sizing_factor: 0.3–1.0 (applied to ALL proposals)
  - block_new_entries:    bool (e.g., 10 min around major event)
  - events_today:         list[MacroEventSignal]
  - rationale:            LLM explanation string
```

Post-event refresh logic:
```python
# In 09:00 ET heartbeat loop (_maybe_refresh_nis_post_event):
# For each calendar event: if released ≥3 min AND <8 min ago AND not refreshed today:
#   nis.invalidate_macro_cache()
#   ctx = nis.get_macro_context()      ← forces Finnhub + Haiku re-call
#   persist as post_event snapshot to MacroSignalCache
```

### 4.2 Tier 2: Stock Signal (Per-symbol, 1h cache)

```
Source:    Finnhub company news → Claude Haiku LLM scoring
Cadence:   On-demand per proposal (cached 1h per symbol)
Cache:     NewsSignalCache (DB)
Outputs:
  - direction_score:          float [-1.0, +1.0]
  - materiality_score:        float [0.0, 1.0]
  - downside_risk_score:      float [0.0, 1.0]
  - upside_catalyst_score:    float [0.0, 1.0]
  - confidence:               float [0.0, 1.0]
  - already_priced_in_score:  float [0.0, 1.0]
  - action_policy:            'ignore' | 'size_down' | 'block_entry'
  - sizing_multiplier:        float (0.3×–1.5×)
  - rationale:                str
```

### 4.3 NIS and Long/Short Considerations

NIS was designed before L/S was enabled. Its current integration is directionally correct but has a potential blind spot:

- `action_policy = "block_entry"` for a SELL_SHORT proposal blocks a short on negative news — which is actually the **correct** direction for a short position. The NIS logic may be over-filtering shorts where negative news is the thesis.
- `direction_score` is currently used to scale sizing but not to flip or validate the ML direction signal.

**Recommendation for LLM review:** Should NIS Tier 2 `direction_score` be compared against `proposal["direction"]` to validate alignment? A short entry with `direction_score = -0.8` (strong bearish news) should arguably get a **larger** size, not a block.

---

## 5. RISK MANAGER — DETAILED DESIGN

### 5.1 RiskLimits: 14-Parameter Dataclass

All DB-configurable (key: `risk.*`), reloaded every 60 seconds, None-coalesced on missing DB rows.

| Parameter | Default | Config Key |
|-----------|---------|------------|
| MAX_POSITION_SIZE_PCT | 5% | `risk.max_position_size_pct` |
| MAX_SECTOR_CONCENTRATION_PCT | 20% | `risk.max_sector_concentration_pct` |
| MAX_DAILY_LOSS_PCT | 2% | `risk.max_daily_loss_pct` |
| MAX_ACCOUNT_DRAWDOWN_PCT | 5% | `risk.max_account_drawdown_pct` |
| MAX_OPEN_POSITIONS | 5 | `risk.max_open_positions` |
| MAX_PORTFOLIO_HEAT_PCT | 6% | `risk.max_portfolio_heat_pct` |
| NORMAL_VOLATILITY_ATR_RATIO | 2% | `risk.normal_volatility_atr_ratio` |
| STOP_LOSS_BASE_PCT | 2% | `risk.stop_loss_base_pct` |
| max_spread_pct | 0.5% | `risk.max_spread_pct` |
| max_adtv_pct | 1% | `risk.max_adtv_pct` |
| max_correlation | 0.75 | `risk.max_correlation` |
| max_portfolio_beta | 1.30 | `risk.max_portfolio_beta` |
| high_beta_threshold | 1.20 | `risk.high_beta_threshold` |
| max_factor_concentration | 60% | `risk.max_factor_concentration` |

### 5.2 11-Point Validation Sequence (fail-fast)

```
Rule 0:   Symbol halt check         (compliance_tracker.is_symbol_halted)
Rule 0:   Earnings calendar gate    (earnings_calendar.get_earnings_risk)
Rule 0:   Macro event window gate   (macro_calendar.get_context → block_new_entries)
Rule 0:   Intraday position cap     (_open_intraday_count ≥ MAX_INTRADAY_POSITIONS=3)
Rule 0a:  PDT check                 (intraday only; equity < $25k → block at 2 day trades)
Rule 0b:  Gross exposure cap        (total deployed + new > 80% of account)
Rule 0c:  Strategy budget cap       (swing ≤ 70% / intraday ≤ 30% of account)
Rule 1:   Buying power              (cash ≥ required; 1.5× for SELL_SHORT)
Rule 2:   Position size             (trade value ≤ MAX_POSITION_SIZE_PCT × account)
Rule 3:   Sector concentration      (direction-aware: shorts signed negative)
Rule 3b:  Correlation risk          (60-day return correlation ≤ 0.75; hedges negate)
Rule 4:   Daily loss                (today's realized P&L ≤ MAX_DAILY_LOSS_PCT × account)
Rule 5:   Account drawdown          (peak_equity → current ≤ MAX_ACCOUNT_DRAWDOWN_PCT)
Rule 6:   Open positions            (open count ≤ MAX_OPEN_POSITIONS)
Rule 7:   Portfolio heat            (sum of position risks ≤ MAX_PORTFOLIO_HEAT_PCT)
Rule 8:   Dynamic stop loss         (ATR-scaled; base 2%, scales by ATR/price ÷ 0.02)
Rule 8a:  Bid-ask spread            (swing only; spread ≤ 0.5%; IEX > 2% = stale → reject)
Rule 8b:  ADTV liquidity gate       (swing only; trade size ≤ 1% of 20d ADTV)
Rule 9:   Beta-adjusted exposure    (swing only; portfolio beta ≤ 1.30)
Rule 10:  Factor/sector conc.       (swing only; sector ≤ 60% of portfolio)
```

Each rule returns `(bool passed, str reason)`. First failure → immediate REJECTED + reason to ProposalLog.

### 5.3 Direction-Aware Rules

After the audit campaign, all direction-sensitive calculations handle SELL_SHORT correctly:

- **Sector concentration**: `signed_cost = -proposed_cost if direction == "SELL_SHORT" else proposed_cost`. Shorts reduce net sector exposure. Check uses `abs(total_sector_value) / account_value`.
- **Correlation**: Shorts treated as hedges (negative correlation contribution). A long + short in correlated names partially cancels.
- **Buying power**: Short sales require 1.5× in margin (Reg T).
- **Factor concentration**: Short positions subtract from sector exposure total.
- **Beta exposure**: Short positions contribute negative beta to portfolio.
- **Portfolio heat**: `abs(entry - stop) × abs(qty) / account_value` — direction-neutral by construction.

### 5.4 Persistence Across Restarts

- **Peak equity**: Saved to `config_store` (DB key) on each update; reloaded at startup.
- **Intraday count**: Rebuilt from DB on startup (`Trade.status IN ("ACTIVE", "PENDING_FILL")` and `trade_type="intraday"`).
- **PDT state**: `load_day_trades_from_db()` idempotent — clears window range before repopulating.
- **Wash sale**: `load_loss_closes_from_db()` reloads last 30-day loss closes.

### 5.5 Compliance Integration

**PDT Rule** (21.1):
- Tracks round-trips per rolling 5-business-day window
- Blocks intraday entries at count = 2 (not 3) when equity < $25k
- Records: `record_day_trade(symbol)` per completed intraday round-trip

**Wash Sale Awareness** (21.2):
- Warning-only (does not hard-block)
- Direction-aware: re-entering SELL_SHORT after a SELL_SHORT loss triggers warning; re-entering opposite direction does not
- Legacy entries (pre-direction tracking) default to "BUY"

**Settlement Tracking (Reg T, T+1)** (21.3):
- Sale proceeds recorded as unsettled: `record_sale_proceeds(amount)`
- T+1 settlement walks forward by business days (Friday sale → Monday settlement, not Saturday)
- `settled_buying_power()` = total BP − unsettled cash

**Symbol Halt List** (21.4):
- Instant halt without restart: `halt_symbol(symbol, reason)`
- Checked as first gate in RM validation sequence

---

## 6. TRADER AGENT — DETAILED DESIGN

### 6.1 Startup Reconciliation

On every startup, Trader reconciles Alpaca positions with DB:

```
For each Alpaca position:
  Case 1 — DB ACTIVE record exists:
    → Reload in-memory state from DB
    → Recalculate bars_held from entry_date
    → Cross-check direction (qty sign vs. DB direction field)
    → Load stop_price, target_price, highest_price from DB

  Case 2 — No DB record (orphaned Alpaca position):
    → Check for same-day existing trade (partial exit remnant)
    → If found: reuse trade, recompute partial P&L from Order ledger
                (recompute_partial_pnl(db, trade_id, entry_price, direction))
    → If not found: create synthetic ACTIVE trade, derive stops from signal

For each DB ACTIVE record with no Alpaca position:
  → Mark FORCE_CLOSED_NO_POSITION (ghost cleanup)
  → Calculate P&L using latest available price
```

### 6.2 Entry Execution

```
Entry type:
  swing    → limit order (0.1% below ask for long; 0.1% above bid for short)
  intraday → market order

PENDING_FILL record written BEFORE order placement.
  (Ensures trade is recorded even if crash occurs between order and fill)

On fill confirmation:
  PENDING_FILL → ACTIVE
  Slippage calculation (direction-aware):
    Long:  slippage_bps = (filled - intended) / intended × 10,000
    Short: slippage_bps = (intended - filled) / intended × 10,000
    (Short: lower fill price = worse for us = positive slippage)
  active_positions[symbol] populated
```

### 6.3 Position State (In-Memory Dict)

```python
active_positions[symbol] = {
    "entry_price":      float,
    "stop_price":       float,     # updated: trailing stop, breakeven after partial
    "target_price":     float,
    "highest_price":    float,     # updated each bar (for trailing stops)
    "atr":              float,
    "bars_held":        int,       # incremented on each bar
    "trade_id":         int,       # DB primary key
    "trade_type":       "swing" | "intraday",
    "entry_date":       date,
    "direction":        "BUY" | "SELL_SHORT",
    "proposal_uuid":    str,
    "shares":           int,       # quantity — "shares" key (not "quantity")
    "_partial_exited":  bool,      # guard: fire partial exit only once
    "_partial_pnl":     float,     # accumulated from Order ledger
}
```

### 6.4 Exit Logic

Exit conditions checked on each tick/bar:

```
1. Stop hit:
   Long:  current_price ≤ stop_price
   Short: current_price ≥ stop_price

2. Profit target:
   Long:  current_price ≥ target_price
   Short: current_price ≤ target_price

3. Partial exit trigger (fires once):
   Long:  current_price ≥ entry + 1×ATR AND NOT _partial_exited
   Short: current_price ≤ entry − 1×ATR AND NOT _partial_exited
   → Exit 50% of position; move stop to breakeven
   → Record PARTIAL_EXIT Order in ledger

4. Trailing stop:
   Long:  new_stop = highest_price × (1 − trail_pct)
          If new_stop > stop_price → update stop_price
   Short: new_stop = lowest_price × (1 + trail_pct)
          If new_stop < stop_price → update stop_price

5. Time exit:
   Swing:    bars_held ≥ max_hold_bars (config: strategy.max_hold_bars)
   Intraday: any open position at 15:55 ET

6. PM review exit:
   PM sends EXIT/HOLD/EXTEND signal on 30-min review cycle
   Trader honours EXIT signals immediately

7. News exit (intraday):
   Phase 53 news monitor detects adverse event → PM emits NEWS_EXIT
```

### 6.5 P&L Architecture — Order Ledger Authority

**Problem solved:** On restart, if a position had partial exits, `trade.pnl` was zeroed out because the partial P&L was only in memory. The next close would calculate final P&L from scratch, ignoring the already-realized partial gains.

**Solution (Audit #17):** `recompute_partial_pnl(db, trade_id, entry_price, direction)` in `app/database/models.py`:

```python
def recompute_partial_pnl(db, trade_id, entry_price, direction):
    orders = db.query(Order).filter(
        Order.trade_id == trade_id,
        Order.order_type == "PARTIAL_EXIT",
        Order.status == "FILLED",
    ).all()
    total = 0.0
    for o in orders:
        if o.filled_price and o.filled_qty:
            leg = (entry_price - o.filled_price) * o.filled_qty \
                  if direction == "SELL_SHORT" \
                  else (o.filled_price - entry_price) * o.filled_qty
            total += leg
    return total
```

The Order table is immutable once filled. This function is called in:
- Startup reconciler (reactivation path)
- Trader `_execute_exit` (force-close path where `_partial_pnl` may be 0)

Final P&L on close: `total_pnl = final_leg_pnl + recompute_partial_pnl(...)`

---

## 7. DATA LAYER

### 7.1 Data Sources

| Source | Provider | Data Types | Cache |
|--------|----------|------------|-------|
| Daily OHLCV bars | Alpaca | Historical + live; swing features | Session |
| Intraday 5-min bars | Alpaca | Live market hours | Session |
| Earnings history | FMP | Quarterly EPS actual/estimated | 24h in-process |
| Analyst grades | FMP | Upgrade/downgrade history | 24h |
| Fundamental ratios | FMP | Margins, PE, growth | 24h |
| Company news | Finnhub | Articles for NIS Tier 2 | 1h per symbol |
| Economic calendar | Finnhub | FOMC, NFP, CPI etc. | Session |
| Polygon.io | Polygon | News sentiment (historical swing) | 1h / 290s |
| Account state | Alpaca | Cash, equity, positions | Per-request |
| Quotes (live) | Alpaca/IEX | Bid-ask spread, mid | Per-request |

### 7.2 Point-in-Time Safety

Walk-forward and live trading both use `get_earnings_features_at(symbol, as_of)` which filters:
```python
# Only returns earnings records where report_date <= as_of
# Prevents look-ahead bias in backtesting
```

All historical news features via `fetch_news_historical(symbol, as_of, days_back=14)` are similarly PIT-safe.

### 7.3 Key DB Tables

```
Trade          — one row per position lifecycle (PENDING_FILL → ACTIVE → CLOSED)
Order          — one row per market order (entry, exit, partial exit); immutable once filled
ProposalLog    — one row per PM proposal (PENDING → SENT → APPROVED/REJECTED)
RiskMetric     — daily P&L snapshots for RM drawdown calculation
MacroSignalCache — NIS Tier 1 (day-level macro context, one row per day)
NewsSignalCache  — NIS Tier 2 (per-symbol, TTL 1h)
AgentConfig    — key/value store for all runtime configuration (DB-backed)
```

---

## 8. LONG/SHORT DESIGN RATIONALE

### 8.1 Why L/S Instead of Long-Only

Four independent LLM reviews (Claude, ChatGPT, Gemini, DeepSeek) converged on the same diagnosis after 9 failed LambdaRank walk-forward runs: long-only cross-sectional ranking on small-cap momentum had no stable edge in a bear-market-inclusive 6-year window. Key problems:
1. **Label contamination**: long-only model saw 2022 bear market as uniform negative label; couldn't distinguish good short setups
2. **Regime dependency**: momentum works in bull markets, reverses in bear markets; long-only amplifies this
3. **Opportunity cost**: leaving 40%+ of informative signals (negative surprises, fundamental deterioration) unused

**Decision**: Pivot to 40% net long L/S target with configurable gross exposure.

### 8.2 Current L/S Configuration

```
pm.ls_net_exposure_pct  = 0.40  (40% net long; NAV-relative)
pm.ls_top_n_long        = 20    (top 20 long candidates)
pm.ls_top_n_short       = 15    (top 15 short candidates)
pm.pead_enable_shorts   = true
pm.swing_selector       = pead_quality_short
```

Gross exposure math at max utilization:
- Long leg: ~100% of account (20 positions × 5% each)
- Short leg: ~75% of account (15 positions × 5% each)
- Net: ~25% long — slightly below 40% target at full utilization

In practice, risk gates (sector concentration, correlation, portfolio heat) limit actual exposure to well below maximum.

### 8.3 Short Position Mechanics

```
Entry:    SELL_SHORT order via Alpaca (borrows automatically for liquid names)
Stop:     entry × (1 + stop_pct) — above entry (reverse of long)
Target:   entry × (1 - target_pct) — below entry
Trailing: updates lowest_price; stop = lowest × (1 + trail_pct)
Partial:  fires at entry - 1×ATR; stop moves to breakeven (entry × 1.001)

P&L:      (entry_price - exit_price) × qty
Slippage: (intended_fill - actual_fill) / intended_fill × 10,000 bps
          (lower fill = worse for short = positive slippage)

Borrow cost: not currently modeled in live trading;
             walk-forward simulator deducts 0.5% annualized flat (conservative)
```

### 8.4 Net Exposure Monitoring (Gap / Enhancement)

The RM validates individual trades against sector concentration and correlation rules, but **does not yet enforce an explicit net exposure gate at the portfolio level**. The 40% net long target is a sizing intention, not a hard rule enforced per-proposal.

**Known gap**: If longs are fully deployed and shorts are empty (or vice versa), net exposure could drift well beyond ±(0.40 ± 0.15). The plan (MASTER_BACKLOG Phase 1) includes adding a net exposure gate to `validate_trade()` that rejects entries pushing net beyond ±55% or below ±25%.

---

## 9. CONFIGURATION REFERENCE

### All Config Keys (DB-backed, live-editable)

#### Portfolio Manager
```
pm.top_n_stocks              int    10     Max symbols per intraday scan
pm.min_confidence            float  0.55   Min model confidence to propose
pm.position_risk_pct         float  0.02   Base risk per position (2% of NAV)
pm.exit_threshold            float  0.35   PM review exit vote threshold
pm.swing_selector            str    pead_quality_short
pm.pead_enable_shorts        str    true
pm.ls_net_exposure_pct       float  0.40   Target net long exposure
pm.ls_top_n_long             int    20     Long universe size
pm.ls_top_n_short            int    15     Short universe size
```

#### Risk Manager
```
risk.max_position_size_pct        float  0.05   Max 5% of account per position
risk.max_sector_concentration_pct float  0.20   Max 20% in any single sector
risk.max_daily_loss_pct           float  0.02   Max 2% daily loss before lockout
risk.max_account_drawdown_pct     float  0.05   Max 5% peak-to-current drawdown
risk.max_open_positions           int    5      (currently overridden to 7 in DB)
risk.max_portfolio_heat_pct       float  0.06   Max 6% total portfolio heat
risk.normal_volatility_atr_ratio  float  0.02   ATR/price ratio for dynamic stop
risk.stop_loss_base_pct           float  0.02   Base stop loss percentage
risk.max_spread_pct               float  0.005  Max bid-ask spread (swing only)
risk.max_adtv_pct                 float  0.01   Max trade size as % of 20d ADTV
risk.max_correlation              float  0.75   Max 60d return correlation
risk.max_portfolio_beta           float  1.30   Max portfolio-level beta
risk.high_beta_threshold          float  1.20   Beta above this triggers check
risk.max_factor_concentration     float  0.60   Max sector as % of portfolio
```

#### Strategy / Execution
```
strategy.partial_exit_pct         float  0.50   % of position to exit at 1×ATR
strategy.max_hold_bars            int    30     Max bars before time exit
strategy.limit_order_offset_pct   float  0.001  Limit order offset (0.1%)
strategy.limit_order_requote_min  int    45     Minutes before requote attempt
strategy.limit_order_escalate_min int    90     Minutes before market-order fallback
```

---

## 10. KNOWN LIMITATIONS & OPEN QUESTIONS FOR LLM REVIEW

### 10.1 NIS Direction Alignment (High Priority)

NIS `action_policy = "block_entry"` fires when news is negative. For a SELL_SHORT proposal, negative news **validates** the thesis. The current implementation may incorrectly block short entries where negative news is the signal source.

**Question**: Should NIS Tier 2 distinguish between `direction_score < 0` for a BUY proposal (adverse) vs. a SELL_SHORT proposal (confirming)?

### 10.2 Net Exposure Gate Missing

PM constructs proposals with L/S direction, but RM has no portfolio-level net exposure gate. Net long could drift significantly if longs fill and shorts don't (or vice versa, due to borrow unavailability, execution timing).

**Question**: Where in the pipeline should net exposure be enforced — RM pre-trade gate, or PM pre-proposal filter?

### 10.3 PEAD Hold Duration vs. Stop Logic

PEAD signals are annotated `max_hold_days = 5`. The stop/target logic in Trader uses the same ATR-based regime as factor portfolio signals. A 5-day PEAD trade has a fundamentally different exit catalyst (drift decay) than a 20-day momentum trade.

**Question**: Should PEAD positions have different exit logic — e.g., exit on day 5 regardless of price, rather than waiting for ATR-based stop/target?

### 10.4 Borrow Cost Not Modeled in Live Trading

The walk-forward simulator deducts 0.5% annualized borrow cost. Live trading (Alpaca paper) does not deduct borrow. For liquid large-cap names this is minimal; for high-short-interest targets it can be material.

**Question**: Should live trading track estimated borrow cost and include it in P&L reporting?

### 10.5 P0 Walk-Forward Bugs (From MASTER_BACKLOG)

Entry price simulation still uses `prev_close × 1.001` rather than next-session open. Stop-out check uses daily close rather than intraday low. These bugs mean all historical WF results are optimistic (fewer stops triggered, entry prices biased toward prior close).

**Status**: These are documented in MASTER_BACKLOG as P0.1 and P0.2. Not yet fixed.

**Impact**: WF Sharpe figures should be treated as directional indicators, not precise estimates.

### 10.6 No Survivorship Bias Correction

Walk-forward universe uses current S&P/Russell constituents. Approximately 15% of historical performance benefit may be attributable to survivorship bias (audit done 2026-05-18, findings in `docs/survivorship_audit_20260518.md`).

### 10.7 Factor IC Not Yet Validated

`scripts/compute_factor_ic.py` (from MASTER_BACKLOG P0.3) has not been run. We do not yet have empirical IC values for the composite factor score over the 6-year walk-forward window.

Pass criteria (from plan): mean IC ≥ 0.02, t-stat ≥ 2.0.

---

## 11. LIVE PROMOTION CRITERIA

Current paper trading targets before live promotion:
- 3 calendar months paper trading
- Annualized Sharpe ≥ 0.50
- Max drawdown ≤ 15%
- Max single position ≤ 8% NAV
- Live-vs-sim shortfall < 30 bps/day median

---

## 12. FILE MAP

| File | Role |
|------|------|
| `app/agents/portfolio_manager.py` | Daily scheduling, selector routing, proposal construction, NIS integration |
| `app/agents/risk_manager.py` | 11-point validation, RiskLimits TTL cache, account state, correlation/beta |
| `app/agents/risk_rules.py` | RiskLimits dataclass (14 fields), all validate_* functions |
| `app/agents/trader.py` | Entry/exit execution, partial exits, P&L, startup reconciliation |
| `app/agents/compliance.py` | PDT, wash sale, T+1 settlement, symbol halt list |
| `app/ml/pead_scorer.py` | PEAD signal scorer (EPS surprise ±5%) |
| `app/ml/factor_scorer.py` | Composite factor scorer + QualityShortScorer |
| `app/news/intelligence_service.py` | NIS public API (Tier 1 + Tier 2) |
| `app/news/news_monitor.py` | Intraday news event monitoring (Phase 53) |
| `app/strategy/premarket_intel.py` | Regime detection, macro context |
| `app/strategy/macro_calendar.py` | Finnhub economic calendar gate |
| `app/strategy/portfolio_heat.py` | Portfolio heat calculation (direction-neutral) |
| `app/backtesting/agent_simulator.py` | Walk-forward simulator (direction-aware, signed sectors) |
| `app/startup_reconciler.py` | Startup position reconciliation, PENDING_FILL resolution |
| `app/database/models.py` | DB schema + `recompute_partial_pnl()` (Order ledger authority) |
| `app/database/agent_config.py` | CONFIG_SCHEMA, get/set_agent_config |
| `app/data/fmp_provider.py` | FMP earnings features (PIT-safe) |
| `app/data/alpaca_provider.py` | Alpaca OHLCV + live quotes |
| `scripts/compute_factor_ic.py` | IC computation (pending, MASTER_BACKLOG P0.3) |
