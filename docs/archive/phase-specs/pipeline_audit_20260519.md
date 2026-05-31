# MrTrader Walk-Forward Pipeline Audit — 2026-05-19

Scope: post-fix audit of `agent_simulator.py`, Phase H/H+ runners, short scorers,
PEAD scorer, factor scorer, and FMP provider. Three equity-accounting bugs
(MTM, short PMV, hardcoded borrow) were just fixed; this audit looks for what
remains.

---

## 1. CRITICAL BUGS

### C1. `portfolio.equity` still uses the buggy `position_market_value` everywhere it matters
**Location:** `app/backtesting/agent_simulator.py:109-111`, used at lines 282-283, 759, 914, 921.

The MTM fix only changed the `equity_by_date[day]` snapshot (line 435). Every
**live decision** in the loop still calls `portfolio.equity`, which routes
through `position_market_value` (line 89) — the very property the bug-fix
docstring says is "kept for backward compat (no MTM)". Concretely:

- **Peak equity tracking** (line 282) — `peak_equity` is updated from the
  bugged equity. Drawdown gate (`validate_account_drawdown`, line 779) compares
  current bugged equity against bugged peak. Net effect: drawdown is computed
  on cost-basis equity, not MTM equity. Bear-market drawdowns will be
  understated dramatically (cost basis stays flat while real equity falls),
  letting the simulator continue trading deep into drawdowns that the live
  RM would block.
- **Position sizing** (line 914, 921) — `size_position` and the
  `MAX_POSITION_SIZE_PCT` cap use bugged equity. For shorts, this is doubly
  wrong: `position_market_value` adds `entry_price * qty` for shorts (line 89
  treats them as positive longs). So opening a short *inflates* the equity
  used to size the next position, allowing levered short stacking.
- **RM `equity` argument** (line 759) — sector concentration, position size,
  and daily-loss rules all see bugged equity.

**Impact:** Materially distorts results. Long-only longs in an uptrend
won't notice (cost basis ≈ MTM after entry). But: (a) short-heavy strategies
(Phase H A/B/E/F + ABCombined, CombinedLS, factor L/S) all inflate equity
when shorts open; (b) bear-fold drawdown gates fail to fire; (c)
peak-equity-relative DD is meaningless once positions move.

**Fix:** Make `equity` MTM-aware. Either:
1. Cache `today_closes` on the portfolio at top of each loop iteration and
   make `equity` a method that takes nothing (closing over the cache); or
2. Compute MTM equity once per day at the top of the loop (after exits
   process, before scoring) and store on `portfolio._equity_cached`; use that
   for peak update, RM, sizing.

This is the most important remaining bug. The earlier "fix" is cosmetic — it
only corrected the *output* equity curve, not the *decisions* that produced
it.

### C2. Short-entry cash accounting double-counts margin
**Location:** `_process_entries` lines 938-940.

```python
portfolio.cash += trade_cost - tx_cost   # receive short proceeds
portfolio.cash -= trade_cost             # post margin = notional
```

Net effect: cash decreases by `tx_cost`. That's the same cash flow as a long
(`cash -= trade_cost + tx_cost` minus the `trade_cost` of the long position
that would otherwise be debited)... but the long has `entry_price*qty` of
asset value to offset; the short has nothing on the asset side until
`equity_mtm` adds `(entry - close) * qty`.

So on day 0 immediately after a short opens:
- `cash` = starting - tx_cost (close enough to starting)
- MTM PMV contribution = `(entry - close_today) * qty` ≈ 0 (close ≈ entry day-of)
- equity ≈ starting - tx_cost ✓

That's actually correct for *MTM* equity. **But** `position_market_value`
(the property used for sizing/RM/peak) still adds `entry_price * qty`,
which means right after opening a short, bugged equity =
`starting - tx_cost + entry*qty` — i.e. equity has roughly doubled by the
size of the short. This is the same root cause as C1, but the
under-collateralised short-margin treatment makes it especially severe in
factor L/S and ABCombined modes.

**Fix:** Combined with C1 — once `.equity` is MTM, this resolves.

### C3. `equity_by_date` is sparse — Sharpe denominator is wrong on no-bar days
**Location:** `_bars_on` (line 1211) returns None when the symbol has no bar
on `day`; `_today_closes` (line 428-434) only collects symbols that DO have
a bar. Then `equity_mtm` falls back to `entry_price` for missing closes
(line 101). So on days with a market holiday (no SPY bar but indices still
have a date in `all_days`), or any single-symbol gap day, equity is computed
with stale entry prices for the affected names.

More subtly: `trading_days` (line 235) is the **union** of all symbols'
indices. A symbol that doesn't trade on day D contributes nothing, but the
day is still iterated. Daily returns are computed from `eq_vals` (line
1115-1118) without checking whether `eq_vals[i]` and `eq_vals[i-1]` come
from comparable trading sessions. Zero-return placeholder days inflate the
denominator and depress Sharpe.

**Impact:** Sharpe systematically understated, roughly proportional to
fraction of days with partial bars. With Russell 1000 universe and ~6-year
window, this is small (<5%) but non-zero.

**Fix:** Restrict `trading_days` to days where SPY has a bar (canonical
NYSE session), and require `today_close` lookup to fall back to the most
recent prior close, not entry_price.

---

## 2. MODERATE ISSUES

### M1. Cross-fold cache contamination via module-level FMP caches
**Location:** `app/data/fmp_provider.py:33-35` — `_earnings_cache`,
`_grades_cache`, `_institutional_cache` are module-globals with no
fold-aware keying.

The data itself is PIT-filtered downstream (good — see `get_earnings_features_at`
line 110-113 filtering by `as_of`). But the **cache returns the full record
list** as fetched **at the time of the first call**. If walk-forward fold 1
fetches FMP earnings for AAPL in Jan 2020, and the live API returns records
through 2026-05, fold 5 testing in 2024 will *re-use* those records, which
is fine because they're PIT-filtered. **But** if a record was *revised*
post-publication (e.g. restatement, FMP backfill), the cached version is
the post-revision one — i.e. it includes information that wasn't yet
publicly available on the original report date.

**Impact:** Mild forward-look on `epsActual` for restated quarters. PEAD
and QualityShort flags affected. Probably <5bps Sharpe drift but
unverifiable.

**Fix:** Use the FMP "as-of" endpoint where available, or persist a
fold-construction-time snapshot to disk.

### M2. PEAD scorer ignores `as_of` business-day logic
**Location:** `app/ml/pead_scorer.py:91-99`. `fmp_days_since_earnings` is
**calendar** days (`fmp_provider.py:122-125`), but `max_days_after = 3` is
treated as if it's trading days. Earnings reported Thursday after-close;
days_since on Friday = 1, Monday = 4 → blocked. So Monday-after-earnings
trades (the canonical PEAD entry) are systematically excluded.

**Impact:** PEAD trade count understated, signal underutilized. Likely
*conservative* (no false positives) but biases the Sharpe estimate
downward for the PEAD scorer.

**Fix:** Use trading-day delta or set `max_days_after=5`.

### M3. Sharpe annualization assumes 252 daily returns when series may be shorter
**Location:** `_compute_result` line 1119-1122. `daily_rets` is built from
`equity_by_date.items()` (sorted). If a fold has, say, 60 calendar days but
the equity series has only 40 entries (sparse), `_sharpe(..., 252)` still
annualizes with √252. Result is correct *if* each entry represents one
trading day; given C3's mixed sparse/dense behaviour, this is borderline.

**Impact:** Small. Combined with C3.

### M4. Sharpe is averaged across folds (not pooled or weighted)
**Location:** `WalkForwardReport.avg_sharpe` line 206-207: `np.mean(fold sharpes)`.

Folds have unequal trade counts and unequal length (last fold may be
truncated by `end_all`, see line 647 `min(raw_test_end_dt, end_all)`). Simple
mean over-weights short, low-N folds. The accepted convention for
walk-forward is pooled-returns Sharpe (concatenate daily returns across
folds, then compute one Sharpe) or N-weighted.

**Impact:** Reported `avg_sharpe` is noisier than the true OOS Sharpe.
Headline metrics may pass/fail the 0.80 gate by random fold weighting.

**Fix:** Add `pooled_sharpe` computed from concatenated daily-return series.

### M5. Trader trend filter is computed on `bars_yesterday`, but entry is at today's open
**Location:** `_process_entries` line 841 + `_trader_signal` line 666-744.

`_trader_signal` is called with `bars_yesterday`; stops/targets are
percent-offsets of yesterday's close. But the actual entry happens at
`today_bar.open` (line 846). Stops therefore aren't calibrated to entry —
e.g. on a 3% gap up, the configured 2% stop is already underwater. The
no-chase filter (line 850-867) catches gross cases but a 1-2% gap easily
passes.

**Impact:** Stop distance underestimated on gap-up days → stops fire more
than ATR formula intended. Likely <0.1 Sharpe bias but worth verifying.

**Fix:** Re-anchor stops on `entry_price`, not yesterday's close. The
target/stop dollar offset should be recomputed at entry.

### M6. `_close_position` margin return uses entry_price
**Location:** Line 1067-1069. On short close:
```python
portfolio.cash += pos.entry_price * pos.quantity + gross_pnl - tx_cost
```
`gross_pnl = (entry - exit) * qty`, so this expands to
`cash += (2*entry - exit)*qty - tx_cost`. Sanity check: started with margin
of `entry*qty` debited; should return `entry*qty - (exit-entry)*qty -
tx_cost = (2*entry - exit)*qty - tx_cost`. ✓ algebraically correct.

But: when the short went **against** us and exit > 2*entry (price more than
doubled), cash *decreases* — correct in principle, but the simulator never
margin-called. Borrow cost continues to accrue at `entry*qty * rate/252`
(line 988), which understates margin once the position is well underwater.

**Impact:** Tail-risk understated on runaway shorts. Materially affects
fold-2 (2022) and any sustained squeeze regime.

**Fix:** Borrow cost should be on `current_notional = today_close * qty`,
not `entry_price * qty`.

### M7. PIT universe filter uses Russell 1000 endpoints, not daily membership
**Location:** `walkforward_tier3.py:680-681` — `_pit_union("russell1000",
tr_start, te_end)` builds the union of members at fold start AND end. So a
stock added to R1000 *during* the test fold is in the universe from day 1
of the test — mild look-ahead.

**Impact:** Very small (R1000 adds are continuous, not concentrated). But
formally a PIT violation. Documented in code as "captures mid-fold
adds/removes" (intentional choice for survivorship correction) — but the
correct fix is daily PIT membership lookup, not union.

---

## 3. MINOR / DEFENSIVE

### N1. `len(daily_rets) < 2` fallback uses per-trade pnl_pct as if they were daily
**Location:** Line 1119: `ret_series = daily_rets if len >= 2 else
[t.pnl_pct for t in accepted_trades]`. If a fold has only 1 equity snapshot,
Sharpe is computed on per-trade returns annualized as if daily. This is
catastrophically wrong (over-annualization by √252 on multi-day returns) —
but only triggers in degenerate folds with ≤1 trading day. Worth removing
the fallback in favor of `0.0`.

### N2. RSI computation uses simple mean, not Wilder's smoothing
**Location:** Line 690-696. Standard RSI uses Wilder EMA. Simple mean is
fine for a threshold check but mildly off from the model's training-time
RSI if it used Wilder's. Check `FeatureEngineer` for consistency.

### N3. `_bars_on` returns the first matching row
**Location:** Line 1217. Daily bars should be unique per day; if there are
duplicate timestamps (data error) only the first is used. Add an assertion.

### N4. `equity_curve` is sorted by date but `monthly_pnl` uses entry_date
**Location:** Line 1142. PnL booked on entry month, not exit month —
unconventional. Most performance attribution uses realization (exit) date.

### N5. The opportunity-score gate (line 365-406) is a long-only filter applied to L/S strategies
The score includes `ma_score` (SPY > MA20) which biases against bear-friendly
short books. When `use_opportunity_score=True` is combined with
`factor_scorer=None` (it's intentionally gated off when a scorer is present —
good), but mixing modes via `_skip_entries` blocks shorts during bear regime
exactly when they should be active. Phase H+ runs use `no_prefilters=True`
and don't enable `use_opportunity_score`, so OK in practice — but a footgun.

### N6. FMP grade classifier maps to "init"/"maintain" by string heuristics
**Location:** `fmp_provider.py:307-327`. `_classify_action` is best-effort
text matching; edge cases (e.g. "Buy → Buy" with `action_hint=""` → returns
"maintain") quietly drop signal. Low impact, but the analyst scorer's net
momentum will be undercounted.

### N7. Short technical-filter bypass
**Location:** Line 886. Comment says "longs only; shorts bypass technical
filter". So short entries skip the EMA-200/RSI/volume checks that gate
longs. That's design, not a bug — but it means short books are denser and
faster-turnover than long books. Worth verifying turnover/cost analysis
weights this asymmetry.

---

## 4. CONFIRMED CORRECT

- **Fold boundaries** (`walkforward_tier3.py:625-648`) — non-overlapping
  segments, `purge_days=10` between train_end and test_start, `embargo_days`
  defaulting to purge applied after test_end. Train/test cleanly separated
  for swing labels (5-day horizon).
- **`_bars_up_to(exclude_today=True)`** (line 1204-1209) — correctly uses
  `<`, not `≤`. PM features computed on yesterday's bars.
- **Short P&L sign and exit logic** (`_process_exits` line 1020-1037,
  `_close_position` line 1064-1066) — direction-conditional check_exit;
  short stop is `today_high >= stop_price` (above entry); target is
  `today_low <= target_price` (below entry); `gross_pnl = (entry - exit) *
  qty`. All correct.
- **Intrabar stop/target detection** (line 1010-1019, 1024-1031) — checked
  against `today_low`/`today_high`, not close. Correctly captures
  intraday-touch fills, with realistic fill price at the stop/target level.
- **VIX extreme-regime gate** (`short_scorers.py` `EXTREME_VIX = 40`) —
  scorers return [] above 40, no entries.
- **PEAD `as_of` filter** (`fmp_provider.py:110-114`) — earnings records
  filtered `<= as_of`. PEAD scorer respects `max_days_after`. PIT-clean
  within the limits of cache freshness (see M1).
- **Borrow cost** is now configurable (`short_borrow_rate_annual=0.05`,
  applied each bar via line 987-990). Daily accrual rate
  `entry*qty*rate/252` is correct (modulo M6 above re: marking to current
  notional).
- **MAX_HOLD override for PEAD hold-5** (line 1005-1007, 1032-1034) —
  threaded through both long and short exit paths.
- **Factor scorer regime gate** (`factor_scorer.py:280-288`) — sits in cash
  when SPY < MA200 OR VIX >= 30. Computed PIT from `closes[<=as_of]`.

---

## Overall verdict

**Not yet trustworthy for paper-trading go/no-go.**

The headline bug — `portfolio.equity` still routes through the un-fixed
`position_market_value` for **all decision logic** (sizing, RM gates, peak
DD, drawdown rule) — is more impactful than the original three bugs that
were "fixed". The MTM repair only corrected the *reported* equity curve;
the *simulator* still sizes positions, accumulates peak, and runs RM
checks against cost-basis equity that double-counts shorts. For long-only
factor mode this is benign; for everything in Phase H/H+ (which is the
entire current research campaign) it inflates short-side allocation and
suppresses drawdown gating.

**Minimum required fixes before trusting numbers:**

1. **C1**: Refactor `_PortfolioState.equity` to be MTM-based (cached at
   top of each iteration) — used by sizing, RM, peak.
2. **C3**: Anchor `trading_days` to a canonical session (SPY index), and
   forward-fill `today_closes` from the most recent prior bar.
3. **M6**: Borrow cost on current notional, not entry notional.

Recommended but not blocking: M2 (PEAD calendar-day off-by-one), M4
(pooled Sharpe alongside fold-mean), M7 (daily PIT membership). M5 and
N1 are low-cost cleanup.

After C1+C3+M6 are landed, re-run the full Phase H+ sweep and recompute
gates. If the rankings of A/B/D/E/F/G against the 0.80/-0.30 gate change
materially under the corrected accounting, hold paper-trading until the
sweep stabilizes. If rankings hold, the pipeline is reliable enough to
go.
