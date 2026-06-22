# Panelist 4 — App Robustness / Infra Audit (raw output, Opus 4.8, 2026-06-22)

> Mandate: where is the system fragile; what defect could cause real loss or silent wrong-trading;
> what makes it materially more robust BEFORE IBKR capital. Repo-grounded. Brutal honesty.

---

# MrTrader Live-Trading Robustness Audit — Pre-IBKR / Pre-Enforce

The system is unusually disciplined for a solo operation (fail-closed sleeves, frozen risk policy,
read-only adapter, shadow-first gate). But several docs claims are **stronger than the code supports**,
and there are concrete loss/wrong-trade vectors that bite the moment the gate flips to enforce or IBKR
is wired. Ranked by blast radius × likelihood.

## 1. Top failure modes (ranked)

### FM-1 — The cash sleeve bypasses the whole-book gate entirely — HIGH × HIGH
`run_cash_rebalance` (`cash_sleeve.py`) places live `place_market_order` with **no** `whole_book_gate`
call; only `trend_sleeve.run_trend_rebalance` is wired. The roadmap's R0.5 acceptance ("nothing trades
the gate hasn't seen") is factually false today. Today bounded (T-bills), but a 2nd non-trend sleeve
inherits the hole silently. **Fix:** shared gate-helper called by every sleeve. (S)
> [RED-TEAM: mostly debunked as a *risk* — the cash sleeve only buys cash-equivalents excluded from
> gross by construction → immaterial. Don't gate it; the real items are FM-2/FM-3.]

### FM-2 — Reconciliation + new kill-switch state machine are built but wired to NOTHING — HIGH × MEDIUM
`reconciliation.py` and `kill_switch_state.py` are imported only by tests/scripts/docs — **zero live
imports.** The live path uses the legacy `startup_reconciler.reconcile()` (startup-only, never gates a
trade) and the legacy `kill_switch.is_active` flag. **There is no reconciliation-before-trade on the
live path** — if DB and broker disagree (uncommitted fill, manual close, partial), the weekly rebalance
trades anyway. This is the "2 a.m. blow-up" the roadmap names as #1, unguarded. **Fix:** wire
`reconciliation.reconcile(...)` into each sleeve's pre-execution path + feed `on_reconciliation_fail()`
→ live kill state. (M)

### FM-3 — Enforce-mode false-block from the cash-ETF mapping gap — MEDIUM × HIGH
Gate builds the proposed book from `all_positions`; `is_cash_equivalent` keys off `instrument_master`
(only SGOV/BIL/SHV). `cash_sleeve.CASH_ETFS` has 8. If `pm.cash_universe` is set to e.g. VGSH/USFR/BILS:
those count toward gross AND are unmapped → unconditional breach → in **enforce** the entire trend
rebalance fail-closes every week on a legal cash config. **Fix:** single source of truth — register all
`CASH_ETFS` in `instrument_master` + `book_state._FACTOR_MAP{}`; add a subset test. (S)

### FM-4 — The gate evaluates the wrong proposed book — MEDIUM × MEDIUM
`trend_sleeve.py` passes `approved` (post per-order-gate) intents, and only trend; if PEAD re-enables,
its orders route independently and the trend gate run won't include in-flight PEAD intents. Two sleeves
can each pass while jointly breaching net-beta/gross. **Fix:** when ≥2 sleeves live, gate the union of
pending intents (the R2 Constructor's job); until then enforce the one-risk-sleeve invariant. (M/S)

### FM-5 — Per-order commit-then-continue double-trade window — MEDIUM × LOW
Order places, then per-order `db.commit()` (good). If the broker order succeeds and the process dies
before commit, the position exists with no DB row. Startup adoption now routes it via `classify_sleeve`
→ trend (2026-06-22 fix), and `client_order_id=trend-{date}-{sym}` makes same-day retries idempotent —
but cross-restart resume isn't deterministic. Real fix = per-order lifecycle persistence. Low priority.

### FM-6 — `apply_risk_gate` fat-finger off-by-design on the per-name cap — LOW × MEDIUM
Blocks a buy if `cost > max_position_pct*nav*1.1 or cost > nav` → a single buy up to 27.5% of NAV passes.
The whole-book gate's 0.25 cap catches it — only in enforce. `compute_trend_deltas` already clamps to
the cap, so it's a defense-in-depth gap, not a live bug. **Fix:** tighten the multiplier or flip enforce.

### FM-7 — Macro-refresh on the rebalance thread — LOW × LOW
`_refresh_macro_history_bounded` (20s daemon join) called twice/rebalance → up to 40s stall + two
concurrent yfinance writers to the same parquet (credit governor off by default → one call today).
**Fix:** refresh macro once, pass the frame to both governors. (S)

## 2. Shadow-gate → enforce readiness
Shadow path is provably inert/fail-safe (good). Blockers before enforce: **A (FM-3)** cash-ETF mapping;
**B (FM-1)** cash sleeve ungated (acceptable if docs say so); **C (FM-2)** enforce gives a risk-cap
check but still NO reconciliation-before-trade → enforce ≠ safe-to-add-capital. Minimum safe enforce:
fix FM-3; one risk sleeve live; ≥1 real Monday rebalance + first cash deploy clean in shadow; then flip.

## 3. Pre-IBKR-capital checklist (and a correction to "R0 is terminal")
R0's *measurement* layer is real and good; R0's *control* layer (reconciliation gate, kill state
machine, dead-man, flatten) is **scaffolding not yet load-bearing.** Hard no-go gates before any IBKR
dollar: (1) reconciliation-before-trade wired + fail-closed on every order path, per-venue cash never
cross-counted (`ibkr_min_cash_reserve_frac` has zero readers); (2) whole-book gate in enforce covering
all order paths; (3) an **out-of-band flatten** that closes whatever the broker reports (doesn't exist);
(4) an **external dead-man watchdog** setting HALT on stale heartbeat (`dead_man_check` logic exists,
nothing calls it; no watchdog binary); (5) IBKR adapter behind `BrokerAdapter` with **verify-on-connect**
on every futures multiplier (`instrument_master._FUTURES` ships placeholder multipliers `verified=False`;
trading on an unverified multiplier e.g. 6J at 12.5M is catastrophic — a test must assert no order routes
for `verified=False`); (6) per-order idempotency + lifecycle persistence before futures.

## 4. Observability / monitoring gaps
No reconciliation alerting on the live path; gate-breach visibility is shadow/email-only and a wedged
gate is invisible (swallowed exception logs at `debug` then proceeds — promote to warning + notifier);
no heartbeat/staleness monitor for the daemon (`dead_man_check` uncalled). `back_validation.py`
intended-vs-actual TE is a genuine strength (trend-only, BUILDING). Highest-value add: alert on a
reconciliation break + a "gate did not run / errored" alert.

## 5. Data-quality / PIT integrity in the LIVE path — CLEAN
The `factor_scorer.py` PIT look-ahead is real but confined to the parked fundamentals sleeve (not live).
Both live governors filter to settled closes strictly before today (matches backtest shift(1)). Trend
sizing fails closed on missing SPY / <60% universe / NAV / data failure. No live-path look-ahead found.

## The 3 things I'd fix before flipping the gate to enforce or wiring IBKR
1. **Single source of truth for cash-equivalent tickers** (FM-3) — register all `CASH_ETFS`; add subset test.
2. **Wire reconciliation-before-trade into the live path + alert on breaks** (FM-2) — the real "#1 safety gate," currently inert.
3. **Route every sleeve through the whole-book gate via a shared helper** (FM-1) so "no order escapes the gate" is structural.
(Plus the IBKR-specific no-go items: out-of-band broker-only flatten + external dead-man — both nonexistent.)

## Over-engineering to avoid (respect the data-gating)
Don't build the Constructor / ERC / Ledoit-Wolf / stress-covariance stack; don't generalize the
tail-corr governor for a 1-sleeve book; no options risk logic / proxy-netting / golden-date replay CI /
LLM monitoring layer (all premature); don't flip the daemon to subprocess as part of the enforce flip.

**The single most dangerous gap: the distance between what the docs say R0 delivers and what the code
wires. Reconciliation-before-trade and the kill-switch state machine are written, tested, and not
connected. Close that before treating R0 as the IBKR no-go gate.**
