# Alpha-v10 R0.3 + R0.4 — Measurement & Safety Foundation (shadow, 2026-06-21)

The R0 "Minimum-Viable-Safety" building blocks from the Portfolio-Brain roadmap, built autonomously
while IBKR pends. **All SHADOW / read-only — they control nothing** (no order path is wired; the
Alpaca adapter is structurally incapable of trading). Each: Opus design → adversarial deep-dive →
fix/iterate → tests. These are the substrate the R0.5 whole-book risk gate and R1 IBKR build will use.

## R0.3 — canonical instrument master + read-only broker abstraction
- **`app/live_trading/instrument_master.py`** — the venue-agnostic contract master: `instrument_id`
  ↔ each venue's broker symbol + static spec (asset class, multiplier, currency, cash-equivalent).
  Seeded with the LIVE Alpaca universe (10 trend ETFs + T-bill cash sleeve, verified) and a
  representative IBKR futures set as **placeholders flagged `verified=False`** (multipliers MUST be
  verify-on-connect in R1, per the P2 spec). Fail-closed: an unknown symbol → `None` (the gate
  treats it as un-sizable).
- **`app/live_trading/broker_adapter.py`** — the `BrokerAdapter` Protocol (read side: health /
  account / positions / normalize) + `AlpacaReadOnlyAdapter` wrapping the existing client. **No
  order methods exist, AND the wrapped client is held behind a read-only proxy** that raises on any
  non-whitelisted method (defense-in-depth: a caller cannot reach `submit_order` via the adapter).
  Normalizes Alpaca's account/positions into canonical `AccountState` / `CanonicalPosition`.

## R0.4 — consolidated book-state + reconciliation + kill-switch
- **`app/live_trading/book_state.py`** — assembles ONE cross-venue `BookState` from the read-only
  adapters: positions, per-venue accounts, gross/net, and the **netted FACTOR-EXPOSURE vector**
  (equity beta, rates DV01, USD, commodity, vol) that catches stacked SPY-on-Alpaca + ES-on-IBKR
  beta the per-trade RM never could. Exposures use **full signed notional `qty·price·mult`** (NOT
  the broker `market_value` — IBKR reports futures MV as daily P&L ≈ 0, which would understate
  stacked beta to ~0). Cash-equivalents excluded from gross; unmapped-factor + stale-price
  instruments flagged (fail-closed candidates). Factors carry per-key UNITS (never summed across).
- **`app/live_trading/reconciliation.py`** — reconciliation-before-trade, **keyed by
  `(venue, instrument_id)`** (an id-only key could silently drop a cross-venue position and report a
  false MATCH — the one failure this gate must never have). **FAIL-CLOSED** on any material position
  break (DB-only, broker-only, qty mismatch beyond fractional-share tolerance) or cash beyond
  `max($100, 5bps·NAV)`; pending orders accounted for; "cash NOT checked" surfaced if cash inputs
  are omitted. (Shadow: computes the verdict; gating the live path is R0.5.)
- **`app/live_trading/kill_switch_state.py`** — the cross-venue kill-switch state machine
  (NORMAL → HALT_NEW_RISK → CANCEL_ONLY → FLATTEN_NON_CORE → FLATTEN_ALL → MANUAL_LOCK). Risk-
  increasing orders only in NORMAL; **auto triggers (dead-man / reconciliation-fail) escalate to
  HALT_NEW_RISK and are HARD-CAPPED at CANCEL_ONLY — a flaky watchdog can never auto-liquidate**;
  de-escalation / any FLATTEN_* / leaving MANUAL_LOCK requires an explicit human.

## Deep-dive fixes applied (the review found loss-vectors-when-wired; all fixed in shadow)
- **[BLOCKER] reconciliation now keyed by `(venue, instrument_id)`** (was id-only → cross-venue
  false-MATCH risk).
- **[BLOCKER] book-state factor exposure uses full notional, not broker MV** (IBKR futures-MV trap).
- **[MAJOR] kill-switch auto-escalation capped at CANCEL_ONLY** (no auto-flatten).
- **[MAJOR] read-only client proxy** so the wrapped Alpaca client can't be used to trade.
- Plus: gross/net/factors derive from one quantity (no desync); stale-price + unmapped flags;
  `settled_cash` no longer aliased to total cash; fractional-share `qty_tol`; factor-unit tags.

## Not yet (R0.5 / R1, documented for the next builder)
Per-venue cash reconciliation calls; thread-safe/persisted (Postgres `control_flags`) kill-switch
for the concurrent order path; IBKR adapter + verify-on-connect of the futures specs; wiring the
reconciliation + kill-switch to actually GATE the live order path (R0.5). The read-only modules are
intentionally inert until then.

## Tests
`tests/test_instrument_master_adapter.py` (7) · `tests/test_book_state_reconciliation_killswitch.py`
(16). All green; flake8 `app/` clean. **Live book UNCHANGED.**
