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

## R0.4b — consolidated risk-surface report (read-only, validated on LIVE Alpaca)
`scripts/run_book_state_report.py` runs the R0 measurement layer against the live broker account(s)
and prints the consolidated book + netted factor exposures. **Validated on the live account
2026-06-21:** NAV $101,409, gross $20,569 (20.3% of NAV; cap 80%), net equity-beta +$16.3k
(0.16× NAV; cap 1.00), USD +$4.3k (UUP), all 5 trend ETFs mapped, zero unmapped/stale. Read-only —
the adapter cannot trade.

## R0.5 — the whole-book risk gate (SHADOW-first; wired into the live trend rebalance)
`app/live_trading/whole_book_gate.py` evaluates a PROPOSED book against the risk-policy-v1 hard caps
(gross-ex-cash, net equity beta, single-instrument & book notional, unmapped fail-closed) and returns
an allow/block verdict. **Wired into `trend_sleeve.run_trend_rebalance` (the risk-bearing sleeve;
cash-sleeve trades are cash-equivalent → gate-exempt by design).** Modes via
`pm.whole_book_gate_mode` (default **`shadow`**):
- **shadow** — computes the proposed-book caps, **LOGS + emails (`whole_book_gate_breach`) what it
  WOULD block, but blocks nothing**; the rebalance proceeds exactly as today.
- **enforce** — a breach HOLDS the rebalance (fail-closed: a missed rebalance, never a bad trade).
- **off** — not evaluated.

**Shadow safety (deep-dive-verified):** the wiring is double-wrapped fail-safe — `shadow_gate_from_
intents` never raises (returns `allow=True, error` on any failure), the sleeve call is in its own
try/except, no input is mutated, and no broker/API call is made (the gate runs on the in-memory
positions + intents already fetched). **A totally broken gate is inert in shadow** — tomorrow's live
rebalance is untouched. Enforce-facing hardening already applied: missing/zero price → fail-closed
breach (no hidden false-allow); an enforce-mode gate wiring failure fails CLOSED (HOLD).

**Rollout:** run in shadow through ≥1 week of live rebalances (incl. the Mon 2026-06-22 trend
rebalance + first cash deploy), confirm it never spuriously flags, then flip `pm.whole_book_gate_mode`
→ `enforce` (a config set, no code change).

## Not yet (later R0.5 / R1, for the next builder)
Per-venue cash reconciliation calls; thread-safe/persisted (Postgres `control_flags`) kill-switch for
the concurrent order path; wiring the **reconciliation + kill-switch** (not just the risk gate) into
the live path; IBKR adapter + verify-on-connect. The read-only book-state/reconciliation/kill-switch
modules remain inert until then; only the whole-book gate is wired (shadow).

## Tests
`tests/test_instrument_master_adapter.py` (7) · `tests/test_book_state_reconciliation_killswitch.py`
(16) · `tests/test_whole_book_gate.py` (9). All green; flake8 `app/` clean. **Live book UNCHANGED
(gate in shadow — logs only).**
