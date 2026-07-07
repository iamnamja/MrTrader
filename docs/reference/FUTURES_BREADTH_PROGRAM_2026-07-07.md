# Futures Breadth Program (Alpha-v10 R1.3) — 2026-07-07

**Why this exists:** the R1.3 before-live gate (`scripts/run_futures_16market_revalidation.py`, DECISIONS
2026-07-07) found the carry+xSMOM book's diversification value (Track-B t vs live ETF-trend) **collapses
2.61 → −0.20** on the 16 IBKR markets in `instrument_master`, while the full 76-market book works (t=2.61).
The edge is **breadth-driven**; 16 markets is a self-imposed limit ("a small representative set"), not an
IBKR limit. This program answers, phase by phase, **whether expanding the tradeable universe restores a
DEPLOYABLE, diversifying futures book — or whether futures-live should be shelved.** Every phase is
Opus-reviewed; the execution stack (roll policy/monitor/holiday calendar/hybrid roll/live signal) is
already built + inert and does NOT change here — this is a DATA/UNIVERSE + validation program.

**Hard rule:** no futures capital until FB3 passes (edge survives on the ACTUAL IBKR-tradeable set) AND
FB4 clears (roll/delivery safe on every new market). ETF-trend + cash stays the sole live book; R1.1/R1.2
(ETF→IBKR cutover) is UNAFFECTED and proceeds independently.

---

## Phases

### FB0 — Breadth-sensitivity analysis ✅ **DONE (2026-07-07)** — `scripts/run_futures_breadth_sweep.py`
Reused the gate machinery on the full-76 Norgate panels (load once; subset columns; momentum RECOMPUTED
per subset). Independent Opus review: **methodologically sound** (anchor t(IBKR-16) reconciles EXACTLY to
the gate's −0.20; full-76 = +2.61). Findings:
- **Breadth curve (random subsets, the selection-free estimate):** E[t] rises with #markets — k=16 +0.30,
  24 +0.68, 32 +1.04, 40 +1.42, **48 +2.04**, 56 +1.96, 64 +2.27, 76 +2.61 (σ ~0.6). → **~48 markets
  (48 ± 8) to reliably clear t>2 — roughly TRIPLING from 16.**
- **Bucket direction (16 + one family, Δt):** **commodities diversify most** — grains +1.44, energy +1.25,
  softs +0.92, livestock +0.64; **more financials/FX HURT-or-add-nothing** — fx −1.12, intl_rates −0.54,
  us_rates +0.25, metals −0.06 (the 16 is already financial-heavy → the missing breadth is COMMODITY).
- **Greedy forward-selection was OVERFIT and removed** (its "clears t>2 at n=19" picked intl-equity/
  intl-rates markets the bucket analysis rates harmful — a direct overfit tell; never a target).
- **Caveats that make deliverable t THINNER than shown (Opus):** `liquid_universe` survivorship inflates
  ALL t (incl. 2.61, itself only ~1 SE above the bar; null_zoo DSR already "borderline"); and the flat
  3bps roll cost UNDERSTATES the real roll/execution cost of the commodity complex that the expansion
  most depends on. Track-B t is noisy (~0.5–1.0 SE) → the "~48" is a band, not a threshold.

**Decision gate — RECOMMENDATION: DEFER / SHELVE (FB-SHELVE), do not expand.** Restoring the edge needs a
~3× expansion (~48 markets) concentrated in the hardest-to-trade commodity complex (many foreign/specialty
contracts → IBKR permissions, margin, per-market roll/FND rules), for a book that is a MARGINAL, survivorship-
and-cost-inflated diversifier even at full breadth (t≈2.6, standalone SR 0.67). The operational cost/complexity
is disproportionate to the marginal, uncertain gain. **Owner call: FB-SHELVE (recommended) vs proceed FB1**
(only worth it if IBKR commodity execution proves genuinely cheap).

### FB1 — IBKR contract availability + cost audit
For the FB0-target markets, confirm IBKR actually offers them at a workable exchange / market-data
permission / margin / liquidity for a small paper→live account (many 76-universe markets are foreign —
Euro bonds, Asian indices, softs — needing specific permissions). **Deliverable:** the IBKR-tradeable
subset of the FB0 target + the gaps + any data-permission cost.

### FB2 — Expand `instrument_master` + verify-on-connect
Add the FB1-confirmed contracts (root / multiplier / exchange / currency / ibkr_symbol) to
`instrument_master`; run verify-on-connect (`reqContractDetails`) to validate each spec — the #1-killer
multiplier guard — exactly as the original 16 were verified (0 critical). **Deliverable:** expanded,
spec-verified futures universe.

### FB3 — Re-run the before-live gate on the expanded universe
Re-run the breadth gate on the ACTUAL expanded IBKR-tradeable set → Track-B t. **PASS (t > ~2) → futures-
live viable; FAIL → shelve.** This is the go/no-go for futures capital. **Deliverable:** pass/fail + the
deployable universe.

### FB4 — Roll / delivery safety on the expanded set
Validate `qualify_future` + the roll policy (FND rules, last-trade cap, hybrid OI-crossover) + the delivery
monitor across EVERY new market — new families have different delivery specs (softs, Euro rates), so the
per-market FND taxonomy must be extended + verified (the Opus roll-review flagged energy already).
**Deliverable:** verified roll/delivery safety for every tradeable market + updated FND rules.

### FB5 — Signal re-extraction + sizing on the expanded universe (SHADOW)
`current_target_weights` now spans the expanded tradeable set; re-validate the shadow signal. Address the
documented before-live signal refinements: **forward-target (un-shifted) extraction** (remove the ~1-day
PIT lag), **gross renormalisation** on the tradeable subset, and a **fresh data feed** (the Norgate mirror
lags). **Deliverable:** a correct, forward-looking shadow signal on the expanded universe.

### FB6 — Futures-live readiness gate (owner-present)
All before-live items cleared: edge re-validated (FB3), roll/delivery safe (FB4), signal refined (FB5),
PLUS the standing R1 prereqs — daemon subprocess decouple (R0.2 P4), `kill_switch_sm` → enforce, Gateway
Read-Only-API OFF, and the enforce-flip soak clean. **Deliverable:** go/no-go for tiny-live futures
(1–2 lots/market, instant rollback), then soak → scale.

### FB-SHELVE — (conditional) shelve futures-live
If FB0 says the required expansion isn't worth it, or FB3 fails: keep ETF-trend + cash as the live book,
leave futures in research, and redirect effort to other edges (equity audit, event-driven/PEAD, VRP re-work).
Record the verdict; the built execution stack stays as dormant, tested infrastructure.

---

## Status
- **FB0** — ✅ DONE (2026-07-07). Verdict: **~48 markets (3×) needed, commodity-first; marginal + cost-inflated
  even at full breadth → RECOMMEND FB-SHELVE.** Opus-reviewed (methodology sound).
- **FB1–FB6** — NOT STARTED; gated on the owner choosing to proceed over the FB-SHELVE recommendation.
- **FB-SHELVE** — the recommended path (awaiting owner confirmation).
