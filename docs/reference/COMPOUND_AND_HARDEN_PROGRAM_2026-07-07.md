# Compound-and-Harden Program (CH) — 2026-07-07

**Why this is the active program:** the 5-LLM Comprehensive Review (unanimous; SSOT
`docs/reference/prompts/20260707_Comp_Review/SYNTHESIS_AND_DECISION.md`, DECISIONS 2026-07-07) concluded the
null second-edge result is the **market, not the method** → **stop the broad hunt; compound and harden the
one edge (ETF trend + cash).** This program executes that decision: make the single trend edge **antifragile
and condition-reactive** (the "adapt to market conditions" system the owner wanted, done the robust way),
instrument it with a **live-forward scorecard**, close the last live-path risk gap, and permit exactly **one**
pre-registered terminating search. It is edge-**hardening + operating**, not edge-discovery.

## Governing rules (from the synthesis — non-negotiable)
- **Every sizing change must (a) BEAT constant-gross trend out-of-sample on CPCV, with the new parameters
  charged to the DSR trial count, AND (b) NOT regress the BEAR regime-conditional Sharpe.** If it doesn't
  beat static, ship nothing — you've only added knobs. The dual gate is because the CH0a CPCV folds
  under-sample stress (no BEAR fold), so a mean_sharpe-only gate would reward benign-regime gains while
  degrading the tail antifragile sizing exists to protect (CH0a Opus review, 2026-07-07).
- **Continuous tilts, not hard regime switches** (switches are where whipsaw + overfitting live).
- **Shadow-first, always** — every new sizing/gate multiplier logs what it *would* do (applies 1.0) behind
  a default-off flag until it clears its gate + a shadow soak; then flip, instantly reversible.
- **DON'T:** buy alpha data · build a regime→strategy-selection layer (nothing to rotate into) · resurrect
  the futures book (marginal + cost-inflated even at full breadth per FB0) · override the live system
  discretionarily. Futures-live stays SHELVED (FB-SHELVE).

## Where we start FROM (already built — this program ASSEMBLES + adds 3 new pieces)
Live + enforce: whole-book risk gate (gross / net-equity-beta / notional / unmapped) + reconciliation-
before-trade. Built: regime detector (`regime_model_v9`), VIX-term crash governor (live), credit/curve
governors (parked/off), drawdown ladder (shadow/off), vol-targeting/inverse-vol sizing, macro-NIS sizing.
**The genuinely NEW work is CH2's three multipliers + CH0's scorecard + CH1's per-name gate.**

---

## Phases

### CH0 — Baseline + live-forward scorecard (measure before you change) ← **START**
You cannot gate anything without (a) the benchmark it must beat and (b) the harness to observe live behavior.
- **CH0a — constant-gross trend BASELINE:** ✅ DONE (2026-07-07). Frozen record in
  `docs/reference/ch0_trend_baseline.json` (`scripts/ch0_baseline.py`, pinned `end=2026-07-07` + data
  fingerprint for immutability; Opus-reviewed, no CRITICAL/MAJOR-correctness). **The bar CH2 must clear:**
  CPCV mean_sharpe **0.7009** (path-t 5.49, point_SR 0.70); standalone Sharpe 0.724 / maxDD −13.9% /
  Calmar 0.49. **Regime-conditional Sharpe (the CH2 target profile): BULL +4.11 (26% days) / NEUTRAL
  +0.44 (45%) / BEAR −0.77 (29%)** — trend earns everything in bull/neutral and bleeds in bear, which is
  exactly the whipsaw/tail failure mode CH2a/CH2c attack. Fold-coverage is LOW (n_folds=8, no BEAR fold)
  → hence the dual gate above.
- **CH0b — live-forward scorecard:** ✅ CODE DONE (2026-07-07; accrues live from Mon 2026-07-13).
  EXTENDS the existing P1-4 intended-vs-actual harness (`app/live_trading/back_validation.py`) rather than
  building anew: now persists **each individual governor's multiplier** (crash / credit / ladder — was only
  the composite `overlay_mult`) + the **ungoverned counterfactual book** (all multipliers = 1.0) at every
  live rebalance (`trend_sleeve` stashes `ungoverned_weights`; idempotent SQLite migration). The report adds
  the **static-vs-governed counterfactual** (`governor_pnl = governed_cum − ungoverned_cum`, +ve = de-risk
  helped) + a **regime-conditional breakdown** (BULL/NEUTRAL/BEAR — SAME taxonomy as the CH0a baseline, so
  the live slice lines up with the CH2 gate) attributing WHERE governing helped or hurt. Opus-reviewed
  (SAFE-TO-MERGE, no CRITICAL/MAJOR; provably inert to live trades — the new fields are write-once summary
  fields read only by the scorecard; Monday capture verified no-silent-NULL). Emits via the existing weekly
  notifier. **Deliverable met:** baseline (CH0a) + scorecard (CH0b). *Foundation for CH2/CH3/CH5.* **The
  live counterfactual accrues weekly starting the first enforce rebalance (2026-07-13) — it cannot be
  backfilled, hence landed BEFORE the soak.**

### CH1 — Close the per-name correlation/heat gap on the live path (hardening, not an emergency)
The live trend+cash path already passes the whole-book gate (gross/beta/notional) + reconciliation in
enforce — but NOT the agent RiskManager's **per-name correlation / heat / concentration** checks (those
cover only the dead proposal-driven path). Wire those into the live path, **shadow → enforce** (same pattern
as the whole-book gate). **Deliverable:** a per-name correlation/heat gate on the live order path, fail-
closed, shadow-soaked then enforced. Independent of CH2 — can run in parallel.

### CH2 — Antifragile trend sizing: the THREE new pieces (SHADOW → gated)
The core build. Each is a **continuous multiplier** on trend gross, shadow-first, gated vs the CH0a baseline
(beat constant-gross OOS on CPCV, params → DSR), armed only after a shadow soak.
- **CH2a — trend-strength-conditioned gross:** size UP when the trend signal is broad + strong across the
  universe; size DOWN when weak/conflicting — directly attacks TSMOM's whipsaw failure mode (weak-trend
  regimes are where it bleeds).
- **CH2b — correlation-regime gross scaling:** cut gross when realized cross-sectional correlation → 1
  (every position becomes one bet; "diversification" is illusory) — a real-time exposure-honesty control.
- **CH2c — trending-vs-whipsaw-aware crash governor:** the existing VIX governor cuts on *stress* — but
  trend often MAKES money in *trending* crises (2008/2022). Distinguish "trending stress" (stay — it's
  paying) from "whipsaw stress" (cut): de-risk when vol is high **AND** trend signals are conflicting/
  reversing, not merely when vol is high. Stops the governor from cutting winning crisis-trends.
- **Composite rule:** all multipliers compose (× the existing floored governors), can only be armed
  individually behind flags, each with its own CPCV evidence. **Deliverable:** the 3 multipliers, each with
  a "beats constant-gross OOS" gate result; ship only the ones that pass; the rest stay documented + off.

### CH3 — Regime-conditional decomposition (DIAGNOSTIC — informs CH2, no new hunt)
Decompose, by frozen regime labels over the existing purged folds: (a) the LIVE trend edge (when does it
work / whipsaw — feeds CH2a/CH2c), and (b) the two PARKED collinear strategies (`sector_rotation`,
`credit_timing`) — is their 0.51/0.52 collinearity to trend **regime-specific**? This is the only analysis
that could rescue a keeper (a regime-gated version could diversify). **Deliverable:** a regime-conditional
report; if a parked strategy is a *conditional* diversifier, it becomes a CH4-style pre-registered candidate
— otherwise the question is closed. **No new signal search.**

### CH4 — The ONE terminating, pre-registered search: ranging-market MR sleeve
Runs ONLY if it does not delay CH0–CH3. The single well-specified conditional family with a real mechanism
(liquidity provision to overreaction) + a non-overfit regime filter + structural anti-correlation:
- **Regime filter = the COMPLEMENT of the existing trend signal** (low realized-vol + weak trend-strength) —
  reuses a signal we didn't overfit, and is anti-correlated to the live book by construction.
- **Pre-registered gates (all must pass):** off-regime Sharpe ≥ −0.10 (flat, not negative); a mandatory
  **1–5 day detection-LAG test** (re-run with the regime signal lagged — this kills most regime mirages);
  **Track-B residual-alpha vs the live trend book**; all regime params charged to DSR.
- **A written 12-month hunting moratorium goes into DECISIONS BEFORE the run.** If CH4 fails, the moratorium
  binds. *Note: the UNCONDITIONAL ETF mean-reversion was already KILLED (`etf_relative_value`, point_SR 0.026);
  this tests only the conditional form.* **Deliverable:** PASS (a genuine second-sleeve candidate → paper) or
  KILL (+ the moratorium). Terminating either way.

### CH5 — Live-forward accrual + pre-committed review (the compounding)
Run the hardened, instrumented single-edge book on live paper; the CH0b scorecard accrues the record and
updates sizing confidence. **Pre-committed review at 12 months** — revisit the hunt ONLY if the live trend
edge has held up as expected (if not, that's a *different* problem — decay — to diagnose, not more sleeves).
The enforce-flip soak (started 2026-07-07) already begins this clock. **Deliverable:** the accruing live
track record + the discipline of not re-hunting before the review date.

---

## Dependencies & ordering
CH0 (foundation) → CH1 (parallel, independent safety) → CH2 (needs CH0a baseline to gate) ← CH3 (cheap,
informs CH2) → CH4 (only if CH0–3 not delayed) → CH5 (continuous, starts now). Every code phase ships with
tests + an independent Opus deep-dive, per this project's discipline.

## Status
- **CH0a** — ✅ DONE (2026-07-07): constant-gross baseline frozen + Opus-reviewed (see CH0 above).
- **CH0b** — ✅ CODE DONE (2026-07-07): scorecard persists per-governor multipliers + ungoverned
  counterfactual + regime attribution; Opus-reviewed, inert to live trades. **Live counterfactual accrues
  weekly from the 2026-07-13 enforce rebalance** (can't be backfilled → landed first).
- **CH0** — ✅ complete (baseline + scorecard). **NEXT: CH1** (per-name correlation/heat gate on the live
  path, shadow→enforce) — independent of CH2, can start now.
- CH1–CH5 — planned. CH4 is gated on a pre-committed moratorium; CH5's clock has started (enforce soak).
