# Compound-and-Harden Program (CH) — 2026-07-07

**Why this is the active program:** the 5-LLM Comprehensive Review (unanimous; SSOT
`docs/reference/prompts/20260707_Comp_Review/SYNTHESIS_AND_DECISION.md`, DECISIONS 2026-07-07) concluded the
null second-edge result is the **market, not the method** → **stop the broad hunt; compound and harden the
one edge (ETF trend + cash).** This program executes that decision: make the single trend edge **antifragile
and condition-reactive** (the "adapt to market conditions" system the owner wanted, done the robust way),
instrument it with a **live-forward scorecard**, close the last live-path risk gap, and permit exactly **one**
pre-registered terminating search. It is edge-**hardening + operating**, not edge-discovery.

## Governing rules (from the synthesis — non-negotiable)
- **Every sizing change must BEAT constant-gross trend out-of-sample on CPCV, with the new parameters
  charged to the DSR trial count.** If it doesn't beat static, ship nothing — you've only added knobs.
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
- **CH0a — constant-gross trend BASELINE:** a frozen CPCV/backtest record of the current constant-gross
  trend book (Sharpe, tail, turnover, regime-conditional profile) = the bar every CH2 change must clear OOS.
- **CH0b — live-forward scorecard:** a periodic report attributing live-vs-backtest divergence — realized
  slippage, missed-rebalance impact, each governor's decisions, turnover, exposure, and the **static-vs-
  governed counterfactual** (what the book WOULD have done ungoverned). Bayesian into sizing over time.
  Reuses the existing paper track + notifier + dashboard. **Deliverable:** the baseline + the scorecard,
  emitting weekly. *This is the foundation for CH2/CH3/CH5.*

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
- **CH0** — 🔄 next (2026-07-07).
- CH1–CH5 — planned. CH4 is gated on a pre-committed moratorium; CH5's clock has started (enforce soak).
