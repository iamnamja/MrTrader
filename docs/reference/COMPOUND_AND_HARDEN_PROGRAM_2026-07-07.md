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

### CH0 — Baseline + live-forward scorecard (measure before you change) — ✅ **DONE (2026-07-07)**
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
  helped) + a **regime-conditional breakdown** (BULL/NEUTRAL/BEAR — same label taxonomy as the CH0a
  baseline) attributing WHERE governing helped or hurt. *(Caveat: the live regime map thresholds VIX
  against a recent expanding window, not CH0a's deep-history one, so the live slice is APPROXIMATELY —
  converging, not identically — comparable to the frozen baseline profile; the CH2 decision gate itself runs
  on CPCV with the deep-history map, so this diagnostic mismatch doesn't affect the gate.)* Opus-reviewed
  (SAFE-TO-MERGE, no CRITICAL/MAJOR; provably inert to live trades — the new fields are write-once summary
  fields read only by the scorecard; Monday capture verified no-silent-NULL). Emits via the existing weekly
  notifier. **Deliverable met:** baseline (CH0a) + scorecard (CH0b). *Foundation for CH2/CH3/CH5.* **The
  live counterfactual accrues weekly starting the first enforce rebalance (2026-07-13) — it cannot be
  backfilled, hence landed BEFORE the soak.**

### CH1 — Close the per-name correlation/heat gap on the live path (hardening, not an emergency) — ✅ **SHADOW-LANDED (2026-07-07)**
The live trend+cash path already passes the whole-book gate (gross/beta/notional) + reconciliation in
enforce — but NOT the agent RiskManager's **per-name correlation / heat / concentration** checks (those
cover only the dead proposal-driven path). CH1 wires those into the live path, **shadow → enforce** (same
pattern as the whole-book gate). **Built:** `app/live_trading/per_name_gate.py` (mirrors `whole_book_gate.py`
— fail-safe, never raises; `pm.per_name_gate_mode` default **shadow**), wired into `run_trend_rebalance`
after the whole-book gate. Book-level metrics off data ALREADY in scope (no new I/O):
- **correlation** = NAV-weighted signed avg pairwise book corr (`prices_df.pct_change().corr()`) — a COARSE
  "the 10-name book collapsed to genuinely ONE bet" backstop, thresholded HIGH (0.90/0.95, above the
  structural ~0.85 equity-ETF corr) so it does NOT hold normal equity-led weeks; the nuanced continuous
  response is CH2b, not this gate.
- **concentration** = per-name cap `max(policy 0.25, configured max_position_pct)` (defense-in-depth clip-
  integrity; can't permanent-hold when `trend_max_position_pct` is raised).
- **heat** = proposed-book open-risk vs 6% (loose for the stopless trend book; backstop).
Deliberately did **NOT** reuse the RM's sector/factor-concentration check — the all-ETF universe maps to
`UNKNOWN` and would breach every week (permanent fail-close). Opus-reviewed (SAFE-TO-MERGE as shadow-default,
provably inert to live trades, no live-path bug). **⚠️ BEFORE ENFORCE:** the correlation threshold is
PROVISIONAL — the gate RECORDS `weighted_avg_book_corr` every week; calibrate the enforce threshold from the
observed shadow-soak distribution (and reconcile it with any `trend_max_position_pct` change) before flipping
`pm.per_name_gate_mode`→enforce. **Deliverable met:** a fail-closed per-name gate on the live order path,
shadow-soaking now. Independent of CH2 — ran in parallel.

### CH2 — Antifragile trend sizing: the THREE new pieces (SHADOW → gated) ← **IN PROGRESS**
The core build. Each is a **continuous multiplier** on trend gross, shadow-first, gated vs the CH0a baseline
by the **DUAL gate — (a) beat constant-gross CPCV mean_sharpe 0.7009 OOS with params → DSR, AND (b) don't
regress the BEAR regime-conditional Sharpe** (the folds under-sample stress) — armed only after a shadow soak.
**Shared gate harness built:** `app/research/ch2_sizing.py::gate_multiplier` (governed = base × m[t] through
the SAME `evaluate_sleeve`→CPCV path CH0a was frozen on + the BEAR prong via `regime_conditional_sharpe`).
- **CH2a — trend-strength-conditioned gross** — ❌ **KILLED (2026-07-08):** lever UP broad+clean trends /
  cut whipsaw, tested TWO ways (clarity + breadth) so it's not a one-measure strawman. All 4 pre-registered
  configs decisively WORSE (−0.07 to −0.19 mean_sharpe; primary Δ−0.194). Mechanism: the signals have ≈0
  forward-Sharpe correlation, so conditioning gross on them only injects a Var(m) noise penalty (SR loss
  monotonic in m-dispersion; ~82% noise / ~18% turnover). Registered `ch2a_trend_strength_gross` (KILLED).
  DECISIONS 2026-07-08.
- **CH2b — correlation-regime gross scaling** — ❌ **KILLED (2026-07-08):** cut gross when the held-book
  weighted correlation → 1. All 3 pre-registered configs FAIL every prong OOS (primary mean_sharpe 0.6607,
  Δ−0.040, improvement p 0.68 n.s., BEAR regresses −0.795); high held-book correlation is predominantly a
  strong equity-BULL signal (trend's best periods), not a crisis signal — correlation is direction-blind.
  Ship nothing. Registered `ch2b_correlation_gross` (KILLED). DECISIONS 2026-07-08. *Directly motivates CH2c
  (add the direction signal).*
- **CH2c — trending-vs-whipsaw-aware crash governor** — ❌ **KILLED (2026-07-08):** de-risk only when stressed
  AND trends are choppy (whipsaw = 1 − trend clarity), so a broad *trending* crash (2008/2022) is NOT cut.
  Hypothesis vindicated — it beats the plain VIX governor (+0.003, and pulls BEAR back from −0.883 to −0.791) —
  but vs constant-gross only +0.0006 (n.s., p 0.56) + still regresses BEAR → ship nothing. Registered
  `ch2c_whipsaw_governor` (KILLED). **⚠️ CH3 flag (scoped, NOT a verdict):** measured ONLY on the trend
  book's Sharpe + bear-regime Sharpe, the plain VIX governor (LIVE) doesn't beat static + worsens the bear
  tail — but this harness does NOT measure its actual mandate (portfolio drawdown/tail-insurance), which
  mechanically lowers Sharpe. A question for CH3 on the right objective, not "turn it off." DECISIONS 2026-07-08.
- **Composite rule:** all multipliers compose (× the existing floored governors), can only be armed
  individually behind flags, each with its own CPCV evidence. **Deliverable:** the 3 multipliers, each with
  a DUAL-gate result (beats constant-gross OOS mean_sharpe AND no BEAR-regime-Sharpe regression); ship only
  the ones that pass; the rest stay documented + off.
- **★ CH2 RESULT (2026-07-08): all three KILLED — antifragile sizing on the available signals does NOT beat
  constant-gross.** CH2a trend-strength/breadth (−0.07/−0.19), CH2b correlation (−0.040), CH2c whipsaw governor
  (+0.0006, n.s.). None of the trend-quality signals (correlation, VIX-whipsaw, strength/breadth) has forward-
  Sharpe edge → conditioning gross on them only injects a Var(m) noise penalty. A valuable POSITIVE result:
  the constant-gross trend book's sizing is already near-optimal; reactive trend-quality knobs degrade it.
  **Scope:** does NOT generalize to vol-TARGETING (`book_vol_target`, a different mechanism, untested here).
  No live change (nothing passed). The hardened gate harness (`app/research/ch2_sizing.py`) remains for any
  future sizing idea.

### CH3 — Regime-conditional decomposition (DIAGNOSTIC — no new hunt) — ✅ **DONE (2026-07-08)**
Read-only (`scripts/ch3_regime_diagnostic.py` → `docs/reference/ch3_regime_diagnostic.json`), by frozen
BULL/NEUTRAL/BEAR labels. **Q1 (trend by regime, confirmatory):** BULL SR +4.11 / NEUTRAL +0.44 / BEAR −0.77
— structure real, not tradeable via sizing (CH2). **Q2 (parked collinearity → NEITHER is a diversifier, both CLOSED):** a candidate must decorrelate AND stay
active + non-losing in the decorrelating regime. `sector_rotation` corr-to-trend collapses to 0.14 in BEAR
(n=1218 incl GFC) BUT is active and **LOSES −34%/yr there (Sharpe −1.26)** → decorrelates by making a DIFFERENT
losing bet, not a hedge → NOT a candidate. `credit_timing` collinear (0.33) + loses → CLOSED. **No CH4 candidate
from the parked strategies.** (The first draft nominated sector_rotation on corr alone; the review caught that
decorrelation is mechanically confounded for long-flat strategies → added a standalone-value check that reversed
it.) **Q3 (governor tail — resolves the CH2c flag):** on the RIGHT objective (`evaluate_overlay`) the plain VIX
governor **HELPS (shallower tail, Sharpe preserved)** — full maxDD −1.8pp shallower + contiguous crisis
dd_improve COVID −4.25% / BEAR_2022 −1.3% / GFC −0.4%, at −0.003 Sharpe → **KEEP the governor** (the CH2c Sharpe
drag is the expected insurance premium). **Deliverable met:** the regime report; both parked strategies closed;
governor confirmed. No live change. DECISIONS 2026-07-08.

### CH4 — The ONE terminating, pre-registered search: ranging-market MR sleeve ← **NEXT**
Runs ONLY if it does not delay CH0–CH3 (done). **CH3 produced NO candidate** — both parked strategies closed
(sector_rotation decorrelates in BEAR but loses 34%/yr there; credit_timing collinear), so CH4 is the
originally-named **ranging-market MR sleeve**: pre-register it, gate it, and let the moratorium bind if it fails.
The single well-specified conditional family with a real mechanism
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
- **CH0** — ✅ complete (baseline + scorecard).
- **CH1** — ✅ SHADOW-LANDED (2026-07-07): per-name correlation/heat/concentration gate on the live path
  (`per_name_gate.py`, `pm.per_name_gate_mode` default shadow); Opus-reviewed, inert to live trades.
  Shadow-soaking; correlation enforce-threshold to be calibrated from the soak (see CH1 above).
- **CH2** — ✅ COMPLETE (2026-07-08): all 3 sizing multipliers KILLED — **antifragile sizing does NOT beat
  constant-gross** (CH2a trend-strength −0.07/−0.19, CH2b correlation −0.040, CH2c whipsaw +0.0006 n.s.).
  A positive result: keep the trend book's sizing SIMPLE. No live change.
- **CH3** — ✅ DONE (2026-07-08): regime diagnostic. NEITHER parked strategy is a conditional diversifier —
  `sector_rotation` decorrelates in BEAR (0.14) but LOSES −34%/yr there; `credit_timing` collinear → BOTH
  CLOSED, no CH4 candidate. The live plain crash governor HELPS on its drawdown mandate (maxDD −1.8pp, COVID
  −4.25%) → KEEP it (resolves the CH2c flag). No live change.
- **CH4** — 🔄 IN PROGRESS (2026-07-08): the ONE terminating search = the ranging-market MR sleeve.
  **PRE-REGISTERED** (frozen spec + gates + the 12-month moratorium): `docs/reference/CH4_MR_PREREGISTRATION_2026-07-08.md`,
  DECISIONS 2026-07-08. Run pending (build the sleeve per spec → the 5 frozen gates → PASS to paper, or KILL +
  the moratorium binds). → then CH5 (accrual).
- CH3–CH5 — planned. CH4 is gated on a pre-committed moratorium; CH5's clock has started (enforce soak).
