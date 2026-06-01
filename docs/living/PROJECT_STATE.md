# MrTrader — Project State

**One-screen view of what's happening right now. Updated at session start/end when focus changes.**

> **Update rule:** Human updates this at session boundaries. Keep it to one screen. This is NOT a planning doc (that's MASTER_BACKLOG.md) and NOT a history doc (that's ML_EXPERIMENT_LOG.md). It answers: "If I open the laptop cold, what do I need to know in 30 seconds?"

**Last updated:** 2026-05-31

---

## Active Phase
**Honest baseline validation on the fully-corrected pipeline**

All 3 phases of the gate integrity overhaul are merged (PRs #329–#334). Pipeline cleared for CPCV by 3 Opus 4.8 review passes. Now establishing the two honest baselines we've never had: swing v224 CPCV and a clean intraday v63 CPCV re-run (the prior +5.14 predates deployment tracking + the active regime gate).

---

## Model Status
→ See `docs/living/MODEL_STATUS.md` for full details.

- **swing v224** — trained 2026-05-29; never CPCV-validated post-audit. **CPCV launching tonight.**
- **intraday_meta v63** — CPCV +5.143 (2026-05-30) but that run predates Phase 1 deployment tracking + Phase 2 regime gate. **Re-running CPCV tonight** to get deployment-adjusted Sharpe.
- **regime v5** — active

---

## RUNNING NOW (2026-05-31): First honest swing CPCV
`logs/p0_swing_v224_cpcv_perfold.log` — swing per-fold CPCV C(6,2), `--per-fold-retrain --as-of 2026-05-29`.
This is the FIRST genuinely out-of-sample ML result in project history. Each fold trains a fresh
model on only its own window (Opus-verified no-leak + per-fold OOS guard, verdict "RUN IT"). ETA ~20-60min.
**The gap between this number and the old frozen +0.08-0.55 swing numbers = the leakage that was inflating them.**
Expect it to be honest-low. A pass (mean Sharpe > 0.80) would be the first promotable swing result.

## Superseded Jobs (2026-05-31, earlier)
1. **CPCV swing v224** (~4h) → `logs/p0_swing_v224_cpcv.log` — first honest swing baseline
2. **CPCV intraday v63 re-run** (~4h) → `logs/p0_intraday_v63_cpcv_postaudit.log` — deployment-adjusted Sharpe on fully-corrected pipeline
3. Macro history + yfinance incremental data updates (~12min)

**Tomorrow:** `/morning` for results → Opus 4.8 analyzes for oddities (esp. if intraday Sharpe stays implausibly high → dig for residual artifacts) → decision tree below.

---

## Decision Tree (after tonight's results)
```
Intraday v63 deployment-adjusted Sharpe?
  > 1.0  → Intraday is REAL edge → lead strategy → begin PEAD as 2nd strategy
  0.3-1.0 → Marginal; the +5.14 was largely a deployment artifact (Opus est. 3-7% real)
  < 0.3  → Artifact confirmed → intraday not deployable as-is

Swing v224 CPCV?
  Passes (mean > 0.80) → run paper-trade prep
  Fails → long-only cross-sectional ranking exhausted (all 9 LX experiments failed,
          incl. LX9-A beta-neutral on corrected pipeline: +0.031, F2=-0.70).
          Next swing work = PEAD (earnings momentum, F2-immune) or proper L/S short model.
```

---

## Strategic Context: Why NOT re-run old models
The pipeline bugs (PF gate non-functional, DSR n_obs=0, pit_union look-ahead, Calmar formula)
affected **measurement**, not the underlying signal. The F2 structural loss (Aug 2024 VIX spike
destroys long-only beta-exposed swing models) is real market behavior — no bug fix changes it.
LX9-A confirms this: beta-neutralized, post-audit, still -0.70 on F2.

**Worth re-running:** only v224 CPCV (never run) + intraday v63 (deployment unknown).
**Not worth re-running:** LX2–LX9, v186–v223 in isolation — failures were F2-structural, not bug-induced.

---

## Data Review (queued for analysis)
After tonight's results, evaluate whether the current data set is the bottleneck:
- Current: yfinance daily/5min, Polygon 5min cache, FMP fundamentals, macro_history, sector ETFs
- Question: are we missing data points that would unlock edge? (options IV/skew, short interest,
  VIX3M term structure, alt data). Opus to assess what's worth downloading vs purchasing.

---

## Blockers / Risks
- Intraday +5.14 Sharpe is implausible for retail equity (daily IR 0.32) — likely a low-deployment
  artifact. Tonight's re-run with deployment tracking will quantify it.
- Swing long-only approach may be structurally exhausted — need forward-looking pivot decision.

---

## Recently Completed
- 2026-05-31: 3-phase gate integrity overhaul merged (PRs #329–#334); 3 Opus review passes; CLEARED FOR CPCV
- 2026-05-31: notify_watcher auto-starts with uvicorn; docs restructured (living/reference/archive)
- 2026-05-31: 13-round adversarial audit complete; PIPELINE_ARCHITECTURE.md is SSOT
