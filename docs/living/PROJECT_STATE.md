# MrTrader — Project State

**One-screen view of what's happening right now. Updated at session start/end when focus changes.**

> **Update rule:** Human updates this at session boundaries. Keep it to one screen. This is NOT a planning doc (that's MASTER_BACKLOG.md) and NOT a history doc (that's ML_EXPERIMENT_LOG.md). It answers: "If I open the laptop cold, what do I need to know in 30 seconds?"

**Last updated:** 2026-06-01 (overnight autonomous session COMPLETE)

---

## ☀️ MORNING SUMMARY (2026-06-01) — read this first

**Headline: the honest pipeline killed two illusions and found one real edge.**

| Strategy | Honest OOS CPCV (per-fold, leak-free) | Verdict |
|---|---|---|
| Swing (long-only cross-sectional) | +0.22, t=0.17, 50% pos | ❌ DEAD (noise) |
| Intraday v63 | -2.80, t=-6.85, PF 0.94 | ❌ DEAD (cost-drag); struck fake +5.14 (memorization) |
| **PEAD (post-earnings drift)** | **+0.546, t=2.26, 95% pos, P5 +0.009** | ✅ **REAL EDGE** (clears 0.50 paper gate; short of 0.80) |

**PEAD is the first genuine, statistically-significant, economically-grounded positive result** the project
has produced. Best config: long-only, VIX>30 crisis block ON, k=8, no priced-in filter. The VIX block was
the key lever — it trimmed the crisis-fold left tail (P5 -0.288 → +0.009, %pos 80% → 95%). Mean 0.546 is
short of the 0.80 promotion gate but comfortably clears the 0.50 PAPER gate. PEAD is event-driven (F2-immune),
rules-based (no leakage risk).

**What this means / next steps (your call):**
1. **Paper-trade PEAD** — it clears the 0.50 paper gate today; collect real fills for 1-2 earnings seasons.
2. **Push PEAD toward 0.80** — remaining levers (Opus deep-dive): hold-extension (5→10→15d), earnings-quality
   split (beat+guidance via analyst revisions), threshold tuning. See `SWING_STRATEGY_DIRECTION.md`.
3. **STOP** all long-only price-feature ML (swing + intraday both confirmed dead).
4. Higher-ceiling future bets: true dollar-neutral L/S (purpose-built short signal), options-PEAD (IV-crush).

**Caveats on PEAD:** edge rests on ~5-8 fold outcomes concentrated in a few windows (N_eff=8); ~15%
survivorship upper-bound; verify FMP `date`=announcement-date PIT (5-min spot check). Solid signal, under-powered.

**9 PRs merged overnight (#335–#348):** save-guard, swing+intraday per-fold retraining, 6 latent integration
bugs (all surfaced only on real runs — mocked tests passed vacuously), daily-source fix (aggregate_5min),
real-data smoke test (#345, guards the empty-matrix class), swing deep-dive doc (#344), PEAD instrumentation
fix (#348, real DSR/regime). Full suite green throughout. notify_watcher auto-starts with uvicorn.

(DEFINITIVE PEAD run with real DSR finishing as of this writing — see logs/p0_pead_cpcv_DEFINITIVE.log;
result will be appended to ML_EXPERIMENT_LOG.)

---

## 🌙 OVERNIGHT PLAN (completed)
All 7 planned items done: intraday definitive (−2.80, dead) → daily-source fixes (#343/#346) → intraday
oddity analysis → real-data smoke test (#345) → PEAD eval (+0.546, the win) → swing deep-dive (#344). Plus
PEAD instrumentation fix (#348) for honest DSR. Process held: branch → Opus implement → Opus review → tests
→ merge → document.

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
