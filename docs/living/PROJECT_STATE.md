# MrTrader — Project State

**One-screen view of what's happening right now. Updated at session start/end when focus changes.**

> **Update rule:** Human updates this at session boundaries. Keep it to one screen. This is NOT a planning doc (that's MASTER_BACKLOG.md) and NOT a history doc (that's ML_EXPERIMENT_LOG.md). It answers: "If I open the laptop cold, what do I need to know in 30 seconds?"

**Last updated:** 2026-06-03 (PEAD UI visibility shipped — selector attribution + PEAD tracking tab; ranker Spike-A CPCV running)

---

## 🧭 ACTIVE DIRECTION (2026-06-02): Alpha v2 — `docs/living/ALPHA_V2_PLAN.md`
PEAD long-only is **live in paper**. The next phase is **structural, not signal-hunting** (from a 5-LLM external review + a code-grounded plan): de-risk PEAD honestly, then test whether the "dead" swing ranker was just strangled by a 5-position long-only book by re-running it **dollar-neutral, sector-neutral, high-breadth, residualized**. **Locked decisions:** short-interest data first; carve a post-2024-06-01 historical holdout; dollar-neutral shorting approved in paper; keep live PEAD running (pause only if leave-one-crisis-out fails). **First moves:** PEAD cost-sensitivity sweep → crisis-block robustness (leave-one-crisis-out). See the plan for the full sequenced roadmap.

**PHASE-1 PEAD DE-RISK COMPLETE (2026-06-02):** §1.1 cost ✅ GO · §1.2 crisis ✅ GO · §1.3 significance ❌ FAIL. PEAD is **real-but-underpowered** (event-level bootstrap p=0.19, HAC t=1.04; CPCV t=2.26 was optimistic) — a long-biased up-trend drift harvester (~0.40 SR, 87% P&L up-trends). **Keep paper-trading as a small diversifier; never a capital centerpiece** (CAPITAL-HOLD confirmed).

**🏁 RANKER VERDICT — DEAD (2026-06-03):** the §3.1 dollar-neutral high-breadth ranker (the Phase-2 capital hope) is **dead**, established rigorously. The first run was invalid (the "dollar-neutral" book actually ran ~35% net-long at 0.35 gross — the L/S engine was fed a one-sided long proposal pool and never re-sized held positions). Fixed end-to-end across 3 phases (observability → neutral-at-target-gross → full-cross-section scoring + breadth). On the **corrected, genuinely-neutral book** (net$ −0.01, gross 0.73), the decisive CPCV (k=8, N_eff=8) gave **Sharpe +0.14, t=0.18, %pos 67%, dep-adj +0.12** → **no cross-sectional alpha**; the long-only +0.22/+1.06 was confirmed **market beta**. Cross-sectional ML ranking is now exhausted (swing noise · intraday cost-drag · ranker null). **→ PEAD is the SOLE validated edge.** Also shipped this session: PEAD-aware entry gate (PEAD had been 0-filling — post-earnings gappers hit the swing 1.5%/0.5% thresholds; now live + filling) and PEAD cockpit v2 (live book + signal log + live-vs-backtest). **NEXT: strategy reassessment** — pivot research to the event-driven edge *family* (PEAD's DNA: discrete event → drift, F2-immune, rules-based) + productionize PEAD; the §3.3 short-interest / Spike-B residualization items are shelved (they were gated on ranker life). Alpha-v3 plan pending owner direction.

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

### UPDATE (2026-06-01 workday) — PEAD lever sweep + second-edge hunt COMPLETE
All high-EV experiments now run. PEAD config tuning is exhausted; the short-side second-edge hunt failed.

| Experiment | Mean | t-stat | Verdict |
|---|---|---|---|
| **PEAD long-only (baseline)** | **+0.546** | **2.26** | ✅ KEEPER, paper-ready |
| PEAD hold-15 | +0.411 | 1.19 | ❌ killed (drift wants longer hold) |
| PEAD long-short | +0.456 | 2.61 | robust lower-return variant (regime-pass) |
| PEAD earnings-quality | +0.449 | 1.02 | ❌ killed (power collapse) |
| QualityShort shorts-only | **-0.903** | -3.19 | ❌ ANTI-EDGE (old +5.95 was inflated) |

**Conclusion:** PEAD long-only +0.546 is the SOLE validated edge. Config tuning can't reach 0.80 (~0.5-0.7
academic ceiling confirmed). Two short approaches ruled out honestly (inverted-long LX7 +0.036; fundamental-
deterioration QualityShort -0.903) — shorting beaten-down names bleeds (they rally on bounces).

### UPDATE (2026-06-01) — Small/mid-cap PEAD expansion REJECTED (honest result)
Built a survivorship-safe small/mid-cap PEAD harness (Polygon grouped-daily flat files, 8755 delisted names
retained, [$2M,$50M] ADV band, top-300/day, 20bps cost, delisted-haircut wired) — 2 Opus pre-run correctness
fixes (PR #362: haircut no-op + eligibility lookahead), full suite green. **Result: mean +0.361, t-stat +0.95
(coin flip), P5 -1.368 → FAILS, and is WEAKER than R1K large-cap PEAD (+0.546, t=2.26).** Opus oddity review:
REAL failure, result trustworthy (the symbology-gap suspect was disproven — FMP covers the tradeable universe;
low trade count is the shared 5-position cap). The literature's "event edges stronger in small-caps" did NOT
survive honest survivorship + cost modeling. **R1K large-cap PEAD remains the sole edge. Small/mid not worth a re-run.**

**The free-data experiment ladder is now fully exhausted** (swing dead, intraday dead, PEAD levers exhausted,
short edges dead, insider weak, buyback no-data, small/mid-cap rejected). The two decisions below are now the
only remaining moves without new infra/data spend.

### UPDATE (2026-06-02) — Gate recalibrated + PEAD long-only WIRED & ACTIVATED for paper
Both decisions executed (Opus design → 2 review passes each → merged):
- **PR #365 — significance-first two-tier promotion gate** (replaces mean-Sharpe≥0.80 relic). PAPER tier
  (t≥2.0, %pos≥0.75, P5≥0.0, mean≥0.35) + CAPITAL tier (t≥2.5, n_folds≥10, mean≥0.50, explicit sign-off).
  Event-sparsity regime waiver (paper-only, flagged) lets PEAD's `worst_regime_sharpe=None` pass paper with
  `requires_human_review`. **Verdicts (real gate): PEAD → PAPER PASS / CAPITAL HOLD; all else FAIL.** Legacy
  `GATE_MODE=mean_sharpe` is a verified no-op. WF-only runs are INCONCLUSIVE (no longer auto-RETIRE in cron).
- **PR #366 — PEAD long-only wired into live paper** (the live path existed but ran a wrong config). Fixed:
  VIX>30 crisis block now fires (daily VIX series injected; **fail-closed** if unavailable), hold=40 (was 5),
  priced-in filter OFF, scorer pins every validated param. Owner-chosen **risk-managed variant** (keeps
  regime/NIS/opportunity/macro/RM overlays → expected tracking error vs the clean +0.546, logged for
  attribution), **marketable entries** (PEAD-scoped, avoids below-ask adverse selection), **full swing budget**.
  9 `pm.pead_*` config keys (defaults=validated). Observability: `app/live_trading/pead_tracker.py` daily row +
  weekly Sharpe-vs-0.546 rollup email.

**🟢 ACTIVATED 2026-06-02:** `pm.swing_selector` set `pead_quality_short` → **`pead`** (pure long-only; drops
the dead QualityShort anti-edge shorts). **⚠️ REQUIRES UVICORN RESTART** to load the new wiring — run
`.\serve.ps1` (or restart `uvicorn app.main:app`). PEAD begins paper-trading on the next premarket cycle
(08:00 ET analyze → 09:50 ET send). Watch `data/pead_tracking.db` daily rows + the weekly rollup email.
**Capital is withheld** (PEAD is CAPITAL-HOLD pending k≥10 re-run OR live-paper confirmation) — this is paper only.

**🔵 DECISIONS NOW YOURS (autonomous experiment ladder exhausted):**
1. **Paper-trade PEAD long-only** — clears the 0.50 paper gate today (t=2.26, 95% pos, PF 1.54, DSR-pass).
2. **Is 0.80 the right promotion gate?** For a PF-1.54 / 95%-positive / DSR-pass / Calmar-0.77 real edge,
   0.80 + P5>-0.30 may be too strict (multi-LLM reviews flagged it). A deliberate gate decision is warranted.
3. **STOP** long-only price-feature ML (swing+intraday dead) AND fundamental/inverted-long shorts (both dead).
4. **Higher-ceiling future bets** (need new infra/data, not just config): options-PEAD (IV-crush), or the two
   remaining untested shorts (MeanReversion, AnalystRevision momentum) — but LOW priority after 2 short failures.

**Caveats on PEAD:** edge rests on ~8 fold outcomes (N_eff=8); ~15% survivorship upper-bound; verify FMP
`date`=announcement-date PIT (5-min spot check). Solid, economically-grounded, but under-powered.

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
