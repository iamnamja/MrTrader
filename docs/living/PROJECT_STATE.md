# MrTrader — Project State

**One-screen view of what's happening right now. Updated at session start/end when focus changes.**

> **Update rule:** Human updates this at session boundaries. Keep it to one screen. This is NOT a planning doc (that's MASTER_BACKLOG.md) and NOT a history doc (that's ML_EXPERIMENT_LOG.md). It answers: "If I open the laptop cold, what do I need to know in 30 seconds?"

**Last updated:** 2026-06-10 (4y options data acquired; OPT-5 implied filter → FRAGILE/parked; live-path + test-infra hardening; **5-LLM review synthesized → ALPHA-v6 direction chosen + documented**)

## 🧭 NOW (2026-06-10): operating the live book; options = data asset (not a sleeve); external review out
- **Live book unchanged = PEAD (telemetry size) + TSMOM trend.** First REAL trend paper rebalance still **Mon 2026-06-15 09:45 ET** — verified READY (read-only shadow sim) and reliability-hardened (per-order commit + 1800s weekly misfire grace, #430). Today is quiet/no-trades and that's correct: swing ML ranker dormant by design (`pm.swing_ml_live_enabled=false`), PEAD found 0 qualifying earnings signals, trend not due until Monday.
- **OPT-5 implied-move filter → ❌ FRAGILE / PARKED (#433).** The threshold-robustness sweep showed the PEAD lift exists only at ratio=1.0 and *inverts* at 1.25 → overfit-suspect. Don't pursue without a powered + pre-registered re-test. Broader options program stays PAUSED (short-vol = risk premium, not alpha).
- **Options data → full 4y local store acquired (#440).** 2022-06-09 → 2026-06-08, ~112.8M bars, 733 names, ~6.18M contracts — the max the Polygon Developer plan serves. We now own the complete copy **even if the subscription is cancelled.**
- **Reliability / test-infra hardening (this session):** kill-switch strict-bool restore (#434); test→prod log/DB bleed closed via one shared `is_test_mode()` (#435/#436); weekly regime retrain + fixed 3-class gate revived from an abandoned PR (#439); Finnhub+FMP economic-calendar 403 log-spam silenced (#437/#441 — the calendar needs a paid tier on BOTH providers; falls back to FRED macro + hardcoded FOMC/NFP).
- **External quant review prepped:** `docs/reference/prompts/EXTERNAL_QUANT_REVIEW_PROMPT.md` + the `20260610_Quant_Options_Review/` kit — soliciting world-class-quant feedback on the next phase across multiple LLMs (harness soundness, options models, architecture gaps).
- **✅ DIRECTION CHOSEN → ALPHA-v6** (SSOT: [NEXT_PHASE_BLUEPRINT_2026-06.md](../reference/NEXT_PHASE_BLUEPRINT_2026-06.md)). The 5-LLM synthesis (Gemini/DeepSeek/Grok/ChatGPT/Claude, deep-dived by Fable 5, code-grounded) converged: our harness over-corrected into a **Type-II / false-negative machine** — a t≥2.0 gate on ~8 folds of ≤4y data kills *true* Sharpe-0.5–0.7 edges (t≈SR·√years), so 100% KILL (incl. confirmed-real index VRP) is a miscalibrated ruler, not an empty opportunity set. **7-phase plan:** **P0** calibrate the ruler (gate positive/negative controls — TSMOM-on-4y decisive — + two-track acceptance [alpha vs book-delta] + research registry) → **P1** live-book fidelity (replay-diff / fill-quality / NBBO spread-calibration) ∥ **P2** options *feature* layer (persisted greeks / surface-quality / BMO-AMC snapshots) → **P3 (centerpiece)** earnings-event panel + event-LEVEL PEAD inference + PEAD v2 continuous score → **P4** options-as-equity-signal XS sleeve (CPIV/skew/O-S/term-slope) → **P5** trend broadening (19y) → **P6** index-VRP micro-sleeve behind the book gate. **Options = signal-first, not execution.** Reviews archived under `docs/reference/prompts/20260610_Quant_Options_Review/responses/`.
- **▶️ P0 STARTED — gate-calibration harness SHIPPED (PR #444):** `scripts/walkforward/gate_calibration.py` scores positive (`tsmom_4y` decisive, `tsmom_19y`, `xmom_12_1`, `pead_baseline`, `spy_buyhold`) + negative (balanced/beta nulls, `leaky_tplus1`) controls through the PRODUCTION gate to MEASURE its false-negative/false-positive rates. Built + 2× Fable-5 adversarial review (1 BLOCKER + 3 MAJOR fixed), 49 tests, changes no threshold. **⏳ NEXT: run the full control suite → OC table → the decisive `tsmom_4y` number** (does our known-good +0.71/19y sleeve pass its own gate on 4y?). Until that runs, every historical KILL has an unknown false-negative rate. Pre-registration + pending results in `ML_EXPERIMENT_LOG.md`. Remaining P0: two-track acceptance gate + research registry + `event_regime_sharpes()` (own PRs).
- **Operate Monday's first real trend rebalance** (still Mon 2026-06-15 09:45 ET) in parallel.

---

## 🧭 PRIOR (2026-06-09): options program PAUSED; operating the validated live book
- **Live book = PEAD (telemetry size) + TSMOM trend.** The **dead cross-sectional swing ML ranker is now OFF in the live path** (`pm.swing_ml_live_enabled=false`) — it was silently producing ~30/32 recent trades of a validated-null strategy (#418). The Trader's ML-score gate is now observable (`REJECTED_ML_SCORE`).
- **Trend armed for paper:** `pm.trend_enabled=true`, `pm.trend_shadow=false` → first REAL paper rebalance **Mon 2026-06-15 09:45 ET** (weekly cadence, ~7 ETFs, 40% alloc), via the orchestrator scheduler.
- **PEAD** clears the 0.55 entry gate (confidence 0.65–0.90); event-sparse (fires on earnings). Live-sizing fidelity fixed (sizes off PEAD's own ATR stop, not the swing stop).
- **Alpha-v5 Options Program (OPT-0..4): built, validated, PAUSED.** Two Opus-certified verdicts: single-name earnings IV-crush = KILL (cost-killed); index short-vol = KILL standalone but **VRP real + cost-robust** (PF 2.24/1.75), just Sharpe-weak. Key insight: our gate is an *alpha* gate, short-vol is a *risk premium* (wrong ruler). Harness retained; revisit later as a book diversifier with a risk-premium framework + more data. See DECISIONS 2026-06-09.
- **Next:** OPT-5 (options-data-as-signal: implied-move filter for PEAD + put-skew risk-off, judged on the host sleeve's gate — sanctioned parting options win, data kept) + continued live-book hardening. ⚠️ Recurring CI flake (`test_agent_simulator_rebalance.py` DB-lock) still costs a rerun on most PRs.

---

## 🧭 P3 (2026-06-08): Live regime-aware sleeve allocator — wired, shipped DISABLED
`app/live_trading/sleeve_allocator_live.py` turns the backtest allocator into a live,
kill-switchable book layer. **Ships `pm.allocator_enabled=false` → byte-identical to
today's static budgets** (trend 0.40 / PEAD telemetry). When enabled it recomputes weekly
(before the trend rebalance), persists effective sleeve weights to agent_config, and both
sleeves read them with a fixed-weight fallback (disabled/stale/warmup/error → static).
Default scheme `equal` (the validated winner); `vol`/`regime` are live-capable but OFF
until `scripts/run_book_allocator.py --emit-config` selects them (expected after the 3rd
sleeve). PEAD regime double-tilt guarded (Opus review). To enable later:
`python -m scripts.set_allocator_config --enable`. **NEXT: the 3rd uncorrelated sleeve**,
then re-run the gate to (likely) activate vol/regime.


---

## 🧭 ACTIVE DIRECTION (2026-06-06): Alpha-v4 — Portfolio of Uncorrelated Premia
**SSOT:** `docs/reference/QUANT_REVIEW_SYNTHESIS_2026-06.md` · **Index:** `MASTER_BACKLOG.md` (Alpha-v4 table).

Five independent world-class-quant LLM reviews (ChatGPT, Gemini, Grok, DeepSeek, Claude) converged: **the architecture is good — re-aim, don't rebuild.** The wall is the *opportunity set + a biased ruler*, not technique. Stop hunting one hero edge; assemble **~4 uncorrelated sleeves → book SR ≈ 0.8**, on an honest harness. PEAD is **not** proven alpha (p≈0.19, 87% P&L in up-trends = conditional beta) — keep small, pair with a crisis-positive sleeve.

**Locked (2026-06-06):** ① **PEAD → telemetry** (`pm.pead_size_mult` 3.0→**1.0**, `pm.pead_max_position_pct` 0.10→**0.05**, applied live restart-free — reverses B4). ② **Gates** → lower bar (~0.45) + reweight to robustness (residual-α-t + fold-consistency primary; keep survivability floor). ③ **Targeted re-arch** (keep execution; rework research harness; add sleeve+regime-allocator). ④ **Free-data only** until a spike justifies a paid purchase.

**Phases (EV/effort):** **P0** validation integrity (`is_trained` guard → full-coverage CPCV for rules-based; sequential-WF baseline; fold-coverage report; gate recalibration; freeze dead XS-ML) → **P1** PEAD reckoning (neutralization + FF5 attribution + gapper slippage; decision gate) → **P2** Trend/TSMOM ETF sleeve (crisis-diversifier, book-level eval) → **P3** regime-aware allocator (the unlock; must beat static weights net of turnover) → **P4** gated high-ceiling bets (PEAD 2.0 · options-VRP spike · squeeze-conditioning). **Regime policy:** "attribute, don't amputate" — switch *allocation* across full-history sleeves, never train a model per regime (§4b).

**PHASE 0.1 + PHASE 1 DONE (2026-06-06):** `is_trained` guard fix shipped (#394) → PEAD CPCV now full-coverage (skips 50%→8). On the honest harness PEAD is **NOT a standalone edge**: unbiased CPCV mean +0.578 but t=1.81 (<2)/p5=−0.796 (worse tail revealed); **CAPM beta-isolation is decisive — α=−1.29%/yr, HAC α t=−0.95, β=0.14; the beta-removed (market-hedged) Sharpe is −0.37** (hedge out SPY and it loses money). The positive Sharpe is small β riding the 2020–26 bull. Gapper-slippage collapses the edge by 50bps. → **PEAD = weak market-beta-driven risk-on satellite, kept at telemetry size (1.0×/5%, live), never a centerpiece.** Retires "PEAD is the sole edge." PEAD 2.0 dropped (was gated on neutralized PEAD showing life). See ML_EXPERIMENT_LOG Phase 1. Tool: `scripts/pead_phase1_attribution.py`.

**PHASE 2 DONE (2026-06-06) — TSMOM trend sleeve ✅ KEEP.** Built a vectorized PIT-safe TSMOM sleeve (`app/strategy/tsmom.py`, 10-ETF multi-asset, long-flat, inverse-vol, weekly) + validation (`scripts/run_tsmom.py`). Independent deep-dive: no look-ahead; fixed a cost-timing off-by-one. **Standalone Sharpe +0.714** (2007-26, incl. all crises), crisis-positive in *slow* bears (2008 +7% vs SPY −37%) but whipsawed in *fast* shocks (COVID −6%). **corr to PEAD +0.25** (diversifying). **Exit gate PASSES:** PEAD+trend (50/50, vol-matched) beats PEAD-alone on Sharpe (0.31→0.92) AND drawdown (−13.7%→−8.3%). Trend is the stronger sleeve; PEAD the weak satellite. See ML_EXPERIMENT_LOG Phase 2.

**PHASE 3 DONE (2026-06-06) — sleeve allocator built; SHIP SIMPLE FIXED-WEIGHT.** Built `app/strategy/sleeve_allocator.py` (general N-sleeve: equal/vol/regime, inverse-vol risk parity, regime hysteresis + EWMA blend, PIT) + `scripts/run_book_allocator.py`. Independent deep-dive CLEAN (no look-ahead). On the PEAD+trend overlap (2020-26): **equal-capital Sharpe +1.082 / maxDD −5.1% / 0× turnover** beats static vol-weight (+0.715) beats regime-tilt (+0.593 / −6.4% / 2.6×). **Regime tilt FAILS its margin** (worse on every metric) → regime layer OFF. **Inverse-vol is fooled by PEAD's sparse-low vol** (over-weights the weak sleeve) → loses to naive equal-capital. → **ship simple fixed-weight book; vol+regime layers kept as OFF scaffold; revisit with more sleeves / longer overlap.** ("Complexity must earn it.") See ML_EXPERIMENT_LOG Phase 3.

**ALPHA-v4 ARC (Phases 0-3) COMPLETE.** Net: PEAD = weak beta satellite (kept at telemetry size); TSMOM trend = the strongest sleeve (Sharpe +0.71 19y, diversifying); the book = the two at simple fixed weights (smart weighting doesn't yet earn its complexity on a 2-sleeve / thin overlap).

**✅ LIVE MULTI-SLEEVE WIRING SHIPPED (2026-06-06, shadow-first):** the trend sleeve now trades live as a **standalone weekly ETF rebalancer** (`app/live_trading/trend_sleeve.py`) alongside PEAD — NOT a `swing_selector` value. Direct Alpaca placement with a lightweight risk gate (kill-switch / gross cap trend+PEAD≤80% / fat-finger), fired by the orchestrator daily 09:45 ET → runs only on `pm.trend_rebalance_weekday` (Mon) when the market is open (fail-closed `get_clock`, covers holidays). **Equal-capital 50/50** (`pm.trend_allocation_pct=0.40`); PEAD dialed to telemetry (schema defaults rebaselined 1.0/0.05). Positions tagged `selector="trend"` and excluded from the Trader's stop/target loop (rebalancer-managed). Tracking via `trend_tracker.py` (+0.71 ref, weekly rollup). **Deploys DORMANT + SHADOW** (`pm.trend_enabled=false`, `pm.trend_shadow=true`). **To activate: restart uvicorn, then `python -m scripts.set_trend_config`** (set `pm.trend_enabled=true` for shadow; add `--arm` for real orders). 32 new tests pass; flake8 clean. **REMAINING (owner-gated):** verify a shadow run on the live data tier (≥253 daily bars per ETF), then arm.

**PRIOR NEXT (now done):** ~~live multi-sleeve wiring — get the trend sleeve trading at a simple fixed weight alongside PEAD~~. This places live ETF orders in the paper book → a deliberate live-system change needing owner go-ahead. Then **Phase 4** (options-VRP feasibility spike / squeeze-conditioning; PEAD 2.0 dropped after Phase 1). Phase 0 remainder (sequential-WF baseline, fold-coverage report, gate recalibration, freeze dead XS-ML) remains as harness hygiene. **CI `database is locked` flake — FIXED (2026-06-06):** per-worker SQLite isolation — feature_store / pead_tracker / notifier DB paths are now env-overridable and conftest points each xdist worker at its own temp dir (set before collection); the bare `universe_history` connect hardened (timeout=30). Validated under -n 8 (2767 passed, 0 locks).

---

## 🗄️ PRIOR DIRECTION (2026-06-02, superseded): Alpha v2 — `docs/living/ALPHA_V2_PLAN.md`
PEAD long-only is **live in paper**. The next phase is **structural, not signal-hunting** (from a 5-LLM external review + a code-grounded plan): de-risk PEAD honestly, then test whether the "dead" swing ranker was just strangled by a 5-position long-only book by re-running it **dollar-neutral, sector-neutral, high-breadth, residualized**. **Locked decisions:** short-interest data first; carve a post-2024-06-01 historical holdout; dollar-neutral shorting approved in paper; keep live PEAD running (pause only if leave-one-crisis-out fails). **First moves:** PEAD cost-sensitivity sweep → crisis-block robustness (leave-one-crisis-out). See the plan for the full sequenced roadmap.

**PHASE-1 PEAD DE-RISK COMPLETE (2026-06-02):** §1.1 cost ✅ GO · §1.2 crisis ✅ GO · §1.3 significance ❌ FAIL. PEAD is **real-but-underpowered** (event-level bootstrap p=0.19, HAC t=1.04; CPCV t=2.26 was optimistic) — a long-biased up-trend drift harvester (~0.40 SR, 87% P&L up-trends). **Keep paper-trading as a small diversifier; never a capital centerpiece** (CAPITAL-HOLD confirmed).

**🏁 RANKER VERDICT — DEAD (2026-06-03):** the §3.1 dollar-neutral high-breadth ranker (the Phase-2 capital hope) is **dead**, established rigorously. The first run was invalid (the "dollar-neutral" book actually ran ~35% net-long at 0.35 gross — the L/S engine was fed a one-sided long proposal pool and never re-sized held positions). Fixed end-to-end across 3 phases (observability → neutral-at-target-gross → full-cross-section scoring + breadth). On the **corrected, genuinely-neutral book** (net$ −0.01, gross 0.73), the decisive CPCV (k=8, N_eff=8) gave **Sharpe +0.14, t=0.18, %pos 67%, dep-adj +0.12** → **no cross-sectional alpha**; the long-only +0.22/+1.06 was confirmed **market beta**. Cross-sectional ML ranking is now exhausted (swing noise · intraday cost-drag · ranker null). **→ PEAD is the SOLE validated edge.** Also shipped this session: PEAD-aware entry gate (PEAD had been 0-filling — post-earnings gappers hit the swing 1.5%/0.5% thresholds; now live + filling) and PEAD cockpit v2 (live book + signal log + live-vs-backtest). **NEXT: strategy reassessment** — pivot research to the event-driven edge *family* (PEAD's DNA: discrete event → drift, F2-immune, rules-based) + productionize PEAD; the §3.3 short-interest / Spike-B residualization items are shelved (they were gated on ranker life). Alpha-v3 plan pending owner direction.

**🧭 ALPHA-v3 TRACK-A SWEEP COMPLETE — PEAD STILL SOLE EDGE (2026-06-03 PM):** owner approved Alpha-v3 (A: build a 2nd event edge; B: aggressively ramp PEAD in paper). Built the reusable **EventEdgeStrategy** harness (A0, PEAD byte-identical) + acquired **short-interest/short-volume data** (Polygon/FINRA, 540k rows, PIT-safe). Tested the two top candidates — **both NULL**: **A1 analyst up/downgrade drift** ❌ (CPCV looked best-in-campaign at +0.894/t=2.85 but was a **52% fold-skip artifact**; neutralized L/S +0.342/t=1.24, full-window CAPM alpha t=0.20 → noise) and **A2 dollar-neutral short-interest factor** ❌ (**−1.213, t=−3.53** — the Boehmer/Asquith anomaly *reversed* in the meme era). Cross-sectional ML ranker + A1 + A2 all dead. **PEAD (+0.546, real-but-underpowered) remains the SOLE validated edge.** Also fixed: nightly intraday retrain crash (orphan `n_workers` kwarg). **NEXT → Track B (owner-gated, at restart):** B1 realized-Sharpe EOD pipeline (the self-certification clock) + B2 Friday cron + B4 aggressive paper-allocation ramp. The event-edge engine (harness + beta-isolation discipline + FINRA/SI + analyst-grades data) is retained for future candidates.

**🟢 TRACK B SHIPPED (2026-06-04):** B1 (realized-Sharpe EOD pipeline) + B2 (Friday rollup) were **already built/scheduled** (16:30 ET, 12 tests) — the self-certification clock is live. **B4 aggressive paper ramp** built + merged: config-driven `pm.pead_size_mult`=3.0 + `pm.pead_max_position_pct`=0.10 (live-tunable via agent_config), PEAD-specific `apply_pead_size_ramp`; RM made PEAD-aware so the 10% per-name cap isn't clipped to the global 5% (aggregate still bounded by the 80% gross cap); ADV-participation instrumentation added (slippage already per-fill). 19 tests; PAPER ONLY. **Deploys at the next uvicorn restart.** Tune the ramp live by editing `pm.pead_size_mult` / `pm.pead_max_position_pct` in agent_config (no redeploy). **B5 SPY<200d trend filter** (2026-06-04) replaces the VIX>30 block — CPCV-validated **+0.661 vs +0.546** (every metric better, same window); wired live config-reversible (`pm.pead_regime_control="trend"`, ma=200), fail-CLOSED to VIX if SPY unavailable. **PEAD now stands down in downtrends (SPY<200d), not just VIX spikes** — protects the ramped book. Deploys at restart.

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

- **swing v229** — trained 2026-06-06 (age 4d); XS swing ranker is OFF in the live path (`pm.swing_ml_live_enabled=false`).
- **intraday_meta v65** — trained 2026-06-06 (age 4d).
- **regime v5** — active (age 20d).

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
