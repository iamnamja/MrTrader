# MrTrader — Model Status

**This is the single source of truth for active model versions.**

> **Update rule:** Updated by Claude as the final step of any retrain, promotion, or revert. Updated by human when manually changing the active paper-trade model. If this file and the DB disagree, trust the DB and update this file.

**Last updated:** 2026-06-12

---

## 🚦 H1 VERDICT — PEAD DEMOTED at event level (2026-06-12, #456)
Event-level re-adjudication of the live PEAD edge (`H1-PEAD-EVENTLEVEL-20260611`, one-shot R4): on a 9,774-qualified R1K panel (2019→2026), two-way (announce_date×firm) CGM-clustered, **10d SPY-hedged mean −8.3bp, t=−0.77, one-sided p=0.7804 → DEMOTE** (negative at every horizon; bootstrap p=0.66; robust to all LOCO). **PEAD is NOT an event-level edge → live book = trend-plus-cash.** ⚠️ **OWNER-GATED:** flipping the live PEAD sleeve → 0 (keep tracker for telemetry) is an owner action, NOT auto-executed (registry decision=None). The PEAD-long-only section below is superseded by this verdict for any CAPITAL question.

---

## Alpha-v6 infra — slow fuses lit (P1c + P2) — 2026-06-11 (#455)
- **P1c — nightly NBBO snapshot logger** (`scripts/log_options_nbbo.py`): 15:55 ET options-snapshot logger via **Alpaca** (`feed=indicative`, free) → `data/options_spread_obs.parquet`. The Polygon $79 plan serves no options NBBO. ⚠️ **The recurring schedule activates only after a uvicorn restart**; runnable manually meanwhile. Accumulating toward the calibrated spread table (~4-6 wks).
- **P2 — computed-greeks backfill** (`scripts/backfill_computed_greeks.py`): one-time resumable pass over the 112.8M-bar store → `data/options_greeks/` (per-contract IV/delta/gamma/vega/theta, European-warm-start American refine, as-traded Polygon `adjusted=false` closes). STARTED 2026-06-11.

---

## Sleeve allocator — LIVE-WIRED, gate-controlled (DISABLED by default) — 2026-06-08

The book's sleeve weights are now set by a live, kill-switchable allocator
(`app/live_trading/sleeve_allocator_live.py`) — **shipped `pm.allocator_enabled=false` =
today's static budgets (trend 0.40 / PEAD telemetry)**. Default scheme `equal` (Phase-3
validated winner); `vol`/`regime` live-capable but OFF until the book-level gate
(`scripts/run_book_allocator.py`) selects them (on 2 sleeves equal beats both). Enable via
`python -m scripts.set_allocator_config --enable`. Recomputes weekly before the trend
rebalance; readers fall back to static on disabled/stale/warmup/error. PEAD regime
double-tilt guarded. See DECISIONS 2026-06-08.

---

## TSMOM trend sleeve — WIRED for paper (DORMANT + SHADOW) — 2026-06-06

The validated trend sleeve (`app/strategy/tsmom.py`, standalone Sharpe **+0.71**, 19y,
crisis-diversifier) is now wired to trade live as a **standalone weekly ETF rebalancer**
(`app/live_trading/trend_sleeve.py`), running *alongside* PEAD — NOT a `swing_selector`
value. Fires from the orchestrator daily at 09:45 ET, runs only on
`pm.trend_rebalance_weekday` (default Mon) when the market is open (fail-closed via
`AlpacaClient.get_clock`).

- **Deploys DORMANT + SHADOW**: `pm.trend_enabled=false`, `pm.trend_shadow=true`. Shadow
  logs would-be orders to `decision_audit` (`strategy="trend"`, `block_reason="shadow"`)
  without sending. Arm via `python -m scripts.set_trend_config` (set `trend_enabled=true`
  for shadow; add `--arm` + `trend_enabled=true` to send real orders).
- **Equal-capital 50/50** with PEAD: `pm.trend_allocation_pct=0.40` (40% trend / 40% PEAD
  under the 80% gross cap). Per-ETF cap `pm.trend_max_position_pct=0.25`. Universe =
  SPY,QQQ,IWM,EFA,EEM,TLT,IEF,GLD,DBC,UUP.
- **Risk gate** (direct placement, bypasses PM→RM): kill-switch, gross cap (trend+PEAD ≤
  80%), fat-finger, per-name cap; fail-closed on data/NAV/clock failure. Whole shares only.
- **Positions tagged `selector="trend"`/`trade_type="trend"`** and excluded from the
  Trader's stop/target exit loop (rebalancer-managed only).
- **Observability**: `app/live_trading/trend_tracker.py` — daily gross/turnover/P&L rows +
  weekly realized-Sharpe-vs-+0.71 rollup email (`notifier.enqueue("trend_weekly")`).
- **Expected tracking error** vs the +0.71 backtest (Alpaca vs yfinance price adjustment;
  wall-clock Monday vs modular 5-day rebalance) — surfaced by the tracker, not swallowed.

**⚠️ Requires uvicorn restart** to load the new scheduler job + executor. Then run
`scripts/set_trend_config.py` to set the live config (no further restart).

---

## PEAD long-only — WIRED for paper (NOT yet activated) — 2026-06-02

> **2026-06-06 update:** PEAD dialed to **telemetry size** alongside the trend sleeve —
> `pm.pead_size_mult` 3.0→1.0, `pm.pead_max_position_pct` 0.10→0.05 (schema defaults now
> rebaselined, not just DB values). PEAD is the weak market-beta satellite; trend is the
> primary sleeve.

---

## 🧊 FROZEN — dead XS-ML retrain (Alpha-v4 P0, 2026-06-07)

The **swing XS-ranker** (`swing_vNNN`, per-fold CPCV +0.22 / t=0.17 — noise) and the
**intraday 5-min ML** (`intraday_meta_vNNN`, -2.80 / t=-6.85 — cost-drag) are DEAD on the
honest harness and are now **frozen as non-production benchmarks** — no more nightly retrain.
They were still retraining every night because `orchestrator._trigger_retraining` ignored
`RETRAIN_WEEKDAY=-1` / `INTRADAY_ENABLED=False`; that is now fixed. Controls:
`SWING_ENABLED=False`, `INTRADAY_ENABLED=False` (both honored by `retrain_cron`), and the
orchestrator now honors `RETRAIN_WEEKDAY`. **`regime_model_v5` stays ACTIVE** (separate
`regime_training.py` path); PEAD + TSMOM are rules-based (never retrained). Re-enable: flip
the bool in `app/ml/retrain_config.py`.


The live PEAD selector path (`pm.swing_selector="pead"` → `_analyze_swing_pead`) has been
**re-wired to faithfully run the validated +0.546 CPCV config** (branch `feat/pead-live-wiring`).
A prior Opus review found the live path silently diverged from the backtest; that is now fixed.

- **Risk-managed live variant** (owner decision): the validated long-only PEAD config runs with
  the live risk overlays KEPT on top (regime sizing multiplier, NIS news sizing, opportunity-score
  gate, macro-calendar block, RM 10-rule chain) and the live `_calculate_quantity` sizing (NOT
  forced 5% equal-weight). Expected tracking error vs the clean backtest — surfaced by the
  observability layer, not silently swallowed.
- **Marketable entries** (owner decision): PEAD entries route as a marketable limit (crosses the
  spread, ask+10bps for longs) so fills track the backtest's next-open assumption — scoped to
  `selector=="pead"` only; swing/intraday entry routing unchanged.
- **Full swing budget** (owner decision): PEAD uses the entire ~70% swing book; swing/intraday ML
  stay dormant.
- **Config-aligned to +0.546**: long_threshold=0.05, short_threshold=-0.05, max_days_after=3,
  long_short=False, vix_block_all=30, vix_block_short=100, vix_conf_ref=100,
  max_announce_day_move=1.0 (priced-in filter OFF), require_positive_revision=False,
  **max_hold=40 trading days**, long-only. All pinned via `pm.pead_*` agent_config keys
  (defaults = validated values) so the config is inspectable and cannot silently drift.
- **FIX A**: a daily VIX **series** is now injected under `symbols_data["^VIX"]` so the crisis
  block (VIX>30 → no entries) actually fires — previously a scalar/no-key bug meant the block,
  which is credited with the entire edge (P5 −0.29→+0.01), NEVER fired live.
- **Observability**: per-day tracking artifact (`app/live_trading/pead_tracker.py`,
  `data/pead_tracking.db`) records signals/entered/filled/fill-rate/gross/daily+cum P&L/VIX/
  vix_block_fired and per-overlay suppression counts; weekly rollup emails realized Sharpe vs the
  +0.546 expectation via `notifier.enqueue("pead_weekly", …)`.
- **Paper observability now FULLY wired (2026-06-02, branch `feat/pead-eod-pnl-weekly-rollup`):**
  EOD step in `_run_eod_jobs` upserts the PEAD book's real daily P&L (gross/realized/unrealized via
  `Trade.selector=="pead"` + Alpaca MTM), and a Friday-only rollup (guarded to skip <3 deployed
  days) emails the weekly Sharpe-vs-0.546 — closing the prior "Sharpe: n/a forever" gap. Requires a
  uvicorn restart to activate the new EOD code.
- **Status:** wired + fully tested (`tests/test_pead_live_wiring.py`); **NOT activated**.
  Activation = flip `pm.swing_selector` to `"pead"` (separate deliberate step).

---

## Active Models (Paper Trading)

| Model | Version | Status | Last WF Sharpe | Last CPCV Result | Notes |
|---|---|---|---|---|---|
| swing | v224 | ⚠️ UNVERIFIABLE | INVALID (in-sample) | Cannot run — trained_through=None | Saved 2026-05-29, predates trained_through feature (PR #311, 2026-05-30). Retrain required. |
| intraday_meta | v63 | ⚠️ UNVERIFIABLE | **INVALID (in-sample memorization)** | +5.143 STRUCK FROM RECORD | Saved 2026-05-22. +5.14 was scored on its own training data — see below. Retrain required. |
| regime | v5 | ACTIVE | — | — | Regime classifier; AUC gate separate |

> ## 🔴 CRITICAL (2026-05-31): Both ML models are UNVERIFIABLE; prior results are in-sample
>
> Both `intraday_v63` (saved 2026-05-22) and `swing_v224` (saved 2026-05-29) were trained and
> saved **before** the `trained_through` feature existed (PR #311, 2026-05-30 23:18). Neither
> artifact records its training cutoff, so the OOS guard correctly refuses to validate them.
>
> **The intraday +5.143 CPCV / +6.618 WF result is in-sample memorization, NOT edge.** Confirmed
> by Opus 4.8: the model's 730-day training window (2024-05 → 2026-05) fully contains all three
> CPCV test folds (2024-11 → 2026-04). The run also started 9 minutes before the OOS guard commit
> landed, so it had zero out-of-sample verification. Per-fold Sharpe of 2.05/9.64/8.16 is the
> signature of memorization. **These numbers are struck from the record.**
>
> **Swing v224 has the identical disease** (same frozen-full-window architecture, 6-year window).
>
> **DEEPER STRUCTURAL FINDING:** frozen-model CPCV (one pre-trained model scored across all folds)
> is *structurally incapable* of being out-of-sample when the model is trained on the full window.
> The OOS invariant `te_start > trained_through + purge` can only hold if test folds are AFTER the
> training cutoff — i.e. in the sacred holdout. The honest fix is **true per-fold retraining**
> (retrain inside each fold on only that fold's train window). See PIPELINE_ARCHITECTURE.md KL-10.
>
> **UPDATE (2026-05-31, Phase 1 landed):** the swing per-fold-retrain mechanism now exists
> (`--per-fold-retrain`, swing only). Each fold trains a fresh model on its own `[tr_start, tr_end]`
> window (in-memory, no re-fetch; date-spine clamped to `train_end` so no label leaks). The run is
> flagged `is_true_walkforward=True`, and `REQUIRE_TRUE_WF_FOR_PROMOTION` (default False, flip True in
> Phase 3) will make frozen runs report-only. Swing must now be re-run with `--per-fold-retrain` to
> earn a promotion-grade number. Intraday per-fold is Phase 2. See PIPELINE_ARCHITECTURE.md KL-10b.
>
> **UPDATE (2026-05-31, Phase 2 landed):** intraday per-fold-retrain now exists too. `--per-fold-retrain`
> trains a fresh intraday model per fold on its own `[tr_start, tr_end]` 5-min window (same-day label →
> leak-free with no FORWARD_DAYS purge; per-symbol daily bars fetched once and cached). Feasibility:
> a reduced liquidity universe is forced (`--intraday-top-n`, default 150) and the design recommends
> reduced-universe WF (k=4), not full CPCV (WARN when cpcv_k>4). The frozen intraday CPCV numbers below
> are INVALID (in-sample); intraday must be re-run with `--per-fold-retrain` (reduced universe) to earn
> an honest number. See PIPELINE_ARCHITECTURE.md KL-10b.

> **UPDATE (2026-05-31, #343 — daily-feature data-quality fix):** ANY intraday per-fold (or
> frozen) result produced BEFORE this PR used **degraded daily features**. The per-symbol DAILY bars
> that feed the 52-week-position / vol-percentile features were fetched via the Alpaca provider, which
> caps at ~100 recent daily bars on this deployment — so those features silently fell back to their
> 0.5/1.0/0.0 defaults across most of the training window (the 5-min Polygon bars were fine; only the
> daily features were blinded). #343 routes the daily fetch through `INTRADAY_DAILY_FEATURE_PROVIDER`
> (default `"yfinance"`, full multi-year history). Any future intraday WF/CPCV must be re-run on the
> fixed daily-feature path before the numbers are comparable; the new shallow-coverage WARNING will
> flag if daily depth is still inadequate on a real run.

---

## Current Gate Status

| Gate | Status | Notes |
|---|---|---|
| **Swing CPCV (per-fold, HONEST)** | ❌ **FAILED — no edge** | mean +0.22 ± 3.13, t-stat 0.17, 50% pos, DSR p=0.30. First genuine OOS. |
| **Intraday CPCV (per-fold, HONEST)** | ❌ **FAILED — no edge (cost-drag)** | mean -2.80, t-stat -6.85, 0% pos, PF 0.94. Gross edge below cost hurdle. First genuine OOS. |
| Intraday CPCV (frozen) | ⚠️ INVALID (in-sample) | +5.143 — STRUCK (memorization) |
| **PEAD CPCV (honest, definitive)** | 🟡 **REAL EDGE — paper-gate-ready** | mean +0.546, t-stat +2.26, 95% pos, P5 +0.009, DSR p=1.0, PF 1.54, Calmar 0.77. Passes 6 gates; short of 0.80 promotion bar. Clears 0.50 PAPER gate. First positive edge. |
| **Small/mid-cap PEAD CPCV** | ❌ **REJECTED — weaker venue** | mean +0.361, t-stat +0.95 (N_eff=8, coin flip), P5 -1.368, 76% pos. Survivorship-safe (Polygon flat files, 8755 delisted retained), 20bps cost, delisted-haircut wired. Opus oddity review: REAL failure, not under-tested (symbology gap disproven). Below R1K +0.546 and the ≥+0.70 pre-reg bar. Not worth a re-run. R1K PEAD stays sole edge. |

> ## 🔴 BOTH ML STRATEGIES HAVE NO HONEST OOS EDGE (2026-06-01)
> Swing long-only per-fold CPCV **+0.22, t=0.17** (noise); intraday per-fold CPCV **-2.80, t=-6.85**
> (cost-drag dominated, gross PF 0.94 below break-even). Struck frozen intraday +5.14 was in-sample
> memorization. Convergent across independent horizons: price/technical features, long-only, after
> costs → no tradeable alpha in this universe. Pivot (`SWING_STRATEGY_DIRECTION.md`): PEAD #1
> (runnable, +0.349 on record, F2-immune), dollar-neutral L/S #2. STOP long-only price-feature ML.

> ## ✅ FIRST HONEST SWING RESULT (2026-05-31): swing v224 has NO out-of-sample edge
>
> Per-fold-retrain CPCV C(6,2), `--as-of 2026-05-29` — genuinely out-of-sample (each fold trains a
> fresh model on its own causal window, verified no-leak, 20d horizon, trained_through=fold train_end):
> - **Mean Sharpe +0.220 ± 3.130, path t-stat 0.17** (statistically indistinguishable from zero)
> - 50% paths positive (coin flip), DSR p=0.30, P5 -3.97 / P95 +3.68
> - Deployment-adjusted Sharpe +0.166 ≈ raw mean → **not a deployment artifact; there is simply no edge**
> - PF 1.256 / Calmar 6.53 "OK" are short-window aggregation artifacts (62-day test windows from
>   purge=embargo=85), NOT evidence of edge — ignore them.
> - worst_regime_sharpe=None: real CPCV instrumentation gap (per-fold equity too sparse to fill
>   regime buckets); gate correctly fails-closed. Does not change verdict.
>
> **Opus 4.8 verdict:** Long-only cross-sectional swing ranking is **exhausted**. Convergent evidence:
> 9 LX experiments failed (LX9-A beta-neutral +0.031, F2=-0.70), honest LX1 baseline +0.079, and now
> the first true OOS CPCV of the production v224 architecture is noise (+0.22, t=0.17). The recurring
> failure (long-only beta exposure destroyed in VIX spikes) is structural to the strategy class.
> The oddities (53% fold-skip biased toward friendly 2023-24 bull folds, omitting the 2022 bear) all
> bias the number UPWARD — a cleaner run would be lower, not higher.
>
> **The real win: the validation pipeline is now honest.** It turned the inflated frozen swing numbers
> (+0.08 to +5.14) into the noise they always were. Next: apply it to intraday (Phase 2) + evaluate PEAD.

> **NOTE (2026-05-31):** All WF/CPCV results prior to the 13-round audit (PRs #323–327) must be re-run. Pipeline hardening changes gate behavior for Calmar, PF, and OOS guard. The intraday CPCV result above was produced on the corrected pipeline.
>
> **Known limitations in current gate results:** Deployment-adjusted Sharpe not yet computed (CRITICAL-2 fix pending Phase 1). **Regime gate now ACTIVE (Phase 2, 2026-05-31):** coarse3 BULL/BEAR/NEUTRAL labeler with expanding-quantile VIX (PIT-correct); `worst_regime_sharpe=None` now HARD-FAILS the gate (no more silent pass) unless `ALLOW_NO_REGIME_GATE=True`. WF/CPCV results must now populate `FoldResult.regime_sharpes`. See `docs/living/PIPELINE_ARCHITECTURE.md` §12 for full known-limitations list.

---

## Recently Gate-Failed Models

| Model | Version | Avg WF Sharpe | Gate Result | Reason | Date |
|---|---|---|---|---|---|
| swing | v216 | -0.91 | FAILED | LambdaRank 18-feat 20d. PF=0.00 every fold. | 2026-05-23 |

---

## Version History Quick Reference

See `docs/living/ML_EXPERIMENT_LOG.md` for full fold results and training details.

| Model | Versions | Key milestone |
|---|---|---|
| swing | v186–v224 | v186 last "honest" pre-audit baseline; v217–v224 trained post Phase C features |
| intraday_meta | v51, v61–v63 | v63 first post-audit CPCV pass |
| regime | v1–v5 | v5 current; AUC gate: ≥0.75 + Brier < baseline |
