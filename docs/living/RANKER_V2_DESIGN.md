# RANKER v2 — Dollar-Neutral, Sector-Neutral, Residualized High-Breadth Ranker Re-test

---

## ⭐⭐ OWNER-LOCKED RE-ARCHITECTURE (2026-06-03 — supersedes the Spike-A standalone-driver design below)

Two ~5hr Spike A runs were **invalidated**. Root causes:
- **Failure A — driver infidelity:** the standalone `scripts/run_ranker_v2_cpcv.py` built a bare
  `SwingStrategy(...)` that silently dropped the tier3 CLI defaults that produced the +0.22 baseline
  (ATR stop/target 0.5/1.5, `no_prefilters`, opportunity-score, earnings-blackout) → it ran a
  *different book* than the +0.22 anchor → −1.36, not comparable.
- **Failure B — structural short underfill:** the per-side short-gross sector cap
  (`apply_sector_cap_shorts`, 18/sector at `short_n=60` × `cap=0.30`) + `short_min_adv=50M`
  starved the **sector-concentrated, illiquid bottom-of-momentum R1K short tail** → realized
  short gross ~0.13 vs the 0.40 target → the "dollar-neutral" book ran ~27% net long.

### Locked decisions (recorded at re-architecture)
1. **ABANDON the standalone driver.** Run BOTH arms through `walkforward_tier3.py::_run_cpcv_swing`
   — the exact path that produced +0.22, which inherits the faithful ATR/prefilter/opportunity/
   blackout defaults automatically and already wires the L/S short kwargs into the per-fold CPCV
   path. `git rm scripts/run_ranker_v2_cpcv.py`. KEEP: `net_exposure.py`, the
   `CPCVResult`/`FoldResult`/`SimResult` net-exposure fields, the short-kwarg wiring, the
   `long_budget`/realized-count sizing fix, and the genuinely reusable tests (migrated).
2. **Rebalance-vs-rebalance design** (kills the signal-vs-rebalance confound that the old
   `--baseline` arm had):
   - **CONTROL** = rebalance-mode, LONG-ONLY, high-breadth (60 names, weekly, inverse-vol, 20M ADV).
   - **TREATMENT** = same engine/breadth/cadence + dollar-neutral shorts (60/side,
     `long_gross=short_gross=0.40`) + net-sector cap + SPY beta-hedge overlay.
   - Only neutrality differs between the two arms. The signal-mode +0.22 stays a *separately
     reported historical reference*, NOT the in-experiment delta.
3. **Neutrality method = NET-sector cap + SPY beta-hedge overlay** (replaces the per-side
   short-gross sector cap). Cap *net* sector exposure (long − short per sector), let single-name
   shorts concentrate where liquid, then add a SPY-short overlay to absorb residual net beta →
   net beta ≈ 0 guaranteed even when single-name shorts can't fully neutralize.
4. **Pre-registered acceptance (unchanged):** treatment net Sharpe ≥0.6 @5bps AND path-t≥2.0
   AND %pos≥0.75 AND realized |net beta| ≤0.15 (clean alpha) ⇒ "ranker is real, was strangled
   by long-only-5"; else genuinely dead.

### As-implemented (2026-06-03)
- **Driver retired:** `scripts/run_ranker_v2_cpcv.py` removed; its driver-specific tests
  (`TestDriver`, `TestBaselinePin`, `TestFoldParamGuard`) removed. Reusable tests
  (net-beta capture, long_budget/realized-count sizing, PIT, neutrality) migrated into
  `tests/test_ranker_v2_spike_a.py` exercising the sim/tier3 path directly.
- **Net-sector cap** — new `apply_net_sector_cap` in `app/strategy/portfolio_construction.py`.
  Given the long target book and a worst-first short candidate list, it admits shorts greedily
  (in rank order, NO per-side per-sector count cap) but **skips a short whose admission would push
  that sector's projected |net exposure| above the cap** (net = long_count_in_sector −
  short_count_in_sector, as a fraction of `short_n`). This lets the short tail concentrate
  where the longs are NOT concentrated (the common case for a momentum ranker: longs in winners'
  sectors, shorts in losers' sectors → net per sector naturally bounded) instead of refusing
  shorts simply because too many land in one sector. Wired in `_process_rebalance` behind
  `net_sector_cap` (opt-in; `apply_sector_cap_shorts` per-side path preserved as the default so
  every existing rebalance/L-S run is byte-identical).
- **SPY beta-hedge as residual-net-beta overlay** — the existing `spy_beta_hedge` path
  (`agent_simulator.py`) is extended: when `enable_shorts=True` it now hedges the **residual net
  beta of the WHOLE book (longs − single-name shorts)**, not just the long book, and runs as an
  *overlay* (single-name shorts are kept) rather than replacing the short book. Opt-in via the
  existing `--spy-beta-hedge` flag; long-only and pre-existing `spy_beta_hedge`-without-shorts
  behavior unchanged.
- **tier3 wiring:** `_run_cpcv_swing` now forwards `spy_beta_hedge`, `spy_beta_lookback`,
  `spy_hedge_max_gross`, `net_sector_cap`, `capture_net_exposure`, and `net_beta_lookback`
  (the short kwargs / rebalance flags were already wired). New flags: `--net-sector-cap`,
  `--capture-net-exposure`.

### EXACT commands to run (both arms through tier3 — fidelity is automatic)

Both arms inherit the faithful +0.22 defaults: ATR stop/target **0.5 / 1.5** (`--stop-mult 0.5
--target-mult 1.5`), `--no-prefilters`, `--pm-opportunity-score`, `--earnings-blackout`, and the
same per-fold-retrain CPCV machinery (N_eff=n_folds, OOS guard, sacred-holdout clamp, the gate).
Fold fingerprint matches the +0.22 anchor: `--years 5 --cpcv-k 6 --cpcv-paths 2` (purge/embargo
default to SWING_PURGE_DAYS=85).

**CONTROL — rebalance-mode LONG-ONLY high-breadth (60 names, weekly, inverse-vol, 20M ADV):**
```
python scripts/walkforward_tier3.py --model swing --swing-model-version 224 \
    --cpcv --cpcv-k 6 --cpcv-paths 2 --per-fold-retrain --as-of 2026-05-29 --years 5 \
    --stop-mult 0.5 --target-mult 1.5 --no-prefilters --pm-opportunity-score --earnings-blackout \
    --rebalance-mode --rebalance-target-n 60 --rebalance-days 5 --rebalance-inv-vol \
    --rebalance-min-adv 20000000 --rebalance-sector-cap 0.30 \
    --swing-cost-bps 10
```

**TREATMENT — same engine/breadth/cadence + dollar-neutral shorts (60/side, 0.40/0.40) +
NET-sector cap + SPY beta-hedge overlay:**
```
python scripts/walkforward_tier3.py --model swing --swing-model-version 224 \
    --cpcv --cpcv-k 6 --cpcv-paths 2 --per-fold-retrain --as-of 2026-05-29 --years 5 \
    --stop-mult 0.5 --target-mult 1.5 --no-prefilters --pm-opportunity-score --earnings-blackout \
    --rebalance-mode --rebalance-target-n 60 --rebalance-days 5 --rebalance-inv-vol \
    --rebalance-min-adv 20000000 --rebalance-sector-cap 0.30 \
    --enable-shorts --long-gross 0.40 --short-gross 0.40 --short-target-n 60 \
    --short-min-adv 50000000 --net-sector-cap --spy-beta-hedge --spy-hedge-max-gross 0.40 \
    --capture-net-exposure \
    --swing-cost-bps 10
```

Only neutrality differs between the two arms (shorts + net-sector cap + beta overlay).
`--swing-cost-bps 10` ⇒ `transaction_cost_pct = 10/10000/2 = 5 bps/side` (the registered cost).
The TREATMENT arm prints the realized net beta / net dollar / max|net sector| (capture auto-on
with `--enable-shorts`; `--capture-net-exposure` forces it explicitly). The signal-mode +0.22 is
reported separately as the historical reference, NOT the in-experiment delta.

### Rebalance-engine self-validation anchor (FOUND)

The **60d momentum-baseline CPCV (ML_EXPERIMENT_LOG.md "Momentum Baseline CPCV (2026-05-24)")**:
**mean Sharpe +1.306, 71.4% positive, 14 paths** — a PURE rebalance-mode, long-only,
inverse-vol, regime-gated run with NO ML model (`--rebalance-momentum-baseline`). It proves the
shared rebalance engine produces sane results independent of the ranker. It is the apples-to-apples
self-validation reference for the CONTROL arm (same engine, long-only). The IC-composite v219
5-fold WF (**mean +0.504**, rebalance-mode) is a secondary rebalance-engine anchor. NOTE: those
anchors used `--swing-purge-days 10` and a 6-year window, so they are an *engine-sanity* anchor,
NOT a fold-for-fold match; the in-experiment CONTROL arm (run with the commands above) is the
fold-matched reference. The net-exposure + realistic-universe regression tests are the integrity
checks on top of the self-validation.

---

## RANKER v2 — Dollar-Neutral, Sector-Neutral, Residualized High-Breadth Ranker Re-test

**Status:** 🟡 DESIGN (Spike-A standalone-driver design — SUPERSEDED by the re-architecture above). Decisive experiment for Alpha v2 §3.1.
**Author:** Staff quant/eng design pass, ground-truthed against the codebase 2026-06-02.
**Question this experiment answers:** Was the cross-sectional swing ranker genuinely edgeless, or merely strangled by the 5-position long-only constraint (Fundamental-Law transfer-coefficient collapse)?
**Decision weight:** This drives weeks of work. If the high-breadth dollar-neutral residualized book clears the pre-registered bar, the ranker becomes the price-factor component of the Phase-3 north-star core book; if it still fails, the ranker is proven dead the right way and we pivot.

---

## ⭐ OWNER-LOCKED DECISIONS (2026-06-02 — recorded at implementation of Spike A)

1. **Staged sequencing.** Spike A now (book change alone, existing ranker + existing
   label, NO residualization). Residualization / Spike B is built ONLY if Spike A
   warrants it (i.e. breadth+neutrality materially lifts the +0.22 baseline).
2. **Primary book (registered):** dollar-neutral (net beta ≈ 0), sector-neutral,
   **60 names/side** (long top-60 / short bottom-60), **weekly** rebalance,
   **gross ≤ 80% NAV**, **inverse-vol** sizing. Implemented as `long_gross=0.40`,
   `short_gross=0.40` (→ `net_target=0`, gross=0.80), `rebalance_days=5`,
   `rebalance_inv_vol=True`, `rebalance_target_n=60`, `short_target_n=60`.
   Sensitivity arms (easy flags, not all required in Spike A): 40/side, 80/side,
   monthly rebalance (`--long-n/--short-n`, `--baseline-rebalance-days`).
3. **Pre-registered FINAL acceptance (full experiment, esp. Spike B):** net Sharpe
   ≥ 0.6 @5bps/side AND CPCV path-t ≥ 2.0 AND Newey-West HAC t ≥ 2.0 AND %positive
   ≥ 0.75. **For Spike A specifically:** the directional read is whether
   breadth+neutrality moves the Sharpe materially above the +0.22 long-only-5
   baseline (toward ~0.5) and lifts the t-stat. A strong Spike A motivates building
   residualization; a flat/negative Spike A is itself decisive evidence the ranker
   is dead even neutral.
4. **Residualization factor set (Spike B only):** market+sector primary, size as
   an ablation. (NOT in Spike A.)

### Spike A — AS IMPLEMENTED (2026-06-02)
- **Net-new wiring (the core gap):** `SwingStrategy.__init__`/`run_fold` now forward
  `enable_shorts`, `long_gross`, `short_gross`, `short_target_n`, `short_min_adv`,
  `short_add_threshold`, `short_drop_threshold` to `AgentSimulator`. Defaults equal
  the `AgentSimulator` constructor defaults and `enable_shorts=False`, so the
  long-only swing path is **byte-identical** (confirmed: 31 existing swing tests pass).
  The CLI per-fold path (`walkforward_tier3.py:_run_cpcv_swing`) wires the same
  kwargs from existing argparse flags (`--enable-shorts`, `--long-gross`, …).
- **Driver:** `scripts/run_ranker_v2_cpcv.py` — dedicated, load-once,
  deterministic (`retrain_as_of`/`--as-of`), ASCII-safe, **artifacts-first**
  (JSON written before the table prints), `--smoke` fast path. Reuses the honest
  OOS machinery (`run_cpcv`, `SwingFoldRetrainer`, `TrainWindowCache`,
  `assert_model_oos`, N_eff=n_folds, the significance gate) — does NOT fork run_cpcv.
- **Same model + same label as the +0.22 baseline:** uses `SWING_RETRAIN`
  (lambdarank model + lambdarank label) via `SwingFoldRetrainer`, `factor_scorer=None`
  → native `AgentSimulator._pm_score`. Spike A isolates the BOOK-CONSTRUCTION effect.
- **Costs:** 5 bps/side headline (the locked bar), parameterized via `--cost-bps`.
- **`--baseline` arm (FIX 2, 2026-06-02 — PINNED):** the +0.22 anchor is the
  v224 per-fold CPCV run `walkforward_tier3.py --model swing --swing-model-version
  224 --cpcv --cpcv-k 6 --cpcv-paths 2 --per-fold-retrain --as-of 2026-05-29`
  (MODEL_STATUS.md / ML_EXPERIMENT_LOG.md "FIRST GENUINE OUT-OF-SAMPLE SWING
  RESULT"). That command does **NOT** pass `--rebalance-mode`, so it was a
  **SIGNAL-MODE long-only book capped at `MAX_OPEN_POSITIONS=5`** (the 16.5%
  deployment + per-trade PF/Calmar artifacts confirm the 5-position signal book) —
  **NOT a 30-name rebalance book and NOT a 5-name rebalance book.** The driver's
  `BASELINE_PIN` records this; the `--baseline` arm now runs `rebalance_mode=False,
  enable_shorts=False, factor_scorer=None` (native ranker) so it genuinely
  reproduces ~+0.22. The old "long-only-5" output label was right about the 5 cap
  but wrongly ran a rebalance book; the arm is now relabeled
  `baseline_signal_long_only_5pos`. Cost parity: the +0.22 run used
  `--swing-cost-bps 5.0` default (one-way 2.5 bps/side) → pass `--cost-bps 2.5`.
  So Spike A's dollar-neutral 60/side is compared AGAINST a 5-position signal-mode
  long-only anchor — i.e. it tests breadth + neutrality jointly vs the true +0.22.
- **`--years` PIN + fold-param guard (BUG 2, 2026-06-02 — silent anchor invalidator).**
  `run_cpcv` builds folds from `total_years` + `retrain_as_of()`, so the
  `(years, cpcv_k, cpcv_paths, purge_days, embargo_days)` tuple fully determines
  every fold window. The driver's `--years` default was **6**, but the recorded
  +0.22 v224 run passed no `--years` and inherited `walkforward_tier3.py`'s default
  of **5** — 6 shifted every fold window (adding the 2020 COVID year) and broke the
  baseline self-validation (−0.83 vs +0.22). **FIX:** `--years` default → **5**;
  `BASELINE_PIN` now pins the full fold fingerprint (`years=5, cpcv_k=6,
  cpcv_paths=2, purge_days=85, embargo_days=85` = the tier3 CLI defaults the +0.22
  run used); a loud `_assert_fold_params` guard fails the BASELINE arm on ANY
  mismatch (the dn arm only LOGS — free to sweep; SMOKE exempted). The parser is now
  `_build_parser()` so the guard/tests read the REAL default.
- **Per-leg sizing fix (BUG 1, 2026-06-02 — the book ran ~76% net long).** In
  `AgentSimulator._process_rebalance` the long leg ignored the `long_budget` that
  `split_gross_budgets` returns — it sized longs to FULL equity × `long_mult`, so
  the "dollar-neutral" book realized long≈0.93×NAV, short≈0.17×NAV, net_dollar≈0.76.
  - **FIX 1a (dead `long_budget`):** the L/S path now sizes the long leg to
    `long_budget` (which already folds in `long_regime_mult`) with
    `gross_exposure_multiplier=1.0` → long gross = `long_gross`×NAV.
  - **FIX 1b (short underfill + symmetric long):** the short leg under-filled
    (~0.17 vs 0.40) because `short_min_adv`(50M)+`sector_cap`(0.30) shrink the
    eligible short set below `short_n`, and the budget was divided by `short_n`.
    Fix: divide the per-name budget by the size of the REALIZED qualifying TARGET
    BOOK (names in `delta.target`/`short_delta.target` with a valid same-day entry
    price — NOT `target_n`/`short_n` and NOT `len(to_add)`); weight across the full
    realized target book and open only the new adds at their per-name weight so each
    rebalance targets `budget` total. `short_min_adv` is NOT relaxed (liquid shorts
    preserved). Verified on a 130-name synthetic 60/60 0.40/0.40 book: once both
    legs populate, **net_dollar ≈ 0** and **gross ≈ 0.80** point-in-time.
  - **Long-only paths untouched:** SIGNAL-mode long-only (the +0.22 path) never
    enters `_process_rebalance`; the long-only REBALANCE path keeps the existing
    full-NAV×`long_mult` sizing byte-identical.
  - `SimResult` gains `mean_gross`/`last_gross` + optional `net_exposure_by_date`
    (per-EOD live-book snapshots) so a POINT-IN-TIME neutrality test inspects the
    live book, not trade flow (the prior unit test summed trade FLOW and missed it).
- **Realized net-beta / net-dollar / net-sector capture (FIX 1, 2026-06-02 — THE
  blocker).** Without this a positive Spike A is unattributable: a dollar-neutral
  book equalizes long/short *dollars*, so net beta ≈ 0 only if longs' and shorts'
  average betas match — for a ranker they systematically don't, so a +0.5 could be
  leftover market beta, not alpha. Now captured PURE-ADDITIVELY:
  - `app/backtesting/net_exposure.py` — `compute_pit_beta` (60d OLS, bars strictly
    `< as_of` → no look-ahead, reuses the spy_beta_hedge convention),
    `compute_book_net_exposure` (signed book net beta =
    `Σ w_long·β_long − Σ w_short·β_short`, net-dollar, signed per-sector net +
    max|sector|), `summarize_net_exposure`, `NET_BETA_ALPHA_THRESHOLD = 0.15`.
  - `AgentSimulator` captures it per EOD when `capture_net_exposure=True`
    (agent_simulator.py:~655, reusing the existing `deployment_by_date` seam),
    surfaces it on `SimResult` (mean/last/max|net beta|, mean/max|net dollar|,
    max|net sector|, `net_exposure_captured`). OFF by default → long-only path is
    byte-identical (gate test `test_long_only_capture_off_is_byte_identical`).
  - `SwingStrategy.run_fold` forwards the flag (defaults to `enable_shorts`) and a
    sector map; `FoldResult` carries the fields; `run_cpcv` aggregates them onto
    new `CPCVResult` fields (`path_mean_net_betas`, …, `net_exposure_captured`) with
    `mean_net_beta`/`max_abs_net_beta`/`net_beta_clean` properties — same
    pure-additive discipline as §1.2 `path_fold_members`.
  - **Interpretation hook:** the driver reports net beta / max|net beta| / net
    dollar / max|net sector| and flags `is_clean_dollar_neutral_alpha`. Per the
    locked rule (§9-Q4), realized `|net beta| > 0.15` ⇒ NOT clean dollar-neutral
    alpha ⇒ re-run beta-neutral.
  Dollar-neutrality is still guaranteed by construction (`net_target=0`); the
  capture now lets us READ a positive Spike A as alpha (net beta ≈ 0) vs a beta
  artifact (net beta materially ≠ 0).
- **Cost sweep (non-blocking, NOT implemented):** the design §7 sweep over
  {2,5,10,15,20} bps is a TODO in the driver — Spike A uses the single 5 bps point.
- **Borrow optimism (unchanged, flagged):** the backtest shorts bottom-of-rank R1K
  names assuming shortability at the default 5%/yr borrow + `short_min_adv=50M` floor.
  §3.3 SI/borrow feed is the live-readiness follow-on, explicitly not blocking.

> **Prior-evidence caveat (be honest):** LX9-A beta-neutral scored +0.031 (F2=−0.70), honest LX1 baseline +0.079, and the production-architecture per-fold CPCV was **+0.22 / t=0.17, 50% pos, DSR p=0.30** (MODEL_STATUS.md:108–135). Convergent weak evidence says the ranker may be dead even neutral. The point of this experiment is to settle it with the **one configuration never tried**: high-breadth (40–80/side) dollar-neutral **with feature+label residualization** — the only setup that both (a) restores breadth/transfer and (b) removes the market+sector beta that an idiosyncratic ranker should never have been graded on.

---

## 0. TL;DR for the owner

- **The book engine already exists.** `AgentSimulator` has a full long/short rebalance engine (`_process_rebalance`, `split_gross_budgets`, `apply_sector_cap`/`apply_sector_cap_shorts`, `compute_inverse_vol_weights`, per-side regime fns, short collateral). Dollar-neutral / sector-capped / inverse-vol / weekly-rebalance is **set-the-kwargs**, not a rewrite (effort drops L→M).
- **Two genuine net-new lifts:** (1) **residualization of features AND labels** in `training.py` (Gemini's core fix; net-new, M–L); (2) **plumbing `enable_shorts`/`long_gross`/`short_gross`/`short_target_n` through `SwingStrategy.run_fold`** — these constructor params exist on the simulator but are **NOT** currently forwarded by the swing per-fold strategy (net-new wiring, S).
- **The honest OOS machinery already supports this path.** `walkforward_tier3 --per-fold-retrain --rebalance-mode` runs true per-fold-retrained swing CPCV through `AgentSimulator` with deterministic `retrain_as_of` anchoring, the OOS guard, the sacred-holdout clamp, N_eff=n_folds, and the two-tier significance gate. We extend it; we do not fork it.
- **Size factor is feasible-but-optional.** FMP `shares_outstanding` (`weightedAverageShsOutDil`) IS in the PIT fundamentals parquet, so a PIT market-cap proxy (price × PIT shares) is buildable. It is the leakiest/lowest-confidence factor — recommend **market+sector first, size as an ablation arm**, not a hard requirement.
- **Recommended spike-first sequencing:** (Spike A) book change alone, existing rank label, dollar-neutral 60/side → does breadth alone move +0.22 toward 0.5? (Spike B) add residualization → does removing market+sector beta surface idiosyncratic signal? Each is a clean attributable step.

---

## 1. Residualization of features AND labels (the core novelty)

### 1.1 Concept
For each PIT decision day `t`, run a **cross-sectional OLS** of each input on a small factor set and keep the **residual**:

- **Label residualization:** regress the forward return `r_{i, t→t+H}` (H = `LABEL_HORIZON_DAYS`, currently 20d for v224 parity — see `build_train_matrix_for_window` horizon override) on the factor matrix `F_t`; the training target becomes the **idiosyncratic residual** `ε^r_{i,t} = r_{i,t} − F_{i,t} β^r_t`.
- **Feature residualization:** for each feature `x^k`, regress `x^k_{i,t}` cross-sectionally on the same `F_t`; the residualized feature is `ε^{x,k}_{i,t} = x^k_{i,t} − F_{i,t} β^k_t`.

The ranker then **predicts the residual forward return from residualized features** — i.e. it is forced to learn *idiosyncratic* selection, not market timing or sector rotation (which the dollar-neutral, sector-neutral book cannot harvest anyway, so grading the ranker on raw returns was always mismatched).

### 1.2 Factor set `F_t` (per-day cross-sectional design matrix)
Columns of `F_t` for the cross-section live on day `t`:

1. **Market beta** — `β_i` of each name to SPY, estimated PIT over a **trailing 60-trading-day rolling window strictly ≤ t** (daily log returns; OLS slope vs SPY daily returns). The residualization regressor uses `β_i × r^SPY` style market loading; in the simplest, most defensible form we use a single market column = each name's PIT 60d beta (so the cross-sectional regression removes the linear beta tilt of the cross-section). SPY bars are already loaded in every fold (`SwingStrategy.fetch_data` injects `SPY`).
2. **Sector dummies** — 11 GICS-ish sectors via `SECTOR_MAP` / `SECTOR_ETF_MAP` (XLK/XLC/XLY/XLP/XLF/XLV/XLI/XLE/XLB/XLRE/XLU — confirmed in `app/ml/fundamental_fetcher.py:34`). One-hot with drop-one to avoid collinearity with the intercept. Sector membership is from `SECTOR_MAP` (static; treat as a known, non-leaking attribute).
3. **Size (OPTIONAL — ablation arm, not default).** A PIT market-cap proxy = `close_t × shares_outstanding_PIT`, where `shares_outstanding` comes from the FMP PIT store (`app/data/fmp_fundamentals.py:54,269`, `weightedAverageShsOutDil`, looked up with `lookup_pit_from_index` ≤ t exactly as `_build_train_matrix_for_window_impl` already does for fundamentals). Use **log market cap**, then **winsorize/standardize cross-sectionally**. *Honesty flag:* FMP shares are point-in-time-as-reported but coverage is incomplete across the R1K and across history; names without PIT shares get dropped from the size column or fall back to an **ADV proxy** (log of trailing-60d average daily dollar volume, fully price-derived, zero extra data dependency). Recommendation: ship **market+sector** as the default residualization; run **+size(mktcap)** and **+size(ADV proxy)** as two ablation arms to measure whether size matters at all.

### 1.3 PIT / leakage discipline (the highest-risk part)
- **Betas and sector means are computed from data ≤ the decision day only.** Market beta uses a trailing 60d window strictly `< t` (mirror the existing strict-`<` SPY lookups in `agent_simulator.py` and `factor_scorer.py:275`). No same-day SPY bar.
- **The cross-sectional regression on day `t` uses only the cross-section observed on day `t`.** This is inherently PIT (it conditions only on day-`t` values), but the *betas/size feeding it* must be backward-only.
- **Label residualization must NOT use the realized cross-sectional mean of the forward return as a leak vector.** The forward return `r_{t→t+H}` is a future quantity; we regress it on **contemporaneous-at-`t` loadings** (`β_i`, sector dummy, size at `t`). We do NOT regress on any t+H quantity. The residual is the part of the future return orthogonal to the *day-t* factor exposures — that is legitimate and is exactly what an idiosyncratic ranker should predict.
- **Per-fold purge already covers the H-day label horizon** (`SWING_PURGE_DAYS=85 = FEATURE_LOOKBACK(60)+LABEL_HORIZON(20)+buffer(5)`; invariant asserted at import in `retrain_config.py`). Residualization adds the 60d beta lookback, which is ≤ the existing 60d feature lookback — **purge does not need to grow**.
- **Mandatory synthetic-leak test before trusting any number.** Build a synthetic panel with a known idiosyncratic signal + injected market/sector factors; verify (a) residualization removes the injected factor exposure, (b) shuffling future labels collapses residual-IC to ~0, (c) the residualized pipeline produces no higher in-sample IC on pure-noise residuals than chance.

### 1.4 Where it slots in (exact code sites)
- **New module:** `app/ml/residualize.py` — `compute_pit_betas(symbols_data, spy, as_of, lookback=60)`, `residualize_cross_section(values_by_sym, factor_matrix) -> residuals`, and a `ResidualizationConfig(factor_set={"market","sector","size"}, size_mode="mktcap"|"adv"|None)`.
- **Label path:** add `label_scheme="residual_rank"` (and/or `"residual_regression"`) alongside the existing `cross_sectional`/`sector_relative`/`return_regression`/`lambdarank` branches in **both** `_process_symbol_windows_worker` (subprocess, `training.py:350`) and `_compute_cs_thresholds` (`training.py:1602`). The residual forward return is computed in the threshold/label pre-pass where the cross-section for window `w` is already assembled per day (the same place `cs_thresholds` is built), because residualization is inherently **cross-sectional per day** — it cannot be done in the per-symbol worker in isolation. **Architecture note:** this is the one structural change — today labels are computed per-symbol with precomputed scalar thresholds; residual labels require a per-day cross-sectional regression. Cleanest implementation: extend `_compute_cs_thresholds` to also return, per `(w_start_idx)`, the fitted `β^r_t` so the worker subtracts `F_{i,t} β^r_t` from each symbol's raw forward return. Feature residualization is analogous but must run over the feature matrix after `engineer_features` — practical approach: residualize features in a **post-assembly pass in `_build_train_matrix_for_window_impl`** (and the matching `_build_rolling_matrix`) after `_windows_to_matrix` returns `X, meta`, grouping rows by `meta["window_idx"]` (the day) and regressing each feature column on `F` within each day-group, since `meta` carries `symbol`, `sector`, and `window_idx`.
- **Inference parity (CRITICAL):** the live/sim scorer must residualize features **identically at scoring time**. The ranker-as-scorer adapter (§2.4) computes the same PIT betas + sector dummies (+ optional size) for the day's cross-section and residualizes the feature vector before `model.predict`. Train/serve skew here would silently invalidate the result — this parity is itself a test target.

---

## 2. Book construction (reuse the existing L/S engine)

### 2.1 Target book
- **Long top-N / short bottom-N** of the **same** ranker (standard market-neutral; NOT the failed QualityShort thesis). N = 40–80/side (owner decision §6).
- **Dollar-neutral:** net target ≈ 0 (`net_target≈0` → `long_gross == short_gross`).
- **Sector-capped** on both books (engine gives a sector *cap*, not strict neutrality — see §2.3 honesty flag).
- **Gross ≤ 80% NAV** (`long_gross + short_gross ≤ 0.80`, i.e. `long_gross = short_gross = 0.40`).
- **Inverse-vol sizing** within each book (`rebalance_inv_vol=True`).
- **Weekly rebalance** (`rebalance_days=5`; owner decision §6).

### 2.2 Mapping to existing vs net-new

| Requirement | Existing engine capability | Net-new? |
|---|---|---|
| Long top-N / short bottom-N | `_process_rebalance` builds long from ranked-best, short from `reversed(ranked)` worst-first (`agent_simulator.py:1652–1674`) | **Reuse** |
| Dollar-neutral net≈0 | `split_gross_budgets(net_target, gross_target)` solves `long=½(gross+net)`, `short=½(gross−net)` (`portfolio_construction.py:255`); `net_target=0` ⇒ equal books. **BUG-1 fix (2026-06-02):** the long leg ignored the returned `long_budget` (sized to full NAV → ~76% net long); now both legs size to their budget / realized qualifying target-book count | **Reuse** (fixed wiring) |
| Gross ≤ 80% | `long_gross + short_gross` (sim ctor); set 0.40/0.40 | **Reuse** (set kwarg) |
| Sector cap both books | `apply_sector_cap` + `apply_sector_cap_shorts` (`agent_simulator.py:1657,1670`) | **Reuse** |
| Inverse-vol sizing | `compute_inverse_vol_weights` (`portfolio_construction.py:260`), `rebalance_inv_vol=True` | **Reuse** |
| Weekly rebalance | `rebalance_days` | **Reuse** (set 5) |
| Short collateral / proceeds accounting | `portfolio.short_collateral`, `_effective_cash` (`agent_simulator.py:1894–1896`) | **Reuse** |
| Per-side regime gates | `long_regime_fn` / `short_regime_fn` | **Reuse** (default off for the clean baseline) |
| **Turn shorts ON in the swing per-fold path** | `enable_shorts`, `long_gross`, `short_gross`, `short_target_n`, `short_min_adv`, `short_add_threshold`, `short_drop_threshold` exist on `AgentSimulator.__init__` (`agent_simulator.py:238–337`) **but are NOT passed by `SwingStrategy.run_fold`** (`swing.py:271–303` omits them) | **NET-NEW (S): plumb these through `SwingStrategy.__init__`/`run_fold` and the `walkforward_tier3` arg wiring (`:1498–1540`).** This is the single concrete simulator-side gap. |

### 2.3 Net-neutrality + sector-neutrality enforcement (honesty flags)
- **Dollar-neutral, not beta-neutral.** `net_target=0` enforces equal long/short *dollars*. Net **beta** ≈ 0 is then approximate (long and short books have similar but not identical aggregate beta). A separate `spy_beta_hedge` path exists in the sim if we later want hard beta-neutrality (owner decision §6: net-neutral vs beta-neutral vs both). **Recommendation:** ship dollar-neutral as the baseline; report realized net beta of the book per fold as a diagnostic; add beta-neutral as an ablation arm only if realized net beta is materially nonzero.
- **Sector cap ≠ sector-neutral.** `apply_sector_cap` bounds each sector's *weight*, it does not force long-sector-weight == short-sector-weight. With residualized labels (sector already regressed out of the target) the residual book should be roughly sector-balanced by construction, and the cap prevents concentration. **Recommendation:** ship sector-cap baseline; report realized per-sector net exposure as a diagnostic; only build strict sector-neutral matching if the diagnostic shows large residual sector tilts.

### 2.4 Ranker-as-scorer adapter (net-new, S)
The per-fold-retrained model is the natural scorer. Two options:
- **(Preferred) Use the native model path.** When `factor_scorer is None`, `AgentSimulator._pm_score`/`_pm_score_cached` already scores every symbol via the model's `predict` and returns `[(sym, conf)]` sorted desc — the rebalance engine takes top-N long / bottom-N short from this. **For residualization parity we must residualize the feature vectors before predict**, so we wrap the model in a thin `ResidualRankerScorer(model, resid_cfg, spy)` that implements `(day, symbols_data, vix_history) -> [(sym, conf)]`: assemble the day-`t` cross-section features (reuse `FeatureEngineer` / `feature_cache`), residualize them per §1.4, call `model.predict`, return sorted scores. Pass it as `factor_scorer`. This guarantees train/serve residualization parity (same `residualize.py` code both sides).
- The existing `FactorPortfolioScorer` returns `(sym, conf, direction)` and the rebalance engine derives shorts from reversed long ranking — so we keep direction implicit (bottom-of-rank = short), which matches the "same ranker, both tails" thesis. No need for scorer-emitted direction.

---

## 3. The CPCV harness (true per-fold OOS)

### 3.1 Reuse, don't fork
The swing per-fold CPCV path composes exactly:
- `run_cpcv(strategy, purge_days=SWING_PURGE_DAYS, embargo_days=..., n_folds, n_paths, total_years, train_years, ...)` (`cpcv.py:573`).
- `SwingStrategy(per_fold_retrain=True, rebalance_mode=True, factor_scorer=ResidualRankerScorer, rebalance_inv_vol=True, rebalance_days=5, rebalance_target_n=N, ...)`.
- Per fold, `SwingStrategy.run_fold` retrains via `SwingFoldRetrainer.train_for_window` (cached by `TrainWindowCache` so combos sharing a train window retrain once), sets `model.trained_through=tr_end`, runs the per-fold OOS guard (`assert_model_oos`), then runs `AgentSimulator` in rebalance L/S mode on `[te_start, te_end]`.
- **Determinism:** `run_cpcv` anchors `end_all = retrain_as_of()` (`cpcv.py:611`), and `SwingFoldRetrainer._seed_for(tr_start, tr_end)` gives reproducible per-window seeds — no re-run-until-favorable boundary risk.
- **Sacred holdout:** clamped at 5 layers; per-fold retrain calls `assert_no_sacred_holdout(tr_end)` (`retrainers.py:60`).
- **N_eff = n_folds** (conservative; `cpcv.py:135`), `unique_obs` multiplicity correction for DSR, the significance-first two-tier gate (`significance_gate_passed`).

### 3.2 What must change in the harness
1. **Plumb shorts (§2.2 net-new S):** add `enable_shorts`, `long_gross`, `short_gross`, `short_target_n`, `short_min_adv`, `short_add_threshold`, `short_drop_threshold` to `SwingStrategy.__init__` and forward them in the `AgentSimulator(...)` call inside `run_fold`. Add the matching `argparse` flags in `walkforward_tier3` and the `SwingStrategy(...)` construction block.
2. **`label_scheme="residual_rank"` propagation:** `SwingFoldRetrainer` reads `label_scheme` from `SWING_RETRAIN` base_config (`retrainers.py:68`); set it to `residual_rank` for this experiment (config flag, not code).
3. **Optional thin driver** `scripts/run_ranker_v2_cpcv.py` mirroring the `run_pead_cpcv.py` load-once/capture pattern: load bars once, build the `ResidualRankerScorer`, construct the L/S `SwingStrategy`, call `run_cpcv`, capture per-path daily-return vectors for significance (§4). The CLI route (`walkforward_tier3 --per-fold-retrain --rebalance-mode --enable-shorts ...`) also works once (1) lands; the dedicated driver is cleaner for capture + cost sweeps. **Recommendation:** dedicated driver — it makes the significance capture and cost-sweep loops first-class.

---

## 4. Power / significance (the whole point of breadth)

### 4.1 What breadth should do
Fundamental Law: `IR ≈ IC × √breadth × transfer-coeff`. A 5-position long-only book on a ~1000-name ranking has transfer-coeff ≈ 0.15 (discards ~85% of the cross-section). Going to 80/side dollar-neutral raises breadth from 5 → ~160 weakly-correlated bets. **For the same IC, the t-stat should rise materially.** This is exactly the significance remedy PEAD structurally lacks (§1.3 of the plan: PEAD's ~19 quarterly clusters can't self-certify; a daily-rebalanced 160-name book is the opposite regime).

### 4.2 Measure the significance GAIN explicitly
- **Primary lens (the right one here):** the standard CPCV **path t-stat with N_eff = n_folds** (`path_sharpe_tstat`, `cpcv.py:121`). This is NOT event-clustered (the ranker rebalances continuously, unlike PEAD's overlapping ~40-day earnings holds), so the daily-return path statistics are the appropriate significance measure. The conservative N_eff floor stays.
- **Secondary lens:** **Newey-West (HAC) t-stat on the book's daily return series** to handle the mild autocorrelation from 5-day rebalance overlap. This reuses the §1.3 significance module (`scripts/walkforward/significance.py`) being built for PEAD — feed it the captured per-path daily returns. (Requires persisting per-path daily-return vectors; `CPCVResult` currently keeps `path_n_obs` but not the vectors — small additive extension, same one PEAD needs.)
- **Diagnostic — the breadth A/B:** run the **same residualized ranker at 5-long (legacy frame)** vs **80/side dollar-neutral**, same folds/seeds, and report Δt-stat. If the ranker has any IC, breadth should lift t from ~0.17-territory toward ≥2.0. If t stays ~0 at high breadth, the IC genuinely is ~0 and the ranker is dead. **This A/B is the cleanest single number the experiment produces.**
- **Report the full honest panel** per fold/path: mean Sharpe, P5, %positive, path t-stat, NW t-stat, realized net beta, realized per-sector net exposure, avg deployment, turnover.

---

## 5. Pre-registered acceptance criterion

> Register BEFORE running (anti-fishing). Numbers are committed here.

**PRIMARY (PASS ⇒ "the signal was always there; long-only-5 was the killer"):**
- Dollar-neutral residualized high-breadth (recommend 60/side as the registered primary config) **net Sharpe ≥ 0.6** (mean across CPCV paths, net of §7 costs), **AND**
- **Path t-stat ≥ 2.0** (N_eff=n_folds) **AND NW t-stat ≥ 2.0** on the daily-return series (clears the PAPER significance tier), **AND**
- **% positive paths ≥ 0.75**, **P5 path Sharpe ≥ −0.40** (paper-tier floors), **AND**
- The **breadth A/B** shows a material t-stat lift from the 5-long frame to high-breadth (directional confirmation that breadth, not luck, drove it).
- **⇒ Trigger:** proceed to §3.3 short-interest + borrow data POC, then paper. Ranker becomes the price-factor leg of the Phase-3 north-star core book.

**STRETCH (capital-grade, do NOT require for the GO decision):** net Sharpe 0.8–1.2 with path t ≥ 2.5 and clean realized net beta — this would justify prioritizing the ranker over PEAD as the centerpiece.

**FAIL (⇒ "ranker is genuinely dead, proven correctly"):**
- High-breadth dollar-neutral residualized net Sharpe < 0.6 **OR** t-stat < 2.0 even at 80/side, **AND** the breadth A/B shows no material t lift.
- **⇒ Trigger:** stop investing in the swing ranker. Skip §1.5 (swing purge fix) and §3.5 swing validation-power upgrade as wasted effort on a corpse. Reallocate to the revisions-momentum sleeve (§3.2) and PEAD hardening. Document the death with the A/B as the evidentiary record.

**Ambiguous band (0.6 ≤ Sharpe but t in [1.5, 2.0]):** treat as "underpowered-but-promising" → increase n_folds/n_paths (cheaper than new data) and re-measure once before deciding; do NOT promote to paper on a sub-2.0 t.

---

## 6. Short leg in the BACKTEST without SI/borrow data

- **The backtest can short bottom-of-rank assuming R1K large-cap shortability.** §3.3 (FINRA SI + borrow feed) is the **LIVE-readiness + borrow-filter follow-on**, NOT a prerequisite for this backtest. The CPCV book just enters short positions at the open via the existing short path (`agent_simulator.py:1868–1904`), pays `short_borrow_rate_annual` (default 0.05) + `transaction_cost_pct` + slippage.
- **Known optimism to flag (revisit with §3.3):** the backtest assumes every bottom-of-rank R1K name is borrowable at the default 5%/yr. In reality a minority of small/low-float R1K names are hard-to-borrow or high-fee, and the *bottom* of a price-feature ranking is biased toward exactly those beaten-down, harder-to-borrow names. **Mitigations for the backtest:** (a) restrict the short universe to a liquidity floor via `short_min_adv` (tighter than the long `rebalance_min_adv`); (b) optionally exclude the smallest-cap decile from the short book even in backtest. Document any residual optimism; §3.3's borrow filter (extends `liquidity_filter`) makes it live-honest later. The §3.3 data is **explicitly not blocking** this experiment per ALPHA_V2_PLAN §3.1 sequencing — but the borrow-feasibility caveat must be stated in the result writeup.

---

## 7. Costs

- **Per-side, both legs, weekly turnover.** A dollar-neutral 60/side weekly-rebalanced book trades far more than the PEAD sleeve (~120 names, ~weekly churn on both sides). Use the §1.1 lesson (realistic per-side costs).
- **Registered baseline:** `transaction_cost_pct = 0.0005` (5 bps one-way) per leg + entry slippage. Note `ENTRY_SLIPPAGE_PCT`/`STOP_SLIPPAGE_PCT` are currently hardcoded constants (`agent_simulator.py:61–62`); §1.1 already requires lifting `ENTRY_SLIPPAGE_PCT` to a sim kwarg — reuse that so the cost sweep is faithful. Short side also pays `short_borrow_rate_annual`.
- **Mandatory cost-sensitivity sweep:** report net Sharpe + t-stat at one-way costs ∈ {2, 5, 10, 15, 20} bps (+ proportional slippage). The acceptance bar (§5) is measured at the **5 bps baseline**; a result that only clears at 2 bps is flagged as cost-fragile. Turnover (annualized two-sided) reported per config so the cost drag is auditable.

---

## 8. Effort breakdown, critical path, risks

### 8.1 Per-component effort

| Component | Effort | Reuse vs net-new |
|---|---|---|
| Plumb `enable_shorts`/gross/short-N through `SwingStrategy` + `walkforward_tier3` args | **S** | Net-new wiring; sim core unchanged |
| Ranker-as-scorer adapter (`ResidualRankerScorer`) | **S** | Net-new thin wrapper over existing `_pm_score`/`predict` |
| `app/ml/residualize.py` (PIT betas, cross-sectional OLS residualizer, size proxy) | **M** | Net-new |
| `label_scheme="residual_rank"` in `_compute_cs_thresholds` + worker + matrix post-pass (feature residualization) | **M–L** | Net-new; touches the per-day cross-sectional label path |
| Synthetic-leak test for residualization | **M** | Net-new; gating prerequisite |
| `scripts/run_ranker_v2_cpcv.py` driver + per-path daily-return capture | **M** | Net-new; mirrors `run_pead_cpcv.py` |
| NW/HAC significance on captured returns | **S** (if §1.3 module exists) / **M** (if not) | Shared with PEAD §1.3 |
| `CPCVResult` per-path daily-return-vector persistence | **S** | Net-new additive field (shared w/ §1.3) |
| Cost-sweep + diagnostics (net beta, sector net, turnover) reporting | **S** | Net-new reporting |

**Whole experiment estimate: M (book/wiring + driver) + M–L (residualization + leak test).** Matches ALPHA_V2_PLAN §3.1's "M (book/wiring) + M–L (residualization)".

### 8.2 Recommended sequencing (critical path)
1. **(S)** Plumb shorts through `SwingStrategy`/`walkforward_tier3`. Unblocks everything.
2. **(M) Spike A — book change alone.** Existing rank label (no residualization), dollar-neutral 60/side, weekly, inverse-vol, through per-fold CPCV. **If breadth alone moves +0.22 → ~0.5, residualization is the cherry; if it doesn't, residualization becomes the real test.** Cheapest decisive signal.
3. **(M–L)** Build `residualize.py` + `residual_rank` label + feature-residualization post-pass. **Gate on the synthetic-leak test before any number is trusted.**
4. **(S)** Ranker-as-scorer adapter with serve-side residualization parity.
5. **(M) Spike B — full residualized dollar-neutral book** through per-fold CPCV. Run the breadth A/B (5-long vs 80/side).
6. **(S)** Cost sweep + NW significance + diagnostics → evaluate against §5 pre-registered bar.

### 8.3 Biggest risks (ranked)
1. **Leakage in residualization (highest).** Betas/size must be strictly backward-looking; the per-day cross-sectional regression is PIT but trivial to break (e.g. fitting betas on the full window, or using t+H info in the label regression). **Mitigation:** the synthetic-leak test is a hard gate; strict-`<` everywhere; betas from ≤t only.
2. **Train/serve residualization skew.** If training residualizes features one way and the scorer another, the OOS number is garbage. **Mitigation:** single `residualize.py` used by both `training.py` and the scorer; add a parity unit test (same input → same residual).
3. **Size-factor feasibility/quality.** FMP shares coverage is incomplete and as-reported; a noisy size column can inject leakage or noise. **Mitigation:** size is an ablation arm, not default; ADV-proxy fallback; drop names without PIT shares from the size column only.
4. **Turnover/cost dominance.** Weekly two-sided rebalance on 120 names can be cost-eaten. **Mitigation:** cost sweep is mandatory; acceptance measured at 5 bps; consider monthly rebalance arm (owner decision §6).
5. **Simulator gaps (lowest — mostly closed).** The L/S engine is built; the only gap is plumbing `enable_shorts` through the swing strategy (S). Sector-cap is a cap not strict neutrality, and net-neutral is dollar not beta — both are **diagnostic-reported and ablation-armed**, not blockers.

---

## 9. Genuine decisions for the human (with recommendations)

| # | Decision | Options | Recommendation |
|---|---|---|---|
| Q1 | **Breadth (names/side)** | 40 vs 60 vs 80 | **60/side primary** (registered). Run 40 and 80 as breadth-sensitivity arms to plot t-stat vs breadth — that curve is itself evidence. Owner locked "40–60/side" for paper (§7-Q3); 60 is the upper edge of that and maximizes power for the decisive test. |
| Q2 | **Rebalance frequency** | weekly vs monthly | **Weekly primary** (matches the north-star core-book spec). Add **monthly** as a cost-mitigation arm — if weekly only clears at <5 bps but monthly clears at 5 bps, monthly is the deployable answer. |
| Q3 | **Factor set (and size proxy)** | market+sector; +size(mktcap); +size(ADV); drop size | **Default: market+sector.** Run **+size(mktcap from PIT FMP shares)** and **+size(ADV proxy)** as two ablation arms. Do NOT make size a hard requirement — it's the leakiest, lowest-coverage factor. Decide size inclusion *after* seeing whether it changes the residual-IC. |
| Q4 | **Net-neutral vs beta-neutral vs both** | dollar-neutral (net≈0); beta-neutral (spy_beta_hedge); both | **Dollar-neutral primary** (engine-native, zero extra risk). Report realized net beta; only add beta-neutral arm if realized net beta is materially nonzero (e.g. |β| > 0.15). |
| Q5 | **Acceptance bar** | Sharpe ≥0.6 & t≥2.0 (registered) vs stricter | **Adopt §5 as registered** (Sharpe ≥0.6 net @5bps AND path-t≥2.0 AND NW-t≥2.0 AND %pos≥0.75). This is the owner's "0.6–0.8" from §3.1, made precise with the significance requirement breadth is meant to deliver. |
| Q6 | **Driver: dedicated script vs CLI flags** | `run_ranker_v2_cpcv.py` vs `walkforward_tier3 --rebalance-mode --enable-shorts` | **Dedicated driver** for first-class significance capture + cost sweeps; CLI route validated as a cross-check once shorts are plumbed. |

---

## 10. Changelog / docs obligations

This experiment touches files under the PIPELINE_ARCHITECTURE update rule (`scripts/walkforward/*`, `app/ml/training.py`, `app/backtesting/agent_simulator.py` wiring). On implementation:
- Update `docs/living/PIPELINE_ARCHITECTURE.md` changelog + gate inventory (new `residual_rank` label, L/S swing rebalance mode, any new feature flags) + Known Limitations (size-factor coverage, dollar-vs-beta-neutral, borrow optimism).
- Log every CPCV run in `docs/living/ML_EXPERIMENT_LOG.md` (the breadth A/B and cost sweep are individual logged runs).
- Update `docs/living/MODEL_STATUS.md` with the verdict (GO/dead) and the A/B evidence.

*Design only — no code written. Next step on owner sign-off: implement step 1 (plumb shorts) + step 2 (Spike A book-change-alone) to get the cheapest decisive signal before building residualization.*
