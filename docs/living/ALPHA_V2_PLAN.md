# MrTrader — Alpha v2 Plan

**Status:** 🟢 **ACTIVE — human decisions locked 2026-06-02 (see §7).** Workstreams may proceed per the sequenced roadmap (§6); each still goes through design→implement→Opus-review→test→merge.
**Author:** Staff quant/eng pass over the synthesized directive, ground-truthed against the codebase.
**Date:** 2026-06-02
**Inputs:** `docs/archive/llm-reviews/2026-06-02/SYNTHESIZED_BUILD_DIRECTIVE.md` (the 5-LLM synthesis — a high-quality INPUT, not gospel), validated against the actual implementation.

**Locked decisions (§7):** (1) **Short interest first** — FINRA SI + borrow feed before options; cheap read-only IV later; defer options-trading infra. (2) **Carve a recent historical holdout** (reserve ~post-2024-06-01 for one-shot promotion eval; dev uses DSR-controlled full-window CPCV). (3) **Dollar-neutral shorting approved in paper** (gated on the borrow-data POC; gross ≤80%, 40–60/side). (4) **Keep live PEAD running** in paper while the de-risk tests run; pause ONLY if §1.2 leave-one-crisis-out fails.

---

## 0. Executive summary

**Thesis.** The honest OOS numbers are real: long-only price-feature ML is exhausted (swing per-fold CPCV +0.22 / t=0.17; intraday −2.80 / t=−6.85, cost-dominated), and the *one* live edge is PEAD R1K (+0.546 / t=2.26, 95% paths positive, clears the 0.50 PAPER gate, short of the 0.80 promotion bar). The next phase is **structural, not signal-hunting**: (1) de-risk and re-measure PEAD honestly before trusting it with capital; (2) test whether the "dead" cross-sectional ranker was strangled by a 5-position long-only book rather than genuinely edgeless, by re-running it **dollar-neutral, sector-neutral, high-breadth, on residualized features+labels**; (3) align the live system to the validated backtest by porting/ablating the live overlays into the simulator. We do **not** chase options-vol selling, signal inversion, or AR(1) N_eff inflation.

**Honest target.** Net book Sharpe **0.6–1.2** at the dollar-neutral-book level. A clean 0.8–1.2 neutral book on free/cheap data is a genuinely good result for this account size; 2+ requires infrastructure (TAQ, colo, sec-lending) we do not have.

**What changes (one paragraph).** PEAD stays the sole live edge and gets stress-tested (cost sweep, leave-one-crisis-out, event-clustered significance, SUE conversion, overlay ablation) before any capital decision. In parallel we spin up the decisive dollar-neutral ranker experiment — which is *less* net-new than the directive assumes because `AgentSimulator` already has a dollar-neutral-capable L/S rebalance engine. The big genuine lifts are (a) feature+label residualization in `training.py`, (b) cheap short-interest/borrow data to run the short leg responsibly, and (c) the event-clustered significance machinery. Everything goes through the existing design→implement→Opus-review→test→merge loop; CPCV/gate integrity and the no-look-ahead invariant are sacred.

---

## 1. Ground-truth corrections (the most important section)

The directive makes specific factual claims about *our* system. Verified against the code:

| # | Directive claim | Verdict | Reality (file:line) |
|---|---|---|---|
| 1 | "Purge = 85 calendar days vs ~62-day test windows → over-purging." | ⚠️ **PARTLY WRONG — conflates two pipelines** | 85d purge is **swing-only** (`SWING_PURGE_DAYS=85 = FEATURE_LOOKBACK(60)+LABEL_HORIZON(20)+buffer(5)`, `retrain_config.py:591-616`; invariant enforced at import). **PEAD CPCV uses `purge_days=10, embargo_days=10`** (`run_pead_cpcv.py:266-271`). PEAD is **not** over-purged. The "62-day test window" figure is the swing per-fold note in `MODEL_STATUS.md:128-129`. So the over-purge critique applies to the swing ranker only, and it is driven by the **feature lookback (60d), not the label horizon** — shortening 60d technicals to 20d is what would let purge shrink toward label_horizon(20)+buffer. |
| 2 | "Sacred holdout 2026-11-09 is in the FUTURE → no historical holdout; every backtest is in-sample at the meta level." | ✅ **CONFIRMED (date) / ⚠️ framing over-claims** | `SACRED_HOLDOUT_START="2026-11-09"` (`retrain_config.py:525`), enforced at 5 layers incl. `run_cpcv` and `retrain_as_of()` clamp. It IS in the future, so there is no held-out *historical* slice. BUT "every backtest is in-sample at the meta level" overstates it: per-fold CPCV with true per-fold retraining (`is_true_walkforward`) is genuinely OOS *within* the historical window (each fold trains on its own causal `[tr_start,tr_end]`, purged). The honest framing: **we have legitimate cross-validated OOS, but no single untouched one-shot holdout** — so cumulative selection bias across experiments is uncontrolled (which is exactly what DSR with `N_TRIALS_TESTED=300` is for). Decision needed (§7-Q2). |
| 3 | "5-position long-only book discards ~85% of cross-sectional edge." | ✅ **CONFIRMED (caps) / breadth diagnosis plausible** | `RiskLimits.MAX_OPEN_POSITIONS=5`, `MAX_POSITION_SIZE_PCT=0.05`, `MAX_SECTOR_CONCENTRATION_PCT=0.20` (`risk_rules.py:19-23`). The swing CPCV ran through `AgentSimulator` with these caps (signal mode), so the +0.22/t=0.17 was produced through a ~5-name long-only frame. The Fundamental-Law argument (can't distinguish "no signal" from "good signal, near-zero transfer") is a **valid hypothesis** worth the decisive test (§3.1) — but note LX9-A (beta-neutral, +0.031, F2=−0.70 per `MODEL_STATUS`) is weak prior evidence the ranker may still be dead. The experiment resolves it honestly either way. |
| 4 | Simulator doesn't support a dollar-neutral, sector-neutral, 40–80/side book → net-new simulator work. | ❌ **WRONG — already substantially built** | `AgentSimulator` has a full L/S rebalance engine: `enable_shorts`, `_process_rebalance` (`agent_simulator.py:1596`), `split_gross_budgets(net_target, gross_target)`, separate long/short books, `apply_sector_cap`/`apply_sector_cap_shorts`, `compute_inverse_vol_weights`, per-side `long_regime_fn`/`short_regime_fn`, short collateral accounting, configurable `short_target_n`/`long_gross`/`short_gross`. Setting `net_target≈0`, `rebalance_target_n=short_target_n=40–80`, `rebalance_inv_vol=True` gives a dollar-neutral, sector-capped, inverse-vol book **today**. This is an **integration/wiring** task (wire the ranker as `factor_scorer` + a thin CPCV driver like `run_pead_cpcv.py`), **not** a simulator rewrite. Lift drops from L to **M**. Caveats: sector-cap is a cap, not strict sector-neutrality; net-beta≈0 is dollar-neutral, not beta-neutral (separate `spy_beta_hedge` path exists). |
| 5 | Training can residualize features AND labels (idiosyncratic-return target). | ⚠️ **PARTIAL — label residualization exists in spirit; feature residualization is net-new** | `training.py` already has `label_scheme` variants `sector_relative` (label = beat sector-relative threshold, `:389-396`) and `return_regression` (raw forward return, `:362-363`). The current production label is **cross-sectional rank** (`lambdarank`/`cross_sectional`), NOT raw return. A true Gemini-style residualization (regress forward return on market+sector+size, train on the **residual** return; and residualize *features* the same way) is **new code**: a residualization step in `_build_window_label`/feature assembly. Medium lift, well-scoped, but real. |
| 6 | SUE feasible from FMP (trailing surprises and/or estimate dispersion). | ⚠️ **HALF — trailing surprises YES, dispersion NO** | `fmp_provider.get_earnings_features_at` returns `fmp_surprise_1q` and `fmp_surprise_2q_avg` (`:99-126`); the surprise is the **noisy** `(actual−est)/|est|` clipped to [−1,1] (`:60,83`) — exactly the formula the directive wants replaced. We have the raw per-quarter actual/estimate history to compute **SUE = (actual−est)/σ(trailing 8q surprises)** with modest new wrangling (extend the provider to return the trailing surprise series, compute σ). **Analyst-estimate dispersion is NOT exposed** (`get_analyst_features_at` returns only upgrade/downgrade counts, `:180-213`). So: SUE via trailing-surprise σ = feasible now; dispersion-denominator SUE = needs a new FMP endpoint. |
| 7 | Cost model is a single per-run constant (cost sweep trivial). | ✅ **CONFIRMED** | PEAD: `transaction_cost_pct=0.0005` passed to `PEADStrategy` and into `AgentSimulator` per run (`run_pead_cpcv.py:253`, `:153`). Slippage is **separate and hardcoded** in the simulator: `ENTRY_SLIPPAGE_PCT=0.0003`, `STOP_SLIPPAGE_PCT=0.0005` (`agent_simulator.py:61-62`), plus `short_borrow_rate_annual` default 0.05. So the cost sweep over `transaction_cost_pct` is trivial; a faithful earnings-window cost sweep should **also** vary entry slippage (currently a constant), which is a tiny parameterization change. |
| 8 | `path_sharpe_tstat` uses N_eff=n_folds; Newey-West / block bootstrap is net-new. | ✅ **CONFIRMED** | `CPCVResult.path_sharpe_tstat` uses `n_eff = max(self.n_folds, 1)` (`cpcv.py:111-126`), gate `CPCV_MIN_TSTAT=2.0` (off by default, `require_tstat_gate=False`). There is **no** Newey-West HAC or event-clustered block bootstrap anywhere. Net-new code; natural home is a new `scripts/walkforward/significance.py` consuming the per-path/per-day return series (which means we must **persist** per-path daily returns — `CPCVResult` currently keeps `path_n_obs` but not the daily-return vectors; small extension). |
| 9 | Live overlays exist only in the live PM; the backtest is already "clean"; ablation = porting overlays INTO the sim. | ✅ **CONFIRMED** | Overlays live in `portfolio_manager.py`: `_regime_sizing_multiplier` (`:4063`), `_compute_opportunity_score` (`:2110`), macro block via `macro_ctx.block_new_entries`, NIS sizing, conviction sizing — all inside `_analyze_swing_pead` (`:1594`). The PEAD CPCV path runs `AgentSimulator(..., no_prefilters=True)` with **none** of these gates active, so the +0.546 IS the clean number. The sim **already has hooks** (`use_opportunity_score`, `macro_blocked_dates`, `benign_blocked_dates`, regime fns) — they are just not exercised in `run_pead_cpcv.py`. Ablation = wire each overlay into the PEAD CPCV driver one at a time and measure ΔSharpe. Confirmed correct, and **lighter than implied** (hooks exist). |
| 10 | Leave-one-crisis-out has 4 crises (2018Q4, COVID-2020, 2022, Aug-2024). | ⚠️ **WRONG — 2018Q4 is out of window → 3 crises** | PEAD CPCV uses `TOTAL_YEARS=6`, `end_all=datetime.now()` ≈ 2026-06, so the window is **≈ 2020-05 → 2026-06**. **2018Q4 is NOT in the window.** In-window stress episodes: **COVID-2020 (tail-only, window starts mid-crash), 2022 bear, Aug-2024 spike, Apr-2025 tariff shock.** So leave-one-crisis-out has effectively **3 clean episodes (2022, Aug-2024, Apr-2025)** plus a partial COVID. The directive's "drop COVID alone" test is partly moot because the window barely contains COVID. The VIX>30 block is also **OFF in the validated config** (`vix_block_all=30` but `vix_conf_ref=100`, long-only; `run_pead_cpcv.py:232-242`) — the +0.546 is long-only **without** an active short-squeeze guard, so the "VIX>30 block carries the entire edge" framing from the brief needs re-checking against which config actually scored +0.546 (see §7-Q5, an uncertainty to confirm). |
| 11 | VIX3M available for the term-structure overlay. | ✅ **CONFIRMED** | `^VIX3M` is cached in `data/macro/macro_history.parquet` alongside `^VIX/HYG/IEF/RSP/SPY` (`macro_history.py:4,12,36`), readable PIT via `get_macro_at(as_of, "vix3m")`. The VIX/VIX3M term-structure overlay (§3.6) is data-ready. |

**Net takeaway:** the directive is right on the big structural calls (breadth, neutrality, sim-live alignment, honest event significance, don't-sell-vol) but wrong/imprecise on several *implementation* facts — most importantly it **over-states the simulator lift** (#4, already built) and **mis-attributes the over-purge** to PEAD when it is a swing-only issue (#1), and it **over-counts the crises** (#10).

---

## 2. Phase 1 — de-risk + re-measure (no new data/redesign)

All Phase-1 items operate on the **existing** PEAD CPCV harness (`run_pead_cpcv.py` + `AgentSimulator` + `cpcv.py`). None require new data except the live-fill audit (which needs live PEAD history that barely exists yet).

### 1.1 Cost-sensitivity table + (deferred) live-fill audit
- **Goal:** how much of +0.546 survives realistic earnings-window costs.
- **Design:** parameterize `run_pead_cpcv.py` to sweep `transaction_cost_pct ∈ {2,5,10,20,35,50 bps}` AND scale `ENTRY_SLIPPAGE_PCT` proportionally (currently a hardcoded 3 bps const in `agent_simulator.py:62` — lift it to a sim kwarg so the sweep is faithful for earnings-window spread blowout). Emit a table of mean/P5/t-stat per cost level. Trivial because cost is already a single per-run constant (correction #7).
- **Effort:** **S.** **Deps:** none. **Acceptance:** net mean Sharpe ≥ 0.40 at 20 bps one-way; if it breaks at ~10 bps, restrict PEAD to tightest-spread names or pause.
- **Live-fill audit:** **BLOCKED on data.** PEAD is *wired but NOT activated* (`MODEL_STATUS.md:11-45`); the `pead_tracking.db` observability exists but has near-zero history. Defer the realized-slippage-vs-modeled-next-open comparison until ≥4 weeks of live fills accrue. Note in roadmap as ongoing.

### 1.2 PEAD crisis-block robustness (highest-priority single test)
- **Goal:** is the edge real or a threshold sidestepping a couple of episodes?
- **Design:** (a) **Leave-one-crisis-out** over the **3 in-window episodes** (2022, Aug-2024, Apr-2025) + partial COVID (correction #10) — drop each episode's dates from the CPCV test returns and recompute mean/t. (b) **Threshold sweep** on `vix_block_all ∈ {25,28,30,33,35}` and `vix_block_short` — stable across range = robust. (c) **Generic regime control**: replace the discrete VIX block with **portfolio vol-targeting** (scale gross to a target annualized vol) and/or an SPY<200dma trend cut — both expressible via the sim's `long_regime_fn` hook and a new vol-target scaler. **First confirm which config actually produced +0.546** (§7-Q5): the committed config has the VIX block effectively OFF, so the brief's "block carries the edge" may be about a *different* config.
- **Effort:** **M.** **Deps:** none (VIX block confirmed active — §7-Q5 resolved). **Acceptance:** edge survives a *generic* regime control AND leave-one-crisis-out; if only the exact VIX>30 threshold works or one episode carries it, **pause live PEAD** (§7-Q4).

### 1.3 Event-clustered block bootstrap + Newey-West
- **Goal:** a significance estimate whose unit of independence is the earnings *event/season*, not the fold/day. The directive and DeepSeek are right that overlapping ~40-day holds make the true independent-bet count ≈ seasons/years, so t=2.26 is optimistic.
- **Design:** new `scripts/walkforward/significance.py`. (a) **Event-clustered block bootstrap**: resample earnings-event clusters by quarter/season with replacement to build the null Sharpe distribution. (b) **Newey-West (HAC) t-stat** on the overlapping daily path-return series. Requires persisting per-path **daily return vectors** in `CPCVResult` (currently only `path_n_obs` is kept — small extension, correction #8). Keep `N_eff=n_folds` as the conservative floor; do **NOT** adopt AR(1) N_eff (§5).
- **Effort:** **M–L** (new module + CPCVResult plumbing + tests). **Deps:** none. **Acceptance:** PEAD block-bootstrap p<0.05 **and** Newey-West t≥2.0; expect a weaker honest number than 2.26.

### 1.4 Convert PEAD to SUE
- **Goal:** replace the noisy `(actual−est)/|est|` (explodes for tiny estimates) with SUE.
- **Design:** extend `fmp_provider.get_earnings_features_at` to also return the **trailing surprise series** (we already store per-quarter actual/estimate, correction #6) and compute **SUE = (actual−est)/σ(trailing 8q surprises)**. Add a `surprise_mode="sue"` switch to `PEADScorer` (sort/enter on SUE deciles, not a flat ±5% cutoff). Dispersion-denominator variant is deferred (no FMP dispersion endpoint yet).
- **Effort:** **M.** **Deps:** none (data already cached). **Acceptance:** SUE-PEAD net Sharpe ≥ current with less fat-tailed path distribution.

### 1.5 Fix the harness (purge + holdout)
- **Goal:** stop starving swing-ranker power; resolve holdout ambiguity. **Note:** this is a **swing-pipeline** fix, not PEAD (correction #1).
- **Design:** (a) **Shorten swing feature lookbacks** (60d→20d technicals) so `FEATURE_LOOKBACK_DAYS` drops and `SWING_PURGE_DAYS` can fall toward `LABEL_HORIZON(20)+buffer ≈ 25–30d` — but the invariant `_assert_purge_horizon_invariant()` (`retrain_config.py:599`) must stay satisfied, and the 252d momentum feature is a **label-side** lookback that does not gate purge (it's excluded per `:596`). Validate with a synthetic-leak test before/after. (b) **Holdout policy:** decide (§7-Q2) between carving a real historical holdout now (e.g. reserve post-2024-06-01) vs. relabeling 2026-11-09 as "forward paper validation," and enforce the chosen policy with an assertion.
- **Effort:** **M** (mostly the leak-test + retrain). **Deps:** §7-Q2 decision; only matters if the swing ranker is revived (§3.1). **Acceptance:** swing purge ≤30d with no synthetic-leak detection; holdout policy documented + asserted.

### 1.6 Overlay ablation harness → demote RM to safety net
- **Goal:** prove each live overlay adds Sharpe or strip it; make live==backtest.
- **Design:** the sim already has the hooks (correction #9). Extend `run_pead_cpcv.py` into an ablation driver: baseline clean PEAD vs. PEAD+each overlay (regime sizing via `long_regime_fn`, opportunity gate via `use_opportunity_score`, macro block via `macro_blocked_dates`, NIS/conviction sizing) head-to-head in the **same CPCV**. Adopt the rule: **every sizing/concentration rule migrates into the PM/sim and is mirrored in the backtest; RM keeps only catastrophic safety functions** (kill switch, margin breach, data-dropout, fat-finger). "If you can't backtest a rule, you can't run it live."
- **Effort:** **M.** **Deps:** none. **Acceptance:** each retained overlay improves net Sharpe **or** cuts max-DD ≥15% without cutting Sharpe >0.1; everything else removed; the live `_analyze_swing_pead` config == the validated CPCV config.

---

## 3. Phase 2 — structural (spike before commitment)

### 3.1 ⭐ Dollar-neutral residualized ranker re-test (THE decisive experiment)
- **Goal:** was the cross-sectional ML ever dead, or strangled by the 5-position long-only frame?
- **Design approach (this codebase):** two pieces.
  1. **Book change — mostly wiring (correction #4).** Drive `AgentSimulator` in `rebalance_mode=True, enable_shorts=True` with `rebalance_target_n=short_target_n=40–80`, `split_gross_budgets(net_target≈0, gross_target≤0.80)`, `rebalance_inv_vol=True`, sector caps on both books. The short side = bottom of the **same** ranker (standard market-neutral, NOT the failed QualityShort thesis). Wire the ranker as `factor_scorer` and add a thin CPCV driver mirroring `run_pead_cpcv.py`. **Net-new: a ~150-line driver + ranker-as-scorer adapter.** Modify: none of the sim core.
  2. **Residualization — net-new (correction #5).** Add a residualization stage in `training.py`: regress forward return on market+sector(+size) factors, train on the **residual**; residualize features the same way. New `label_scheme="residual_rank"` + a feature-residualization helper.
- **Net-new vs modify:** new = CPCV driver, ranker adapter, residualization stage, factor-loading computation; modify = `training.py` label path, sim driver kwargs only.
- **Spike first:** run the **book change alone** (existing rank label, dollar-neutral 60/side) before building residualization — if breadth alone moves +0.22 toward 0.5+, residualization is the cherry; if it doesn't, residualization is the test of whether the ranker has *any* idiosyncratic signal.
- **Effort:** **M** (book/wiring) + **M–L** (residualization). **Deps:** §3.3 short-interest/borrow data to run the short leg *responsibly* in paper; §7-Q3 shorting appetite. **Acceptance:** dollar-neutral residualized high-breadth net Sharpe clears ~0.6–0.8 → signal was always there; if it still fails → ranker genuinely dead, proven correctly.

### 3.2 Analyst-revision momentum sleeve (information-diffusion)
- **Goal:** harvest the best untapped *free* edge; combine same-mechanism signals to raise breadth of one durable effect.
- **Design:** promote `fmp_analyst_momentum_30d` (`fmp_provider.py:207`, currently only a PEAD sub-filter) to a standalone cross-sectional signal via a new `factor_scorer`. Then combine revisions + SUE-PEAD + drift into one diffusion sleeve (z-score composite) — same underreaction mechanism, so stacking raises breadth rather than diluting.
- **Effort:** **S–M** (data already exposed). **Deps:** §1.4 (SUE). **Acceptance:** revisions positive standalone (block-bootstrap p<0.05); combined sleeve Sharpe ≥ best single component, lower variance.

### 3.3 Short-interest data acquisition (cheap) — standalone + enables the short leg
- **Goal:** unlock §3.1's short leg responsibly + add a documented large-cap edge.
- **Design:** acquire **FINRA short interest** (near-free, semi-monthly) + a modest borrow-availability/cost feed; new provider module + parquet cache mirroring `macro_history.py`. Test ΔSI, days-to-cover, crowded-short as a cross-sectional signal (Boehmer-Jones-Zhang). Borrow data feeds a short-leg feasibility filter (you can't responsibly short what's hard/expensive to borrow).
- **Effort:** **M** (new pipeline). **Deps:** §7-Q1 budget. **Acceptance:** POC short-interest signal positive IC; borrow filter integrated into the short-leg eligibility (extends `liquidity_filter`).

### 3.4 Read-only options IV as a PEAD signal-quality filter (NOT trading)
- **Goal:** salvageable part of the options enthusiasm; zero options trading.
- **Design:** cheap historical+live ATM-IV/implied-move feed (new provider). Use implied move to gate/scale PEAD confidence in `PEADScorer` (a +X% surprise against a much larger priced-in move is weaker). No options positions, no Greeks.
- **Effort:** **M** (new data feed + scorer hook). **Deps:** §7-Q1 budget. **Acceptance:** IV-gated PEAD IC > price-only PEAD IC.

### 3.5 Validation-power upgrade
- **Goal:** make the CAPITAL gate reachable by a real edge without weakening it.
- **Design:** raise CPCV to **k=12–15** with higher combinatoric depth (more unique paths; keep `N_eff=n_folds` floor and the `unique_obs` multiplicity correction in `cpcv.py:152-165`). Add **eigenvalue-based effective-trial count** for DSR (decompose the correlation matrix of the ~300 trial return streams so correlated HPO variants don't count as full independent trials) — requires **persisting trial return streams** (not currently stored). The shortened swing purge (§1.5) is the prerequisite for more folds there; PEAD's purge=10 already permits more folds.
- **Effort:** **M** (k bump) + **L** (eigenvalue trials + stream persistence). **Deps:** §1.5 (swing). **Acceptance:** a synthetic true-Sharpe≈0.7 edge clears the upgraded gate; correlated trial variants no longer over-deflate DSR.

### 3.6 Regime overlay redesign (general replacement for VIX>30)
- **Goal:** replace the hand-tuned threshold with a regime-general, independently-validated control.
- **Design:** **VIX/VIX3M term structure** (data-ready, correction #11): backwardation (ratio>1)=stress→cut gross; contango=normal. Combine with portfolio vol-targeting + SPY-trend. Expressed via the sim's `long_regime_fn`/`short_regime_fn` hooks + a `get_macro_at("vix3m")` reader. Validate the overlay as an independent Sharpe-improver, not bolted on by faith.
- **Effort:** **M.** **Deps:** §1.2. **Acceptance:** term-structure+vol-target overlay improves OOS Sharpe of the core book independently and would have de-risked Aug-2024 without being fit to specific crises.

### 3.7 (Lower priority) Honest reversal tests
- **Design:** (a) short-term (1–5d) reversal, large-cap, dollar-neutral 40–80/side via the §3.1 book, realistic costs + explicit bid-ask-bounce modeling (much close-to-close "reversal" is microstructure noise we can't capture). (b) overnight reversal after large intraday down-moves — **needs intraday data + PIT news filter; spike the data dependency first** (it is NOT cleanly validated on daily panels). No 5-name versions; no guaranteed t-stats.
- **Effort:** **M.** **Deps:** §3.1 infra. **Acceptance:** clears the honest gate net of realistic costs, or documented dead.

---

## 4. Phase 3 — north-star integration

**Composition.** A single dollar-neutral, sector-neutral multi-factor **core book** + an event **sleeve** + a regime **overlay**, with the RM demoted to a safety net.

- **Core book (dollar-neutral, sector-neutral multi-factor):** composite z-score of [revisions momentum + SUE-PEAD tilt + 12-1 momentum + quality + low-vol], sector-neutralized, long top 40–80 / short bottom 40–80, net≈0, gross ≤80% NAV, inverse-vol sizing, weekly rebalance. Built on the **existing** `_process_rebalance` engine (correction #4) with a composite `factor_scorer`.
- **Event sleeve (PEAD-family):** SUE-PEAD + standalone revisions drift, 20–40 positions, ~20–40d hold, crisis-controlled via vol-target/term-structure (§3.6), not a fitted VIX threshold.
- **Overlay:** VIX/VIX3M term-structure + vol-target + SPY-trend, validated independently (§3.6), applied via `long_regime_fn`/`short_regime_fn`.
- **RM = asymmetric safety net:** kill switch, margin breach, data-dropout, fat-finger only. All construction logic lives in the PM/sim and is mirrored in backtest (the §1.6 rule, made permanent).

**Migration from current architecture.** (1) Phase 1 hardens PEAD + ablates overlays → live PEAD == backtest. (2) §3.1 proves/kills the ranker; if it lives, it becomes the core book's price-factor component. (3) §3.2/§3.4/§3.6 add the revisions sleeve, IV filter, and validated overlay. (4) Compose via a single composite `factor_scorer` feeding the rebalance engine. (5) Strip every unbacktested RM/PM rule. **Kill list:** intraday 5-min, thesis shorts, the 5-position cap (`MAX_OPEN_POSITIONS=5`), the long-only default, the discrete VIX block, every unvalidated live overlay.

---

## 5. DO NOT BUILD (re-justified for our constraints)

- **Selling options vol around earnings (IV-crush "free money").** Reject. We have **no options data, no Greeks risk system, no options execution**. It is a negative-skew steamroller (win ~80%, lose it all on the gap), the *opposite* tail of everything validated, and earnings-window single-name option spreads are wide (5–15% of premium) — "bypasses cost drag" is false. Keep only the **read-only IV-as-signal** filter (§3.4).
- **"Invert the dead intraday signal → +2.8 Sharpe."** Mathematically illiterate. Our intraday −2.80 is **net of costs**; gross PF was 0.94 (`MODEL_STATUS.md:109`) — already below break-even *before* costs. Inverting a cost-dominated, sub-break-even gross signal and paying costs on the other side yields another negative. No TAQ/L2/colo → the horizon is dead for our infra. Leave it dead.
- **Quenouille AR(1) N_eff on daily returns.** Reject. It produces N_eff≈900–1200 and t≈25–40, which would re-inflate exactly what 13 audit rounds deflated. Keep `N_eff=n_folds` (`cpcv.py:125`) as the conservative floor; refine only with Newey-West / event-clustered block bootstrap (§1.3).
- **Institutional-priced data now ($25–100k Compustat PIT / S3 / Ortex).** Scaled for a $50M fund; we are $100k paper. Start cheap/free: FINRA SI, low-cost IV (§3.3/§3.4).
- **5-name reversal book / any "guaranteed t-stat."** A 5-name reversal contradicts the entire breadth argument. Reversal is a real *category* (§3.7) — test it high-breadth with honest costs and microstructure modeling, or not at all. Treat every projected/guaranteed Sharpe across all reviews as a hypothesis.

---

## 6. Sequenced roadmap

| Phase | Workstream | Effort | Depends on | Acceptance gate | Blocking decision |
|---|---|---|---|---|---|
| P1 | 1.1 Cost-sensitivity sweep | S | — | net SR ≥0.40 @20bps | ✅ **DONE 2026-06-02 — PASS (+0.402@20bps; anchor +0.548≈+0.546; cost-robust but t<2.0/%pos 57% by 20bps). Working assumption 15bps→SR≈0.45. See ML_EXPERIMENT_LOG.** |
| P1 | 1.2 Crisis-block robustness | M | — | survives leave-one-crisis-out + generic regime control | Q4 (pause-if-fail) |
| P1 | 1.3 Event-clustered bootstrap + NW | M–L | — | block-boot p<0.05 & NW t≥2.0 | — |
| P1 | 1.4 SUE conversion | M | — | SUE-PEAD ≥ current, cleaner tails | — |
| P1 | 1.5 Swing purge + holdout policy | M | Q2 | purge ≤30d, no synthetic leak; policy asserted | Q2 |
| P1 | 1.6 Overlay ablation → RM safety net | M | — | each overlay earns its place; live==backtest | — |
| P2 | 3.3 Short-interest + borrow POC | M | Q1 | positive IC; borrow filter live | Q1 |
| P2 | **3.1 Dollar-neutral residualized ranker** | M(+M–L) | 3.3, Q3 | SR ≥0.6–0.8 net or proven dead | Q3 |
| P2 | 3.2 Revisions + diffusion sleeve | S–M | 1.4 | revisions p<0.05; sleeve ≥ best component | — |
| P2 | 3.4 Read-only IV PEAD filter | M | Q1 | IV-gated IC > price-only IC | Q1 |
| P2 | 3.5 Validation-power upgrade | M(+L) | 1.5 | synthetic 0.7-SR edge clears gate | — |
| P2 | 3.6 Regime overlay redesign | M | 1.2 | overlay improves OOS SR independently | — |
| P2+ | 3.7 Reversal tests (high-breadth) | M | 3.1 infra | clears gate or documented dead | — |
| P3 | §4 north-star integration | L | P1+P2 | net book SR 0.6–1.2 in CPCV + forward paper | Q1–Q4 |

---

## 7. Open questions & human decisions

> Decisions LOCKED by the owner 2026-06-02. Recorded below.

1. **Data budget allocation (the one real disagreement).** Four reviews rank options #1; the structural argument ranks short interest #1.
   - ✅ **DECIDED: (a) short-interest first.** FINRA SI + a modest borrow feed (near-free) before anything else — it's a standalone edge AND unlocks the dollar-neutral short leg (§3.1). A cheap read-only IV feed comes second (§3.4). Full options-*trading* infra deferred. (Dollar cap/timing to be set when the SI POC is scoped.)

2. **Historical holdout vs forward-paper-only (§1.5, correction #2).**
   - ✅ **DECIDED: (a) carve a recent historical holdout.** Reserve ~post-2024-06-01 as an untouched one-shot holdout for the *promotion candidate only*; development continues on full-window DSR-controlled CPCV. Enforce with an assertion (touching the holdout in dev raises).

3. **Operational appetite for shorting (§3.1, north star).**
   - ✅ **DECIDED: yes — dollar-neutral in paper,** gated on the §3.3 borrow-data POC. Conservative gross (≤80%), 40–60 names/side, net beta ≈ 0. This is the highest-EV structural move and the clean fix for the long-only-beta trap (F2).

4. **Pause live PEAD while the de-risk tests (§1.1–§1.3) run, or keep it trading?** PEAD is **already live and trading in paper** (activated 2026-06-02, `swing_selector=pead`).
   - ✅ **DECIDED: keep it running.** Paper capital, risk filters gate entries, and the live fills feed the §1.1 audit. **Pre-committed pause trigger:** revert `swing_selector` only if §1.2 (leave-one-crisis-out) FAILS — i.e. if the edge proves to hinge on a single episode.

5. **(NEW — RESOLVED, no longer open)** *Does the VIX>30 block actually fire in the +0.546 config?* **Yes — confirmed by code, not an open question.** `pead_scorer.py:134` (`if vix is not None and vix > self.vix_block_all: return []`) is a HARD block that fires whenever VIX>30, active in the committed long-only config (`vix_block_all=30`). It is **independent** of `vix_conf_ref=100` (a separate confidence-*damping* knob, effectively off) and `vix_block_short=100` (short-squeeze guard, irrelevant for long-only — those were misread as making the block inert). VIX>30 occurs on real days inside the 2020-05→2026-06 window (2022, Aug-2024, Apr-2025, COVID tail), and `PROJECT_STATE.md` documents the block lifting the crisis-fold tail P5 −0.288→+0.009 / %pos 80%→95%. **§1.2 has no Q5 dependency; it is valid and correctly the top-priority test.**

6. **(NEW) Fork a new dollar-neutral simulator vs. extend the existing one?** Correction #4 shows `AgentSimulator` already supports dollar-neutral L/S via `_process_rebalance`. But that engine is large and carries swing/intraday/PEAD baggage.
   - *Options:* (a) extend in place (thin driver, reuse the rebalance engine); (b) fork a clean `NeutralBookSimulator`.
   - **Recommendation:** **(a) extend in place** for the §3.1 spike (fastest path to the decisive answer, zero new sim-correctness risk). Only consider a fork if the composite north-star book (§4) outgrows the rebalance engine's assumptions. **Confirm with owner** — this determines §3.1 scope.

7. **(NEW) Does shortening swing feature lookbacks (§1.5) materially change the ranker, given it may be dead anyway?** If §3.1 proves the ranker dead even dollar-neutral, the swing purge fix (§1.5) and validation-power upgrade for swing (§3.5) are wasted effort.
   - **Recommendation:** **sequence §3.1's book-only spike BEFORE §1.5** so we don't invest in swing-harness fixes for a corpse. Confirm sequencing.

---

## 8. Highest-conviction first move

**Run §1.1 (cost-sensitivity sweep) immediately.** Rationale: it is **S effort** (cost is a single per-run constant, correction #7), unblocked, and it answers the single most decision-relevant question about the *only* live edge — does +0.546 survive realistic earnings-window costs? If PEAD breaks at ~10 bps it cannot go live regardless of everything else, which immediately reprioritizes the whole plan toward §3.1 (the neutral ranker) as the sole structural hope. It also forces us to lift `ENTRY_SLIPPAGE_PCT` to a kwarg, a prerequisite for every later honest cost test. Run §1.2 (crisis-block robustness — highest *priority*) right behind it; with §7-Q5 resolved (the VIX>30 block is confirmed active), it is unblocked.

---

*Discipline note: every workstream goes through design→implement→Opus-review→test→merge; CPCV/gate integrity and the no-look-ahead invariant are sacred; never adopt a change (e.g. AR(1) N_eff) that re-inflates significance. Prefer the structural arguments over the exciting ones. This plan is a DRAFT pending the §7 decisions.*
