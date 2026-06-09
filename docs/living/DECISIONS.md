# Decision Log
Append-only record of significant architectural and strategic decisions.
Format: `## YYYY-MM-DD — Title` then context, decision, rationale, consequences.

---

## 2026-06-09 — Alpha-v5 OPT-1b: options data layer — survivorship from the OPRA day files, PIT via holiday-aware knowable_date

**Context**: The pricing engine (OPT-1a) needs historical OHLCV + a contract universe to price against. Polygon Developer serves NO historical IV/greeks/OI (only the current snapshot) and NO historical NBBO — so this layer carries only PIT OHLCV + the universe; IV/greeks are computed by the engine. The two ways an options backtest silently lies are **survivorship bias** (building the universe from today's active chain drops every contract that expired worthless — the modal outcome for short premium) and **look-ahead** (using an EOD bar before it printed).

**Decision**:
1. **Survivorship by construction.** The universe is built FROM the OPRA daily flat files (`us_options_opra/day_aggs_v1`, every contract that *actually traded* that day, expired included) — not from the REST active chain. A contract enters the store the first day it prints a bar and is never removed. `fetch_contracts`/`get_current_snapshot` (REST) exist only for live/validation and are never on any historical path. (`app/data/options_provider.py`, `scripts/backfill_options.py`.)
2. **PIT via holiday-aware knowable_date.** An EOD bar trade-dated D is knowable the next *trading* day; `knowable_date = D + 1 NYSE business day` using a proper NYSE holiday calendar (observes Good Friday; not Columbus/Veterans) so it never lands on a closed session. Every historical accessor filters `knowable_date <= as_of`. Contract metadata (strike/expiry/type) is decoded from the OCC ticker; `knowable_date` for a contract is the MIN over its bars (first knowable). (`docs/reference/OPTIONS_DATA.md`.)
3. **Storage**: `data/options_bars.parquet` (long OHLCV + knowable_date) + `data/options_contracts.parquet` (derived from the bars, always consistent). Provider `PolygonOptionsProvider` implements the OPT-0 `OptionsDataProvider` contract; `polygon_s3.py` extended with `get_options_day_file` + a `dataset` param.
4. **Opus 4.8 adversarial review (look-ahead / survivorship focus) confirmed the architecture sound and drove fixes**: holiday-aware knowable_date (was weekday-only `BDay`, could stamp a holiday); datetime-resolution + dtype coercion on load (parquet round-trips ms, fresh bars are ns → concat crash; now forced to ns); coverage-start guard (an empty universe before our data window logs a DATA-GAP warning instead of masquerading as "no contracts existed"); per-day logging of dropped adjusted/non-standard roots (split-driven gaps become visible). 22 tests (OCC parse, prefix-root disambiguation, PIT no-look-ahead, survivorship incl./excl. expired, holiday knowable_date, dtype coercion, merge revision-keep, multi-underlying alignment). S3 path smoke-tested: 3 days × SPY = 19,392 bars / 8,939 contracts, 791 expired retained, PIT confirmed.

**Rationale**: Deriving the universe from traded bars is *more* survivorship-safe than the REST reference endpoint (the files are ground truth of what traded) and avoids REST pagination for history. Computing knowable_date holiday-aware (vs the SI provider's padded weekday lag) keeps the options lag exact (+1 trading day) without staleness. We backfill a focused liquid universe (index ETFs + large caps), not all of OPRA — the IV-crush/VRP strategies only need the names we trade.

**Consequences**: A multi-year, PIT, survivorship-safe options OHLCV + universe store is available behind the frozen contract. Unblocks OPT-2 (contract-level simulator marks against this data + the OPT-1a engine). Not a WF/CPCV pipeline change yet → `PIPELINE_ARCHITECTURE.md` untouched until the options simulator lands (OPT-2).

---

## 2026-06-09 — Alpha-v5 OPT-1a: options pricing/greeks engine (BS + Bjerksund-Stensland + CRR) validated vs live snapshot

**Context**: Polygon Developer serves IV/greeks only in the *current* snapshot, so all historical IV/greeks must be **computed** — the program's confidence keystone (a wrong pricer silently corrupts every options backtest). OPT-1a builds that engine and proves it against the one window with ground truth (the served-IV snapshot).

**Decision**:
1. **Engine** (`app/options/pricing_engine.py`, pure/no-I/O, implements the OPT-0 `OptionsPricingEngine` Protocol): Black-Scholes-European closed-form price + greeks; **Bjerksund-Stensland 1993** American approximation (calls direct; puts via the put-call transform P(S,K,r,b)=C(K,S,r−b,−b)); **CRR binomial** as an independent reference; bisection IV solver; American greeks via kink-aware central finite differences.
2. **Validation** (`scripts/validate_options_engine.py`, the keystone): recompute American IV from EOD close + spot + real per-underlying dividend yield + rate, compare to Polygon's served IV/delta over 15 underlyings, PASS/FAIL on the OPT-1 tolerance. **Result: PASS** — near-ATM median |IV err| 0.0072, all-contract bias +0.0068 (both < 0.010), engine-delta vs served-delta median |err| 0.0011 (greeks essentially exact). A **day-vol ≥ 10 liquidity filter** removes a +0.022 tail bias shown to be a *data-timing artifact* (snapshot pairs an option's stale last trade with the live spot) — absent in EOD-bar backtests.
3. **Adversarial review (Opus 4.8) found + fixed 3 bugs before merge**: **(CRITICAL)** for dividend-yield > rate calls the BjS h(T) term flips positive and the trigger boundary degenerates, underpricing ~95% (0.0017 vs 0.30 true) → route the strongly-negative-carry regime to the exact CRR binomial. **(HIGH)** the IV solver marched to the bracket top (3.0) for deep-ITM American prices pinned to the intrinsic floor → return `None` (vol is unrecoverable from a price = intrinsic). **(MEDIUM)** central-difference gamma spiked ~10× at the early-exercise boundary strike → one-sided 2nd difference on the smooth side. All three have regression tests (18 unit tests total: textbook BS, put-call parity, American≥European, BjS↔CRR cross-check, IV round-trip, greeks signs, contract conformance).

**Rationale**: BjS-1993 is fast and accurate in its valid regime but has a known degenerate carry regime; rather than ship a higher-order approximation, we fall back to CRR (exact in the limit) only for the rare contracts that need it — fast path stays fast, edge cases stay correct. Computing IV per-contract from its own price means we match each strike's own smile point, so the validation tests the *engine*, not a smile model.

**Consequences**: The historical IV/greeks engine is trustworthy (validated to <1 vol-pt near-ATM, exact greeks). `validate_options_engine.py` is now a repeatable nightly health gate (PASS/FAIL exit). Residual +0.0068 bias is flat-rate + crude-dividend; OPT-2 wires a real rate series. Unblocks OPT-1b (data layer) → OPT-2 (simulator). Not a WF/CPCV pipeline change yet, so `PIPELINE_ARCHITECTURE.md` is untouched until the options simulator lands (OPT-2).

---

## 2026-06-09 — Alpha-v5: Options Strategy Program launched (Polygon Developer); OPT-0 charter + spike PASS

**Context**: Free-data 3rd-sleeve candidates are exhausted (reversal/carry/estimate-revision all
eliminated — the opportunity set is fished out on free data). Owner subscribed **Polygon Options
Developer ($79/mo)** to pursue the highest-ceiling edge: the options variance risk premium.

**Decision**: Launch a phased **Options Program** (Alpha-v5) — SSOT `docs/living/OPTIONS_PROGRAM.md`.
Build a **resilient five-layer base** (data ⟂ pricing engine ⟂ simulator ⟂ pluggable strategy ⟂
reused gates/allocator/live; four frozen contracts in `app/options/contracts.py`) and explore MANY
options strategies, each validated KEEP/KILL on the SAME `run_cpcv` + significance gate + CAPM
residual-α we trust for equities, plus an options-specific **spread-stress sweep (KEEP must survive
2×)** and capacity check. Foundation first, but prove the whole pipeline end-to-end with ONE
strategy (earnings IV-crush) before building the catalog. Phases OPT-0…OPT-8; owner checkpoints at
OPT-0, OPT-3 (first verdict), OPT-8 (arm live).

**Forced data-architecture facts (Developer tier)**: Polygon serves IV/greeks/OI only in the CURRENT
snapshot → **we compute historical IV/greeks ourselves** (BS-European + Bjerksund-Stensland); **no
historical NBBO** → mark off EOD close + model/stress the spread; **no historical OI** → liquidity via
volume/notional; survivorship cured via the expired-contract universe.

**Rationale / de-risking**: the OPT-0 feasibility spike (`scripts/spike_options_iv_check.py`) proved
the confidence keystone — computed BS-European IV matches Polygon's served IV to **0.86 vol-points
median, unbiased, near-ATM** (the contracts VRP trades); all-contract bias (+0.035) is the expected
ITM/OTM + dividend gap that BjS + real dividends close in OPT-1. So computing historical IV from EOD
close is accurate enough to backtest with confidence. **Consequences**: the program's foundational
risk (computed-IV accuracy) is retired up front; OPT-1 (data + engine) green-lit. Live execution
(Alpaca options `OptionLegRequest`/mleg) is supported by the SDK but not yet wired — a later build.

---

## 2026-06-08 — Alpha-v4 P4: short-term reversal sleeve KILLED (cost-dead); 3rd-sleeve slot stays open

**Context**: Sought a 3rd uncorrelated premium after PEAD + TSMOM trend. Owner chose
short-term cross-sectional reversal (the momentum complement). Built + validated a
dollar-neutral, PIT, survivorship-safe sleeve (`app/strategy/reversal.py` + `run_reversal.py`).

**Decision**: **KILL / benchmark-only — do not live-wire.** The reversal signal is real but
weak (gross +0.40, t=1.28 @2bps) and **cost-dead**: ~159x/yr turnover → ~16%/yr cost drag →
**-0.90 Sharpe at a realistic 10bps**; adding it *drags* the book (equal-capital +1.145 →
+0.138). It IS genuinely uncorrelated (β~0.10, corr +0.13/+0.03 to PEAD/trend) — the concept
is right, the tradeable edge isn't.

**Rationale**: Opus 4.8 adversarial review verified the KILL is real, not a bug (sign / cost
single-charge / look-ahead / dollar-neutrality / liquidity-masking all correct). Short-term
reversal is the most-arbitraged anomaly; this is the expected null. **Explicitly NOT
filter-hunted** to rescue it (only-trade-in-high-VIX etc.) — that's the B5 trap on the STOP
list. **Consequences**: the harness is retained as a reusable validated null (7 tests); the
3rd-sleeve slot remains OPEN. Next candidates: options-VRP feasibility spike (needs paid IV
data — a spend decision), cross-asset carry (free data), or squeeze-conditioning (existing
SI data, but a PEAD conditioner rather than a standalone premium).

---

## 2026-06-08 — Alpha-v4 P3: live regime-aware sleeve allocator (gate-controlled, default-equal, kill-switchable)

**Context**: The live book ran two independent sleeves at static budgets (trend
`pm.trend_allocation_pct=0.40`, PEAD telemetry `pm.pead_size_mult=1.0`). The allocator
(`app/strategy/sleeve_allocator.py`) existed only as a backtest library. The Phase-3
book-level gate (`scripts/run_book_allocator.py`, margin >0.10 Sharpe AND no-worse-DD)
found on the 2-sleeve overlap: **equal +1.082 > vol +0.715 > regime +0.593** — regime is
worst. So vol/regime are not justified on the current book.

**Decision**: Wire the allocator into the LIVE book as a kill-switchable layer
(`app/live_trading/sleeve_allocator_live.py`) that **ships DISABLED** (`pm.allocator_enabled=false`)
→ byte-identical to today. When enabled it recomputes weekly (before the trend rebalance),
persists effective weights to `agent_config`, and both sleeve readers
(`effective_trend_allocation` / `effective_pead_size_mult`) consult them with a
**fixed-weight fallback** on disabled / stale / warmup / any error. Default scheme `equal`;
`vol`/`regime` are live-capable but stay OFF until `run_book_allocator.py --emit-config`
selects them (expected after a 3rd sleeve).

- **Live regime label** = the persisted, staleness-aware live score
  (`get_regime_context`) mapped **RISK_ON→BULL, RISK_CAUTION→NEUTRAL, RISK_OFF→BEAR,
  unknown→NEUTRAL** (the safe no-tilt key).
- **PEAD regime double-tilt avoided (from the Opus pre-merge review):** PM already applies
  a per-name `_regime_sizing_multiplier` to PEAD, so under `scheme=regime` the allocator
  does **not** also scale PEAD's `size_mult` (would compound the same regime bet) —
  PEAD's own per-name mult is its sole regime tilt; the allocator's regime tilt flows only
  to the TREND budget. `equal`/`vol` map PEAD normally (no regime component).
- **Gross-cap safety:** effective trend allocation is clamped to ≤0.80, and the trend
  sleeve's existing `apply_risk_gate` independently caps total (trend+PEAD) gross ≤80% on
  actual positions regardless of allocator output.

**Quality loop**: adversarial Opus 4.8 review found 1 Critical (the double-tilt, now
guarded) + doc/robustness items (now fixed); a second Opus pass verified all resolved with
no regression (SHIP).

**Known limitations (must clear before enabling `vol`/`regime`):** (1) the LIVE regime
path uses a one-shot tilt (no hysteresis/EWMA blend — those need a label series), so it
differs from the validated backtest `apply_regime_tilt` and must be re-validated before
activation; (2) the trend sleeve writes its tracker weekly, so the live per-sleeve vol
estimate is coarse until ~`pm.allocator_min_deployed_days` of history accrue (the warmup
guard keeps it in static fallback until then). **Consequences**: zero behavior change today;
infra + re-runnable gate ready to activate when the 3rd sleeve makes vol/regime earn it.

---

## 2026-06-07 — Alpha-v4 P0 gate recalibration: robustness over Sharpe level; residual-alpha-t diagnostic-first

**Context**: On N_eff≈8 a high Sharpe *level* selects for overfitting (5-LLM review:
"lower the SR≥0.80 gate — real edges live at 0.4–0.7; weight fold-consistency +
neutralized-t over level"). The significance gate already made t-stat / %pos / P5 /
worst-regime the primary criteria, but the headline bar and the missing
market-residualized check needed to land.

**Decision**:
- **Retire the legacy SR≥0.80 promotion bar.** `GATE_MODE='significance'` stays the
  default (the legacy `mean_sharpe`/0.80 path is kept only for reproducibility).
  Lower `CAPITAL_GATE_MIN_MEAN_SHARPE` 0.50→**0.45**; keep PAPER 0.35, min-fold/P5
  floors, DSR/PF/Calmar, and the worst-regime survivability floor. The Sharpe floor
  is now a materiality *backstop*, not the discriminator.
- **Add residual-alpha-t (CAPM/HAC) as a DIAGNOSTIC, not a gate — yet.** Per owner
  decision, it enters *diagnostic-first*: computed on the concatenated OOS book
  returns vs SPY, reported in `print()`/JSON + a `t<1` WARN log, and **explicitly
  excluded from gate pass/fail** until validated (it reproduces the known PEAD
  beta-driven verdict and a genuine-alpha case in tests). It graduates to a primary
  blocking criterion in a later PR — mirroring how the significance gate itself was
  rolled out behind a faithful-reproduction proof.

**Rationale**: a blocking gate on a brand-new metric over ~8 effective folds could
mis-promote/mis-retire before it's trusted; diagnostic-first de-risks that while
still surfacing the single most important robustness signal (does the edge survive
hedging out the market?). **Consequences**: every CPCV run now prints residual-α-t +
β + hedged-Sharpe; gate verdicts are unchanged this PR (proven by test). Canonical
estimator is `scripts/walkforward/attribution.capm_alpha` (shared with the PEAD
attribution script). See PIPELINE_ARCHITECTURE Gate Inventory + changelog.

---

## 2026-06-06 — Live TSMOM trend sleeve: standalone weekly rebalancer (Alpha-v4 live wiring)

**Context**: Alpha-v4 Phases 0–3 are complete. The TSMOM trend sleeve
(`app/strategy/tsmom.py`, validated standalone Sharpe +0.71, the book's crisis
diversifier) was the strongest sleeve but not live. The task: trade it live in the
paper account alongside PEAD at a simple fixed weight.

**Decisions**:
- **Standalone weekly executor, NOT a `pm.swing_selector` value.** The selector is
  mutual-exclusion (one daily stock scan producing entry signals); the trend sleeve
  is a weekly rebalance-to-target on a fixed 10-ETF basket that must run *alongside*
  PEAD. It lives in `app/live_trading/trend_sleeve.py`, fired by a daily orchestrator
  job (09:45 ET) with an in-function weekday guard (`pm.trend_rebalance_weekday`,
  live-tunable) + a fail-closed market-open check (`AlpacaClient.get_clock` — the
  weekday cron has no holiday calendar).
- **Direct Alpaca placement with a lightweight risk gate** (kill-switch, gross cap
  `trend+PEAD ≤ 80%`, fat-finger, per-name cap), NOT the PM→RM→Trader proposal queue
  (those rules are entry-signal-shaped and map poorly onto rebalance trims/sells).
- **Equal-capital 50/50**: `pm.trend_allocation_pct` default 0.70→**0.40** (trend 40%
  / PEAD 40% under the 80% gross cap — matches the Phase-3-validated equal-capital
  book, which beat vol-weight and regime-tilt).
- **PEAD dialed to telemetry** in the schema defaults too: `pm.pead_size_mult` 3.0→1.0,
  `pm.pead_max_position_pct` 0.10→0.05 (the DB values were already dialed; rebaselining
  the defaults prevents a DB reset from silently re-ramping PEAD). `test_pead_ramp_b4`
  expectations updated to match.
- **Shadow-first, dormant-by-default**: `pm.trend_enabled` default `false`,
  `pm.trend_shadow` default `true` (logs would-be orders to `decision_audit` with
  `block_reason="shadow"`, sends nothing). Owner arms via `scripts/set_trend_config.py`.
- **Trend positions tagged `selector="trend"`/`trade_type="trend"` and excluded from the
  Trader's per-tick stop/target exit loop** (`_check_exit` guard) — the weekly
  rebalancer is their sole manager; otherwise the synthetic stops the reconciler
  attaches would liquidate the sleeve mid-week.
- **Fail-closed everywhere**: kill-switch, data-fetch failure, missing core symbol
  (SPY), or NAV-fetch failure → no orders. Whole shares only (Alpaca wrapper is int-only).

**Consequences**: trend coexists with PEAD as a peer sleeve; live-vs-backtest
divergence (Alpaca vs yfinance adjustment, wall-clock vs modular rebalance) is tracked
by `app/live_trading/trend_tracker.py` (+0.71 reference, weekly rollup email). Known
limits / backlog: the gross-cap formula now lives in 3 places (trend imports the
canonical `risk_manager.GROSS_EXPOSURE_CAP`); fixed 40/40 forgoes the validated
vol-weighting + BEAR regime tilt in `sleeve_allocator.py` (deliberate "ship simple
first" — revisit when more sleeves / longer overlap earn it).

---

## 2026-06-02 — Significance-first two-tier promotion gate (replaces mean-Sharpe≥0.80)

**Context**: The promotion gate's primary discriminator was `mean_sharpe ≥ 0.80`
(swing) / `≥ 1.00` (intraday). Those thresholds were calibrated against numbers
that have since been struck as in-sample artifacts (intraday +5.14, QualityShort
+3.25). A bare mean-Sharpe threshold cannot distinguish a `+0.22 / t=0.17` noise
result from a `+0.546 / t=2.26` genuine-signal result — both are below 0.80, yet
one is statistically significant and one is pure noise. The 0.80 bar was a
frozen-WF relic: it rejected the real signal (PEAD) for the same reason it rejected
the noise, providing no actual discrimination.

**Decision**: Adopt a **significance-first two-tier** gate behind a `GATE_MODE`
flag (default `"significance"`; `"mean_sharpe"` reproduces the legacy gate exactly
for reversibility + historical re-scoring).
- Primary discriminators become statistical: path-Sharpe **t-stat** (N_eff=n_folds,
  flipped from WARN to BLOCK), sign-consistency (`pct_positive`), and the tail
  (`p5_sharpe`). Mean Sharpe is demoted to an economic-materiality FLOOR.
- **PAPER** tier (forward-validate, no capital): t≥2.0, %pos≥0.75, P5≥0.0,
  mean≥0.35, plus PF/Calmar/regime backstops.
- **CAPITAL** tier (real money): PAPER + mean≥0.50 + n_folds≥10 + (t≥2.5 OR a
  documented live-paper confirmation). The higher t-stat is a multiple-testing
  haircut (~10–15 strategy shots); n_folds≥10 is a statistical-power floor.
- A standard WF report (single point estimate, no path distribution) HARD-FAILS
  under significance — it cannot fabricate a t-stat; CPCV is required.

**Rationale**: Promotion should be gated on whether an edge is statistically real
and economically material, not on clearing an absolute Sharpe number that was set
against contaminated baselines. The two-tier split lets a genuinely-significant-
but-still-developing edge go to PAPER (forward-validate with no money at risk)
while reserving CAPITAL for results that also clear the multiple-testing haircut.

**Consequences**:
- Re-scoring every CPCV result on record (`scripts/rescore_gates.py`) promotes
  **only PEAD R1K → PAPER PASS / CAPITAL HOLD**. Every other strategy (Swing
  +0.22/t0.17, Intraday −2.80, Small/mid PEAD +0.361/t0.95/P5−1.368, QualityShort
  −0.903, Insider +0.228/t0.88) FAILs all tiers. The LEGACY(0.80) column is
  all-FAIL — confirming 0.80 never promoted any of these anyway; it just failed
  to separate the one real signal from the noise.
- PEAD is cleared to PAPER (forward validation), NOT capital — it lacks both the
  t≥2.5 haircut margin (2.26) and the n_folds≥10 power floor (8).
- `mean_sharpe` mode is a verified no-op vs pre-Phase-4 main (full legacy gate
  test corpus passes unchanged).
- No change to DSR math, N_eff=n_folds, OOS/sacred-holdout machinery, the
  simulators, or the PEAD scorer.

---

## 2026-06-02 — Significance-gate review fixes: PEAD paper PASS is a FLAGGED event-sparsity waiver, not unconditional

**Context**: An independent review of the significance-gate branch found three
blocking defects. (1) Under `GATE_MODE="significance"` a WF-only retrain hard-failed
`WalkForwardReport.gate_passed()`, and `retrain_cron.py` fed that boolean into
`record_tier3_result(gate_passed=False)`, which sets `status="RETIRED"` and rolls
back — so every scheduled WF retrain auto-retired the fresh model. The capital tier
was also unreachable (no caller ever requested `tier="capital"`). (2) The real PEAD
CPCVResult has `worst_regime_sharpe=None` due to event-sparsity (`<REGIME_MIN_OBS`
same-regime trading days — documented "not a bug"), and the backstop failed-closed
on None, so the REAL PEAD FAILED the paper gate the whole exercise was meant to pass.
(3) `rescore_gates.py` reimplemented the threshold math and hardcoded
`backstops_ok=True`, so its "PEAD PASS" was fiction, not the real gate.

**Decision**:
- **Tri-state outcome (FIX-1)**: distinguish "gate failed → retire" from "cannot
  evaluate for promotion → keep status." `GateOutcome{PROMOTE,RETIRE,INCONCLUSIVE}`;
  significance+WF → `INCONCLUSIVE` (report-only). The cron keeps the current model
  status on `INCONCLUSIVE` (no retire/rollback). Capital is reached only by an
  explicit promotion run (`--gate-tier capital`), never by the cron retrain.
- **Event-sparsity regime waiver (FIX-2)**: `worst_regime_sharpe=None` has two
  causes, now disambiguated by `CPCVResult.regime_insufficient_obs` (set from raw
  per-regime obs counts captured before the REGIME_MIN_OBS filter). For
  EVENT-SPARSITY only, the **PAPER** (zero-capital) tier waives the regime backstop
  AND flags `requires_human_review`. The **CAPITAL** tier never auto-waives (requires
  explicit `regime_waiver_approved`). A DATA-BUG None still fails closed on both.
- **Real-gate rescore (FIX-3)**: the artifact now runs the production gate.

**Rationale**: The waiver is the minimum needed to let an event-sparse strategy
reach forward-validation without opening a global fail-open. Scoping it to (a)
paper only, (b) event-sparsity only, (c) with a mandatory human-review flag keeps
the regime backstop fully enforced everywhere real capital or real regime data is at
stake. The corrected statement of the result: **PEAD R1K → PAPER PASS *with a
mandatory `requires_human_review` flag* (via the event-sparsity waiver) / CAPITAL
HOLD** — the prior "unconditional PASS" framing overstated it.

**Consequences**:
- PEAD reaches paper for forward validation but is explicitly tagged for human
  review because it was promoted without real regime data.
- A scheduled WF retrain under significance no longer auto-retires the fresh model;
  it logs INCONCLUSIVE and waits for an explicit CPCV promotion decision.
- Capital promotion of an event-sparse strategy is impossible without a documented
  `--regime-waiver-approved` human sign-off.

---

## 2026-05-23 — Adopt Opus 4.7 Four-Phase Plan

**Context**: v216 Walk-Forward gate failed (avg Sharpe -0.91, PF=0.00 every fold). Five independent LLM reviews (Claude, ChatGPT, Gemini, Grok, Deepseek) all flagged the same core issue: jumped straight to L4 (full agent stack) without validating at L1 (rank-IC) or L2 (decile spread).

**Decision**: Adopt Opus 4.7's four-phase plan:
1. WF Trustworthiness → 2. Signal Measurement → 3. Modelling → 4. Portfolio/Execution

**Rationale**: Each layer must pass independently before proceeding. Without isolating signal from execution, it's impossible to know whether PF=0.00 comes from bad features, bad labels, bad sizing, or bad simulation.

**Consequences**:
- NO retraining until Phase 2 (L2 decile spread) gate passes
- NO regime-conditional models until factor attribution confirms residual alpha
- PIT audit is the highest-risk gate: if fundamentals have look-ahead, all prior results are invalid

---

## 2026-05-23 — Fix 10 WF Simulation Bugs (PR #256)

**Context**: Opus 4.7 deep code review found 10 simulation bugs in walkforward_tier3.py and agent_simulator.py.

**Decision**: Fixed all 10 bugs:
1. MTM pricing used stale prices (off-by-one)
2. Sharpe annualization used calendar days not trading days
3. DSR formula missing sqrt(V[SR]) scaling
4. DSR N_obs used fold count not observation count
5. CPCV look-ahead: used future fold's training data for embedding
6. Force-close fired after MTM, double-counted last day P&L
7. Halt-day MTM used next day's open (look-ahead)
8. Sector ETF signal loaded same-day (look-ahead on rebalance date)
9. Short series annualization used wrong N in sqrt formula
10. profit_factor sentinel: returned 999 instead of 0 when no losses

**Consequences**: WF results are now trustworthy at the simulation level. v216 rerun gave Sharpe -0.91 (improved from -1.8+ but still gate failed).

---

## 2026-05-22 — Restore swing_v215 as Active Model

**Context**: v216 LambdaRank model trained with 18 features, 20d horizon. Walk-forward gate failed.

**Decision**: Restore v215 as the active paper-trading model while diagnostics run.

**Rationale**: v215 had better WF results than v216 post-bug-fixes. Running on broken simulation results (pre-fix) was producing misleading metrics. Running paper trading on v215 while investigating is safer than using a gate-failed model.

---

## 2026-05-20 — Adopt L/S Equity as Primary Strategy Direction

**Context**: Long-only swing strategy with ATR stops consistently fails WF gate. Opus analysis suggests the stop-loss asymmetry requires hit-rate ≥ 33% with 2:1 R:R — not achievable with IC ≈ 0.

**Decision**: Target Long/Short equity for production. Top-N long + bottom-N short, dollar-neutral.

**Rationale**: Removes the dependency on absolute return prediction (hard). L/S only requires relative ranking (easier). Eliminates directional beta. Enables full capital utilization in both bull and bear markets.

**Consequences**: Phase 4 must implement dollar-neutral construction with borrow filter.

---

## 2026-05-23 — Execute Phase 4 First If L2 Decile Sharpe >= 0.60

**Context**: Null benchmark showed random portfolio Sharpe = +0.669 vs v216 WF = -0.91 (z=-9.87). The execution layer is 9.87 sigma worse than random chance. L2 decile spread is running to determine if underlying signal exists.

**Decision (pending L2 result)**: If L2 Sharpe >= 0.60, skip Phase 3 (label redesign) and go directly to Phase 4 (execution fix: remove ATR stops, increase position count, L/S conversion).

**Rationale**: With execution destroying 1.5+ Sharpe units vs random, fixing execution is higher ROI than fixing labels. The 2021 IC = +0.023 suggests signal exists in bull regimes. The execution pathology (ATR stops + low position count) is the dominant failure mode.

**If L2 < 0.20**: No signal exists. Must rebuild features. Phase 3 before Phase 4.

---

## 2026-05-23 — Remove ATR Stops From Swing Strategy

**Context**: Null benchmark (no stops) achieves Sharpe +0.669. WF (with ATR stops) achieves -0.91.

**Decision**: The ATR stop mechanism should be disabled for initial Phase 4 testing. The stops are creating a negative feedback loop:
1. Low IC → random win rate ~50%
2. ATR stop triggers on small adverse moves, cutting many positions early  
3. Remaining positions run longer but the overall win rate < breakeven for 2:1 R:R
4. Net effect: stops increase transaction costs while not improving win rate

**Do NOT**: Add wider stops or tighter stops as a fix. The stop mechanism itself needs testing without stops first. If L2 without stops shows Sharpe > 0.60, that is the baseline.

---

## 2026-05-23 — Fold 2 Diagnosis: Opportunity Score Gate + ATR Stops (Phase 1.6)

**Context**: v216 WF Fold 2 (test: 2022-06-04..2023-05-24) had 95 trades vs 300+ in all other folds. Fold 2 covers the post-peak-inflation, aggressive-Fed-hiking period.

**Findings**:
1. Cross-sectional vol in Fold 2 = 1.04x other folds — NOT dramatically higher (test starts after the worst of the 2022 crash)
2. Symbol coverage: 769 vs 750 avg — similar, NOT a data sparsity issue
3. Primary suppressor: **opportunity score gate** (`score < 0.35 = skip`, `0.35-0.65 = cap at 2 candidates`). Model trained on 2020-2022 bull data assigns low scores to 2022 bear-market patterns → gate skips most entries
4. Secondary suppressor: ATR stops cut the few entries that pass the gate before HOLD_DAYS

**Decision**: Phase 4 isolation test must disable BOTH mechanisms:
- `--no-pm-opportunity-score` (disable opportunity score gate)
- Remove ATR stops (already decided)

**Note**: v216 WF used purge=10d not 85d. All v216 results have potential leakage and must be re-run with purge=85d post-Phase 4.

---

## 2026-05-23 — Phase 4 Before Phase 3 (Opus 4.7 Override)

**Context**: L2 decile spread returned Sharpe=0.397 (marginal, 0.20-0.60 range). Original decision tree said "Phase 3 first." Opus 4.7 reviewed all findings.

**Decision**: Run Phase 4 (execution fix) BEFORE Phase 3 (label redesign).

**Rationale**:
1. Null benchmark shows execution destroys ~1.6 Sharpe vs random. Phase 4 is a config change (1-2 days), Phase 3 is weeks.
2. Cannot measure label improvements through WF when execution layer masks signal. Phase 4 first establishes honest baseline.
3. Signal clearly exists in right regime (2021/2025 L/S Sharpe = +1.1). Short side is the structural problem, not features.
4. 2023 inversion (-1.29) is a crowded-short squeeze in narrow Mag7 rally — short-side failure, not long-side.

**Phase 4 Spec**:
- Disable opportunity score gate (`--no-pm-opportunity-score`)
- Remove ATR stops
- Position count: n=40 long, n=40 short
- Re-run v216 WF with 85d purge

**Phase 3 Spec (after Phase 4 baseline)**:
- Long-only labels: top-quintile binary (drop full cross-sectional rank)
- 10d horizon (not 20d) — doubles training samples
- Rolling 3-year window (not expanding)
- Add regime features as inputs (breadth, dispersion, VIX term structure)
- Kill sign-flipping features (per-year IC audit)
- Short side: separate model with quality overlay, NOT symmetric decile rank

**If Phase 4 WF Sharpe > +0.3**: proceed to Phase 3 with confidence.
**If Phase 4 WF Sharpe < 0**: investigate execution bug before any label work.

---

## 2026-05-24 — Opus 4.7 WF Code Audit: 10 Critical/Major Bugs Found

**Context**: After Phase 4 v2 WF (avg Sharpe +0.046, 78 trades) and L2 Sharpe=0.397, commissioned a thorough Opus 4.7 audit of walkforward_tier3.py and agent_simulator.py looking for bugs, look-ahead, and realism issues.

**Findings (prioritized)**:

1. **CRITICAL — Embargo never enforced in fold boundaries** (walkforward_tier3.py L689)
   - `raw_test_end_dt = train_end_dt + segment_days` → fold N test ends exactly where fold N+1 trains. Embargo_days was logged but had zero effect on boundary math.
   - **Fix**: `raw_test_end_dt = train_end_dt + segment_days - embargo_days`

2. **MAJOR — no_atr_stops defeated by check_exit trailing ratchet** (agent_simulator.py L1250)
   - When `no_atr_stops=True`, sentinel stop prices replaced with real trailing stops on first profitable bar, defeating the phase 4 isolation.
   - **Fix**: Only persist `new_stop` from check_exit when `not self.no_atr_stops`

3. **MAJOR — PF=999 sentinel inflates avg_profit_factor gate** (walkforward_tier3.py L269-271)
   - `avg_profit_factor` averaged PF=999 (all-wins fold) with real PFs, yanking mean far above gate threshold.
   - **Fix**: Cap individual PFs at 5.0 before averaging

4. **MAJOR — Silent trade loss when end-date data missing** (agent_simulator.py L514)
   - FORCE_CLOSE silently skipped positions with no bar data — trade never recorded, affecting trade count and equity.
   - **Fix**: Exit at entry_price with warning log when no bar data available

5. **MAJOR (deferred) — Calmar=0 "not computed" free-passes gate** (walkforward_tier3.py L292)
   - `avg_calmar == 0` was treated as "skip gate" rather than "gate fail". Ambiguous sentinel.
   - **Decision**: Document for future fix; change sentinel to NaN requires broader test updates.

6. **MAJOR (deferred) — Short buying power check uses full notional** (agent_simulator.py L889)
   - Short entries checked against cash balance using full notional (Reg-T 100%), over-rejecting shorts.
   - **Decision**: Defer; only affects short-side entries. Long-only Phase 3 is unaffected.

**Fixes implemented**: Items 1-4 committed in feat/wf-opus-audit branch.

**Consequences**: Previous WF results (all phases) used the defective embargo formula. Re-running Phase 4 v3 with corrected boundaries is required to get clean results. Embargo fix shrinks test windows by ~85 days each fold — with purge=85 and embargo=85, effective test window is 456-85=371 trading days per fold.

---

## 2026-06-03 — PEAD UI visibility: selector attribution + PEAD tracking panel

**Context**: The dashboard surfaced only "swing" and "intraday" proposals. PEAD — the sole live capital strategy — rode under the "Swing Proposals" tab, indistinguishable from swing-ranker proposals, and its rich daily scoreboard (`data/pead_tracking.db`: signals→entered→filled funnel, fill rate, gross deployed, daily/cum P&L, VIX blocks, per-overlay suppression counts) had **zero UI surface** (weekly email only). With PEAD live and currently 0-filling (price-ran / spread gates), there was no way to see *why* without querying SQLite by hand.

**Decision**:
1. **Data model** — added `selector` (VARCHAR(32), indexed) to `proposal_log`, mirroring `Trade.selector`. Chosen over deriving PEAD-ness by joining `proposal_uuid → trades.selector` so that **unfilled** PEAD proposals are attributable too (the join only covers proposals that became trades). Migration `scripts/migrations/2026_06_proposal_log_selector.py` is idempotent and backfills historical `dir_{selector}_*` batches (backfilled 271 rows: 150 quality_short, 121 pead).
2. **API** — `selector` threaded into all 3 PM `ProposalLog` persist sites; exposed on `/proposal-log` (response field + `selector` filter param) and on positions/trades responses. New `/api/dashboard/pead/tracking` wraps `pead_tracker.read_daily` with a window summary (funnel totals, fill rate, suppression counts, cumulative P&L).
3. **Frontend** — shared `SelectorBadge` across proposals/positions/trades; selector column + filter on the swing proposals table; new top-level **PEAD** tab (KPI row + signal→fill funnel + suppression breakdown + daily table) — the first UI view of the live PEAD book.

**Consequences**: PEAD is now first-class in the dashboard; the funnel/suppression view makes the live 0-fill situation diagnosable at a glance. The live PM/RM/Trader path is unchanged except the additive `selector` write (nullable, default `""`) — a server restart is needed to deploy the routes/UI but **not** for any behavior change. Built in an isolated git worktree to protect the in-flight ranker CPCV run. Not a WF/CPCV pipeline change, so `PIPELINE_ARCHITECTURE.md` is intentionally untouched.

---

## 2026-06-03 — Cross-sectional ML ranking is dead; close the ranker line, pivot to the event-driven edge family

**Context**: Alpha-v2 §3.1 hypothesized the "dead" swing ranker (+0.22, t=0.17) was merely *strangled* by a 5-position long-only book, and would show alpha if re-run **dollar-neutral, sector-neutral, high-breadth**. The first L/S run looked invalid-positive then invalid-negative; rigorous diagnosis found the book was never actually neutral (it ran ~35% net-long at 0.35 gross: the L/S rebalance was fed a one-sided **long proposal pool** of ~50 names — `_pm_score`'s `proposal_pool_size` cap + `min_confidence` floor — so the 60-long book absorbed the whole ranked set and the short leg starved; held positions were also never re-sized).

**Decision**: Fixed the validity end-to-end across 3 phases — **(1)** net-exposure observability (surface realized net beta/dollar/gross + result JSON), **(2)** dollar-neutral-at-target-gross (full-book resize each rebalance + breadth admission), **(3)** full cross-sectional scoring for the L/S arm + adequate power (k=8). On the **corrected, genuinely-neutral book** (realized net$ −0.01, gross 0.73, ~60 shorts), the decisive CPCV (N_eff=8) gave **mean Sharpe +0.14, path-t +0.18, %pos 67%, deployment-adj +0.12, DSR p 0.03** → **no cross-sectional alpha.** The long-only +0.22/+1.06 was **confirmed market beta** (neutralizing collapses it to noise).

**Rationale**: This is the *third* honest CPCV null from the cross-sectional-ML-ranking direction (swing long-only = noise; intraday v63 = cost-drag; dollar-neutral ranker = beta-only). t=0.18 is unambiguously null, not borderline — more CPCV power (purged-CV) cannot rescue a flat-zero signal, so we did not invest in it. The one validated edge (PEAD) is **event-driven, rules-based, economically grounded** — a different species from cross-sectional ranking. The data says alpha lives in the event-driven family, not in ML-ranking price/fundamental features.

**Consequences**: **The cross-sectional-ML-ranking line is closed.** The Alpha-v2 §3.3 (short-interest as a ranker feature) and Spike-B (residualized features) items are **shelved** — they were predicated on the ranker showing life. **PEAD is the sole validated edge** and now trades live (the entry-gate fix unblocked fills). Next direction (pending owner steer): pivot research to a **second event-driven edge** (analyst-revision drift / short-interest-squeeze-as-event / guidance) to diversify PEAD, and productionize PEAD (live track record + the §1.2b trend-filter). The validity-fix *infrastructure* (observability, neutral-at-gross L/S engine, full-ranking, net-exposure capture) is retained as reusable tooling even though the thesis died — its value is precisely that it prevented deploying a beta book as "alpha." See `ML_EXPERIMENT_LOG.md` (§3.1 Phase 1-3) for the run record.
