"""
Single source of truth for the weekly auto-retrain architecture and gate thresholds.

When you develop and validate a new model architecture (new model_type,
HPO settings, feature set, etc.), update SWING_RETRAIN or INTRADAY_RETRAIN
here and document the change in docs/ML_EXPERIMENT_LOG.md. The weekly
scheduled retrain will automatically use the updated config.

Rules:
  - Only change this after a new architecture passes the Tier 3 walk-forward gate
  - The config here should mirror the exact settings used for the current
    ACTIVE model version (so retrains produce the same architecture on fresh data)
  - hpo_trials controls Optuna hyperparameter search per weekly retrain;
    0 = use fixed params from model.py, >0 = search that many trials first
  - If the new model fails the walk-forward gate, the previous ACTIVE model
    is automatically restored — the new version is marked RETIRED
"""

import os as _os
import sys as _sys

# ── Per-fold retraining (true out-of-sample WF/CPCV) ──────────────────────────
PER_FOLD_RETRAIN: bool = False
"""Default per-fold-retrain mode when not overridden by --per-fold-retrain CLI.
When True, WF/CPCV retrain a fresh model inside each fold on only that fold's
training window (genuinely out-of-sample). When False (default), the legacy
frozen-model 'generalization test' runs (NOT out-of-sample, cannot promote).
See docs/living/PIPELINE_ARCHITECTURE.md KL-10."""

REQUIRE_TRUE_WF_FOR_PROMOTION: bool = False
"""When True, gate_passed() returns False for any run that is not a true
per-fold walk-forward (is_true_walkforward=False), even if all metric thresholds
pass. This makes frozen-mode runs report-only. Default False during Phase 1
rollout (swing per-fold proving); flip to True in Phase 3 once swing per-fold
is validated, so frozen runs can no longer promote project-wide."""

PER_FOLD_SWING_HPO_TRIALS: int = 0
"""HPO trials for per-fold swing retraining. 0 = use frozen production
hyperparameters (no per-fold HPO). Per-fold HPO is both prohibitively expensive
and methodologically wrong (in-fold selection). Keep 0."""

PER_FOLD_INTRADAY_HPO_TRIALS: int = 0
"""HPO trials for per-fold INTRADAY retraining. 0 = use the frozen production
hyperparameters (FROZEN_HPO_PARAMS in app/ml/intraday_training.py) — no per-fold
HPO. Same rationale as swing: per-fold HPO is prohibitively expensive (an Optuna
search inside every fold of every CPCV combination) and methodologically wrong
(in-fold model selection leaks the test distribution). Keep 0. Mirrors
PER_FOLD_SWING_HPO_TRIALS. See docs/living/PIPELINE_ARCHITECTURE.md KL-10."""

# ── Worker / parallelism caps ─────────────────────────────────────────────────
# Single source of truth for process/thread counts across all training entry points.
#
# Root cause of prior OOM: 5 parallel folds × 12 workers = 60 Python processes,
# each loading ~400-600 MB of numpy/pandas/scipy DLLs on Windows spawn → exhausted
# paging file. Fix: (1) serialize walk-forward folds on Windows via MAX_FOLD_WORKERS=1,
# (2) cap process pools via MAX_WORKERS, (3) cap BLAS/XGBoost threads via MAX_THREADS.
#
# With folds serialized the effective process count is MAX_WORKERS (not folds×workers),
# so MAX_WORKERS can be raised well above 4 on machines with adequate RAM.
#
# Tuned for 24-core / 32 GB Windows dev machine:
#   MAX_WORKERS=8   → 8 processes × ~500 MB DLL = ~4 GB overhead, 28 GB for data
#   MAX_THREADS=16  → 16 BLAS/XGBoost threads, 8 cores reserved for OS + I/O
#   MAX_FOLD_WORKERS=1 → folds run serially; prevents the multiplicative OOM
#
# To adjust for a different machine, change only these three constants.
MAX_WORKERS: int = 8 if _sys.platform == "win32" else (_os.cpu_count() or 8)
MAX_THREADS: int = 16 if _sys.platform == "win32" else (_os.cpu_count() or 24)
MAX_FOLD_WORKERS: int = 1 if _sys.platform == "win32" else 4

# ── Retraining schedule ──────────────────────────────────────────────────────
# Day of week to run the weekly retrain (0=Monday … 6=Sunday).
# Retrain only fires on this day — server restarts on other days don't trigger it.
# Change to a different day if Wednesday conflicts with market events.
# Set to -1 to disable scheduled retrains (Phase C diagnostic mode — retraining
# current 69-feature arch is counterproductive; re-enable after v200 ships).
RETRAIN_WEEKDAY: int = -1  # disabled — Phase C in progress (was: 2 = Wednesday)

# ── Regime model retrain cadence + gate (independent of RETRAIN_WEEKDAY) ─────
# The regime classifier is macro-level (slow-moving: VIX term structure, credit
# spreads, breadth) — daily retraining adds noise without benefit, and it must stay
# active even while the swing/intraday retrain is frozen (RETRAIN_WEEKDAY=-1), since
# regime_model_v5 carries the live book's sizing. 7 days (weekly) matches the cadence
# at which regime features meaningfully shift.
REGIME_RETRAIN_INTERVAL_DAYS: int = 7
# Promotion gate for the regime model. The model is 3-class (BULL/NEUTRAL/BEAR) scored
# by cross-entropy log-loss (random baseline = log(3) ≈ 1.099) — NOT a 2-class Brier
# score. (The old 0.22 threshold was a 2-class Brier value mis-applied to 3-class CE.)
# v5 reference: macro_F1 min=0.728, log_loss mean=0.358 → PASSES both.
REGIME_GATE_MACRO_F1_MIN: float = 0.60
REGIME_GATE_LOG_LOSS_MAX: float = 0.45

# ── Per-model retrain freeze (Alpha-v4 P0, 2026-06-07) ───────────────────────
# The large-cap daily cross-sectional swing ranker is DEAD on the honest harness
# (per-fold CPCV +0.22, t=0.17 — noise; MODEL_STATUS.md). Stop wasting nightly
# compute on it and stop promoting noise. Frozen as a non-production benchmark.
# Set True to resume the nightly swing retrain. (5/5-LLM review: "freeze the dead
# large-cap daily XS-ML.")  Honored by scripts/retrain_cron.run_swing.
SWING_ENABLED: bool = False

# ── Phase C validated feature keep-list (from A1 IC diagnostic 2026-05-13) ──
# Only these 14 features showed positive, stable IC (IR >= 0.40 at h5).
# All other features (technical noise, WQ alphas, short-momentum) are dropped.
# Source: data/diagnostics/feature_ic/20260513T124800Z/ic_summary.csv
PHASE_C_FEATURE_KEEP_LIST: tuple = (
    # Tier 1 — strong, stable IC (IR >= 1.0)
    "momentum_252d_ex1m",    # IR=1.99, IC grows with horizon (long-horizon factor)
    "vol_regime",            # IR=1.87
    "profit_margin",         # IR=1.40
    "operating_margin",      # IR=1.24
    "price_to_52w_high",     # IR=1.11
    "pe_ratio",              # IR=1.05 (sign: lower PE = better)
    # Tier 2 — additive, positive IC (IR 0.40–1.0)
    "range_expansion",       # IR=0.89
    "price_to_52w_low",      # IR=0.76
    "gross_margin",          # IR=0.73
    "volume_trend",          # IR=0.71
    "vrp",                   # IR=0.55
    "revenue_growth",        # IR=0.44 (mapped from revenue_growth_yoy)
    "near_52w_high",         # IR=0.46
    "trend_consistency_63d",  # IR=0.40
)

# ── Phase C+ feature keep-list: 14 IC features + 3 Opus interaction terms ────
# For v201+. Adds cross-terms among high-IC features per Opus 4.7 recommendation.
PHASE_C_PLUS_FEATURE_KEEP_LIST: tuple = PHASE_C_FEATURE_KEEP_LIST + (
    "ix_momentum_vol",  # momentum_252d_ex1m x vol_regime (trending + low-vol)
    "ix_quality_at_high",  # price_to_52w_high x profit_margin (quality near highs)
    "ix_vrp_range",  # vrp x range_expansion (vol premium confirming breakout)
)

# ── Phase C v2 feature keep-list: 17 + 2 sector-neutral features ──────────────
# v207+: adds sector-neutral momentum features (stock momentum minus sector ETF
# 20d momentum) to remove sector-concentration bias.  Opus 4.7 identified this
# as the highest-EV fix after regime gating failed — the 17-feature set has an
# absolute momentum bias that makes the ranker a concentrated sector bet in
# trending periods (energy 2022, tech 2024), which is the primary cause of
# Fold 2 and Fold 4 losses.  Both features are PIT-clean (ETF bars from
# sector_etf_history.parquet, not live API calls).
PHASE_C_V2_FEATURE_KEEP_LIST: tuple = PHASE_C_PLUS_FEATURE_KEEP_LIST + (
    "momentum_20d_sector_neutral",   # 20d stock return minus sector ETF 20d return
    "momentum_60d_sector_neutral",   # 60d stock return minus sector ETF 20d return
)

# ── Swing model (Phase C — LambdaRank on 14 IC-validated features) ───────────
# v200: XGBoost LambdaRank, 14 features, 21d forward rank target, 5 folds
# Previous: XGBoost binary classifier, 69 features, triple_barrier — Sharpe +0.106
# Rationale: A1 shows only 14/69 features have positive IC. LambdaRank trains
#   for cross-sectional ranking (which is what we do in the PM) rather than
#   per-symbol binary classification. A3 shows signal clearly exists (factor
#   Sharpe +1.399) — ML should learn to exploit it better than rule-based ranking.
SWING_RETRAIN: dict = dict(
    model_type="lambdarank",
    label_scheme="lambdarank",
    hpo_trials=50,              # Optuna: num_leaves, lr, min_child, lambda_l2
    fetch_fundamentals=False,   # avoid OOM on Windows
    n_workers=MAX_WORKERS,
    walk_forward_folds=5,
    walk_forward_years=6,
    exclude_risk_off_days=True,
    use_union_label=False,
    feature_keep_list=PHASE_C_V2_FEATURE_KEEP_LIST,  # v216: +sector-neutral momentum (19 features)
)

# ── Option C (label horizon) ─────────────────────────────────────────────────
# LABEL_HORIZON_DAYS: forward-return horizon for the swing label (default 5).
# Hypothesis C: 5d label is too noisy in trending regimes (AI rally 2023-24
# played out over 10-15 days). Larger horizon = thesis has time to develop.
#
# When != 5, training.py mutates module-level FORWARD_DAYS / STEP_DAYS /
# EMBARGO_WINDOWS at train_model entry. Set via env or by editing here.
#
# LABEL_ABS_HURDLE: optional ADDITIONAL absolute-return floor for cross-sectional
# label (in addition to the top-20% Sharpe-normalized cut). Currently swing has
# NO absolute hurdle; this is a new opt-in gate. Scales linearly with horizon
# (5d:0.003 → 10d:0.006 → 15d:0.009). Set to 0.0 to disable.
LABEL_HORIZON_DAYS: int = 20  # v216: 20d aligns with 40-bar max_hold_bars (was 5d — too noisy)
LABEL_ABS_HURDLE_5D: float = 0.0  # 0.0 = disabled (default). 0.003 = 0.3% per 5d.

# ── Option B (regime-split training) ─────────────────────────────────────────
# REGIME_SPLIT_VIX_THRESHOLD: when > 0, train two separate models — one fit on
# rows where VIX < threshold (calm/trending) and one on VIX >= threshold
# (shock/volatile). Inference selects model based on current VIX level.
#
# Hypothesis B: fold 4 (tariff shock, high VIX) and fold 3 (AI rally, low VIX)
# need opposite mean-reversion vs momentum biases. A single model cannot serve
# both. Default 20.0 (S&P long-run median ~19). Range [15, 30] per spec.
#
# 0.0 = disabled (single-model path, current behaviour).
REGIME_SPLIT_VIX_THRESHOLD: float = 0.0

# Gate thresholds matching the manual training gates (docs/ML_EXPERIMENT_LOG.md)
# LEGACY — used ONLY when GATE_MODE="mean_sharpe" (the reproduction path). Under the
# default GATE_MODE="significance", mean Sharpe is an economic-materiality FLOOR
# (PAPER_GATE_MIN_MEAN_SHARPE / CAPITAL_GATE_MIN_MEAN_SHARPE), NOT the primary
# discriminator. Do NOT delete — these reproduce the historical 0.80 behavior. See
# the Phase-4 significance-first block below and docs/living/DECISIONS.md.
SWING_GATE = dict(
    min_avg_sharpe=0.80,        # LEGACY (mean_sharpe mode): avg Sharpe across all folds
    min_fold_sharpe=-0.30,      # LEGACY (mean_sharpe mode): no single fold below this
)

# ── Intraday model ───────────────────────────────────────────────────────────
# DISABLED (Phase C, 2026-05-13): A1 IC diagnostic shows no intraday alpha at
# daily bar level. 15bps round-trip cost wipes any marginal edge. Mothballed
# until a dedicated intraday-IC diagnostic (5-min bar level) confirms signal.
# Re-enable by setting INTRADAY_ENABLED = True.
# Alpha-v4 P0 (2026-06-07): this flag now ALSO freezes the nightly intraday
# RETRAIN (honored by scripts/retrain_cron.run_intraday + retrain_intraday.main),
# not just live intraday scanning. Intraday 5-min XS-ML is DEAD (per-fold CPCV
# -2.80, t=-6.85; MODEL_STATUS.md). Set True to resume scanning AND retraining.
INTRADAY_ENABLED: bool = False

INTRADAY_RETRAIN: dict = dict(
    days=730,
    n_workers=MAX_WORKERS,
    fetch_spy=True,
    use_ranker=False,
    top_n_by_liquidity=None,    # None = full Russell 1000 universe
    exclude_risk_off_days=True,  # Phase R6/88: down-weight (0.3×) not exclude RISK_OFF
    wf_folds=5,                  # Phase 88: 5 folds (was 3)
)

INTRADAY_DAILY_FEATURE_PROVIDER: str = "aggregate_5min"
"""Source for the per-symbol DAILY bars that feed intraday 52-week / vol-percentile
features. The intraday 5-min bars come from the Polygon cache; the daily bars feed a
separate set of features. History of this knob:
  - 'alpaca' (original): caps at ~100 recent daily bars on this deployment — silently
    degraded 52w/vol features to 0.5 defaults across most of the training window.
  - 'yfinance' (PR #343): full multi-year daily history, BUT (a) its cache compared the
    requested range against datetime.date and raised "can't compare datetime.datetime to
    datetime.date" when the per-fold path passed datetime — returning 0/703 symbols (now
    type-fixed), and (b) it can rate-limit (YFRateLimitError) on 150-700 symbols.
  - 'aggregate_5min' (DEFAULT): aggregate the ALREADY-LOADED in-memory 5-min bars to
    daily OHLCV (open=first, high=max, low=min, close=last, volume=sum). Zero network,
    no rate-limit, and daily coverage == the 5-min span the folds are built from, so a
    daily/5-min span mismatch can never empty the per-fold matrix. This is the most
    robust choice for the per-fold CPCV path and is therefore the default.
    TRADEOFF: the 5-min cache is ~2yr, so windows near the cache START get <252 trading
    days of backward lookback for the 52w/vol features (degrades to a shorter window,
    NOT to 0.5 defaults). For a 504-day run with ~2yr (≈504-day) of 5-min cache, only
    the earliest ~252 trading days have a truncated 52w lookback; the rest get the full
    252d. Materially: the 52w/vol features are a small slice of the intraday feature set
    and degrade gracefully, so this does not invalidate the result.
  - 'polygon': Polygon daily cache + S3 (full history) if available.
Set to 'yfinance'/'alpaca'/'polygon' to use a network/full-history source instead."""

# LEGACY — used ONLY when GATE_MODE="mean_sharpe". See SWING_GATE note above.
INTRADAY_GATE = dict(
    min_avg_sharpe=1.00,        # LEGACY (mean_sharpe mode): recalibrated from 1.50 — best honest 365d result +0.786 (v51 Run B)
    min_fold_sharpe=-0.30,      # LEGACY (mean_sharpe mode): no single fold below this floor
)

# ── Deflated Sharpe Ratio — selection bias correction ────────────────────────
# N_TRIALS_TESTED: the number of distinct model variants evaluated across the
# project's lifetime. Used by the Deflated Sharpe Ratio (Bailey & López de Prado
# 2014) to penalise observed Sharpe for selection bias from iterative development.
#
# This is the SINGLE SOURCE OF TRUTH for this value. It must be imported by
# any code computing DSR; never hardcode a duplicate. Update whenever we
# materially try a new model architecture, label scheme, or feature set.
# Record the update in docs/ML_EXPERIMENT_LOG.md with the justification.
#
# History:
#   15   — original (under-stated: only counted explicit HPO trials, not all variants)
#   200  — corrected 2026-05-11 (R1): iterations 1-6 + phases 18-87 + R-series ≈ 200
#   200  — 2026-05-12: v192-v195 retrains + R5 + R2 ablation = ~205; rounded 200
#   250  — updated 2026-05-30: LX1–LX9 campaign (9 experiments) + v217–v224 retrains
#          + 4 audit rounds with repeated WF re-runs ≈ 225 total; rounded to 250.
#          Higher N makes DSR harder to pass — correct direction for selection bias.
#   300  — updated 2026-05-31: audit rounds 5-13 (9 more pipeline-verification WF runs)
#          + Phase 1/2/3 gate-overhaul validation runs ≈ 275 total; rounded to 300.
#          Conservative (higher N = harder DSR) — correct direction.
N_TRIALS_TESTED: int = 300

# ── CRITICAL-1: DSR implausibility ceiling ────────────────────────────────────
SHARPE_IMPLAUSIBILITY_CEILING: float = 3.0
"""Annualised Sharpe above which a result flags for mandatory human review instead
of auto-promotion. A daily-Sharpe > 3.0 net of costs is empirically implausible
for a retail equity strategy — capital-deployment artifacts or leakage are the
likeliest cause. Does NOT block gate_passed() — enforced by the promotion runner
(walkforward_tier3.py) via requires_human_review(). See CRITICAL-1."""

DSR_SATURATION_P: float = 0.999
"""DSR p-value above which the DSR gate is reported as SATURATED (non-binding).
A NOTE is emitted so reviewers know DSR provided no selection-bias screening.
Basis: deflated_sharpe_ratio saturates to ~1.0 for Sharpe > ~2 at n_obs~250."""

# ── CRITICAL-2: Capital deployment tracking ───────────────────────────────────
TARGET_DEPLOYMENT_PCT: float = 1.0
"""Target capital deployment fraction for deployment-adjusted Sharpe (1.0 = 100%
fully invested). Daily returns are rescaled by (target / actual_deployment) before
computing the diagnostic deployment-adjusted Sharpe so strategies with different
position sizing are comparable. Diagnostic only — never a gate. See CRITICAL-2."""

MIN_DEPLOYMENT_PCT_WARN: float = 0.10
"""Avg capital deployment below which a LOW DEPLOYMENT WARNING is issued.
WARNING only — does not block gate_passed(). Basis: 3% sizing × 5 max positions
= 15% ceiling; < 10% means most days are flat and Sharpe denominator is tiny."""

DEP_ADJ_MAX_SCALE: float = 50.0
"""Maximum scale factor for deployment-adjusted Sharpe per day. Caps
(target_deployment / actual_deployment) to prevent a near-zero-deployment day
from exploding the diagnostic. Diagnostic only."""

# ── MEDIUM-1: Calmar vol-floor ────────────────────────────────────────────────
USE_CALMAR_VOL_FLOOR: bool = True
"""When True, no-DD folds compute Calmar using max(0.5 * monthly_vol, MIN_CALMAR_FLOOR_DD)
as a floor drawdown instead of CAL_NO_DD_SENTINEL=+5.0. Prevents tight-stop intraday
strategies from trivially passing the Calmar gate on no-drawdown folds.
Set False to reproduce legacy sentinel behavior (for baseline comparisons)."""

MIN_CALMAR_FLOOR_DD: float = 0.01
"""Absolute floor drawdown (1%) when no-DD fold has insufficient daily return data
to estimate a vol-based floor. Prevents the free-pass CAL_NO_DD_SENTINEL on
short folds. See MEDIUM-1."""

# ── MEDIUM-3: Data span gate ──────────────────────────────────────────────────
MIN_DATA_SPAN_TRADING_DAYS: int = 250
"""Hard minimum trading-day span data must cover before WF/CPCV runs.
Below this, folds degenerate (e.g. 55-day yfinance fallback → 9-day folds).
Raises DataSpanError when ENFORCE_MIN_DATA_SPAN=True. See MEDIUM-3."""

ENFORCE_MIN_DATA_SPAN: bool = True
"""When True, raise DataSpanError if data span < MIN_DATA_SPAN_TRADING_DAYS.
Set False for warn-only legacy behavior (e.g. intentional short-window experiments)."""

# ── Phase 2: Regime gate overhaul ────────────────────────────────────────────
REGIME_SCHEME: str = "coarse3"
"""Regime label scheme for fold stratification and worst-regime-sharpe gate.
'coarse3' (default): BULL/BEAR/NEUTRAL — 3 buckets with expanding-quantile VIX
  thresholds (PIT-correct, no look-ahead). Gives enough obs per bucket for the
  gate to be meaningful.
'legacy16': VIX-quartile(4) × trend(2) × momentum(2) = up to 16 buckets. The
  original scheme; produces too-sparse per-bucket obs for most fold windows.
  Retained for backward compat and baseline comparison."""

REGIME_BULL_VIX_PCTILE: float = 25.0
"""VIX expanding-percentile threshold below which a day is BULL-eligible (coarse3).
Default 25 = lower quartile. VIX below this AND SPY above REGIME_SPY_MA_BULL-day
MA → BULL label."""

REGIME_BEAR_VIX_PCTILE: float = 75.0
"""VIX expanding-percentile threshold above which a day is BEAR (coarse3).
Default 75 = upper quartile. VIX above this OR SPY below REGIME_SPY_MA_BEAR-day
MA → BEAR label."""

REGIME_SPY_MA_BULL: int = 50
"""SPY MA period for BULL regime detection (coarse3). SPY must be above this
MA for a day to qualify as BULL. Default 50 (50-day MA)."""

REGIME_SPY_MA_BEAR: int = 200
"""SPY MA period for BEAR regime detection (coarse3). SPY below this MA → BEAR
regardless of VIX level. Default 200 (200-day MA)."""

REGIME_MIN_OBS: int = 20
"""Minimum daily-return observations required per regime bucket before that
bucket's Sharpe is included in worst_regime_sharpe. Below this, the bucket is
too noisy (sqrt(252) on 5 returns is not a Sharpe). Default 20 trading days."""

REGIME_VIX_WARMUP_DAYS: int = 60
"""Minimum VIX history before a day gets a non-NEUTRAL coarse3 label. Below
this, the expanding-quantile thresholds are unstable. Default 60 trading days."""

ALLOW_NO_REGIME_GATE: bool = False
"""When False (default), worst_regime_sharpe=None causes gate_passed() to return
False with a clear 'REGIME DATA INSUFFICIENT' message. This prevents the silent
pass that has historically made the regime gate a no-op.
Set True to restore old silent-pass behavior (for diagnostic runs or legacy
comparisons where regime data is unavailable)."""

# ── Phase 3: CPCV path t-stat ─────────────────────────────────────────────────
CPCV_MIN_TSTAT: float = 2.0
"""Minimum path-Sharpe t-statistic for CPCV significance.
t = mean_path_sharpe / (std_path_sharpe / sqrt(N_eff)), where N_eff = n_folds
(NOT n_paths — the C(k,p) paths reuse the same k folds and are strongly
correlated; treating them as n_paths independent experiments overstates
significance by ~sqrt(n_paths/n_folds)). t > 2.0 ≈ 95% one-sided.
Gate is off by default (require_tstat_gate=False); reported as WARNING when
below threshold. See HIGH-3."""

# ── Phase-4: significance-first two-tier promotion gate ───────────────────────
# Replaces mean-Sharpe>=0.80 (swing) / >=1.00 (intraday) as the PRIMARY
# discriminator with a significance-first multi-criteria gate.
#
# WHY: 0.80/1.00 were calibrated against now-struck IN-SAMPLE numbers (intraday
# +5.14, QualityShort +3.25). A bare mean-Sharpe threshold cannot separate a
# +0.22 / t=0.17 noise result from a +0.546 / t=2.26 genuine-signal result — both
# are below 0.80 yet one is significant and one is noise. The fix is to gate on
# statistical significance (path-Sharpe t-stat, sign-consistency, tail floor)
# with mean Sharpe relegated to an economic-materiality floor. See
# docs/living/DECISIONS.md and docs/living/PIPELINE_ARCHITECTURE.md §7.
#
# Behind a flag for reversibility + historical re-scoring:
#   "significance" (default, owner-approved): new two-tier gate below.
#   "mean_sharpe"  (legacy): faithful reproduction of the 0.80/1.00 mean gate
#                  with t-stat WARN-only — a true no-op vs pre-Phase-4 main.
GATE_MODE: str = "significance"   # "significance" (new) | "mean_sharpe" (legacy)

# Two tiers:
#   PAPER   — forward-validate, deploy to paper, NO real capital.
#   CAPITAL — real money; PAPER conditions PLUS stricter t-stat / power floors.

# PAPER (forward-validate) tier — deploy to paper, NO real capital
PAPER_GATE_MIN_TSTAT: float = 2.0          # ~p<0.05 one-sided; == existing CPCV_MIN_TSTAT
PAPER_GATE_MIN_PCT_POSITIVE: float = 0.75  # sign-consistency floor (existing %pos gate)
PAPER_GATE_MIN_P5_SHARPE: float = 0.0      # tail floor — STRICTER than legacy MIN_FOLD -0.30
PAPER_GATE_MIN_MEAN_SHARPE: float = 0.35   # economic-materiality floor (matches pre-reg insider bar)

# CAPITAL tier — real money; PAPER conditions PLUS:
CAPITAL_GATE_MIN_TSTAT: float = 2.5        # multiple-testing haircut (~10-15 strategy shots)
CAPITAL_GATE_MIN_N_FOLDS: int = 10         # power floor (N_eff=n_folds; 8 is too few for capital)
# Alpha-v4 P0 recalibration (2026-06-07): 0.50 → 0.45. The legacy SR≥0.80 promotion
# bar is RETIRED (GATE_MODE='significance' is the default; 'mean_sharpe'/0.80 is kept
# only for legacy reproducibility). On N_eff≈8 a high Sharpe LEVEL selects for
# overfitting — real edges live at ~0.4–0.7 — so robustness (t-stat + fold-consistency
# + worst-regime survivability floor, and the residual-alpha-t DIAGNOSTIC) is primary
# and the Sharpe floor is the economic-materiality backstop, not the discriminator.
CAPITAL_GATE_MIN_MEAN_SHARPE: float = 0.45
CAPITAL_GATE_REQUIRE_PAPER_CONFIRMATION: bool = True  # OR-path: t>=2.5 OR documented live-paper confirmation

# ── Alpha-v6 Phase 0: Track B (book-delta) acceptance ─────────────────────────
# PRE-REGISTERED criteria for the Track B gate (scripts/walkforward/book_gate.py;
# NEXT_PHASE_BLUEPRINT_2026-06.md Phase 0 / consensus C2). Track B judges
# crisis-diversifiers / risk premia (TSMOM extensions, index VRP, tail hedges)
# on their CONTRIBUTION TO THE COMBINED BOOK at a small fixed risk budget —
# NOT on the standalone Track-A significance gate, whose worst-regime backstop
# a crisis-diversifier intentionally fails (P0 RESULT 2026-06-10: tsmom t=6.72
# passes the significance core, fails PAPER only on worst_regime_sharpe;
# lowering the t-bar was empirically refuted — 3/5 zero-SR nulls cleared t>=2.0).
# Frozen 2026-06-10 BEFORE any Track B candidate was evaluated. The gate is a
# PURE function of these constants — it never mutates them.
# Track B NEVER auto-promotes to CAPITAL: it gates PAPER-level book inclusion
# only; capital requires explicit owner sign-off plus a codified tail budget.
TRACKB_MIN_SHARPE_DELTA: float = 0.10   # combined-book Sharpe must improve by >= this
TRACKB_MIN_CALMAR_DELTA: float = 0.0    # book Calmar delta must be >= this (must not worsen)
TRACKB_MAX_DD_DELTA: float = 0.0        # max allowed book maxDD worsening (0 = must not worsen;
#                                         maxDD <= 0 convention, sleeve_allocator._maxdd)
TRACKB_MAX_CORR: float = 0.30           # candidate corr to the existing book must be < this
TRACKB_MIN_STANDALONE_SR: float = 0.20  # candidate's own vol-targeted Sharpe must be > this
TRACKB_MAX_RISK_BUDGET: float = 0.25    # candidate blended at <= this fraction of book risk.
#   AMENDED 2026-06-11 (0.10 -> 0.25), owner-approved REGISTERED amendment. The original 0.10 was
#   arbitrarily conservative: at b=0.10 the TRACKB_MIN_SHARPE_DELTA>=0.10 bar implicitly demands a
#   candidate standalone SR >~ 1.11 (since dSR ~= b*(SR_cand - corr*SR_book)), which NO realistic
#   diversifier can clear. The first real run (TSMOM vs PEAD book, #450) improved the book on EVERY
#   metric (Sharpe 0.41->0.50, maxDD shallower, Calmar up, lower vol) yet failed dSharpe by 0.0115.
#   A diversifier is realistically a minority-but-meaningful share of book risk (the validated
#   Phase-2/3 book ran trend at ~40-50%); 0.25 = a quarter, the CONSERVATIVE end of the
#   owner-endorsed 25-40% range, chosen on principle (NOT reverse-engineered to make TSMOM pass —
#   the budget-vs-dSharpe curve is reported alongside the re-test). See DECISIONS 2026-06-11.
TRACKB_JOINT_TAIL_PCTL: float = 0.01    # tail-overlap test: tail size = max(3, floor(pctl *
#                                         n_days)) worst days of EACH book (the base book, and
#                                         the vol-targeted candidate) on the shared evaluation index
TRACKB_MAX_TAIL_OVERLAP: float = 0.30   # PASS iff |base_worst INTERSECT cand_worst| / n_tail
#                                         <= this. Registered 2026-06-10, before any Track B
#                                         candidate was evaluated. This IS the blueprint's
#                                         registered criterion ("candidate's worst-1% days must
#                                         not coincide with the book's worst-1%"); an earlier
#                                         mean-return-on-tail-days implementation was an
#                                         UNREGISTERED divergence — maskable by one large
#                                         positive tail day, and ~43% false-REJECT on genuinely
#                                         independent diversifiers. Under independence the
#                                         expected overlap fraction is ~n_tail/n_days (~1%),
#                                         far below 0.30; a co-crasher overlaps heavily and fails.
# Track B also enforces SHARPE_IMPLAUSIBILITY_CEILING (above, 3.0) on the candidate's
# standalone vol-targeted Sharpe — an SR > 3.0 FAILS the gate as look-ahead/degenerate,
# mirroring Track A's CRITICAL-1 ceiling intent.

# ── Feature flags ─────────────────────────────────────────────────────────────
# USE_NIS_FEATURES: include NIS/macro LLM sentiment features in swing training.
#   False (default): NIS excluded — ~80% NaN creates a time-proxy (NaN = pre-May-2025
#   regime). XGBoost learns the time index, not the sentiment. AUC inflated on 2025 folds.
#   True: re-enable only after ≥2yr backfill via scripts/backfill_stock_nis_history.py
#   and scripts/backfill_macro_nis_llm.py. NIS still used at PM gate layer regardless.
#   Why: same pattern as USE_REALIZED_R_LABELS — code/infra preserved, behavior gated.
USE_NIS_FEATURES: bool = False

# ── Model artifact integrity ──────────────────────────────────────────────────
REQUIRE_TRAINED_THROUGH: bool = True
"""When True (default), model save() refuses to persist a model whose
trained_through is None. A model without a verifiable training cutoff cannot be
OOS-gate-evaluated, and shipping one (as happened with v63/v224) produces silently
in-sample results. Set False ONLY for explicit diagnostic saves that will never
be gate-evaluated."""

# ── Phase 93: FMP quarterly fundamentals ─────────────────────────────────────
# When True and data/fundamentals/fmp_fundamentals_history.parquet exists,
# training/inference loads PIT-correct quarterly fundamentals from FMP and
# OVERRIDES any value sourced from the older EDGAR annual parquet. Falls back
# to EDGAR (or zeros) when the FMP parquet is absent. Toggle False to A/B vs
# the EDGAR-only baseline without removing the parquet.
USE_FMP_FUNDAMENTALS: bool = True

# Quarters of history to fetch per symbol during full backfill.
# 40q ≈ 10y; 100q is FMP standard plan max.
FMP_QUARTERLY_LOOKBACK_QUARTERS: int = 40

# USE_REALIZED_R_LABELS: switch intraday labeling from cross-sectional top-20%
#   to absolute realized-R threshold. Failed walk-forward gate (avg Sharpe -4.514,
#   AUC ~0.55). Structurally incompatible with cs_normalize. Keep False.
USE_REALIZED_R_LABELS: bool = False

# MIN_REALIZED_R: threshold for realized-R labeling (only applies when
#   USE_REALIZED_R_LABELS=True). 0.35 was tried — gate failed.
MIN_REALIZED_R: float = 0.35


# ── P1: BenignModel — regime-filtered training ────────────────────────────────
# When BENIGN_FILTER_ENABLED=True, the training pipeline filters out any
# (symbol, window) row where the PIT composite regime score < BENIGN_REGIME_THRESHOLD.
# This prevents the model from learning patterns that only appear during bear markets
# (where our signals have no demonstrated edge).
#
# Default: False — existing retrain behaviour is unchanged until explicitly enabled.
# Enable via CLI flag: --benign-model (sets this to True for the run).
BENIGN_FILTER_ENABLED: bool = False

# Regime threshold: rows with composite_score < this are excluded from training.
# 0.5 = at least 3 of 5 components must be bullish (spy_ma50, spy_ma200, vix_term,
# breadth, credit). Historically filters ~21% of days (2018-2026).
BENIGN_REGIME_THRESHOLD: float = 0.5

# P1 feature keep-lists. When BENIGN_FILTER_ENABLED=True, the worker retains ONLY
# features in the relevant list after computing the full feature dict.
# Verified against app/ml/features.py output — names must match exactly.
#
# v217 feature set (2026-05-24): 17 cross-regime rank-stable features selected via
# IC audit (Spearman IC vs 20d forward returns, 2019-2026, regime-split by year).
# Selection criteria: positive 2022 bear-year IC OR meaningful overall IC IR.
# Dropped from v216: gross_margin (-0.039 2022 IC), revenue_growth (-0.019 2022),
# trend_consistency_63d (-0.014 2022), momentum_60d_sector_neutral (≈0 overall),
# ix_vrp_range (≈0 overall), near_52w_high (redundant with price_to_52w_high),
# ix_quality_at_high (redundant interaction).
BENIGN_SWING_FEATURES: tuple = (
    # Momentum (4) — regime-stable, positive IC in 2022 bear
    "momentum_252d_ex1m",
    "ix_momentum_vol",
    "price_to_52w_high",
    "price_to_52w_low",
    # Counter-trend / mean-reversion (3) — positive 2022 IC, orthogonal to momentum
    "reversal_5d_vol_weighted",
    "downtrend",
    "range_expansion",
    # Quality / fundamentals (3) — FMP PIT-correct values; positive 2022 IC
    "operating_margin",
    "profit_margin",
    "pe_ratio",
    # Risk / volatility regime (3)
    "vol_regime",
    "vrp",
    "vol_percentile_52w",
    # Flow (1)
    "volume_trend",
    # WorldQuant alphas (3) — positive 2022 IC, diversified signal
    "wq_alpha35",
    "wq_alpha40",
    "wq_alpha43",
)  # 17 total — v217

BENIGN_INTRADAY_FEATURES: tuple = (
    # Momentum (6)
    "rsi_14", "momentum_5d", "momentum_20d", "price_change_pct",
    "consecutive_days", "adx_14",
    # Trend (5)
    "price_above_ema20", "price_above_ema50", "ema20_dist", "uptrend", "downtrend",
    # Volatility (4)
    "atr_norm", "volatility", "vol_percentile_52w", "parkinson_vol",
    # Volume (3)
    "volume_ratio", "volume_trend", "vwap_distance_20d",
    # 52-week position (2)
    "price_to_52w_high", "near_52w_high",
    # Regime / macro (5)
    "vix_term_ratio", "spy_above_ma50", "spy_above_ma200",
    "breadth_rsp_spy_ratio_20d", "credit_hyg_ief_20d",
)  # 25 total


# ── P0: Sacred holdout enforcement ────────────────────────────────────────────
# SACRED_HOLDOUT_START defines the boundary date — data ON or AFTER this date is
# RESERVED for the single, final promotion-candidate evaluation. It must NEVER
# be touched during development, tuning, walk-forward, CPCV, or any iterative
# experiment. Doing so reintroduces selection bias and invalidates every honest
# baseline number we have.
#
# Enforcement (defense in depth — guard fires at multiple layers):
#   1. ModelTrainer.train_model() before fetching bars
#   2. ModelTrainer._build_rolling_matrix() on the assembled date spine
#   3. scripts/walkforward_tier3.py main() before kicking off any folds
#   4. scripts/walkforward/cpcv.py run_cpcv() when computing end_all
#   5. scripts/walkforward/engine.py FoldEngine.run() when computing end_all
#
# Bypass: pass `allow_sacred_holdout=True` (Python) or `--allow-sacred-holdout`
# (CLI). Bypass logs a prominent banner warning. Use ONLY for the final
# promotion candidate run, ONCE. Do not develop with this flag set.
#
# Date format is ISO (YYYY-MM-DD). Boundary is INCLUSIVE: end_date == this date
# is rejected (the holdout day itself is sacred).
#
# History:
#   2025-11-09 — original boundary (set 2026-05-09). Exhausted: all v182-v185
#                retrains used --allow-sacred-holdout during pipeline debugging.
#   2026-11-09 — reset 2026-05-10. Previous boundary considered consumed.
#                Next promotion candidate evaluation must use this boundary.
SACRED_HOLDOUT_START: str = "2026-11-09"


def _parse_sacred_holdout_start():
    from datetime import date as _date
    y, m, d = SACRED_HOLDOUT_START.split("-")
    return _date(int(y), int(m), int(d))


def assert_no_sacred_holdout(end_date, *, allow_sacred_holdout: bool = False,
                             context: str = "training") -> None:
    """Hard-fail guard: reject any run whose data window reaches the sacred holdout.

    Raises RuntimeError if `end_date >= SACRED_HOLDOUT_START` and bypass not set.
    Logs a prominent banner warning if bypass IS set.

    Args:
        end_date: a `datetime.date`, `datetime.datetime`, pandas Timestamp,
                  or ISO date string representing the END of the data window.
        allow_sacred_holdout: explicit bypass for the one-shot promotion run.
        context: free-text label included in error/warning messages
                 ("training", "walk-forward", "cpcv", etc.).
    """
    import logging as _logging
    from datetime import date as _date, datetime as _datetime

    if end_date is None:
        return  # nothing to check

    # Coerce input → date
    if isinstance(end_date, str):
        end_d = _date.fromisoformat(end_date[:10])
    elif isinstance(end_date, _datetime):
        end_d = end_date.date()
    elif isinstance(end_date, _date):
        end_d = end_date
    else:
        # pandas.Timestamp and similar — has .date()
        end_d = end_date.date() if hasattr(end_date, "date") else end_date

    boundary = _parse_sacred_holdout_start()
    if end_d < boundary:
        return  # safe — entirely before holdout

    if allow_sacred_holdout:
        log = _logging.getLogger(__name__)
        banner = "=" * 72
        log.warning(banner)
        log.warning("SACRED HOLDOUT BYPASS ACTIVE — context=%s", context)
        log.warning("end_date=%s reaches/exceeds SACRED_HOLDOUT_START=%s",
                    end_d.isoformat(), boundary.isoformat())
        log.warning("This must ONLY be the final, one-shot promotion-candidate run.")
        log.warning("If this is anything else, ABORT NOW.")
        log.warning(banner)
        return

    raise RuntimeError(
        f"SACRED HOLDOUT VIOLATION ({context}): end_date={end_d.isoformat()} "
        f"reaches or exceeds SACRED_HOLDOUT_START={boundary.isoformat()}. "
        f"Data on/after this date is reserved for the single, final "
        f"promotion-candidate evaluation. To intentionally bypass for that "
        f"one-shot run, pass allow_sacred_holdout=True (or --allow-sacred-holdout). "
        f"See app/ml/retrain_config.py for details."
    )


# ── BUG-17/18: Purge horizon invariant ────────────────────────────────────────
# The swing OOS purge_days (85) must exceed feature_lookback + label_horizon + buffer.
# This invariant is checked at import time so a config drift (e.g. bumping
# LABEL_HORIZON_DAYS) is caught immediately rather than silently producing leakage.
SWING_PURGE_DAYS: int = 85        # must match --swing-purge-days CLI default
FEATURE_LOOKBACK_DAYS: int = 60   # max rolling feature window (252d excluded; it's for labels)


def _assert_purge_horizon_invariant() -> None:
    """Raise if swing purge_days < feature_lookback + label_horizon + buffer.

    Called at module import so any config drift is caught at startup, not at
    a future WF run hours later.
    """
    buffer = 5
    required = FEATURE_LOOKBACK_DAYS + LABEL_HORIZON_DAYS + buffer
    if SWING_PURGE_DAYS < required:
        raise RuntimeError(
            f"Purge invariant violated: SWING_PURGE_DAYS={SWING_PURGE_DAYS} < "
            f"FEATURE_LOOKBACK_DAYS({FEATURE_LOOKBACK_DAYS}) + "
            f"LABEL_HORIZON_DAYS({LABEL_HORIZON_DAYS}) + buffer({buffer}) = {required}. "
            f"Increase SWING_PURGE_DAYS in retrain_config.py or reduce LABEL_HORIZON_DAYS."
        )


_assert_purge_horizon_invariant()


# ── BUG-20: Deterministic retrain boundary ────────────────────────────────────

def retrain_as_of():
    """Return the WF boundary date for this retrain cycle: last completed Friday.

    Anchoring to last Friday makes weekly gate decisions deterministic — two
    retrains on the same cycle (e.g. crash-restart) use identical fold boundaries.
    Clamped below SACRED_HOLDOUT_START to prevent holdout contamination.
    """
    from datetime import date as _date, timedelta as _td
    today = _date.today()
    # Roll back to the most recent Friday (weekday 4); if today is Friday, use today.
    offset = (today.weekday() - 4) % 7
    last_friday = today - _td(days=offset if offset else 0)
    holdout = _parse_sacred_holdout_start()
    return min(last_friday, holdout - _td(days=1))
