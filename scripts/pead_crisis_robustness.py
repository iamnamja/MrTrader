"""
ALPHA v2 §1.2 — PEAD crisis-block robustness harness.

THE GO/NO-GO FOR LIVE PEAD. PEAD's validated long-only edge (CPCV mean Sharpe
+0.546) leans on a hard VIX>30 crisis block (no new entries when VIX>30). That
block lifts the crisis-fold tail P5 from -0.288 to +0.009 and %positive 80%->95%.
But VIX>30 is a hand-tuned threshold whose value rests on only ~3 crisis episodes
in the 2020-05..2026 data window. §1.2 asks: is the edge real, or is one hand-tuned
threshold sidestepping a couple of crises? This is the plan's pre-committed PAUSE
TRIGGER: if the edge only survives at VIX>30-exactly, or is carried by a single
episode, we PAUSE live PEAD.

This harness MIRRORS scripts/pead_cost_sensitivity.py exactly:
  * load PEAD data ONCE (re-fetch dominates runtime),
  * re-run the C(8,2) CPCV under each variant on BYTE-IDENTICAL data + folds,
    deterministic + anchored to retrain_as_of(),
  * a BASELINE row (the committed VIX>30 config) self-validates ~+0.546,
  * ASCII-safe output, artifacts written FIRST then a guarded table print.

────────────────────────────────────────────────────────────────────────────────
THREE SUB-ANALYSES (all on the one shared data load)
────────────────────────────────────────────────────────────────────────────────
A. THRESHOLD SWEEP — re-run with vix_block_all in {25,28,30,33,35, inf(no block)}.
   inf == block disabled == the raw edge with NO crisis protection. Stable across
   25..35 => the exact 30 is not load-bearing (robust). Good ONLY at 30 => overfit.
   The inf row quantifies what the block actually buys (expect P5 tail to drop).

B. LEAVE-ONE-CRISIS-OUT (LOCO) — PEAD does NOT trade during VIX>30 (it's blocked),
   so "removing a crisis" tests the POST-crisis drift rebound, not the blocked days.
   Mechanism: run the committed config ONCE (capturing each fold's daily equity
   curve), then for each crisis window recompute every path's Sharpe from the SAME
   equity curves with that window's calendar days MASKED OUT before differencing.
   This isolates "does +0.546 survive if one episode's P&L is removed" without
   re-simulating. The no-removal row recomputes from the unmasked curves and MUST
   match the committed baseline (self-check). If removing any single episode (esp.
   2022) collapses mean Sharpe toward 0 or flips %pos, the edge is episode-dependent.

C. GENERIC REGIME CONTROL — replace the discrete VIX>30 block with a regime-GENERAL
   control (NOT fit to crisis dates) and see if the edge survives:
     (c1) vol_target: a PIT exposure scalar = min(1, target_vol / realized SPY vol).
     (c2) trend: exposure scalar 0.0 when SPY < its 200d SMA (PIT), else 1.0.
   IMPORTANT — how the scalar actually acts in the equal-weight +0.546 book: it is
   NOT continuous gross vol-targeting. The committed PEAD book is EQUAL-WEIGHT
   (pead_conviction_size OFF), so the scalar has only two real effects:
     1. ENTRY BLOCK: when the scalar < regime_control_floor (default 0.50) it blocks
        ALL new entries that day (a vol/trend-gated hard block — the generic analogue
        of the discrete VIX>30 block). Above the floor, no entry is blocked.
     2. a coarse CONVICTION-SIZE tilt: the scalar multiplies signal confidence, which
        only feeds conviction sizing. With the equal-weight book the per-day gross is
        fixed and the band is narrow/saturating (~[0.75-1.25x] of equal weight), so
        between the floor and 1.0 the scalar barely moves realized gross — it does NOT
        smoothly scale total exposure down with rising vol.
   Implemented in PEADScorer (the single PIT entry-decision point, where the VIX
   block already lives) as opt-in kwargs; the discrete VIX block is turned OFF
   (vix_block_all=inf) so the generic control governs. PIT: vol/trend use only SPY
   closes <= the decision day. If PEAD survives under such a generic vol/trend-gated
   entry block, VIX>30 was just one instance of "de-risk in stress" and can be
   replaced -> robust. If it ONLY survives under the exact VIX>30 block -> overfit
   -> PAUSE.

Usage:
    # Full run (re-fetches ~1000 symbols / 6yr, runs the full CPCV many times — LONG):
    python scripts/pead_crisis_robustness.py

    # Just one sub-analysis:
    python scripts/pead_crisis_robustness.py --only threshold
    python scripts/pead_crisis_robustness.py --only loco
    python scripts/pead_crisis_robustness.py --only control

    # Fast smoke (tiny universe + short window + tiny CPCV; for tests/CI):
    python scripts/pead_crisis_robustness.py --smoke
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv()

from app.ml.retrain_config import MAX_THREADS as _max_threads

os.environ.setdefault("OMP_NUM_THREADS", str(_max_threads))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(_max_threads))

logger = logging.getLogger("pead_crisis_robustness")

# ── A. Threshold ladder. inf (math.inf) disables the block entirely (no protection).
DEFAULT_VIX_THRESHOLDS: List[float] = [25.0, 28.0, 30.0, 33.0, 35.0, math.inf]
SMOKE_VIX_THRESHOLDS: List[float] = [30.0, math.inf]

# Committed/validated config: VIX>30 block. Its CPCV mean Sharpe is ~+0.546; the
# BASELINE row should reproduce this closely (self-validation, like the cost sweep
# anchor). The baseline is the vix_block_all=30 threshold row.
BASELINE_VIX_BLOCK = 30.0
BASELINE_EXPECTED_SHARPE = 0.546
BASELINE_REPRO_TOL = 0.05  # |baseline - expected| above this -> harness divergence (LOUD)

# ── B. LOCO self-check: the (none removed) row recomputes path Sharpes from the SAME
# captured curves using the SAME grouping run_cpcv returned (result.path_fold_members),
# so it must reproduce run_cpcv's OWN mean Sharpe to a tight identity tolerance. A miss
# of more than this proves the LOCO grouping or curve-capture diverged from the real CPCV
# (real grouping/keying errors diverge by >> 1e-3). The floor is set by rounding, NOT
# float epsilon: run_cpcv stores each fold's sharpe as round(sharpe, 3) (AgentSimulator
# result.sharpe_ratio), while the LOCO baseline recomputes unrounded from the same curves
# — so per-path means can differ by up to ~5e-4 from that 3dp rounding alone. 1e-3 stays
# far below any real grouping divergence while not false-alarming on benign rounding.
LOCO_SELF_CHECK_TOL = 1e-3

# ── B. Crisis windows: VIX>30 episodes that fall INSIDE the ~2020-05..2026 window.
# (COVID-2020 spike is largely BEFORE the 2020-05 start -> only its tail is in-window;
#  2018Q4 is OUT of window entirely.) Dates are calendar ranges [start, end] inclusive.
# Each window masks the POST-crisis drift period it brackets (VIX>30 days are blocked
# anyway, so the masked P&L is the rebound when VIX falls back below 30).
CRISIS_WINDOWS: List[Dict] = [
    {"name": "covid_tail_2020", "start": date(2020, 5, 1), "end": date(2020, 6, 15),
     "note": "COVID spike mostly pre-window; only the elevated-VIX tail is in-window (partial)"},
    {"name": "bear_2022", "start": date(2022, 1, 21), "end": date(2022, 10, 21),
     "note": "2022 bear: repeated VIX>30 spikes (Jan/Feb, Jun, Sep-Oct) — the dominant episode"},
    {"name": "yen_carry_aug2024", "start": date(2024, 8, 1), "end": date(2024, 8, 12),
     "note": "Aug-2024 yen carry-unwind VIX spike (brief)"},
    {"name": "tariff_apr2025", "start": date(2025, 4, 3), "end": date(2025, 5, 9),
     "note": "Apr-2025 tariff crash VIX spike"},
]

# ── C. Generic controls to test (label -> scorer kwargs). vix_block_all=inf turns the
# discrete block OFF so the generic control alone governs de-risking.
GENERIC_CONTROLS: List[Dict] = [
    {"label": "vol_target(0.16)", "regime_control": "vol_target",
     "regime_control_target_vol": 0.16, "regime_control_floor": 0.50},
    {"label": "trend(SPY<200d)", "regime_control": "trend",
     "regime_control_trend_ma": 200, "regime_control_floor": 0.50},
]
SMOKE_GENERIC_CONTROLS: List[Dict] = [GENERIC_CONTROLS[0]]

LOG_DIR = ROOT / "logs"
ARTIFACT_DIR = ROOT / "logs"


def _setup_logging(log_path: Path) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, mode="w", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [pead_crisis] %(message)s",
        handlers=handlers,
        force=True,
    )


class _EquityCapturingStrategy:
    """Wrap a PEADStrategy to record each fold's (te_start, te_end, equity_curve).

    Mirrors pead_cost_sensitivity._ReturnCapturingStrategy: proxies everything to the
    inner strategy and only intercepts run_fold. We capture the per-fold daily equity
    curve (sorted [(date, equity), ...]) so the LOCO analysis can recompute masked
    path Sharpes WITHOUT re-simulating. reset() clears the accumulator per variant.
    """

    def __init__(self, inner):
        self._inner = inner
        # list of (fold_id, te_start, te_end, equity_curve) in run order
        self.fold_equity: List[Tuple[int, date, date, list]] = []

    def reset(self) -> None:
        self.fold_equity = []

    def run_fold(self, fold_idx, n_folds, tr_start, tr_end, te_start, te_end):
        fold = self._inner.run_fold(fold_idx, n_folds, tr_start, tr_end, te_start, te_end)
        # The captured curve is attached to the LAST run_fold; cpcv groups folds into
        # combos, so we also need to know which combo a fold belongs to. cpcv passes a
        # globally-unique fold_idx (combo_idx*n_boundaries + ti + 1), so fold_idx alone
        # disambiguates. We store the curve keyed by that id; the harness reconstructs
        # path membership from the captured order (see _recompute_paths_from_curves).
        # We cannot see combos here, so the harness re-derives them deterministically.
        return fold

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _sharpe_from_equity(equity_curve: list, mask_range: Optional[Tuple[date, date]] = None) -> Tuple[float, int]:
    """Recompute the daily-return Sharpe from an equity curve, mirroring AgentSimulator.

    equity_curve is sorted [(date, equity), ...]. We optionally DROP the (date, equity)
    points whose date falls in [mask_range.start, mask_range.end] inclusive, then
    difference the SURVIVING equity points into daily returns and annualize on 252 via
    StrategySimulator._sharpe — byte-identical to the production Sharpe path.

    Returns (sharpe, n_return_obs). Masking removes a contiguous chunk of dates; the
    return spanning the gap is dropped (we difference consecutive SURVIVING points but
    skip the single boundary diff across the removed window so a crisis's own rebound
    return is not re-introduced).
    """
    from app.backtesting.strategy_simulator import StrategySimulator
    if not equity_curve:
        return 0.0, 0
    pts = [(d.date() if hasattr(d, "date") else d, float(v)) for d, v in equity_curve]
    if mask_range is not None:
        lo, hi = mask_range
        kept = [(d, v, (lo <= d <= hi)) for d, v in pts]
    else:
        kept = [(d, v, False) for d, v in pts]
    surviving = [(d, v, masked) for (d, v, masked) in kept if not masked]
    if len(surviving) < 3:
        return 0.0, 0
    # Daily returns between consecutive surviving points. Skip any diff whose two
    # endpoints straddle a removed window (i.e. there was >=1 masked day strictly
    # between them) so the masked episode's P&L truly does not enter the series.
    survivor_dates = [d for d, _, _ in surviving]
    survivor_vals = [v for _, v, _ in surviving]
    # Build a set of masked dates for the straddle check.
    masked_dates = {d for d, _, masked in kept if masked}
    # Determine, for each consecutive surviving pair, whether any masked date lies
    # strictly between them. Since the curve is daily-sorted and masked dates are a
    # contiguous range, a straddle exists iff lo <= some_date <= hi sits between the
    # two surviving dates.
    rets: List[float] = []
    for i in range(1, len(survivor_vals)):
        d_prev, d_cur = survivor_dates[i - 1], survivor_dates[i]
        if mask_range is not None and any(d_prev < md < d_cur for md in masked_dates):
            continue  # straddles the removed window -> drop this boundary return
        prev, cur = survivor_vals[i - 1], survivor_vals[i]
        rets.append((cur - prev) / max(prev, 1e-9))
    if len(rets) < 2:
        return 0.0, 0
    return float(StrategySimulator._sharpe(rets, 252)), len(rets)


# ─── CPCV path membership ───────────────────────────────────────────────────────────
# NOTE: LOCO does NOT reconstruct the CPCV path grouping. run_cpcv is the SINGLE SOURCE
# OF TRUTH — it records exactly which global fold ids each surviving path aggregated
# (post purge/overlap guards) in CPCVResult.path_fold_members, and the LOCO capture run
# consumes THAT directly. A prior version reimplemented the grouping here and silently
# diverged (it omitted the BUG-23 expanding-window overlap guard), so every LOCO row was
# computed on the wrong folds. The reconstruction has been DELETED; do not reintroduce it.


def _path_metrics(path_sharpes: List[float]) -> Dict:
    """Mean / t-stat (N_eff=n_folds via CPCVResult) / %pos / P5 / P95 over a path list."""
    import numpy as np
    if not path_sharpes:
        return {"mean_sharpe": 0.0, "pct_positive": 0.0, "p5_sharpe": 0.0,
                "p95_sharpe": 0.0, "n_paths": 0, "_arr": np.array([], dtype=float)}
    arr = np.array(path_sharpes, dtype=float)
    return {
        "mean_sharpe": round(float(np.mean(arr)), 4),
        "pct_positive": round(float(np.mean(arr > 0)), 4),
        "p5_sharpe": round(float(np.percentile(arr, 5)), 4),
        "p95_sharpe": round(float(np.percentile(arr, 95)), 4),
        "n_paths": len(arr),
        "_arr": arr,  # internal; stripped before serialization
    }


def _tstat(arr, n_folds: int) -> float:
    """Path-Sharpe t-stat with N_eff = n_folds (mirror CPCVResult.path_sharpe_tstat)."""
    import numpy as np
    if len(arr) < 2:
        return 0.0
    sd = float(np.std(arr))
    if sd <= 1e-12:
        return 0.0
    return round(float(np.mean(arr)) / (sd / math.sqrt(max(n_folds, 1))), 4)


# ───────────────────────────────── Harness ─────────────────────────────────────────

def _make_scorer(vix_block_all: float = BASELINE_VIX_BLOCK, **control_kwargs):
    """Build the validated long-only PEAD scorer with a chosen VIX block / generic control."""
    from app.ml.pead_scorer import PEADScorer
    return PEADScorer(
        long_threshold=0.05,
        short_threshold=-0.05,
        long_short=False,
        vix_block_all=vix_block_all,
        vix_block_short=100.0,
        vix_conf_ref=100.0,
        max_announce_day_move=1.0,
        require_positive_revision=False,
        min_analyst_momentum=0.0,
        **control_kwargs,
    )


def run_analysis(
    smoke: bool = False,
    only: Optional[str] = None,
    symbols: Optional[List[str]] = None,
    total_years: Optional[int] = None,
    cpcv_k: Optional[int] = None,
    cpcv_paths: Optional[int] = None,
    purge_days: int = 10,
    embargo_days: int = 10,
    vix_thresholds: Optional[List[float]] = None,
    generic_controls: Optional[List[Dict]] = None,
) -> Dict:
    """Run the three §1.2 sub-analyses on a SINGLE shared data load.

    `only` in {None, 'threshold', 'loco', 'control'} runs all or one sub-analysis.
    """
    from app.utils.constants import RUSSELL_1000_TICKERS
    from scripts.run_pead_cpcv import PEADStrategy, CPCV_K, CPCV_PATHS, TOTAL_YEARS
    from scripts.walkforward.cpcv import run_cpcv

    _years = total_years if total_years is not None else (1 if smoke else TOTAL_YEARS)
    _k = cpcv_k if cpcv_k is not None else (4 if smoke else CPCV_K)
    _paths = cpcv_paths if cpcv_paths is not None else CPCV_PATHS
    _thresholds = vix_thresholds if vix_thresholds is not None else (
        SMOKE_VIX_THRESHOLDS if smoke else DEFAULT_VIX_THRESHOLDS)
    _controls = generic_controls if generic_controls is not None else (
        SMOKE_GENERIC_CONTROLS if smoke else GENERIC_CONTROLS)

    if symbols is None:
        symbols = list(RUSSELL_1000_TICKERS)[:8] if smoke else list(RUSSELL_1000_TICKERS)

    logger.info(
        "PEAD crisis-robustness: smoke=%s only=%s years=%d CPCV C(%d,%d) symbols=%d "
        "thresholds=%s controls=%s",
        smoke, only, _years, _k, _paths, len(symbols),
        [("inf" if math.isinf(t) else t) for t in _thresholds],
        [c["label"] for c in _controls],
    )

    # ── Fetch data ONCE; build strategy ONCE (mirror the cost sweep) ────────────────
    base = PEADStrategy(scorer=_make_scorer(), symbols=symbols, transaction_cost_pct=0.0005)
    t0 = time.time()
    try:
        from app.ml.retrain_config import retrain_as_of
        end_all = datetime.combine(retrain_as_of(), datetime.min.time())
    except Exception:
        end_all = datetime.now()
    start_all = end_all - timedelta(days=_years * 365 + 30)
    base.fetch_data(start_all, end_all)
    logger.info("Single shared data load complete in %.1fs", time.time() - t0)

    strategy = _EquityCapturingStrategy(base)

    def _run_cpcv_with(scorer) -> Tuple[object, Dict[int, list]]:
        """Swap in `scorer`, reset capture, run the full CPCV. Returns (result, {fold_id: equity_curve})."""
        base.scorer = scorer
        strategy.reset()
        result = run_cpcv(
            strategy=strategy, purge_days=purge_days, embargo_days=embargo_days,
            n_folds=_k, n_paths=_paths, total_years=_years,
        )
        return result, None

    out: Dict = {
        "smoke": smoke,
        "config": {"total_years": _years, "cpcv_k": _k, "cpcv_paths": _paths,
                   "purge_days": purge_days, "embargo_days": embargo_days,
                   "n_symbols": len(symbols)},
        "baseline_expected_sharpe": BASELINE_EXPECTED_SHARPE,
        "baseline_repro_tol": BASELINE_REPRO_TOL,
    }

    run_all = only is None

    # ════════════════════ A. THRESHOLD SWEEP ════════════════════════════════════════
    baseline_result = None
    if run_all or only == "threshold":
        rows: List[Dict] = []
        for thr in _thresholds:
            label = "inf (no block)" if math.isinf(thr) else f"VIX>{thr:.0f}"
            t_lvl = time.time()
            result, _ = _run_cpcv_with(_make_scorer(vix_block_all=thr))
            m = _path_metrics(list(result.path_sharpes))
            row = {
                "vix_block_all": ("inf" if math.isinf(thr) else round(thr, 2)),
                "label": label,
                "is_baseline": (not math.isinf(thr) and abs(thr - BASELINE_VIX_BLOCK) < 1e-9),
                "mean_sharpe": m["mean_sharpe"],
                "path_tstat": _tstat(m["_arr"], result.n_folds),
                "pct_positive": m["pct_positive"],
                "p5_sharpe": m["p5_sharpe"],
                "p95_sharpe": m["p95_sharpe"],
                "n_paths": m["n_paths"],
            }
            rows.append(row)
            if row["is_baseline"]:
                baseline_result = result
            logger.info("Threshold %s -> mean Sharpe %+.3f t=%+.2f %%pos=%.0f%% P5=%+.3f (%.1fs)",
                        label, row["mean_sharpe"], row["path_tstat"],
                        row["pct_positive"] * 100, row["p5_sharpe"], time.time() - t_lvl)
        # Self-validation against the committed +0.546 (baseline = VIX>30 row).
        base_row = next((r for r in rows if r["is_baseline"]), None)
        repro = (base_row is not None
                 and abs(base_row["mean_sharpe"] - BASELINE_EXPECTED_SHARPE) <= BASELINE_REPRO_TOL)
        if base_row is not None and not repro:
            logger.error(
                "BASELINE SELF-VALIDATION FAILED: VIX>30 mean Sharpe %+.3f does NOT reproduce "
                "validated %+.3f (|diff|=%.3f > tol %.3f). The harness DIVERGES from the "
                "committed config; threshold rows may be untrustworthy. INVESTIGATE.",
                base_row["mean_sharpe"], BASELINE_EXPECTED_SHARPE,
                abs(base_row["mean_sharpe"] - BASELINE_EXPECTED_SHARPE), BASELINE_REPRO_TOL,
            )
        elif base_row is not None:
            logger.info("BASELINE self-validation OK: VIX>30 %+.3f reproduces expected %+.3f (tol +/-%.3f)",
                        base_row["mean_sharpe"], BASELINE_EXPECTED_SHARPE, BASELINE_REPRO_TOL)
        out["threshold_sweep"] = {
            "rows": rows,
            "baseline_reproduces": repro,
            "interpretation": (
                "stable across 25-35 => exact 30 not load-bearing (robust); good ONLY at 30 "
                "=> overfit; inf(no block) row = raw edge with NO crisis protection"),
        }

    # ════════════════════ B. LEAVE-ONE-CRISIS-OUT (LOCO) ════════════════════════════
    if run_all or only == "loco":
        # Run the COMMITTED config once, capturing per-fold equity curves, then recompute
        # masked path Sharpes WITHOUT re-simulating. Reuse the threshold-sweep baseline
        # run if available; otherwise run it now (still capturing curves).
        loco_result, fold_curves = _loco_capture_run(
            base, strategy, _make_scorer(vix_block_all=BASELINE_VIX_BLOCK),
            purge_days, embargo_days, _k, _paths, _years,
        )
        # SINGLE SOURCE OF TRUTH: consume run_cpcv's OWN path grouping. The capture
        # run above ran the committed config through run_cpcv, which recorded exactly
        # which global fold ids each surviving path aggregated (post purge/overlap
        # guards) in result.path_fold_members. We group the captured per-fold equity
        # curves by THAT, so LOCO grouping is identical to the real CPCV by
        # construction — a reconstruction can never diverge from it.
        paths = [list(members) for members in loco_result.path_fold_members]
        in_window = [w for w in CRISIS_WINDOWS if _window_in_data(w, fold_curves)]
        rows: List[Dict] = []
        # No-removal baseline (mask=None) — MUST match the committed full run.
        baseline_row = _loco_row(
            "(none removed)", None, paths, fold_curves, loco_result.n_folds,
            note="baseline; recomputed from unmasked curves (self-check)")
        rows.append(baseline_row)
        # ── LOCO self-check (bug 2): the no-removal grouping recomputes path Sharpes
        # from the SAME curves with the SAME grouping run_cpcv used, so the mean MUST
        # equal run_cpcv's own mean Sharpe almost exactly. A divergence proves the
        # grouping/curve capture is broken — fail LOUDLY rather than report a verdict
        # off the wrong folds.
        _real_mean = float(loco_result.mean_sharpe)
        _loco_mean = float(baseline_row["mean_sharpe"])
        _self_check_ok = abs(_loco_mean - _real_mean) <= LOCO_SELF_CHECK_TOL
        if not _self_check_ok:
            logger.error(
                "LOCO SELF-CHECK FAILED: (none removed) mean Sharpe %+.6f does NOT "
                "reproduce run_cpcv's own mean Sharpe %+.6f (|diff|=%.6f > tol %.0e). "
                "The LOCO grouping/curve-capture DIVERGES from the real CPCV - every "
                "LOCO row is computed on the WRONG folds. INVESTIGATE; do NOT trust "
                "the verdict.",
                _loco_mean, _real_mean, abs(_loco_mean - _real_mean), LOCO_SELF_CHECK_TOL,
            )
        else:
            logger.info(
                "LOCO self-check OK: (none removed) mean Sharpe %+.6f reproduces "
                "run_cpcv mean Sharpe %+.6f (|diff|=%.2e <= tol %.0e)",
                _loco_mean, _real_mean, abs(_loco_mean - _real_mean), LOCO_SELF_CHECK_TOL,
            )
        baseline_row["self_check_ok"] = _self_check_ok
        baseline_row["real_cpcv_mean_sharpe"] = round(_real_mean, 6)
        for w in CRISIS_WINDOWS:
            inw = w in in_window
            rows.append(_loco_row(
                w["name"], (w["start"], w["end"]), paths, fold_curves, loco_result.n_folds,
                note=w["note"] + ("" if inw else "  [OUT OF DATA WINDOW — no effect]"),
                in_window=inw, start=w["start"], end=w["end"]))
        out["loco"] = {
            "rows": rows,
            "self_check_ok": _self_check_ok,
            "self_check_tol": LOCO_SELF_CHECK_TOL,
            "real_cpcv_mean_sharpe": round(_real_mean, 6),
            "crisis_windows": [
                {"name": w["name"], "start": w["start"].isoformat(), "end": w["end"].isoformat(),
                 "in_window": (w in in_window), "note": w["note"]}
                for w in CRISIS_WINDOWS],
            "interpretation": (
                "if removing any single episode (esp. bear_2022) collapses mean Sharpe toward 0 "
                "or flips %pos, the edge is episode-dependent (fragile); if it survives every "
                "single-episode removal it is broad-based"),
        }

    # ════════════════════ C. GENERIC REGIME CONTROL ════════════════════════════════
    if run_all or only == "control":
        rows: List[Dict] = []
        # VIX-block baseline (committed) for side-by-side comparison.
        if baseline_result is not None:
            bm = _path_metrics(list(baseline_result.path_sharpes))
            rows.append({
                "control": "VIX>30 block (baseline)", "mean_sharpe": bm["mean_sharpe"],
                "path_tstat": _tstat(bm["_arr"], baseline_result.n_folds),
                "pct_positive": bm["pct_positive"], "p5_sharpe": bm["p5_sharpe"],
                "p95_sharpe": bm["p95_sharpe"], "n_paths": bm["n_paths"], "is_baseline": True})
        else:
            result, _ = _run_cpcv_with(_make_scorer(vix_block_all=BASELINE_VIX_BLOCK))
            bm = _path_metrics(list(result.path_sharpes))
            rows.append({
                "control": "VIX>30 block (baseline)", "mean_sharpe": bm["mean_sharpe"],
                "path_tstat": _tstat(bm["_arr"], result.n_folds),
                "pct_positive": bm["pct_positive"], "p5_sharpe": bm["p5_sharpe"],
                "p95_sharpe": bm["p95_sharpe"], "n_paths": bm["n_paths"], "is_baseline": True})
        # Each generic control: VIX block OFF (inf), control governs.
        for ctrl in _controls:
            kw = {k: v for k, v in ctrl.items() if k != "label"}
            t_lvl = time.time()
            result, _ = _run_cpcv_with(_make_scorer(vix_block_all=math.inf, **kw))
            m = _path_metrics(list(result.path_sharpes))
            rows.append({
                "control": ctrl["label"], "mean_sharpe": m["mean_sharpe"],
                "path_tstat": _tstat(m["_arr"], result.n_folds),
                "pct_positive": m["pct_positive"], "p5_sharpe": m["p5_sharpe"],
                "p95_sharpe": m["p95_sharpe"], "n_paths": m["n_paths"], "is_baseline": False})
            logger.info("Control %s (VIX block OFF) -> mean Sharpe %+.3f t=%+.2f %%pos=%.0f%% P5=%+.3f (%.1fs)",
                        ctrl["label"], m["mean_sharpe"], rows[-1]["path_tstat"],
                        m["pct_positive"] * 100, m["p5_sharpe"], time.time() - t_lvl)
        out["generic_control"] = {
            "rows": rows,
            "interpretation": (
                "control = a vol/trend-gated ENTRY BLOCK (scalar<floor -> block all new entries) "
                "plus a coarse saturating [0.75-1.25x] conviction-size tilt; in the equal-weight "
                "+0.546 book it is NOT continuous gross vol-targeting (gross per day is fixed). "
                "If PEAD survives under this generic entry block (not fit to crises), VIX>30 was "
                "just one instance of 'de-risk in stress' -> robust, can replace the block; if it "
                "ONLY survives under the exact VIX>30 block -> overfit -> PAUSE"),
        }

    # Strip internal numpy arrays before returning (not JSON-serializable, internal-only).
    _strip_internal(out)
    return out


def _loco_capture_run(base, strategy, scorer, purge_days, embargo_days, _k, _paths, _years):
    """Run the committed CPCV once while capturing per-fold equity curves.

    We monkey-wrap the inner run_fold to stash each fold's equity_curve keyed by the
    GLOBAL fold id cpcv assigns (combo_idx*n_boundaries+ti+1), so LOCO can recompute
    masked path Sharpes. Returns (CPCVResult, {fold_id: equity_curve}).
    """
    from scripts.walkforward.cpcv import run_cpcv
    base.scorer = scorer
    strategy.reset()
    captured: Dict[int, list] = {}
    inner_run_fold = base.run_fold

    def _wrapped(fold_idx, n_folds, tr_start, tr_end, te_start, te_end):
        fold = inner_run_fold(fold_idx, n_folds, tr_start, tr_end, te_start, te_end)
        # Re-run is NOT needed — the SimResult equity_curve is what we need, but
        # FoldResult does not carry it. We instead re-derive from the sim by reading
        # the equity curve the strategy stashed. PEADStrategy.run_fold builds a sim
        # internally; we capture via a side channel set on `base` (see below).
        ec = getattr(base, "_last_equity_curve", None)
        if ec is not None:
            captured[fold_idx] = ec
        return fold

    base.run_fold = _wrapped  # type: ignore[assignment]
    try:
        result = run_cpcv(
            strategy=strategy, purge_days=purge_days, embargo_days=embargo_days,
            n_folds=_k, n_paths=_paths, total_years=_years,
        )
    finally:
        base.run_fold = inner_run_fold  # type: ignore[assignment]
    return result, captured


def _window_in_data(window: Dict, fold_curves: Dict[int, list]) -> bool:
    """True if the crisis window overlaps ANY captured fold equity-curve date range."""
    lo, hi = window["start"], window["end"]
    for ec in fold_curves.values():
        if not ec:
            continue
        dates = [d.date() if hasattr(d, "date") else d for d, _ in ec]
        if dates and not (hi < dates[0] or lo > dates[-1]):
            return True
    return False


def _loco_row(label, mask_range, paths, fold_curves, n_folds, note="",
              in_window=True, start=None, end=None) -> Dict:
    """Recompute path Sharpes from captured fold curves with `mask_range` removed."""
    path_sharpes: List[float] = []
    for members in paths:
        fold_sharpes = []
        for fid in members:
            ec = fold_curves.get(fid)
            if not ec:
                continue
            s, _ = _sharpe_from_equity(ec, mask_range)
            fold_sharpes.append(s)
        if fold_sharpes:
            import numpy as np
            path_sharpes.append(float(np.mean(fold_sharpes)))
    m = _path_metrics(path_sharpes)
    return {
        "removed": label,
        "start": start.isoformat() if start else None,
        "end": end.isoformat() if end else None,
        "in_window": in_window,
        "mean_sharpe": m["mean_sharpe"],
        "path_tstat": _tstat(m["_arr"], n_folds),
        "pct_positive": m["pct_positive"],
        "p5_sharpe": m["p5_sharpe"],
        "p95_sharpe": m["p95_sharpe"],
        "n_paths": m["n_paths"],
        "note": note,
    }


def _strip_internal(obj):
    if isinstance(obj, dict):
        obj.pop("_arr", None)
        for v in obj.values():
            _strip_internal(v)
    elif isinstance(obj, list):
        for v in obj:
            _strip_internal(v)


# ───────────────────────────────── Reporting ───────────────────────────────────────

def _print_report(out: Dict) -> None:
    smoke = "   [SMOKE]" if out.get("smoke") else ""
    print()
    print("=" * 100)
    print("PEAD CRISIS-BLOCK ROBUSTNESS  (Alpha v2 1.2)" + smoke)
    print("=" * 100)

    if "threshold_sweep" in out:
        ts = out["threshold_sweep"]
        print("\nA. THRESHOLD SWEEP  (vix_block_all)")
        print("-" * 100)
        print(f"{'threshold':>16} {'meanSharpe':>11} {'t-stat':>8} {'%pos':>7} {'P5':>9} {'P95':>9} {'paths':>6}")
        for r in ts["rows"]:
            mark = "  <- BASELINE (committed +0.546)" if r["is_baseline"] else ""
            print(f"{r['label']:>16} {r['mean_sharpe']:>+11.3f} {r['path_tstat']:>+8.2f} "
                  f"{r['pct_positive']*100:>6.0f}% {r['p5_sharpe']:>+9.3f} {r['p95_sharpe']:>+9.3f} "
                  f"{r['n_paths']:>6d}{mark}")
        print(f"  baseline self-validation: {'REPRODUCES' if ts['baseline_reproduces'] else 'DIVERGES - INVESTIGATE'}")
        print(f"  {ts['interpretation']}")

    if "loco" in out:
        lc = out["loco"]
        print("\nB. LEAVE-ONE-CRISIS-OUT  (mask crisis date range from equity, recompute Sharpe)")
        print("-" * 100)
        print(f"{'removed':>20} {'window':>23} {'inWin':>6} {'meanSharpe':>11} {'t-stat':>8} {'%pos':>7} {'P5':>9} {'paths':>6}")
        for r in lc["rows"]:
            win = f"{r['start']}..{r['end']}" if r.get("start") else "-"
            inw = "yes" if r.get("in_window") else "no"
            print(f"{r['removed']:>20} {win:>23} {inw:>6} {r['mean_sharpe']:>+11.3f} "
                  f"{r['path_tstat']:>+8.2f} {r['pct_positive']*100:>6.0f}% {r['p5_sharpe']:>+9.3f} {r['n_paths']:>6d}")
        _sc = lc.get("self_check_ok")
        _rm = lc.get("real_cpcv_mean_sharpe")
        print(f"  LOCO self-check: {'OK' if _sc else 'DIVERGES - INVESTIGATE'} "
              f"((none removed) reproduces run_cpcv mean Sharpe {_rm:+.4f} "
              f"within tol {lc.get('self_check_tol')})")
        print(f"  {lc['interpretation']}")

    if "generic_control" in out:
        gc = out["generic_control"]
        print("\nC. GENERIC REGIME CONTROL  (replaces VIX>30; VIX block OFF for control rows)")
        print("-" * 100)
        print(f"{'control':>26} {'meanSharpe':>11} {'t-stat':>8} {'%pos':>7} {'P5':>9} {'P95':>9} {'paths':>6}")
        for r in gc["rows"]:
            mark = "  <- baseline" if r.get("is_baseline") else ""
            print(f"{r['control']:>26} {r['mean_sharpe']:>+11.3f} {r['path_tstat']:>+8.2f} "
                  f"{r['pct_positive']*100:>6.0f}% {r['p5_sharpe']:>+9.3f} {r['p95_sharpe']:>+9.3f} "
                  f"{r['n_paths']:>6d}{mark}")
        print(f"  {gc['interpretation']}")
    print("=" * 100)
    print()


def _write_artifacts(out: Dict, stamp: str) -> Dict[str, Path]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = ARTIFACT_DIR / f"pead_crisis_robustness_{stamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    paths = {"json": json_path}
    # One flat CSV per sub-analysis present.
    for key, fields in (
        ("threshold_sweep", ["label", "vix_block_all", "is_baseline", "mean_sharpe",
                             "path_tstat", "pct_positive", "p5_sharpe", "p95_sharpe", "n_paths"]),
        ("loco", ["removed", "start", "end", "in_window", "mean_sharpe", "path_tstat",
                  "pct_positive", "p5_sharpe", "p95_sharpe", "n_paths", "note"]),
        ("generic_control", ["control", "is_baseline", "mean_sharpe", "path_tstat",
                             "pct_positive", "p5_sharpe", "p95_sharpe", "n_paths"]),
    ):
        if key in out:
            csv_path = ARTIFACT_DIR / f"pead_crisis_{key}_{stamp}.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                w.writeheader()
                for r in out[key]["rows"]:
                    w.writerow(r)
            paths[key] = csv_path
    logger.info("Wrote artifacts: %s", ", ".join(str(p) for p in paths.values()))
    return paths


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="PEAD crisis-block robustness harness (Alpha v2 1.2)")
    p.add_argument("--smoke", action="store_true", help="fast smoke: tiny universe + short window + tiny CPCV")
    p.add_argument("--only", choices=["threshold", "loco", "control"], default=None,
                   help="run only one sub-analysis (default: all three)")
    args = p.parse_args(argv)

    try:
        if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S") + ("_smoke" if args.smoke else "")
    log_path = LOG_DIR / f"pead_crisis_robustness_{stamp}.log"
    _setup_logging(log_path)

    out = run_analysis(smoke=args.smoke, only=args.only)

    # Artifacts FIRST (durable), then a guarded table print (console-encoding safe).
    paths = _write_artifacts(out, stamp)
    out["artifacts"] = {k: str(v) for k, v in paths.items()}
    try:
        _print_report(out)
    except Exception as exc:  # pragma: no cover - defensive console guard
        logger.warning("Report print failed (%s: %s); artifacts already written to %s",
                       type(exc).__name__, exc, paths)

    logger.info("PEAD crisis-robustness complete. Log: %s", log_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
