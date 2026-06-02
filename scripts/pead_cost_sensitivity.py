"""
ALPHA v2 §1.1 — PEAD cost-sensitivity sweep.

Question: how much of the committed PEAD long-only edge (CPCV mean Sharpe +0.546,
t=2.26, 95% paths positive at 5 bps) survives REALISTIC earnings-window transaction
costs? Earnings-window spreads are wider than the optimistic 5 bps, so we sweep the
per-side cost and report where the edge degrades.

────────────────────────────────────────────────────────────────────────────────
DESIGN (see also docstring of run_pead_cpcv.build_pead_scorer)
────────────────────────────────────────────────────────────────────────────────
1. SWEPT QUANTITY — `cost_bps` is the TOTAL one-way cost charged on EACH fill side
   (entry, and every exit). It is driven entirely through `transaction_cost_pct`
   (the AgentSimulator charges it per-side at entry AND exit → round-trip = 2×).
   During the sweep, entry/stop SLIPPAGE is set to ZERO so the table number is
   exactly interpretable as "X bps per side" with no double-counting. (The default
   PEAD config's 3 bps entry / 5 bps stop slippage is FOLDED INTO the swept cost:
   at e.g. 5 bps the sweep charges 5 bps per side flat, which is comparable to the
   committed run's 5 bps fee + ~3 bps entry-slip on the entry leg — the sweep is a
   slightly cleaner, monotone lever.) Commissions are ~0 on the paper broker, so
   spread/slippage IS the relevant friction and that is exactly what `cost_bps`
   models.

2. COMPARABILITY — every cost level runs on BYTE-IDENTICAL data, folds, signals and
   PEAD config. We fetch data ONCE, build the strategy ONCE, then re-run CPCV at
   each cost level mutating ONLY (transaction_cost_pct, entry_slippage_pct,
   stop_slippage_pct). No re-fetch, no re-derive of folds (folds are anchored to
   retrain_as_of() inside run_cpcv → deterministic across levels).

3. The validated PEAD scorer config (long-only, vix_block_all=30, hold=40,
   priced-in off, long_threshold=0.05) is reused verbatim via build_pead_scorer().
   The ONLY thing varied across the table is cost.

OUTPUT — a results table {2,5,10,20,35,50} bps → mean Sharpe, path t-stat
(N_eff=n_folds), %pos, P5, avg net return/fold; plus the implied BREAK-EVEN cost
(mean Sharpe crosses 0) and the cost where mean crosses +0.40 (the §1.1 acceptance
threshold). Printed, logged under logs/, and written as CSV + JSON artifacts.

4. ANCHOR ROW — in addition to the pure-cost ladder, a clearly-labelled ANCHOR row
   runs the EXACT committed/validated config on the SAME data + folds: 5 bps fee +
   3 bps entry-slip + 5 bps stop-slip (slippage ADDITIVE). It exists to (a) self-
   validate that this harness reproduces the validated ~+0.546 mean Sharpe — a
   mismatch beyond a tight tolerance is logged LOUDLY as a harness divergence — and
   (b) sit beside the pure-cost ladder for honest comparison. Because it carries
   slippage it is NOT on the pure per-side cost axis, so it is flagged
   (slippage_included=True, cost_bps=None) and EXCLUDED from the break-even / +0.40
   interpolation, which is computed over the pure-cost ladder only.

Usage:
    # Full sweep (re-fetches ~1000 symbols / 6yr, runs 6× CPCV — LONG, ~hours):
    python scripts/pead_cost_sensitivity.py

    # Fast smoke (2 cost levels, tiny universe + short window, tiny CPCV):
    python scripts/pead_cost_sensitivity.py --smoke

    # Custom levels:
    python scripts/pead_cost_sensitivity.py --cost-bps 5 20 50
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv()

from app.ml.retrain_config import MAX_THREADS as _max_threads

os.environ.setdefault("OMP_NUM_THREADS", str(_max_threads))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(_max_threads))

logger = logging.getLogger("pead_cost_sweep")

# Default cost ladder (bps, one-way per side). §1.1 acceptance is at 20 bps.
DEFAULT_COST_BPS = [2.0, 5.0, 10.0, 20.0, 35.0, 50.0]
SMOKE_COST_BPS = [5.0, 50.0]

# §1.1 acceptance: net mean Sharpe must stay ≥ this at 20 bps one-way.
ACCEPTANCE_SHARPE = 0.40
ACCEPTANCE_COST_BPS = 20.0

# The EXACT committed/validated PEAD config (CPCV mean Sharpe +0.546). Its effective
# friction is NOT a pure per-side cost: it is a fee + ADDITIVE slippage triplet, so it
# does NOT lie on the pure-cost ladder axis. We run it as a clearly-labelled ANCHOR row
# (slippage_included=True) to self-validate that this harness reproduces ~+0.546, and we
# EXCLUDE it from the break-even / +0.40 interpolation (which is pure-cost-only).
ANCHOR_LABEL = "committed (+0.546 anchor)"
ANCHOR_TX_COST_PCT = 0.0005       # 5 bps fee per side
ANCHOR_ENTRY_SLIPPAGE_PCT = 0.0003  # 3 bps entry slippage (additive)
ANCHOR_STOP_SLIPPAGE_PCT = 0.0005   # 5 bps stop slippage (additive)
# Expected mean Sharpe of the validated run; the anchor should reproduce this closely.
ANCHOR_EXPECTED_SHARPE = 0.546
ANCHOR_REPRO_TOL = 0.05  # |anchor - expected| above this → harness divergence, surface LOUDLY

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
        format="%(asctime)s %(levelname)s [pead_cost_sweep] %(message)s",
        handlers=handlers,
        force=True,
    )


class _ReturnCapturingStrategy:
    """Wraps a PEADStrategy to record per-fold net total_return for the sweep table.

    The cost knobs live on the wrapped strategy (transaction_cost_pct,
    entry/stop_slippage_pct). The wrapper proxies everything to it and only
    intercepts run_fold to accumulate fold.total_return. We reset the accumulator
    before each cost level via reset_returns().
    """

    def __init__(self, inner):
        self._inner = inner
        self.fold_returns: List[float] = []

    def reset_returns(self) -> None:
        self.fold_returns = []

    def run_fold(self, *args, **kwargs):
        fold = self._inner.run_fold(*args, **kwargs)
        try:
            self.fold_returns.append(float(fold.total_return))
        except Exception:
            pass
        return fold

    def __getattr__(self, name):
        # Proxy all other attribute access (symbols_data, model, model_type,
        # allow_in_sample, transaction_cost_pct, slippage, etc.) to the inner
        # strategy so run_cpcv sees an unmodified PEADStrategy interface.
        return getattr(self._inner, name)


def _interp_crossing(levels: List[float], values: List[float], target: float) -> Optional[float]:
    """Linear-interpolated cost (bps) where `values` first crosses `target` going down.

    levels and values are paired and assumed sorted by ascending cost. Returns the
    interpolated cost where value == target between the last point ABOVE target and
    the first point at/below it. None if it never crosses within the swept range.
    """
    for i in range(1, len(levels)):
        v_hi, v_lo = values[i - 1], values[i]
        if v_hi >= target > v_lo:
            # crossing between levels[i-1] and levels[i]
            x_hi, x_lo = levels[i - 1], levels[i]
            if v_hi == v_lo:
                return x_lo
            frac = (v_hi - target) / (v_hi - v_lo)
            return x_hi + frac * (x_lo - x_hi)
    # If even the lowest cost is already at/below target, no useful crossing.
    return None


def run_sweep(
    cost_bps_levels: List[float],
    smoke: bool = False,
    symbols: Optional[List[str]] = None,
    total_years: Optional[int] = None,
    cpcv_k: Optional[int] = None,
    cpcv_paths: Optional[int] = None,
    purge_days: int = 10,
    embargo_days: int = 10,
) -> Dict:
    """Run the PEAD CPCV at each cost level on a SINGLE shared data load.

    Returns a dict with the per-level rows and the derived break-even / +0.40 costs.
    """
    from app.utils.constants import RUSSELL_1000_TICKERS
    from scripts.run_pead_cpcv import (
        PEADStrategy,
        build_pead_scorer,
        CPCV_K,
        CPCV_PATHS,
        TOTAL_YEARS,
    )
    from scripts.walkforward.cpcv import run_cpcv

    _years = total_years if total_years is not None else (1 if smoke else TOTAL_YEARS)
    _k = cpcv_k if cpcv_k is not None else (4 if smoke else CPCV_K)
    _paths = cpcv_paths if cpcv_paths is not None else CPCV_PATHS

    if symbols is None:
        if smoke:
            # Tiny universe for a fast end-to-end smoke (data fetch dominates runtime).
            symbols = list(RUSSELL_1000_TICKERS)[:8]
        else:
            symbols = list(RUSSELL_1000_TICKERS)

    logger.info(
        "PEAD cost-sensitivity sweep: levels=%s bps  smoke=%s  years=%d  CPCV C(%d,%d)  "
        "symbols=%d",
        cost_bps_levels, smoke, _years, _k, _paths, len(symbols),
    )

    # ── Build scorer + strategy ONCE; fetch data ONCE ───────────────────────────
    scorer = build_pead_scorer()
    base = PEADStrategy(scorer=scorer, symbols=symbols, transaction_cost_pct=0.0005)

    t0 = time.time()
    # Determinism: anchor the data window to retrain_as_of() (same as run_cpcv's
    # fold anchoring) so folds and data are identical across cost levels and reruns.
    try:
        from app.ml.retrain_config import retrain_as_of
        end_all = datetime.combine(retrain_as_of(), datetime.min.time())
    except Exception:
        end_all = datetime.now()
    start_all = end_all - timedelta(days=_years * 365 + 30)
    base.fetch_data(start_all, end_all)
    logger.info("Single shared data load complete in %.1fs", time.time() - t0)

    strategy = _ReturnCapturingStrategy(base)

    def _simulate(tx_cost_pct: float, entry_slip: float, stop_slip: float) -> Dict:
        """Re-run CPCV on the SHARED data/folds at one (cost, slippage) triplet.

        Only mutates the three friction knobs on `base`; everything else (data,
        folds, signals, config) is identical across calls. Returns the metric dict
        WITHOUT the cost-axis/label fields (the caller adds those)."""
        base.transaction_cost_pct = tx_cost_pct
        base.entry_slippage_pct = entry_slip
        base.stop_slippage_pct = stop_slip
        strategy.reset_returns()
        result = run_cpcv(
            strategy=strategy,
            purge_days=purge_days,
            embargo_days=embargo_days,
            n_folds=_k,
            n_paths=_paths,
            total_years=_years,
        )
        avg_net_ret = (
            float(sum(strategy.fold_returns) / len(strategy.fold_returns))
            if strategy.fold_returns else 0.0
        )
        return {
            "mean_sharpe": round(result.mean_sharpe, 4),
            "path_tstat": round(result.path_sharpe_tstat, 4),
            "pct_positive": round(result.pct_positive, 4),
            "p5_sharpe": round(result.p5_sharpe, 4),
            "p95_sharpe": round(result.p95_sharpe, 4),
            "avg_net_return_per_fold": round(avg_net_ret, 6),
            "n_paths": result.n_combinations,
            "n_folds_eff": result.n_folds,
        }

    rows: List[Dict] = []
    for cost_bps in cost_bps_levels:
        cost_frac = cost_bps / 1e4  # bps → fraction
        # Drive the FULL per-side cost through transaction_cost_pct; zero out
        # slippage so the table number is exactly "cost_bps per side".
        logger.info("---- Cost level %.1f bps (frac=%.6f) ----", cost_bps, cost_frac)
        t_lvl = time.time()
        metrics = _simulate(cost_frac, 0.0, 0.0)
        row = {
            "cost_bps": round(cost_bps, 4),
            "label": f"{cost_bps:.1f} bps (pure cost)",
            "slippage_included": False,
            **metrics,
        }
        rows.append(row)
        logger.info(
            "Cost %.1f bps -> mean Sharpe %+.3f  t=%+.2f  %%pos=%.0f%%  P5=%+.3f  "
            "avg_net_ret/fold=%+.2f%%  (%.1fs)",
            cost_bps, metrics["mean_sharpe"], metrics["path_tstat"],
            metrics["pct_positive"] * 100, metrics["p5_sharpe"],
            metrics["avg_net_return_per_fold"] * 100, time.time() - t_lvl,
        )

    # ── ANCHOR ROW — the EXACT committed/validated config (fee + additive slippage) ──
    # Same loaded data + same folds; only the friction triplet differs. NOT on the
    # pure-cost axis (it carries slippage), so it is appended SEPARATELY and is
    # EXCLUDED from the break-even / +0.40 interpolation below.
    logger.info(
        "---- ANCHOR: committed config  tx=%.4f entry_slip=%.4f stop_slip=%.4f ----",
        ANCHOR_TX_COST_PCT, ANCHOR_ENTRY_SLIPPAGE_PCT, ANCHOR_STOP_SLIPPAGE_PCT,
    )
    t_anchor = time.time()
    anchor_metrics = _simulate(
        ANCHOR_TX_COST_PCT, ANCHOR_ENTRY_SLIPPAGE_PCT, ANCHOR_STOP_SLIPPAGE_PCT,
    )
    anchor_row = {
        "cost_bps": None,  # NOT a pure-cost level — has no single per-side cost number
        "label": ANCHOR_LABEL,
        "slippage_included": True,
        "tx_cost_pct": ANCHOR_TX_COST_PCT,
        "entry_slippage_pct": ANCHOR_ENTRY_SLIPPAGE_PCT,
        "stop_slippage_pct": ANCHOR_STOP_SLIPPAGE_PCT,
        **anchor_metrics,
    }
    anchor_reproduces = abs(anchor_metrics["mean_sharpe"] - ANCHOR_EXPECTED_SHARPE) <= ANCHOR_REPRO_TOL
    logger.info(
        "ANCHOR (committed +0.546) -> mean Sharpe %+.3f  t=%+.2f  %%pos=%.0f%%  (%.1fs)",
        anchor_metrics["mean_sharpe"], anchor_metrics["path_tstat"],
        anchor_metrics["pct_positive"] * 100, time.time() - t_anchor,
    )
    if anchor_reproduces:
        logger.info(
            "ANCHOR self-validation OK: %+.3f reproduces expected %+.3f (tol +/-%.3f)",
            anchor_metrics["mean_sharpe"], ANCHOR_EXPECTED_SHARPE, ANCHOR_REPRO_TOL,
        )
    else:
        logger.error(
            "ANCHOR SELF-VALIDATION FAILED: harness mean Sharpe %+.3f does NOT reproduce "
            "validated %+.3f (|diff|=%.3f > tol %.3f) - the sweep harness DIVERGES from the "
            "committed config; the pure-cost ladder may be untrustworthy. INVESTIGATE.",
            anchor_metrics["mean_sharpe"], ANCHOR_EXPECTED_SHARPE,
            abs(anchor_metrics["mean_sharpe"] - ANCHOR_EXPECTED_SHARPE), ANCHOR_REPRO_TOL,
        )

    # Break-even / +0.40 interpolation is computed over the PURE-COST ladder ONLY.
    # The anchor carries slippage, so it is not on the same cost axis — exclude it.
    pure_rows = [r for r in rows if not r["slippage_included"]]
    levels = [r["cost_bps"] for r in pure_rows]
    sharpes = [r["mean_sharpe"] for r in pure_rows]
    breakeven_bps = _interp_crossing(levels, sharpes, 0.0)
    cross_040_bps = _interp_crossing(levels, sharpes, ACCEPTANCE_SHARPE)

    # Append the anchor AFTER interpolation so it cannot contaminate it.
    rows.append(anchor_row)

    # §1.1 acceptance check at 20 bps (interpolate if 20 not in the ladder).
    sharpe_at_acc = None
    if ACCEPTANCE_COST_BPS in levels:
        sharpe_at_acc = sharpes[levels.index(ACCEPTANCE_COST_BPS)]
    elif len(levels) >= 2 and levels[0] <= ACCEPTANCE_COST_BPS <= levels[-1]:
        for i in range(1, len(levels)):
            if levels[i - 1] <= ACCEPTANCE_COST_BPS <= levels[i]:
                x0, x1 = levels[i - 1], levels[i]
                y0, y1 = sharpes[i - 1], sharpes[i]
                frac = (ACCEPTANCE_COST_BPS - x0) / (x1 - x0) if x1 != x0 else 0.0
                sharpe_at_acc = y0 + frac * (y1 - y0)
                break

    summary = {
        "rows": rows,
        "anchor": {
            "label": ANCHOR_LABEL,
            "tx_cost_pct": ANCHOR_TX_COST_PCT,
            "entry_slippage_pct": ANCHOR_ENTRY_SLIPPAGE_PCT,
            "stop_slippage_pct": ANCHOR_STOP_SLIPPAGE_PCT,
            "mean_sharpe": anchor_metrics["mean_sharpe"],
            "expected_sharpe": ANCHOR_EXPECTED_SHARPE,
            "repro_tol": ANCHOR_REPRO_TOL,
            "reproduces_validated": anchor_reproduces,
            "excluded_from_interpolation": True,
        },
        "breakeven_cost_bps": (round(breakeven_bps, 2) if breakeven_bps is not None else None),
        "cross_0.40_cost_bps": (round(cross_040_bps, 2) if cross_040_bps is not None else None),
        "acceptance_cost_bps": ACCEPTANCE_COST_BPS,
        "acceptance_sharpe_threshold": ACCEPTANCE_SHARPE,
        "mean_sharpe_at_acceptance": (round(sharpe_at_acc, 4) if sharpe_at_acc is not None else None),
        "acceptance_passed": (sharpe_at_acc is not None and sharpe_at_acc >= ACCEPTANCE_SHARPE),
        "smoke": smoke,
        "config": {
            "total_years": _years,
            "cpcv_k": _k,
            "cpcv_paths": _paths,
            "purge_days": purge_days,
            "embargo_days": embargo_days,
            "n_symbols": len(symbols),
        },
    }
    return summary


def _print_table(summary: Dict) -> None:
    rows = summary["rows"]
    print()
    print("=" * 92)
    print("PEAD COST-SENSITIVITY SWEEP  (Alpha v2 1.1)" + ("   [SMOKE]" if summary["smoke"] else ""))
    print("=" * 92)
    hdr = (f"{'cost(bps)':>10} {'meanSharpe':>11} {'t-stat':>8} {'%pos':>7} "
           f"{'P5':>8} {'P95':>8} {'avgNetRet/fold':>15} {'paths':>6}")
    print(hdr)
    print("-" * 92)
    for r in rows:
        cost_lbl = f"{r['cost_bps']:>10.1f}" if r.get("cost_bps") is not None else f"{'ANCHOR':>10}"
        print(f"{cost_lbl} {r['mean_sharpe']:>+11.3f} {r['path_tstat']:>+8.2f} "
              f"{r['pct_positive'] * 100:>6.0f}% {r['p5_sharpe']:>+8.3f} {r['p95_sharpe']:>+8.3f} "
              f"{r['avg_net_return_per_fold'] * 100:>+14.2f}% {r['n_paths']:>6d}"
              + ("  <- " + r["label"] if r.get("slippage_included") else ""))
    print("-" * 92)
    anc = summary.get("anchor")
    if anc is not None:
        repro = "REPRODUCES" if anc["reproduces_validated"] else "DOES NOT REPRODUCE - HARNESS DIVERGES"
        print(f"  ANCHOR ({anc['label']}): fee={anc['tx_cost_pct']*1e4:.0f}bps "
              f"+ entry-slip={anc['entry_slippage_pct']*1e4:.0f}bps + stop-slip={anc['stop_slippage_pct']*1e4:.0f}bps "
              f"(slippage additive, NOT on pure-cost axis)")
        print(f"    mean Sharpe {anc['mean_sharpe']:+.3f}  vs validated {anc['expected_sharpe']:+.3f} "
              f"(tol +/-{anc['repro_tol']:.3f}) -> {repro}")
        print(f"    [excluded from break-even / +0.40 interpolation]")
    print("-" * 92)
    be = summary["breakeven_cost_bps"]
    c40 = summary["cross_0.40_cost_bps"]
    print(f"  Break-even cost (mean Sharpe -> 0):     "
          f"{be if be is not None else 'beyond swept range'} bps")
    print(f"  Cost where mean Sharpe -> +0.40:        "
          f"{c40 if c40 is not None else 'beyond swept range'} bps")
    msa = summary["mean_sharpe_at_acceptance"]
    print(f"  Mean Sharpe at {summary['acceptance_cost_bps']:.0f} bps (acceptance): "
          f"{msa if msa is not None else 'n/a'}  "
          f"(gate >= {summary['acceptance_sharpe_threshold']}) -> "
          f"{'PASS' if summary['acceptance_passed'] else 'FAIL/REVIEW'}")
    print("=" * 92)
    print()


def _write_artifacts(summary: Dict, stamp: str) -> Dict[str, Path]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = ARTIFACT_DIR / f"pead_cost_sweep_{stamp}.csv"
    json_path = ARTIFACT_DIR / f"pead_cost_sweep_{stamp}.json"

    fieldnames = [
        "cost_bps", "label", "slippage_included",
        "tx_cost_pct", "entry_slippage_pct", "stop_slippage_pct",
        "mean_sharpe", "path_tstat", "pct_positive",
        "p5_sharpe", "p95_sharpe", "avg_net_return_per_fold", "n_paths", "n_folds_eff",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        # extrasaction='ignore' is unnecessary (all keys are in fieldnames) but the
        # pure-cost rows simply leave the anchor-only triplet columns blank.
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in summary["rows"]:
            w.writerow(r)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Wrote artifacts: %s , %s", csv_path, json_path)
    return {"csv": csv_path, "json": json_path}


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="PEAD cost-sensitivity sweep (Alpha v2 §1.1)")
    p.add_argument("--smoke", action="store_true",
                   help="fast end-to-end smoke: 2 cost levels, tiny universe + short window")
    p.add_argument("--cost-bps", type=float, nargs="+", default=None,
                   help="explicit one-way cost levels in bps (overrides defaults)")
    args = p.parse_args(argv)

    # Best-effort: render Unicode cleanly in terminals that support it. The script's
    # strings are already ASCII-safe (the durable artifacts never depend on this), so
    # a failure here is harmless.
    try:
        if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    if args.cost_bps is not None:
        levels = sorted(args.cost_bps)
    elif args.smoke:
        levels = SMOKE_COST_BPS
    else:
        levels = DEFAULT_COST_BPS

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S") + ("_smoke" if args.smoke else "")
    log_path = LOG_DIR / f"pead_cost_sweep_{stamp}.log"
    _setup_logging(log_path)

    summary = run_sweep(levels, smoke=args.smoke)

    # Persist the durable artifacts FIRST. They are the real output of the sweep and
    # must never be lost to a console-encoding error during table rendering. We then
    # print the table guarded by try/except so a failed print can't abort the run
    # after a ~1hr sweep with the results already computed.
    paths = _write_artifacts(summary, stamp)
    summary["artifacts"] = {k: str(v) for k, v in paths.items()}

    try:
        _print_table(summary)
    except Exception as exc:  # pragma: no cover - defensive console guard
        logger.warning(
            "Table print failed (%s: %s); artifacts already written to %s",
            type(exc).__name__, exc, paths,
        )

    logger.info("PEAD cost-sensitivity sweep complete. Log: %s", log_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
