"""
Insider-Buying-Cluster CPCV — honest-pipeline validation of a candidate SECOND edge.

Near-verbatim clone of scripts/run_pead_cpcv.py (the proven PEAD CPCV harness),
swapping only the scorer: PEADScorer -> InsiderClusterScorer. Inherits all the
honest-pipeline correctness (per-fold n_obs for DSR, regime_sharpes, real
profit_factor, PIT universe via te_start, OOS guard trained_through=date.min,
5bps costs, C(8,2) k=8 ~6yr).

Usage:
    python scripts/run_insider_cpcv.py [--hypothesis-id HYP-ID | --exploratory]
"""
import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

from app.ml.retrain_config import MAX_THREADS as _max_threads
os.environ.setdefault("OMP_NUM_THREADS", str(_max_threads))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(_max_threads))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [insider_cpcv] %(message)s",
)
logger = logging.getLogger(__name__)

CPCV_K = 8       # number of fold groups
CPCV_PATHS = 2   # test groups per combination → C(8,2)=28 paths
TOTAL_YEARS = 6


class InsiderStrategy:
    """
    Thin strategy adapter that runs AgentSimulator with InsiderClusterScorer for
    each CPCV fold. Implements the interface expected by
    scripts/walkforward/cpcv.run_cpcv. Mirrors PEADStrategy exactly.
    """

    model_type = "insider"

    def __init__(self, scorer, symbols, transaction_cost_pct=0.0005):
        self.scorer = scorer
        self.symbols = list(symbols)
        self.transaction_cost_pct = transaction_cost_pct
        self.symbols_data: Dict[str, pd.DataFrame] = {}
        self.spy_prices = None
        self.all_days_sorted = []
        # OOS guard: rules-based strategies have no ML training cutoff.
        # Use date.min so every test fold is trivially after the "training" date.
        from datetime import date as _date
        self.model = type("_NoModel", (), {"trained_through": _date.min})()
        self.allow_in_sample = False

    def fetch_data(self, start: datetime, end: datetime) -> None:
        import yfinance as yf
        from app.utils.constants import RUSSELL_1000_TICKERS
        t0 = time.time()
        syms = self.symbols or list(RUSSELL_1000_TICKERS)
        logger.info("Downloading daily bars %s → %s for %d symbols", start.date(), end.date(), len(syms))
        for sym in syms:
            try:
                df = yf.download(sym, start=start.date().isoformat(),
                                 end=end.date().isoformat(), progress=False, auto_adjust=True)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.columns = [c.lower() for c in df.columns]
                if len(df) >= 210:
                    self.symbols_data[sym] = df
            except Exception:
                pass

        spy_raw = yf.download("SPY", start=start.date().isoformat(),
                              end=end.date().isoformat(), progress=False, auto_adjust=True)
        if isinstance(spy_raw.columns, pd.MultiIndex):
            spy_raw.columns = spy_raw.columns.get_level_values(0)
        spy_raw.columns = [c.lower() for c in spy_raw.columns]
        self.spy_prices = spy_raw["close"]
        self.symbols_data["SPY"] = spy_raw

        # Download VIX for regime gating in the scorer
        try:
            vix_raw = yf.download("^VIX", start=start.date().isoformat(),
                                  end=end.date().isoformat(), progress=False, auto_adjust=True)
            if isinstance(vix_raw.columns, pd.MultiIndex):
                vix_raw.columns = vix_raw.columns.get_level_values(0)
            vix_raw.columns = [c.lower() for c in vix_raw.columns]
            if len(vix_raw) > 0:
                self.symbols_data["^VIX"] = vix_raw
                logger.info("VIX data loaded: %d days", len(vix_raw))
        except Exception as e:
            logger.warning("VIX download failed (regime gate disabled): %s", e)

        # Pre-warm insider caches for all symbols (one paginated fetch each) so the
        # CPCV inner loop slices in memory — no per-day API spam.
        try:
            from app.data.fmp_provider import get_insider_trades_fmp
            loaded = [s for s in self.symbols_data if s not in ("SPY", "^VIX", "VIX")]
            logger.info("Pre-fetching insider Form-4 purchases for %d symbols...", len(loaded))
            n_with = 0
            for s in loaded:
                recs = get_insider_trades_fmp(s)
                if recs:
                    n_with += 1
            logger.info("Insider prefetch complete: %d/%d symbols have purchase history",
                        n_with, len(loaded))
        except Exception as e:
            logger.warning("Insider prefetch failed: %s", e)

        all_days = sorted({
            d.date() if hasattr(d, "date") else d
            for df in self.symbols_data.values()
            for d in df.index
        })
        self.all_days_sorted = all_days

        # Pre-compute global regime map over the full evaluation window so VIX
        # quartile thresholds are stable across folds (mirrors swing.py:174-179).
        try:
            from scripts.walkforward.regime import load_regime_map as _lrm
            self._global_regime_map = _lrm(
                start.date() if hasattr(start, "date") else start,
                end.date() if hasattr(end, "date") else end,
            )
        except Exception:
            self._global_regime_map = {}

        logger.info("Data loaded: %d symbols, %d days in %.1fs",
                    len(self.symbols_data), len(all_days), time.time() - t0)

    def run_fold(self, fold_idx, n_folds, tr_start, tr_end, te_start, te_end):
        from app.backtesting.agent_simulator import AgentSimulator
        from app.data.universe_history import pit_union as _pit_union, historical_trade_symbols as _hist_syms
        from scripts.walkforward.gates import FoldResult, compute_profit_factor, compute_calmar, compute_k_ratio, fold_years

        # Use te_start (not te_end) for PIT universe — avoids leaking symbols that
        # joined the index mid-test-period into the universe at test start.
        extra = _hist_syms(tr_start, te_start, trade_type="swing")
        pit_members = set(_pit_union("russell1000", tr_start, te_start, extra_symbols=extra))
        _synthetic = {"^VIX", "VIX", "SPY"}
        fold_symbols_data = {
            s: d for s, d in self.symbols_data.items()
            if s in pit_members or s in _synthetic
        }

        sim = AgentSimulator(
            model=None,
            factor_scorer=self.scorer,
            transaction_cost_pct=self.transaction_cost_pct,
            no_prefilters=True,
        )
        result = sim.run(
            fold_symbols_data,
            start_date=te_start,
            end_date=te_end,
            spy_prices=self.spy_prices,
        )

        from scripts.walkforward.regime import compute_regime_sharpes as _crs

        stop_exits = result.exit_breakdown.get("STOP", 0)
        n_trades = int(result.total_trades)
        stop_rate = float(stop_exits) / max(n_trades, 1)
        trades_list = getattr(result, "trades", None) or []
        trade_returns = [t.pnl_pct for t in trades_list if hasattr(t, "pnl_pct")]
        equity_curve = getattr(result, "equity_curve", [])
        n_obs = max(len(equity_curve) - 1, 0)
        regime_sharpes = _crs(equity_curve, te_start, te_end,
                              regime_map=getattr(self, "_global_regime_map", None))
        years = fold_years(te_start, te_end)
        sharpe = float(result.sharpe_ratio)
        total_ret = float(result.total_return_pct)
        max_dd = float(result.max_drawdown_pct)
        win_rate = float(result.win_rate)

        logger.info(
            "Fold %d done — %d trades, Sharpe %.3f, return %.1f%%",
            fold_idx, n_trades, sharpe, total_ret * 100,
        )
        return FoldResult(
            fold=fold_idx,
            train_start=tr_start, train_end=tr_end,
            test_start=te_start, test_end=te_end,
            trades=n_trades,
            win_rate=win_rate,
            sharpe=sharpe,
            max_drawdown=max_dd,
            total_return=total_ret,
            stop_exit_rate=stop_rate,
            model_version=0,
            profit_factor=getattr(result, "profit_factor", compute_profit_factor(trade_returns)),
            calmar_ratio=compute_calmar(total_ret, max_dd, years),
            k_ratio=compute_k_ratio(equity_curve),
            n_obs=n_obs,
            regime_sharpes=regime_sharpes,
        )


def main() -> int:
    from app.ml.insider_scorer import InsiderClusterScorer
    from app.utils.constants import RUSSELL_1000_TICKERS
    from scripts.walkforward.cpcv import run_cpcv
    from scripts.walkforward.registry_enforcement import add_arguments, begin_run

    ap = argparse.ArgumentParser(description="Insider-cluster CPCV (k=8, paths=2)")
    add_arguments(ap)  # --hypothesis-id / --exploratory (all-optional)
    args = ap.parse_args()
    # Registry enforcement FAILS FAST — before the multi-hour fetch/run.
    run = begin_run(args.hypothesis_id, exploratory=args.exploratory)

    scorer = InsiderClusterScorer()
    strategy = InsiderStrategy(
        scorer=scorer,
        symbols=list(RUSSELL_1000_TICKERS),
        transaction_cost_pct=0.0005,
    )

    end_all = datetime.now()
    start_all = end_all - timedelta(days=TOTAL_YEARS * 365 + 30)
    strategy.fetch_data(start_all, end_all)

    logger.info("Running Insider CPCV: k=%d paths=%d → C(%d,%d)=%d test paths",
                CPCV_K, CPCV_PATHS, CPCV_K, CPCV_PATHS,
                len(list(__import__("itertools").combinations(range(CPCV_K), CPCV_PATHS))))

    result = run_cpcv(
        strategy=strategy,
        purge_days=10,
        embargo_days=10,
        n_folds=CPCV_K,
        n_paths=CPCV_PATHS,
        total_years=TOTAL_YEARS,
    )

    result.print()

    gate_detail = result.gate_detail()
    gate_ok = all(v for _, v in gate_detail.values())
    verdict = "CPCV GATE PASSED" if gate_ok else "CPCV GATE FAILED"
    logger.info(
        "Insider CPCV %s — mean_sharpe=%.3f  p5=%.3f  p95=%.3f",
        verdict, result.mean_sharpe, result.p5_sharpe, result.p95_sharpe,
    )

    # Best-effort registry recording (never crashes the run). decision=None:
    # promotion is owner-gated, never auto-decided from one run.
    if run is not None:
        run.record({
            "mean_sharpe": result.mean_sharpe,
            "p5_sharpe": result.p5_sharpe,
            "p95_sharpe": result.p95_sharpe,
            "tstat": result.path_sharpe_tstat,
            "n_paths": len(result.path_sharpes),
            "gate_detail": gate_detail,
            "gate_ok": gate_ok,
        }, decision=None)

    try:
        from app.notifications.notifier import _smtp_send
        _smtp_send(
            subject=f"MrTrader Insider CPCV: {verdict} (mean={result.mean_sharpe:.3f})",
            html_body=f"""
<h2>Insider-Cluster CPCV Result</h2>
<p><b>{verdict}</b></p>
<ul>
  <li>Mean Sharpe: {result.mean_sharpe:.3f}</li>
  <li>P5 Sharpe: {result.p5_sharpe:.3f}</li>
  <li>P95 Sharpe: {result.p95_sharpe:.3f}</li>
  <li>Gate detail: {gate_detail}</li>
  <li>N paths: {len(result.path_sharpes)}</li>
</ul>
""",
        )
    except Exception as _e:
        logger.warning("Email notification failed: %s", _e)

    return 0 if gate_ok else 2


if __name__ == "__main__":
    sys.exit(main())
