"""
QualityShort CPCV — honest-pipeline validation of the QualityShortScorer short leg.

This is a near-verbatim clone of scripts/run_pead_cpcv.py (the PEAD CPCV harness),
swapping only the scorer: QualityShortScorer(legs_mode="shorts_only").

Why clone PEAD exactly:
  The PEAD harness encodes every honest-pipeline correctness fix the project has
  shipped — per-fold n_obs (real DSR denominator), per-fold regime_sharpes (real
  worst-regime gate), result.profit_factor (real PF), PIT universe via te_start,
  the rules-based OOS guard (trained_through=date.min), 5bps costs, and the
  C(8,2)=28-path / k=8 / ~6yr fold geometry. Reusing the identical adapter means
  this short-strategy result inherits all of that — no empty-matrix / DSR-n_obs
  regressions.

Shorts execution (the #1 risk for a short strategy):
  QualityShortScorer(legs_mode="shorts_only") emits ONLY (sym, -conf, "short")
  tuples. In AgentSimulator's factor_scorer path (factor_scorer set, rebalance_mode
  off, enable_shorts off — exactly as PEAD long_short=True uses), _process_entries
  parses the 3-tuple, sets is_short=True for direction=="short", and opens a real
  short: receives proceeds + posts collateral, applies a daily borrow cost, uses
  short stop/target semantics (stop ABOVE entry, target BELOW), exits via
  max_hold_bars, and books P&L = (entry - exit) * qty. This is the same path the
  PEAD short leg (PEAD_LONG_SHORT=1) exercises — verified in agent_simulator.py
  (_process_entries ~L1098-1280, _process_exits ~L1819-1940, _close_position
  ~L1950-1966). No enable_shorts / rebalance_mode needed; those are for the
  separate rebalance path.

PIT-safety:
  QualityShortScorer reads fundamentals from the FMP PIT parquet
  (data/fundamentals/fmp_fundamentals_history.parquet), filtering
  as_of_date <= scoring-day (as_of_date == filingDate, the date the 10-Q/K became
  public, ~45d after period end) plus a 120-day staleness reject. The optional
  earnings-surprise flag uses get_earnings_features_at(sym, as_of), which filters
  earnings records to date <= as_of. Composite scores use bars strictly before the
  scoring day (_build_closes_and_bars masks index < day). No look-ahead.

Usage:
    python scripts/run_qualityshort_cpcv.py
"""
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
    format="%(asctime)s %(levelname)s [qshort_cpcv] %(message)s",
)
logger = logging.getLogger(__name__)

CPCV_K = 8       # number of fold groups (matches PEAD harness)
CPCV_PATHS = 2   # test groups per combination → C(8,2)=28 paths
TOTAL_YEARS = 6


class QualityShortStrategy:
    """
    Thin strategy adapter that runs AgentSimulator with QualityShortScorer for each
    CPCV fold. Implements the interface expected by scripts/walkforward/cpcv.run_cpcv.

    Identical to scripts/run_pead_cpcv.py:PEADStrategy except for the model_type
    label. The scorer is injected by main().
    """

    model_type = "qualityshort"

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

        # Download VIX for regime gating in QualityShortScorer
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

        all_days = sorted({
            d.date() if hasattr(d, "date") else d
            for df in self.symbols_data.values()
            for d in df.index
        })
        self.all_days_sorted = all_days

        # Pre-compute global regime map over the full evaluation window so VIX
        # quartile thresholds are stable across folds (mirrors swing.py:174-179).
        # Required for compute_regime_sharpes / worst-regime-sharpe gate.
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

        # Hold-period lever (shared with PEAD): QShort positions exit via
        # max_hold_bars (default 40 ≈ 8wk). QSHORT_MAX_HOLD_BARS env var caps it.
        _hold_override = (int(os.environ["QSHORT_MAX_HOLD_BARS"])
                          if os.environ.get("QSHORT_MAX_HOLD_BARS") else None)
        sim = AgentSimulator(
            model=None,
            factor_scorer=self.scorer,
            transaction_cost_pct=self.transaction_cost_pct,
            no_prefilters=True,
            max_hold_bars_override=_hold_override,
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
        # PF: use AgentSimulator's result.profit_factor directly (computed with
        # _PF_NO_LOSS_SENTINEL); SimResult has no trade_returns attr. Fall back to
        # per-trade pnl_pct extraction. Mirrors swing.py:317-343.
        trades_list = getattr(result, "trades", None) or []
        trade_returns = [t.pnl_pct for t in trades_list if hasattr(t, "pnl_pct")]
        equity_curve = getattr(result, "equity_curve", [])
        # n_obs = trading-day return observations for DSR. equity_curve is one
        # (date, equity) point per trading day; diffs give daily returns → len-1.
        # Mirrors swing.py:320-323. Without this DSR falls back to ~path-count.
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
    from app.ml.short_scorers import QualityShortScorer
    from app.utils.constants import RUSSELL_1000_TICKERS
    from scripts.walkforward.cpcv import run_cpcv

    # Shorts-only QualityShort: short fundamentally deteriorating names (operating
    # margin <= 0, revenue_growth <= 0, debt/equity >= 1.5, negative surprise),
    # >= flags_required of those flags. NO long leg — this isolates the short edge
    # so the CPCV number is the short signal's honest Sharpe (not a blended L/S).
    scorer = QualityShortScorer(
        top_n=20,
        max_shorts=15,
        vix_threshold=30.0,
        flags_required=2,
        legs_mode="shorts_only",
    )
    logger.info("QualityShortScorer(legs_mode='shorts_only', flags_required=2, max_shorts=15)")

    strategy = QualityShortStrategy(
        scorer=scorer,
        symbols=list(RUSSELL_1000_TICKERS),
        transaction_cost_pct=0.0005,
    )

    end_all = datetime.now()
    start_all = end_all - timedelta(days=TOTAL_YEARS * 365 + 30)
    strategy.fetch_data(start_all, end_all)

    logger.info("Running QualityShort CPCV: k=%d paths=%d → C(%d,%d)=%d test paths",
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
        "QualityShort CPCV %s — mean_sharpe=%.3f  p5=%.3f  p95=%.3f  pct_pos=%.1f%%  tstat=%.2f  n_paths=%d  n_skipped=%d",
        verdict, result.mean_sharpe, result.p5_sharpe, result.p95_sharpe,
        result.pct_positive * 100, result.path_sharpe_tstat,
        len(result.path_sharpes), result.n_skipped,
    )

    try:
        from app.notifications.notifier import _smtp_send
        _smtp_send(
            subject=f"MrTrader QualityShort CPCV: {verdict} (mean={result.mean_sharpe:.3f})",
            html_body=f"""
<h2>QualityShort CPCV Result (shorts-only)</h2>
<p><b>{verdict}</b></p>
<ul>
  <li>Mean Sharpe: {result.mean_sharpe:.3f} (gate: ≥0.80)</li>
  <li>P5 Sharpe: {result.p5_sharpe:.3f} (gate: ≥-0.30)</li>
  <li>P95 Sharpe: {result.p95_sharpe:.3f}</li>
  <li>% positive: {result.pct_positive:.1%}</li>
  <li>Path t-stat: {result.path_sharpe_tstat:.2f}</li>
  <li>Gate detail: {gate_detail}</li>
  <li>N paths: {len(result.path_sharpes)} / skipped {result.n_skipped}</li>
</ul>
""",
        )
    except Exception as _e:
        logger.warning("Email notification failed: %s", _e)

    return 0 if gate_ok else 2


if __name__ == "__main__":
    sys.exit(main())
