"""
Phase G: CPCV validation of the PEAD scorer.

Combinatorial Purged Cross-Validation (k=6, paths=2 → C(6,2)=15 test paths).
Higher statistical power than standard 5-fold WF; catches overfitting to specific
fold boundaries.

Usage:
    python scripts/run_pead_cpcv.py
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
    format="%(asctime)s %(levelname)s [pead_cpcv] %(message)s",
)
logger = logging.getLogger(__name__)

CPCV_K = 8       # number of fold groups
CPCV_PATHS = 2   # test groups per combination → C(6,2)=15 paths
TOTAL_YEARS = 6


class PEADStrategy:
    """
    Thin strategy adapter that runs AgentSimulator with PEADScorer for each CPCV fold.
    Implements the interface expected by scripts/walkforward/cpcv.run_cpcv.
    """

    model_type = "pead"

    def __init__(self, scorer, symbols, transaction_cost_pct=0.0005,
                 entry_slippage_pct=None, stop_slippage_pct=None):
        self.scorer = scorer
        self.symbols = list(symbols)
        self.transaction_cost_pct = transaction_cost_pct
        # §1.1 cost-sensitivity: optional slippage overrides threaded into each
        # per-fold AgentSimulator. None → AgentSimulator uses its module-constant
        # defaults (byte-identical to the committed +0.546 run).
        self.entry_slippage_pct = entry_slippage_pct
        self.stop_slippage_pct = stop_slippage_pct
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
        logger.info("Downloading daily bars %s -> %s for %d symbols", start.date(), end.date(), len(syms))
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

        # Download VIX for regime gating in PEADScorer
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

        # Hold-period lever: PEAD positions exit via max_hold_bars (default 40 ≈ 8wk,
        # which over-stays and gives back drift). PEAD_MAX_HOLD_BARS env var caps it
        # (e.g. 15 ≈ 3wk, matching the academic drift half-life). Unset → default 40.
        _hold_override = (int(os.environ["PEAD_MAX_HOLD_BARS"])
                          if os.environ.get("PEAD_MAX_HOLD_BARS") else None)
        # PEAD_CONVICTION_SIZE=1 → size new long entries by clip(SUE_z,0,3)/realized_vol,
        # gross-normalized to the equal-weight book (same names, same per-day gross,
        # only the per-name weight changes). Default OFF → committed equal-weight
        # +0.546 config is byte-identical. PIT-safe SUE (expanding cross-section) +
        # realized vol (bars strictly before entry day).
        _conviction = os.environ.get("PEAD_CONVICTION_SIZE") == "1"
        # §1.1: only pass slippage kwargs when explicitly overridden, so the
        # default PEAD run uses AgentSimulator's module-constant defaults and is
        # byte-identical to the committed +0.546 config.
        _slip_kwargs = {}
        if self.entry_slippage_pct is not None:
            _slip_kwargs["entry_slippage_pct"] = self.entry_slippage_pct
        if self.stop_slippage_pct is not None:
            _slip_kwargs["stop_slippage_pct"] = self.stop_slippage_pct
        sim = AgentSimulator(
            model=None,
            factor_scorer=self.scorer,
            transaction_cost_pct=self.transaction_cost_pct,
            no_prefilters=True,
            max_hold_bars_override=_hold_override,
            pead_conviction_size=_conviction,
            **_slip_kwargs,
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
        # §1.2 LOCO side-channel: stash this fold's per-day equity curve so the
        # crisis-robustness harness can recompute masked path Sharpes WITHOUT
        # re-simulating. Harmless to the default CPCV run (just an extra attribute).
        self._last_equity_curve = equity_curve
        # §1.3 significance side-channel (PURE-ADDITIVE): stash this fold's realized
        # trade objects so the event-clustered significance harness can capture the
        # per-TRADE return stream tagged with each trade's earnings-proximate entry
        # date + symbol, WITHOUT re-simulating. Byte-identical to the committed run
        # (just an extra attribute; never read by run_cpcv or the gate). Mirrors the
        # §1.2 _last_equity_curve discipline.
        self._last_trades = list(trades_list)
        # n_obs = trading-day return observations for DSR. equity_curve is one
        # (date, equity) point per trading day; diffs give daily returns → len-1.
        # Mirrors swing.py:320-323. Without this DSR falls back to ~path-count.
        n_obs = max(len(equity_curve) - 1, 0)
        _regime_obs: dict = {}
        regime_sharpes = _crs(equity_curve, te_start, te_end,
                              regime_map=getattr(self, "_global_regime_map", None),
                              obs_counts=_regime_obs)
        years = fold_years(te_start, te_end)
        sharpe = float(result.sharpe_ratio)
        total_ret = float(result.total_return_pct)
        max_dd = float(result.max_drawdown_pct)
        win_rate = float(result.win_rate)

        logger.info(
            "Fold %d done - %d trades, Sharpe %.3f, return %.1f%%",
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
            regime_obs_counts=_regime_obs,
        )


def build_pead_scorer():
    """Construct the validated PEAD scorer (the committed +0.546 long-only config).

    Extracted so the §1.1 cost-sensitivity sweep reuses the EXACT same scorer
    config — the only thing the sweep varies is transaction cost / slippage.
    Honors the same PEAD_LONG_SHORT / PEAD_QUALITY_GATE env switches as main().
    """
    from app.ml.pead_scorer import PEADScorer

    # Best config found: long-only, no VIX gate, no priced-in filter.
    # CPCV campaign (4 runs): Run 3 (this config) achieved mean=0.349 — the best result.
    # Priced-in filter (Run 4) hurt: large announce-day gaps have strongest drift continuation.
    # Default = best long-only config (+0.546 honest CPCV). PEAD_LONG_SHORT=1 enables
    # the short leg (long positive-surprise + short negative-surprise) with a VIX>20
    # squeeze-guard on shorts — the untried dollar-neutral test.
    _ls = os.environ.get("PEAD_LONG_SHORT") == "1"
    # Earnings-quality split (last high-EV lever): PEAD_QUALITY_GATE=1 requires a
    # long signal to be "EPS beat + analysts revising up" (positive analyst-revision
    # momentum as-of the scoring day) rather than a bare beat. Higher-conviction
    # drift, fewer trades. PIT-safe (analyst feature windowed to <= scoring day).
    # Default OFF → committed long-only +0.546 config is unchanged.
    _quality_gate = os.environ.get("PEAD_QUALITY_GATE") == "1"
    scorer = PEADScorer(
        long_threshold=0.05,
        short_threshold=-0.05,
        long_short=_ls,
        vix_block_all=30.0,
        vix_block_short=(20.0 if _ls else 100.0),  # squeeze guard when short leg active
        vix_conf_ref=100.0,
        max_announce_day_move=1.0,  # disabled — large gaps retain drift signal
        require_positive_revision=_quality_gate,
        min_analyst_momentum=0.0,
    )
    if _quality_gate:
        logger.info("PEAD_QUALITY_GATE=1 - long signals require positive analyst revision")
    if os.environ.get("PEAD_CONVICTION_SIZE") == "1":
        logger.info(
            "PEAD_CONVICTION_SIZE=1 - long entries weighted by clip(SUE_z,0,3)/realized_vol, "
            "gross-normalized to the equal-weight book (same entry set, no leverage confound)"
        )
    return scorer


def main() -> int:
    from app.utils.constants import RUSSELL_1000_TICKERS
    from scripts.walkforward.cpcv import run_cpcv

    scorer = build_pead_scorer()
    strategy = PEADStrategy(
        scorer=scorer,
        symbols=list(RUSSELL_1000_TICKERS),
        transaction_cost_pct=0.0005,
    )

    end_all = datetime.now()
    start_all = end_all - timedelta(days=TOTAL_YEARS * 365 + 30)
    strategy.fetch_data(start_all, end_all)

    logger.info("Running PEAD CPCV: k=%d paths=%d -> C(%d,%d)=%d test paths",
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
        "PEAD CPCV %s - mean_sharpe=%.3f  p5=%.3f  p95=%.3f",
        verdict, result.mean_sharpe, result.p5_sharpe, result.p95_sharpe,
    )

    try:
        from app.notifications.notifier import _smtp_send
        _smtp_send(
            subject=f"MrTrader PEAD CPCV: {verdict} (mean={result.mean_sharpe:.3f})",
            html_body=f"""
<h2>PEAD CPCV Result</h2>
<p><b>{verdict}</b></p>
<ul>
  <li>Mean Sharpe: {result.mean_sharpe:.3f} (gate: ≥0.80)</li>
  <li>P5 Sharpe: {result.p5_sharpe:.3f} (gate: ≥-0.30)</li>
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
