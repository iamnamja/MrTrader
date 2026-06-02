"""
Small/Mid-Cap PEAD CPCV harness (survivorship-safe).

Clones scripts/run_pead_cpcv.py — inheriting the verified honest-pipeline
machinery (per-fold n_obs + regime_sharpes + profit_factor, OOS guard
trained_through=date.min, CPCV C(8,2)/k=8/6yr, FoldResult construction) — with
the following changes for the small/mid-cap event-drift experiment:

  - Universe: survivorship-safe PIT small/mid-cap ADV band (scripts/
    build_smallmid_universe.py), NOT RUSSELL_1000_TICKERS.
  - Prices: Polygon delisted-inclusive grouped-daily (cached panel), NOT yfinance.
  - run_fold universe: per-fold PIT eligibility set (band-eligible as-of the test
    window) replaces the hardcoded pit_union("russell1000").
  - delisted_haircut=0.70: a name held through its delisting books a -70% loss
    (conservative-realistic for small-cap delistings), NOT P&L=0.
  - transaction_cost_pct=0.0020 (20bps): small-cap realistic vs 5bps large-cap.
  - Per-fold min-history guard: skip names with < MIN_HISTORY_BARS before
    te_start so conviction/vol/features aren't computed on a stub series.

The PEAD scorer config is the validated long-only baseline (long_threshold=0.05,
vix_block_all=30, max_announce_day_move=1.0 disabled, long_short=False).

Build only. Run with:
    python scripts/build_smallmid_universe.py --years 6      # ONCE, builds cache
    python scripts/run_pead_smallmid_cpcv.py                  # uses the cache
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
    format="%(asctime)s %(levelname)s [pead_smallmid_cpcv] %(message)s",
)
logger = logging.getLogger(__name__)

CPCV_K = 8       # number of fold groups
CPCV_PATHS = 2   # test groups per combination → C(8,2)=28 paths
TOTAL_YEARS = 6

# Small/mid-cap realism levers (named constants; see docstring).
TRANSACTION_COST_PCT = 0.0020   # 20bps — small-cap realistic (vs 5bps large-cap)
DELISTED_HAIRCUT = 0.70         # held-through-delisting books -70% (vs P&L=0)
# min trailing bars before te_start to include a name (ADV window 20 + vol/conviction buffer)
MIN_HISTORY_BARS = 60


class SmallMidPEADStrategy:
    """
    PEAD strategy adapter for the survivorship-safe small/mid-cap universe.

    Mirrors PEADStrategy in run_pead_cpcv.py but sources prices from Polygon
    grouped-daily (delisted-inclusive) and uses a PIT ADV-band universe instead
    of the hardcoded Russell-1000 pit_union.
    """

    model_type = "pead"

    def __init__(self, scorer, panel: pd.DataFrame, eligibility: pd.DataFrame,
                 transaction_cost_pct: float = TRANSACTION_COST_PCT):
        self.scorer = scorer
        self.panel = panel
        self.eligibility = eligibility
        self.transaction_cost_pct = transaction_cost_pct
        self.symbols_data: Dict[str, pd.DataFrame] = {}
        self.spy_prices = None
        self.all_days_sorted = []
        self.data_source = "polygon_grouped_daily_smallmid"
        # OOS guard: rules-based strategies have no ML training cutoff.
        from datetime import date as _date
        self.model = type("_NoModel", (), {"trained_through": _date.min})()
        self.allow_in_sample = False

    def fetch_data(self, start: datetime, end: datetime) -> None:
        """
        Build symbols_data from the cached survivorship-safe panel (NOT yfinance).
        SPY + VIX are still fetched from Polygon/yfinance for benchmark + regime gate.
        """
        t0 = time.time()
        if self.panel is None or self.panel.empty:
            raise RuntimeError(
                "Small/mid-cap panel is empty. Run "
                "`python scripts/build_smallmid_universe.py --years 6` first to build the cache."
            )

        panel = self.panel.copy()
        panel["date"] = pd.to_datetime(panel["date"])
        # Pivot the long panel into per-symbol OHLCV-ish frames. Grouped-daily gives
        # close + volume; PEAD/AgentSimulator use close for entries/exits and
        # close*volume for ADV. We synthesize open/high/low = close (the sim's stop
        # logic is disabled for PEAD via no_atr_stops paths; entries use close).
        for sym, grp in panel.groupby("symbol"):
            g = grp.sort_values("date").set_index("date")
            df = pd.DataFrame({
                "open": g["close"].astype(float),
                "high": g["close"].astype(float),
                "low": g["close"].astype(float),
                "close": g["close"].astype(float),
                "volume": g["volume"].astype(float),
            }, index=g.index)
            df = df[~df.index.duplicated(keep="last")].sort_index()
            self.symbols_data[sym] = df

        # SPY benchmark + VIX regime gate (small auxiliary fetch; not the universe).
        import yfinance as yf
        spy_raw = yf.download("SPY", start=start.date().isoformat(),
                              end=end.date().isoformat(), progress=False, auto_adjust=True)
        if isinstance(spy_raw.columns, pd.MultiIndex):
            spy_raw.columns = spy_raw.columns.get_level_values(0)
        spy_raw.columns = [c.lower() for c in spy_raw.columns]
        self.spy_prices = spy_raw["close"]
        self.symbols_data["SPY"] = spy_raw
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

        # Pre-compute global regime map (stable VIX quartiles across folds).
        try:
            from scripts.walkforward.regime import load_regime_map as _lrm
            self._global_regime_map = _lrm(
                start.date() if hasattr(start, "date") else start,
                end.date() if hasattr(end, "date") else end,
            )
        except Exception:
            self._global_regime_map = {}

        logger.info("Data loaded: %d symbols (incl. delisted), %d days in %.1fs",
                    len(self.symbols_data), len(all_days), time.time() - t0)

    def run_fold(self, fold_idx, n_folds, tr_start, tr_end, te_start, te_end):
        from app.backtesting.agent_simulator import AgentSimulator
        from scripts.walkforward.gates import (
            FoldResult, compute_profit_factor, compute_calmar, compute_k_ratio, fold_years,
        )
        from scripts.build_smallmid_universe import symbols_eligible_in_window

        _te_start = te_start.date() if hasattr(te_start, "date") else te_start
        _te_end = te_end.date() if hasattr(te_end, "date") else te_end

        # PIT small/mid-cap universe: a name is in the fold universe if it was
        # ADV-band eligible on any day in the TEST window. Replaces the hardcoded
        # pit_union("russell1000"). Survivorship-safe (eligibility includes
        # delisted names up to delisting) and PIT (trailing-20d ADV).
        pit_members = symbols_eligible_in_window(self.eligibility, _te_start, _te_end)
        _synthetic = {"^VIX", "VIX", "SPY"}

        # Per-fold min-history guard (M-1): require >= MIN_HISTORY_BARS bars strictly
        # before te_start so conviction/vol/features aren't computed on a stub series.
        _te_start_ts = pd.Timestamp(_te_start)
        fold_symbols_data = {}
        _skipped_short = 0
        for s, d in self.symbols_data.items():
            if s in _synthetic:
                fold_symbols_data[s] = d
                continue
            if s not in pit_members:
                continue
            prior = d.loc[d.index < _te_start_ts]
            if len(prior) < MIN_HISTORY_BARS:
                _skipped_short += 1
                continue
            fold_symbols_data[s] = d

        logger.info(
            "Fold %d universe: %d band-eligible names, %d in-data (>= %d bars), %d skipped (short history)",
            fold_idx, len(pit_members), len(fold_symbols_data) - len(_synthetic & set(fold_symbols_data)),
            MIN_HISTORY_BARS, _skipped_short,
        )

        _hold_override = (int(os.environ["PEAD_MAX_HOLD_BARS"])
                          if os.environ.get("PEAD_MAX_HOLD_BARS") else None)
        _conviction = os.environ.get("PEAD_CONVICTION_SIZE") == "1"
        sim = AgentSimulator(
            model=None,
            factor_scorer=self.scorer,
            transaction_cost_pct=self.transaction_cost_pct,
            no_prefilters=True,
            max_hold_bars_override=_hold_override,
            pead_conviction_size=_conviction,
            delisted_haircut=DELISTED_HAIRCUT,  # C-2 fix: held-through-delist books haircut, not 0
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
        # n_obs = trading-day return observations for DSR (equity-curve diffs → len-1).
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
    from app.ml.pead_scorer import PEADScorer
    from scripts.walkforward.cpcv import run_cpcv
    from scripts.build_smallmid_universe import load_cached

    cached = load_cached()
    if cached is None:
        logger.error(
            "No cached small/mid-cap universe found. Build it first:\n"
            "    python scripts/build_smallmid_universe.py --years %d", TOTAL_YEARS,
        )
        return 1
    panel, elig = cached
    logger.info("Loaded cached universe: %d panel rows, %d eligible (symbol,day) rows, %d distinct symbols",
                len(panel), len(elig), elig["symbol"].nunique() if not elig.empty else 0)

    # Validated long-only baseline (same scorer config as the +0.546 large-cap run).
    _ls = os.environ.get("PEAD_LONG_SHORT") == "1"
    _quality_gate = os.environ.get("PEAD_QUALITY_GATE") == "1"
    scorer = PEADScorer(
        long_threshold=0.05,
        short_threshold=-0.05,
        long_short=_ls,
        vix_block_all=30.0,
        vix_block_short=(20.0 if _ls else 100.0),
        vix_conf_ref=100.0,
        max_announce_day_move=1.0,   # disabled — large gaps retain drift signal
        require_positive_revision=_quality_gate,
        min_analyst_momentum=0.0,
    )

    strategy = SmallMidPEADStrategy(
        scorer=scorer,
        panel=panel,
        eligibility=elig,
        transaction_cost_pct=TRANSACTION_COST_PCT,
    )

    end_all = datetime.now()
    start_all = end_all - timedelta(days=TOTAL_YEARS * 365 + 30)
    strategy.fetch_data(start_all, end_all)

    logger.info("Running small/mid-cap PEAD CPCV: k=%d paths=%d → C(%d,%d)=%d paths "
                "(cost=%.2f%%, delisted_haircut=%.0f%%)",
                CPCV_K, CPCV_PATHS, CPCV_K, CPCV_PATHS,
                len(list(__import__("itertools").combinations(range(CPCV_K), CPCV_PATHS))),
                TRANSACTION_COST_PCT * 100, DELISTED_HAIRCUT * 100)

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
        "Small/mid-cap PEAD CPCV %s — mean_sharpe=%.3f  p5=%.3f  p95=%.3f",
        verdict, result.mean_sharpe, result.p5_sharpe, result.p95_sharpe,
    )

    try:
        from app.notifications.notifier import _smtp_send
        _smtp_send(
            subject=f"MrTrader Small/Mid PEAD CPCV: {verdict} (mean={result.mean_sharpe:.3f})",
            html_body=f"""
<h2>Small/Mid-Cap PEAD CPCV Result</h2>
<p><b>{verdict}</b></p>
<ul>
  <li>Mean Sharpe: {result.mean_sharpe:.3f} (gate: >=0.80)</li>
  <li>P5 Sharpe: {result.p5_sharpe:.3f} (gate: >=-0.30)</li>
  <li>P95 Sharpe: {result.p95_sharpe:.3f}</li>
  <li>Cost: {TRANSACTION_COST_PCT*100:.2f}%  Delisted haircut: {DELISTED_HAIRCUT*100:.0f}%</li>
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
