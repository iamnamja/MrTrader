"""
Reusable event-edge CPCV adapter (Alpha-v3 A0).

Generalizes the proven PEAD harness (scripts/run_pead_cpcv.py) so any
discrete-event -> measurable-drift edge (analyst-revision drift, short-squeeze,
guidance, ...) can be validated through the SAME trusted path: per-fold
AgentSimulator + scripts.walkforward.cpcv.run_cpcv + the gate + the
event-clustered significance / crisis-robustness side-channels.

A *scorer* is any callable implementing the AgentSimulator factor-scorer contract:

    scorer(day, symbols_data, vix_history=None) -> list[(symbol, confidence, direction)]

EventEdgeStrategy is rules-based (no ML training): trained_through = date.min and
allow_in_sample = False, so every CPCV test fold is trivially out-of-sample.

To add an edge: write a scorer, then either instantiate EventEdgeStrategy(scorer,
symbols, model_type="...") or subclass it and override `_fold_sim_kwargs` for
edge-specific AgentSimulator knobs. `PEADStrategy` (run_pead_cpcv.py) is the
reference subclass and remains byte-identical to the committed +0.546 config.
"""

import logging
import time
from datetime import date as _date
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Symbols that are data inputs, not tradable names.
_SYNTHETIC = {"^VIX", "VIX", "SPY"}


class EventEdgeStrategy:
    """Thin per-fold AgentSimulator adapter for an event-edge scorer.

    Implements the interface scripts.walkforward.cpcv.run_cpcv expects:
    ``model_type``, ``fetch_data(start, end)``, ``run_fold(...)`` plus the
    attributes run_cpcv reads (``symbols_data``, ``all_days_sorted``,
    ``model.trained_through``, ``allow_in_sample``).
    """

    model_type = "event"
    # PIT universe construction (PEAD uses russell1000 / swing trade history).
    pit_index = "russell1000"
    pit_trade_type = "swing"
    # Download ^VIX (scorers may regime-gate on it).
    download_vix = True
    # Alpha-v4 P0.1: rules-based — NO model is ever fit (see __init__:
    # model.trained_through == date.min). run_fold uses the train window ONLY for
    # PIT universe construction, never for training, so a training window that
    # spans a prior test fold cannot leak. run_cpcv reads this flag to bypass the
    # BUG-23 overlap guard (which exists only to protect a *trained* window),
    # restoring full, unbiased CPCV fold coverage for event/rules strategies.
    is_trained = False

    def __init__(self, scorer, symbols, *, model_type: str | None = None,
                 transaction_cost_pct: float = 0.0005,
                 entry_slippage_pct: float | None = None,
                 stop_slippage_pct: float | None = None,
                 max_hold_bars_override: int | None = None,
                 no_prefilters: bool = True):
        if model_type is not None:
            self.model_type = model_type
        self.scorer = scorer
        self.symbols = list(symbols)
        self.transaction_cost_pct = transaction_cost_pct
        # Only forwarded to AgentSimulator when explicitly set (else its
        # module-constant defaults apply) -- preserves byte-identical defaults.
        self.entry_slippage_pct = entry_slippage_pct
        self.stop_slippage_pct = stop_slippage_pct
        self.max_hold_bars_override = max_hold_bars_override
        self.no_prefilters = no_prefilters

        self.symbols_data: Dict[str, pd.DataFrame] = {}
        self.spy_prices = None
        self.all_days_sorted: list = []
        self._global_regime_map: dict = {}
        # Rules-based: no ML cutoff -> every test fold is OOS.
        self.model = type("_NoModel", (), {"trained_through": _date.min})()
        self.allow_in_sample = False
        # Side-channels populated per fold (read by significance / LOCO harnesses).
        self._last_equity_curve: list = []
        self._last_trades: list = []

    # ── data ──────────────────────────────────────────────────────────────────

    def fetch_data(self, start: datetime, end: datetime) -> None:
        import yfinance as yf
        from app.utils.constants import RUSSELL_1000_TICKERS

        t0 = time.time()
        syms = self.symbols or list(RUSSELL_1000_TICKERS)
        logger.info("Downloading daily bars %s -> %s for %d symbols",
                    start.date(), end.date(), len(syms))
        for sym in syms:
            try:
                df = yf.download(sym, start=start.date().isoformat(),
                                 end=end.date().isoformat(), progress=False,
                                 auto_adjust=True)
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

        if self.download_vix:
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

        # Global regime map over the full window so VIX quartile thresholds are
        # stable across folds (mirrors swing.py). Needed for worst-regime gate.
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

    # ── per-fold sim configuration (override point) ────────────────────────────

    def _fold_sim_kwargs(self, tr_start, te_start) -> dict:
        """AgentSimulator kwargs for this fold. Subclasses extend for edge knobs.

        Base set: no_prefilters + optional explicit slippage / hold overrides.
        Anything omitted falls through to AgentSimulator's module defaults.
        """
        kw: dict = {"no_prefilters": self.no_prefilters}
        if self.max_hold_bars_override is not None:
            kw["max_hold_bars_override"] = self.max_hold_bars_override
        if self.entry_slippage_pct is not None:
            kw["entry_slippage_pct"] = self.entry_slippage_pct
        if self.stop_slippage_pct is not None:
            kw["stop_slippage_pct"] = self.stop_slippage_pct
        return kw

    def _fold_universe(self, tr_start, te_start) -> set:
        """PIT-safe tradable universe for this fold (uses te_start to avoid
        leaking names that joined the index mid-test-period)."""
        from app.data.universe_history import (
            pit_union as _pit_union, historical_trade_symbols as _hist_syms,
        )
        extra = _hist_syms(tr_start, te_start, trade_type=self.pit_trade_type)
        return set(_pit_union(self.pit_index, tr_start, te_start, extra_symbols=extra))

    # ── per-fold run ───────────────────────────────────────────────────────────

    def run_fold(self, fold_idx, n_folds, tr_start, tr_end, te_start, te_end):
        from app.backtesting.agent_simulator import AgentSimulator
        from scripts.walkforward.gates import (
            FoldResult, compute_profit_factor, compute_calmar, compute_k_ratio, fold_years,
        )
        from scripts.walkforward.regime import compute_regime_sharpes as _crs
        from scripts.walkforward.regime import event_regime_sharpes as _ers

        pit_members = self._fold_universe(tr_start, te_start)
        fold_symbols_data = {
            s: d for s, d in self.symbols_data.items()
            if s in pit_members or s in _SYNTHETIC
        }

        sim = AgentSimulator(
            model=None,
            factor_scorer=self.scorer,
            transaction_cost_pct=self.transaction_cost_pct,
            **self._fold_sim_kwargs(tr_start, te_start),
        )
        result = sim.run(
            fold_symbols_data,
            start_date=te_start,
            end_date=te_end,
            spy_prices=self.spy_prices,
        )

        stop_exits = result.exit_breakdown.get("STOP", 0)
        n_trades = int(result.total_trades)
        stop_rate = float(stop_exits) / max(n_trades, 1)
        trades_list = getattr(result, "trades", None) or []
        trade_returns = [t.pnl_pct for t in trades_list if hasattr(t, "pnl_pct")]
        equity_curve = getattr(result, "equity_curve", [])
        # Side-channels for the significance / crisis-robustness harnesses
        # (pure-additive; never read by run_cpcv or the gate).
        self._last_equity_curve = equity_curve
        self._last_trades = list(trades_list)
        n_obs = max(len(equity_curve) - 1, 0)
        # Alpha-v4 P0: dated OOS daily returns for the residual-alpha (CAPM/HAC) diagnostic
        # — (date, ret) aligned to the curve's own dates. PURE-ADDITIVE; mirrors swing.py.
        # Without this the EventEdge/PEAD CPCV emitted no residual-α (the alpha-vs-beta check),
        # which matters because PEAD's base edge is beta-driven (so does any enhancement to it).
        _eq_vals = [v for _, v in equity_curve]
        _daily_rets_dated = [
            (equity_curve[i][0],
             (_eq_vals[i] - _eq_vals[i - 1]) / max(_eq_vals[i - 1], 1e-9))
            for i in range(1, len(_eq_vals))
        ] if len(_eq_vals) >= 2 else []
        _regime_obs: dict = {}
        regime_sharpes = _crs(equity_curve, te_start, te_end,
                              regime_map=getattr(self, "_global_regime_map", None),
                              obs_counts=_regime_obs)
        # Alpha-v6 P0 (blueprint X6): EVENT-LEVEL per-regime Sharpes. Each CLOSED
        # trade is one event; its whole-holding return is bucketed by the regime
        # of its ENTRY day. With hundreds of events per window every regime
        # bucket clears REGIME_MIN_OBS in EVENT units, so run_cpcv can populate
        # worst_regime_sharpe even when the daily binning above starves out
        # (the case that used to trigger the paper-only event-sparsity waiver).
        # PURE-ADDITIVE: new FoldResult fields, never touches regime_sharpes.
        _event_regime_obs: dict = {}
        _event_returns = [
            (t.entry_date, float(t.pnl_pct)) for t in trades_list
            if getattr(t, "exit_date", None) is not None
            and getattr(t, "pnl_pct", None) is not None
        ]
        event_sharpes = _ers(_event_returns,
                             getattr(self, "_global_regime_map", None),
                             obs_counts=_event_regime_obs)
        years = fold_years(te_start, te_end)
        sharpe = float(result.sharpe_ratio)
        total_ret = float(result.total_return_pct)
        max_dd = float(result.max_drawdown_pct)
        win_rate = float(result.win_rate)

        logger.info("Fold %d done - %d trades, Sharpe %.3f, return %.1f%%",
                    fold_idx, n_trades, sharpe, total_ret * 100)
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
            event_regime_sharpes=event_sharpes,
            event_regime_obs_counts=_event_regime_obs,
            daily_returns_dated=_daily_rets_dated,
        )


# Re-export the scorer contract type for documentation / type hints.
ScorerResult = List[Tuple[str, float, int]]
