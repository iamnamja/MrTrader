"""
Options strategy CPCV adapter — OPT-3.

Duck-types to scripts.walkforward.event_edge.EventEdgeStrategy so scripts.walkforward.cpcv.
run_cpcv drives it UNCHANGED — but each fold runs the OPT-2 OptionsSimulator (not the equity
AgentSimulator) over a set of option positions produced by a position builder, and returns a
FoldResult with daily_returns_dated populated (so the significance gate + CAPM residual-α +
fold-coverage all apply). This is the ONLY disposable layer of the options program: a new
options strategy is a new position builder + a runner; nothing below the adapter changes.

A *position builder* is any callable:

    builder(symbol, event_date, get_chain, underlying_closes) -> Optional[OptionPosition]

where get_chain(symbol, as_of) -> DataFrame[contract, contract_type, strike, expiration] is a
PIT (knowable_date <= as_of) chain accessor. The reference builder is the earnings IV-crush
iron condor (build_ivcrush_iron_condor).

Rules-based: is_trained=False, model.trained_through=date.min -> every test fold is OOS and
run_cpcv bypasses the overlap guard for full coverage (same as EventEdgeStrategy).

NOTE (known limitation, documented in OPTIONS_PROGRAM.md): the underlying universe is the set
of currently-liquid optionable names, so there is a mild SURVIVORSHIP bias at the *underlying*
level (the option *contract* universe is survivorship-safe via OPT-1b). For a short-vol /
IV-crush edge this is second-order, but a fully PIT underlying universe is a future refinement.
Underlying closes are RAW (auto_adjust=False) so they match the unadjusted OCC strikes across
the in-window splits (NVDA/AMZN/TSLA/GOOGL).
"""
from __future__ import annotations

import logging
import time
from datetime import date as _date
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional

import pandas as pd

from app.backtesting.options_simulator import (
    OptionsSimulator, OptionsSpreadCostModel, OptionLeg, OptionPosition, daily_returns_dated,
)
from app.data.options_provider import load_options_bars, load_options_contracts

logger = logging.getLogger(__name__)


# ── Position builder: earnings IV-crush iron condor ────────────────────────────

def build_ivcrush_iron_condor(symbol, event_date, get_chain, underlying_closes, *,
                              min_dte=10, max_dte=45, target_dte=25, wing_steps=2,
                              expected_move=None, short_em_mult=1.3,
                              allow_atm=False) -> Optional[OptionPosition]:
    """Short-vol-into-earnings, DEFINED RISK. Enter an iron condor the trading day BEFORE the
    report and exit the trading day AFTER (bracketing BMO/AMC), so the post-earnings IV crush
    reprices the short premium lower.

    Canonical earnings short-vol parameterization (OPT-3 fair-test revision):
      * Expiry nearest ``target_dte`` (~25 DTE monthly) within [min_dte, max_dte] — NOT the
        nearest weekly. A ~2-3 DTE weekly is the worst tenor for short-vol (tiny vega to
        harvest, max gamma on a breach); ~25 DTE maximizes the vega/gamma ratio.
      * Short strikes ~``short_em_mult`` × expected move OUTSIDE spot (1.3× — pushes breach
        probability into the ~20-30% range where credit + crush win on average; 1.0× breaches
        ~40% and loses), long wings ``wing_steps`` strikes further out.
    ``expected_move`` is a fractional move (e.g. 0.06 = 6%). When None we SKIP the event
    (no position) unless allow_atm=True — the ATM strawman is a known guaranteed-loser and is
    not traded in production (it injected losers into the first naive run). Returns None when
    the event can't be traded (the adapter/sim count drops)."""
    closes = underlying_closes.get(symbol)
    if not closes:
        return None
    if expected_move is None:
        if not allow_atm:
            return None        # no expected-move estimate -> skip (never trade ATM strawman)
        em = 0.0
    else:
        em = expected_move * short_em_mult
    dates = sorted(closes)
    prior = [x for x in dates if x < event_date]
    after = [x for x in dates if x > event_date]
    if not prior or not after:
        return None
    entry, exit_d = max(prior), min(after)
    spot = closes[entry]
    if spot <= 0:
        return None

    chain = get_chain(symbol, entry)   # PIT chain metadata (knowable <= entry)
    if chain is None or chain.empty:
        return None
    lo, hi = event_date + timedelta(days=min_dte), event_date + timedelta(days=max_dte)
    exp_col = pd.to_datetime(chain["expiration"]).dt.date
    span_exps = sorted({e for e in exp_col if lo <= e <= hi})
    if not span_exps:
        return None
    # nearest to the target tenor (canonical monthly), not the nearest weekly
    expiry = min(span_exps, key=lambda e: abs((e - event_date).days - target_dte))
    leg_chain = chain[pd.to_datetime(chain["expiration"]).dt.date == expiry]
    calls = sorted(leg_chain[leg_chain["contract_type"] == "call"]["strike"].unique())
    puts = sorted(leg_chain[leg_chain["contract_type"] == "put"]["strike"].unique())
    if len(calls) < wing_steps + 1 or len(puts) < wing_steps + 1:
        return None

    # Short-strike targets: outside the expected move (canonical) or ATM (strawman).
    call_target = spot * (1.0 + em)
    put_target = spot * (1.0 - em)

    sc = [k for k in calls if k >= call_target]
    if not sc:
        return None
    short_call_k = sc[0]
    sc_idx = calls.index(short_call_k)
    if sc_idx + wing_steps >= len(calls):
        return None
    long_call_k = calls[sc_idx + wing_steps]
    sp = [k for k in puts if k <= put_target]
    if not sp:
        return None
    short_put_k = sp[-1]
    sp_idx = puts.index(short_put_k)
    if sp_idx - wing_steps < 0:
        return None
    long_put_k = puts[sp_idx - wing_steps]

    def _occ(strike, kind):
        cp = "C" if kind == "call" else "P"
        return f"O:{symbol}{expiry.strftime('%y%m%d')}{cp}{int(round(strike * 1000)):08d}"

    legs = [
        OptionLeg(_occ(short_call_k, "call"), side=-1, qty=1),
        OptionLeg(_occ(long_call_k, "call"), side=+1, qty=1),
        OptionLeg(_occ(short_put_k, "put"), side=-1, qty=1),
        OptionLeg(_occ(long_put_k, "put"), side=+1, qty=1),
    ]
    return OptionPosition(legs, entry_date=entry, exit_date=exit_d,
                          label=f"{symbol}@{event_date}")


# ── Adapter ────────────────────────────────────────────────────────────────────

class OptionsStrategy:
    """CPCV adapter that drives the OPT-2 OptionsSimulator from an event position builder."""

    model_type = "options_ivcrush"
    pit_index = "russell1000"
    pit_trade_type = "swing"
    is_trained = False
    per_fold_retrain = False

    def __init__(self, symbols, position_builder: Callable = build_ivcrush_iron_condor, *,
                 model_type: Optional[str] = None, spread_mult: float = 1.0,
                 cost_model: Optional[OptionsSpreadCostModel] = None,
                 starting_capital: float = 1_000_000.0):
        if model_type:
            self.model_type = model_type
        self.symbols = list(symbols)
        self.position_builder = position_builder
        self.spread_mult = spread_mult
        self.cost_model = cost_model or OptionsSpreadCostModel()
        self.starting_capital = starting_capital

        self.symbols_data: Dict[str, pd.DataFrame] = {}
        self.spy_prices = None
        self.all_days_sorted: list = []
        self._global_regime_map: dict = {}
        self._underlying_closes: Dict[str, Dict[_date, float]] = {}  # sym -> {date: raw close}
        self._earnings: Dict[str, List[_date]] = {}
        self._chain_by_sym: Dict[str, pd.DataFrame] = {}   # sym -> contract metadata (sorted)
        self._all_bars: Optional[pd.DataFrame] = None       # full options OHLCV (filtered/fold)
        self.model = type("_NoModel", (), {"trained_through": _date.min})()
        self.allow_in_sample = False

    # ── data ────────────────────────────────────────────────────────────────────

    def fetch_data(self, start: datetime, end: datetime) -> None:
        import yfinance as yf
        from app.data.fmp_provider import get_earnings_history_fmp

        t0 = time.time()
        for sym in self.symbols:
            try:
                # RAW closes (auto_adjust=False) to match unadjusted OCC strikes across splits.
                df = yf.download(sym, start=start.date().isoformat(),
                                 end=end.date().isoformat(), progress=False, auto_adjust=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.columns = [c.lower() for c in df.columns]
                if len(df) >= 60 and "close" in df.columns:
                    self.symbols_data[sym] = df
                    self._underlying_closes[sym] = {
                        (t.date() if hasattr(t, "date") else t): float(v)
                        for t, v in df["close"].items()}
            except Exception:
                pass

        spy = yf.download("SPY", start=start.date().isoformat(),
                          end=end.date().isoformat(), progress=False, auto_adjust=False)
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)
        spy.columns = [c.lower() for c in spy.columns]
        self.spy_prices = spy["close"]
        self.symbols_data["SPY"] = spy

        for sym in self.symbols:
            try:
                recs = get_earnings_history_fmp(sym)
                dts = sorted({datetime.strptime(r["date"], "%Y-%m-%d").date()
                              for r in recs if r.get("date")})
                if dts:
                    self._earnings[sym] = dts
            except Exception:
                pass

        # Pre-index the options store ONCE (the per-event/per-fold hot path then slices small
        # per-symbol frames instead of scanning the 45M-row / 1.9M-row global frames).
        contracts = load_options_contracts(refresh=True)
        if not contracts.empty:
            for sym, grp in contracts.groupby("underlying"):
                self._chain_by_sym[sym] = grp.reset_index(drop=True)
        self._all_bars = load_options_bars(refresh=True)

        all_days = sorted({(d.date() if hasattr(d, "date") else d)
                           for df in self.symbols_data.values() for d in df.index})
        self.all_days_sorted = all_days
        try:
            from scripts.walkforward.regime import load_regime_map as _lrm
            self._global_regime_map = _lrm(start.date(), end.date())
        except Exception:
            self._global_regime_map = {}

        n_events = sum(len(v) for v in self._earnings.values())
        logger.info("Data: %d underlyings, %d earnings events, %d option contracts, %d days "
                    "in %.1fs", len(self._underlying_closes), n_events,
                    sum(len(v) for v in self._chain_by_sym.values()),
                    len(all_days), time.time() - t0)

    # ── PIT chain accessor (fast) ────────────────────────────────────────────────

    def _get_chain(self, symbol: str, as_of: _date) -> Optional[pd.DataFrame]:
        df = self._chain_by_sym.get(symbol)
        if df is None or df.empty:
            return None
        return df[df["knowable_date"] <= pd.Timestamp(as_of)]

    def _events_in_window(self, te_start, te_end):
        for sym, dts in self._earnings.items():
            if sym not in self._underlying_closes:
                continue
            for ed in dts:
                if te_start <= ed <= te_end:
                    yield sym, ed

    def _expected_move(self, sym: str, event_date: _date, lookback: int = 8,
                       floor: float = 0.04, cap: float = 0.25) -> Optional[float]:
        """PIT expected earnings move = mean |2-day bracket return| over the symbol's last
        `lookback` PAST earnings (strictly before event_date), from raw closes. Floored/capped
        so a thin history can't place absurd strikes. None if no past earnings move computable."""
        closes = self._underlying_closes.get(sym)
        past = [e for e in self._earnings.get(sym, []) if e < event_date]
        if not closes or not past:
            return None
        dates = sorted(closes)
        moves = []
        for e in past[-lookback:]:
            before = [d for d in dates if d < e]
            after = [d for d in dates if d > e]
            if before and after:
                c0, c1 = closes[max(before)], closes[min(after)]
                if c0 > 0:
                    moves.append(abs(c1 / c0 - 1.0))
        if not moves:
            return None
        return max(floor, min(cap, sum(moves) / len(moves)))

    def run_fold(self, fold_idx, n_folds, tr_start, tr_end, te_start, te_end):
        from scripts.walkforward.gates import (
            FoldResult, compute_calmar, compute_k_ratio, fold_years,
        )
        from scripts.walkforward.regime import compute_regime_sharpes as _crs

        events = list(self._events_in_window(te_start, te_end))
        positions: List[OptionPosition] = []
        for sym, ed in events:
            em = self._expected_move(sym, ed)
            pos = self.position_builder(sym, ed, self._get_chain, self._underlying_closes,
                                        expected_move=em)
            if pos is not None:
                positions.append(pos)

        # Bars only for the contracts we actually trade, PIT-filtered (knowable <= te_end).
        wanted = {leg.contract for p in positions for leg in p.legs}
        und_syms = {p.label.split("@")[0] for p in positions}
        if wanted and self._all_bars is not None and not self._all_bars.empty:
            bars = self._all_bars[
                self._all_bars["contract"].isin(wanted)
                & (self._all_bars["knowable_date"] <= pd.Timestamp(te_end))]
        else:
            bars = pd.DataFrame()
        und_prices = {s: self._underlying_closes.get(s, {}) for s in und_syms}

        sim = OptionsSimulator(bars, underlying_prices=und_prices,
                               cost_model=self.cost_model,
                               starting_capital=self.starting_capital)
        result = sim.run(positions, te_start, te_end, spread_mult=self.spread_mult)

        equity_curve = result.equity_curve
        dr_tuples = sorted(daily_returns_dated(result).items())
        n_obs = max(len(equity_curve) - 1, 0)
        _regime_obs: dict = {}
        regime_sharpes = _crs(equity_curve, te_start, te_end,
                              regime_map=getattr(self, "_global_regime_map", None),
                              obs_counts=_regime_obs)
        years = fold_years(te_start, te_end)
        total_ret = float(result.total_return_pct) / 100.0
        max_dd = float(result.max_drawdown_pct) / 100.0
        logger.info("Fold %d: %d events -> %d positions (dropped %d), Sharpe %.3f, ret %.2f%%",
                    fold_idx, len(events), len(positions),
                    getattr(result, "dropped_positions", 0),
                    float(result.sharpe_ratio), total_ret * 100)
        return FoldResult(
            fold=fold_idx, train_start=tr_start, train_end=tr_end,
            test_start=te_start, test_end=te_end,
            trades=int(result.total_trades), win_rate=float(result.win_rate),
            sharpe=float(result.sharpe_ratio), max_drawdown=max_dd, total_return=total_ret,
            stop_exit_rate=0.0, model_version=0,
            profit_factor=float(result.profit_factor),
            calmar_ratio=compute_calmar(total_ret, max_dd, years),
            k_ratio=compute_k_ratio(equity_curve),
            n_obs=n_obs, regime_sharpes=regime_sharpes, regime_obs_counts=_regime_obs,
            daily_returns_dated=dr_tuples,
        )
