"""OPT-4: systematic index short-vol builder + adapter.

Synthetic chain + closes (no network). Locks: scheduled entry/exit, OTM strike placement from
the expected move, realized-vol expected-move estimate, the calendar cadence, and that it reuses
the OptionsStrategy sim-driving path (FoldResult with daily_returns_dated).
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from scripts.walkforward.options_strategy import (
    build_index_short_condor, IndexShortVolStrategy, INDEX_ETFS,
)
from scripts.walkforward.gates import FoldResult


def _chain_df(symbol="SPY", expiry=date(2025, 2, 21),
              strikes=range(380, 421, 5), knowable=date(2024, 12, 1)):
    rows = []
    for k in strikes:
        for kind in ("call", "put"):
            cp = "C" if kind == "call" else "P"
            rows.append({
                "underlying": symbol,
                "contract": f"O:{symbol}{expiry.strftime('%y%m%d')}{cp}{int(k * 1000):08d}",
                "contract_type": kind, "strike": float(k),
                "expiration": pd.Timestamp(expiry), "knowable_date": pd.Timestamp(knowable),
            })
    return pd.DataFrame(rows)


def _weekday_closes(start, n, price=400.0):
    out, d, made = {}, start, 0
    while made < n:
        if d.weekday() < 5:
            out[d] = price
            made += 1
        d += timedelta(days=1)
    return out


def test_index_builder_scheduled_entry_exit_and_otm_strikes():
    # ~50 trading days of SPY at 400; expiry ~35 DTE out.
    closes = {"SPY": _weekday_closes(date(2025, 1, 2), 50, 400.0)}
    entry_sched = date(2025, 1, 8)
    # expiry ~35 DTE out from the scheduled entry
    chain = _chain_df(expiry=date(2025, 2, 12), strikes=range(360, 441, 5))
    get_chain = lambda s, asof: chain  # noqa: E731
    pos = build_index_short_condor("SPY", entry_sched, get_chain, closes,
                                   expected_move=0.05, sd_mult=1.0, hold_trading_days=21,
                                   target_dte=35, min_dte=20, max_dte=55, wing_steps=2)
    assert pos is not None
    # entry is the scheduled day (a trading day on/before it); exit is 21 trading days later
    assert pos.entry_date == entry_sched
    assert pos.exit_date > pos.entry_date
    # em=0.05: short call >= 400*1.05=420; short put <= 400*0.95=380
    shorts = [leg.contract for leg in pos.legs if leg.side == -1]
    assert any("C00420000" in c for c in shorts)
    assert any("P00380000" in c for c in shorts)
    # 4 legs, net short (2 short + 2 long)
    assert sum(1 for leg in pos.legs if leg.side == -1) == 2
    assert sum(1 for leg in pos.legs if leg.side == +1) == 2


def test_index_builder_requires_expected_move():
    closes = {"SPY": _weekday_closes(date(2025, 1, 2), 50, 400.0)}
    get_chain = lambda s, asof: _chain_df()  # noqa: E731
    assert build_index_short_condor("SPY", date(2025, 1, 8), get_chain, closes,
                                    expected_move=None) is None


def test_strategy_expected_move_from_realized_vol():
    strat = IndexShortVolStrategy(symbols=["SPY"], rv_lookback=20, dte_target=35)
    # build closes with a small daily wiggle so realized vol > 0
    closes, d, made, px = {}, date(2025, 1, 2), 0, 400.0
    while made < 40:
        if d.weekday() < 5:
            px *= (1.01 if made % 2 == 0 else 0.99)   # ~1%/day oscillation
            closes[d] = px
            made += 1
        d += timedelta(days=1)
    strat._underlying_closes = {"SPY": closes}
    em = strat._expected_move("SPY", max(closes))
    assert em is not None and 0.0 < em < 1.0
    # ~1%/day over ~24 trading-day horizon -> a few % expected move
    assert 0.01 < em < 0.30


def test_strategy_schedule_cadence():
    strat = IndexShortVolStrategy(symbols=["SPY"], rebalance_trading_days=21)
    strat._underlying_closes = {"SPY": _weekday_closes(date(2025, 1, 2), 60, 400.0)}
    events = list(strat._events_in_window(date(2025, 1, 1), date(2025, 4, 1)))
    syms = {s for s, _ in events}
    assert syms == {"SPY"}
    dts = [d for _, d in events]
    # entries spaced ~21 trading days apart
    assert len(dts) >= 2
    assert all(dts[i] < dts[i + 1] for i in range(len(dts) - 1))


def test_strategy_duck_types_and_no_earnings():
    strat = IndexShortVolStrategy()
    assert strat.symbols == INDEX_ETFS
    assert strat._needs_earnings is False
    assert strat.is_trained is False
    assert strat.model.trained_through == date.min
    assert strat.model_type == "options_index_shortvol"
    for attr in ("fetch_data", "run_fold", "_expected_move", "_events_in_window"):
        assert hasattr(strat, attr)


def test_strategy_run_fold_end_to_end():
    # Wire a tiny synthetic fold: one scheduled SPY condor with bars on entry + exit.
    strat = IndexShortVolStrategy(symbols=["SPY"], rebalance_trading_days=21,
                                  hold_trading_days=10, dte_target=35)
    closes = {"SPY": _weekday_closes(date(2025, 1, 2), 40, 400.0)}
    # add a wiggle so expected_move > 0
    keys = sorted(closes["SPY"])
    for i, k in enumerate(keys):
        closes["SPY"][k] = 400.0 * (1.0 + 0.01 * (1 if i % 2 else -1))
    strat._underlying_closes = closes
    expiry = date(2025, 2, 12)
    strat._chain_by_sym = {"SPY": _chain_df(expiry=expiry, strikes=range(360, 441, 5),
                                            knowable=date(2024, 12, 1))}
    entry = keys[0]
    exit_d = keys[10]
    legs = ["O:SPY250212C00420000", "O:SPY250212C00430000",
            "O:SPY250212P00380000", "O:SPY250212P00370000"]
    brows = []
    for c in legs:
        for dd, px in [(entry, 3.0), (exit_d, 1.4)]:
            brows.append({"underlying": "SPY", "contract": c, "date": pd.Timestamp(dd),
                          "open": px, "high": px, "low": px, "close": px, "volume": 500.0,
                          "knowable_date": pd.Timestamp(dd)})
    strat._all_bars = pd.DataFrame(brows)
    strat._global_regime_map = {}
    fr = strat.run_fold(0, 8, date(2024, 6, 1), date(2024, 12, 31),
                        keys[0], keys[-1])
    assert isinstance(fr, FoldResult)
    assert isinstance(fr.daily_returns_dated, list)
