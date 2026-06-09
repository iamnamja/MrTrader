"""OPT-3: earnings IV-crush position builder + CPCV adapter wiring.

Synthetic chain + closes (no network). Verifies the iron-condor builder picks the right
strikes/expiry/dates, is PIT (uses a chain knowable as-of entry), and that the adapter's
run_fold produces a FoldResult with daily_returns_dated. The actual KEEP/KILL verdict comes
from the live CPCV run on the backfilled data (scripts/run_options_ivcrush_cpcv.py).
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from scripts.walkforward.options_strategy import build_ivcrush_iron_condor, OptionsStrategy
from scripts.walkforward.gates import FoldResult


def _chain_df(symbol="AAPL", expiry=date(2025, 1, 17), strikes=(180, 185, 190, 195, 200),
              knowable=date(2024, 12, 1)):
    rows = []
    for k in strikes:
        for kind in ("call", "put"):
            cp = "C" if kind == "call" else "P"
            rows.append({
                "underlying": symbol,
                "contract": f"O:{symbol}{expiry.strftime('%y%m%d')}{cp}{int(k * 1000):08d}",
                "contract_type": kind, "strike": float(k),
                "expiration": pd.Timestamp(expiry),
                "knowable_date": pd.Timestamp(knowable),
            })
    return pd.DataFrame(rows)


def test_builder_picks_atm_condor_strikes():
    chain = _chain_df(strikes=(180, 185, 190, 195, 200))
    get_chain = lambda s, asof: chain  # noqa: E731
    closes = {"AAPL": {date(2025, 1, 6): 190.0, date(2025, 1, 7): 191.0,
                       date(2025, 1, 8): 192.0}}
    # earnings on 2025-01-07 -> entry 2025-01-06 (spot 190), exit 2025-01-08
    pos = build_ivcrush_iron_condor("AAPL", date(2025, 1, 7), get_chain, closes,
                                    min_dte=2, max_dte=45, wing_steps=1)
    assert pos is not None
    assert pos.entry_date == date(2025, 1, 6) and pos.exit_date == date(2025, 1, 8)
    sides = {leg.contract: leg.side for leg in pos.legs}
    # spot 190: short call @190 (>=spot), long call @195 (wing); short put @190, long put @185
    assert sides["O:AAPL250117C00190000"] == -1   # short call ATM
    assert sides["O:AAPL250117C00195000"] == +1    # long call wing
    assert sides["O:AAPL250117P00190000"] == -1    # short put ATM
    assert sides["O:AAPL250117P00185000"] == +1    # long put wing


def test_builder_returns_none_without_chain():
    get_chain = lambda s, asof: None  # noqa: E731
    closes = {"AAPL": {date(2025, 1, 6): 190.0, date(2025, 1, 8): 192.0}}
    assert build_ivcrush_iron_condor("AAPL", date(2025, 1, 7), get_chain, closes) is None


def test_builder_respects_dte_window():
    # Only a far expiry (>45 DTE) exists -> no position.
    chain = _chain_df(expiry=date(2025, 6, 20))
    get_chain = lambda s, asof: chain  # noqa: E731
    closes = {"AAPL": {date(2025, 1, 6): 190.0, date(2025, 1, 8): 192.0}}
    assert build_ivcrush_iron_condor("AAPL", date(2025, 1, 7), get_chain, closes) is None


def test_builder_pit_chain_filtered_by_asof():
    # A chain knowable only AFTER entry must not be used (PIT). The adapter's _get_chain
    # filters knowable<=as_of; here we emulate by returning empty for the early as_of.
    chain = _chain_df(knowable=date(2025, 1, 10))  # knowable after the 01-06 entry

    def get_chain(s, asof):
        return chain[chain["knowable_date"] <= pd.Timestamp(asof)]
    closes = {"AAPL": {date(2025, 1, 6): 190.0, date(2025, 1, 8): 192.0}}
    assert build_ivcrush_iron_condor("AAPL", date(2025, 1, 7), get_chain, closes) is None


def test_adapter_run_fold_produces_foldresult():
    # Wire a tiny synthetic end-to-end: one AAPL earnings event, full chain + option bars.
    strat = OptionsStrategy(symbols=["AAPL"])
    expiry = date(2025, 1, 17)
    strat._chain_by_sym = {"AAPL": _chain_df(expiry=expiry, knowable=date(2024, 12, 1))}
    strat._underlying_closes = {"AAPL": {date(2025, 1, 6): 190.0, date(2025, 1, 7): 191.0,
                                         date(2025, 1, 8): 192.0}}
    strat._earnings = {"AAPL": [date(2025, 1, 7)]}
    # option bars for the four condor legs (em=None here -> ATM shorts @190, 2-wide wings
    # @200C / @180P) on entry (01-06) and exit (01-08)
    legs = ["O:AAPL250117C00190000", "O:AAPL250117C00200000",
            "O:AAPL250117P00190000", "O:AAPL250117P00180000"]
    brows = []
    for c in legs:
        for d, px in [(date(2025, 1, 6), 3.0), (date(2025, 1, 8), 1.5)]:
            brows.append({"underlying": "AAPL", "contract": c, "date": pd.Timestamp(d),
                          "open": px, "high": px, "low": px, "close": px, "volume": 50.0,
                          "knowable_date": pd.Timestamp(d)})
    strat._all_bars = pd.DataFrame(brows)
    strat._global_regime_map = {}

    fr = strat.run_fold(0, 8, date(2024, 6, 1), date(2024, 12, 31),
                        date(2025, 1, 1), date(2025, 1, 31))
    assert isinstance(fr, FoldResult)
    assert fr.trades == 1                       # one condor entered
    assert isinstance(fr.daily_returns_dated, list)
    assert fr.n_obs >= 1


def test_adapter_duck_types_event_edge_interface():
    strat = OptionsStrategy(symbols=["AAPL"])
    assert strat.is_trained is False
    assert strat.model.trained_through == date.min
    assert strat.allow_in_sample is False
    for attr in ("model_type", "fetch_data", "run_fold", "symbols_data",
                 "all_days_sorted", "spy_prices"):
        assert hasattr(strat, attr)
