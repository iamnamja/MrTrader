"""OPT-5: ImpliedMoveProvider — pre-earnings implied move (ATM straddle / spot), PIT.

Synthetic parquet (no network). Locks the ATM-straddle math, front-expiry selection, and the
PIT cutoff (the observation-day bar, knowable obs+1, is hidden from a same-day decision).
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from app.data.options_signal import ImpliedMoveProvider


def _write_bars(path, rows):
    """rows: (contract, 'YYYY-MM-DD' obs, close, 'YYYY-MM-DD' knowable)."""
    recs = [{"underlying": "SPY", "contract": c, "date": pd.Timestamp(d),
             "open": v, "high": v, "low": v, "close": v, "volume": 100.0,
             "knowable_date": pd.Timestamp(k)} for c, d, v, k in rows]
    pd.DataFrame(recs).to_parquet(path, index=False)


def test_implied_move_atm_straddle(tmp_path):
    p = tmp_path / "options_bars.parquet"
    # spot ~400, front expiry 2025-02-21. ATM call@400=8, put@400=7 -> straddle 15 -> 3.75%.
    # also a wing 410C@3 (should be ignored — not ATM) and a far expiry (ignored).
    _write_bars(p, [
        ("O:SPY250221C00400000", "2025-02-10", 8.0, "2025-02-11"),
        ("O:SPY250221P00400000", "2025-02-10", 7.0, "2025-02-11"),
        ("O:SPY250221C00410000", "2025-02-10", 3.0, "2025-02-11"),
        ("O:SPY250620C00400000", "2025-02-10", 25.0, "2025-02-11"),  # far expiry
    ])
    imp = ImpliedMoveProvider(bars_path=p, contracts_path=p, min_dte=1, max_dte=60)
    # decide later (knowable) -> obs-day bar visible. straddle 8+7=15, /400 = 0.0375
    im = imp.implied_move("SPY", date(2025, 2, 10), 400.0, knowable_asof=date(2025, 2, 20))
    assert abs(im - 0.0375) < 1e-9


def test_implied_move_pit_cutoff_hides_unknowable_bar(tmp_path):
    p = tmp_path / "options_bars.parquet"
    _write_bars(p, [
        ("O:SPY250221C00400000", "2025-02-10", 8.0, "2025-02-11"),
        ("O:SPY250221P00400000", "2025-02-10", 7.0, "2025-02-11"),
    ])
    imp = ImpliedMoveProvider(bars_path=p, contracts_path=p)
    # deciding ON 2025-02-10 (knowable_asof=that day) -> the obs bar (knowable 02-11) is hidden.
    assert imp.implied_move("SPY", date(2025, 2, 10), 400.0,
                            knowable_asof=date(2025, 2, 10)) is None
    # default cutoff (obs+5d) makes it visible again.
    assert imp.implied_move("SPY", date(2025, 2, 10), 400.0) is not None


def test_implied_move_none_when_no_chain(tmp_path):
    p = tmp_path / "options_bars.parquet"
    _write_bars(p, [("O:SPY250221C00400000", "2025-02-10", 8.0, "2025-02-11")])  # call only
    imp = ImpliedMoveProvider(bars_path=p, contracts_path=p)
    # no put -> no straddle -> None
    assert imp.implied_move("SPY", date(2025, 2, 10), 400.0,
                            knowable_asof=date(2025, 2, 20)) is None
    # unknown symbol -> None
    assert imp.implied_move("NOPE", date(2025, 2, 10), 400.0) is None
    # bad spot -> None
    assert imp.implied_move("SPY", date(2025, 2, 10), 0.0) is None


def test_implied_move_respects_dte_window(tmp_path):
    p = tmp_path / "options_bars.parquet"
    # only a far expiry (>60 DTE) -> outside window -> None
    far = (date(2025, 2, 10) + timedelta(days=120)).strftime("%y%m%d")
    _write_bars(p, [
        (f"O:SPY{far}C00400000", "2025-02-10", 20.0, "2025-02-11"),
        (f"O:SPY{far}P00400000", "2025-02-10", 18.0, "2025-02-11"),
    ])
    imp = ImpliedMoveProvider(bars_path=p, contracts_path=p, max_dte=60)
    assert imp.implied_move("SPY", date(2025, 2, 10), 400.0,
                            knowable_asof=date(2025, 2, 20)) is None
