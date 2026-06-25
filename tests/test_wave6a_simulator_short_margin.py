"""Alpha-v10 audit Wave 6a — signal-mode shorts must reserve margin (research-tier sim).

In SIGNAL mode, opening a short previously did a net-zero cash hack (cash += proceeds; cash -= notional)
and never set short_collateral, while buying_power returned raw cash — so the RM buying-power gate saw
~full buying power after a short and the short book could over-leverage. Fix: signal-mode shorts now
reserve collateral = entry notional (parity with REBALANCE mode), buying_power nets the reserve, and
the signal-mode cover (_close_position) releases it. Invariants pinned here: equity_mtm and the
round-trip realized P&L are UNCHANGED; only buying_power (the gate input) is corrected.
"""
from __future__ import annotations

from datetime import date

from app.backtesting.agent_simulator import AgentSimulator, _Position, _PortfolioState


def _sim(tx=0.0):
    s = AgentSimulator.__new__(AgentSimulator)
    s.transaction_cost_pct = tx
    return s


# ── buying_power nets the short reserve; long-only is unchanged ───────────────────
def test_buying_power_nets_short_collateral():
    p = _PortfolioState(cash=100_000.0, peak_equity=100_000.0)
    assert p.buying_power == 100_000.0           # long-only: collateral 0 -> raw cash
    p.short_collateral = 40_000.0
    assert p.buying_power == 60_000.0            # reserve subtracted -> gate can't be defeated


def test_equity_mtm_unchanged_by_reserve_bookkeeping():
    # OLD bookkeeping for a 40k short: cash = cash0 - tx; collateral = 0
    # NEW bookkeeping:                   cash = cash0 + 40k - tx; collateral = 40k
    # equity_mtm = cash - collateral + short_upnl must be identical at entry.
    old = _PortfolioState(cash=100_000.0, peak_equity=100_000.0)        # cash0 - tx (tx=0)
    new = _PortfolioState(cash=140_000.0, peak_equity=100_000.0)
    new.short_collateral = 40_000.0
    pos = _Position(symbol="X", entry_date=date(2026, 1, 2), entry_price=40.0, stop_price=44.0,
                    target_price=36.0, quantity=1000, highest_price=40.0, direction="short")
    old.positions["X"] = pos
    new.positions["X"] = pos
    # at entry (current == entry) short_upnl == 0 -> both equities equal
    assert abs(old.equity_mtm({"X": 40.0}) - new.equity_mtm({"X": 40.0})) < 1e-6


# ── short cover releases the reserve and books correct P&L ────────────────────────
def test_short_cover_releases_collateral_and_pnl():
    s = _sim(tx=0.0)
    # post-entry state for a 1000-share short @ $40 (notional 40k): cash credited proceeds + reserve
    p = _PortfolioState(cash=140_000.0, peak_equity=100_000.0)
    p.short_collateral = 40_000.0
    pos = _Position(symbol="X", entry_date=date(2026, 1, 2), entry_price=40.0, stop_price=44.0,
                    target_price=36.0, quantity=1000, highest_price=40.0, direction="short")
    p.positions["X"] = pos
    # cover at $36 -> profit (40-36)*1000 = 4000
    trade, _tx = s._close_position(pos, date(2026, 1, 5), 36.0, "TARGET", p)
    assert abs(trade.pnl - 4000.0) < 1e-6
    assert abs(p.short_collateral - 0.0) < 1e-6          # reserve fully released
    # cash: 140k (post-entry) - 36*1000 (cover) = 104k; equity = cash - collateral = 104k
    assert abs(p.cash - 104_000.0) < 1e-6
    assert abs(p.equity_decision - 104_000.0) < 1e-6     # = starting 100k + 4k profit


def test_short_roundtrip_equity_matches_realized_pnl_with_tx():
    s = _sim(tx=0.001)   # 10 bps
    notional = 40_000.0
    qty, entry = 1000, 40.0
    tx_entry = notional * 0.001
    # simulate the NEW entry bookkeeping: cash += proceeds - tx_entry; collateral += notional
    p = _PortfolioState(cash=100_000.0 + notional - tx_entry, peak_equity=100_000.0)
    p.short_collateral = notional
    pos = _Position(symbol="X", entry_date=date(2026, 1, 2), entry_price=entry, stop_price=44.0,
                    target_price=36.0, quantity=qty, highest_price=entry, direction="short")
    p.positions["X"] = pos
    exit_price = 36.0
    trade, _ = s._close_position(pos, date(2026, 1, 5), exit_price, "TARGET", p)
    tx_exit = exit_price * qty * 0.001
    expected_pnl = (entry - exit_price) * qty - tx_exit      # entry tx already charged at entry
    assert abs(trade.pnl - expected_pnl) < 1e-6
    assert abs(p.short_collateral) < 1e-6
    # final equity == starting 100k + gross profit - both tx legs
    expected_equity = 100_000.0 + (entry - exit_price) * qty - tx_entry - tx_exit
    assert abs(p.equity_decision - expected_equity) < 1e-6
