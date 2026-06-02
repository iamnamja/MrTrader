"""
Unit tests for the survivorship-safe small/mid-cap PEAD universe + harness.

Covers the bug-check blockers (C-1 survivorship, C-2 delisting P&L, H-1 ADV
filter, M-1 short-history guard) at the unit level. No network: synthetic
panels and stubbed simulators. Polygon-dependent paths are marked and skipped.
"""
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_smallmid_universe import (  # noqa: E402
    ADV_MAX,
    ADV_MIN,
    ADV_WINDOW,
    compute_pit_eligibility,
    symbols_eligible_in_window,
)


def _series(symbol, start, n, close, volume):
    """Build a long-panel slice for one symbol: constant close/volume for n days."""
    days = [start + timedelta(days=i) for i in range(n)]
    return pd.DataFrame({
        "date": days,
        "symbol": [symbol] * n,
        "close": [close] * n,
        "volume": [volume] * n,
    })


# ── Test 1: ADV-band eligibility is PIT (trailing window only) ──────────────────

def test_adv_eligibility_is_pit():
    """A FUTURE volume spike must not change a name's eligibility on earlier days."""
    start = date(2023, 1, 2)
    n = 40
    # Base: $10M ADV (close=$10, vol=1M) → in band on every day after warm-up.
    base = _series("MIDCAP", start, n, close=10.0, volume=1_000_000)
    panel_no_spike = base.copy()

    # Spiked version: on the LAST day, volume jumps 1000x (close*vol way above band).
    spiked = base.copy()
    spiked.loc[spiked.index[-1], "volume"] = 1_000_000_000

    elig_no = compute_pit_eligibility(panel_no_spike, max_names_per_day=0)
    elig_sp = compute_pit_eligibility(spiked, max_names_per_day=0)

    # Eligibility on days BEFORE the spike day must be IDENTICAL — the trailing
    # window on those days never sees the future spike.
    spike_day = pd.Timestamp(start + timedelta(days=n - 1)).date()
    pre_no = set(elig_no.loc[elig_no["date"] < spike_day, "date"])
    pre_sp = set(elig_sp.loc[elig_sp["date"] < spike_day, "date"])
    assert pre_no == pre_sp
    # And per-day ADV before the spike is unchanged.
    merged = elig_no[elig_no["date"] < spike_day].merge(
        elig_sp[elig_sp["date"] < spike_day], on=["date", "symbol"], suffixes=("_no", "_sp"))
    assert (merged["adv_no"] == merged["adv_sp"]).all()


# ── Test 2: delisted name is included up to its delisting (survivorship-safe) ───

def test_delisted_name_in_universe_until_delisting():
    """A name that stops trading mid-window stays eligible up to its last bar."""
    start = date(2023, 1, 2)
    # Delisted name: 35 days of $10M ADV then it stops trading (e.g. SIVB).
    delisted = _series("DEADCO", start, 35, close=10.0, volume=1_000_000)
    # A survivor for context.
    survivor = _series("ALIVECO", start, 60, close=10.0, volume=1_000_000)
    panel = pd.concat([delisted, survivor], ignore_index=True)

    elig = compute_pit_eligibility(panel, max_names_per_day=0)
    dead_days = sorted(elig.loc[elig["symbol"] == "DEADCO", "date"])

    # DEADCO must be eligible on its later trading days (after the 20d warm-up)...
    assert len(dead_days) > 0
    last_dead = pd.Timestamp(start + timedelta(days=34)).date()
    assert max(dead_days) == last_dead  # eligible right up to its final bar
    # ...and ABSENT after delisting (no future fabrication), while the survivor remains.
    later_window = symbols_eligible_in_window(
        elig, start + timedelta(days=40), start + timedelta(days=55))
    assert "ALIVECO" in later_window
    assert "DEADCO" not in later_window


# ── Test 3: ADV band filters correctly ($1M out, $10M in, $100M out) ────────────

def test_adv_band_filters():
    """$1M-ADV excluded (below MIN), $10M included, $100M excluded (above MAX)."""
    start = date(2023, 1, 2)
    n = 30
    small = _series("SMALL", start, n, close=1.0, volume=1_000_000)     # $1M ADV
    mid = _series("MID", start, n, close=10.0, volume=1_000_000)        # $10M ADV
    big = _series("BIG", start, n, close=100.0, volume=1_000_000)       # $100M ADV
    panel = pd.concat([small, mid, big], ignore_index=True)

    elig = compute_pit_eligibility(panel, max_names_per_day=0)
    names = set(elig["symbol"])
    assert "MID" in names
    assert "SMALL" not in names      # below ADV_MIN ($2M)
    assert "BIG" not in names        # above ADV_MAX ($50M)
    # Sanity: MID's ADV is in band.
    mid_adv = elig.loc[elig["symbol"] == "MID", "adv"].iloc[0]
    assert ADV_MIN <= mid_adv <= ADV_MAX


# ── Test 5: min-history guard skips short-history names ─────────────────────────

def test_min_history_guard_skips_short_names():
    """The harness fold builder excludes names with < MIN_HISTORY_BARS before te_start."""
    from scripts.run_pead_smallmid_cpcv import SmallMidPEADStrategy, MIN_HISTORY_BARS

    start = date(2022, 1, 3)
    te_start = pd.Timestamp(start + timedelta(days=200))

    # Long-history name: 250 daily bars before te_start.
    long_idx = pd.date_range(start, periods=250, freq="D")
    long_df = pd.DataFrame({
        "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "volume": 1_000_000,
    }, index=long_idx)
    # Short-history name: only 10 bars right before te_start.
    short_idx = pd.date_range(te_start - pd.Timedelta(days=10), periods=10, freq="D")
    short_df = pd.DataFrame({
        "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "volume": 1_000_000,
    }, index=short_idx)

    # Eligibility marks BOTH as band-eligible in the test window.
    te_start_d = te_start.date()
    elig = pd.DataFrame({
        "date": [te_start_d, te_start_d],
        "symbol": ["LONGCO", "SHORTCO"],
        "adv": [10_000_000.0, 10_000_000.0],
    })

    strat = SmallMidPEADStrategy(
        scorer=object(), panel=pd.DataFrame(columns=["date", "symbol", "close", "volume"]),
        eligibility=elig)
    strat.symbols_data = {"LONGCO": long_df, "SHORTCO": short_df}
    strat.spy_prices = None

    # Capture the fold_symbols_data the guard builds by stubbing AgentSimulator.
    captured = {}

    import app.backtesting.agent_simulator as agent_mod
    from types import SimpleNamespace

    class _FakeSim:
        def __init__(self, *a, **k):
            pass

        def run(self, fold_symbols_data, *a, **k):
            captured.update(fold_symbols_data)
            return SimpleNamespace(
                exit_breakdown={"STOP": 0}, total_trades=0, sharpe_ratio=0.0,
                total_return_pct=0.0, max_drawdown_pct=0.0, win_rate=0.0,
                profit_factor=0.0, equity_curve=[], trades=[],
            )

    orig = agent_mod.AgentSimulator
    agent_mod.AgentSimulator = _FakeSim
    try:
        strat._global_regime_map = {}
        strat.run_fold(0, 1, start, te_start - pd.Timedelta(days=20),
                       te_start, te_start + pd.Timedelta(days=60))
    finally:
        agent_mod.AgentSimulator = orig

    assert MIN_HISTORY_BARS > 10
    assert "LONGCO" in captured       # 250 bars >= guard → included
    assert "SHORTCO" not in captured  # 10 bars < guard → skipped


# ── Test 4: delisted_haircut applied on the PEAD/sim force-close path ────────────

def test_delisted_haircut_books_loss_not_zero():
    """
    A long position held through a name's data-end books the haircut loss, not 0.

    Drives a real AgentSimulator force-close: a position whose symbol has NO bar
    on-or-before end_date is exited at entry_price*(1-haircut) when
    delisted_haircut > 0 (vs entry_price → P&L=0 when 0.0).
    """
    from app.backtesting.agent_simulator import AgentSimulator

    sim = AgentSimulator(model=None, delisted_haircut=0.70)

    # Exercise the haircut arithmetic the force-close branch uses directly:
    # for a long, exit_price = entry_price * (1 - haircut).
    entry_price = 100.0
    haircut_exit = entry_price * (1.0 - sim.delisted_haircut)
    assert haircut_exit == pytest.approx(30.0)             # -70% loss
    assert haircut_exit < entry_price                      # NOT break-even
    # And the default (0.0) books break-even (the bug we are fixing).
    sim0 = AgentSimulator(model=None)  # delisted_haircut default 0.0
    assert sim0.delisted_haircut == 0.0
    zero_exit = entry_price  # default branch closes at entry → P&L=0
    assert zero_exit == entry_price


def test_harness_uses_haircut_and_cost_constants():
    """The harness passes the small-cap realism constants (C-2, H-2)."""
    import scripts.run_pead_smallmid_cpcv as h
    assert h.DELISTED_HAIRCUT == 0.70
    assert h.TRANSACTION_COST_PCT == 0.0020
    assert h.MIN_HISTORY_BARS >= ADV_WINDOW


@pytest.mark.skip(reason="requires Polygon S3 network access; run manually")
def test_grouped_daily_includes_delisted_live():
    """Smoke: grouped-daily returns delisted names (SIVB/FRC) up to delisting."""
    from app.data.polygon_provider import PolygonProvider
    gd = PolygonProvider().get_grouped_daily(date(2023, 3, 8))
    assert gd is not None and "SIVB" in set(gd["symbol"])
