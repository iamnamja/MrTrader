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

def _ohlcv(index, close, volume=1_000_000.0):
    return pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": volume},
        index=index,
    )


def _build_held_then_dataended_fold(last_bar_offset_td, n_fold_days=20):
    """
    Build symbols_data + (te_start, te_end) for a fold where a single factor-scored
    LONG name enters on te_start, then its data ENDS `last_bar_offset_td` trading
    days into the fold (the rest of the fold has no bar for it), while SPY's
    calendar runs the full fold. Returns (symbols_data, te_start, te_end, last_close).

    last_bar_offset_td: index into the fold's business-day calendar where the
    target's last bar sits (0 = te_start). The gap-to-end is therefore
    (n_fold_days-1 - last_bar_offset_td) trading days.
    """
    warm_start = date(2023, 1, 2)
    # Warm-up so per-name feature/sizing math has history; te_start after warm-up.
    warm_idx = pd.bdate_range(warm_start, periods=60)
    te_start = (warm_idx[-1] + pd.tseries.offsets.BDay(1)).date()
    fold_idx = pd.bdate_range(te_start, periods=n_fold_days)
    te_end = fold_idx[-1].date()

    full_idx = warm_idx.append(fold_idx)
    last_close = 50.0  # flat price so no stop/target/max-hold exit fires early

    # SPY runs the entire span (anchors the trading calendar).
    spy = _ohlcv(full_idx, close=400.0)
    # Target name: warm-up + fold bars only up to last_bar_offset_td, then data ends.
    target_idx = warm_idx.append(fold_idx[: last_bar_offset_td + 1])
    target = _ohlcv(target_idx, close=last_close)

    symbols_data = {"SPY": spy, "DEADCO": target}
    return symbols_data, te_start, te_end, last_close


def _run_fold_with_scorer(symbols_data, te_start, te_end, delisted_haircut):
    """Run a real AgentSimulator fold that buys DEADCO on te_start via a scorer."""
    from app.backtesting.agent_simulator import AgentSimulator

    def scorer(day, syms, vix_history=None):
        # Only propose the target on the fold-open day.
        if day == te_start:
            return [("DEADCO", 0.95)]
        return []

    sim = AgentSimulator(
        model=None,
        factor_scorer=scorer,
        no_prefilters=True,
        delisted_haircut=delisted_haircut,
    )
    return sim.run(symbols_data, start_date=te_start, end_date=te_end)


def test_delisted_haircut_bites_on_data_ended_while_held():
    """
    FIX 1: a LONG held through its data-end (>= 3 fold trading days with no bar)
    books the haircut loss off its LAST CLOSE — not the full last close (P&L~0).

    Drives the REAL force-close branch in AgentSimulator.run(): the name enters
    on te_start, trades a couple bars, then data ends mid-fold while SPY's
    calendar runs to te_end. With delisted_haircut=0.70 the FORCE_CLOSE must
    exit at last_close*(1-0.70).
    """
    # Last bar at fold-day 2 → gap of (20-1)-2 = 17 trading days >= tolerance.
    symbols_data, te_start, te_end, last_close = _build_held_then_dataended_fold(
        last_bar_offset_td=2, n_fold_days=20)

    res = _run_fold_with_scorer(symbols_data, te_start, te_end, delisted_haircut=0.70)
    # NOTE: AgentSimulator._normalize_reason maps "FORCE_CLOSE" -> "MAX_HOLD".
    # DEADCO has exactly one trade (the end-of-fold close), so filter by symbol.
    fc = [t for t in res.trades if t.symbol == "DEADCO"]
    assert len(fc) == 1, f"expected one close for DEADCO, got {[t.symbol for t in res.trades]}"
    trade = fc[0]
    # Exit must be the haircut off LAST CLOSE, NOT the full last close.
    assert trade.exit_price == pytest.approx(last_close * (1.0 - 0.70))   # ~15.0
    # Entry fills near the flat price (small entry slippage at the open).
    assert trade.entry_price == pytest.approx(last_close, rel=0.01)
    # Books a deep loss (~ -70% from last close), not break-even.
    assert trade.pnl < 0
    assert trade.pnl_pct < -0.5

    # Control: with the default haircut (0.0) it is a strict no-op — the same
    # data-ended name force-closes at the FULL last close (P&L ~ 0).
    symbols_data2, te_start2, te_end2, last_close2 = _build_held_then_dataended_fold(
        last_bar_offset_td=2, n_fold_days=20)
    res0 = _run_fold_with_scorer(symbols_data2, te_start2, te_end2, delisted_haircut=0.0)
    fc0 = [t for t in res0.trades if t.symbol == "DEADCO"]
    assert len(fc0) == 1
    assert fc0[0].exit_price == pytest.approx(last_close2)                # full last close
    # ~break-even (only entry slippage + tx cost, no haircut).
    assert fc0[0].pnl_pct == pytest.approx(0.0, abs=0.02)


def test_haircut_not_applied_when_name_trades_to_fold_boundary():
    """
    FIX 1 guard: a held LONG that trades right up to the fold boundary (last bar
    on/near te_end, < 3 trading-day gap) is a NORMAL end-of-fold MTM close — it
    must NOT be haircut, even with delisted_haircut=0.70.
    """
    # Last bar on the final fold day → zero-day gap; not a delisting.
    symbols_data, te_start, te_end, last_close = _build_held_then_dataended_fold(
        last_bar_offset_td=19, n_fold_days=20)
    res = _run_fold_with_scorer(symbols_data, te_start, te_end, delisted_haircut=0.70)
    fc = [t for t in res.trades if t.symbol == "DEADCO"]
    assert len(fc) == 1
    # Closed at full last close (no haircut) → ~flat P&L.
    assert fc[0].exit_price == pytest.approx(last_close)
    assert fc[0].pnl_pct == pytest.approx(0.0, abs=0.02)


def test_delisted_haircut_constructor_clamps():
    """Constructor sanity: haircut clamped to [0,1]; default is 0.0 (no-op)."""
    from app.backtesting.agent_simulator import AgentSimulator
    assert AgentSimulator(model=None).delisted_haircut == 0.0
    assert AgentSimulator(model=None, delisted_haircut=0.70).delisted_haircut == 0.70
    assert AgentSimulator(model=None, delisted_haircut=5.0).delisted_haircut == 1.0
    assert AgentSimulator(model=None, delisted_haircut=-1.0).delisted_haircut == 0.0


def test_universe_selection_is_as_of_te_start_pit():
    """
    FIX 2: per-fold universe is the eligibility snapshot AS-OF te_start — it must
    NOT admit a name that only becomes band-eligible LATER in the test window
    (that union-over-window form is universe-membership look-ahead).
    """
    from scripts.build_smallmid_universe import (
        symbols_eligible_as_of, symbols_eligible_in_window,
    )

    te_start = date(2024, 1, 2)
    te_end = date(2024, 1, 31)
    # EARLY is eligible on te_start; LATE only becomes eligible mid-window.
    elig = pd.DataFrame({
        "date": [te_start, date(2024, 1, 15), date(2024, 1, 15)],
        "symbol": ["EARLY", "EARLY", "LATE"],
        "adv": [10_000_000.0, 10_000_000.0, 10_000_000.0],
    })

    as_of = symbols_eligible_as_of(elig, te_start)
    assert as_of == {"EARLY"}                 # only the te_start snapshot
    assert "LATE" not in as_of                # no mid-window leak (PIT)

    # The old union-over-window form WOULD have leaked LATE — documents the fix.
    window = symbols_eligible_in_window(elig, te_start, te_end)
    assert "LATE" in window


def test_eligible_as_of_uses_latest_snapshot_on_or_before():
    """as-of picks the latest eligibility day <= as_of; depends only on data <= as_of."""
    from scripts.build_smallmid_universe import symbols_eligible_as_of
    elig = pd.DataFrame({
        "date": [date(2024, 1, 5), date(2024, 1, 9), date(2024, 1, 20)],
        "symbol": ["A", "B", "C"],
        "adv": [10e6, 10e6, 10e6],
    })
    # as_of between the 9th and 20th → latest snapshot is the 9th (B); C (future) excluded.
    assert symbols_eligible_as_of(elig, date(2024, 1, 12)) == {"B"}
    # Before any eligibility day → empty (no look-back fabrication).
    assert symbols_eligible_as_of(elig, date(2024, 1, 1)) == set()


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
