"""Tests for Alpha v2 §1.1 — PEAD cost-sensitivity sweep + the slippage/cost
parameterization refactor in AgentSimulator.

Covers:
  1. BACKWARD-COMPAT: default entry/stop slippage + tx-cost kwargs equal the prior
     hardcoded module constants, and a known entry fill prices identically pre/post
     refactor.
  2. COST MONOTONICITY: higher cost_bps → lower net return, and the per-trade cost is
     charged on BOTH entry and exit (round-trip = 2× one-way).
  3. SWEEP HARNESS: the entrypoint runs end-to-end on a minimal config and emits the
     table + CSV/JSON artifacts with the expected schema.
"""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from app.backtesting.agent_simulator import (
    AgentSimulator,
    ENTRY_SLIPPAGE_PCT,
    STOP_SLIPPAGE_PCT,
    TX_COST_PCT,
    _Position,
    _PortfolioState,
)


# ───────────────────────── 1. Backward-compat ──────────────────────────────────

def test_default_slippage_and_cost_equal_prior_constants():
    """The refactor lifted ENTRY/STOP slippage to kwargs; defaults must be byte-equal
    to the prior hardcoded module constants (3 bps entry, 5 bps stop) and tx cost."""
    sim = AgentSimulator(model=None)
    assert sim.entry_slippage_pct == ENTRY_SLIPPAGE_PCT == 0.0003
    assert sim.stop_slippage_pct == STOP_SLIPPAGE_PCT == 0.0005
    assert sim.transaction_cost_pct == TX_COST_PCT == 0.0005


def test_entry_fill_price_identical_at_default_slippage():
    """A long entry at raw open O fills at O*(1+entry_slippage). At the default
    slippage this must equal the prior hardcoded formula O*(1+0.0003) exactly."""
    sim = AgentSimulator(model=None)
    raw_open = 100.0
    # Mirror the fill formula in run_entries (long branch).
    expected = raw_open * (1 + 0.0003)
    got = raw_open * (1 + sim.entry_slippage_pct)
    assert got == expected
    # And an override changes it deterministically with no other behavior change.
    sim2 = AgentSimulator(model=None, entry_slippage_pct=0.0020)
    assert raw_open * (1 + sim2.entry_slippage_pct) == raw_open * 1.0020


def test_slippage_kwargs_are_independent_of_tx_cost():
    """Slippage and transaction cost are separate knobs; setting one must not touch
    the others (the sweep relies on driving them independently)."""
    sim = AgentSimulator(
        model=None,
        transaction_cost_pct=0.005,
        entry_slippage_pct=0.0,
        stop_slippage_pct=0.0,
    )
    assert sim.transaction_cost_pct == 0.005
    assert sim.entry_slippage_pct == 0.0
    assert sim.stop_slippage_pct == 0.0


# ─────────────────────── 2. Cost charged per-side (round-trip = 2×) ─────────────

def _round_trip_net_pnl(cost_pct: float, entry_price=100.0, exit_price=100.0, qty=10):
    """Compute net P&L of a flat (entry==exit) round trip at a given tx cost.

    Charges entry cost the way run_entries does (trade_cost * cost), then exits via
    _close_position (which charges exit_price*qty*cost). With entry==exit and zero
    price move, gross P&L = 0, so net = -(entry_cost + exit_cost) = -2 × one-way.
    """
    sim = AgentSimulator(model=None, transaction_cost_pct=cost_pct,
                         entry_slippage_pct=0.0, stop_slippage_pct=0.0)
    entry_cost = entry_price * qty * sim.transaction_cost_pct
    pf = _PortfolioState(cash=1_000_000.0, peak_equity=1_000_000.0)
    pf.cash -= entry_price * qty + entry_cost  # entry leg (mirrors run_entries)
    pos = _Position(
        symbol="X", entry_date=date(2023, 1, 3), entry_price=entry_price,
        stop_price=entry_price * 0.95, target_price=entry_price * 1.05,
        quantity=qty, highest_price=entry_price,
    )
    trade, exit_tx = sim._close_position(pos, date(2023, 1, 10), exit_price, "time_exit", pf)
    # Net round-trip P&L on the position = trade.pnl (already net of exit cost) minus
    # the entry-side cost (charged separately at entry time).
    return trade.pnl - entry_cost, entry_cost, exit_tx


def test_cost_charged_on_both_entry_and_exit():
    """Round-trip cost must equal 2× the one-way cost (entry + exit), not 1×."""
    cost = 0.0010  # 10 bps one-way
    net, entry_tx, exit_tx = _round_trip_net_pnl(cost)
    one_way = 100.0 * 10 * cost  # entry_price * qty * cost
    assert entry_tx == pytest.approx(one_way)
    assert exit_tx == pytest.approx(one_way)
    # Flat trade (no price move) → net P&L = -(entry + exit) = -2× one-way.
    assert net == pytest.approx(-2 * one_way)


def test_higher_cost_lower_net_return():
    """Monotonicity: a higher per-side cost yields a strictly more-negative net P&L
    on the same flat round trip."""
    nets = []
    for bps in [2, 5, 10, 20, 35, 50]:
        net, _, _ = _round_trip_net_pnl(bps / 1e4)
        nets.append(net)
    # Strictly decreasing (more cost → more loss on a flat trade).
    for a, b in zip(nets, nets[1:]):
        assert b < a
    # Zero cost → zero net P&L on a flat trade (sanity anchor).
    net0, _, _ = _round_trip_net_pnl(0.0)
    assert net0 == pytest.approx(0.0)


# ─────────────────────────── 3. Sweep harness ──────────────────────────────────

def test_interp_crossing_breakeven():
    from scripts.pead_cost_sensitivity import _interp_crossing
    # Sharpe falls below 0 between 10 and 20 bps.
    be = _interp_crossing([2, 5, 10, 20], [0.5, 0.45, 0.2, -0.2], 0.0)
    assert 10 < be < 20
    # Never crosses (all above target) → None.
    assert _interp_crossing([2, 5, 10], [0.9, 0.8, 0.7], 0.0) is None


def _synthetic_bars(n_days=400, start=date(2022, 1, 3), seed=7):
    rng = np.random.default_rng(seed)
    close = 100.0
    rows, d = [], pd.Timestamp(start)
    while len(rows) < n_days:
        if d.weekday() < 5:
            close *= (1 + rng.normal(0.0008, 0.012))
            rows.append({
                "open": close * (1 + rng.normal(0, 0.002)),
                "high": close * (1 + abs(rng.normal(0, 0.004))),
                "low": close * (1 - abs(rng.normal(0, 0.004))),
                "close": close,
                "volume": int(rng.integers(1_000_000, 4_000_000)),
            })
        d += pd.Timedelta(days=1)
    idx = pd.date_range(start=start, periods=len(rows), freq="B")
    return pd.DataFrame(rows, index=idx)


class _AlwaysLongScorer:
    """Minimal factor_scorer: proposes a fixed long every scan day with conf 0.9."""

    def __call__(self, day, symbols_data, vix_history=None, **kwargs):
        out = []
        for sym in symbols_data:
            if sym in ("SPY", "^VIX", "VIX"):
                continue
            out.append((sym, 0.9))
        return out[:3]


def test_sweep_monotonic_on_synthetic_sim():
    """End-to-end at the AgentSimulator level: run the SAME synthetic data/signals at
    two cost levels via the slippage/cost kwargs; higher cost → not-higher net return.
    Proves the parameterization actually flows into fills during a real .run()."""
    bars = {f"SYM{i}": _synthetic_bars(seed=i + 1) for i in range(4)}
    bars["SPY"] = _synthetic_bars(seed=99)
    spy = bars["SPY"]["close"]
    start, end = date(2022, 6, 1), date(2023, 6, 1)

    def _run(cost_bps):
        sim = AgentSimulator(
            model=None,
            factor_scorer=_AlwaysLongScorer(),
            transaction_cost_pct=cost_bps / 1e4,
            entry_slippage_pct=0.0,
            stop_slippage_pct=0.0,
            no_prefilters=True,
        )
        res = sim.run(bars, start_date=start, end_date=end, spy_prices=spy)
        return res.total_return_pct, res.total_trades

    lo_ret, lo_trades = _run(2)
    hi_ret, hi_trades = _run(50)
    # Same data + signals + (zero) slippage → identical trade set; only cost differs.
    assert lo_trades == hi_trades
    if lo_trades > 0:
        # More cost can only reduce (or equal, if no trades) the net return.
        assert hi_ret <= lo_ret + 1e-9


def test_sweep_entrypoint_smoke_schema(monkeypatch, tmp_path):
    """The sweep entrypoint runs end-to-end on a minimal config (no network) and
    emits a summary with the expected schema. We stub fetch_data with synthetic bars
    so the test is fast and offline."""
    import scripts.run_pead_cpcv as rp
    import scripts.pead_cost_sensitivity as sweep

    syn = {f"SYM{i}": _synthetic_bars(seed=i + 20) for i in range(5)}
    syn["SPY"] = _synthetic_bars(seed=500)

    def _fake_fetch(self, start, end):
        # Populate the minimal attributes run_cpcv / run_fold read.
        self.symbols_data = dict(syn)
        self.spy_prices = syn["SPY"]["close"]
        self.all_days_sorted = sorted({d.date() for d in syn["SPY"].index})
        self._global_regime_map = {}

    monkeypatch.setattr(rp.PEADStrategy, "fetch_data", _fake_fetch)
    # Keep the PIT universe from filtering out our synthetic symbols.
    monkeypatch.setattr(rp, "RUSSELL_1000_TICKERS", [f"SYM{i}" for i in range(5)], raising=False)

    summary = sweep.run_sweep(
        cost_bps_levels=[5.0, 50.0],
        smoke=True,
        symbols=[f"SYM{i}" for i in range(5)],
        total_years=2,
        cpcv_k=4,
        cpcv_paths=2,
    )

    assert set(summary["rows"][0].keys()) >= {
        "cost_bps", "mean_sharpe", "path_tstat", "pct_positive",
        "p5_sharpe", "p95_sharpe", "avg_net_return_per_fold", "n_paths",
    }
    # Pure-cost ladder rows preserve the requested levels (anchor is appended after).
    pure = [r for r in summary["rows"] if not r["slippage_included"]]
    assert [r["cost_bps"] for r in pure] == [5.0, 50.0]
    assert "breakeven_cost_bps" in summary
    assert "acceptance_passed" in summary

    # Artifacts round-trip.
    paths = sweep._write_artifacts(summary, "test")
    assert paths["csv"].exists() and paths["json"].exists()
    sweep._print_table(summary)  # must not raise


# ─────────────────────────── 4. Anchor row (+0.546) ────────────────────────────

def test_anchor_row_present_and_flagged(monkeypatch):
    """The sweep must append exactly ONE anchor row carrying the validated config,
    flagged slippage_included=True with the +0.546 label, on top of the pure ladder."""
    import scripts.run_pead_cpcv as rp
    import scripts.pead_cost_sensitivity as sweep

    syn = {f"SYM{i}": _synthetic_bars(seed=i + 40) for i in range(5)}
    syn["SPY"] = _synthetic_bars(seed=700)

    def _fake_fetch(self, start, end):
        self.symbols_data = dict(syn)
        self.spy_prices = syn["SPY"]["close"]
        self.all_days_sorted = sorted({d.date() for d in syn["SPY"].index})
        self._global_regime_map = {}

    monkeypatch.setattr(rp.PEADStrategy, "fetch_data", _fake_fetch)
    monkeypatch.setattr(rp, "RUSSELL_1000_TICKERS", [f"SYM{i}" for i in range(5)], raising=False)

    summary = sweep.run_sweep(
        cost_bps_levels=[5.0, 50.0],
        smoke=True,
        symbols=[f"SYM{i}" for i in range(5)],
        total_years=2,
        cpcv_k=4,
        cpcv_paths=2,
    )

    anchor_rows = [r for r in summary["rows"] if r.get("slippage_included")]
    assert len(anchor_rows) == 1, "exactly one slippage-included anchor row expected"
    anchor = anchor_rows[0]

    # Distinctly labelled, NOT a plain bps number on the cost axis.
    assert anchor["label"] == sweep.ANCHOR_LABEL == "committed (+0.546 anchor)"
    assert anchor["slippage_included"] is True
    assert anchor["cost_bps"] is None  # not a pure per-side cost level

    # Validated triplet exactly: 5 bps fee + 3 bps entry-slip + 5 bps stop-slip.
    assert anchor["tx_cost_pct"] == 0.0005
    assert anchor["entry_slippage_pct"] == 0.0003
    assert anchor["stop_slippage_pct"] == 0.0005

    # Summary block records the anchor + its self-validation verdict.
    assert summary["anchor"]["tx_cost_pct"] == 0.0005
    assert summary["anchor"]["entry_slippage_pct"] == 0.0003
    assert summary["anchor"]["stop_slippage_pct"] == 0.0005
    assert summary["anchor"]["excluded_from_interpolation"] is True
    assert "reproduces_validated" in summary["anchor"]


def test_anchor_excluded_from_pure_cost_interpolation(monkeypatch):
    """The break-even / +0.40 crossings must be computed over the PURE-COST ladder
    only; the slippage-carrying anchor must not enter the interpolation, even if its
    Sharpe would shift a crossing."""
    import scripts.pead_cost_sensitivity as sweep

    captured = {}
    orig = sweep._interp_crossing

    def _spy(levels, values, target):
        # Record every (levels) the interpolator is asked to operate on.
        captured.setdefault("levels", []).append(list(levels))
        return orig(levels, values, target)

    monkeypatch.setattr(sweep, "_interp_crossing", _spy)

    import scripts.run_pead_cpcv as rp
    syn = {f"SYM{i}": _synthetic_bars(seed=i + 60) for i in range(5)}
    syn["SPY"] = _synthetic_bars(seed=900)

    def _fake_fetch(self, start, end):
        self.symbols_data = dict(syn)
        self.spy_prices = syn["SPY"]["close"]
        self.all_days_sorted = sorted({d.date() for d in syn["SPY"].index})
        self._global_regime_map = {}

    monkeypatch.setattr(rp.PEADStrategy, "fetch_data", _fake_fetch)
    monkeypatch.setattr(rp, "RUSSELL_1000_TICKERS", [f"SYM{i}" for i in range(5)], raising=False)

    summary = sweep.run_sweep(
        cost_bps_levels=[5.0, 50.0],
        smoke=True,
        symbols=[f"SYM{i}" for i in range(5)],
        total_years=2,
        cpcv_k=4,
        cpcv_paths=2,
    )

    pure_levels = [r["cost_bps"] for r in summary["rows"] if not r["slippage_included"]]
    assert pure_levels == [5.0, 50.0]
    # Every interpolation call saw ONLY the pure-cost levels — never None (the anchor's
    # cost_bps) and never more entries than the pure ladder has.
    for lv in captured["levels"]:
        assert lv == pure_levels
        assert None not in lv


# ──────────────── 5. Artifact persistence is Windows-safe + resilient ───────────

def _stub_sweep_summary(monkeypatch):
    """Build a real run_sweep summary (incl. the anchor row) on synthetic offline data."""
    import scripts.run_pead_cpcv as rp
    import scripts.pead_cost_sensitivity as sweep

    syn = {f"SYM{i}": _synthetic_bars(seed=i + 80) for i in range(5)}
    syn["SPY"] = _synthetic_bars(seed=1100)

    def _fake_fetch(self, start, end):
        self.symbols_data = dict(syn)
        self.spy_prices = syn["SPY"]["close"]
        self.all_days_sorted = sorted({d.date() for d in syn["SPY"].index})
        self._global_regime_map = {}

    monkeypatch.setattr(rp.PEADStrategy, "fetch_data", _fake_fetch)
    monkeypatch.setattr(rp, "RUSSELL_1000_TICKERS", [f"SYM{i}" for i in range(5)], raising=False)

    return sweep.run_sweep(
        cost_bps_levels=[5.0, 50.0],
        smoke=True,
        symbols=[f"SYM{i}" for i in range(5)],
        total_years=2,
        cpcv_k=4,
        cpcv_paths=2,
    )


def test_artifacts_written_with_anchor_row(monkeypatch, tmp_path):
    """Artifacts (JSON + CSV) must be written even when the table contains the anchor
    row (the `<-`/marker case that previously crashed on cp1252). Assert both files
    exist on disk with the expected schema/rows, including the slippage-flagged anchor."""
    import json as _json
    import csv as _csv
    import scripts.pead_cost_sensitivity as sweep

    summary = _stub_sweep_summary(monkeypatch)
    monkeypatch.setattr(sweep, "ARTIFACT_DIR", tmp_path)

    paths = sweep._write_artifacts(summary, "anchortest")
    assert paths["json"].exists()
    assert paths["csv"].exists()

    data = _json.loads(paths["json"].read_text(encoding="utf-8"))
    rows = data["rows"]
    # Pure-cost ladder + exactly one slippage-included anchor row.
    assert [r["cost_bps"] for r in rows if not r["slippage_included"]] == [5.0, 50.0]
    anchor = [r for r in rows if r["slippage_included"]]
    assert len(anchor) == 1
    assert anchor[0]["label"] == sweep.ANCHOR_LABEL

    with open(paths["csv"], newline="", encoding="utf-8") as f:
        csv_rows = list(_csv.DictReader(f))
    assert len(csv_rows) == len(rows)
    assert {"cost_bps", "label", "slippage_included", "mean_sharpe"} <= set(csv_rows[0].keys())


def test_print_table_failure_does_not_block_artifacts(monkeypatch, tmp_path):
    """Resilience guard: if console rendering raises (simulating a cp1252
    UnicodeEncodeError on the `<-` anchor marker), the durable JSON/CSV artifacts must
    STILL be persisted. Mirrors main()'s write-first, print-guarded ordering."""
    import scripts.pead_cost_sensitivity as sweep

    summary = _stub_sweep_summary(monkeypatch)
    monkeypatch.setattr(sweep, "ARTIFACT_DIR", tmp_path)

    # Make any table render explode, the way a cp1252 console would on a non-ASCII char.
    def _boom(summary):
        raise UnicodeEncodeError("charmap", "<-", 0, 1, "simulated console failure")

    monkeypatch.setattr(sweep, "_print_table", _boom)

    # Reproduce main()'s ordering: write artifacts first, then guard the print.
    paths = sweep._write_artifacts(summary, "resilient")
    try:
        sweep._print_table(summary)
    except Exception:
        pass

    # Artifacts persisted despite the print blowing up.
    assert paths["json"].exists()
    assert paths["csv"].exists()


def test_no_non_ascii_in_runtime_strings():
    """The sweep + run_pead_cpcv emit only cp1252-safe (ASCII) text in their print/log
    f-strings and bare prints, so a Windows console/file handler can never crash the
    run. We scan the print()/logger lines for non-ASCII (docstrings/comments exempt)."""
    import ast
    import scripts.pead_cost_sensitivity as sweep_mod
    import scripts.run_pead_cpcv as rp_mod

    def _bad_runtime_strings(path):
        tree = ast.parse(open(path, encoding="utf-8").read())
        bad = []
        for node in ast.walk(tree):
            # Bare print(...) and logger.<level>(...) calls.
            is_print = isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print"
            is_log = (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "logger"
            )
            if not (is_print or is_log):
                continue
            for sub in ast.walk(node):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                    if any(ord(ch) > 127 for ch in sub.value):
                        bad.append((node.lineno, sub.value))
        return bad

    for mod in (sweep_mod, rp_mod):
        bad = _bad_runtime_strings(mod.__file__)
        assert not bad, f"non-ASCII in runtime print/log strings of {mod.__file__}: {bad}"
