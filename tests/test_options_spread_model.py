"""P2-4 — calibrated option spread cost model: binning, calibration, fallback, cost wiring."""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from app.options.spread_model import (
    CalibratedSpreadModel, calibrate, moneyness_bin, dte_bin, MIN_OBS, MONEYNESS_EDGES,
)
from app.backtesting.options_simulator import (
    OptionsSpreadCostModel, CalibratedOptionsSpreadCostModel, OptionsSimulator,
    OptionLeg, OptionPosition,
)
from app.data.options_provider import BARS_COLS


# ── binning ───────────────────────────────────────────────────────────────────
def test_moneyness_and_dte_bins():
    assert moneyness_bin(1.00) == 4          # ATM (>=0.975, <1.025)
    assert moneyness_bin(0.50) == 0          # deep ITM call / deep OTM put
    assert moneyness_bin(1.30) == len(MONEYNESS_EDGES)  # deep OTM call (>1.20)
    assert dte_bin(3) == 0 and dte_bin(10) == 1 and dte_bin(30) == 2 and dte_bin(120) == 4


# ── calibration ───────────────────────────────────────────────────────────────
def _obs(rows):
    """rows: (underlying, contract_type, moneyness, dte, spread_pct[, obs_date])."""
    recs = []
    for r in rows:
        und, ct, m, d, sp = r[:5]
        od = r[5] if len(r) > 5 else "2026-06-12"
        # supply a valid two-sided quote so the bid/ask sanity filter passes
        recs.append({"underlying": und, "contract_type": ct, "moneyness": m, "dte": d,
                     "spread_pct": sp, "bid": 1.0, "ask": 1.0 * (1 + sp), "mid": 1.0 * (1 + sp / 2),
                     "obs_date": od})
    return pd.DataFrame(recs)


def test_calibrate_medians_and_buckets():
    # 3 SPY ATM-30D calls with spreads 0.01,0.02,0.03 -> median 0.02
    df = _obs([("SPY", "call", 1.0, 30, 0.01), ("SPY", "call", 1.0, 30, 0.02),
               ("SPY", "call", 1.0, 30, 0.03)])
    m = calibrate(df)
    assert m.n_obs == 3
    key = f"SPY|call|{moneyness_bin(1.0)}|{dte_bin(30)}"
    assert m.und_buckets[key] == pytest.approx(0.02)
    assert m.und_bucket_n[key] == 3
    assert m.global_median == pytest.approx(0.02)


def test_calibrate_drops_bad_rows():
    df = _obs([("SPY", "call", 1.0, 30, 0.02)])
    # inject bad rows: NEGATIVE/crossed spread, NaN moneyness, crossed quote (ask<bid)
    df = pd.concat([df, pd.DataFrame([
        {"underlying": "SPY", "contract_type": "call", "moneyness": 1.0, "dte": 30,
         "spread_pct": -0.1, "bid": 1.0, "ask": 0.9, "mid": 0.95, "obs_date": "2026-06-12"},
        {"underlying": "SPY", "contract_type": "call", "moneyness": float("nan"), "dte": 30,
         "spread_pct": 0.05, "bid": 1.0, "ask": 1.05, "mid": 1.02, "obs_date": "2026-06-12"},
        {"underlying": "SPY", "contract_type": "call", "moneyness": 1.0, "dte": 30,
         "spread_pct": 0.05, "bid": 2.0, "ask": 1.0, "mid": 1.5, "obs_date": "2026-06-12"},
    ])], ignore_index=True)
    m = calibrate(df)
    assert m.n_obs == 1  # only the one clean row survives (locked sp==0 would be kept, see other test)


def test_calibrate_pit_filter():
    df = _obs([("SPY", "call", 1.0, 30, 0.02, "2026-06-10"),
               ("SPY", "call", 1.0, 30, 0.99, "2026-06-20")])
    m = calibrate(df, as_of=date(2026, 6, 15))
    assert m.n_obs == 1  # the 6-20 obs is excluded
    assert m.global_median == pytest.approx(0.02)


# ── fallback hierarchy ────────────────────────────────────────────────────────
def _model():
    mb, db = moneyness_bin(1.0), dte_bin(10)   # ATM, short-dated
    return CalibratedSpreadModel(
        und_buckets={f"SPY|call|{mb}|{db}": 0.02}, und_bucket_n={f"SPY|call|{mb}|{db}": MIN_OBS},
        buckets={f"call|{mb}|{db}": 0.10}, bucket_n={f"call|{mb}|{db}": MIN_OBS},
        moneyness_marginal={f"call|{mb}": 0.20}, moneyness_marginal_n={f"call|{mb}": MIN_OBS},
        dte_marginal={f"call|{db}": 0.30}, dte_marginal_n={f"call|{db}": MIN_OBS},
        global_median=0.50, conservative_global=0.80,
        calibrated_from="2026-06-11", calibrated_through="2026-06-16", n_obs=1000,
    )


def test_fallback_uses_underlying_bucket_when_available():
    m = _model()
    assert m.predict_full_spread_pct(1.0, 10, "call", "SPY") == pytest.approx(0.02)


def test_fallback_to_panel_when_underlying_thin():
    m = _model()
    # unknown underlying -> skip und bucket -> panel bucket
    assert m.predict_full_spread_pct(1.0, 10, "call", "ZZZZ") == pytest.approx(0.10)


def test_fallback_chain_marginals_then_global():
    m = _model()
    # below MIN_OBS on und + panel -> moneyness marginal
    m.und_bucket_n = {}
    m.bucket_n = {}
    assert m.predict_full_spread_pct(1.0, 10, "call", "SPY") == pytest.approx(0.20)
    m.moneyness_marginal_n = {}
    assert m.predict_full_spread_pct(1.0, 10, "call", "SPY") == pytest.approx(0.30)  # dte marginal
    m.dte_marginal_n = {}
    # last resort is the CONSERVATIVE global (p75), not the median
    assert m.predict_full_spread_pct(1.0, 10, "call", "SPY") == pytest.approx(0.80)


def test_half_spread_is_half_of_full():
    m = _model()
    assert m.half_spread_pct(1.0, 10, "call", "SPY") == pytest.approx(0.01)  # 0.02 / 2


def test_missing_context_returns_conservative_global():
    m = _model()
    # missing/NaN context -> conservative global (high), never the optimistic median
    assert m.predict_full_spread_pct(None, None, None) == pytest.approx(0.80)
    assert m.predict_full_spread_pct(float("nan"), 10, "call", "SPY") == pytest.approx(0.80)


def test_covers_date_anachronism_guard():
    m = _model()
    assert m.covers_date(date(2026, 6, 12)) is True
    assert m.covers_date(date(2022, 1, 3)) is False     # before the calibration window
    assert m.covers_date(date(2026, 7, 1)) is False     # after
    assert CalibratedSpreadModel().covers_date(date(2026, 6, 12)) is False  # empty model


def test_save_load_roundtrip(tmp_path):
    m = _model()
    p = tmp_path / "m.json"
    m.save(p)
    m2 = CalibratedSpreadModel.load(p)
    assert m2.predict_full_spread_pct(1.0, 10, "call", "SPY") == pytest.approx(0.02)
    assert m2.global_median == pytest.approx(0.50)
    assert m2.conservative_global == pytest.approx(0.80)
    assert m2.calibrated_from == "2026-06-11"


def test_load_rejects_mismatched_bin_edges(tmp_path):
    import json as _json
    m = _model()
    p = tmp_path / "m.json"
    m.save(p)
    d = _json.loads(p.read_text())
    d["moneyness_edges"] = [0.5, 1.5]   # edges changed since save -> indices now meaningless
    p.write_text(_json.dumps(d))
    with pytest.raises(ValueError):
        CalibratedSpreadModel.load(p)


def test_calibrate_keeps_locked_quotes_and_uses_p75_global():
    # sp==0 (locked ask==bid) must be KEPT (it's a real tight quote on a liquid name).
    df = _obs([("SPY", "call", 1.0, 30, 0.0), ("SPY", "call", 1.0, 30, 0.0),
               ("SPY", "call", 1.0, 30, 0.40)])
    # fix the locked rows so bid==ask passes the sanity filter (a>=b, mid>0)
    df.loc[df["spread_pct"] == 0.0, ["bid", "ask", "mid"]] = 1.0
    m = calibrate(df)
    assert m.n_obs == 3                       # locked quotes retained
    assert m.global_median == pytest.approx(0.0)   # median of [0, 0, 0.40]
    assert m.conservative_global > m.global_median  # p75 > median (conservative)


# ── cost models ───────────────────────────────────────────────────────────────
def test_calibrated_cost_uses_half_spread_and_fee():
    m = _model()
    cm = CalibratedOptionsSpreadCostModel(model=m, per_contract_fee=0.65)
    # SPY ATM 10D: half=0.01 -> 2.0 * 0.01 * 1 * 100 + 0.65 = 2.65
    assert cm.entry_exit_cost(2.0, 1.0, moneyness=1.0, dte=10, contract_type="call",
                              underlying="SPY") == pytest.approx(2.65)
    # stress 2x doubles the spread part only
    assert cm.entry_exit_cost(2.0, 2.0, moneyness=1.0, dte=10, contract_type="call",
                              underlying="SPY") == pytest.approx(4.65)
    # missing context -> CONSERVATIVE global half (0.80/2=0.40): 2.0*0.40*100 + 0.65 = 80.65
    assert cm.entry_exit_cost(2.0, 1.0) == pytest.approx(80.65)


def test_flat_cost_model_ignores_context():
    cm = OptionsSpreadCostModel(spread_pct=0.01, per_contract_fee=0.65)
    base = cm.entry_exit_cost(5.0, 1.0)
    with_ctx = cm.entry_exit_cost(5.0, 1.0, moneyness=1.3, dte=60, contract_type="put",
                                  underlying="XYZ")
    assert base == with_ctx == pytest.approx(5.65)


# ── simulator wiring: context is computed + passed ────────────────────────────
class _RecordingCost:
    def __init__(self):
        self.calls = []

    def entry_exit_cost(self, premium, spread_mult=1.0, *, moneyness=None, dte=None,
                        contract_type=None, underlying=None):
        self.calls.append({"premium": premium, "moneyness": moneyness, "dte": dte,
                           "contract_type": contract_type, "underlying": underlying})
        return 0.0


def test_simulator_passes_contract_context_to_cost_model():
    C = "O:SPY251219C00100000"  # strike 100, expiry 2025-12-19, call, SPY
    bars = pd.DataFrame([{"underlying": "SPY", "contract": C, "date": pd.Timestamp("2025-12-01"),
                          "open": 5.0, "high": 5.0, "low": 5.0, "close": 5.0, "volume": 100.0,
                          "knowable_date": pd.Timestamp("2025-12-01")}], columns=BARS_COLS)
    und = {"SPY": {date(2025, 12, 1): 105.0, date(2025, 12, 19): 110.0}}
    rec = _RecordingCost()
    sim = OptionsSimulator(bars, underlying_prices=und, cost_model=rec, starting_capital=100_000)
    sim.run([OptionPosition([OptionLeg(C, +1, 1)], entry_date=date(2025, 12, 1))],
            date(2025, 12, 1), date(2025, 12, 19))
    assert rec.calls, "cost model was never called"
    c = rec.calls[0]
    assert c["contract_type"] == "call" and c["underlying"] == "SPY"
    assert c["moneyness"] == pytest.approx(100.0 / 105.0)  # strike/spot on entry
    assert c["dte"] == (date(2025, 12, 19) - date(2025, 12, 1)).days
