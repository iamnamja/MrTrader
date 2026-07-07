"""R1.3 — futures_signal.current_target_weights: extract the live carry+xsmom book target weights,
map to canonical instrument_ids, filter to the IBKR-tradeable universe. Never raises."""
import numpy as np
import pandas as pd

from app.live_trading import futures_signal as fs


def _panel_last_row_weights():
    # 2 rows; the LAST row is the target the extractor should read.
    return pd.DataFrame(
        {"ES": [0.10, 0.20], "GC": [0.00, -0.10], "NOPE": [0.00, 0.05]},
        index=pd.to_datetime(["2026-06-16", "2026-06-17"]))


def test_current_target_weights_combines_maps_and_filters(monkeypatch):
    from app.research import futures_carry as fc
    from app.research import futures_data as fd
    from app.research import futures_factors as ff
    idx = pd.to_datetime(["2026-06-16", "2026-06-17"])
    panel = _panel_last_row_weights()
    # DISTINCT carry vs xsmom weight panels so the 0.5/0.5 average is genuinely exercised (not just
    # "take one sleeve's last row"): last rows carry {ES:0.2, GC:0.0}, xsmom {ES:0.4, GC:-0.2, NOPE:0.1}
    # → combined {ES:0.3, GC:-0.1, NOPE:0.05}; NOPE non-tradeable → dropped.
    carry_w = pd.DataFrame({"ES": [0.1, 0.2], "GC": [0.0, 0.0], "NOPE": [0.0, 0.0]}, index=idx)
    xsmom_w = pd.DataFrame({"ES": [0.3, 0.4], "GC": [0.0, -0.2], "NOPE": [0.0, 0.1]}, index=idx)
    monkeypatch.setattr(fd, "liquid_universe", lambda **k: ["ES", "GC", "NOPE"])
    monkeypatch.setattr(fd, "returns_panel", lambda uni, **k: panel)
    monkeypatch.setattr(fd, "synthetic_price_panel", lambda uni, **k: panel)
    monkeypatch.setattr(fc, "carry_panel", lambda uni, **k: panel)
    monkeypatch.setattr(ff, "xs_momentum_signal", lambda prices, **k: panel)
    calls = {"n": 0}

    def _bt(returns, sig, cfg, return_weights=False):
        calls["n"] += 1
        return carry_w if calls["n"] == 1 else xsmom_w      # 1st call = carry, 2nd = xsmom
    monkeypatch.setattr(fc, "carry_backtest", _bt)
    out = fs.current_target_weights()
    assert out == {"FUT.ES": 0.3, "FUT.GC": -0.1}           # 0.5·carry + 0.5·xsmom, NOPE dropped


def test_current_target_weights_drops_dust(monkeypatch):
    from app.research import futures_carry as fc
    from app.research import futures_data as fd
    from app.research import futures_factors as ff
    tiny = pd.DataFrame({"ES": [0.0, 0.004], "GC": [0.0, 0.5]},
                        index=pd.to_datetime(["2026-06-16", "2026-06-17"]))
    monkeypatch.setattr(fd, "liquid_universe", lambda **k: ["ES", "GC"])
    monkeypatch.setattr(fd, "returns_panel", lambda uni, **k: tiny)
    monkeypatch.setattr(fd, "synthetic_price_panel", lambda uni, **k: tiny)
    monkeypatch.setattr(fc, "carry_panel", lambda uni, **k: tiny)
    monkeypatch.setattr(ff, "xs_momentum_signal", lambda prices, **k: tiny)
    monkeypatch.setattr(fc, "carry_backtest", lambda r, s, c, return_weights=False: tiny)
    out = fs.current_target_weights(min_weight=0.01)
    assert out == {"FUT.GC": 0.5}                    # ES 0.004 < min_weight → dropped


def test_current_target_weights_empty_and_error_degrade_to_empty(monkeypatch):
    from app.research import futures_data as fd
    monkeypatch.setattr(fd, "liquid_universe", lambda **k: ["ES"])
    monkeypatch.setattr(fd, "returns_panel", lambda uni, **k: pd.DataFrame())
    assert fs.current_target_weights() == {}         # empty returns → {}

    def _boom(*a, **k):
        raise RuntimeError("data feed down")
    monkeypatch.setattr(fd, "liquid_universe", _boom)
    assert fs.current_target_weights() == {}         # any error → {} (never raises)


def test_carry_backtest_return_weights_gives_a_panel():
    from app.research import futures_carry as fc
    n, mkts = 200, ["ES", "GC", "CL", "ZC", "ZS", "NQ"]     # >= min_xs_width(5), > vol_lookback(60)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    rng = np.arange(n)
    returns = pd.DataFrame({m: 0.001 * np.sin(rng / (5 + i)) for i, m in enumerate(mkts)}, index=idx)
    carry = pd.DataFrame({m: 0.01 * np.cos(rng / (7 + i)) for i, m in enumerate(mkts)}, index=idx)
    ret_series = fc.carry_backtest(returns, carry, fc.CarryConfig())
    W = fc.carry_backtest(returns, carry, fc.CarryConfig(), return_weights=True)
    assert isinstance(ret_series, pd.Series)               # default path unchanged
    assert isinstance(W, pd.DataFrame) and list(W.columns) == mkts
    assert W.abs().to_numpy().sum() > 0                     # real (non-degenerate) weights produced


def test_signal_live_flag_registered_and_settable(db_session):
    from app.database.agent_config import get_agent_config, set_agent_config
    assert get_agent_config(db_session, "ibkr.futures_signal_live") == "false"   # default OFF
    set_agent_config(db_session, "ibkr.futures_signal_live", "true")
    assert get_agent_config(db_session, "ibkr.futures_signal_live") == "true"
