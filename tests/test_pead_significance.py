"""Tests for Alpha v2 §1.3 — PEAD honest significance harness (scripts/pead_significance.py)
and its pure-additive _last_trades instrumentation in run_pead_cpcv.PEADStrategy.

Covers (per the §1.3 spec):
  1. EVENT-CLUSTERED BLOCK BOOTSTRAP behaves: strong synthetic edge -> p<0.05; pure noise
     -> p~=0.5; clustered CI is WIDER than an iid CI on autocorrelated/clustered input.
  2. NEWEY-WEST HAC: matches a hand-checked reference value on a small series; HAC t < OLS t
     on positively-autocorrelated returns.
  3. STRATIFICATION: trades bucketed by SPY><200d are PIT (a future SPY move does not change
     a past trade's regime tag); cluster key = calendar quarter.
  4. DETERMINISM: same seed -> identical bootstrap result.
  5. BASELINE self-validation reconciles with +0.546 (on the smoke schema).
  6. PURE-ADDITIVE instrumentation: _last_trades does not change CPCV results; full run
     emits the expected schema + artifacts.
  7. ASCII-safe runtime strings.
"""
from datetime import date

import math
import numpy as np
import pandas as pd
import pytest


# ───────────────────────── 1. Event-clustered block bootstrap ──────────────────────

def test_bootstrap_strong_edge_is_significant():
    """A strong, consistent positive edge -> bootstrap p (H0 Sharpe<=0) well below 0.05."""
    from scripts.pead_significance import event_clustered_bootstrap
    rng = np.random.default_rng(0)
    # 12 clusters, 20 trades each, mean +1% sd 1% -> trade Sharpe ~ 1.0 (very significant).
    rets, clus = [], []
    for q in range(12):
        for _ in range(20):
            rets.append(float(rng.normal(0.01, 0.01)))
            clus.append(f"2020Q{q % 4 + 1}_{q}")
    out = event_clustered_bootstrap(rets, clus, n_resamples=2000, seed=7)
    assert out["observed_sharpe"] > 0
    assert out["p_value"] < 0.05
    assert out["ci_low"] > 0  # CI excludes 0 for a strong edge


def test_bootstrap_pure_noise_is_not_significant():
    """Zero-mean noise -> observed Sharpe ~0 and bootstrap p ~0.5 (not significant)."""
    from scripts.pead_significance import event_clustered_bootstrap
    # Seed 5 draws genuinely-centered noise (observed Sharpe ~0) so the re-centered null
    # p lands near 0.5 — the spec's "pure noise -> p~=0.5" behavior. (With a sample whose
    # observed Sharpe happens positive by chance, the p would correctly read smaller; the
    # statistic is honest, this seed just isolates the truly-zero case.)
    rng = np.random.default_rng(5)
    rets, clus = [], []
    for q in range(20):
        for _ in range(25):
            rets.append(float(rng.normal(0.0, 0.02)))
            clus.append(f"C{q}")
    out = event_clustered_bootstrap(rets, clus, n_resamples=2000, seed=11)
    assert abs(out["observed_sharpe"]) < 0.05
    assert 0.35 < out["p_value"] < 0.65  # ~0.5 for centered noise


def test_clustering_widens_ci_vs_iid_on_clustered_input():
    """On input where the edge is concentrated in a few clusters (strong within-cluster
    correlation), resampling WHOLE clusters must yield a WIDER CI than the iid per-trade
    bootstrap (which ignores the dependence and understates uncertainty)."""
    from scripts.pead_significance import event_clustered_bootstrap, iid_bootstrap_ci
    rng = np.random.default_rng(5)
    rets, clus = [], []
    # 8 clusters; each cluster has a strong COMMON shift (all its trades move together),
    # so the real independent unit is the cluster, not the trade.
    for q in range(8):
        shift = float(rng.normal(0.0, 0.03))  # cluster-level common component
        for _ in range(30):
            rets.append(shift + float(rng.normal(0.0, 0.002)))
            clus.append(f"Q{q}")
    clustered = event_clustered_bootstrap(rets, clus, n_resamples=3000, seed=21)
    iid = iid_bootstrap_ci(rets, n_resamples=3000, seed=21)
    assert clustered["ci_width"] > iid["ci_width"], \
        "clustered CI must be wider than iid CI on cluster-correlated input"


def test_bootstrap_handles_degenerate_input():
    from scripts.pead_significance import event_clustered_bootstrap
    out = event_clustered_bootstrap([0.01], ["A"], n_resamples=100, seed=1)
    assert out["n_resamples"] == 0 and out["p_value"] == 1.0


# ──────────────── 1b. Textbook return-shift bootstrap null (the §1.3 upgrade) ───────

def test_bootstrap_textbook_null_noise_calibrated():
    """Centered noise -> the textbook return-shift null gives p ~ 0.5."""
    from scripts.pead_significance import event_clustered_bootstrap
    rng = np.random.default_rng(5)
    rets, clus = [], []
    for q in range(20):
        for _ in range(25):
            rets.append(float(rng.normal(0.0, 0.02)))
            clus.append(f"C{q}")
    out = event_clustered_bootstrap(rets, clus, n_resamples=2000, seed=11)
    assert abs(out["observed_sharpe"]) < 0.05
    assert 0.35 < out["p_value"] < 0.65


def test_bootstrap_textbook_null_strong_edge_significant():
    """A strong synthetic edge -> the textbook return-shift null gives p < 0.05."""
    from scripts.pead_significance import event_clustered_bootstrap
    rng = np.random.default_rng(0)
    rets, clus = [], []
    for q in range(12):
        for _ in range(20):
            rets.append(float(rng.normal(0.01, 0.01)))
            clus.append(f"2020Q{q % 4 + 1}_{q}")
    out = event_clustered_bootstrap(rets, clus, n_resamples=2000, seed=7)
    assert out["observed_sharpe"] > 0
    assert out["p_value"] < 0.05


def test_bootstrap_ci_uses_unshifted_returns():
    """The percentile CI must be computed on the UN-shifted resamples: for a strong positive
    edge the CI brackets the (positive) observed Sharpe, not zero."""
    from scripts.pead_significance import event_clustered_bootstrap
    rng = np.random.default_rng(3)
    rets, clus = [], []
    for q in range(12):
        for _ in range(20):
            rets.append(float(rng.normal(0.01, 0.01)))
            clus.append(f"Q{q}")
    out = event_clustered_bootstrap(rets, clus, n_resamples=2000, seed=7)
    # CI is about the actual (un-shifted) statistic, so it sits around the positive observed.
    assert out["ci_low"] > 0
    assert out["ci_low"] <= out["observed_sharpe"] <= out["ci_high"]


# ──────── 1c. Daily-series de-dup by CALENDAR WINDOW (the CRITICAL fold-overlap bug) ──

from types import SimpleNamespace


def _curve(start: date, n: int, base: float = 100000.0, step: float = 100.0):
    """A simple equity curve of n daily points starting at `start` (business days)."""
    idx = pd.date_range(start=pd.Timestamp(start), periods=n, freq="B")
    return [(ts.date(), base + i * step) for i, ts in enumerate(idx)]


def test_daily_series_dedups_overlapping_fold_ids_to_each_day_once():
    """REGRESSION for the fold-overlap bug. run_cpcv gives DISTINCT global fold ids to the
    SAME calendar test window across combos (global id = combo_idx*K + ti + 1). For PEAD the
    captured curve for a given ti is byte-identical across combos, so the SAME window appears
    under many fold ids. The OLD by-fold-id concat repeated each calendar day ~Kx; the FIXED
    by-window dedup must yield each calendar day EXACTLY ONCE."""
    from scripts.pead_significance import _pool_unique_daily_returns, validate_daily_series

    K = 4  # CPCV k -> global id = combo_idx*K + ti + 1
    # Two disjoint calendar windows (ti=1 and ti=2), each appearing under MANY combos.
    win_ti1 = _curve(date(2022, 1, 3), 6)       # window A
    win_ti2 = _curve(date(2022, 6, 1), 6)       # window B (disjoint, later)
    fold_curves = {}
    members = []
    for combo_idx in range(6):  # window A & B each captured under 6 distinct global ids
        gid_a = combo_idx * K + 1 + 1   # ti=1
        gid_b = combo_idx * K + 2 + 1   # ti=2
        fold_curves[gid_a] = {"trades": [], "equity": list(win_ti1)}
        fold_curves[gid_b] = {"trades": [], "equity": list(win_ti2)}
        members.append([gid_a, gid_b])
    result = SimpleNamespace(path_fold_members=members)

    series = _pool_unique_daily_returns(result, fold_curves)
    # 2 windows x (6 points -> 5 returns) = 10 unique daily returns. Old buggy concat would
    # have produced 6x that (~60) with each calendar day repeated 6x adjacently.
    assert len(series) == 10, f"expected 10 deduped daily returns, got {len(series)}"

    audit = validate_daily_series(series, result, fold_curves)
    assert audit["ok"] is True
    assert audit["n_distinct_windows"] == 2
    assert audit["series_len"] == 10
    assert audit["n_unique_days"] == 10  # each calendar day exactly once


def test_daily_series_is_chronological_and_no_duplicate_dates():
    """The pooled daily series concatenates the disjoint windows in chronological order and
    contains no duplicate calendar day."""
    from scripts.pead_significance import _pool_unique_daily_returns_audit

    K = 3
    win_early = _curve(date(2021, 2, 1), 5)
    win_late = _curve(date(2021, 9, 1), 5)
    fold_curves = {
        0 * K + 1 + 1: {"trades": [], "equity": list(win_late)},   # late window first id
        1 * K + 1 + 1: {"trades": [], "equity": list(win_late)},   # dup id, same late window
        0 * K + 2 + 1: {"trades": [], "equity": list(win_early)},  # early window
    }
    result = SimpleNamespace(path_fold_members=[[2, 3], [5]])
    audit = _pool_unique_daily_returns_audit(result, fold_curves)
    assert audit["n_distinct_windows"] == 2
    # Chronological: early window's dates precede the late window's dates.
    flat = [d for dates in audit["win_dates"] for d in dates]
    assert flat == sorted(flat)
    assert len(flat) == len(set(flat))  # no duplicate dates


def test_daily_series_self_validation_fires_on_duplicated_input():
    """The loud self-validation must REJECT a daily series that was built by the OLD buggy
    by-fold-id concatenation (same window repeated -> length inflated / days repeated)."""
    from scripts.pead_significance import validate_daily_series

    K = 4
    win = _curve(date(2022, 1, 3), 6)  # one window
    fold_curves = {}
    members = []
    for combo_idx in range(5):
        gid = combo_idx * K + 1 + 1  # same ti=1 window under 5 distinct ids
        fold_curves[gid] = {"trades": [], "equity": list(win)}
        members.append([gid])
    result = SimpleNamespace(path_fold_members=members)

    # Simulate the OLD buggy series: concat the SAME window 5x (5 ids x 5 returns = 25).
    pts = win
    one_window = [(pts[i][1] - pts[i - 1][1]) / max(pts[i - 1][1], 1e-9)
                  for i in range(1, len(pts))]
    buggy_series = one_window * 5  # length 25, calendar days repeated 5x

    audit = validate_daily_series(buggy_series, result, fold_curves)
    assert audit["ok"] is False
    assert "inflat" in audit["reason"].lower() or "!=" in audit["reason"]


def test_daily_series_self_validation_passes_on_clean_series():
    """A correctly-deduped series passes the self-validation cleanly."""
    from scripts.pead_significance import (_pool_unique_daily_returns,
                                           validate_daily_series)
    K = 4
    win_a = _curve(date(2022, 1, 3), 6)
    win_b = _curve(date(2022, 6, 1), 6)
    fold_curves = {
        0 * K + 1 + 1: {"trades": [], "equity": list(win_a)},
        1 * K + 1 + 1: {"trades": [], "equity": list(win_a)},
        0 * K + 2 + 1: {"trades": [], "equity": list(win_b)},
    }
    result = SimpleNamespace(path_fold_members=[[2, 3], [6]])
    series = _pool_unique_daily_returns(result, fold_curves)
    audit = validate_daily_series(series, result, fold_curves)
    assert audit["ok"] is True
    assert audit["series_len"] == len(series) == 10


# ───────────────────────── 2. Newey-West / HAC ─────────────────────────────────────

def test_newey_west_matches_hand_checked_reference():
    """Pin the Bartlett-kernel HAC to a hand-computed reference on a tiny series.

    series r = [1,2,3,4,5] (units arbitrary); T=5, rbar=3.
    dev = [-2,-1,0,1,2]; gamma0 = sum(dev^2)/T = 10/5 = 2.0.
    L=1: gamma1 = sum_{t=1}^{4} dev[t]*dev[t-1] / T
              = ((-1)(-2)+(0)(-1)+(1)(0)+(2)(1))/5 = (2+0+0+2)/5 = 0.8.
    Bartlett weight at k=1, L=1: 1 - 1/(1+1) = 0.5.
    S_hac = gamma0 + 2*0.5*gamma1 = 2.0 + 0.8 = 2.8.
    var(mean) = S_hac/T = 2.8/5 = 0.56 ; t_hac = rbar/sqrt(var) = 3/sqrt(0.56) = 4.00891862...
    """
    from scripts.pead_significance import newey_west_tstat
    out = newey_west_tstat([1.0, 2.0, 3.0, 4.0, 5.0], lag=1)
    assert out["lag"] == 1
    assert out["t_hac"] == pytest.approx(3.0 / math.sqrt(0.56), abs=1e-4)


def test_hac_t_below_ols_t_on_positive_autocorrelation():
    """On positively-autocorrelated returns, the HAC t (which adds positive autocovariance
    to the variance of the mean) must be SMALLER than the naive OLS t."""
    from scripts.pead_significance import newey_west_tstat
    rng = np.random.default_rng(9)
    # AR(1) with positive phi and positive mean.
    n, phi, mu = 400, 0.6, 0.001
    r = np.zeros(n)
    eps = rng.normal(0, 0.01, n)
    r[0] = mu + eps[0]
    for t in range(1, n):
        r[t] = mu + phi * (r[t - 1] - mu) + eps[t]
    out = newey_west_tstat(r.tolist(), lag=20)
    assert out["t_ols"] > 0
    assert out["t_hac"] < out["t_ols"], "HAC t must shrink vs OLS t under positive autocorr"


def test_hac_equals_ols_at_lag_zero_for_iid():
    """At lag 0 the HAC variance is just gamma0/T; on iid data t_hac ~ t_ols (gamma0 uses
    1/T, OLS uses 1/(T-1), so they agree closely for large T)."""
    from scripts.pead_significance import newey_west_tstat
    rng = np.random.default_rng(2)
    r = rng.normal(0.001, 0.01, 2000).tolist()
    out = newey_west_tstat(r, lag=0)
    assert out["t_hac"] == pytest.approx(out["t_ols"], rel=0.01)


# ───────────────────────── 3. Stratification PIT + cluster key ─────────────────────

def test_cluster_key_is_calendar_quarter():
    from scripts.pead_significance import cluster_key
    assert cluster_key(date(2022, 1, 15)) == "2022Q1"
    assert cluster_key(date(2022, 4, 1)) == "2022Q2"
    assert cluster_key(date(2022, 9, 30)) == "2022Q3"
    assert cluster_key(date(2022, 12, 31)) == "2022Q4"


def test_spy_trend_is_pit():
    """A trade's trend tag uses only SPY closes <= entry; a FUTURE move cannot change it."""
    from scripts.pead_significance import spy_trend_at
    idx = pd.date_range("2021-01-04", periods=300, freq="B")
    # Uptrend through the entry day -> 'up'.
    closes = pd.Series(np.linspace(100, 200, 300), index=idx)
    entry = idx[250].date()
    tag_before = spy_trend_at(entry, closes, ma_window=200)
    assert tag_before == "up"
    # Crash the SPY AFTER the entry day; the entry-day tag must NOT change (PIT).
    crashed = closes.copy()
    crashed.iloc[251:] = 10.0
    assert spy_trend_at(entry, crashed, ma_window=200) == tag_before


def test_spy_trend_down_and_insufficient_history():
    from scripts.pead_significance import spy_trend_at
    idx = pd.date_range("2021-01-04", periods=300, freq="B")
    # Price below its 200d MA at entry -> 'down'.
    closes = pd.Series(np.concatenate([np.full(200, 150.0), np.linspace(150, 80, 100)]),
                       index=idx)
    assert spy_trend_at(idx[260].date(), closes, ma_window=200) == "down"
    # Fewer than ma_window closes before entry -> None (untagged).
    assert spy_trend_at(idx[50].date(), closes, ma_window=200) is None


# ───────────────────────── 4. Determinism ──────────────────────────────────────────

def test_bootstrap_is_deterministic_for_seed():
    from scripts.pead_significance import event_clustered_bootstrap
    rng = np.random.default_rng(0)
    rets = [float(x) for x in rng.normal(0.005, 0.02, 200)]
    clus = [f"Q{i//25}" for i in range(200)]
    a = event_clustered_bootstrap(rets, clus, n_resamples=500, seed=99)
    b = event_clustered_bootstrap(rets, clus, n_resamples=500, seed=99)
    assert a == b
    c = event_clustered_bootstrap(rets, clus, n_resamples=500, seed=100)
    assert c["p_value"] != a["p_value"] or c["ci_low"] != a["ci_low"]


# ───────────────────────── 5/6. Capture harness, self-validation, artifacts ────────

def _synthetic_bars(n_days=520, start=date(2021, 1, 4), seed=7):
    rng = np.random.default_rng(seed)
    close = 100.0
    rows, d = [], pd.Timestamp(start)
    while len(rows) < n_days:
        if d.weekday() < 5:
            close *= (1 + rng.normal(0.0008, 0.012))
            rows.append({"open": close * (1 + rng.normal(0, 0.002)),
                         "high": close * (1 + abs(rng.normal(0, 0.004))),
                         "low": close * (1 - abs(rng.normal(0, 0.004))),
                         "close": close, "volume": int(rng.integers(1e6, 4e6))})
        d += pd.Timedelta(days=1)
    idx = pd.date_range(start=start, periods=len(rows), freq="B")
    return pd.DataFrame(rows, index=idx)


def _stub_run(monkeypatch, smoke=True):
    import scripts.run_pead_cpcv as rp
    import scripts.pead_significance as sig

    syn = {f"SYM{i}": _synthetic_bars(seed=i + 20) for i in range(5)}
    syn["SPY"] = _synthetic_bars(seed=500)

    def _fake_fetch(self, start, end):
        self.symbols_data = dict(syn)
        self.spy_prices = syn["SPY"]["close"]
        self.all_days_sorted = sorted({d.date() for d in syn["SPY"].index})
        self._global_regime_map = {}

    monkeypatch.setattr(rp.PEADStrategy, "fetch_data", _fake_fetch)
    monkeypatch.setattr(rp, "RUSSELL_1000_TICKERS", [f"SYM{i}" for i in range(5)], raising=False)
    return sig.run_analysis(smoke=smoke, symbols=[f"SYM{i}" for i in range(5)],
                            total_years=2, cpcv_k=4, cpcv_paths=2, n_resamples=100)


def test_full_run_schema(monkeypatch):
    out = _stub_run(monkeypatch)
    assert {"baseline_self_validation", "anchor_cpcv_path_tstat",
            "event_clustered_bootstrap", "iid_bootstrap", "newey_west_hac",
            "trend_stratification"} <= set(out)
    a = out["anchor_cpcv_path_tstat"]
    assert "path_tstat" in a and a["n_eff_folds"] == 4
    b = out["event_clustered_bootstrap"]
    assert {"observed_sharpe", "p_value", "ci_low", "ci_high", "n_clusters"} <= set(b)
    rows = out["trend_stratification"]["rows"]
    assert {r["regime"] for r in rows} == {"up", "down"}


def test_baseline_self_validation_reconciles(monkeypatch):
    """The path Sharpe recomputed from captured curves MUST reproduce run_cpcv's own mean
    Sharpe (the +0.546 reconciliation lens) within tolerance — proving the capture stream
    is faithful to the committed CPCV run."""
    import scripts.pead_significance as sig
    out = _stub_run(monkeypatch)
    bsv = out["baseline_self_validation"]
    assert bsv["self_check_ok"] is True, "capture diverged from run_cpcv mean Sharpe"
    assert abs(bsv["recomputed_path_sharpe"] - bsv["real_cpcv_mean_sharpe"]) <= sig.SELF_CHECK_TOL


def test_pure_additive_last_trades_does_not_change_cpcv(monkeypatch):
    """The §1.3 _last_trades stash is pure-additive: a CPCV run with the capture wrapper
    (which reads _last_trades) yields BYTE-IDENTICAL path Sharpes to a plain run_cpcv with
    the same scorer/data/folds."""
    import scripts.run_pead_cpcv as rp
    import scripts.pead_significance as sig
    from scripts.walkforward.cpcv import run_cpcv

    syn = {f"SYM{i}": _synthetic_bars(seed=i + 20) for i in range(5)}
    syn["SPY"] = _synthetic_bars(seed=500)

    def _mk():
        s = rp.PEADStrategy(scorer=sig._make_scorer(), symbols=[f"SYM{i}" for i in range(5)])
        s.symbols_data = dict(syn)
        s.spy_prices = syn["SPY"]["close"]
        s.all_days_sorted = sorted({d.date() for d in syn["SPY"].index})
        s._global_regime_map = {}
        return s

    plain = run_cpcv(strategy=_mk(), purge_days=10, embargo_days=10,
                     n_folds=4, n_paths=2, total_years=2)
    result, _curves = sig._capture_run(_mk(), 10, 10, 4, 2, 2)
    assert plain.path_sharpes == result.path_sharpes
    assert plain.path_fold_members == result.path_fold_members


def test_run_fold_stashes_last_trades(monkeypatch):
    """run_pead_cpcv.PEADStrategy.run_fold sets _last_trades (pure-additive side channel),
    populated from the sim result's trades, without affecting the FoldResult."""
    from types import SimpleNamespace
    import app.backtesting.agent_simulator as agent_mod
    import app.data.universe_history as uni_mod
    from scripts.run_pead_cpcv import PEADStrategy

    class _FakeTrade:
        def __init__(self, pnl_pct):
            self.pnl_pct = pnl_pct
            self.symbol = "AAPL"
            self.entry_date = date(2022, 2, 1)
            self.exit_date = date(2022, 3, 1)
            self.pnl = pnl_pct * 1000

    trades = [_FakeTrade(0.02), _FakeTrade(-0.01)]
    eq = [(date(2022, 1, 3), 100000.0), (date(2022, 1, 4), 100500.0),
          (date(2022, 1, 5), 100800.0)]
    sim_result = SimpleNamespace(
        exit_breakdown={"STOP": 0}, total_trades=2, sharpe_ratio=0.5,
        total_return_pct=0.1, max_drawdown_pct=0.05, win_rate=0.55,
        profit_factor=1.4, equity_curve=eq, trades=trades)

    class _FakeSim:
        def __init__(self, *a, **k):
            pass

        def run(self, *a, **k):
            return sim_result

    monkeypatch.setattr(agent_mod, "AgentSimulator", _FakeSim, raising=True)
    monkeypatch.setattr(uni_mod, "pit_union", lambda *a, **k: ["AAPL"], raising=True)
    monkeypatch.setattr(uni_mod, "historical_trade_symbols", lambda *a, **k: [], raising=True)

    strat = PEADStrategy(scorer=SimpleNamespace(), symbols=["AAPL"])
    strat.symbols_data = {"AAPL": SimpleNamespace()}
    strat.spy_prices = None
    strat._global_regime_map = {}
    fr = strat.run_fold(0, 1, date(2021, 1, 1), date(2021, 12, 31),
                        date(2022, 1, 1), date(2022, 12, 31))
    assert hasattr(strat, "_last_trades")
    assert len(strat._last_trades) == 2
    assert fr.profit_factor == pytest.approx(1.4)


def test_full_run_artifacts(monkeypatch, tmp_path):
    import scripts.pead_significance as sig
    out = _stub_run(monkeypatch)
    monkeypatch.setattr(sig, "ARTIFACT_DIR", tmp_path)
    paths = sig._write_artifacts(out, "test")
    assert paths["json"].exists()
    assert paths["summary"].exists()
    assert paths["trend"].exists()
    sig._print_report(out)  # must not raise


# ───────────────────────── 7. ASCII-safe runtime strings ───────────────────────────

def test_no_non_ascii_in_runtime_strings():
    """print()/logger strings must be cp1252-safe (ASCII) so a Windows console/file handler
    cannot crash the run. Docstrings/comments are exempt."""
    import ast
    import scripts.pead_significance as mod

    tree = ast.parse(open(mod.__file__, encoding="utf-8").read())
    bad = []
    for node in ast.walk(tree):
        is_print = isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print"
        is_log = (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                  and isinstance(node.func.value, ast.Name) and node.func.value.id == "logger")
        if not (is_print or is_log):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                if any(ord(ch) > 127 for ch in sub.value):
                    bad.append((node.lineno, sub.value))
    assert not bad, f"non-ASCII in runtime print/log strings: {bad}"
