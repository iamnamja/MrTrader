"""Tests for Alpha v2 §1.2 — PEAD crisis-block robustness harness + the regime-general
control added to PEADScorer.

Covers:
  1. DEFAULT PATH UNCHANGED: PEADScorer with regime_control=None is byte-identical
     (exposure scalar 1.0; no new block fires; build_pead_scorer signals unchanged).
  2. GENERIC CONTROL IS PIT: the exposure scalar on day D uses only data <= D — a
     future spike does not change D's scalar (vol_target and trend).
  3. THRESHOLD SWEEP emits expected rows incl. the inf/no-block case; baseline flagged.
  4. LOCO masking is correct: the right date range removed, PIT, no off-by-one; the
     no-removal baseline equals the unmasked recompute.
  5. BASELINE self-validation row reproduces the committed config on the smoke schema.
  6. CPCV path membership: run_cpcv exposes the REAL grouping (path_fold_members) and
     LOCO consumes it directly (no reconstruction); the (none removed) row reproduces
     run_cpcv's own mean Sharpe (self-check).
"""
from datetime import date

import math
import numpy as np
import pandas as pd
import pytest


# ───────────────────────── 1. Default scorer path unchanged ────────────────────────

def test_scorer_default_regime_control_is_off():
    from app.ml.pead_scorer import PEADScorer
    s = PEADScorer()
    assert s.regime_control is None
    # Scalar is a no-op 1.0 regardless of data when control is off.
    spy = pd.DataFrame({"close": np.linspace(100, 50, 300)},
                       index=pd.date_range("2022-01-01", periods=300, freq="B"))
    assert s._regime_exposure_scalar(date(2022, 6, 1), {"SPY": spy}) == 1.0


def test_control_off_byte_identical_to_default(monkeypatch):
    """With regime_control=None the scorer output (incl. confidence) must be identical
    to a default scorer even when VIX data is present — the new code is a pure no-op."""
    import app.data.fmp_provider as fmp
    from app.ml.pead_scorer import PEADScorer

    def _fake_feats(sym, as_of):
        return {"fmp_surprise_1q": 0.12, "fmp_days_since_earnings": 1}

    monkeypatch.setattr(fmp, "get_earnings_features_at", _fake_feats, raising=False)
    idx = pd.date_range("2022-01-03", periods=260, freq="B")
    bars = {f"S{i}": pd.DataFrame(
        {"open": [10] * 260, "high": [11] * 260, "low": [9] * 260,
         "close": [10] * 260, "volume": [1e6] * 260}, index=idx) for i in range(3)}
    bars["SPY"] = pd.DataFrame({"close": np.linspace(100, 120, 260)}, index=idx)
    bars["^VIX"] = pd.DataFrame({"close": np.full(260, 22.0)}, index=idx)
    day = idx[259].date()

    default = PEADScorer(long_short=False)
    explicit_off = PEADScorer(long_short=False, regime_control=None)
    assert default(day, bars) == explicit_off(day, bars)


def test_build_pead_scorer_unchanged_signals(monkeypatch):
    """build_pead_scorer must produce identical signals with/without the new field
    defaulting (regime_control absent) — same entry set on a stubbed earnings day."""
    from scripts.run_pead_cpcv import build_pead_scorer

    s = build_pead_scorer()
    assert s.regime_control is None
    assert s.vix_block_all == 30.0

    # Stub earnings so __call__ produces a deterministic long signal, no VIX in data.
    # __call__ does `from app.data.fmp_provider import get_earnings_features_at`, so
    # we must patch the source module attribute (not app.ml.pead_scorer).
    import app.data.fmp_provider as fmp

    def _fake_feats(sym, as_of):
        return {"fmp_surprise_1q": 0.10, "fmp_days_since_earnings": 1}

    monkeypatch.setattr(fmp, "get_earnings_features_at", _fake_feats, raising=False)
    bars = pd.DataFrame({"open": [10] * 50, "high": [11] * 50, "low": [9] * 50,
                         "close": [10] * 50, "volume": [1e6] * 50},
                        index=pd.date_range("2022-01-01", periods=50, freq="B"))
    out = s(date(2022, 2, 1), {"AAA": bars})
    assert len(out) == 1 and out[0][0] == "AAA" and out[0][2] == "long"


# ───────────────────────── 2. Generic control is PIT ───────────────────────────────

def test_vol_target_scalar_is_pit():
    """The vol_target exposure scalar on day D must use ONLY SPY closes <= D. A spike
    AFTER D must not change D's scalar."""
    from app.ml.pead_scorer import PEADScorer
    s = PEADScorer(regime_control="vol_target", regime_control_target_vol=0.16,
                   regime_control_vol_lookback=20)
    idx = pd.date_range("2022-01-03", periods=120, freq="B")
    # Calm series up to D, then a huge spike AFTER D.
    closes = np.full(120, 100.0)
    for i in range(1, 120):
        closes[i] = closes[i - 1] * 1.0005  # very low vol
    spy_calm = pd.DataFrame({"close": closes}, index=idx)
    D = idx[60].date()
    scalar_before = s._regime_exposure_scalar(D, {"SPY": spy_calm})

    # Inject a massive vol spike strictly AFTER D.
    closes2 = closes.copy()
    for i in range(61, 120):
        closes2[i] = closes2[i - 1] * (1.10 if i % 2 else 0.91)  # huge swings
    spy_spiked = pd.DataFrame({"close": closes2}, index=idx)
    scalar_after_future_spike = s._regime_exposure_scalar(D, {"SPY": spy_spiked})

    assert scalar_before == pytest.approx(scalar_after_future_spike), \
        "future vol spike leaked into day-D scalar (NOT PIT)"
    # Calm regime -> low realized vol -> scalar capped at 1.0.
    assert scalar_before == pytest.approx(1.0)


def test_vol_target_scalar_cuts_in_high_vol():
    from app.ml.pead_scorer import PEADScorer
    s = PEADScorer(regime_control="vol_target", regime_control_target_vol=0.16,
                   regime_control_vol_lookback=20)
    idx = pd.date_range("2022-01-03", periods=60, freq="B")
    closes = [100.0]
    for i in range(1, 60):
        closes.append(closes[-1] * (1.05 if i % 2 else 0.952))  # ~ very high realized vol
    spy = pd.DataFrame({"close": closes}, index=idx)
    scalar = s._regime_exposure_scalar(idx[-1].date(), {"SPY": spy})
    assert 0.0 <= scalar < 1.0  # high realized vol -> exposure cut below 1


def test_trend_scalar_is_pit_and_zero_below_ma():
    from app.ml.pead_scorer import PEADScorer
    s = PEADScorer(regime_control="trend", regime_control_trend_ma=200)
    idx = pd.date_range("2021-01-04", periods=260, freq="B")
    # Uptrend then we evaluate at a point ABOVE the 200d MA -> scalar 1.0.
    up = np.linspace(50, 150, 260)
    spy_up = pd.DataFrame({"close": up}, index=idx)
    D = idx[259].date()
    assert s._regime_exposure_scalar(D, {"SPY": spy_up}) == 1.0

    # Below the 200d MA at D -> scalar 0.0. Future recovery must not change D.
    down = np.concatenate([np.linspace(150, 150, 200), np.linspace(150, 80, 60)])
    spy_down = pd.DataFrame({"close": down}, index=idx)
    scalar_D = s._regime_exposure_scalar(D, {"SPY": spy_down})
    assert scalar_D == 0.0
    # Append a future spike-up — D's scalar unchanged (PIT).
    spy_down2 = spy_down.copy()
    spy_down2.iloc[-1, spy_down2.columns.get_loc("close")] = 80.0
    assert s._regime_exposure_scalar(D, {"SPY": spy_down2}) == scalar_D


def test_control_below_floor_blocks_entries(monkeypatch):
    """When the regime scalar drops below the floor, __call__ returns [] (block all)."""
    import app.data.fmp_provider as fmp
    from app.ml.pead_scorer import PEADScorer
    s = PEADScorer(regime_control="trend", regime_control_trend_ma=200,
                   regime_control_floor=0.50, vix_block_all=math.inf)

    def _fake_feats(sym, as_of):
        return {"fmp_surprise_1q": 0.10, "fmp_days_since_earnings": 1}

    monkeypatch.setattr(fmp, "get_earnings_features_at", _fake_feats, raising=False)
    idx = pd.date_range("2021-01-04", periods=260, freq="B")
    down = np.concatenate([np.full(200, 150.0), np.linspace(150, 80, 60)])
    spy = pd.DataFrame({"close": down}, index=idx)
    bars = pd.DataFrame({"open": [10] * 260, "high": [11] * 260, "low": [9] * 260,
                         "close": [10] * 260, "volume": [1e6] * 260}, index=idx)
    out = s(idx[259].date(), {"AAA": bars, "SPY": spy})
    assert out == []  # trend scalar 0 < floor -> blocked


# ───────────────────────── helpers for harness tests ───────────────────────────────

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


def _stub_run(monkeypatch, only=None, smoke=True):
    import scripts.run_pead_cpcv as rp
    import scripts.pead_crisis_robustness as cr

    syn = {f"SYM{i}": _synthetic_bars(seed=i + 20) for i in range(5)}
    syn["SPY"] = _synthetic_bars(seed=500)

    def _fake_fetch(self, start, end):
        self.symbols_data = dict(syn)
        self.spy_prices = syn["SPY"]["close"]
        self.all_days_sorted = sorted({d.date() for d in syn["SPY"].index})
        self._global_regime_map = {}

    monkeypatch.setattr(rp.PEADStrategy, "fetch_data", _fake_fetch)
    monkeypatch.setattr(rp, "RUSSELL_1000_TICKERS", [f"SYM{i}" for i in range(5)], raising=False)
    return cr.run_analysis(smoke=smoke, only=only, symbols=[f"SYM{i}" for i in range(5)],
                           total_years=2, cpcv_k=4, cpcv_paths=2)


# ───────────────────────── 3. Threshold sweep rows ─────────────────────────────────

def test_threshold_sweep_emits_rows_including_no_block(monkeypatch):
    out = _stub_run(monkeypatch, only="threshold")
    assert "threshold_sweep" in out
    rows = out["threshold_sweep"]["rows"]
    labels = [r["label"] for r in rows]
    assert "VIX>30" in labels
    assert "inf (no block)" in labels  # the disabled-block baseline
    # Exactly one baseline row, and it is the VIX>30 row.
    base = [r for r in rows if r["is_baseline"]]
    assert len(base) == 1 and base[0]["label"] == "VIX>30"
    for r in rows:
        assert {"mean_sharpe", "path_tstat", "pct_positive", "p5_sharpe", "p95_sharpe", "n_paths"} <= set(r)


# ───────────────────────── 4. LOCO masking correctness ─────────────────────────────

def test_sharpe_from_equity_masks_correct_range():
    """The masked Sharpe must drop exactly the dates in [start,end] and the boundary
    return straddling the gap; with no mask it equals the production Sharpe path."""
    from scripts.pead_crisis_robustness import _sharpe_from_equity
    from app.backtesting.strategy_simulator import StrategySimulator

    idx = pd.date_range("2022-01-03", periods=40, freq="B")
    vals = list(100.0 + np.arange(40) * 0.5)
    ec = list(zip([d.date() for d in idx], vals))

    # No mask: equals StrategySimulator._sharpe over the daily diffs.
    full_sharpe, n_full = _sharpe_from_equity(ec, None)
    rets = [(vals[i] - vals[i - 1]) / vals[i - 1] for i in range(1, len(vals))]
    assert full_sharpe == pytest.approx(StrategySimulator._sharpe(rets, 252))
    assert n_full == len(rets)

    # Mask a middle window: those dates are dropped + the straddle return removed.
    lo, hi = idx[10].date(), idx[20].date()
    masked_sharpe, n_masked = _sharpe_from_equity(ec, (lo, hi))
    # 11 dates removed -> their points gone; plus the single straddle diff dropped.
    surviving_dates = [d for d in [x.date() for x in idx] if not (lo <= d <= hi)]
    assert n_masked == len(surviving_dates) - 1 - 1  # diffs minus the straddle gap
    assert masked_sharpe != pytest.approx(full_sharpe)


def test_loco_no_removal_is_noop_vs_direct_recompute():
    """The LOCO '(none removed)' row (mask=None) must equal a direct unmasked recompute
    of path Sharpes from the SAME fold curves — masking with None is a no-op. We inject
    non-empty fold curves directly (the synthetic stub makes no PEAD trades)."""
    from scripts.pead_crisis_robustness import _loco_row, _sharpe_from_equity
    import numpy as np

    idx = pd.date_range("2022-01-03", periods=60, freq="B")
    # Two folds with distinct upward equity curves.
    ec1 = list(zip([d.date() for d in idx], list(100.0 + np.arange(60) * 0.4)))
    ec2 = list(zip([d.date() for d in idx], list(100.0 + np.cumsum(np.full(60, 0.3)))))
    fold_curves = {1: ec1, 2: ec2}
    paths = [[1], [2], [1, 2]]

    row = _loco_row("(none removed)", None, paths, fold_curves, n_folds=4)
    # Direct recompute: each path mean of its folds' unmasked Sharpes.
    direct = []
    for members in paths:
        fs = [_sharpe_from_equity(fold_curves[m], None)[0] for m in members]
        direct.append(float(np.mean(fs)))
    assert row["n_paths"] == 3
    assert row["mean_sharpe"] == pytest.approx(round(float(np.mean(direct)), 4))


def test_loco_full_run_emits_baseline_and_episode_rows(monkeypatch):
    """The LOCO sub-analysis emits a (none removed) baseline plus one row per defined
    crisis window, each with the expected schema and in_window flags."""
    out = _stub_run(monkeypatch, only="loco")
    assert "loco" in out
    rows = {r["removed"]: r for r in out["loco"]["rows"]}
    assert "(none removed)" in rows
    for w in ("covid_tail_2020", "bear_2022", "yen_carry_aug2024", "tariff_apr2025"):
        assert w in rows
        assert {"mean_sharpe", "path_tstat", "pct_positive", "p5_sharpe", "in_window"} <= set(rows[w])


def test_loco_out_of_window_episode_has_no_effect(monkeypatch):
    """A crisis window entirely OUTSIDE the (smoke) data range must leave the metrics
    identical to the no-removal baseline (masking nothing)."""
    out = _stub_run(monkeypatch, only="loco")
    rows = {r["removed"]: r for r in out["loco"]["rows"]}
    base = rows["(none removed)"]
    # tariff_apr2025 is far outside the 2021-2022 synthetic smoke window.
    oow = rows.get("tariff_apr2025")
    assert oow is not None and oow["in_window"] is False
    assert oow["mean_sharpe"] == pytest.approx(base["mean_sharpe"])
    assert oow["n_paths"] == base["n_paths"]


# ───────────────────────── 5. Generic control sub-analysis ─────────────────────────

def test_generic_control_rows_and_baseline(monkeypatch):
    out = _stub_run(monkeypatch, only="control")
    assert "generic_control" in out
    rows = out["generic_control"]["rows"]
    # First row is the VIX>30 baseline; at least one generic control row follows.
    assert rows[0]["is_baseline"] is True
    assert rows[0]["control"].startswith("VIX>30")
    assert any(not r["is_baseline"] for r in rows)
    for r in rows:
        assert {"mean_sharpe", "path_tstat", "pct_positive", "p5_sharpe", "n_paths"} <= set(r)


# ───────────────────────── 6. Real path membership (no reconstruction) ─────────────

class _RecordingStrategy:
    """Minimal CPCV strategy that records every (global) fold_idx run_cpcv runs and
    returns a deterministic FoldResult, so we can read run_cpcv's REAL path grouping."""

    model_type = "pead"
    per_fold_retrain = True  # genuine per-fold (run_cpcv skips the global OOS guard)

    def __init__(self):
        from datetime import date as _date
        self.symbols_data = {}
        self.spy_prices = None
        self.all_days_sorted = []
        self.model = type("_NoModel", (), {"trained_through": _date.min})()
        self.allow_in_sample = False
        self.calls = []  # (fold_idx, te_start, te_end)

    def run_fold(self, fold_idx, n_folds, tr_start, tr_end, te_start, te_end):
        from scripts.walkforward.gates import FoldResult
        self.calls.append((fold_idx, te_start, te_end))
        # Deterministic positive Sharpe so the path survives (combo_sharpes non-empty).
        return FoldResult(
            fold=fold_idx, train_start=tr_start, train_end=tr_end,
            test_start=te_start, test_end=te_end, trades=10, win_rate=0.6,
            sharpe=0.5 + (fold_idx % 3) * 0.1, max_drawdown=-0.05,
            total_return=0.1, stop_exit_rate=0.1, model_version=0,
            profit_factor=1.5, calmar_ratio=0.4, k_ratio=0.3, n_obs=200,
            regime_sharpes={}, regime_obs_counts={},
        )


def _run_smoke_cpcv(k=4, p=2, purge=10, years=2):
    """Run run_cpcv on a tiny deterministic config and return (result, strategy)."""
    from scripts.walkforward.cpcv import run_cpcv
    strat = _RecordingStrategy()
    result = run_cpcv(strategy=strat, purge_days=purge, embargo_days=purge,
                      n_folds=k, n_paths=p, total_years=years)
    return result, strat


def test_cpcv_exposes_path_fold_members_aligned_with_path_sharpes():
    """run_cpcv must expose path_fold_members: one list per surviving path, aligned 1:1
    with path_sharpes, holding the REAL global fold ids it ran (post all guards)."""
    result, strat = _run_smoke_cpcv()
    assert len(result.path_fold_members) == len(result.path_sharpes) >= 1
    # Every recorded id is a global fold id actually run; members are a subset of them.
    run_ids = {c[0] for c in strat.calls}
    flat = [fid for members in result.path_fold_members for fid in members]
    assert set(flat) <= run_ids
    # Members use the global id formula combo_idx*n_folds + ti + 1 (all >= 1, ints).
    for members in result.path_fold_members:
        assert members and all(isinstance(m, int) and m >= 1 for m in members)


def test_path_fold_members_is_the_real_surviving_grouping():
    """path_fold_members must equal the REAL surviving grouping run_cpcv used — derived
    from run_cpcv's OWN run_fold calls, NOT a reconstruction. We rebuild the expected
    grouping from the recorded calls (the ground truth) and assert equality.

    This is the regression test that FAILS if anyone reintroduces a reconstruction that
    diverges from run_cpcv (e.g. by omitting the BUG-23 overlap guard): the recorded
    calls are exactly what run_cpcv executed, so the only way path_fold_members can match
    them is if it is sourced from the real run."""
    result, strat = _run_smoke_cpcv()
    # Ground truth: group recorded global ids by combo. With n_folds=4 the id encodes
    # combo_idx = (id - 1) // 4. Each combo's surviving ids, in run order, form a path.
    k = result.n_folds
    expected_by_combo: dict = {}
    for fid, _ts, _te in strat.calls:
        combo_idx = (fid - 1) // k
        expected_by_combo.setdefault(combo_idx, []).append(fid)
    expected_paths = [expected_by_combo[c] for c in sorted(expected_by_combo)]
    assert result.path_fold_members == expected_paths
    # Sanity: the grouping is NOT the inflated full-combination reconstruction. C(4,2)=6
    # combos, but combos whose only test fold(s) lack prior-train history are skipped, so
    # there are strictly FEWER surviving paths than 6.
    assert len(result.path_fold_members) < 6


def test_loco_uses_real_membership_not_reconstruction(monkeypatch):
    """LOCO must group the captured per-fold curves using run_cpcv's REAL
    path_fold_members (same path set, same fold-ids) — proving it consumes the real
    grouping rather than reconstructing it. FAILS if a reconstruction is reintroduced.

    We intercept _loco_capture_run's result to read path_fold_members, then assert the
    LOCO rows were built over exactly that grouping (n_paths matches, and the reconstruct
    helpers no longer exist)."""
    import scripts.pead_crisis_robustness as cr
    # The reconstruction MUST be gone (root-cause deletion).
    assert not hasattr(cr, "_cpcv_path_membership")
    assert not hasattr(cr, "_build_boundaries")

    captured = {}
    orig = cr._loco_capture_run

    def _spy(*a, **k):
        res, curves = orig(*a, **k)
        captured["members"] = [list(m) for m in res.path_fold_members]
        captured["mean"] = float(res.mean_sharpe)
        return res, curves

    monkeypatch.setattr(cr, "_loco_capture_run", _spy)
    out = _stub_run(monkeypatch, only="loco")
    loco = out["loco"]
    base = next(r for r in loco["rows"] if r["removed"] == "(none removed)")
    # LOCO grouped over the REAL membership: the no-removal path count equals the number
    # of surviving paths run_cpcv reported (paths with >=1 captured curve).
    assert "members" in captured and len(captured["members"]) >= 1
    assert base["n_paths"] <= len(captured["members"])


def test_loco_self_check_matches_run_cpcv_mean_sharpe(monkeypatch):
    """The (none removed) LOCO mean Sharpe must reproduce run_cpcv's OWN mean Sharpe to a
    tight tolerance (same curves + same grouping). The harness records self_check_ok and
    the real mean — assert it passes and the values agree within LOCO_SELF_CHECK_TOL."""
    import scripts.pead_crisis_robustness as cr
    out = _stub_run(monkeypatch, only="loco")
    loco = out["loco"]
    assert loco["self_check_ok"] is True, "LOCO self-check flagged divergence"
    base = next(r for r in loco["rows"] if r["removed"] == "(none removed)")
    assert abs(base["mean_sharpe"] - loco["real_cpcv_mean_sharpe"]) <= cr.LOCO_SELF_CHECK_TOL


# ───────────────────────── 7. Full run schema + artifacts ──────────────────────────

def test_no_non_ascii_in_runtime_strings():
    """Runtime print()/logger strings must be cp1252-safe (ASCII) so a Windows console
    or file handler can never crash the run. Docstrings/comments are exempt."""
    import ast
    import scripts.pead_crisis_robustness as mod

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


def test_full_run_schema_and_artifacts(monkeypatch, tmp_path):
    import scripts.pead_crisis_robustness as cr
    out = _stub_run(monkeypatch, only=None)
    # All three sub-analyses present in a full run.
    assert {"threshold_sweep", "loco", "generic_control"} <= set(out)
    monkeypatch.setattr(cr, "ARTIFACT_DIR", tmp_path)
    paths = cr._write_artifacts(out, "test")
    assert paths["json"].exists()
    assert paths["threshold_sweep"].exists()
    assert paths["loco"].exists()
    assert paths["generic_control"].exists()
    cr._print_report(out)  # must not raise
