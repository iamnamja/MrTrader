"""CH0a — the pure helpers of the constant-gross trend baseline (standalone metrics + regime-conditional
Sharpe). The slow CPCV `build_baseline()` is exercised by running the script, not the unit suite."""
import numpy as np
import pandas as pd

from scripts import ch0_baseline as ch0


def _series(n=300, mu=0.0004, sd=0.01, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(rng.normal(mu, sd, n), index=idx)


def test_sharpe_and_standalone_metrics():
    r = _series()
    m = ch0.standalone_metrics(r)
    assert m["n_obs"] == 300
    # Sharpe = round(mean/std*sqrt(252), 3).
    assert m["sharpe"] == round(r.mean() / r.std() * np.sqrt(252), 3)
    assert m["ann_vol"] > 0 and m["max_drawdown"] <= 0
    # short series → empty (guard)
    assert ch0.standalone_metrics(r.iloc[:5]) == {}


def test_regime_conditional_sharpe_groups_by_label(monkeypatch):
    r = _series(n=250)
    # synthetic regime map: first half "calm", second half "stress"
    from scripts.walkforward import regime as _rg
    dates = list(r.index)
    rmap = {d.date(): ("calm" if i < 125 else "stress") for i, d in enumerate(dates)}
    monkeypatch.setattr(_rg, "load_regime_map", lambda a, b, **k: rmap)
    out = ch0.regime_conditional_sharpe(r)
    assert set(out) == {"calm", "stress"}
    assert out["calm"]["n_days"] + out["stress"]["n_days"] == 250
    assert abs(out["calm"]["frac_days"] + out["stress"]["frac_days"] - 1.0) < 1e-9
    # each regime's Sharpe matches a direct compute on its slice
    calm = r.iloc[:125]
    assert abs(out["calm"]["sharpe"] - round(calm.mean() / calm.std() * np.sqrt(252), 3)) < 1e-6


def test_regime_sparse_labels_ffill_and_full_coverage(monkeypatch):
    """Sparse labels (not every date) must forward-fill PIT, and dates BEFORE the first label
    must bucket as UNLABELED — never be silently dropped. frac_days must sum to 1.0."""
    r = _series(n=200)
    from scripts.walkforward import regime as _rg
    dates = list(r.index)
    # label only every 20th day, and start AFTER day 10 (first 10 dates have no PIT label)
    rmap = {dates[i].date(): ("A" if (i // 20) % 2 == 0 else "B")
            for i in range(10, 200, 20)}
    monkeypatch.setattr(_rg, "load_regime_map", lambda a, b, **k: rmap)
    out = ch0.regime_conditional_sharpe(r)
    assert "UNLABELED" in out and out["UNLABELED"]["n_days"] == 10  # the leading pre-label days
    assert sum(v["n_days"] for v in out.values()) == 200  # no vanished days
    assert abs(sum(v["frac_days"] for v in out.values()) - 1.0) < 1e-9


def test_calmar_guard_and_obs_floor():
    # positive-only series → max_dd == 0 → calmar is NaN (guarded, no ZeroDivision)
    up = pd.Series(np.full(50, 0.001), index=pd.date_range("2020-01-01", periods=50, freq="B"))
    m = ch0.standalone_metrics(up)
    assert m["max_drawdown"] == 0.0 and np.isnan(m["calmar"])
    # exactly 20 obs is below MIN_OBS (21) → empty (the two guards agree)
    assert ch0.standalone_metrics(up.iloc[:20]) == {}
