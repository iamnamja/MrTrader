"""CH0a (Compound-and-Harden) — freeze the CONSTANT-GROSS ETF-trend BASELINE.

This is the immutable benchmark that every CH2 antifragile-sizing change must BEAT out-of-sample
(the gate: "governed beats constant-gross on CPCV, new params charged to DSR"). "Constant-gross" =
the static trend policy with ALL antifragile/governor multipliers = 1.0 — which is exactly what
`sleeves.live_trend_book_returns()` produces (`tsmom_backtest(closes, TSMOMConfig())`: inverse-vol
sizing + caps + weekly rebalance, NO crash/credit/curve/ladder overlays, `book_vol_target=None`).

We run that ungoverned book through the SAME uniform gate (`sleeve_lab.evaluate_sleeve` →
purged/embargoed CPCV → Ruler-v2) plus a regime-conditional Sharpe breakdown (`regime.load_regime_map`),
and write a versioned, reviewable artifact to `docs/reference/ch0_trend_baseline.json`. Re-running
overwrites it; the git history is the version log. This is READ-ONLY analysis — it trades nothing.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import date, datetime, timezone
from typing import Dict

import numpy as np
import pandas as pd

ARTIFACT = "docs/reference/ch0_trend_baseline.json"
ANN = 252
# Pinned end date for the FROZEN baseline window (immutability): re-running at the same commit
# reproduces the same window instead of silently extending to "today". Regeneration bumps this
# + the git commit = a new versioned record, never a silent refresh.
BASELINE_END = date(2026, 7, 7)
MIN_OBS = 21  # need > 20 for a std-based Sharpe (see _sharpe); keep the two guards consistent


def _sharpe(r: pd.Series, ppy: int = ANN) -> float:
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(ppy)) if len(r) >= MIN_OBS and r.std() > 0 else float("nan")


def standalone_metrics(rets: pd.Series) -> Dict[str, float]:
    """Standalone (non-CPCV) profile of the constant-gross return series."""
    r = rets.dropna()
    if len(r) < MIN_OBS:
        return {}
    cum = (1.0 + r).cumprod()
    max_dd = float((cum / cum.cummax() - 1.0).min())
    ann_ret = float(r.mean() * ANN)
    return {
        "n_obs": int(len(r)),
        "sharpe": round(_sharpe(r), 3),
        "ann_return": round(ann_ret, 4),
        "ann_vol": round(float(r.std() * np.sqrt(ANN)), 4),
        "max_drawdown": round(max_dd, 4),
        "calmar": round(ann_ret / abs(max_dd), 3) if max_dd < 0 else float("nan"),
    }


def regime_conditional_sharpe(rets: pd.Series) -> Dict[str, Dict[str, float]]:
    """Sharpe of the constant-gross book WITHIN each regime label — the CH2 target profile
    (which regimes trend wins vs whipsaws). Uses the backtest-aligned regime map (PIT: the label
    is ffill'd from the last known regime, and returns are already PIT in the backtest)."""
    from scripts.walkforward.regime import load_regime_map
    r = rets.dropna()
    rmap = load_regime_map(r.index.min().date(), r.index.max().date())
    labels = pd.Series({pd.Timestamp(d): v for d, v in rmap.items()}).sort_index()
    # ffill = PIT (last known label at-or-before each date). Any return date BEFORE the first
    # regime label has no PIT label → bucket it as UNLABELED rather than let groupby silently
    # drop it (so n_days sum + frac_days total to the full sample, no vanished days).
    aligned = labels.reindex(r.index, method="ffill").where(lambda s: s.notna(), "UNLABELED")
    out: Dict[str, Dict[str, float]] = {}
    for label, idx in r.groupby(aligned).groups.items():
        rr = r.loc[idx]
        out[str(label)] = {"n_days": int(len(rr)), "sharpe": round(_sharpe(rr), 3),
                           "ann_return": round(float(rr.mean() * ANN), 4),
                           "frac_days": round(len(rr) / len(r), 3)}
    return out


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _data_fingerprint(closes: pd.DataFrame) -> Dict[str, object]:
    """A verifiable fingerprint of the exact price panel the baseline was frozen from — so a
    regeneration can be checked against the committed record (yfinance auto-adjust silently
    re-adjusts history over time; this surfaces such drift instead of hiding it)."""
    last = {c: round(float(closes[c].dropna().iloc[-1]), 6) for c in closes.columns}
    payload = json.dumps({"n_rows": int(len(closes)),
                          "first": str(closes.index.min().date()),
                          "last": str(closes.index.max().date()),
                          "last_close": dict(sorted(last.items()))}, sort_keys=True)
    return {"sha256": hashlib.sha256(payload.encode()).hexdigest()[:16],
            "n_rows": int(len(closes)), "last_close_per_symbol": dict(sorted(last.items()))}


def build_baseline() -> dict:
    from scripts.walkforward.sleeve_lab import Sleeve, evaluate_sleeve
    from scripts.walkforward.sleeves import LIVE_TREND_UNIVERSE, fetch_universe_closes, \
        live_trend_book_returns

    # Pin `end` so the frozen window is reproducible at this commit (not silently "today").
    rets = live_trend_book_returns(end=BASELINE_END)
    closes = fetch_universe_closes(LIVE_TREND_UNIVERSE, end=BASELINE_END)  # same cache/end = same data
    spy = closes["SPY"]
    sleeve = Sleeve(label="ch0_trend_baseline", component_type="risk_premium",
                    returns=rets, spy_prices=spy, notes="CH0a constant-gross ETF-trend baseline")
    rep = evaluate_sleeve(sleeve)
    return {
        "artifact": "CH0a — constant-gross ETF-trend baseline (the CH2 out-of-sample bar)",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "baseline_end": str(BASELINE_END),
        "universe": list(LIVE_TREND_UNIVERSE),
        "policy": "tsmom_backtest(TSMOMConfig()): inverse-vol, per-name+gross caps, weekly rebal, "
                  "NO governor overlays, book_vol_target=None (= all antifragile multipliers 1.0)",
        "data_fingerprint": _data_fingerprint(closes),
        "window": [rep.window_start, rep.window_end],
        "cpcv_gate": {
            "n_obs": rep.n_obs, "n_folds": rep.n_folds,
            "mean_sharpe": round(float(rep.mean_sharpe), 4),
            "path_sharpe_tstat": round(float(rep.path_sharpe_tstat), 4),
            "point_sr": round(float(rep.point_sr), 4),
            "hac_p_one_sided": round(float(rep.hac_p_one_sided), 4),
            "worst_regime_sharpe": (round(float(rep.worst_regime_sharpe), 4)
                                    if rep.worst_regime_sharpe is not None else None),
            # NOTE: paper/capital_passed are regime-WAIVED for component_type="risk_premium"
            # (ruler_v2 waives the worst-regime floor for premia) — they are NOT the CH0
            # deliverable. The CH2 bar is mean_sharpe + the regime_conditional profile below.
            "paper_passed": bool(rep.paper_passed), "capital_passed": bool(rep.capital_passed),
            "pass_fields_are_regime_waived": True,
        },
        "standalone": standalone_metrics(rets),
        "regime_conditional_sharpe": regime_conditional_sharpe(rets),
        "note": "CH2 gate (BOTH required): a governed/antifragile trend book must (a) BEAT this CPCV "
                "mean_sharpe out-of-sample with new params charged to the DSR trial count, AND (b) NOT "
                "regress the BEAR regime-conditional Sharpe (the CPCV folds under-sample stress, so a "
                "mean_sharpe-only gate would reward benign-regime gains while degrading the tail that "
                "antifragile sizing exists to protect). Fail either → ship nothing.",
    }


def main() -> int:
    baseline = build_baseline()
    with open(ARTIFACT, "w") as f:
        json.dump(baseline, f, indent=2, default=str)
    g = baseline["cpcv_gate"]
    print(f"CH0a baseline -> {ARTIFACT}")
    print(f"  window {baseline['window'][0]}..{baseline['window'][1]}  n_obs={g['n_obs']}")
    print(f"  CPCV mean_sharpe={g['mean_sharpe']}  path-t={g['path_sharpe_tstat']}  "
          f"point_SR={g['point_sr']}  worst_regime_SR={g['worst_regime_sharpe']}")
    print(f"  standalone: {baseline['standalone']}")
    print("  regime-conditional Sharpe:")
    for lab, m in baseline["regime_conditional_sharpe"].items():
        print(f"    {lab:<10} SR {m['sharpe']:+.2f}  ({m['frac_days']:.0%} of days)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
