"""universe_expansion.py — pre-registered universe-expansion study (2026-07-22).

Does adding a genuinely orthogonal macro ETF (or group) to the live 10-ETF trend universe improve
the constant-gross trend book on the SAME CPCV path CH0a was frozen on — beating baseline mean_sharpe
SIGNIFICANTLY and without regressing the BEAR tail? Report-only; reuses the CH2 DUAL-gate harness
(same bar, same paired-bootstrap significance) so the verdict is apples-to-apples with CH0a/CH2.

Pre-registration: docs/reference/UNIVERSE_EXPANSION_PREREGISTRATION_2026-07-22.md.
Run: `PYTHONPATH=. python -m app.research.universe_expansion` → results JSON + printed table.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from app.research.ch2_sizing import (
    BASELINE_END, build_base, load_baseline_bar, paired_delta_sharpe_pvalue,
    gate_decision, _evaluate)

RESULTS = Path("docs/reference/universe_expansion_results.json")
SIG_ALPHA = 0.05                      # paired-bootstrap significance threshold (prong 2)

# Candidates — orthogonal macro legs (NOT more US equity; deep 2007+ history verified).
INDIVIDUAL = ["USO", "UNG", "DBA", "FXE", "FXY", "VNQ", "TIP", "SLV"]
GROUPS: Dict[str, List[str]] = {
    "grp_commodities_granular": ["USO", "UNG", "DBA"],
    "grp_fx": ["FXE", "FXY"],
    "grp_real_assets": ["VNQ", "TIP"],
    "grp_kitchen_sink": ["USO", "UNG", "DBA", "FXE", "FXY", "VNQ", "TIP", "SLV"],
}


def _candidate_returns(added: List[str]) -> pd.Series:
    """Constant-gross trend book returns for baseline-10 + `added`, same engine/end as the baseline."""
    from scripts.walkforward.sleeves import LIVE_TREND_UNIVERSE, fetch_universe_closes
    from app.strategy.tsmom import TSMOMConfig, tsmom_backtest
    universe = list(LIVE_TREND_UNIVERSE) + list(added)
    closes = fetch_universe_closes(universe, end=BASELINE_END)
    # tsmom uses only the columns present; pin the config universe to the expanded set.
    return tsmom_backtest(closes, TSMOMConfig(universe=universe)).returns.dropna()


def _regime_bear(rets: pd.Series):
    from scripts.ch0_baseline import regime_conditional_sharpe
    prof = regime_conditional_sharpe(rets)
    return prof.get("BEAR", {}).get("sharpe"), prof


def _row(label: str, added: List[str], base: pd.Series, spy: pd.Series,
         bar: Dict[str, float], n_trials: int) -> dict:
    rets = _candidate_returns(added)
    rep = _evaluate(rets, spy, label=f"universe:{label}", n_trials=n_trials)
    bear, prof = _regime_bear(rets)
    pval = paired_delta_sharpe_pvalue(rets, base)
    beats, bear_ok, both = gate_decision(rep.mean_sharpe, bear, bar)
    significant = pval < SIG_ALPHA
    verdict = "PASS" if (both and significant) else "FAIL"
    return {
        "label": label, "added": added, "n_universe": 10 + len(added),
        "mean_sharpe": round(float(rep.mean_sharpe), 4),
        "delta_mean": round(float(rep.mean_sharpe) - bar["mean_sharpe"], 4),
        "min_fold_sharpe": (round(float(_mf), 4) if (_mf := getattr(rep, "min_fold_sharpe", None))
                            is not None and _mf == _mf else None),
        "bear_sharpe": (round(float(bear), 4) if bear is not None else None),
        "delta_bear": (round(float(bear) - bar["bear_sharpe"], 4)
                       if bear is not None and bar["bear_sharpe"] is not None else None),
        "paired_pvalue": round(float(pval), 4),
        "beats": beats, "bear_no_regression": bear_ok, "significant": significant,
        "verdict": verdict,
        "regime_profile": {k: round(float(v.get("sharpe")), 4)
                           for k, v in prof.items() if v.get("sharpe") is not None},
    }


def run_study() -> dict:
    base, spy, _ = build_base()
    bar = load_baseline_bar()
    n_trials = 1 + len(INDIVIDUAL) + len(GROUPS)     # multiplicity disclosed to the DSR secondary

    # In-run baseline sanity: re-evaluate the frozen 10-ETF book — should reproduce ~0.7009.
    base_rep = _evaluate(base, spy, label="universe:baseline10", n_trials=n_trials)
    base_bear, base_prof = _regime_bear(base)
    rows = [{
        "label": "baseline10", "added": [], "n_universe": 10,
        "mean_sharpe": round(float(base_rep.mean_sharpe), 4),
        "delta_mean": round(float(base_rep.mean_sharpe) - bar["mean_sharpe"], 4),
        "bear_sharpe": (round(float(base_bear), 4) if base_bear is not None else None),
        "paired_pvalue": None, "verdict": "BASELINE",
        "regime_profile": {k: round(float(v.get("sharpe")), 4)
                           for k, v in base_prof.items() if v.get("sharpe") is not None},
    }]

    for sym in INDIVIDUAL:
        rows.append(_row(f"add_{sym}", [sym], base, spy, bar, n_trials))
    for name, syms in GROUPS.items():
        rows.append(_row(name, syms, base, spy, bar, n_trials))

    passers = [r for r in rows if r.get("verdict") == "PASS"]
    out = {
        "artifact": "Universe-expansion study (pre-registered 2026-07-22)",
        "baseline_end": str(BASELINE_END), "baseline_bar": bar,
        "sig_alpha": SIG_ALPHA, "n_trials_disclosed": n_trials,
        "in_run_baseline_mean_sharpe": round(float(base_rep.mean_sharpe), 4),
        "n_candidates": len(rows) - 1, "n_pass": len(passers),
        "passers": [r["label"] for r in passers], "rows": rows,
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=2))
    return out


def _print(out: dict) -> None:
    print(f"\nUNIVERSE-EXPANSION STUDY — baseline bar mean_sharpe={out['baseline_bar']['mean_sharpe']}"
          f" BEAR={out['baseline_bar']['bear_sharpe']} | in-run baseline reproduced="
          f"{out['in_run_baseline_mean_sharpe']}\n")
    hdr = f"{'candidate':22s} {'nU':>3s} {'meanSR':>7s} {'d_mean':>7s} {'BEAR':>7s} {'d_bear':>7s} {'pval':>6s}  verdict"
    print(hdr)
    print("-" * len(hdr))
    for r in out["rows"]:
        print(f"{r['label']:22s} {r['n_universe']:>3d} {r.get('mean_sharpe', '-'):>7} "
              f"{r.get('delta_mean', '-'):>7} {str(r.get('bear_sharpe', '-')):>7} "
              f"{str(r.get('delta_bear', '-')):>7} {str(r.get('paired_pvalue', '-')):>6}  {r['verdict']}")
    print(f"\n-> {out['n_pass']}/{out['n_candidates']} PASS the DUAL gate + significance: "
          f"{out['passers'] or 'NONE'}\n")


if __name__ == "__main__":
    _print(run_study())
