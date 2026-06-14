"""
run_trend_broaden_rulerv2.py — re-score the P5 broadened-trend sleeve through the LIVE
Ruler-v2 gate (REPORT-ONLY, exploratory).

Workstream (1) of the 2026-06-14 research push: the old gate PARKED P5 trend-broadening
(broadened Sharpe 0.30 / t=1.31 vs the live 10-ETF sleeve's 0.72 / t=3.18). The question
this answers: does Ruler-v2 REVIVE it? We score the broadened sleeve as a
component_type="risk_premium" diversifier (so the worst-regime backstop is WAIVED — the
best case for a revival), through a proper CPCV via the proven SeriesReturnStrategy
adapter. Prints the Ruler-v2 PAPER + CAPITAL verdicts. Changes nothing; promotes nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402
load_dotenv()


def main() -> int:
    from scripts.run_trend_broadening import _fetch, BROADENED_SPEC
    from app.strategy.tsmom import TSMOMConfig, tsmom_backtest
    from scripts.walkforward.gate_calibration import (
        SeriesReturnStrategy, FULL_N_FOLDS, FULL_N_PATHS,
        FULL_PURGE_DAYS, FULL_EMBARGO_DAYS,
    )
    from scripts.walkforward.cpcv import run_cpcv
    from scripts.walkforward.regime import load_regime_map
    from app.research import ruler_v2

    prices = _fetch(BROADENED_SPEC["universe"])
    cfg = TSMOMConfig(**{k: (tuple(v) if isinstance(v, list) and k == "lookbacks" else v)
                         for k, v in BROADENED_SPEC.items()})
    rets = tsmom_backtest(prices, cfg).returns.dropna()
    print(f"[broaden] {prices.shape[1]} ETFs, broadened sleeve {rets.index[0].date()} -> "
          f"{rets.index[-1].date()} ({len(rets)} days)")

    spy = prices["SPY"] if "SPY" in prices.columns else None
    regime_map = load_regime_map(rets.index[0].date(), rets.index[-1].date())
    strat = SeriesReturnStrategy("trend_broadened", rets, spy_prices=spy,
                                 regime_map=regime_map)
    result = run_cpcv(strategy=strat, purge_days=FULL_PURGE_DAYS,
                      embargo_days=FULL_EMBARGO_DAYS, n_folds=FULL_N_FOLDS,
                      n_paths=FULL_N_PATHS, total_years=None)
    # Score as a declared diversifier so the worst-regime backstop is WAIVED (best case
    # for a revival — isolates whether significance/plausibility, not regime, is binding).
    result.component_type = "risk_premium"

    print("\n" + "=" * 78)
    print("  P5 BROADENED TREND — re-scored through Ruler-v2 (component_type=risk_premium)")
    print("=" * 78)
    print(f"  mean_sharpe={result.mean_sharpe:+.3f}  path_t={result.path_sharpe_tstat:+.2f}  "
          f"n_folds={result.n_folds}  n_oos={len(result.oos_returns_dated)}")
    for tier in ("paper", "capital"):
        d = ruler_v2.evaluate(result, tier=tier)
        passed = ruler_v2.gate_passed(result, tier=tier)
        failed = [k for k, (_v, ok) in d.items()
                  if not ok and k not in ruler_v2.INFORMATIONAL_KEYS]
        psr = d.get("point_sr_floor", ("?", None))[0]
        hac_p = d.get("hac_significance", ("?", None))[0]
        print(f"  {tier.upper():7} -> {'PASS' if passed else 'FAIL':4}  "
              f"point_SR={psr if isinstance(psr, str) else round(psr, 3)}  "
              f"hac_p={hac_p if isinstance(hac_p, str) else round(hac_p, 4)}  "
              f"failed={failed}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
