"""
Phase C3 — Build and validate RegimeRuleScorer (regime_v4).

Replaces regime_v3 (broken: 100% NEUTRAL from Phase A4 diagnostic).

Validation:
  - 2025-02-01 → 2025-05-13 (tariff shock period): expect ≥60% RISK_OFF/NEUTRAL days
  - Full history: print regime label distribution by year

Saves:
  app/ml/models/regime_model_v4.pkl
  app/ml/models/regime_model_v4_meta.json
  data/diagnostics/regime_v4/<timestamp>/validation.md

Usage:
    python scripts/build_regime_v4.py
    python scripts/build_regime_v4.py --vix-cap 20 --no-save   # test threshold
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import numpy as np
import pandas as pd

from app.ml.regime_classifier import RegimeRuleScorer
from app.notifications import notifier as _notifier

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_DIR = ROOT / "app" / "ml" / "models"
DAILY_CACHE = ROOT / "data" / "cache" / "daily"
MACRO_PATHS = [
    ROOT / "data" / "macro" / "macro_history.parquet",
    ROOT / "data" / "macro_history.parquet",
]
OUT_BASE = ROOT / "data" / "diagnostics" / "regime_v4"


def load_macro_for_regime(start: str = "2019-01-01") -> pd.DataFrame:
    """Load macro_history; supplement with SPY/VIX from daily cache if columns missing."""
    macro = None
    for p in MACRO_PATHS:
        if p.exists():
            try:
                df = pd.read_parquet(p)
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.set_index("date")
                elif not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                macro = df.loc[start:]
                logger.info("Loaded macro history: %d rows from %s", len(macro), p)
                break
            except Exception as e:
                logger.warning("Could not load %s: %s", p, e)

    # Supplement with SPY from daily cache if not already present
    spy_path = DAILY_CACHE / "SPY.parquet"
    if spy_path.exists():
        spy_df = pd.read_parquet(spy_path)
        spy_df.index = pd.to_datetime(spy_df.index)
        spy_close = spy_df["close"].loc[start:]
        spy_ma200 = spy_close.rolling(200).mean()
        spy_above = (spy_close > spy_ma200).astype(float)

        if macro is None:
            macro = pd.DataFrame(index=spy_close.index)

        if "spy_close" not in macro.columns:
            macro["spy_close"] = spy_close.reindex(macro.index)
        if "spy_ma200" not in macro.columns:
            macro["spy_ma200"] = spy_ma200.reindex(macro.index)
        macro["spy_above_ma200"] = spy_above.reindex(macro.index)

    # Add VIX from daily cache
    vix_path = DAILY_CACHE / "VIX.parquet"
    if not vix_path.exists():
        vix_path = DAILY_CACHE / "^VIX.parquet"
    if vix_path.exists() and (macro is None or "vix" not in macro.columns):
        vix_df = pd.read_parquet(vix_path)
        vix_df.index = pd.to_datetime(vix_df.index)
        vix_close = vix_df["close"].loc[start:]
        if macro is None:
            macro = pd.DataFrame(index=vix_close.index)
        macro["vix"] = vix_close.reindex(macro.index)

    if macro is None or macro.empty:
        raise RuntimeError("Could not load any macro data")

    # If VIX still missing, use 20 as neutral default
    if "vix" not in macro.columns:
        logger.warning("VIX not found in macro data — using default 20.0")
        macro["vix"] = 20.0

    if "spy_above_ma200" not in macro.columns:
        logger.warning("SPY/MA200 not found — defaulting to True (optimistic)")
        macro["spy_above_ma200"] = 1.0

    return macro


def validate(scorer: RegimeRuleScorer, macro: pd.DataFrame) -> dict:
    """Run validation and print regime distribution."""
    scores = scorer.predict_proba_series(macro)
    labels = scores.apply(scorer.label)

    # Full period distribution
    full_dist = labels.value_counts(normalize=True).to_dict()

    # Tariff shock 2025-02 → 2025-05
    shock_mask = (scores.index >= "2025-02-01") & (scores.index <= "2025-05-13")
    shock_labels = labels[shock_mask]
    shock_dist = shock_labels.value_counts(normalize=True).to_dict()
    shock_risk_off_pct = shock_dist.get("RISK_OFF", 0) + shock_dist.get("NEUTRAL", 0)

    # Per-year distribution
    yearly = {}
    for yr in sorted(labels.index.year.unique()):
        yr_labels = labels[labels.index.year == yr]
        yearly[int(yr)] = {k: round(v, 3) for k, v in yr_labels.value_counts(normalize=True).items()}

    # Regime v3 comparison (was 100% NEUTRAL)
    logger.info("Full period distribution: %s", full_dist)
    logger.info("Tariff shock (2025-02 to 2025-05): RISK_OFF/NEUTRAL = %.1f%%", shock_risk_off_pct * 100)

    gate_pass = shock_risk_off_pct >= 0.60
    logger.info("Validation gate (≥60%% RISK_OFF/NEUTRAL in tariff shock): %s",
                "PASS" if gate_pass else "FAIL (adjust thresholds)")

    return {
        "full_distribution": {k: round(v, 3) for k, v in full_dist.items()},
        "shock_risk_off_pct": round(shock_risk_off_pct, 3),
        "yearly_distribution": yearly,
        "gate_pass": gate_pass,
        "n_days": len(labels),
        "date_range": f"{labels.index.min().date()} → {labels.index.max().date()}",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vix-cap",       type=float, default=25.0)
    parser.add_argument("--breadth-thresh", type=float, default=0.40)
    parser.add_argument("--no-save",       action="store_true")
    parser.add_argument("--start",         default="2019-01-01")
    args = parser.parse_args()

    t0 = time.time()
    scorer = RegimeRuleScorer(vix_cap=args.vix_cap, breadth_thresh=args.breadth_thresh)
    logger.info("Built RegimeRuleScorer %s (vix_cap=%.0f, breadth_thresh=%.2f)",
                scorer.VERSION, scorer.vix_cap, scorer.breadth_thresh)

    macro = load_macro_for_regime(args.start)
    results = validate(scorer, macro)
    runtime_s = time.time() - t0

    # ── Output ──
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = OUT_BASE / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    yr_rows = "\n".join(
        f"| {yr} | {d.get('BULL', 0)*100:.0f}% | {d.get('NEUTRAL', 0)*100:.0f}% | {d.get('RISK_OFF', 0)*100:.0f}% |"
        for yr, d in sorted(results["yearly_distribution"].items())
    )
    verdict = "PASS" if results["gate_pass"] else "FAIL"
    md = f"""# RegimeRuleScorer v4 — Validation Report

**Generated:** {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}

## Configuration
- VIX cap: {args.vix_cap}
- Breadth threshold: {args.breadth_thresh}
- SPY weight: {scorer.spy_weight}, VIX weight: {scorer.vix_weight}, Breadth weight: {scorer.breadth_weight}

## Full Period Distribution ({results['date_range']})
| Label | Fraction |
|---|---|
{chr(10).join(f"| {k} | {v*100:.1f}% |" for k, v in sorted(results['full_distribution'].items()))}

## Validation Gate: {verdict}
- Tariff shock (2025-02 → 2025-05) RISK_OFF/NEUTRAL: **{results['shock_risk_off_pct']*100:.1f}%** (need ≥60%)
- Gate {'PASSED' if results['gate_pass'] else 'FAILED'}

## Comparison: regime_v3 (Phase A4)
- regime_v3: NEUTRAL **100%** of all days → gate broken, useless
- regime_v4: {results['full_distribution'].get('NEUTRAL', 0)*100:.1f}% NEUTRAL, {results['full_distribution'].get('RISK_OFF', 0)*100:.1f}% RISK_OFF, {results['full_distribution'].get('BULL', 0)*100:.1f}% BULL

## Per-Year Regime Distribution

| Year | BULL | NEUTRAL | RISK_OFF |
|---|---|---|---|
{yr_rows}

## Notes
- Rule-based (no ML): interpretable, no train/test split needed
- SPY>200d MA signal driven by Phase A3 B2 (Sharpe +0.808 with this single rule alone)
- VIX cap prevents entry during volatility spikes
- Breadth (if available) adds market internals signal
"""
    (out_dir / "validation.md").write_text(md, encoding="utf-8")
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    logger.info("Artifacts: %s", out_dir)

    # ── Save model ──
    if not args.no_save:
        model_path = MODEL_DIR / "regime_model_v4.pkl"
        scorer.save(model_path)
        logger.info("Saved regime_v4 to %s", model_path)

    # ── Notify ──
    outcome = (
        f"PASS — {results['shock_risk_off_pct']*100:.1f}% RISK_OFF/NEUTRAL in tariff shock (gate ≥60%)"
        if results["gate_pass"]
        else f"FAIL — only {results['shock_risk_off_pct']*100:.1f}% RISK_OFF/NEUTRAL (need ≥60%)"
    )
    _notifier.enqueue("diag_complete", {
        "script":   "build_regime_v4.py (Phase C3)",
        "duration": f"{runtime_s:.0f}s",
        "outcome":  outcome,
        "artifacts": [str(p) for p in sorted(out_dir.iterdir())],
        "summary_html": f"<pre>{md[:2000]}</pre>",
    })

    print(f"\n{'='*60}")
    print(f"  Regime v4 — {verdict}")
    print(f"  Tariff shock RISK_OFF/NEUTRAL: {results['shock_risk_off_pct']*100:.1f}%")
    print(f"  Full period: {results['full_distribution']}")
    print(f"{'='*60}\n")
    return 0 if results["gate_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
