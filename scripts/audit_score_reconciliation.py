"""
G-Pre.1 — Score Reconciliation Audit.

Checks whether FactorPortfolioScorer rankings and model.predict rankings
are measuring the same thing on WF dates.

The WF has been run using scorer_instance=FactorPortfolioScorer (the hand-crafted
composite callable). The trained LambdaRank model (model.predict_with_vix) is a
completely separate system. If their rankings differ materially, all prior WF Sharpe
numbers describe the scorer, NOT the trained model.

Pass criteria (Opus 4.7 review):
  - Spearman >= 0.85 on >= 90% of sampled dates
  - Top-10 overlap >= 7/10 on >= 90% of sampled dates

Usage:
    python scripts/audit_score_reconciliation.py [--n-dates 20] [--model swing_v214]
"""
import argparse
import logging
import os
import random
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.ml.retrain_config import MAX_THREADS as _max_threads
os.environ.setdefault("OMP_NUM_THREADS", str(_max_threads))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(_max_threads))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [score_recon] %(message)s")
logger = logging.getLogger(__name__)


def _load_model(model_name: str):
    import pickle
    model_path = ROOT / "app" / "ml" / "models" / f"{model_name}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    with open(model_path, "rb") as f:
        return pickle.load(f)


def _download_symbols_data(symbols: list, start: str, end: str) -> dict:
    import yfinance as yf
    import pandas as pd
    logger.info("Downloading %d symbols %s->%s (batch) ...", len(symbols), start, end)
    try:
        raw = yf.download(symbols, start=start, end=end, auto_adjust=True,
                          progress=False, threads=True)
    except Exception as e:
        logger.error("Batch download failed: %s", e)
        return {}

    if raw.empty:
        return {}

    # Handle MultiIndex columns (yfinance >= 0.2)
    if isinstance(raw.columns, pd.MultiIndex):
        closes_df = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw.iloc[:, 0]
    else:
        closes_df = raw[["Close"]].rename(columns={"Close": symbols[0]})

    data = {}
    for sym in closes_df.columns:
        sym_closes = closes_df[sym].dropna()
        if len(sym_closes) > 50:
            # Build OHLCV DataFrame expected by feature engineer
            if isinstance(raw.columns, pd.MultiIndex):
                sym_df = pd.DataFrame({
                    "open": raw.get("Open", pd.DataFrame()).get(sym, sym_closes),
                    "high": raw.get("High", pd.DataFrame()).get(sym, sym_closes),
                    "low": raw.get("Low", pd.DataFrame()).get(sym, sym_closes),
                    "close": sym_closes,
                    "volume": raw.get("Volume", pd.DataFrame()).get(sym, pd.Series(dtype=float)),
                }).dropna(subset=["close"])
            else:
                sym_df = pd.DataFrame({"close": sym_closes})
            data[str(sym)] = sym_df

    logger.info("Downloaded %d/%d symbols with data", len(data), len(symbols))
    return data


def _compute_scorer_ranks(scorer, day, symbols_data, vix_history):
    try:
        results = scorer(day, symbols_data, vix_history)
    except Exception as e:
        logger.warning("Scorer failed on %s: %s", day, e)
        return None
    if not results:
        return None
    syms = [r[0] for r in results]
    confs = [r[1] for r in results]
    import pandas as pd
    return pd.Series(confs, index=syms).sort_values(ascending=False)


def _compute_model_ranks(model, day, symbols_data, vix_series):
    import pandas as pd
    from app.ml.features import FeatureEngineer
    fe = FeatureEngineer()
    features_by_symbol = {}
    for sym, df in symbols_data.items():
        import pandas as pd_inner
        bars_up_to = df[df.index < pd_inner.Timestamp(day)]
        if len(bars_up_to) < 60:
            continue
        try:
            feats = fe.engineer_features(
                sym, bars_up_to, fetch_fundamentals=False,
                as_of_date=day, regime_score=0.5,
                vix_history=vix_series,
                vix_value=None,
            )
            if feats:
                features_by_symbol[sym] = feats
        except Exception:
            pass

    if not features_by_symbol:
        return None

    sym_list = list(features_by_symbol.keys())
    model_feat_names = getattr(model, "feature_names", None)
    try:
        if model_feat_names:
            X = np.array([
                [features_by_symbol[s].get(f, 0.0) for f in model_feat_names]
                for s in sym_list
            ])
        else:
            X = np.array([list(features_by_symbol[s].values()) for s in sym_list])
        X = np.nan_to_num(X, nan=0.0)
        vix_now = float(vix_series.get(pd.Timestamp(day), vix_series.mean())) if vix_series is not None else 20.0
        _, probas = model.predict_with_vix(X, vix_level=vix_now)
    except Exception as e:
        logger.warning("Model predict failed on %s: %s", day, e)
        return None

    return pd.Series(probas, index=sym_list).sort_values(ascending=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-dates", type=int, default=20)
    parser.add_argument("--model", default="swing_v214")
    parser.add_argument("--top-n", type=int, default=10, help="Overlap check for top-N")
    args = parser.parse_args()

    from scipy.stats import spearmanr
    import pandas as pd

    logger.info("Loading model: %s", args.model)
    model = _load_model(args.model)
    logger.info("Model loaded. Features: %d", len(getattr(model, "feature_names", [])))

    from app.ml.factor_scorer import FactorPortfolioScorer
    scorer = FactorPortfolioScorer(
        top_n=20, long_short=False, vix_threshold=30.0, spy_ma_window=200,
        require_positive_momentum_days=0,
    )

    # Sample dates in WF range (last 6 years, skip most recent 30d)
    end_all = date.today() - timedelta(days=30)
    start_all = date.today() - timedelta(days=6 * 365)
    all_weekdays = [
        start_all + timedelta(days=i)
        for i in range((end_all - start_all).days)
        if (start_all + timedelta(days=i)).weekday() < 5
    ]
    sampled_dates = sorted(random.sample(all_weekdays, min(args.n_dates, len(all_weekdays))))
    logger.info("Sampled %d dates from %s to %s", len(sampled_dates), sampled_dates[0], sampled_dates[-1])

    # Download a universe subset (use top-100 R1K for speed)
    from app.utils.constants import RUSSELL_1000_TICKERS
    universe = list(RUSSELL_1000_TICKERS)[:100]
    data_start = (sampled_dates[0] - timedelta(days=400)).isoformat()
    data_end = (sampled_dates[-1] + timedelta(days=5)).isoformat()
    symbols_data = _download_symbols_data(universe, data_start, data_end)
    if len(symbols_data) < 30:
        logger.error("Too few symbols downloaded (%d) — aborting", len(symbols_data))
        sys.exit(1)

    # Build close matrix and VIX for scorer
    closes = pd.DataFrame({sym: df["close"] for sym, df in symbols_data.items()})
    try:
        import yfinance as yf
        vix_raw = yf.download("^VIX", start=data_start, end=data_end, auto_adjust=True, progress=False)
        vix_series = vix_raw["Close"].squeeze()
    except Exception:
        vix_series = None

    # Inject closes into scorer (it reads from symbols_data internally via 'close' key)
    results_log = []
    for day in sampled_dates:
        scorer_ranks = _compute_scorer_ranks(scorer, day, symbols_data, vix_series)
        model_ranks = _compute_model_ranks(model, day, symbols_data, vix_series)

        if scorer_ranks is None or model_ranks is None or len(scorer_ranks) < 5 or len(model_ranks) < 5:
            logger.debug("Skipping %s (insufficient rankings)", day)
            continue

        common = scorer_ranks.index.intersection(model_ranks.index)
        if len(common) < 5:
            logger.debug("Skipping %s (too few common symbols: %d)", day, len(common))
            continue

        sr_common = scorer_ranks[common]
        mr_common = model_ranks[common]
        sp = spearmanr(sr_common.values, mr_common.values).statistic

        top_n = args.top_n
        top_scorer = set(scorer_ranks.head(top_n).index)
        top_model = set(model_ranks.head(top_n).index)
        overlap = len(top_scorer & top_model)

        results_log.append({
            "date": day, "spearman": sp, "overlap": overlap, "n_common": len(common)
        })
        logger.info("  %s  spearman=%.3f  top-%d overlap=%d/%d  n_common=%d",
                    day, sp, top_n, overlap, top_n, len(common))

    if not results_log:
        logger.error("No valid dates to compare")
        sys.exit(1)

    spearman_vals = [r["spearman"] for r in results_log]
    overlap_vals = [r["overlap"] for r in results_log]
    top_n = args.top_n

    pass_spearman_pct = sum(1 for s in spearman_vals if s >= 0.85) / len(spearman_vals)
    pass_overlap_pct = sum(1 for o in overlap_vals if o >= int(top_n * 0.7)) / len(overlap_vals)

    print("\n=== SCORE RECONCILIATION RESULTS ===")
    print(f"  Dates evaluated:         {len(results_log)}")
    print(f"  Spearman mean:           {np.mean(spearman_vals):.3f}")
    print(f"  Spearman min:            {np.min(spearman_vals):.3f}")
    print(f"  Spearman >= 0.85:         {pass_spearman_pct:.0%} (need >= 90%)")
    print(f"  Top-{top_n} overlap mean: {np.mean(overlap_vals):.1f}/{top_n}")
    print(f"  Top-{top_n} overlap >= {int(top_n*0.7)}: {pass_overlap_pct:.0%} (need >= 90%)")

    gate_pass = pass_spearman_pct >= 0.90 and pass_overlap_pct >= 0.90
    verdict = "PASS" if gate_pass else "FAIL"
    print(f"\n  VERDICT: {verdict}")
    if not gate_pass:
        print("\n  WARNING: Scorer and model are measuring DIFFERENT things.")
        print("  All prior WF Sharpe numbers describe the FactorPortfolioScorer,")
        print("  NOT the trained LambdaRank model. Re-run WF experiments using")
        print("  model.predict to understand what the model actually does.")

    return 0 if gate_pass else 2


if __name__ == "__main__":
    sys.exit(main())
