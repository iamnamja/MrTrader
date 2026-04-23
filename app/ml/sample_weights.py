"""
Sample weighting for the swing model training pipeline.

Five multiplicative weight components — each normalised to mean=1 so they
don't inflate or deflate the effective sample count:

  recency         Newer windows get up to 2.5x weight vs oldest.
                  Captures regime drift: a 2024 bull-market pattern matters
                  more for predicting 2025 than a 2020 COVID-crash pattern.

  outcome_margin  Decisive outcomes (price ran far past target/stop) are
                  stronger signal than ones that barely triggered.
                  A trade that hit +6% in a 3%-target window is much more
                  informative than one that scraped +3.01%.

  vol_regime      Upweight windows whose volatility percentile is close to
                  the current market vol regime.  Trains the model on patterns
                  that are relevant to today's conditions.

  liquidity       Higher average volume = cleaner price signals, less noise.
                  Low-liquidity windows produce noisy labels.

  sector_diversity  Downweight samples from over-represented sectors so no
                  sector (e.g. 20 bank stocks) dominates the gradient.
"""

import logging
from collections import Counter
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def compute_sample_weights(
    window_indices: List[int],       # 0-based position of each sample in the full window list
    total_windows: int,              # total number of distinct time windows
    outcome_returns: List[float],    # actual return achieved (pnl / entry_price)
    vol_percentiles: List[float],    # vol_percentile_52w at feature time
    avg_volumes: List[float],        # mean daily volume over feature window
    sector_labels: List[str],        # sector string per sample (e.g. "Technology")
    vix_regime_buckets: Optional[List[float]] = None,  # vix_regime_bucket [0,1] per sample
    target_pct: float = 0.03,        # label target threshold (for outcome margin calc)
    current_vol_percentile: float = 0.5,  # today's vol percentile (for regime match)
    recency_max_multiplier: float = 2.5,  # oldest window gets weight 1, newest gets this
    vol_match_bandwidth: float = 0.20,    # Gaussian bandwidth for vol regime matching
    low_vix_multiplier: float = 1.5,     # max upweight for low-VIX windows (bucket≈0)
) -> np.ndarray:
    """
    Compute per-sample importance weights combining 5 factors.

    Returns normalised weights array (mean ≈ 1.0) of shape (n_samples,).
    """
    n = len(window_indices)
    if n == 0:
        return np.array([])

    # ── 1. Recency ────────────────────────────────────────────────────────────
    # Linear ramp: oldest window → 1.0, newest → recency_max_multiplier
    denom = max(total_windows - 1, 1)
    recency = np.array([
        1.0 + (idx / denom) * (recency_max_multiplier - 1.0)
        for idx in window_indices
    ])

    # ── 2. Outcome margin ─────────────────────────────────────────────────────
    # Weight by how decisively the label was determined.
    # For winners: excess return above target; for losers: depth of stop hit.
    # Floor at 0.5 so no sample gets near-zero weight.
    margin = np.array([
        max(0.5, 1.0 + abs(r) / max(target_pct, 1e-6))
        for r in outcome_returns
    ])

    # ── 3. Volatility regime match ────────────────────────────────────────────
    # Gaussian kernel: samples with vol_percentile close to current regime
    # get higher weight. Bandwidth ~0.20 means ±20pp vol gets ~60% weight.
    vol_arr = np.array(vol_percentiles)
    vol_match = np.exp(-0.5 * ((vol_arr - current_vol_percentile) / vol_match_bandwidth) ** 2)
    vol_match = np.clip(vol_match, 0.3, 1.0)  # floor at 0.3 — don't zero out cross-regime

    # ── 4. Liquidity ──────────────────────────────────────────────────────────
    # log-scale so a 10x volume difference gives ~2.3x weight, not 10x.
    vol_arr2 = np.array([max(v, 1.0) for v in avg_volumes])
    med_vol = float(np.median(vol_arr2))
    liquidity = np.log1p(vol_arr2 / med_vol)
    liquidity = np.clip(liquidity / np.mean(liquidity), 0.5, 2.0)

    # ── 5. Sector diversity ───────────────────────────────────────────────────
    # Weight each sample by 1 / count_of_its_sector, then normalise.
    sector_counts = Counter(sector_labels)
    sector_w = np.array([1.0 / max(sector_counts[s], 1) for s in sector_labels])

    # ── 6. VIX regime (Phase 26a) ─────────────────────────────────────────────
    # Low-VIX windows (bucket≈0 = calm market) get up to low_vix_multiplier weight.
    # High-VIX windows (bucket≈1 = fear spike) get weight=1.0 (no upweight).
    # Rationale: in high-VIX regimes, 70% of trades stop out regardless of signal
    # quality; downweighting them focuses gradient on learnable patterns.
    if vix_regime_buckets is not None and len(vix_regime_buckets) == n:
        vix_arr = np.array(vix_regime_buckets, dtype=float)
        # bucket=0 → weight=low_vix_multiplier, bucket=1 → weight=1.0
        vix_regime_w = 1.0 + (1.0 - np.clip(vix_arr, 0.0, 1.0)) * (low_vix_multiplier - 1.0)
    else:
        vix_regime_w = np.ones(n)

    # ── Combine multiplicatively ──────────────────────────────────────────────
    combined = recency * margin * vol_match * liquidity * sector_w * vix_regime_w

    # Normalise to mean=1 so total effective sample count stays ~n
    mean_w = combined.mean()
    if mean_w <= 0 or not np.isfinite(mean_w):
        return np.ones(n, dtype=np.float32)
    combined = combined / mean_w

    # Final safety: clip to [0.1, 10] and ensure all positive finite
    combined = np.clip(combined, 0.1, 10.0)
    if not np.all(np.isfinite(combined)):
        return np.ones(n, dtype=np.float32)

    logger.debug(
        "Sample weights: recency=[%.2f,%.2f] margin=[%.2f,%.2f] "
        "vol=[%.2f,%.2f] liq=[%.2f,%.2f] sector=[%.2f,%.2f] vix=[%.2f,%.2f] combined=[%.2f,%.2f]",
        recency.min(), recency.max(),
        margin.min(), margin.max(),
        vol_match.min(), vol_match.max(),
        liquidity.min(), liquidity.max(),
        sector_w.min(), sector_w.max(),
        vix_regime_w.min(), vix_regime_w.max(),
        combined.min(), combined.max(),
    )

    return combined.astype(np.float32)
