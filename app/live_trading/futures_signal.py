"""
futures_signal.py — Alpha-v10 R1.3: LIVE target weights for the futures book (carry + xsmom).

Extracts the CURRENT target-weight vector from the SAME validated construction as the research
`futures_book` (0.5·carry + 0.5·xsmom, each via `carry_backtest`'s cross-sectional z-score × inverse-vol,
clipped, gross-capped, book-vol-targeted). The book's levered position panel `W` is exposed by
`carry_backtest(return_weights=True)`; the live target = 0.5·carry_W.iloc[-1] + 0.5·xsmom_W.iloc[-1],
computed cross-sectionally over the FULL liquid universe (so the z-scores are correct) then FILTERED to
the IBKR-tradeable instruments we actually hold contracts for.

SHADOW / observational only. This feeds `run_futures_rebalance` ONLY when `ibkr.futures_signal_live` is
on (default OFF), and that rebalance places NOTHING (P2.3/R1 shadow). Caveats for the live cutover
(documented, NOT yet addressed — nothing trades on this today):
  - `W.iloc[-1]` is the book's position on the LAST DATA date (a ~1 trading-day PIT lag vs a true
    forward target) — fine for shadow observation; a forward-target extraction is a before-live refinement.
  - The Norgate mirror is the data source and may lag; live needs a fresh price/carry feed.
  - The carry/xSMOM edge must be RE-VALIDATED on the hybrid roll before any live futures CAPITAL.
  - This returns the 16-MARKET SLICE of the full ~76-market book: z-scores are computed cross-sectionally
    over the FULL liquid universe (correct signal), but we only trade the instruments with IBKR contracts,
    so the gross on our subset is a fraction of the book's. Whether the edge survives on the tradeable
    subset (and whether to renormalise the subset's gross) is a before-live sizing question — NOT done
    here (faithful raw slice).
Never raises — returns {} on any failure (no weights → no would-be orders).
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Sequence

log = logging.getLogger(__name__)


def current_target_weights(*, universe: Optional[Sequence[str]] = None, cfg=None,
                           min_weight: float = 0.01) -> Dict[str, float]:
    """The current carry+xsmom book target weights, keyed by canonical `instrument_id` (FUT.<root>),
    restricted to the IBKR-tradeable futures universe. `min_weight` drops dust. Returns {} on any error."""
    try:
        from app.live_trading import instrument_master as im
        from app.research import futures_carry as fc
        from app.research import futures_data as fd
        from app.research import futures_factors as ff

        uni = list(universe) if universe is not None else fd.liquid_universe()
        cfg = cfg or fc.CarryConfig()
        returns = fd.returns_panel(uni)
        if returns is None or returns.empty:
            return {}
        carry = fc.carry_panel(uni)
        mom = ff.xs_momentum_signal(fd.synthetic_price_panel(uni)).reindex_like(returns)

        carry_w = fc.carry_backtest(returns, carry, cfg, return_weights=True)
        xsmom_w = fc.carry_backtest(returns, mom, cfg, return_weights=True)
        if carry_w is None or xsmom_w is None or carry_w.empty or xsmom_w.empty:
            return {}
        combined = 0.5 * carry_w.iloc[-1] + 0.5 * xsmom_w.iloc[-1]   # per-market weight (aligned by market)

        # map each tradeable Norgate market -> our canonical instrument_id; drop dust + non-tradeable.
        by_root = {str(inst.root).upper(): iid for iid, inst in im.futures_instruments().items()}
        out: Dict[str, float] = {}
        for mkt, w in combined.items():
            iid = by_root.get(str(mkt).upper())
            if iid is not None and w == w and abs(float(w)) >= min_weight:   # w==w drops NaN
                out[iid] = round(float(w), 6)
        log.info("ibkr futures LIVE signal: %d/%d tradeable target weight(s) from carry+xsmom book "
                 "(gross %.3f) — SHADOW, places nothing", len(out), len(by_root),
                 sum(abs(v) for v in out.values()))
        return out
    except Exception as e:  # noqa: BLE001 — a signal failure must NEVER crash the rebalance
        log.warning("ibkr futures signal extraction failed -> {} (no orders): %s", e)
        return {}
