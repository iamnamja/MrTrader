"""Phase RA — RebalanceMixin for PortfolioManager.

Adds REBALANCE execution mode: on a fixed cadence, score all universe symbols,
apply portfolio construction constraints, and emit close/open orders to rotate
the portfolio toward the top-ranked target set.

Injected into PortfolioManager via multiple inheritance. All state lives on
the PM instance (self), so there are no external dependencies beyond what PM
already holds.
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Sentinel — RebalanceMixin initialises this in _init_rebalance_state()
_NOT_SET = object()


class RebalanceMixin:
    """Mixin providing REBALANCE execution mode for PortfolioManager.

    The host class must provide:
      - self.model              — PortfolioSelectorModel (swing)
      - self.regime_detector    — RegimeDetector
      - self._get_universe()    — List[str]
      - self._fetch_swing_features() — Dict[str, Dict[str, float]]
      - self._alpaca            — data provider
      - self.logger             — Logger
      - self._normalize_for_inference(X, symbols, model) — np.ndarray
      - self._get_current_equity() — float
      - self._get_open_positions() — Dict[str, Any]   (symbol → position)
      - self._submit_close_proposal(symbol, reason) — coroutine
      - self._submit_open_proposal(symbol, dollar_size, rank, regime, score) — coroutine
    """

    # ------------------------------------------------------------------
    # State initialisation (call from PortfolioManager.__init__)
    # ------------------------------------------------------------------

    def _init_rebalance_state(self) -> None:
        self._rebalance_bar_counter: int = 0
        self._last_rebalance_date: Optional[date] = None

    # ------------------------------------------------------------------
    # Schedule check
    # ------------------------------------------------------------------

    def _should_rebalance(self, today: date) -> bool:
        """Return True if a rebalance cycle should run today."""
        from app.config import settings
        if self._last_rebalance_date is None:
            return True
        elapsed = (today - self._last_rebalance_date).days
        # Convert calendar days to approximate trading days (5/7 factor)
        approx_trading_days = elapsed * 5 / 7
        return approx_trading_days >= settings.rebalance_days

    # ------------------------------------------------------------------
    # Core rebalance cycle
    # ------------------------------------------------------------------

    async def _run_rebalance_cycle(self, today: date) -> None:
        """Score universe, compute target portfolio, emit close/open orders."""
        import numpy as np
        from app.config import settings
        from app.strategy.portfolio_construction import (
            apply_sector_cap,
            compute_equal_weights,
            compute_target_portfolio,
            liquidity_filter,
        )

        self.logger.info("REBALANCE cycle starting (target_positions=%d)", settings.target_positions)

        # --- 1. Score universe -------------------------------------------
        try:
            features_by_symbol = await self._get_loop().run_in_executor(
                None, self._fetch_swing_features
            )
        except Exception as exc:
            self.logger.error("REBALANCE: feature fetch failed — %s", exc)
            return

        symbols = sorted(features_by_symbol.keys())
        if not symbols:
            self.logger.warning("REBALANCE: empty universe, skipping")
            return

        model_feature_names = getattr(self.model, "feature_names", None)
        if model_feature_names:
            X = np.array([
                [features_by_symbol[s].get(f, 0.0) for f in model_feature_names]
                for s in symbols
            ])
        else:
            X = np.array([list(features_by_symbol[s].values()) for s in symbols])
        X = np.nan_to_num(X)
        X = self._normalize_for_inference(X, symbols, self.model)

        try:
            _, scores = self.model.predict(X)
        except Exception as exc:
            self.logger.error("REBALANCE: model prediction failed — %s", exc)
            return

        # Sort descending by score → ranked list
        ranked_pairs = sorted(zip(symbols, scores), key=lambda p: p[1], reverse=True)
        ranked_symbols = [s for s, _ in ranked_pairs]
        score_of = {s: float(sc) for s, sc in ranked_pairs}
        self.logger.info("REBALANCE: scored %d symbols, top5=%s",
                         len(ranked_symbols), ranked_symbols[:5])

        # --- 2. Liquidity filter -----------------------------------------
        try:
            bars_map = await self._get_loop().run_in_executor(
                None, lambda: self._fetch_bars_for_liquidity(symbols)
            )
            eligible = liquidity_filter(
                bars_map, today,
                min_avg_daily_dollar_vol=settings.min_avg_daily_dollar_vol,
            )
        except Exception as exc:
            self.logger.warning("REBALANCE: liquidity filter failed (%s) — using full universe", exc)
            eligible = set(ranked_symbols)

        ranked_eligible = [s for s in ranked_symbols if s in eligible]
        self.logger.info("REBALANCE: %d / %d symbols pass liquidity filter",
                         len(ranked_eligible), len(ranked_symbols))

        # --- 3. Sector cap -----------------------------------------------
        sector_map = self._get_sector_map(ranked_eligible)
        capped = apply_sector_cap(
            ranked_eligible, sector_map,
            cap=settings.sector_cap,
            n_target=settings.target_positions,
        )

        # --- 4. Target portfolio -----------------------------------------
        current_holdings = list(self._get_open_positions().keys())
        delta = compute_target_portfolio(
            capped, current_holdings,
            n_target=settings.target_positions,
            add_rank_threshold=settings.add_rank_threshold,
            drop_rank_threshold=settings.drop_rank_threshold,
        )
        self.logger.info(
            "REBALANCE: target=%d  add=%d  drop=%d  hold=%d",
            len(delta.target), len(delta.to_add), len(delta.to_drop), len(delta.held),
        )

        # --- 5. Regime gross exposure ------------------------------------
        regime = self.regime_detector.get_regime()
        gross_mult = self.regime_detector.gross_exposure_multiplier()
        equity = self._get_current_equity()

        weights = compute_equal_weights(delta.to_add, equity, gross_mult)

        # --- 6. Cap rotation at rebalance_max_rotation_pct ---------------
        max_rotations = max(1, int(len(delta.target) * settings.rebalance_max_rotation_pct))
        to_drop = delta.to_drop[:max_rotations]
        to_add = delta.to_add[:max_rotations]

        # --- 7. Emit close orders ----------------------------------------
        for sym in to_drop:
            try:
                await self._submit_close_proposal(sym, reason="REBALANCE_DROP")
            except Exception as exc:
                self.logger.error("REBALANCE: close proposal failed for %s — %s", sym, exc)

        # --- 8. Emit open orders -----------------------------------------
        rank_of = {s: i + 1 for i, s in enumerate(ranked_eligible)}
        for sym in to_add:
            dollar_size = weights.get(sym, 0.0)
            if dollar_size <= 0:
                continue
            rank = rank_of.get(sym, len(ranked_eligible))
            try:
                await self._submit_open_proposal(
                    sym, dollar_size,
                    rank=rank,
                    regime=regime,
                    score=score_of.get(sym, 0.0),
                )
            except Exception as exc:
                self.logger.error("REBALANCE: open proposal failed for %s — %s", sym, exc)

        self._last_rebalance_date = today
        self.logger.info(
            "REBALANCE cycle complete — regime=%s gross_mult=%.2f closed=%d opened=%d",
            regime, gross_mult, len(to_drop), len(to_add),
        )

    # ------------------------------------------------------------------
    # Helpers (thin wrappers — host PM provides the real implementation)
    # ------------------------------------------------------------------

    def _get_loop(self):
        import asyncio
        return asyncio.get_event_loop()

    def _fetch_bars_for_liquidity(self, symbols: List[str]) -> Dict[str, Any]:
        """Fetch 65 days of daily bars for liquidity check."""
        try:
            return self._alpaca.get_bars_batch(symbols, "1D", 65)
        except Exception:
            return {}

    def _get_sector_map(self, symbols: List[str]) -> Dict[str, str]:
        """Return GICS sector for each symbol from DB or static map."""
        try:
            from app.data.sector_map import get_sector_map
            return get_sector_map(symbols)
        except Exception:
            return {}

    def _get_current_equity(self) -> float:
        try:
            acct = self._alpaca.get_account()
            return float(acct.equity)
        except Exception:
            return 20_000.0  # fallback

    def _get_open_positions(self) -> Dict[str, Any]:
        try:
            positions = self._alpaca.get_all_positions()
            return {p.symbol: p for p in positions}
        except Exception:
            return {}

    async def _submit_close_proposal(self, symbol: str, reason: str) -> None:
        self.logger.info("REBALANCE CLOSE: %s  reason=%s", symbol, reason)

    async def _submit_open_proposal(
        self,
        symbol: str,
        dollar_size: float,
        rank: int,
        regime: str,
        score: float,
    ) -> None:
        self.logger.info(
            "REBALANCE OPEN: %s  $%.0f  rank=%d  regime=%s  score=%.4f",
            symbol, dollar_size, rank, regime, score,
        )
