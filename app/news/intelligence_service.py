"""
News Intelligence Service — public API wiring Tier 1 + Tier 2.

Usage:
    from app.news.intelligence_service import nis
    ctx = nis.get_macro_context()          # MacroContext, cached for the day
    sig = nis.get_stock_signal("AAPL")     # NewsSignal, cached ~1 hour
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Optional

from app.news.signal import (
    ActionPolicy,
    MacroContext,
    MacroEventSignal,
    NewsSignal,
)

logger = logging.getLogger(__name__)

# ── Day-level cache (avoids re-calling Tier 1 many times per day) ─────────────
_macro_cache: dict[str, MacroContext] = {}   # date_str → MacroContext
_stock_cache: dict[str, NewsSignal] = {}     # symbol → NewsSignal (TTL via evaluated_at)
_STOCK_SIGNAL_TTL_SECS = 3600               # 1 hour before re-fetching


class NewsIntelligenceService:
    """
    Orchestrates Finnhub data fetching + LLM scoring into typed signals.

    Fail-open: every method returns a neutral fallback on any error.
    """

    # ── Tier 1: Macro Context ─────────────────────────────────────────────────

    def get_macro_context(self, force_refresh: bool = False) -> MacroContext:
        """
        Return today's MacroContext (Tier 1).

        Cached for the calendar day.  Pass force_refresh=True to re-fetch
        (e.g. after FOMC announcement time passes and actual is now available).
        """
        today = date.today().isoformat()
        if not force_refresh and today in _macro_cache:
            return _macro_cache[today]

        ctx = self._build_macro_context()
        _macro_cache[today] = ctx
        return ctx

    def _build_macro_context(self) -> MacroContext:
        try:
            from app.news.sources.finnhub_source import fetch_economic_calendar
            from app.news.llm_scorer import macro_classify

            events = fetch_economic_calendar(days_ahead=1, min_impact="medium")
            result = macro_classify(events)

            if result is None:
                logger.info("NIS Tier 1: LLM unavailable — using neutral MacroContext")
                return MacroContext.neutral()

            event_signals = self._build_event_signals(events, result)
            ctx = MacroContext(
                as_of=datetime.now(timezone.utc),
                events_today=event_signals,
                block_new_entries=bool(result.get("block_new_entries", False)),
                global_sizing_factor=float(result.get("sizing_factor", 1.0)),
                overall_risk=result.get("risk_level", "LOW"),
                rationale=result.get("rationale", ""),
            )
            logger.info(
                "NIS Tier 1: risk=%s sizing=%.2f block=%s — %s",
                ctx.overall_risk, ctx.global_sizing_factor,
                ctx.block_new_entries, ctx.rationale,
            )
            self._persist_macro(ctx, events, result)
            return ctx

        except Exception as exc:
            logger.warning("NIS Tier 1 failed (using neutral): %s", exc)
            return MacroContext.neutral()

    def _build_event_signals(
        self, events: list[dict], llm_result: dict
    ) -> list[MacroEventSignal]:
        """Map each Finnhub event + the day-level LLM result to MacroEventSignal."""
        signals = []
        for e in events:
            evt_time = e["event_time"]
            time_str = evt_time.strftime("%H:%M UTC") if hasattr(evt_time, "strftime") else str(evt_time)[:16]
            signals.append(MacroEventSignal(
                event_type=e["event_type"],
                event_time=time_str,
                risk_level=llm_result.get("risk_level", "LOW"),
                direction=llm_result.get("direction", "NEUTRAL"),
                sizing_factor=float(llm_result.get("sizing_factor", 1.0)),
                block_new_entries=bool(llm_result.get("block_new_entries", False)),
                consensus_summary=llm_result.get("consensus_summary", ""),
                rationale=llm_result.get("rationale", ""),
                scorer_tier="haiku",
                evaluated_at=datetime.now(timezone.utc),
                already_priced_in=e.get("estimate") == e.get("prior"),
            ))
        return signals

    def _persist_macro(self, ctx: MacroContext, events: list[dict], raw: dict) -> None:
        try:
            from app.database.session import get_session
            from app.database.models import MacroSignalCache
            with get_session() as db:
                today_str = date.today().isoformat()
                existing = db.query(MacroSignalCache).filter_by(date=today_str).first()
                if existing:
                    return  # already stored today
                db.add(MacroSignalCache(
                    date=today_str,
                    prompt_version="macro_v1",
                    risk_level=ctx.overall_risk,
                    direction=raw.get("direction", "NEUTRAL"),
                    sizing_factor=ctx.global_sizing_factor,
                    block_new_entries=ctx.block_new_entries,
                    rationale=ctx.rationale,
                    events_payload=events if isinstance(events, list) else [],
                ))
                db.commit()
        except Exception as exc:
            logger.debug("MacroSignalCache persist failed (non-fatal): %s", exc)

    # ── Tier 2: Stock Signal ──────────────────────────────────────────────────

    def get_stock_signal(
        self,
        symbol: str,
        sector: str = "Unknown",
        lookback_hours: int = 24,
        macro_context: Optional[MacroContext] = None,
    ) -> NewsSignal:
        """
        Return a NewsSignal for symbol (Tier 2).

        Cached ~1 hour per symbol.  Returns NewsSignal.neutral() on any error.
        """
        import time as _time
        cached = _stock_cache.get(symbol)
        if cached is not None:
            age = (_time.time() - cached.evaluated_at.replace(tzinfo=timezone.utc).timestamp()
                   if cached.evaluated_at.tzinfo else
                   _time.time() - cached.evaluated_at.timestamp())
            if age < _STOCK_SIGNAL_TTL_SECS:
                return cached

        sig = self._build_stock_signal(symbol, sector, lookback_hours, macro_context)
        _stock_cache[symbol] = sig
        return sig

    def _build_stock_signal(
        self,
        symbol: str,
        sector: str,
        lookback_hours: int,
        macro_context: Optional[MacroContext],
    ) -> NewsSignal:
        try:
            from app.news.sources.finnhub_source import fetch_company_news
            from app.news.llm_scorer import stock_score

            articles = fetch_company_news(symbol, lookback_hours=lookback_hours)
            macro_summary = (
                macro_context.rationale
                if macro_context and macro_context.rationale
                else "No significant macro events today"
            )

            result = stock_score(
                symbol=symbol,
                articles=articles,
                sector=sector,
                macro_context_summary=macro_summary,
                lookback_hours=lookback_hours,
            )

            if result is None:
                return NewsSignal.neutral(symbol)

            sig = NewsSignal(
                symbol=symbol,
                evaluated_at=datetime.now(timezone.utc),
                direction_score=result["direction_score"],
                materiality_score=result["materiality_score"],
                downside_risk_score=result["downside_risk_score"],
                upside_catalyst_score=result["upside_catalyst_score"],
                confidence=result["confidence"],
                already_priced_in_score=result["already_priced_in_score"],
                action_policy=result.get("action_policy", "ignore"),
                sizing_multiplier=result["sizing_multiplier"],
                rationale=result.get("rationale", ""),
                top_headlines=[a["headline"] for a in articles[:5]],
                scorer_tier="haiku",
            )
            logger.debug(
                "NIS Tier 2 %s: policy=%s size=%.2f dir=%.2f — %s",
                symbol, sig.action_policy, sig.sizing_multiplier,
                sig.direction_score, sig.rationale,
            )
            self._persist_stock(sig, result)
            return sig

        except Exception as exc:
            logger.warning("NIS Tier 2 %s failed (using neutral): %s", symbol, exc)
            return NewsSignal.neutral(symbol)

    def _persist_stock(self, sig: NewsSignal, raw: dict) -> None:
        try:
            from app.database.session import get_session
            from app.database.models import NewsSignalCache
            import hashlib, json
            cache_key = hashlib.sha256(
                f"stock_v1|{sig.symbol}|{sig.rationale}".encode()
            ).hexdigest()[:64]
            with get_session() as db:
                existing = db.query(NewsSignalCache).filter_by(cache_key=cache_key).first()
                if existing:
                    return
                db.add(NewsSignalCache(
                    symbol=sig.symbol,
                    cache_key=cache_key,
                    prompt_version="stock_v1",
                    direction_score=sig.direction_score,
                    materiality_score=sig.materiality_score,
                    downside_risk_score=sig.downside_risk_score,
                    upside_catalyst_score=sig.upside_catalyst_score,
                    confidence=sig.confidence,
                    already_priced_in_score=sig.already_priced_in_score,
                    action_policy=sig.action_policy,
                    sizing_multiplier=sig.sizing_multiplier,
                    rationale=sig.rationale,
                    top_headlines=sig.top_headlines,
                ))
                db.commit()
        except Exception as exc:
            logger.debug("NewsSignalCache persist failed (non-fatal): %s", exc)

    # ── Batch helpers ─────────────────────────────────────────────────────────

    def get_stock_signals_batch(
        self,
        symbols: list[str],
        sector_map: Optional[dict[str, str]] = None,
        macro_context: Optional[MacroContext] = None,
    ) -> dict[str, NewsSignal]:
        """Score multiple symbols; returns {symbol: NewsSignal}."""
        sector_map = sector_map or {}
        return {
            sym: self.get_stock_signal(
                sym,
                sector=sector_map.get(sym, "Unknown"),
                macro_context=macro_context,
            )
            for sym in symbols
        }

    def invalidate_stock_cache(self, symbol: Optional[str] = None) -> None:
        """Clear in-memory stock signal cache (all or one symbol)."""
        if symbol:
            _stock_cache.pop(symbol, None)
        else:
            _stock_cache.clear()

    def invalidate_macro_cache(self) -> None:
        """Clear day-level macro cache to force re-fetch."""
        _macro_cache.clear()


# Singleton
nis = NewsIntelligenceService()
