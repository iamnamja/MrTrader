"""
Pre-market Intelligence — situational awareness before the first trade of day.

Runs once between 09:00–09:25 ET each trading day via the orchestrator.

Responsibilities:
  20.1  Economic calendar gate — flag high-impact macro events (FOMC, NFP, CPI, GDP)
        and adjust sizing / block new entries accordingly.
  20.2  Overnight gap analysis — for each open position, compare today's open vs
        prior close; request immediate PM re-evaluation or auto-exit on large gaps.
  20.3  SPY pre-market context — SPY pre-market % feeds intraday sizing decisions.
  20.4  SEC 8-K monitor — poll EDGAR for material filings on held symbols.
"""
from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import requests

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")

# ── Constants ─────────────────────────────────────────────────────────────────

OVERNIGHT_GAP_REEVAL_PCT = 0.03    # 3% adverse gap → PM re-evaluation
OVERNIGHT_GAP_EXIT_PCT = 0.05      # 5% adverse gap → immediate auto-exit
SPY_FUTURES_HALVE_PCT = -0.015     # SPY pre-mkt < -1.5% → halve intraday sizes
SPY_FUTURES_BLOCK_PCT = -0.025     # SPY pre-mkt < -2.5% → no new intraday entries
EDGAR_POLL_INTERVAL = 900          # check 8-K filings every 15 min
MATERIAL_8K_ITEMS = {"1.01", "2.02", "5.02", "8.01"}  # material item codes

# FRED calendar event keywords that matter
HIGH_IMPACT_EVENTS = {
    "fomc": "FOMC",
    "federal open market committee": "FOMC",
    "nonfarm payroll": "NFP",
    "nfp": "NFP",
    "consumer price index": "CPI",
    "cpi": "CPI",
    "gross domestic product": "GDP",
    "gdp": "GDP",
}


class PremarketIntelligence:
    """
    Gathers and caches pre-market intelligence for the current trading session.
    Call `run_premarket_routine()` once per day before market open.
    """

    def __init__(self):
        self._today: Optional[str] = None       # YYYY-MM-DD of last run
        self._macro_flags: Dict[str, Any] = {}  # event_type → details
        self._spy_premarket_pct: float = 0.0
        self._last_8k_check: float = 0.0
        self._nis_macro_context = None           # MacroContext from NIS (or None)

    # ─── Public API ───────────────────────────────────────────────────────────

    def run_premarket_routine(self, open_positions: List[str], redis_send_fn=None) -> Dict[str, Any]:
        """
        Run all pre-market checks. Returns a summary dict with flags and actions taken.

        Args:
            open_positions: list of ticker symbols currently held
            redis_send_fn: optional callable(queue, msg) to send PM reeval requests
        """
        today = date.today().isoformat()
        self._today = today

        summary: Dict[str, Any] = {
            "date": today,
            "macro_flags": {},
            "spy_premarket_pct": 0.0,
            "intraday_blocked": False,
            "intraday_sizing_factor": 1.0,
            "gaps": {},
            "8k_filings": [],
        }

        # 1. Economic calendar — try NIS first, fall back to legacy FMP/hardcoded
        try:
            from app.news.intelligence_service import nis
            nis.invalidate_macro_cache()
            nis_ctx = nis.get_macro_context()
            self._nis_macro_context = nis_ctx
            # Persist snapshot so the API can serve it after a server restart
            try:
                from app.database.decision_audit import persist_nis_macro_snapshot
                persist_nis_macro_snapshot(nis_ctx)
            except Exception:
                pass
            # Back-fill legacy _macro_flags for backwards-compat
            macro: Dict[str, Any] = {}
            for evt in nis_ctx.events_today:
                macro[evt.event_type] = {
                    "name": evt.event_type,
                    "time": evt.event_time,
                    "impact": evt.risk_level.lower(),
                    "block": evt.block_new_entries,
                    "sizing_factor": evt.sizing_factor,
                    "rationale": evt.rationale,
                }
            if not macro:
                macro = self._fetch_macro_events(today)
            summary["macro_flags"] = macro
            self._macro_flags = macro
            logger.info(
                "NIS macro context: risk=%s sizing=%.2f block=%s — %s",
                nis_ctx.overall_risk, nis_ctx.global_sizing_factor,
                nis_ctx.block_new_entries, nis_ctx.rationale,
            )
        except Exception as exc:
            logger.warning("NIS macro context failed, falling back to legacy: %s", exc)
            try:
                macro = self._fetch_macro_events(today)
                summary["macro_flags"] = macro
                self._macro_flags = macro
            except Exception as exc2:
                logger.warning("Macro calendar fetch failed: %s", exc2)

        # 2. SPY pre-market context
        try:
            spy_pct = self._fetch_spy_premarket()
            self._spy_premarket_pct = spy_pct
            summary["spy_premarket_pct"] = round(spy_pct, 4)
            if spy_pct <= SPY_FUTURES_BLOCK_PCT:
                summary["intraday_blocked"] = True
                logger.warning("SPY pre-mkt %.1f%% — blocking new intraday entries", spy_pct * 100)
            elif spy_pct <= SPY_FUTURES_HALVE_PCT:
                summary["intraday_sizing_factor"] = 0.5
                logger.info("SPY pre-mkt %.1f%% — halving intraday sizes", spy_pct * 100)
        except Exception as exc:
            logger.warning("SPY pre-market fetch failed: %s", exc)

        # 3. FOMC day: block intraday + halve all sizing
        if "FOMC" in self._macro_flags:
            summary["intraday_blocked"] = True
            summary["intraday_sizing_factor"] = min(summary["intraday_sizing_factor"], 0.5)
            logger.warning("FOMC day — intraday blocked, all sizing halved")

        # 4. NFP Friday: block intraday before 10 AM
        if "NFP" in self._macro_flags:
            now_et = datetime.now(ET)
            if now_et.hour < 10:
                summary["intraday_blocked"] = True
                logger.warning("NFP day — intraday blocked until 10:00 AM ET")

        # 5. Overnight gap analysis
        if open_positions:
            try:
                gaps = self._check_overnight_gaps(open_positions, redis_send_fn)
                summary["gaps"] = gaps
            except Exception as exc:
                logger.warning("Overnight gap analysis failed: %s", exc)

        # 6. SEC 8-K monitor
        if open_positions:
            try:
                filings = self._check_8k_filings(open_positions, redis_send_fn)
                summary["8k_filings"] = filings
            except Exception as exc:
                logger.warning("8-K check failed: %s", exc)

        # 7. Regime model scoring
        try:
            regime_ctx = self._run_regime_scoring("premarket")
            if regime_ctx:
                summary["regime_score"] = regime_ctx["regime_score"]
                summary["regime_label"] = regime_ctx["regime_label"]
        except Exception as exc:
            logger.warning("Regime scoring in premarket routine failed: %s", exc)

        logger.info("Pre-market routine complete: %s", summary)
        return summary

    # ─── Macro Calendar ───────────────────────────────────────────────────────

    def _fetch_macro_events(self, today: str) -> Dict[str, Any]:
        """
        Fetch today's high-impact economic events from Nasdaq Data Link (free tier)
        or fall back to a simple FOMC date list.
        Returns dict of {event_type: details}.
        """
        found: Dict[str, Any] = {}

        # Try FMP economic calendar (free tier, 3 days)
        try:
            from app.config import settings
            fmp_key = getattr(settings, "fmp_api_key", None)
            if fmp_key:
                url = f"https://financialmodelingprep.com/api/v3/economic_calendar?from={today}&to={today}&apikey={fmp_key}"
                resp = requests.get(url, timeout=8)
                if resp.status_code == 200:
                    events = resp.json() or []
                    for ev in events:
                        name = (ev.get("event") or "").lower()
                        impact = (ev.get("impact") or "").lower()
                        if impact in ("high", "medium"):
                            for keyword, event_type in HIGH_IMPACT_EVENTS.items():
                                if keyword in name:
                                    found[event_type] = {
                                        "name": ev.get("event"),
                                        "time": ev.get("date"),
                                        "impact": impact,
                                    }
                                    break
                    return found
        except Exception as exc:
            logger.debug("FMP economic calendar failed: %s", exc)

        # Fallback: FOMC meeting dates (hardcoded 2026 schedule)
        fomc_dates_2026 = {
            "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
            "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-16",
        }
        if today in fomc_dates_2026:
            found["FOMC"] = {"name": "FOMC Meeting", "time": "14:00 ET", "impact": "high"}

        # NFP = first Friday of each month
        d = date.fromisoformat(today)
        if d.weekday() == 4:  # Friday
            first_day = d.replace(day=1)
            days_to_first_fri = (4 - first_day.weekday()) % 7
            first_friday = first_day + timedelta(days=days_to_first_fri)
            if d == first_friday:
                found["NFP"] = {"name": "Non-Farm Payrolls", "time": "08:30 ET", "impact": "high"}

        return found

    # ─── SPY Pre-market ───────────────────────────────────────────────────────

    def _fetch_spy_premarket(self) -> float:
        """
        Return SPY pre-market % change vs prior close using Alpaca bars.
        Positive = gap up, negative = gap down.
        """
        from app.integrations import get_alpaca_client
        client = get_alpaca_client()
        bars = client.get_bars("SPY", timeframe="1Day", limit=3)
        if bars is None or len(bars) < 2:
            return 0.0
        prior_close = float(bars["close"].iloc[-2])
        latest_close = float(bars["close"].iloc[-1])
        if prior_close <= 0:
            return 0.0
        return (latest_close - prior_close) / prior_close

    # ─── Overnight Gap Analysis ───────────────────────────────────────────────

    def _check_overnight_gaps(
        self, symbols: List[str], redis_send_fn=None
    ) -> Dict[str, Dict]:
        """
        For each held symbol, compute overnight gap = (today_open - prior_close) / prior_close.
        Adverse gaps trigger PM re-evaluation (>3%) or exit signal (>5%).
        """
        from app.integrations import get_alpaca_client
        client = get_alpaca_client()
        gaps: Dict[str, Dict] = {}

        for symbol in symbols:
            try:
                bars = client.get_bars(symbol, timeframe="1Day", limit=3)
                if bars is None or len(bars) < 2:
                    continue
                prior_close = float(bars["close"].iloc[-2])
                # Phase 73: prefer actual today's open from 1-min bars; fall back gracefully
                today_open = None
                try:
                    intraday = client.get_bars(symbol, timeframe="1Min", limit=5)
                    if intraday is not None and not intraday.empty and "open" in intraday.columns:
                        today_open = float(intraday["open"].iloc[0])
                except Exception:
                    pass
                if today_open is None:
                    today_open = float(
                        bars["open"].iloc[-1] if "open" in bars.columns
                        else bars["close"].iloc[-1]
                    )
                if prior_close <= 0:
                    continue
                gap_pct = (today_open - prior_close) / prior_close
                gaps[symbol] = {"gap_pct": round(gap_pct, 4)}

                if gap_pct <= -OVERNIGHT_GAP_EXIT_PCT:
                    gaps[symbol]["action"] = "AUTO_EXIT"
                    logger.warning(
                        "%s: adverse overnight gap %.1f%% → queuing auto-exit",
                        symbol, gap_pct * 100,
                    )
                    if redis_send_fn:
                        redis_send_fn("trader_exit_requests", {
                            "symbol": symbol,
                            "action": "EXIT",
                            "reason": f"overnight_gap_{gap_pct*100:.1f}pct",
                        })
                elif gap_pct <= -OVERNIGHT_GAP_REEVAL_PCT:
                    gaps[symbol]["action"] = "REEVAL"
                    logger.info(
                        "%s: adverse overnight gap %.1f%% → queuing PM reeval",
                        symbol, gap_pct * 100,
                    )
                    if redis_send_fn:
                        redis_send_fn("pm_reeval_requests", {
                            "symbol": symbol,
                            "reason": f"overnight_gap_{gap_pct*100:.1f}pct",
                            "current_price": today_open,
                        })
                else:
                    gaps[symbol]["action"] = "OK"
            except Exception as exc:
                logger.debug("Gap check failed for %s: %s", symbol, exc)

        return gaps

    # ─── SEC 8-K Monitor ─────────────────────────────────────────────────────

    def _check_8k_filings(
        self, symbols: List[str], redis_send_fn=None
    ) -> List[Dict]:
        """
        Poll EDGAR EFTS for 8-K filings on held symbols in the last 24 hours.
        Material filings trigger immediate PM re-evaluation.
        """
        now = time.monotonic()
        if now - self._last_8k_check < EDGAR_POLL_INTERVAL:
            return []
        self._last_8k_check = now

        material_filings: List[Dict] = []
        today = date.today().isoformat()

        for symbol in symbols:
            try:
                url = (
                    f"https://efts.sec.gov/LATEST/search-index?q=%22{symbol}%22"
                    f"&dateRange=custom&startdt={today}&enddt={today}&forms=8-K"
                )
                resp = requests.get(
                    url,
                    headers={"User-Agent": "MrTrader research@example.com"},
                    timeout=8,
                )
                if resp.status_code != 200:
                    continue
                hits = resp.json().get("hits", {}).get("hits", [])
                for hit in hits:
                    src = hit.get("_source", {})
                    # Check for material item types
                    filing_items = {
                        s.strip() for s in str(src.get("file_type", "")).split(",")
                    }
                    description = (src.get("display_names") or "").lower()
                    is_material = any(
                        item in description or item in filing_items
                        for item in MATERIAL_8K_ITEMS
                    )
                    if is_material or hits:  # any 8-K on a held symbol is worth noting
                        entry = {
                            "symbol": symbol,
                            "filing_date": today,
                            "description": src.get("display_names", ""),
                        }
                        material_filings.append(entry)
                        logger.warning(
                            "%s: 8-K filing detected — queuing PM reeval", symbol
                        )
                        if redis_send_fn:
                            redis_send_fn("pm_reeval_requests", {
                                "symbol": symbol,
                                "reason": "sec_8k_filing",
                                "current_price": 0.0,
                            })
                        break  # one reeval per symbol per check
            except Exception as exc:
                logger.debug("8-K check failed for %s: %s", symbol, exc)

        return material_filings

    # ─── Session State Accessors ──────────────────────────────────────────────

    @property
    def macro_flags(self) -> Dict[str, Any]:
        return dict(self._macro_flags)

    @property
    def macro_context(self):
        """Return the NIS MacroContext for today, or None if NIS unavailable."""
        return self._nis_macro_context

    @property
    def spy_premarket_pct(self) -> float:
        return self._spy_premarket_pct

    def intraday_sizing_factor(self) -> float:
        """Return the intraday size multiplier for this session (0.5 or 1.0)."""
        spy = self._spy_premarket_pct
        if spy <= SPY_FUTURES_HALVE_PCT:
            return 0.5
        if self._nis_macro_context is not None:
            return min(1.0, self._nis_macro_context.global_sizing_factor)
        if "FOMC" in self._macro_flags:
            return 0.5
        return 1.0

    def _all_todays_events_released(self) -> bool:
        """
        Return True if all high-impact macro events scheduled for today have
        already passed their release time (+ 5 min buffer).

        This lets the NIS pre-release block expire naturally once every event
        on the calendar has printed — without needing a manual override.
        FOMC at 2:00 PM keeps the block active until 2:05 PM ET.
        """
        try:
            from app.calendars.macro import MacroCalendar
            cal = MacroCalendar()
            now_et = datetime.now(ET)
            today_str = now_et.strftime("%Y-%m-%d")
            ctx = cal.get_context()
            if not ctx.events_today:
                return True  # no events today — block is stale, lift it
            for evt in ctx.events_today:
                if evt.date_str != today_str:
                    continue
                h, m = map(int, evt.time_str.split(":"))
                release_dt = now_et.replace(hour=h, minute=m, second=0, microsecond=0)
                # Still within 5-min buffer after release
                if now_et < release_dt.replace(minute=m + 5) if m <= 54 else release_dt:
                    return False  # at least one event not yet released
            return True  # all events have printed
        except Exception:
            return False  # conservative: keep blocking if we can't determine

    def is_intraday_blocked(self) -> bool:
        """Return True if no new intraday entries should be placed this session."""
        spy = self._spy_premarket_pct
        if spy <= SPY_FUTURES_BLOCK_PCT:
            return True
        # NIS-driven block (replaces binary FOMC check)
        if self._nis_macro_context is not None:
            if self._nis_macro_context.block_new_entries:
                # Lift block once all scheduled events for today have been released
                if self._all_todays_events_released():
                    logger.info("NIS intraday block lifted — all today's macro events released")
                    return False
                return True
        else:
            if "FOMC" in self._macro_flags:
                return True
            if "NFP" in self._macro_flags:
                now_et = datetime.now(ET)
                if now_et.hour < 10:
                    return True
        return False

    def is_swing_blocked(self) -> bool:
        """
        Return True if swing entries should be blocked today.
        NIS Tier 1 provides risk-scored blocking — HIGH risk with genuine uncertainty
        blocks swing. SPY pre-market gap < -2.5% or live intraday drawdown > 2% also block.
        """
        if self._spy_premarket_pct <= SPY_FUTURES_BLOCK_PCT:
            return True
        # NIS-driven block (consensus-aware, not binary)
        if self._nis_macro_context is not None:
            if self._nis_macro_context.block_new_entries:
                # Lift block once all scheduled events for today have been released
                if self._all_todays_events_released():
                    logger.info("NIS swing block lifted — all today's macro events released")
                    return False
                logger.info(
                    "Swing entries blocked by NIS: risk=%s — %s",
                    self._nis_macro_context.overall_risk,
                    self._nis_macro_context.rationale,
                )
                return True
        else:
            if "FOMC" in self._macro_flags:
                return True
        # Check live intraday SPY drawdown from today's open
        intraday_drawdown = self._get_spy_intraday_drawdown()
        if intraday_drawdown <= -0.02:
            logger.warning(
                "Swing entries blocked: SPY intraday drawdown %.1f%% from open",
                intraday_drawdown * 100,
            )
            return True
        return False

    def _get_spy_intraday_drawdown(self) -> float:
        """
        Return SPY's % change from today's open to now.
        Negative = down from open. Cached for 5 minutes to avoid hammering the API.
        """
        now = datetime.now(ET)
        cache_key = "_spy_intraday_cache"
        cached = getattr(self, cache_key, None)
        if cached and (now - cached["ts"]).total_seconds() < 300:
            return cached["value"]
        try:
            from app.integrations import get_alpaca_client
            client = get_alpaca_client()
            bars = client.get_bars("SPY", timeframe="5Min", limit=80)
            if bars is None or len(bars) < 2:
                return 0.0
            today_str = now.strftime("%Y-%m-%d")
            today_bars = bars[bars.index.strftime("%Y-%m-%d") == today_str]
            if len(today_bars) < 2:
                return 0.0
            open_price = float(today_bars["open"].iloc[0])
            current_price = float(today_bars["close"].iloc[-1])
            drawdown = (current_price - open_price) / open_price if open_price > 0 else 0.0
            setattr(self, cache_key, {"ts": now, "value": drawdown})
            return drawdown
        except Exception:
            return 0.0

    # ─── Regime Model Integration (Phase R3) ─────────────────────────────────

    def _run_regime_scoring(self, trigger: str) -> Optional[dict]:
        """Score the regime model and persist to regime_snapshots. Returns score dict or None."""
        try:
            from app.ml.regime_model import RegimeModel
            result = RegimeModel.instance().score(
                as_of_date=date.today(),
                trigger=trigger,
            )
            logger.info(
                "Regime score [%s]: %.4f (%s)",
                trigger, result["regime_score"], result["regime_label"],
            )
            return result
        except Exception as exc:
            logger.error("Regime scoring failed (trigger=%s): %s", trigger, exc)
            return None

    def _schedule_regime_reeval_jobs(self, scheduler) -> None:
        """Register APScheduler one-shot jobs for post-macro-event regime re-evals."""
        try:
            from app.calendars.macro import MacroCalendar
            cal = MacroCalendar()
            ctx = cal.get_context()
            today_str = date.today().isoformat()
            for evt in ctx.events_today:
                if evt.date_str != today_str:
                    continue
                try:
                    h, m = map(int, evt.time_str.split(":"))
                    # Schedule re-eval 5 minutes after event release
                    reeval_m = m + 5
                    reeval_h = h
                    if reeval_m >= 60:
                        reeval_m -= 60
                        reeval_h += 1
                    trigger_name = f"post_{evt.event_type.lower()}"
                    run_dt = datetime.now(ET).replace(
                        hour=reeval_h, minute=reeval_m, second=0, microsecond=0
                    )
                    if run_dt > datetime.now(ET):
                        scheduler.add_job(
                            lambda tn=trigger_name: self._run_regime_scoring(tn),
                            trigger="date",
                            run_date=run_dt,
                            id=f"regime_reeval_{trigger_name}",
                            replace_existing=True,
                        )
                        logger.info(
                            "Regime re-eval job scheduled: %s at %02d:%02d ET",
                            trigger_name, reeval_h, reeval_m,
                        )
                except Exception as exc:
                    logger.warning("Could not schedule regime re-eval for %s: %s", evt.event_type, exc)
        except Exception as exc:
            logger.warning("_schedule_regime_reeval_jobs failed: %s", exc)

    def _startup_regime_catchup(self) -> None:
        """
        Called on startup. If premarket regime scoring was missed (server was down at 7am
        and it's now before 11:30 ET), run a catchup score with trigger='startup_catchup'.
        Also re-schedules any post-event re-evals that are still in the future.
        """
        try:
            now_et = datetime.now(ET)
            if now_et.hour > 11 or (now_et.hour == 11 and now_et.minute >= 30):
                logger.info("Regime startup catchup skipped — past 11:30 ET")
                return

            # Check if we already have a premarket row for today
            from app.database.session import get_session
            from app.database.models import RegimeSnapshot
            today = date.today()
            with get_session() as session:
                existing = session.query(RegimeSnapshot).filter(
                    RegimeSnapshot.snapshot_date == today,
                    RegimeSnapshot.snapshot_trigger.in_(["premarket", "startup_catchup"]),
                ).first()

            if existing:
                logger.info("Regime startup catchup skipped — premarket row already exists for %s", today)
            else:
                logger.info("Regime startup catchup: running missed premarket scoring")
                self._run_regime_scoring("startup_catchup")
        except Exception as exc:
            logger.error("_startup_regime_catchup failed: %s", exc)

    def get_regime_context(self) -> Optional[dict]:
        """
        Return the latest regime score for today from regime_snapshots.
        Returns None if no row exists yet (before first premarket run).
        """
        try:
            from app.database.session import get_session
            from app.database.models import RegimeSnapshot
            today = date.today()
            with get_session() as session:
                row = (
                    session.query(RegimeSnapshot)
                    .filter(
                        RegimeSnapshot.snapshot_date == today,
                        RegimeSnapshot.snapshot_trigger != "backfill",
                    )
                    .order_by(RegimeSnapshot.snapshot_time.desc())
                    .first()
                )
                if row is None:
                    return None
                age_hours = (
                    (datetime.now(ET).replace(tzinfo=None) - row.snapshot_time).total_seconds() / 3600
                    if row.snapshot_time else 99
                )
                result = {
                    "regime_score": row.regime_score,
                    "regime_label": row.regime_label,
                    "trigger": row.snapshot_trigger,
                    "snapshot_time": row.snapshot_time.isoformat() if row.snapshot_time else None,
                    "age_hours": round(age_hours, 2),
                }
                # Warn if score is stale during market hours
                now_et = datetime.now(ET)
                if 9 <= now_et.hour < 16 and age_hours > 4:
                    logger.warning(
                        "Regime score is stale (%.1fh old) — applying 20%% uncertainty haircut",
                        age_hours,
                    )
                    result["regime_score"] = round(row.regime_score * 0.8, 4)
                    result["stale"] = True
                return result
        except Exception as exc:
            logger.error("get_regime_context failed: %s", exc)
            return None

    def get_market_context(self) -> dict:
        """
        Return a summary of current macro context for use by Trader's entry check.
        Includes intraday SPY drawdown, pre-market pct, macro flags, block status.
        """
        ctx = {
            "spy_premarket_pct": round(self._spy_premarket_pct * 100, 2),
            "spy_intraday_drawdown_pct": round(self._get_spy_intraday_drawdown() * 100, 2),
            "macro_flags": list(self._macro_flags.keys()),
            "intraday_blocked": self.is_intraday_blocked(),
            "swing_blocked": self.is_swing_blocked(),
            "sizing_factor": self.intraday_sizing_factor(),
        }
        if self._nis_macro_context is not None:
            ctx["nis_risk_level"] = self._nis_macro_context.overall_risk
            ctx["nis_sizing_factor"] = self._nis_macro_context.global_sizing_factor
            ctx["nis_rationale"] = self._nis_macro_context.rationale
        return ctx


# Module-level singleton
premarket_intel = PremarketIntelligence()
