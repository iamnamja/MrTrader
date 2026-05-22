"""
Portfolio Manager Agent — daily ML-driven instrument selection.

Cycle:
  1. At 09:30 ET: run swing model (daily bars) → proposals tagged trade_type="swing"
  2. At 09:45 ET: run intraday model (5-min bars) → proposals tagged trade_type="intraday"
  3. Send all proposals to Risk Manager via Redis queue `trade_proposals`
  4. At 17:00 ET: retrain swing model with latest daily data
"""

import asyncio
import logging
import time as _time
import uuid
import numpy as np
from datetime import datetime
from typing import Any, Coroutine, Dict, List, Optional
from zoneinfo import ZoneInfo

from app.agents.base import BaseAgent
from app.agents.news_monitor import news_monitor
from app.ml.features import FeatureEngineer
from app.ml.cs_normalize import cs_normalize, cs_normalize_branch_a
from app.ml.intraday_features import BRANCH_B_FEATURES as _INTRADAY_BRANCH_B, FEATURE_NAMES as _INTRADAY_FEATURE_NAMES
from app.ml.model import PortfolioSelectorModel
from app.ml.training import ModelTrainer
from app.utils.constants import RUSSELL_1000_TICKERS
from app.ml.retrain_config import MAX_WORKERS, MAX_THREADS, INTRADAY_ENABLED

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")

TRADE_PROPOSALS_QUEUE = "trade_proposals"
EXIT_REQUESTS_QUEUE = "trader_exit_requests"     # PM → Trader
REEVAL_REQUESTS_QUEUE = "pm_reeval_requests"     # Trader → PM
TOP_N_STOCKS = 10
TOP_N_INTRADAY = 5             # fewer intraday picks per session
MIN_CONFIDENCE = 0.55          # minimum model probability to propose a trade
EXIT_THRESHOLD = 0.35          # re-score below this → exit signal
POSITION_RISK_PCT = 0.02       # base risk per trade (2% of strategy budget)
# Intraday scan windows (ET hour, minute). Shifted 11:00→10:45 to avoid lunch;
# 13:30→13:00 to give the smaller afternoon universe more reaction time.
# Lunch block (11:15–13:00) is enforced separately in the heartbeat loop.
INTRADAY_SCAN_WINDOWS = [(9, 45), (10, 45), (13, 0)]
INTRADAY_COOLDOWN_HOURS = 2     # min gap between entries in the same symbol
INTRADAY_DAILY_LOSS_CAP_PCT = 0.01   # stop new intraday entries if day loss > 1% of account
SWING_PREMARKET_HOUR = 8       # 08:00 ET: run swing model analysis pre-market
SWING_SEND_HOUR = 9            # 09:50 ET: send swing proposals after open volatility settles
SWING_SEND_MINUTE = 50
REVIEW_INTERVAL_MINUTES = 30   # PM re-scores open positions every 30 minutes
EARNINGS_EXIT_DAYS = 3         # exit swing positions this many days before earnings
INTRADAY_PROPOSAL_STALE_MIN = 30       # proposals expire after 30 min (was 90)
INTRADAY_AFTERNOON_CANDIDATES = 100    # top-N by score from 9:45 scan reused for 10:45/13:00
# SPY intraday % thresholds for dynamic confidence/size adjustment
SPY_HARD_STOP_PCT = -1.5   # below this: no new intraday longs all day
SPY_CAUTION_PCT = -0.75    # below this: raise min_confidence→0.65, size→50%
SPY_CHASE_PCT = 2.0        # above this: raise min_confidence→0.65 (gap caution)
SPY_ADAPTIVE_DELTA = 1.5   # adaptive re-scan if SPY moves this much from last scan

# ── Phase 85: PM abstention gates ────────────────────────────────────────────
# Gate 1A: skip scan if SPY first-hour (bars 0-11) high-low range < 0.45%
SPY_MIN_FIRST_HOUR_RANGE = 0.0020
# Gate 1B: reduce to 1 pick if top-decile score spread above median < threshold
SCORE_SPREAD_MIN = 0.08
# Gate 1C: melt-up compression — SPY up >2.5% over 5d, vol <0.60%/d, first hour also compressed
MELTUP_5D_RETURN_MIN = 0.025
MELTUP_5D_VOL_MAX = 0.006
MELTUP_FIRST_HOUR_MAX = 0.005

# ── Capital allocation ────────────────────────────────────────────────────────
SWING_BUDGET_PCT = 0.70        # 70% of account reserved for swing trades
INTRADAY_BUDGET_PCT = 0.30     # 30% of account reserved for intraday trades
GROSS_EXPOSURE_CAP = 0.80      # never deploy more than 80% of account at once


def _confidence_scalar(prob: float) -> float:
    """Linear scale: prob=MIN_CONFIDENCE → 0.5x, prob=1.0 → 2.0x."""
    lo, hi = MIN_CONFIDENCE, 1.0
    return float(np.clip(0.5 + 1.5 * (prob - lo) / max(hi - lo, 1e-6), 0.5, 2.0))


class PortfolioManager(BaseAgent):
    """
    Runs on a 60-second heartbeat.
    09:30: swing model selection (daily bars)
    09:45: intraday model selection (5-min bars)
    17:00: retrain swing model
    """

    def __init__(self):
        super().__init__("portfolio_manager")
        self.feature_engineer = FeatureEngineer()
        self.model = PortfolioSelectorModel(model_type="xgboost")           # swing
        self.intraday_model = PortfolioSelectorModel(model_type="xgboost")  # intraday
        from app.ml.retrain_config import SWING_RETRAIN
        self.trainer = ModelTrainer(
            model_type=SWING_RETRAIN["model_type"],
            label_scheme=SWING_RETRAIN.get("label_scheme", "triple_barrier"),
            hpo_trials=SWING_RETRAIN["hpo_trials"],
            n_workers=SWING_RETRAIN["n_workers"],
            feature_keep_list=SWING_RETRAIN.get("feature_keep_list", None),
        )
        # Last-known-good scores per symbol (persisted across review cycles).
        # Used to detect suspicious near-zero scores from scorer failures.
        self._last_good_score: dict = {}  # symbol -> (score, timestamp)
        self._analyzed_today: bool = False       # 08:00 pre-market analysis done
        self._selected_today: bool = False       # 09:50 proposals sent
        self._selected_intraday_today: bool = False  # legacy compat (missed-task check)
        self._intraday_windows_run: set = set()  # Phase 51: (hour,min) windows already scanned
        self._intraday_symbol_last_entry: Dict[str, float] = {}  # symbol → monotonic ts of last entry
        self._retrained_today: bool = False
        self._earnings_prefetched_today: bool = False  # 06:00 earnings calendar prefetch
        self._premarket_run_today: bool = False  # 09:00 premarket routine done
        self._benchmark_recorded_today: bool = False
        self._weekly_report_generated_today: bool = False
        self._last_date: Optional[str] = None
        self._swing_proposals: List[Dict[str, Any]] = []  # cached from 08:00 analysis
        self._pending_approvals: Dict[str, float] = {}   # symbol → monotonic time of proposal send (Phase 70)
        self._last_review_minute: int = -1       # last minute a 30-min review ran
        self._last_heartbeat_hour: int = -1      # last hour a loop-alive log was emitted
        # Per-task stats: attempts, failures, last duration (ms), last run timestamp
        self._task_stats: Dict[str, Dict[str, Any]] = {}
        # Gap 1: cache features at inference so we can log top inputs per decision
        self._last_swing_features: Dict[str, Dict[str, float]] = {}
        self._last_intraday_features: Dict[str, Dict[str, float]] = {}
        # Gap 2: EOD backfill + daily summary tracking
        self._eod_jobs_run_today: bool = False
        self._eod_order_cleanup_run_today: bool = False
        # Adaptive intraday universe — top-N candidates from 9:45 scan reused for later windows
        self._morning_intraday_candidates: List[str] = []
        # NIS post-event refresh tracking: set of event IDs already refreshed today
        self._nis_refreshed_event_ids: set = set()
        # SPY % at time of last intraday scan (for adaptive re-scan trigger)
        self._last_scan_spy_pct: float = 0.0
        # Monotonic timestamp of last adaptive re-scan (prevents back-to-back triggers)
        self._last_adaptive_scan_at: float = 0.0
        # Regime context from last premarket / re-eval (Phase R3 — parallel running only)
        self._current_regime_ctx: Optional[dict] = None

    # ─── Scheduling helpers ───────────────────────────────────────────────────

    @staticmethod
    def _in_window(now: datetime, start_h: int, start_m: int, end_h: int, end_m: int) -> bool:
        """Return True if now falls in [start_h:start_m, end_h:end_m)."""
        t = now.hour * 60 + now.minute
        return (start_h * 60 + start_m) <= t < (end_h * 60 + end_m)

    def _get_spy_day_pct(self) -> Optional[float]:
        """Return SPY % change from previous close to current price. None on error."""
        try:
            bars = self._alpaca.get_bars("SPY", timeframe="1Day", limit=2)
            if bars is None or len(bars) < 2:
                return None
            prev_close = float(bars.iloc[-2]["close"])
            current = self._alpaca.get_latest_price("SPY")
            if not current or prev_close <= 0:
                return None
            return (float(current) - prev_close) / prev_close * 100
        except Exception:
            return None

    def _get_spy_intraday_state(self) -> dict:
        """Return SPY first-hour range, 5-day vol/return, and daily bars for Phase 85/86.

        Returns dict with keys: first_hour_range, spy_5d_return, spy_5d_vol, spy_daily_bars.
        All values are None on data error.
        """
        result = {"first_hour_range": None, "spy_5d_return": None, "spy_5d_vol": None,
                  "spy_daily_bars": None}
        try:
            intraday = self._alpaca.get_bars("SPY", timeframe="5Min", limit=78)
            if intraday is not None and len(intraday) >= 12:
                bars_0_11 = intraday.iloc[:12]
                prior_close = float(bars_0_11["open"].iloc[0])
                if prior_close > 0:
                    result["first_hour_range"] = (
                        float(bars_0_11["high"].max()) - float(bars_0_11["low"].min())
                    ) / prior_close
        except Exception:
            pass
        try:
            # Fetch 30 days for Phase 86 market-condition features (need ≥6 for 5d vol)
            daily = self._alpaca.get_bars("SPY", timeframe="1Day", limit=30)
            if daily is not None and len(daily) >= 6:
                closes = daily["close"].astype(float).values
                daily_rets = np.diff(closes) / closes[:-1]
                result["spy_5d_return"] = float(daily_rets[-5:].sum())
                result["spy_5d_vol"] = float(daily_rets[-5:].std())
                result["spy_daily_bars"] = daily  # Phase 86: pass to compute_intraday_features
        except Exception:
            pass
        return result

    async def _withdraw_all_pending(self, reason: str) -> None:
        """Withdraw all pending approvals and clear the map."""
        for symbol in list(self._pending_approvals.keys()):
            self.send_message(EXIT_REQUESTS_QUEUE, {"symbol": symbol, "action": "WITHDRAW", "reason": reason})
            await self.log_decision("PROPOSAL_WITHDRAWN", reasoning={"symbol": symbol, "reason": reason})
        self._pending_approvals.clear()
        self.logger.info("Withdrew all pending approvals — reason: %s", reason)

    async def _run_task(
        self,
        name: str,
        nominal_h: int,
        nominal_m: int,
        coro: Coroutine,
    ) -> None:
        """
        Run a scheduled task and record timing + outcome to agent_decisions.

        Logs a TASK_COMPLETED or TASK_FAILED decision so the dashboard always
        shows whether each named task ran, how long it took, and whether it was
        late starting.
        """
        now = datetime.now(ET)
        late_by = (now.hour * 60 + now.minute) - (nominal_h * 60 + nominal_m)
        stats = self._task_stats.setdefault(
            name, {"attempts": 0, "failures": 0, "last_run_ms": None, "last_run_at": None}
        )
        stats["attempts"] += 1

        if late_by > 1:
            self.logger.warning(
                "Task %s starting %d min late (nominal %02d:%02d, actual %02d:%02d)",
                name, late_by, nominal_h, nominal_m, now.hour, now.minute,
            )
        else:
            self.logger.info("Task %s starting (nominal %02d:%02d)", name, nominal_h, nominal_m)

        t0 = _time.monotonic()
        try:
            await coro
            elapsed_ms = int((_time.monotonic() - t0) * 1000)
            stats["last_run_ms"] = elapsed_ms
            stats["last_run_at"] = datetime.now(ET).isoformat()
            self.logger.info("Task %s completed in %.1fs", name, elapsed_ms / 1000)
            await self.log_decision("TASK_COMPLETED", reasoning={
                "task": name,
                "nominal": f"{nominal_h:02d}:{nominal_m:02d}",
                "late_by_min": max(0, late_by),
                "duration_ms": elapsed_ms,
            })
        except Exception as exc:
            elapsed_ms = int((_time.monotonic() - t0) * 1000)
            stats["failures"] += 1
            self.logger.error(
                "Task %s failed after %.1fs: %s", name, elapsed_ms / 1000, exc, exc_info=True
            )
            await self.log_decision("TASK_FAILED", reasoning={
                "task": name,
                "nominal": f"{nominal_h:02d}:{nominal_m:02d}",
                "duration_ms": elapsed_ms,
                "error": str(exc),
            })
            raise

    # ─── Lazy connectors ─────────────────────────────────────────────────────

    @property
    def _alpaca(self):
        from app.integrations import get_alpaca_client
        return get_alpaca_client()

    # ─── Main Loop ────────────────────────────────────────────────────────────

    def _restore_daily_flags(self) -> None:
        """On startup, restore today's task flags from the AgentDecision log to prevent re-running."""
        from app.database.session import get_session
        from app.database.models import AgentDecision as _AD
        today = datetime.now(ET).date()
        db = get_session()
        try:
            # DB stores naive UTC; compare against UTC midnight for today (ET date)
            from datetime import timezone
            today_utc_start = datetime.combine(today, datetime.min.time()).replace(tzinfo=timezone.utc)
            today_decisions = db.query(_AD.decision_type).filter(
                _AD.agent_name == "portfolio_manager",
                _AD.timestamp >= today_utc_start.replace(tzinfo=None),
            ).all()
            types = {r.decision_type for r in today_decisions}
            if "SWING_PREMARKET_ANALYSIS" in types:
                self._analyzed_today = True
                self.logger.info("Startup: swing_premarket_analysis already ran today — skipping re-run")
            if "INSTRUMENTS_SELECTED" in types or "SWING_PROPOSALS_SENT" in types:
                self._selected_today = True
                self.logger.info("Startup: swing proposals already sent today — skipping re-send")
            if "PREMARKET_INTELLIGENCE" in types:
                self._premarket_run_today = True
            # Restore morning intraday candidates so post-restart scans use the short list
            try:
                import json
                from app.database.config_store import get_config as _gc
                raw = _gc(db, "intraday.morning_candidates")
                if raw:
                    payload = json.loads(raw)
                    if payload.get("date") == today.isoformat() and payload.get("symbols"):
                        self._morning_intraday_candidates = payload["symbols"]
                        self.logger.info(
                            "Startup: restored %d morning intraday candidates from DB",
                            len(self._morning_intraday_candidates),
                        )
            except Exception as _ce:
                self.logger.debug("Could not restore morning candidates: %s", _ce)
        except Exception as e:
            self.logger.warning("Could not restore daily flags from DB: %s", e)
        finally:
            db.close()

    async def run(self):
        self.logger.info("Portfolio Manager started")
        self.status = "running"
        self._try_load_model()
        self._restore_daily_flags()

        # Phase R3: run startup catchup if premarket regime scoring was missed
        try:
            from app.agents.premarket import premarket_intel
            await asyncio.to_thread(premarket_intel._startup_regime_catchup)
            self._current_regime_ctx = premarket_intel.get_regime_context()
        except Exception as _rc_exc:
            self.logger.debug("Regime startup catchup failed (non-fatal): %s", _rc_exc)

        while self.status == "running":
            try:
                now = datetime.now(ET)
                today = now.strftime("%Y-%m-%d")

                # ── Daily reset at midnight ───────────────────────────────────
                if today != self._last_date:
                    if self._last_date is not None:
                        await self._warn_missed_tasks()
                    self._analyzed_today = False
                    self._selected_today = False
                    self._selected_intraday_today = False
                    self._intraday_windows_run = set()
                    self._intraday_symbol_last_entry = {}
                    self._retrained_today = False
                    self._earnings_prefetched_today = False
                    self._premarket_run_today = False
                    self._benchmark_recorded_today = False
                    self._weekly_report_generated_today = False
                    self._eod_jobs_run_today = False
                    self._eod_order_cleanup_run_today = False
                    self._last_swing_features = {}
                    self._morning_intraday_candidates = []
                    self._last_scan_spy_pct = 0.0
                    self._last_adaptive_scan_at = 0.0
                    self._last_intraday_features = {}
                    self._nis_refreshed_event_ids = set()
                    self._last_date = today
                    self._swing_proposals = []
                    self._last_review_minute = -1
                    # Phase 53: clear intraday watchlist — all intraday positions flat at EOD
                    from app.agents.news_monitor import _watched as _nm_watched
                    _nm_watched.clear()
                    self.logger.info("Daily reset complete for %s", today)

                is_weekday = now.weekday() < 5

                # ── Hourly loop-alive heartbeat (INFO only, not to DB) ────────
                if now.hour != self._last_heartbeat_hour:
                    self._last_heartbeat_hour = now.hour
                    self.logger.info(
                        "PM heartbeat %02d:00 — flags: analyzed=%s selected=%s "
                        "intraday=%s retrained=%s model_trained=%s",
                        now.hour,
                        self._analyzed_today, self._selected_today,
                        self._selected_intraday_today, self._retrained_today,
                        self.model.is_trained,
                    )

                # ── Phase 83: DB heartbeat for deadman watchdog ───────────────
                # Writes every 60s during market hours so watchdog.py can detect crash.
                if is_weekday and (9 <= now.hour < 17):
                    try:
                        await asyncio.to_thread(self._write_heartbeat)
                    except Exception as _hb_exc:
                        self.logger.debug("Heartbeat write failed: %s", _hb_exc)

                # ── 06:00–07:59: prefetch earnings calendar ───────────────────
                # Warms cache for all watchlist symbols before trading starts.
                # Fail-closed on swing entries if both sources fail (Phase 81).
                if (
                    is_weekday
                    and self._in_window(now, 6, 0, 8, 0)
                    and not self._earnings_prefetched_today
                ):
                    self._earnings_prefetched_today = True
                    await self._run_task(
                        "earnings_prefetch", 6, 0,
                        self._prefetch_earnings_calendar(),
                    )

                # ── 08:00–09:39: pre-market swing analysis ───────────────────
                # Window closes at 09:40 to guarantee analysis finishes before
                # the 09:50 send step (7-min timeout + margin).
                if (
                    is_weekday
                    and self._in_window(now, 8, 0, 9, 40)
                    and not self._analyzed_today
                ):
                    self._analyzed_today = True
                    await self._run_task(
                        "swing_premarket_analysis", 8, 0,
                        self._analyze_swing_premarket(),
                    )

                # ── 09:00–09:29: pre-market intelligence ─────────────────────
                if (
                    is_weekday
                    and self._in_window(now, 9, 0, 9, 30)
                    and not self._premarket_run_today
                ):
                    self._premarket_run_today = True
                    await self._run_task(
                        "premarket_intelligence", 9, 0,
                        self._run_premarket_intelligence(),
                    )

                # ── 09:50–10:59: send cached swing proposals ──────────────────
                if (
                    is_weekday
                    and self._in_window(now, 9, 50, 11, 0)
                    and not self._selected_today
                ):
                    self._selected_today = True
                    await self._run_task(
                        "swing_proposals_send", 9, 50,
                        self._send_swing_proposals(),
                    )

                # ── Intraday scan windows ─────────────────────────────────────
                # Windows: 09:45, 10:45, 13:00 ET — each runs once per day.
                # 10:45 and 13:00 use the morning candidates list (top-N from 9:45
                # scan) instead of the full 716-symbol universe.
                # Lunch block (11:15–13:00) is respected — no new scans during this window.
                for win_h, win_m in INTRADAY_SCAN_WINDOWS:
                    # Window is "active" for 15 minutes after its scheduled time
                    if (
                        INTRADAY_ENABLED
                        and is_weekday
                        and self._in_window(now, win_h, win_m, win_h, win_m + 15)
                        and (win_h, win_m) not in self._intraday_windows_run
                        and not self._in_window(now, 11, 15, 13, 0)  # lunch block
                    ):
                        self._intraday_windows_run.add((win_h, win_m))
                        self._selected_intraday_today = True  # legacy compat
                        from app.agents.premarket import premarket_intel
                        if premarket_intel.is_intraday_blocked():
                            self.logger.info(
                                "Intraday scan %02d:%02d BLOCKED (macro gate or SPY gap)",
                                win_h, win_m,
                            )
                            await self.log_decision("SELECTION_SKIPPED", reasoning={
                                "reason": "premarket_blocked",
                                "strategy": "intraday",
                                "window": f"{win_h:02d}:{win_m:02d}",
                            })
                        else:
                            # First window (9:45): full universe to build morning candidates list.
                            # Later windows: reuse morning candidates for speed.
                            use_morning = (win_h, win_m) != (9, 45)
                            await self._run_task(
                                f"intraday_selection_{win_h:02d}{win_m:02d}", win_h, win_m,
                                self.select_intraday_instruments(
                                    window=(win_h, win_m),
                                    use_morning_candidates=use_morning,
                                ),
                            )

                # ── Adaptive re-scan: SPY moves >1.5% from last scan reference ──
                # Fires at most once per hour; skips lunch (11:15-13:00);
                # withdraws stale proposals and re-scans morning candidates.
                market_open_intraday = (now.hour > 9 or (now.hour == 9 and now.minute >= 45))
                not_lunch = not self._in_window(now, 11, 15, 13, 0)
                not_eod = now.hour < 15
                adaptive_gap_ok = (_time.monotonic() - self._last_adaptive_scan_at) > 3600
                if (
                    INTRADAY_ENABLED
                    and is_weekday
                    and market_open_intraday
                    and not_lunch
                    and not_eod
                    and adaptive_gap_ok
                    and self._morning_intraday_candidates
                    and self._pending_approvals
                ):
                    try:
                        spy_now = await asyncio.to_thread(self._get_spy_day_pct)
                        if spy_now is not None and abs(spy_now - self._last_scan_spy_pct) >= SPY_ADAPTIVE_DELTA:
                            self.logger.info(
                                "SPY shifted %.1f%% → %.1f%% since last scan — adaptive re-scan triggered",
                                self._last_scan_spy_pct, spy_now,
                            )
                            self._last_adaptive_scan_at = _time.monotonic()
                            await self._withdraw_all_pending("spy_regime_shift")
                            await self._run_task(
                                "intraday_adaptive_rescan", now.hour, now.minute,
                                self.select_intraday_instruments(
                                    window=(now.hour, now.minute),
                                    use_morning_candidates=True,
                                ),
                            )
                    except Exception as _adp_exc:
                        self.logger.debug("Adaptive re-scan check failed: %s", _adp_exc)

                # ── NIS post-event refresh: re-fetch after each calendar event releases ──
                # Fires 3 minutes after each event's scheduled time so Finnhub actuals
                # are available. Uses event ID to ensure each event triggers exactly once.
                if is_weekday:
                    await self._maybe_refresh_nis_post_event(now)

                # ── 09:30–16:00: 30-minute position review + reeval drain ─────
                market_open = (now.hour > 9 or (now.hour == 9 and now.minute >= 30))
                market_close = now.hour < 16
                review_slot = (now.hour * 60 + now.minute) // REVIEW_INTERVAL_MINUTES
                if (
                    is_weekday
                    and market_open
                    and market_close
                    and review_slot != self._last_review_minute
                ):
                    self._last_review_minute = review_slot
                    try:
                        await self._review_open_positions()
                    except Exception as _rev_exc:
                        self.logger.warning("Position review error: %s", _rev_exc)
                    try:
                        await self._handle_reeval_requests()
                    except Exception as _rev_exc:
                        self.logger.warning("Reeval handler error: %s", _rev_exc)
                    try:
                        await self._rescore_pending_approvals()
                    except Exception as _rev_exc:
                        self.logger.warning("Pending approval rescore error: %s", _rev_exc)

                # ── 16:05–16:59: record daily benchmark ───────────────────────
                if (
                    is_weekday
                    and self._in_window(now, 16, 5, 17, 0)
                    and not self._benchmark_recorded_today
                ):
                    self._benchmark_recorded_today = True
                    await self._run_task(
                        "daily_benchmark", 16, 5,
                        self._record_daily_benchmark(),
                    )

                # ── 16:10–16:59 Friday: weekly performance report ─────────────
                if (
                    is_weekday
                    and now.weekday() == 4
                    and self._in_window(now, 16, 10, 17, 0)
                    and not self._weekly_report_generated_today
                ):
                    self._weekly_report_generated_today = True
                    await self._run_task(
                        "weekly_report", 16, 10,
                        self._generate_weekly_report(),
                    )

                # ── 15:55–16:29: cancel unfilled limit orders before close ────
                if (
                    is_weekday
                    and self._in_window(now, 15, 55, 16, 30)
                    and not self._eod_order_cleanup_run_today
                ):
                    self._eod_order_cleanup_run_today = True
                    await self._run_task(
                        "eod_order_cleanup", 15, 55,
                        self._cancel_eod_pending_orders(),
                    )

                # ── 16:30–16:59: EOD jobs (backfill outcomes + daily summary) ─
                if (
                    is_weekday
                    and self._in_window(now, 16, 30, 17, 0)
                    and not self._eod_jobs_run_today
                ):
                    self._eod_jobs_run_today = True
                    await self._run_task(
                        "eod_jobs", 16, 30,
                        self._run_eod_jobs(),
                    )

                # ── 17:00–23:59: retrain model (scheduled weekday only) ───────
                from app.ml.retrain_config import RETRAIN_WEEKDAY
                if (
                    is_weekday
                    and now.weekday() == RETRAIN_WEEKDAY
                    and now.hour >= 17
                    and not self._retrained_today
                ):
                    self._retrained_today = True
                    await self._run_task(
                        "model_retrain", 17, 0,
                        self._retrain(),
                    )

                await asyncio.sleep(60)

            except asyncio.CancelledError:
                self.logger.info("Portfolio Manager cancelled — shutting down")
                self.status = "stopped"
                break
            except Exception as e:
                self.logger.error("Error in portfolio manager loop: %s", e, exc_info=True)
                await self.log_decision("PORTFOLIO_MANAGER_ERROR", reasoning={"error": str(e)})
                await asyncio.sleep(10)

    # ─── Missed-task diagnostics ──────────────────────────────────────────────

    async def _warn_missed_tasks(self) -> None:
        """
        Called at daily reset.  Logs a WINDOW_MISSED decision for every critical
        task that should have run yesterday but didn't.  Visible on the dashboard
        so you can see at a glance which days had gaps.
        """
        missed = []
        # Only report on trading days (weekdays); _last_date is yesterday's date.
        try:
            from datetime import date
            d = date.fromisoformat(self._last_date)
            if d.weekday() >= 5:   # Saturday / Sunday — nothing scheduled
                return
        except Exception:
            return

        if not self._analyzed_today:
            missed.append("swing_premarket_analysis")
        if not self._selected_today:
            missed.append("swing_proposals_send")
        if not self._selected_intraday_today:
            missed.append("intraday_selection")
        if not self._retrained_today:
            missed.append("model_retrain")

        if missed:
            self.logger.warning(
                "⚠ Tasks that did NOT run on %s: %s",
                self._last_date, ", ".join(missed),
            )
            await self.log_decision("WINDOW_MISSED", reasoning={
                "date": self._last_date,
                "missed_tasks": missed,
                "task_stats": {
                    k: {kk: v for kk, v in vv.items()}
                    for k, vv in self._task_stats.items()
                },
            })
        else:
            self.logger.info("All scheduled tasks ran on %s ✓", self._last_date)

    # ─── Ticker Universe ──────────────────────────────────────────────────────

    def _get_universe(self) -> List[str]:
        """Return active tickers from DB watchlist; fall back to RUSSELL_1000_TICKERS."""
        try:
            from app.database.session import get_session
            from app.database.models import WatchlistTicker
            db = get_session()
            try:
                tickers = [
                    r.symbol for r in
                    db.query(WatchlistTicker).filter(WatchlistTicker.active == 1).all()
                ]
                if tickers:
                    return tickers
            finally:
                db.close()
        except Exception as exc:
            self.logger.debug("Watchlist DB unavailable, using RUSSELL_1000: %s", exc)
        return list(RUSSELL_1000_TICKERS)

    # ─── Instrument Selection ─────────────────────────────────────────────────

    def _fetch_swing_features(self) -> Dict[str, Dict[str, float]]:
        """Fetch bars + engineer features. Uses bulk bar fetch (4 API calls for 750 symbols)
        instead of one call per symbol, then parallelises only the CPU-bound feature engineering."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        symbols = self._get_universe()

        # ── Step 1: bulk fetch — 4 API calls instead of 750 ──────────────────
        t0 = _time.monotonic()
        bars_by_symbol = self._alpaca.get_bars_batch(symbols, timeframe="1D", limit=300)
        self.logger.info(
            "Bulk bar fetch: %d/%d symbols in %.1fs",
            len(bars_by_symbol), len(symbols), _time.monotonic() - t0,
        )

        # ── Step 2: feature engineering — CPU-bound, parallelise ─────────────
        def _engineer(symbol: str, bars) -> tuple:
            try:
                feats = self.feature_engineer.engineer_features(symbol, bars, fetch_fundamentals=False)
                return symbol, feats
            except Exception as e:
                self.logger.debug("Feature engineering failed for %s: %s", symbol, e)
                return symbol, None

        features_by_symbol: Dict[str, Dict[str, float]] = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {
                pool.submit(_engineer, sym, bars): sym
                for sym, bars in bars_by_symbol.items()
                if not bars.empty
            }
            for future in as_completed(futures, timeout=120):
                try:
                    symbol, feats = future.result(timeout=30)
                except Exception as e:
                    self.logger.debug("Feature engineering timed out or failed: %s", e)
                    continue
                if feats is not None:
                    features_by_symbol[symbol] = feats
        return features_by_symbol

    def _write_heartbeat(self) -> None:
        """Phase 83: Upsert PM heartbeat row so watchdog.py can detect process death."""
        from app.database.session import get_session
        from app.database.models import ProcessHeartbeat
        from datetime import datetime as _dt
        db = get_session()
        try:
            row = db.query(ProcessHeartbeat).filter_by(process_name="portfolio_manager").first()
            if row is None:
                row = ProcessHeartbeat(process_name="portfolio_manager", started_at=_dt.utcnow())
                db.add(row)
            row.last_beat = _dt.utcnow()
            db.commit()
        except Exception as exc:
            db.rollback()
            self.logger.debug("Heartbeat upsert failed: %s", exc)
        finally:
            db.close()

    async def _maybe_refresh_nis_post_event(self, now: datetime) -> None:
        """
        Re-fetch NIS macro context 3 minutes after each calendar event releases.

        Finnhub populates actuals within ~1-2 min of release. Waiting 3 min gives
        it time to settle, then we invalidate the day cache and rebuild so the
        updated risk level / sizing factor is visible to all downstream gates.
        Each event fires exactly once per day (tracked by event ID).
        """
        POST_EVENT_DELAY_MINUTES = 3
        try:
            from app.news.sources.finnhub_source import fetch_economic_calendar
            from app.news.intelligence_service import nis
            from datetime import timezone

            events = fetch_economic_calendar(days_ahead=0, min_impact="medium")
            for evt in events:
                evt_id = evt.get("id")
                if not evt_id or evt_id in self._nis_refreshed_event_ids:
                    continue
                evt_dt = evt.get("event_time")
                if evt_dt is None:
                    continue
                # Convert to ET for display, compare in UTC
                now_utc = now.astimezone(timezone.utc)
                elapsed_min = (now_utc - evt_dt).total_seconds() / 60
                if POST_EVENT_DELAY_MINUTES <= elapsed_min < POST_EVENT_DELAY_MINUTES + 5:
                    self._nis_refreshed_event_ids.add(evt_id)
                    self.logger.info(
                        "NIS post-event refresh triggered: %s released %.1f min ago — invalidating cache",
                        evt.get("event_type"), elapsed_min,
                    )
                    nis.invalidate_macro_cache()
                    ctx = await asyncio.to_thread(nis.get_macro_context)
                    # Persist updated snapshot for API and audit trail
                    try:
                        from app.database.decision_audit import persist_nis_macro_snapshot
                        await asyncio.to_thread(
                            persist_nis_macro_snapshot, ctx,
                            "post_event",
                            evt.get("event_type"),
                            evt.get("event_name"),
                        )
                    except Exception:
                        pass
                    await self.log_decision("NIS_POST_EVENT_REFRESH", reasoning={
                        "event_type": evt.get("event_type"),
                        "event_name": evt.get("event_name"),
                        "elapsed_minutes": round(elapsed_min, 1),
                        "new_risk": ctx.overall_risk,
                        "new_sizing": ctx.global_sizing_factor,
                        "block_entries": ctx.block_new_entries,
                        "rationale": ctx.rationale,
                    })
                    self.logger.info(
                        "NIS post-event refresh complete: risk=%s sizing=%.2f block=%s",
                        ctx.overall_risk, ctx.global_sizing_factor, ctx.block_new_entries,
                    )
        except Exception as exc:
            self.logger.debug("NIS post-event refresh check failed (non-fatal): %s", exc)

    async def _prefetch_earnings_calendar(self):
        """
        06:00 ET: Warm the earnings calendar cache for all watchlist symbols.
        Phase 81: fail-closed — if both Finnhub and FMP fail, swing entries will be
        blocked automatically until the cache is refreshed.
        """
        from app.calendars.earnings import earnings_calendar
        symbols = list(RUSSELL_1000_TICKERS)
        await asyncio.to_thread(earnings_calendar.prefetch, symbols)

    async def _run_premarket_intelligence(self):
        """
        09:00 ET: Run pre-market intelligence routine.
        Fetches macro events, SPY context, checks overnight gaps on open positions,
        polls 8-K filings. Results cached in premarket_intel singleton for the session.
        """
        from app.agents.premarket import premarket_intel
        from app.database.session import get_session
        from app.database.models import Trade

        # Get currently open positions
        open_symbols: List[str] = []
        db = get_session()
        try:
            trades = db.query(Trade).filter_by(status="ACTIVE").all()
            open_symbols = [t.symbol for t in trades]
        except Exception:
            pass
        finally:
            db.close()

        summary = await asyncio.to_thread(
            premarket_intel.run_premarket_routine,
            open_symbols,
            self.send_message,
        )
        self.logger.info("Pre-market intelligence: %s", summary)
        await self.log_decision("PREMARKET_INTELLIGENCE", reasoning=summary)

        # Phase R3: cache regime context for this session's proposal writes
        self._current_regime_ctx = premarket_intel.get_regime_context()

        # Phase 62: pre-warm NIS stock signal cache for candidate universe
        await self._run_morning_nis_digest()

    async def _run_morning_nis_digest(self) -> None:
        """
        Phase 62 — 09:00 ET: Pre-score the swing candidate universe via NIS Tier 2.

        Populates the in-memory stock signal cache so _build_proposals() at 09:50
        reads from cache rather than making live LLM calls on the critical path.
        Also refreshes Tier 1 macro context (force_refresh=True after premarket runs).
        """
        try:
            from app.news.intelligence_service import nis

            # Tier 1: refresh now that premarket data is available
            macro_ctx = await asyncio.to_thread(nis.get_macro_context, True)
            self.logger.info(
                "NIS morning digest: macro risk=%s sizing=%.2f block=%s",
                macro_ctx.overall_risk, macro_ctx.global_sizing_factor, macro_ctx.block_new_entries,
            )

            # Tier 2: pre-score top candidates from yesterday's model ranking
            # Use the symbols already selected by _analyze_swing_premarket
            candidates = [p["symbol"] for p in self._swing_proposals] if self._swing_proposals else []
            if not candidates:
                # Fallback: score a broad set from the universe
                candidates = self._get_universe()[:50]

            sector_map = {}
            try:
                from app.database.session import get_session
                from app.database.models import WatchlistTicker
                db = get_session()
                try:
                    rows = db.query(WatchlistTicker).filter(WatchlistTicker.active == 1).all()
                    sector_map = {r.symbol: (r.sector or "Unknown") for r in rows}
                finally:
                    db.close()
            except Exception:
                pass

            def _pre_score_batch():
                nis.invalidate_stock_cache()
                signals = nis.get_stock_signals_batch(candidates, sector_map=sector_map, macro_context=macro_ctx)
                blocked = [s for s, sig in signals.items() if sig.action_policy == "block_entry"]
                sized_down = [s for s, sig in signals.items() if "size_down" in sig.action_policy]
                return len(signals), blocked, sized_down

            n, blocked, sized_down = await asyncio.to_thread(_pre_score_batch)
            self.logger.info(
                "NIS morning digest: pre-scored %d symbols — %d block_entry, %d size_down",
                n, len(blocked), len(sized_down),
            )
            if blocked:
                self.logger.info("NIS block_entry pre-market: %s", blocked)
            await self.log_decision("NIS_MORNING_DIGEST", reasoning={
                "scored": n,
                "block_entry": blocked,
                "size_down": sized_down,
                "macro_risk": macro_ctx.overall_risk,
                "macro_sizing": macro_ctx.global_sizing_factor,
            })
        except Exception as exc:
            self.logger.warning("NIS morning digest failed (non-fatal): %s", exc)

    async def _cancel_eod_pending_orders(self) -> None:
        """
        15:55 ET: Cancel any PENDING_FILL trades whose limit orders won't fill before close.
        Alpaca cancels unfilled day orders at 4 PM anyway, but doing it ourselves at 15:55
        ensures the DB is updated before shutdown — preventing ghost/duplicate records on restart.
        """
        from app.database.session import get_session
        from app.database.models import Trade
        from app.integrations import get_alpaca_client

        db = get_session()
        try:
            pending = db.query(Trade).filter_by(status="PENDING_FILL").all()
            if not pending:
                self.logger.info("EOD order cleanup: no PENDING_FILL trades to cancel")
                return

            alpaca = get_alpaca_client()
            cancelled = 0
            for trade in pending:
                if not trade.alpaca_order_id:
                    trade.status = "CANCELLED"
                    trade.exit_reason = "EOD_NO_ORDER_ID"
                    cancelled += 1
                    continue
                try:
                    order_status = alpaca.get_order_status(trade.alpaca_order_id)
                    status_str = str((order_status or {}).get("status", "")).lower()
                    if status_str in ("new", "pending_new", "accepted", "held", "partially_filled"):
                        # Still open — cancel it in Alpaca then mark DB
                        try:
                            alpaca.trading_client.cancel_order_by_id(trade.alpaca_order_id)
                        except Exception:
                            pass  # Alpaca may already be cancelling at EOD
                        trade.status = "CANCELLED"
                        trade.exit_reason = "EOD_LIMIT_EXPIRED"
                        cancelled += 1
                        self.logger.info(
                            "EOD cleanup: cancelled PENDING_FILL Trade#%d %s (order %s)",
                            trade.id, trade.symbol, trade.alpaca_order_id,
                        )
                    elif status_str in ("canceled", "expired", "rejected"):
                        trade.status = "CANCELLED"
                        trade.exit_reason = "EOD_LIMIT_EXPIRED"
                        cancelled += 1
                except Exception as exc:
                    self.logger.warning(
                        "EOD cleanup: could not check/cancel Trade#%d %s: %s",
                        trade.id, trade.symbol, exc,
                    )
            db.commit()
            self.logger.info("EOD order cleanup: cancelled %d PENDING_FILL trade(s)", cancelled)
        except Exception as exc:
            db.rollback()
            self.logger.error("EOD order cleanup failed: %s", exc)
        finally:
            db.close()

    async def _run_eod_jobs(self) -> None:
        """
        16:30 ET: Back-fill outcome P&L for decision_audit rows and write daily summary.
        Runs after market close so trades have settled.
        """
        # Gap 2: back-fill realized outcomes for all entered decisions
        try:
            from app.database.decision_audit import backfill_outcomes
            n = await asyncio.to_thread(backfill_outcomes, 14)
            self.logger.info("EOD backfill: %d decision_audit rows updated with outcomes", n)
        except Exception as exc:
            self.logger.warning("EOD backfill failed (non-fatal): %s", exc)

        # Gap 2b: back-fill counterfactual outcomes for gated (blocked) decisions
        try:
            from app.database.decision_audit import backfill_gate_outcomes, backfill_scan_abstention_outcomes
            gate_res = await asyncio.to_thread(backfill_gate_outcomes, 14)
            scan_res = await asyncio.to_thread(backfill_scan_abstention_outcomes, 14)
            self.logger.info(
                "EOD gate backfill: %s gate outcomes, %s scan abstentions updated",
                gate_res, scan_res,
            )
        except Exception as exc:
            self.logger.warning("EOD gate backfill failed (non-fatal): %s", exc)

        # Gap 3: write daily summary row to risk_metrics
        try:
            from app.database.daily_summary import write_daily_summary
            await asyncio.to_thread(write_daily_summary)
            self.logger.info("EOD daily summary written to risk_metrics")
        except Exception as exc:
            self.logger.warning("EOD daily summary failed (non-fatal): %s", exc)

        # Phase R4: log regime divergence for today
        try:
            await asyncio.to_thread(self._log_regime_divergence_today)
        except Exception as exc:
            self.logger.warning("Regime divergence log failed (non-fatal): %s", exc)

    def _log_regime_divergence_today(self) -> None:
        """EOD: log days where regime model and opportunity score disagree."""
        from app.database.session import get_session
        from app.database.models import ProposalLog

        today_start = datetime.now(ET).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
        db = get_session()
        try:
            rows = (
                db.query(ProposalLog)
                .filter(
                    ProposalLog.proposed_at >= today_start,
                    ProposalLog.regime_score_at_scan.isnot(None),
                    ProposalLog.opportunity_score.isnot(None),
                )
                .all()
            )
            if not rows:
                self.logger.info("Regime divergence: no proposals with regime+opportunity scores today")
                return

            n_regime_block = sum(1 for r in rows if r.regime_score_at_scan < 0.35)
            n_opp_block = sum(1 for r in rows if r.opportunity_score is not None and r.opportunity_score < 0.5)
            n_both_block = sum(1 for r in rows if r.regime_score_at_scan < 0.35 and (r.opportunity_score or 1) < 0.5)
            n_diverge = n_regime_block + n_opp_block - 2 * n_both_block

            self.logger.info(
                "REGIME DIVERGENCE TODAY: %d proposals | regime_block=%d opp_block=%d diverge=%d",
                len(rows), n_regime_block, n_opp_block, n_diverge,
            )
        finally:
            db.close()

    async def _record_daily_benchmark(self) -> None:
        """
        16:05 ET: Record today's strategy P&L vs SPY for benchmark comparison (Phase 22).
        """
        try:
            from app.agents.performance_monitor import performance_monitor
            from app.database.session import get_session
            from app.database.models import Trade
            from datetime import datetime as _dt
            today_start = _dt.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            db = get_session()
            try:
                trades = (
                    db.query(Trade)
                    .filter(Trade.status == "CLOSED", Trade.closed_at >= today_start)
                    .all()
                )
                daily_pnl = sum(float(t.pnl) for t in trades if t.pnl is not None)
            finally:
                db.close()
            from app.integrations import get_alpaca_client
            alpaca = get_alpaca_client()
            account = alpaca.get_account()
            account_value = float(account.get("portfolio_value", 100_000))
            await asyncio.to_thread(
                performance_monitor.fetch_and_record_spy_return, daily_pnl, account_value
            )
            self.logger.info("Daily benchmark recorded: daily_pnl=%.2f", daily_pnl)
        except Exception as exc:
            self.logger.warning("Daily benchmark recording failed: %s", exc)

    async def _generate_weekly_report(self) -> None:
        """
        16:10 ET every Friday: generate and log the weekly performance report (Phase 22).
        """
        try:
            from app.agents.performance_monitor import performance_monitor
            from app.database.session import get_session
            db = get_session()
            try:
                report = await asyncio.to_thread(performance_monitor.generate_weekly_report, db)
            finally:
                db.close()
            self.logger.info("Weekly performance report: %s", report)
            await self.log_decision("WEEKLY_PERFORMANCE_REPORT", reasoning=report)
        except Exception as exc:
            self.logger.warning("Weekly report generation failed: %s", exc)

        # Phase R4: regime weekly summary
        try:
            await asyncio.to_thread(self._log_regime_weekly_summary)
        except Exception as exc:
            self.logger.warning("Regime weekly summary failed (non-fatal): %s", exc)

    def _log_regime_weekly_summary(self) -> None:
        """Log regime model weekly summary: RISK_OFF/NEUTRAL/RISK_ON day counts + divergence."""
        from datetime import timedelta
        from app.database.session import get_session
        from app.database.models import RegimeSnapshot, ProposalLog
        from sqlalchemy import func

        cutoff = datetime.now(ET).date() - timedelta(days=7)
        db = get_session()
        try:
            label_counts = (
                db.query(RegimeSnapshot.regime_label, func.count(RegimeSnapshot.id))
                .filter(
                    RegimeSnapshot.snapshot_date >= cutoff,
                    RegimeSnapshot.snapshot_trigger != "backfill",
                )
                .group_by(RegimeSnapshot.regime_label)
                .all()
            )
            counts = {label: n for label, n in label_counts}

            # Intraday proposals sent by regime label
            intra_by_label = (
                db.query(ProposalLog.regime_label_at_scan, func.count(ProposalLog.id))
                .filter(
                    ProposalLog.proposed_at >= datetime.now(ET).replace(tzinfo=None) - timedelta(days=7),
                    ProposalLog.strategy == "intraday",
                    ProposalLog.pm_status.in_(["SENT", "PENDING"]),
                    ProposalLog.regime_label_at_scan.isnot(None),
                )
                .group_by(ProposalLog.regime_label_at_scan)
                .all()
            )
            intra_counts = {label: n for label, n in intra_by_label}

            self.logger.info(
                "REGIME WEEKLY SUMMARY (last 7 days): "
                "RISK_OFF=%d days, NEUTRAL=%d days, RISK_ON=%d days | "
                "Intraday proposals by label: RISK_OFF=%d NEUTRAL=%d RISK_ON=%d",
                counts.get("RISK_OFF", 0), counts.get("NEUTRAL", 0), counts.get("RISK_ON", 0),
                intra_counts.get("RISK_OFF", 0), intra_counts.get("NEUTRAL", 0), intra_counts.get("RISK_ON", 0),
            )
        finally:
            db.close()

    def _fetch_bars_bulk_for_factor(self) -> dict:
        """Fetch 300-day daily bars for the universe — shared between ML and factor paths."""
        symbols = self._get_universe()
        return self._alpaca.get_bars_batch(symbols, timeframe="1D", limit=300)

    async def _analyze_swing_factor_portfolio(self) -> None:
        """Phase D: factor portfolio swing selection (momentum + quality composite).

        Replaces ML model scoring when pm.swing_selector='factor_portfolio'.
        Top-20 by composite score, regime-gated (SPY>MA200 + VIX<30).
        Proposals use confidence = min-max normalised composite score.
        """
        import pandas as pd
        from app.ml.factor_scorer import (
            compute_composite_score, select_top_n, regime_gate_ok,
        )

        universe = self._get_universe()
        self.logger.info(
            "Factor portfolio swing scan: fetching bars for %d symbols...", len(universe)
        )
        t0 = _time.monotonic()
        try:
            bars_by_symbol: dict = await asyncio.wait_for(
                asyncio.to_thread(
                    self._alpaca.get_bars_batch, universe, "1D", 300
                ),
                timeout=1200,
            )
        except asyncio.TimeoutError:
            self.logger.error("Factor portfolio bar fetch timed out — skipping")
            await self.log_decision("SELECTION_SKIPPED", reasoning={
                "reason": "factor_bar_fetch_timeout",
                "strategy": "swing",
                "selector": "factor_portfolio",
            })
            return
        self.logger.info(
            "Factor portfolio bar fetch: %d/%d symbols in %.1fs",
            len(bars_by_symbol), len(universe), _time.monotonic() - t0,
        )

        if not bars_by_symbol:
            self.logger.warning("Factor portfolio: no bars fetched — skipping")
            await self.log_decision("SELECTION_SKIPPED", reasoning={
                "reason": "no_bars",
                "strategy": "swing",
                "selector": "factor_portfolio",
            })
            return

        # Build aligned closes DataFrame (each column = one symbol)
        close_cols: dict[str, pd.Series] = {}
        for sym, df in bars_by_symbol.items():
            if df is not None and not df.empty and "close" in df.columns:
                close_cols[sym] = df["close"]
        closes = pd.DataFrame(close_cols)
        closes.index = pd.to_datetime(closes.index)

        as_of = closes.index[-1]  # most recent bar date

        # Regime gate: SPY > MA200 AND VIX < 30
        spy_ok = True
        try:
            vix_val = self._fetch_vix_level()
            spy_series = closes.get("SPY", pd.Series(dtype=float))
            spy_ok = regime_gate_ok(spy_series, as_of, vix_value=vix_val)
            if not spy_ok:
                self.logger.warning(
                    "Factor portfolio regime gate BLOCKED (SPY<MA200 or VIX>=30, VIX=%.1f)", vix_val
                )
                await self.log_decision("SELECTION_SKIPPED", reasoning={
                    "reason": "factor_regime_gate_blocked",
                    "strategy": "swing",
                    "selector": "factor_portfolio",
                    "vix": round(vix_val, 1),
                })
                return
        except Exception as _rg_exc:
            self.logger.warning("Factor regime gate check failed (permissive): %s", _rg_exc)

        # Compute composite score
        try:
            scores = compute_composite_score(as_of, closes, bars_by_symbol)
        except Exception as _sc_exc:
            self.logger.error("Factor score computation failed: %s", _sc_exc, exc_info=True)
            await self.log_decision("SELECTION_SKIPPED", reasoning={
                "reason": "factor_score_error",
                "strategy": "swing",
                "selector": "factor_portfolio",
                "error": str(_sc_exc),
            })
            return

        if scores.empty:
            self.logger.warning("Factor portfolio: all scores empty — skipping")
            return

        # Exclude currently-held symbols
        from app.database.session import get_session as _gs3
        from app.database.models import Trade as _Trade3
        _hdb3 = _gs3()
        try:
            _open3 = _hdb3.query(_Trade3).filter(
                _Trade3.status.in_(["ACTIVE", "PENDING_FILL"]),
                _Trade3.trade_type == "swing",
            ).all()
            _held3 = {t.symbol for t in _open3}
        finally:
            _hdb3.close()
        scores = scores.drop(index=list(_held3), errors="ignore")

        top_n_count = 20  # Phase D: top-20 equal-weight
        try:
            from app.database.session import get_session as _gs4
            from app.database.agent_config import get_agent_config as _gac4
            _db4 = _gs4()
            try:
                _cfg_top_n = _gac4(_db4, "pm.top_n_stocks")
                if isinstance(_cfg_top_n, int) and _cfg_top_n > 0:
                    top_n_count = _cfg_top_n
            finally:
                _db4.close()
        except Exception:
            pass

        top_syms = select_top_n(scores, n=top_n_count)

        # Map composite score → [0,1] confidence for _build_proposals compatibility
        s_min, s_max = float(scores.min()), float(scores.max())
        s_range = max(s_max - s_min, 1e-6)

        def _norm_conf(sym: str) -> float:
            raw = float(scores.get(sym, s_min))
            return 0.55 + 0.40 * ((raw - s_min) / s_range)  # maps to [0.55, 0.95]

        selected = [(sym, _norm_conf(sym)) for sym in top_syms]

        self.logger.info(
            "Factor portfolio scan: scored %d symbols, top-%d selected: %s",
            len(scores), len(selected), [s for s, _ in selected[:5]],
        )

        self._swing_proposals = await self._build_proposals(selected)

        await self.log_decision(
            "SWING_PREMARKET_ANALYSIS",
            reasoning={
                "selector": "factor_portfolio",
                "candidates": [
                    {"symbol": s, "confidence": round(float(c), 4)} for s, c in selected
                ],
                "total_evaluated": len(scores),
                "send_time": "09:50 ET",
            },
        )

    async def _build_directional_proposals(
        self,
        scored: list,
        selector: str = "pead",
        max_hold_days: int = 0,
    ) -> list:
        """Build trade proposals supporting both long (BUY) and short (SELL_SHORT) directions.

        Args:
            scored: List of (symbol, confidence, direction) where direction is "long" or "short"
                    and confidence is positive for longs, negative for shorts.
            selector: Label for logging/audit trail.
            max_hold_days: If > 0, annotate proposals with max_hold_days metadata so the
                           Trader agent can enforce time-based exits (PEAD hold-5).
        """
        try:
            account = self._alpaca.get_account()
            account_value = account["portfolio_value"]
        except Exception:
            account_value = 20_000.0

        # Exclude currently-held symbols
        from app.database.session import get_session as _gs_dp
        from app.database.models import Trade as _Trade_dp
        _db_dp = _gs_dp()
        try:
            _open_dp = _db_dp.query(_Trade_dp).filter(
                _Trade_dp.status.in_(["ACTIVE", "PENDING_FILL"]),
                _Trade_dp.trade_type == "swing",
            ).all()
            # Include direction in held-set: allow opposite-direction entry on same symbol
            _held_dp = {(t.symbol, getattr(t, "direction", "BUY") or "BUY") for t in _open_dp}
        finally:
            _db_dp.close()

        proposals = []
        for sym, conf, direction in scored:
            _prop_dir = "SELL_SHORT" if direction == "short" else "BUY"
            if (sym, _prop_dir) in _held_dp:
                continue
            price = self._alpaca.get_latest_price(sym)
            if price is None or price <= 0:
                continue

            abs_conf = abs(conf)
            is_short = direction == "short"

            quantity = self._calculate_quantity(
                price, account_value, trade_type="swing", confidence=float(abs_conf)
            )
            _regime_mult, _regime_lbl, _regime_score = self._regime_sizing_multiplier()
            if _regime_mult != 1.0:
                quantity = max(1, int(quantity * _regime_mult))

            if quantity <= 0:
                continue

            if is_short:
                stop_loss = round(price * 1.05, 2)       # 5% above entry
                profit_target = round(price * 0.95, 2)   # 5% below entry
                trade_direction = "SELL_SHORT"
            else:
                stop_loss = round(price * 0.98, 2)
                profit_target = round(price * 1.05, 2)
                trade_direction = "BUY"

            proposal: dict = {
                "symbol": sym,
                "direction": trade_direction,
                "quantity": quantity,
                "entry_price": price,
                "confidence": float(abs_conf),
                "stop_loss": stop_loss,
                "profit_target": profit_target,
                "source_agent": "portfolio_manager",
                "trade_type": "swing",
                "proposal_uuid": str(uuid.uuid4()),
                "selector": selector,
            }
            if max_hold_days > 0:
                proposal["max_hold_days"] = max_hold_days

            # Persist to ProposalLog so the SENT update in send_swing_proposals finds the row
            try:
                from app.database.models import ProposalLog as _PLdp
                from app.database.session import get_session as _gs_pl
                _pl_db = _gs_pl()
                try:
                    _regime = self._current_regime_ctx or {}
                    _pl_row = _PLdp(
                        strategy="swing",
                        batch_id=f"dir_{selector}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                        scan_time=datetime.utcnow(),
                        symbol=sym,
                        ml_score=round(float(abs_conf), 4),
                        confidence=round(float(abs_conf), 4),
                        above_threshold=True,
                        pm_status="SCORED",
                        proposed_at=datetime.utcnow(),
                        proposal_uuid=proposal["proposal_uuid"],
                        direction=trade_direction,
                        entry_price=price,
                        stop_price=proposal.get("stop_loss"),
                        target_price=proposal.get("profit_target"),
                        quantity=proposal.get("quantity"),
                        regime_score_at_scan=_regime.get("regime_score"),
                        regime_label_at_scan=_regime.get("regime_label"),
                    )
                    _pl_db.add(_pl_row)
                    _pl_db.commit()
                finally:
                    _pl_db.close()
            except Exception as _ple:
                self.logger.debug("ProposalLog directional write failed (non-fatal): %s", _ple)

            # NIS overlay: direction-aware — block_entry means OPPOSITE for shorts
            try:
                from app.news.intelligence_service import nis
                from app.agents.premarket import premarket_intel
                _macro_ctx = premarket_intel.macro_context
                _sector = self._get_symbol_sector(sym)
                _news_sig = await asyncio.to_thread(
                    nis.get_stock_signal, sym, _sector, 24, _macro_ctx
                )
                if is_short:
                    # For shorts: negative news (block_entry/size_down) = green light
                    # Positive news (size_up) = headwind → reduce size
                    if _news_sig.action_policy in ("size_up_light", "size_up_heavy"):
                        old_qty = proposal["quantity"]
                        proposal["quantity"] = max(1, int(old_qty * 0.5))
                        self.logger.info(
                            "NIS short headwind %s: positive news → halving qty %d→%d — %s",
                            sym, old_qty, proposal["quantity"], _news_sig.rationale,
                        )
                    # block_entry on negative news = supports short, do nothing (allow)
                else:
                    # For longs: standard NIS logic
                    if _news_sig.action_policy == "block_entry":
                        self.logger.info(
                            "NIS block_entry (long) %s [%s]: %s", sym, selector, _news_sig.rationale
                        )
                        continue
                    if _news_sig.sizing_multiplier != 1.0:
                        old_qty = proposal["quantity"]
                        proposal["quantity"] = max(1, int(old_qty * _news_sig.sizing_multiplier))
                        self.logger.info(
                            "NIS sizing (long) %s: %.2f× (%d→%d) — %s",
                            sym, _news_sig.sizing_multiplier, old_qty,
                            proposal["quantity"], _news_sig.rationale,
                        )
                proposal["news_signal"] = {
                    "action_policy": _news_sig.action_policy,
                    "sizing_multiplier": _news_sig.sizing_multiplier,
                    "rationale": _news_sig.rationale,
                }
            except Exception as _nis_exc:
                self.logger.debug("NIS directional overlay failed for %s (non-fatal): %s", sym, _nis_exc)

            proposals.append(proposal)

        return proposals

    async def _analyze_swing_pead(self) -> None:
        """PEAD scorer: enter long (and optionally short) within 3 days of EPS surprise > ±5%.
        Hold cap: 5 trading days (max_hold_days=5 annotated on each proposal).

        Short support requires pm.pead_enable_shorts=true AND a margin-enabled Alpaca account.
        Default is longs-only to allow paper trading before short order routing is wired.
        """
        from app.ml.pead_scorer import PEADScorer
        from app.database.session import get_session as _gs_pead
        from app.database.agent_config import get_agent_config as _gac_pead
        from datetime import date as _date
        import pandas as _pd

        # Check whether short PEAD signals are enabled (default: longs only for Phase I)
        _shorts_enabled = False
        try:
            _db_pead = _gs_pead()
            try:
                _shorts_enabled = str(_gac_pead(_db_pead, "pm.pead_enable_shorts") or "false").lower() == "true"
            finally:
                _db_pead.close()
        except Exception:
            pass

        universe = self._get_universe()
        self._swing_proposals = []  # always reset before early returns
        self.logger.info("PEAD scan: fetching bars for %d symbols (shorts_enabled=%s)...",
                         len(universe), _shorts_enabled)
        try:
            bars_by_symbol = await asyncio.wait_for(
                asyncio.to_thread(self._alpaca.get_bars_batch, universe, "1D", 60),
                timeout=600,
            )
        except asyncio.TimeoutError:
            self.logger.error("PEAD bar fetch timed out — skipping")
            return

        if not bars_by_symbol:
            self.logger.warning("PEAD: no bars fetched — skipping")
            return

        symbols_data = {s: df for s, df in bars_by_symbol.items() if df is not None and not df.empty}
        scorer = PEADScorer(long_threshold=0.05, short_threshold=-0.05,
                            max_days_after=3, long_short=_shorts_enabled)
        scored = scorer(day=_pd.Timestamp(_date.today()), symbols_data=symbols_data)
        # If shorts disabled, filter to longs only
        if not _shorts_enabled:
            scored = [(s, c, d) for s, c, d in scored if d == "long"]

        self.logger.info("PEAD: %d signals generated (long+short)", len(scored))
        if not scored:
            await self.log_decision("SELECTION_SKIPPED", reasoning={
                "reason": "no_pead_signals", "selector": "pead",
            })
            return

        self._swing_proposals = await self._build_directional_proposals(
            scored, selector="pead", max_hold_days=5,
        )
        await self.log_decision("SWING_PREMARKET_ANALYSIS", reasoning={
            "selector": "pead",
            "signals": [{"symbol": s, "conf": round(abs(c), 4), "direction": d} for s, c, d in scored[:10]],
            "total_signals": len(scored),
            "proposals": len(self._swing_proposals),
            "send_time": "09:50 ET",
        })

    async def _analyze_swing_quality_short(self) -> None:
        """QualityShort scorer (shorts_only): short fundamentally deteriorating stocks."""
        from app.ml.short_scorers import QualityShortScorer

        self._swing_proposals = []  # always reset before early returns
        universe = self._get_universe()
        self.logger.info("QualityShort scan: fetching bars for %d symbols...", len(universe))
        try:
            bars_by_symbol = await asyncio.wait_for(
                asyncio.to_thread(self._alpaca.get_bars_batch, universe, "1D", 300),
                timeout=1200,
            )
        except asyncio.TimeoutError:
            self.logger.error("QualityShort bar fetch timed out — skipping")
            return

        if not bars_by_symbol:
            self.logger.warning("QualityShort: no bars fetched — skipping")
            return

        symbols_data = {s: df for s, df in bars_by_symbol.items() if df is not None and not df.empty}
        scorer = QualityShortScorer(max_shorts=15, flags_required=2, legs_mode="shorts_only")
        import pandas as _pd
        from datetime import date as _date
        scored = scorer(day=_pd.Timestamp(_date.today()), symbols_data=symbols_data)

        self.logger.info("QualityShort: %d short signals generated", len(scored))
        if not scored:
            await self.log_decision("SELECTION_SKIPPED", reasoning={
                "reason": "no_quality_short_signals", "selector": "quality_short",
            })
            return

        self._swing_proposals = await self._build_directional_proposals(
            scored, selector="quality_short", max_hold_days=0,
        )
        await self.log_decision("SWING_PREMARKET_ANALYSIS", reasoning={
            "selector": "quality_short",
            "signals": [{"symbol": s, "conf": round(abs(c), 4), "direction": d} for s, c, d in scored[:10]],
            "total_signals": len(scored),
            "proposals": len(self._swing_proposals),
            "send_time": "09:50 ET",
        })

    async def _analyze_swing_pead_quality_short(self) -> None:
        """Combined: PEAD events (long+short, hold-5) + QualityShort shorts-only.
        PEAD proposals take priority; QualityShort fills remaining capacity.
        """
        await self._analyze_swing_pead()
        pead_proposals = list(self._swing_proposals or [])
        pead_syms = {p["symbol"] for p in pead_proposals}

        await self._analyze_swing_quality_short()
        qs_proposals = [p for p in (self._swing_proposals or []) if p["symbol"] not in pead_syms]

        self._swing_proposals = pead_proposals + qs_proposals
        self.logger.info(
            "pead_quality_short combined: %d PEAD + %d QualityShort = %d total proposals",
            len(pead_proposals), len(qs_proposals), len(self._swing_proposals),
        )

    async def _analyze_swing_premarket(self):
        """
        08:00 ET: Score all universe stocks using yesterday's daily bars.
        Results cached in self._swing_proposals — not sent to Risk Manager yet.
        Sending is deferred to 09:50 so we enter after the open volatility spike.
        """
        model_ver = getattr(self.model, "version", None)

        # Phase D: route to factor portfolio if configured or model not trained
        selector = "ml_model"
        try:
            from app.database.session import get_session as _gs2
            from app.database.agent_config import get_agent_config as _gac
            _db2 = _gs2()
            try:
                selector = _gac(_db2, "pm.swing_selector") or "factor_portfolio"
            finally:
                _db2.close()
        except Exception:
            pass

        if selector == "pead":
            self.logger.info("Pre-market swing analysis starting — PEAD selector")
            await self._analyze_swing_pead()
            return

        if selector == "quality_short":
            self.logger.info("Pre-market swing analysis starting — quality_short selector")
            await self._analyze_swing_quality_short()
            return

        if selector == "pead_quality_short":
            self.logger.info("Pre-market swing analysis starting — pead_quality_short combined selector")
            await self._analyze_swing_pead_quality_short()
            return

        if selector == "factor_portfolio" or not self.model.is_trained:
            self.logger.info(
                "Pre-market swing analysis starting — factor_portfolio selector (model v%s)",
                model_ver,
            )
            await self._analyze_swing_factor_portfolio()
            return

        self.logger.info("Pre-market swing analysis starting — model v%s", model_ver)

        universe = self._get_universe()
        self.logger.info("Fetching features for %d symbols...", len(universe))
        t0 = _time.monotonic()

        # Run blocking network calls in a thread so the event loop stays free
        try:
            features_by_symbol = await asyncio.wait_for(
                asyncio.to_thread(self._fetch_swing_features), timeout=1200
            )
        except asyncio.TimeoutError:
            self.logger.error("Swing feature fetch timed out after 20 minutes — skipping")
            await self.log_decision("SELECTION_SKIPPED", reasoning={
                "reason": "feature fetch timeout",
                "strategy": "swing",
                "model_version": model_ver,
                "universe_size": len(universe),
            })
            return
        fetch_ms = int((_time.monotonic() - t0) * 1000)
        self.logger.info(
            "Swing feature fetch complete — %d/%d symbols in %.1fs",
            len(features_by_symbol), len(universe), fetch_ms / 1000,
        )

        if not features_by_symbol:
            self.logger.warning("Could not build features for any symbol — skipping")
            await self.log_decision("SELECTION_SKIPPED", reasoning={
                "reason": "no features computed",
                "strategy": "swing",
                "model_version": model_ver,
                "universe_size": len(universe),
            })
            return

        self._last_swing_features = features_by_symbol  # Gap 1: cache for decision audit
        symbols = list(features_by_symbol.keys())

        # P1 BenignGate: block all swing signals when macro regime is adverse
        try:
            from app.strategy.benign_gate import BenignGate
            _bg = BenignGate()
            symbols = _bg.gate(symbols, reason="swing_ml")
            if not symbols:
                self.logger.warning("BenignGate blocked all swing signals — adverse regime")
                await self.log_decision("SELECTION_SKIPPED", reasoning={
                    "reason": "benign_gate_adverse_regime",
                    "strategy": "swing",
                    "model_version": model_ver,
                    "universe_size": len(features_by_symbol),
                })
                return
        except Exception as _bg_exc:
            self.logger.warning("BenignGate check failed (non-fatal, proceeding): %s", _bg_exc)

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
            # Option B: regime-aware routing if model was trained with regime split.
            # Falls back to plain predict() if no high-vix sibling is loaded.
            if getattr(self.model, "_highvix_sibling", None) is not None:
                try:
                    _vix_now = self._fetch_vix_level()
                except Exception:
                    _vix_now = None
                _, probabilities = self.model.predict_with_vix(X, vix_level=_vix_now)
                self.logger.info(
                    "Model v%s [regime-split, VIX=%s] scored %d symbols — max=%.3f median=%.3f min=%.3f",
                    model_ver, ("%.1f" % _vix_now) if _vix_now else "?",
                    len(probabilities),
                    max(probabilities), float(np.median(probabilities)), min(probabilities),
                )
            else:
                _, probabilities = self.model.predict(X)
                self.logger.info(
                    "Model v%s scored %d symbols — max=%.3f median=%.3f min=%.3f",
                    model_ver, len(probabilities),
                    max(probabilities), float(np.median(probabilities)), min(probabilities),
                )
        except Exception as e:
            self.logger.error("Model prediction failed: %s", e, exc_info=True)
            await self.log_decision("SELECTION_SKIPPED", reasoning={
                "reason": "prediction error",
                "strategy": "swing",
                "model_version": model_ver,
                "error": str(e),
            })
            return

        min_conf, top_n = MIN_CONFIDENCE, TOP_N_STOCKS
        try:
            from app.database.session import get_session
            from app.database.agent_config import get_agent_config
            _db = get_session()
            try:
                min_conf = get_agent_config(_db, "pm.min_confidence")
                top_n = get_agent_config(_db, "pm.top_n_stocks")
            finally:
                _db.close()
        except Exception:
            pass

        target_upside = self._fetch_target_upside(symbols)
        boosted_probs = []
        for sym, prob in zip(symbols, probabilities):
            upside = target_upside.get(sym, 0.0)
            boost = min(0.05, max(0.0, upside * 0.15)) if upside > 0.10 else 0.0
            boosted_probs.append(float(prob) + boost)

        # Fix 1a: exclude symbols already held so PM never re-proposes open positions
        from app.database.session import get_session as _gs
        from app.database.models import Trade as _Trade
        _hdb = _gs()
        try:
            _held = {t.symbol for t in _hdb.query(_Trade).filter_by(status="ACTIVE").all()}
        finally:
            _hdb.close()
        if _held:
            self.logger.info("Pre-market scan: excluding %d already-held symbol(s): %s", len(_held), sorted(_held))

        ranked = sorted(zip(symbols, boosted_probs), key=lambda x: x[1], reverse=True)
        selected = [(sym, prob) for sym, prob in ranked if prob >= min_conf and sym not in _held][:top_n]

        swing_top5 = [(s, round(p, 3)) for s, p in ranked[:5]]
        self.logger.info(
            "Swing scan: scored %d symbols, %d above %.2f threshold | top5: %s",
            len(ranked), len(selected), min_conf,
            ", ".join(f"{s}={p}" for s, p in swing_top5),
        )
        self.logger.info(
            "Pre-market analysis complete — %d candidates: %s (proposals held until 09:50)",
            len(selected), [s for s, _ in selected],
        )

        self._swing_proposals = await self._build_proposals(selected)

        await self.log_decision(
            "SWING_PREMARKET_ANALYSIS",
            reasoning={
                "candidates": [{"symbol": s, "confidence": round(float(p), 4)} for s, p in selected],
                "total_evaluated": len(symbols),
                "send_time": "09:50 ET",
            },
        )

    def _fetch_vix_level(self) -> float:
        """Fetch current VIX level. Tries yfinance first, falls back to FRED (1-day lag).
        Returns 30.0 (conservative) if both sources fail."""
        import os
        # Primary: yfinance (real-time)
        try:
            import yfinance as yf
            import pandas as pd
            from datetime import date as _date, timedelta as _td
            # Bug fix: ^VIX returns empty with period=…; use start/end window.
            _end = _date.today() + _td(days=1)
            _start = _end - _td(days=14)
            vix = yf.download(
                "^VIX", start=_start.isoformat(), end=_end.isoformat(),
                progress=False, auto_adjust=True,
            )
            if isinstance(vix.columns, pd.MultiIndex):
                vix.columns = vix.columns.get_level_values(0)
            vix.columns = [c.lower() for c in vix.columns]
            if len(vix) > 0:
                return float(vix["close"].iloc[-1])
        except Exception:
            pass

        # Fallback: FRED VIXCLS (previous close, 1-day lag)
        try:
            import requests
            fred_key = os.getenv("FRED_API_KEY")
            if fred_key:
                r = requests.get(
                    "https://api.stlouisfed.org/fred/series/observations",
                    params={"series_id": "VIXCLS", "sort_order": "desc", "limit": 3,
                            "api_key": fred_key, "file_type": "json"},
                    timeout=5,
                )
                if r.status_code == 200:
                    obs = [o for o in r.json().get("observations", []) if o["value"] != "."]
                    if obs:
                        vix_level = float(obs[0]["value"])
                        self.logger.info("VIX from FRED fallback (1d lag): %.1f", vix_level)
                        return vix_level
        except Exception:
            pass

        self.logger.warning("VIX unavailable from all sources — assuming VIX=30 (conservative)")
        return 30.0

    def _compute_opportunity_score(self) -> tuple:
        """
        Phase 88 + 5b: Continuous 0.0–1.0 day opportunity score.
        Returns (score, vix, spy_close, spy_ma20, spy_5d_ret).

        Score components (Phase 5b adds breadth + dispersion):
          - vix_score:      drops as VIX rises (0 at VIX=35, 1 at VIX=15)
          - vix_trend:      penalty if VIX is spiking vs its 5-day avg
          - ma_score:       1.0 if SPY above MA20, 0.4 if below
          - mom_score:      SPY 5d momentum, clamped 0–1
          - breadth_score:  % of universe above MA50 from latest regime snapshot
          - dispersion_score: cross-sectional return dispersion from regime snapshot

        Weights are config-driven (opp_score_* in Settings). When breadth or
        dispersion data is unavailable, their weights are redistributed to the
        remaining components so the score always spans 0–1.

        Caller maps score → max_candidates:
          ≥ 0.65 → normal (TOP_N_INTRADAY)
          0.35–0.64 → reduced (2)
          < 0.35 → skip (0)
        """
        try:
            from app.config import settings as _s
            import yfinance as yf
            import pandas as pd
            from datetime import date as _date, timedelta as _td
            _end = _date.today() + _td(days=1)
            _start = _end - _td(days=80)  # ~40 trading days
            spy = yf.download(
                "SPY", start=_start.isoformat(), end=_end.isoformat(),
                progress=False, auto_adjust=True,
            )
            if isinstance(spy.columns, pd.MultiIndex):
                spy.columns = spy.columns.get_level_values(0)
            spy.columns = [c.lower() for c in spy.columns]
            if len(spy) < 20:
                return 1.0, None, None, None, None

            spy_close = float(spy["close"].iloc[-1])
            spy_ma20 = float(spy["close"].tail(20).mean())
            spy_5d_ret = (spy_close / float(spy["close"].iloc[-6]) - 1.0) if len(spy) >= 6 else 0.0

            vix_level = self._fetch_vix_level()

            # VIX: 1.0 at VIX=15, 0.0 at VIX=35
            vix_score = float(np.clip(1.0 - (vix_level - 15.0) / 20.0, 0.0, 1.0))

            # VIX trend: penalty if VIX spiking vs 5-day avg
            vix_5d_avg = vix_level
            try:
                from datetime import date as _date, timedelta as _td
                _end = _date.today() + _td(days=1)
                _start = _end - _td(days=20)
                vix_hist = yf.download(
                    "^VIX", start=_start.isoformat(), end=_end.isoformat(),
                    progress=False, auto_adjust=True,
                )
                if isinstance(vix_hist.columns, pd.MultiIndex):
                    vix_hist.columns = vix_hist.columns.get_level_values(0)
                vix_hist.columns = [c.lower() for c in vix_hist.columns]
                if len(vix_hist) >= 5:
                    vix_5d_avg = float(vix_hist["close"].tail(5).mean())
            except Exception:
                pass
            vix_trend = float(np.clip(1.0 - (vix_level - vix_5d_avg) / 5.0, 0.0, 1.0))

            # SPY vs MA20: not a binary block — below MA = 0.4 score (reduced, not zero)
            ma_score = 1.0 if spy_close >= spy_ma20 else 0.4

            # SPY 5d momentum: 0.5 at flat, 1.0 at +2%, 0.0 at -2%
            mom_score = float(np.clip(0.5 + spy_5d_ret * 25.0, 0.0, 1.0))

            # Phase 5b: breadth + dispersion from regime context (fail-neutral = 0.5)
            breadth_score: Optional[float] = None
            dispersion_score: Optional[float] = None
            try:
                ctx_feats = (self._current_regime_ctx or {}).get("features", {})
                raw_breadth = ctx_feats.get("breadth_pct_ma50")
                if raw_breadth is not None:
                    # breadth_pct_ma50 is stored as fraction 0–1
                    breadth_score = float(np.clip(raw_breadth, 0.0, 1.0))
                raw_disp = ctx_feats.get("dispersion_pctile")
                if raw_disp is not None:
                    dispersion_score = float(np.clip(raw_disp, 0.0, 1.0))
            except Exception:
                pass

            # Build weighted sum, redistributing missing-component weights
            w_vix = _s.opp_score_vix_weight
            w_vix_trend = _s.opp_score_vix_trend_weight
            w_ma = _s.opp_score_ma_weight
            w_mom = _s.opp_score_mom_weight
            w_breadth = _s.opp_score_breadth_weight if breadth_score is not None else 0.0
            w_disp = _s.opp_score_dispersion_weight if dispersion_score is not None else 0.0

            w_total = w_vix + w_vix_trend + w_ma + w_mom + w_breadth + w_disp
            if w_total <= 0:
                w_total = 1.0

            score = (
                w_vix * vix_score
                + w_vix_trend * vix_trend
                + w_ma * ma_score
                + w_mom * mom_score
                + w_breadth * (breadth_score or 0.5)
                + w_disp * (dispersion_score or 0.5)
            ) / w_total

            self.logger.debug(
                "Opp score=%.3f vix=%.2f vix_trend=%.2f ma=%.2f mom=%.2f "
                "breadth=%s disp=%s",
                score, vix_score, vix_trend, ma_score, mom_score,
                f"{breadth_score:.2f}" if breadth_score is not None else "N/A",
                f"{dispersion_score:.2f}" if dispersion_score is not None else "N/A",
            )
            return score, vix_level, spy_close, spy_ma20, spy_5d_ret
        except Exception as exc:
            self.logger.warning("Opportunity score check failed (%s) — failing open", exc)
            return 1.0, None, None, None, None

    def _market_regime_allows_entries(self) -> bool:
        """
        Swing PM abstention gate. Uses opportunity score < 0.35 as the block
        threshold (equivalent to old VIX>=25 OR SPY<MA20 in most regimes).
        Fails open if data unavailable.
        """
        score, vix, spy_close, spy_ma20, spy_5d_ret = self._compute_opportunity_score()
        if vix is None:
            return True
        allows = score >= 0.35
        if not allows:
            self.logger.info(
                "PM abstention gate ACTIVE — score=%.2f SPY=%.2f MA20=%.2f "
                "VIX=%.1f SPY5d=%.2f%%",
                score, spy_close, spy_ma20, vix, (spy_5d_ret or 0) * 100,
            )
        else:
            self.logger.info(
                "PM abstention gate clear — score=%.2f SPY=%.2f MA20=%.2f VIX=%.1f SPY5d=%.2f%%",
                score, spy_close, spy_ma20, vix, (spy_5d_ret or 0) * 100,
            )
        return allows

    async def _send_swing_proposals(self):
        """
        09:50 ET: Forward cached swing proposals to Risk Manager.
        Open volatility has settled; entries here get cleaner fills.
        Falls back to running a fresh analysis if 08:00 run was missed.
        """
        from app.live_trading.kill_switch import kill_switch
        if kill_switch.is_active:
            self.logger.warning("Kill switch ACTIVE — suppressing all swing proposals")
            await self.log_decision("SELECTION_SKIPPED", reasoning={"reason": "kill_switch"})
            self._swing_proposals = []
            return

        if not self._swing_proposals:
            self.logger.warning(
                "No swing proposals cached at send time — pre-market analysis did not run or "
                "was cleared. Skipping today's swing proposals. Check logs for SELECTION_SKIPPED."
            )
            return

        # Phase 88: graduated opportunity score for swing (mirrors intraday gate)
        swing_opp_score, _vix, _spy, _ma20, _spy5d = await asyncio.to_thread(self._compute_opportunity_score)
        if swing_opp_score < 0.35:
            self.logger.info(
                "Phase 88: opportunity score %.2f < 0.35 — suppressing %d swing proposals",
                swing_opp_score, len(self._swing_proposals),
            )
            await self.log_decision(
                "SWING_ABSTAINED",
                reasoning={"reason": "phase88_opportunity_score_low", "score": round(swing_opp_score, 3)},
            )
            self._swing_proposals = []
            return

        # Phase 59: Macro calendar gate — suppress entries within high-impact event window
        try:
            from app.calendars.macro import macro_calendar
            macro_ctx = macro_calendar.get_context()
            if macro_ctx.block_new_entries:
                event_names = ", ".join(e.event_type for e in macro_ctx.events_today)
                self.logger.info("Macro gate: suppressing swing proposals — within %s window", event_names)
                await self.log_decision(
                    "SWING_ABSTAINED",
                    reasoning={"reason": f"macro event window: {event_names}"},
                )
                self._swing_proposals = []
                return
            if macro_ctx.sizing_factor < 1.0:
                self.logger.info("Macro gate: high-impact day — applying sizing_factor=%.2f to swing proposals",
                                 macro_ctx.sizing_factor)
                for p in self._swing_proposals:
                    p["macro_sizing_factor"] = macro_ctx.sizing_factor
        except Exception as exc:
            self.logger.debug("Macro calendar check failed: %s", exc)

        self.logger.info("Sending %d swing proposals to Risk Manager (09:50)...", len(self._swing_proposals))
        import time as _time
        _send_ts = datetime.utcnow()
        for proposal in self._swing_proposals:
            self.send_message(TRADE_PROPOSALS_QUEUE, proposal)
            self._pending_approvals[proposal["symbol"]] = _time.monotonic()
            self.logger.info(
                "Proposal sent: %s @ $%.2f (confidence=%.2f)",
                proposal["symbol"], proposal["entry_price"], proposal["confidence"],
            )
            # Update ProposalLog: mark as SENT to RM
            try:
                from app.database.models import ProposalLog
                from app.database.session import get_session as _gss
                _sdb = _gss()
                try:
                    _sdb.query(ProposalLog).filter(
                        ProposalLog.proposal_uuid == proposal.get("proposal_uuid"),
                    ).update({"pm_status": "SENT", "sent_to_rm_at": _send_ts})
                    _sdb.commit()
                finally:
                    _sdb.close()
            except Exception:
                pass

        await self.log_decision(
            "INSTRUMENTS_SELECTED",
            reasoning={
                "selected": [
                    {"symbol": p["symbol"], "confidence": round(float(p["confidence"]), 4)}
                    for p in self._swing_proposals
                ],
                "sent_at": "09:50 ET",
            },
        )
        self._swing_proposals = []

    async def select_instruments(self):
        """Public entry point for manual/forced selection.

        Routes to the correct strategy based on time of day so that a restart
        or manual trigger mid-session doesn't fire swing proposals at 2 PM.
        - Before 09:45 ET → swing (pre-market / open)
        - 09:45–15:45 ET  → intraday (market hours)
        - After 15:45 ET  → no-op (too close to close)
        """
        now = datetime.now(ET)
        minutes = now.hour * 60 + now.minute

        if minutes < 9 * 60 + 45:
            # Pre-market / just after open — run swing
            await self._analyze_swing_premarket()
            await self._send_swing_proposals()
        elif minutes < 15 * 60 + 45:
            # Market hours — run intraday; skip if this window already ran
            win_key = (now.hour, now.minute // 15 * 15)  # bucket to 15-min window
            if win_key in self._intraday_windows_run:
                self.logger.info(
                    "select_instruments: intraday window %02d:%02d already ran — skipping duplicate",
                    win_key[0], win_key[1],
                )
                return
            self.logger.info(
                "select_instruments called at %02d:%02d ET — routing to intraday scan",
                now.hour, now.minute,
            )
            self._intraday_windows_run.add(win_key)
            await self.select_intraday_instruments()
        else:
            self.logger.info(
                "select_instruments called at %02d:%02d ET — too late in day, skipping",
                now.hour, now.minute,
            )

    # ─── Intraday Selection ───────────────────────────────────────────────────

    async def select_intraday_instruments(self, window: tuple = (9, 45), use_morning_candidates: bool = False):
        """Run intraday model on 5-min bars, send intraday proposals.

        Args:
            window: (hour, minute) scheduled scan time.
            use_morning_candidates: If True, scan only the top-N symbols cached from the
                9:45 scan instead of the full Russell 1000 universe.  Dramatically reduces
                scan time for 10:45 and 13:00 windows (and adaptive re-scans).
        """
        intraday_ver = getattr(self.intraday_model, "version", None)
        win_str = f"{window[0]:02d}:{window[1]:02d}"
        self.logger.info("Selecting intraday instruments (%s scan) — model v%s", win_str, intraday_ver)

        from app.live_trading.kill_switch import kill_switch
        if kill_switch.is_active:
            self.logger.warning("Kill switch ACTIVE — suppressing intraday scan %s", win_str)
            await self.log_decision("SELECTION_SKIPPED", reasoning={"reason": "kill_switch", "window": win_str})
            return

        if not self.intraday_model.is_trained:
            self.logger.warning("No intraday model available — skipping intraday scan")
            await self.log_decision("SELECTION_SKIPPED", reasoning={
                "reason": "model not trained",
                "strategy": "intraday",
                "model_version": intraday_ver,
            })
            return

        # Phase 88: graduated opportunity score — replaces binary VIX/SPY gate
        opp_score, _vix, _spy, _ma20, _spy5d = await asyncio.to_thread(self._compute_opportunity_score)
        if opp_score < 0.35:
            self.logger.info(
                "Phase 88: opportunity score %.2f < 0.35 — suppressing all intraday entries", opp_score,
            )
            return
        # score 0.35–0.64 → cap at 2 candidates; ≥ 0.65 → normal
        _phase88_max = TOP_N_INTRADAY if opp_score >= 0.65 else 2
        self.logger.info("Phase 88: opportunity score %.2f → max_candidates=%d", opp_score, _phase88_max)

        # Phase 59: Macro calendar gate
        try:
            from app.calendars.macro import macro_calendar
            macro_ctx = macro_calendar.get_context()
            if macro_ctx.block_new_entries:
                event_names = ", ".join(e.event_type for e in macro_ctx.events_today)
                self.logger.info("Macro gate: suppressing intraday entries — within %s window", event_names)
                return
        except Exception as exc:
            self.logger.debug("Macro calendar check failed: %s", exc)

        # Refresh regime context so sizing uses fresh data (not hours-stale premarket snapshot)
        try:
            from app.agents.premarket import premarket_intel
            self._current_regime_ctx = premarket_intel.get_regime_context()
        except Exception as _exc:
            self.logger.debug("Regime context refresh failed: %s", _exc)

        # ── SPY intraday regime gate ───────────────────────────────────────────
        # Fetch once per scan; gate entries and adjust confidence/size.
        spy_pct = await asyncio.to_thread(self._get_spy_day_pct)
        self._last_scan_spy_pct = spy_pct if spy_pct is not None else self._last_scan_spy_pct
        intraday_min_conf = MIN_CONFIDENCE  # may be raised below
        intraday_size_mult = 1.0

        if spy_pct is not None:
            if spy_pct <= SPY_HARD_STOP_PCT:
                self.logger.info(
                    "SPY down %.1f%% — hard stop: suppressing all intraday longs for %s scan",
                    spy_pct, win_str,
                )
                await self.log_decision("SELECTION_SKIPPED", reasoning={
                    "reason": "spy_hard_stop", "spy_pct": round(spy_pct, 2),
                    "strategy": "intraday", "window": win_str,
                })
                return
            elif spy_pct <= SPY_CAUTION_PCT:
                intraday_min_conf = 0.65
                intraday_size_mult = 0.5
                self.logger.info(
                    "SPY down %.1f%% — caution mode: min_conf=%.2f size=50%% for %s scan",
                    spy_pct, intraday_min_conf, win_str,
                )
            elif spy_pct >= SPY_CHASE_PCT:
                intraday_min_conf = 0.65
                self.logger.info(
                    "SPY up %.1f%% — gap caution: min_conf=%.2f for %s scan",
                    spy_pct, intraday_min_conf, win_str,
                )

        # ── Universe selection ─────────────────────────────────────────────────
        # 9:45 scan: full Russell 1000 to build morning candidates list.
        # Later scans: use cached morning candidates (top-N by score) for speed.
        if use_morning_candidates and self._morning_intraday_candidates:
            scan_universe = self._morning_intraday_candidates
            self.logger.info(
                "Using morning candidates universe (%d symbols) for %s scan",
                len(scan_universe), win_str,
            )
        else:
            scan_universe = RUSSELL_1000_TICKERS
            self.logger.info(
                "Using full universe (%d symbols) for %s scan",
                len(scan_universe), win_str,
            )

        def _fetch_intraday_features() -> tuple:
            """
            Two-phase fetch:
            1. Batch-fetch daily bars for the full universe (1 API call per 200 symbols)
               to rank by volume and pre-filter to the top 150 liquid symbols.
            2. Per-symbol 5Min + daily fetch for only those 150 symbols (~300 API calls,
               ~100s at 3 req/s — well within the 5-min timeout).
            """
            from app.ml.intraday_features import compute_intraday_features, MIN_BARS
            from concurrent.futures import ThreadPoolExecutor, as_completed
            from datetime import date as _date

            # ── Phase 1: daily batch to rank by volume, keep top 150 ─────────
            PREFILTER_N = 150
            self.logger.info(
                "Intraday fetch phase 1: daily batch for %d symbols to select top-%d by volume",
                len(scan_universe), PREFILTER_N,
            )
            daily_batch = self._alpaca.get_bars_batch(scan_universe, timeframe="1Day", limit=25)
            self.logger.info("Daily batch: %d symbols returned", len(daily_batch))

            # Rank by prior-day volume descending, keep top N
            vol_ranked = sorted(
                ((sym, float(df.iloc[-2]["volume"]) if len(df) >= 2 else 0.0)
                 for sym, df in daily_batch.items()),
                key=lambda x: x[1], reverse=True,
            )
            prefiltered = [sym for sym, _ in vol_ranked[:PREFILTER_N]]
            self.logger.info(
                "Intraday fetch phase 1 complete: %d → %d symbols after volume filter",
                len(daily_batch), len(prefiltered),
            )

            # ── Phase 2: per-symbol 5Min fetch for prefiltered set ───────────
            # Fetch SPY daily bars here (sync, inside the thread) so the closure
            # is self-contained — spy_daily_bars in the outer scope isn't assigned yet.
            try:
                _spy_state = self._get_spy_intraday_state()
                _spy_daily_bars = _spy_state.get("spy_daily_bars")
            except Exception:
                _spy_daily_bars = None

            def _fetch_one(symbol: str, _spy_daily=_spy_daily_bars) -> Optional[tuple]:
                try:
                    bars = self._alpaca.get_bars(symbol, timeframe="5Min", limit=78)
                    if bars is None or bars.empty or len(bars) < MIN_BARS:
                        return None
                    daily = daily_batch.get(symbol)
                    prior_close = prior_high = prior_low = None
                    prior_day_range = None
                    if daily is not None and len(daily) >= 2:
                        prev = daily.iloc[-2]
                        prior_close = float(prev["close"])
                        prior_high = float(prev["high"])
                        prior_low = float(prev["low"])
                        prior_day_range = prior_high - prior_low
                    feats = compute_intraday_features(
                        bars, prior_close=prior_close,
                        prior_day_high=prior_high, prior_day_low=prior_low,
                        daily_bars=daily,
                        spy_daily_bars=_spy_daily,
                        symbol=symbol,
                        as_of_date=_date.today(),
                    )
                    if feats is None:
                        return None
                    return feats, prior_day_range
                except Exception as exc:
                    self.logger.warning("Intraday feature skip %s: %s", symbol, exc)
                    return None

            feat_result: Dict[str, Dict[str, float]] = {}
            range_result: Dict[str, Optional[float]] = {}
            n_ok = n_fail = 0
            with ThreadPoolExecutor(max_workers=3) as pool:
                futures = {pool.submit(_fetch_one, s): s for s in prefiltered}
                for future in as_completed(futures, timeout=240):
                    sym = futures[future]
                    try:
                        out = future.result(timeout=30)
                    except Exception:
                        n_fail += 1
                        continue
                    if out is not None:
                        feat_result[sym] = out[0]
                        range_result[sym] = out[1]
                        n_ok += 1
                    else:
                        n_fail += 1

            self.logger.info(
                "Intraday feature fetch: %d/%d symbols computed (%d failed/skipped)",
                n_ok, len(prefiltered), n_fail,
            )
            return feat_result, range_result

        try:
            features_by_symbol, prior_ranges_by_symbol = await asyncio.wait_for(
                asyncio.to_thread(_fetch_intraday_features), timeout=300
            )
        except asyncio.TimeoutError:
            self.logger.error("Intraday feature fetch timed out after 5 minutes — skipping")
            return

        if not features_by_symbol:
            self.logger.warning("No intraday features computed")
            return

        symbols = list(features_by_symbol.keys())

        # P1 BenignGate: block all intraday signals when macro regime is adverse
        try:
            from app.strategy.benign_gate import BenignGate
            _bg = BenignGate()
            symbols = _bg.gate(symbols, reason="intraday_ml")
            if not symbols:
                self.logger.warning("BenignGate blocked all intraday signals — adverse regime")
                return
        except Exception as _bg_exc:
            self.logger.warning("BenignGate check failed (non-fatal, proceeding): %s", _bg_exc)

        # Use model's stored feature_names to guarantee correct column order and count.
        # Falls back to dict-value order if model has no feature_names (old pkl).
        model_feat_names = getattr(self.intraday_model, "feature_names", None)
        if model_feat_names:
            X = np.array([
                [features_by_symbol[s].get(f, 0.0) for f in model_feat_names]
                for s in symbols
            ])
        else:
            X = np.array([list(features_by_symbol[s].values()) for s in symbols])
        X = np.nan_to_num(X, nan=0.0)
        # Phase 3a: Branch B global features (vix_regime_level, spy_5d_return_daily,
        # day_of_week) must bypass cs_normalize — they're identical across symbols
        # on a given day and would be zeroed out by cross-sectional z-scoring.
        _model_fns = getattr(self.intraday_model, "feature_names", None) or _INTRADAY_FEATURE_NAMES
        _b2_cols = [_model_fns.index(f) for f in _INTRADAY_BRANCH_B if f in _model_fns]
        X = cs_normalize_branch_a(X, _b2_cols)

        try:
            _, probabilities = self.intraday_model.predict(X)
        except Exception as exc:
            self.logger.error("Intraday model prediction failed: %s", exc)
            return

        # Phase 51: daily intraday P&L cap — stop new entries if day loss > 1% of account
        try:
            from app.database.session import get_session
            from app.database.models import Trade
            from datetime import date
            _db = get_session()
            try:
                today_str = date.today().isoformat()
                today_intraday = _db.query(Trade).filter(
                    Trade.trade_type == "intraday",
                    Trade.status == "CLOSED",
                    Trade.closed_at >= today_str,
                ).all()
                intraday_day_pnl = sum(t.pnl or 0.0 for t in today_intraday)
            finally:
                _db.close()
            try:
                acct = self._alpaca.get_account()
                _acct_val = acct["portfolio_value"]
            except Exception:
                _acct_val = 20_000.0
            if intraday_day_pnl < -(_acct_val * INTRADAY_DAILY_LOSS_CAP_PCT):
                self.logger.info(
                    "Intraday daily loss cap hit (%.2f < -%.0f) — skipping %s scan",
                    intraday_day_pnl, _acct_val * INTRADAY_DAILY_LOSS_CAP_PCT, win_str,
                )
                return
        except Exception as exc:
            self.logger.debug("Daily intraday P&L cap check failed (non-fatal): %s", exc)

        ranked = sorted(zip(symbols, probabilities), key=lambda x: x[1], reverse=True)

        top5 = [(s, round(p, 3)) for s, p in ranked[:5]]
        n_above = sum(1 for _, p in ranked if p >= intraday_min_conf)
        self.logger.info(
            "Intraday scan %s: scored %d symbols, %d above %.2f threshold | top5: %s",
            win_str, len(ranked), n_above, intraday_min_conf,
            ", ".join(f"{s}={p}" for s, p in top5),
        )

        # Persist scan results to ProposalLog (log top-N for DB, does not cap scoring universe)
        _scan_time_utc = datetime.utcnow()
        _intraday_batch_id = f"intra_{_scan_time_utc.strftime('%Y%m%d_%H%M%S')}_{win_str.replace(':', '')}"
        _LOG_TOP_N = 50  # how many rows to write to proposal_log; scoring uses full ranked list
        try:
            from app.database.models import ProposalLog
            from app.database.session import get_session as _gs2
            _idb = _gs2()
            try:
                _intraday_ver = getattr(self.intraday_model, "version", None)
                for _rank, (_sym, _prob) in enumerate(ranked[:_LOG_TOP_N], start=1):
                    _regime = self._current_regime_ctx or {}
                    _row = ProposalLog(
                        strategy="intraday",
                        batch_id=_intraday_batch_id,
                        scan_time=_scan_time_utc,
                        scan_window=win_str,
                        symbol=_sym,
                        rank=_rank,
                        ml_score=round(float(_prob), 4),
                        confidence=round(float(_prob), 4),
                        above_threshold=(_prob >= intraday_min_conf),
                        model_version=str(_intraday_ver) if _intraday_ver else None,
                        pm_status="SCORED",
                        proposed_at=_scan_time_utc,
                        regime_score_at_scan=_regime.get("regime_score"),
                        regime_label_at_scan=_regime.get("regime_label"),
                        regime_trigger_at_scan=_regime.get("trigger"),
                    )
                    _idb.add(_row)
                _idb.commit()
            finally:
                _idb.close()
        except Exception as _loge:
            self.logger.debug("ProposalLog intraday write failed (non-fatal): %s", _loge)

        # Store top-N by score for later scans (10:45, 13:00 use this instead of full universe)
        if not use_morning_candidates:
            self._morning_intraday_candidates = [
                sym for sym, _ in ranked[:INTRADAY_AFTERNOON_CANDIDATES]
            ]
            self.logger.info(
                "Stored %d morning candidates for afternoon scans",
                len(self._morning_intraday_candidates),
            )
            # Persist to DB so candidates survive a restart mid-day
            try:
                import json
                from app.database.session import get_session as _gs
                from app.database.config_store import set_config as _sc
                _db = _gs()
                try:
                    _sc(_db, "intraday.morning_candidates", json.dumps({
                        "date": datetime.now(ET).strftime("%Y-%m-%d"),
                        "symbols": self._morning_intraday_candidates,
                    }), description="Morning intraday candidates — persisted for restart recovery")
                    _db.commit()
                finally:
                    _db.close()
            except Exception as _pe:
                self.logger.debug("Could not persist morning candidates: %s", _pe)

        # ── Phase 85: PM abstention gates ────────────────────────────────────────
        # Fetch SPY intraday state once; gates are non-fatal if data unavailable.
        spy_state = await asyncio.to_thread(self._get_spy_intraday_state)
        first_hour_range = spy_state["first_hour_range"]
        spy_5d_return = spy_state["spy_5d_return"]
        spy_5d_vol = spy_state["spy_5d_vol"]
        spy_daily_bars = spy_state["spy_daily_bars"]  # noqa: F841

        # Gate 1A: SPY first-hour range gate — abstain if market has no intraday range
        if first_hour_range is not None and first_hour_range < SPY_MIN_FIRST_HOUR_RANGE:
            _gate_reason = f"gate1a_spy_range:{first_hour_range*100:.3f}%<{SPY_MIN_FIRST_HOUR_RANGE*100:.2f}%"
            self.logger.info(
                "Phase 85 Gate 1A: SPY first-hour range %.3f%% < %.2f%% — abstaining %s scan",
                first_hour_range * 100, SPY_MIN_FIRST_HOUR_RANGE * 100, win_str,
            )
            await self.log_decision("SELECTION_SKIPPED", reasoning={
                "reason": "phase85_gate1a_first_hour_range",
                "spy_first_hour_range_pct": round(first_hour_range * 100, 3),
                "threshold_pct": SPY_MIN_FIRST_HOUR_RANGE * 100,
                "window": win_str,
            })
            try:
                from app.database.models import ProposalLog
                from app.database.session import get_session as _gs3
                _idb3 = _gs3()
                try:
                    _idb3.query(ProposalLog).filter(
                        ProposalLog.batch_id == _intraday_batch_id,
                    ).update({"scan_gate_block": _gate_reason, "pm_status": "SCAN_GATE_BLOCKED"})
                    _idb3.commit()
                finally:
                    _idb3.close()
            except Exception:
                pass
            try:
                from app.database.decision_audit import write_scan_abstention
                _spy_px = float(spy_state.get("spy_price") or spy_state.get("spy_close") or 0) or None
                write_scan_abstention(
                    gate_type="gate1a_spy_range",
                    gate_detail=_gate_reason,
                    proposal_log_batch_id=_intraday_batch_id,
                    spy_price=_spy_px,
                    spy_first_hour_range_pct=round(first_hour_range * 100, 3),
                )
            except Exception:
                pass
            return

        # Gate 1C: melt-up compression guard — catch sustained low-vol melt-up regime
        if (first_hour_range is not None and spy_5d_return is not None and spy_5d_vol is not None):
            melt_up = (
                spy_5d_return > MELTUP_5D_RETURN_MIN
                and spy_5d_vol < MELTUP_5D_VOL_MAX
                and first_hour_range < MELTUP_FIRST_HOUR_MAX
            )
            if melt_up:
                self.logger.info(
                    "Phase 85 Gate 1C: melt-up compression (5d_ret=%.2f%% vol=%.3f%% fhr=%.3f%%) — abstaining %s",
                    spy_5d_return * 100, spy_5d_vol * 100, first_hour_range * 100, win_str,
                )
                await self.log_decision("SELECTION_SKIPPED", reasoning={
                    "reason": "phase85_gate1c_meltup_compression",
                    "spy_5d_return_pct": round(spy_5d_return * 100, 2),
                    "spy_5d_vol_pct": round(spy_5d_vol * 100, 3),
                    "spy_first_hour_range_pct": round(first_hour_range * 100, 3),
                    "window": win_str,
                })
                try:
                    from app.database.models import ProposalLog
                    from app.database.session import get_session as _gs3
                    _idb3 = _gs3()
                    try:
                        _idb3.query(ProposalLog).filter(
                            ProposalLog.batch_id == _intraday_batch_id,
                        ).update({"scan_gate_block": "gate1c_meltup", "pm_status": "SCAN_GATE_BLOCKED"})
                        _idb3.commit()
                    finally:
                        _idb3.close()
                except Exception:
                    pass
                try:
                    from app.database.decision_audit import write_scan_abstention
                    _spy_px2 = float(spy_state.get("spy_price") or spy_state.get("spy_close") or 0) or None
                    write_scan_abstention(
                        gate_type="gate1c_meltup",
                        gate_detail=f"5d_ret={spy_5d_return*100:.2f}% vol={spy_5d_vol*100:.3f}% fhr={first_hour_range*100:.3f}%",
                        proposal_log_batch_id=_intraday_batch_id,
                        spy_price=_spy_px2,
                        spy_first_hour_range_pct=round(first_hour_range * 100, 3),
                    )
                except Exception:
                    pass
                return

        # Gate 1B: score-spread abstention — reduce picks if model has no strong opinion
        phase85_max_trades = _phase88_max
        scores_only = [p for _, p in ranked]
        if len(scores_only) >= 10:
            top_n_decile = max(1, len(scores_only) // 10)
            score_spread = (
                float(np.mean(sorted(scores_only, reverse=True)[:top_n_decile]))
                - float(np.median(scores_only))
            )
            if score_spread < SCORE_SPREAD_MIN:
                phase85_max_trades = 1
                self.logger.info(
                    "Phase 85 Gate 1B: score spread %.3f < %.3f — capping picks at 1 for %s",
                    score_spread, SCORE_SPREAD_MIN, win_str,
                )

        # min_conf may have been raised by SPY regime gate above (intraday_min_conf)
        min_conf = intraday_min_conf

        # Phase 51: per-symbol cooldown — skip symbols entered within last INTRADAY_COOLDOWN_HOURS
        now_mono = _time.monotonic()
        cooldown_secs = INTRADAY_COOLDOWN_HOURS * 3600
        selected = [
            (sym, prob) for sym, prob in ranked
            if prob >= min_conf
            and (now_mono - self._intraday_symbol_last_entry.get(sym, 0)) > cooldown_secs
        ][:phase85_max_trades]

        if not selected:
            self.logger.info("No intraday candidates above confidence threshold (or all on cooldown)")
            return

        from app.live_trading.kill_switch import kill_switch
        if kill_switch.is_active:
            self.logger.warning("Intraday scan: kill switch active — suppressing proposals")
            return

        self.logger.info("Intraday selected: %s", [s for s, _ in selected])

        try:
            account = self._alpaca.get_account()
            account_value = account["portfolio_value"]
        except Exception:
            account_value = 20_000.0

        for symbol, confidence in selected:
            price = self._alpaca.get_latest_price(symbol)
            if price is None or price <= 0:
                continue

            # Phase 52: intraday entry quality gates
            intra_gate_fail = self._check_intraday_entry_gates(symbol, features_by_symbol.get(symbol, {}))
            if intra_gate_fail:
                self.logger.info("INTRADAY_ENTRY_GATE_BLOCKED %s: %s", symbol, intra_gate_fail)
                try:
                    from app.database.decision_audit import write_decision
                    write_decision(symbol, "intraday", "block",
                                   model_score=float(confidence),
                                   block_reason=f"entry_gate: {str(intra_gate_fail)[:120]}",
                                   price_at_decision=price,
                                   top_features=self._top_features_for(symbol, "intraday"))
                except Exception:
                    pass
                continue

            quantity = self._calculate_quantity(
                price, account_value, trade_type="intraday", confidence=float(confidence)
            )
            # Apply SPY caution sizing multiplier (0.5× in down-market caution mode)
            if intraday_size_mult != 1.0:
                quantity = max(1, int(quantity * intraday_size_mult))
            # Regime-aware sizing: scale by regime score (RISK_OFF → 0.3×, RISK_CAUTION → 0.6×)
            _regime_mult, _regime_lbl, _regime_score = self._regime_sizing_multiplier()
            if _regime_mult != 1.0:
                quantity = max(1, int(quantity * _regime_mult))
                self.logger.info(
                    "Regime sizing intraday %s: %.1f× (%s score=%.2f)",
                    symbol, _regime_mult, _regime_lbl, _regime_score or 0,
                )
            # Phase 3d: vol-targeting sizing (overrides quantity when enabled)
            from app.config import settings as _settings
            _intra_vt_mult = 1.0
            if _settings.vol_targeting_enabled:
                _atr_norm = features_by_symbol.get(symbol, {}).get("atr_norm", 0.0)
                quantity, _intra_vt_mult = self._vol_targeting_quantity(
                    price, account_value, quantity, _atr_norm
                )
                if _intra_vt_mult != 1.0:
                    self.logger.info(
                        "Vol-targeting intraday %s: qty=%d mult=%.2f (atr_norm=%.4f)",
                        symbol, quantity, _intra_vt_mult, _atr_norm,
                    )
            # Use ATR-based stops matching the backtester: 0.4x / 0.8x prior-day range.
            # Fall back to fixed pct if prior range unavailable.
            prior_range = prior_ranges_by_symbol.get(symbol)
            if prior_range and prior_range > 0:
                stop_dist = 0.4 * prior_range
                target_dist = 0.8 * prior_range
                stop_price = round(price - stop_dist, 2)
                target_price = round(price + target_dist, 2)
            else:
                stop_price = round(price * 0.997, 2)   # ~0.3% fallback
                target_price = round(price * 1.006, 2)  # ~0.6% fallback (~2:1)
            proposal: Dict[str, Any] = {
                "symbol": symbol,
                "direction": "BUY",
                "quantity": quantity,
                "entry_price": price,
                "confidence": float(confidence),
                "stop_loss": stop_price,
                "profit_target": target_price,
                "source_agent": "portfolio_manager",
                "trade_type": "intraday",
                "proposal_uuid": str(uuid.uuid4()),
            }
            # NIS overlay: block or resize intraday proposal based on live news signal
            try:
                from app.news.intelligence_service import nis
                from app.agents.premarket import premarket_intel
                _macro_ctx = premarket_intel.macro_context
                _sector = self._get_symbol_sector(symbol)
                _news_sig = await asyncio.to_thread(
                    nis.get_stock_signal, symbol, _sector, 4, _macro_ctx
                )
                if _news_sig.action_policy == "block_entry":
                    self.logger.info(
                        "NIS block_entry intraday %s: %s", symbol, _news_sig.rationale
                    )
                    try:
                        from app.database.decision_audit import write_decision
                        write_decision(symbol, "intraday", "block",
                                       model_score=float(confidence),
                                       block_reason=f"nis_block_entry: {_news_sig.rationale[:120]}",
                                       news_signal=_news_sig, macro_context=_macro_ctx,
                                       price_at_decision=price,
                                       top_features=self._top_features_for(symbol, "intraday"))
                    except Exception:
                        pass
                    continue
                if _news_sig.sizing_multiplier != 1.0:
                    old_qty = proposal["quantity"]
                    proposal["quantity"] = max(1, int(old_qty * _news_sig.sizing_multiplier))
                    self.logger.info(
                        "NIS sizing intraday %s: %.2f× (%d→%d) — %s",
                        symbol, _news_sig.sizing_multiplier, old_qty,
                        proposal["quantity"], _news_sig.rationale,
                    )
                proposal["news_signal"] = {
                    "action_policy": _news_sig.action_policy,
                    "direction_score": _news_sig.direction_score,
                    "materiality_score": _news_sig.materiality_score,
                    "sizing_multiplier": _news_sig.sizing_multiplier,
                    "rationale": _news_sig.rationale,
                }
            except Exception as _exc:
                self.logger.debug("NIS intraday overlay failed for %s (non-fatal): %s", symbol, _exc)

            self.send_message(TRADE_PROPOSALS_QUEUE, proposal)
            self._pending_approvals[proposal["symbol"]] = _time.monotonic()
            self._intraday_symbol_last_entry[symbol] = _time.monotonic()  # Phase 51 cooldown
            news_monitor.watch(symbol)  # Phase 53: start monitoring for news exits
            self.logger.info(
                "Intraday proposal: %s @ $%.2f (confidence=%.2f)",
                symbol, price, confidence,
            )
            try:
                from app.database.models import ProposalLog
                from app.database.session import get_session as _gs4
                _idb4 = _gs4()
                try:
                    _idb4.query(ProposalLog).filter(
                        ProposalLog.batch_id == _intraday_batch_id,
                        ProposalLog.symbol == symbol,
                    ).update({
                        "pm_status": "SENT",
                        "pm_decided_at": datetime.utcnow(),
                        "sent_to_rm_at": datetime.utcnow(),
                        "proposal_uuid": proposal.get("proposal_uuid"),
                        "direction": proposal.get("direction", "BUY"),
                        "entry_price": price,
                        "stop_price": proposal.get("stop_loss"),
                        "target_price": proposal.get("profit_target"),
                        "quantity": proposal.get("quantity"),
                        "nis_signal": proposal.get("news_signal"),
                    })
                    _idb4.commit()
                finally:
                    _idb4.close()
            except Exception:
                pass
            try:
                from app.database.decision_audit import write_decision
                write_decision(
                    symbol, "intraday", "enter",
                    model_score=float(confidence),
                    top_features=self._top_features_for(symbol, "intraday"),
                    regime_sizing_mult=_regime_mult,
                    regime_label=_regime_lbl,
                    regime_score=_regime_score,
                    vol_targeting_mult=_intra_vt_mult if _intra_vt_mult != 1.0 else None,
                )
            except Exception:
                pass

        await self.log_decision(
            "INTRADAY_INSTRUMENTS_SELECTED",
            reasoning={"selected": [{"symbol": s, "confidence": round(float(p), 4)}
                                    for s, p in selected]},
        )

    # ─── 30-Minute Position Review ────────────────────────────────────────────

    async def _review_open_positions(self) -> None:
        """
        Re-score all open swing positions using fresh bars + current model.
        Sends EXIT to trader_exit_requests if:
          - Score drops below EXIT_THRESHOLD
          - Earnings within EARNINGS_EXIT_DAYS days
          - Score has increased significantly → extend target
        Also scans universe for new entry opportunities if budget allows.
        """
        from app.database.session import get_session
        from app.database.models import Trade

        db = get_session()
        try:
            active_trades = (
                db.query(Trade)
                .filter(Trade.status == "ACTIVE", Trade.direction.in_(["BUY", "SELL_SHORT"]))
                .all()
            )
        finally:
            db.close()

        swing_trades = [t for t in active_trades if getattr(t, "signal_type", "") != "intraday"]
        if not swing_trades:
            # Still scan for new opportunities even with no open positions
            await self._scan_new_opportunities()
            return

        if not self.model.is_trained:
            return

        self.logger.info("30-min review: re-scoring %d open position(s)", len(swing_trades))

        def _score_positions():
            results = {}
            for trade in swing_trades:
                try:
                    bars = self._alpaca.get_bars(trade.symbol, timeframe="1D", limit=300)
                    if bars is None or bars.empty:
                        continue
                    feats = self.feature_engineer.engineer_features(
                        trade.symbol, bars, fetch_fundamentals=False
                    )
                    if feats is None:
                        continue
                    model_feature_names = getattr(self.model, "feature_names", None)
                    if model_feature_names:
                        x = [[feats.get(f, 0.0) for f in model_feature_names]]
                    else:
                        x = [list(feats.values())]
                    import numpy as np
                    x = np.nan_to_num(x)
                    x = self._normalize_for_inference(x, [trade.symbol], self.model)
                    # Safety net: if normalization produced all-zeros (e.g. bad data),
                    # refuse to predict rather than returning ~0 and triggering an exit.
                    if x.size > 0 and np.all(x == 0):
                        self.logger.warning(
                            "Re-score %s: feature vector all-zeros after normalization — skipping, holding position.",
                            trade.symbol,
                        )
                        continue
                    _, probs = self.model.predict(x)
                    results[trade.symbol] = {
                        "score": float(probs[0]),
                        "trade_id": trade.id,
                        "entry_price": float(trade.entry_price or 0),
                        "target_price": float(trade.target_price or 0),
                        "atr": (
                            abs(float(trade.target_price or 0) - float(trade.entry_price or 0))
                            if trade.target_price and trade.entry_price else 0.0
                        ),
                    }
                except Exception as exc:
                    self.logger.debug("Re-score failed for %s: %s", trade.symbol, exc)
            return results

        scores = await asyncio.to_thread(_score_positions)

        # Safety: if the majority of positions scored suspiciously low simultaneously,
        # treat this as a scoring failure rather than genuine degradation.
        # Genuine degradation would rarely affect every open position at once.
        _n_positions = len(swing_trades)
        _n_scored = len(scores)
        _n_near_zero = sum(1 for info in scores.values() if info["score"] < 0.05)
        if _n_scored > 0 and _n_near_zero >= max(1, _n_scored * 0.5):
            self.logger.warning(
                "SCORING ANOMALY: %d/%d positions scored near-zero (< 0.05) in same cycle. "
                "Likely a scorer failure, not genuine degradation. Aborting position review "
                "to avoid false mass-exit. Will retry in next review cycle.",
                _n_near_zero, _n_scored,
            )
            return

        # Safety: if no positions were scored at all (all failed), hold everything.
        if not scores and _n_positions > 0:
            self.logger.warning(
                "Re-scoring returned no results for %d open position(s). "
                "Holding all positions — scorer may be failing.",
                _n_positions,
            )
            return

        exit_threshold = EXIT_THRESHOLD
        try:
            from app.database.session import get_session as _gs
            from app.database.agent_config import get_agent_config
            _db = _gs()
            try:
                exit_threshold = get_agent_config(_db, "pm.exit_threshold") or EXIT_THRESHOLD
            finally:
                _db.close()
        except Exception:
            pass

        # Build direction map from the queried trades for score interpretation below
        _trade_direction = {t.symbol: (getattr(t, "direction", "BUY") or "BUY") for t in swing_trades}

        from datetime import datetime as _dt
        for symbol, info in scores.items():
            score = info["score"]
            _is_short_pos = _trade_direction.get(symbol) == "SELL_SHORT"
            # For short positions the ML score is a long-bullish signal, so invert:
            # high score → short thesis is failing → EXIT; low score → short working → hold.
            if _is_short_pos:
                score = 1.0 - score

            # Check if this score is a suspicious sudden collapse vs. last known good.
            # A genuine degradation rarely drops from >0.55 to <0.05 in one cycle.
            _last = self._last_good_score.get(symbol)
            _suspicious = score < 0.05 and _last is not None and _last[0] >= 0.45
            if _suspicious:
                self.logger.warning(
                    "SCORE ANOMALY %s: score collapsed %.2f → %.2f in one cycle "
                    "(last good score %.2f at %s). Holding position, not exiting. "
                    "Will re-check next cycle.",
                    symbol, _last[0], score, _last[0], _last[1].strftime("%H:%M"),
                )
                continue  # skip exit decision for this symbol this cycle

            # Update last-known-good if score looks valid (above noise floor)
            if score >= 0.10:
                self._last_good_score[symbol] = (score, _dt.now())

            action = None
            reason = None

            # Check earnings proximity first (overrides score)
            try:
                from app.strategy.earnings_filter import days_until_earnings
                days = days_until_earnings(symbol)
                if days is not None and 0 < days <= EARNINGS_EXIT_DAYS:
                    action = "EXIT"
                    reason = f"earnings_in_{days}d"
            except Exception:
                pass

            # Phase 72: NIS Tier 2 re-check for held swing positions
            if action is None:
                try:
                    from app.news.intelligence_service import nis
                    from app.agents.premarket import premarket_intel
                    _macro_ctx = premarket_intel.macro_context
                    _sector = self._get_symbol_sector(symbol)
                    _news_sig = await asyncio.to_thread(
                        nis.get_stock_signal, symbol, _sector, 24, _macro_ctx
                    )
                    if _news_sig.action_policy in ("exit_review", "block_entry"):
                        action = "EXIT"
                        reason = f"nis_{_news_sig.action_policy}: {_news_sig.rationale[:80]}"
                        self.logger.info(
                            "NIS exit_review %s (held position): %s",
                            symbol, _news_sig.rationale,
                        )
                except Exception as _exc:
                    self.logger.debug("NIS re-check failed for %s (non-fatal): %s", symbol, _exc)

            # Phase 53: exit flag on negative intraday news
            if action is None and news_monitor.has_negative_news(symbol):
                article = news_monitor.get_latest_article(symbol)
                title = (article or {}).get("title", "?")[:60] if article else "?"
                action = "EXIT"
                reason = f"negative_news: {title}"
                self.logger.warning(
                    "NEWS EXIT FLAG %s: negative article in last 30min — '%s'",
                    symbol, title,
                )

            if action is None:
                if score < exit_threshold:
                    action = "EXIT"
                    reason = f"score_degraded_{score:.2f}"
                elif score > MIN_CONFIDENCE + 0.10 and info.get("atr", 0) > 0:
                    # Score improved — extend target by 0.5 ATR
                    action = "EXTEND_TARGET"
                    reason = f"score_improved_{score:.2f}"

            if action:
                msg = {
                    "symbol": symbol,
                    "action": action,
                    "reason": reason,
                    "score": score,
                }
                if action == "EXTEND_TARGET":
                    msg["extend_atr"] = round(info["atr"] * 0.5, 4)
                self.send_message(EXIT_REQUESTS_QUEUE, msg)
                self.logger.info(
                    "Review → %s %s (score=%.3f reason=%s)",
                    action, symbol, score, reason,
                )

        await self.log_decision(
            "POSITION_REVIEW",
            reasoning={
                "reviewed": list(scores.keys()),
                "actions": {
                    sym: {"score": round(info["score"], 3)}
                    for sym, info in scores.items()
                },
            },
        )

        # Phase 22: record PM cycle telemetry
        try:
            from app.agents.performance_monitor import performance_monitor
            held_scores = [info["score"] for info in scores.values()]
            cycle_decisions: Dict[str, int] = {"EXIT": 0, "HOLD": 0, "EXTEND_TARGET": 0}
            for _, info in scores.items():
                sc = info["score"]
                if sc < exit_threshold:
                    cycle_decisions["EXIT"] += 1
                elif sc > MIN_CONFIDENCE + 0.10:
                    cycle_decisions["EXTEND_TARGET"] += 1
                else:
                    cycle_decisions["HOLD"] += 1
            performance_monitor.record_pm_cycle(held_scores, cycle_decisions)
        except Exception:
            pass

        # After reviewing existing positions, look for new opportunities
        await self._scan_new_opportunities()

    async def _scan_new_opportunities(self) -> None:
        """
        Check if account has budget for new swing entries.
        If so, re-score the universe and send top candidates through RM.
        Called from _review_open_positions every 30 minutes.
        """
        if not self.model.is_trained:
            return
        try:
            account = self._alpaca.get_account()
            account_value = float(account.get("portfolio_value", 0))
        except Exception:
            return

        deployed = self._get_deployed_by_type()
        gross_pct = deployed["total"] / max(account_value, 1)
        if gross_pct >= 0.75:  # stay below 80% hard cap — leave 5% buffer
            return

        # Respect the same macro/regime gates that the PM uses at 09:50 send time.
        # No point scanning and sending proposals that the trader will immediately discard.
        try:
            from app.agents.premarket import premarket_intel
            if premarket_intel.is_swing_blocked():
                self.logger.info(
                    "30-min scan skipped — swing macro gate active (FOMC/SPY drawdown)"
                )
                return
        except Exception:
            pass
        _30min_opp, _, _, _, _ = await asyncio.to_thread(self._compute_opportunity_score)
        if _30min_opp < 0.35:
            self.logger.info("30-min scan skipped — opportunity score %.2f < 0.35", _30min_opp)
            return

        self.logger.info("30-min review: budget available (%.0f%% deployed) — scanning for new entries", gross_pct * 100)

        features_by_symbol = await asyncio.to_thread(self._fetch_swing_features)
        if not features_by_symbol:
            return

        # Exclude already-held symbols
        from app.database.session import get_session
        from app.database.models import Trade
        db = get_session()
        try:
            held = {t.symbol for t in db.query(Trade).filter_by(status="ACTIVE").all()}
        finally:
            db.close()

        symbols = [s for s in features_by_symbol if s not in held]
        if not symbols:
            return

        model_feature_names = getattr(self.model, "feature_names", None)
        import numpy as np
        if model_feature_names:
            X = np.array([[features_by_symbol[s].get(f, 0.0) for f in model_feature_names] for s in symbols])
        else:
            X = np.array([list(features_by_symbol[s].values()) for s in symbols])
        X = np.nan_to_num(X)
        X = self._normalize_for_inference(X, symbols, self.model)

        try:
            _, probabilities = self.model.predict(X)
        except Exception as exc:
            self.logger.debug("New opportunity scan prediction failed: %s", exc)
            return

        ranked = sorted(zip(symbols, probabilities), key=lambda x: x[1], reverse=True)
        candidates = [(s, p) for s, p in ranked if float(p) >= MIN_CONFIDENCE][:TOP_N_STOCKS]

        if not candidates:
            return

        self.logger.info("30-min scan found %d new candidate(s): %s", len(candidates), [s for s, _ in candidates])
        proposals = await self._build_proposals(candidates)
        for proposal in proposals:
            self.send_message(TRADE_PROPOSALS_QUEUE, proposal)

    # ─── Reeval Request Handler ───────────────────────────────────────────────

    async def _handle_reeval_requests(self) -> None:
        """
        Drain pm_reeval_requests queue (Trader → PM).
        Re-score each symbol and respond via trader_exit_requests with
        EXIT, HOLD, or EXTEND_TARGET.
        """
        if not self.model.is_trained:
            return

        requests_processed = 0
        while True:
            msg = await asyncio.to_thread(self.get_message, REEVAL_REQUESTS_QUEUE, 1)
            if msg is None:
                break
            symbol = msg.get("symbol")
            reason = msg.get("reason", "trader_request")
            trade_type = msg.get("trade_type", "swing")
            if not symbol:
                continue

            # Intraday positions: don't re-score with swing model — just HOLD
            if trade_type == "intraday":
                self.send_message(EXIT_REQUESTS_QUEUE, {"symbol": symbol, "action": "HOLD", "reason": "intraday_not_rescored"})
                continue

            try:
                def _rescore():
                    bars = self._alpaca.get_bars(symbol, timeframe="1D", limit=300)
                    if bars is None or bars.empty:
                        return None
                    feats = self.feature_engineer.engineer_features(symbol, bars, fetch_fundamentals=False)
                    if feats is None:
                        return None
                    model_feature_names = getattr(self.model, "feature_names", None)
                    if model_feature_names:
                        x = [[feats.get(f, 0.0) for f in model_feature_names]]
                    else:
                        x = [list(feats.values())]
                    import numpy as np
                    x = np.nan_to_num(x)
                    x = self._normalize_for_inference(x, [symbol], self.model)
                    _, probs = self.model.predict(x)
                    return float(probs[0])

                score = await asyncio.to_thread(_rescore)

                if score is None:
                    action = "HOLD"
                    resp_reason = "rescore_failed"
                elif score < EXIT_THRESHOLD:
                    action = "EXIT"
                    resp_reason = f"rescore_{score:.2f}_below_threshold"
                else:
                    action = "HOLD"
                    resp_reason = f"rescore_{score:.2f}_ok"

                self.send_message(EXIT_REQUESTS_QUEUE, {
                    "symbol": symbol,
                    "action": action,
                    "reason": resp_reason,
                    "score": score,
                    "original_reason": reason,
                })
                self.logger.info(
                    "Reeval %s → %s (score=%.3f reason=%s)",
                    symbol, action, score or 0.0, reason,
                )
                requests_processed += 1

            except Exception as exc:
                self.logger.error("Reeval failed for %s: %s", symbol, exc)
                # Default to HOLD on error — don't exit just because we couldn't score
                self.send_message(EXIT_REQUESTS_QUEUE, {"symbol": symbol, "action": "HOLD", "reason": "reeval_error"})

        if requests_processed:
            self.logger.info("Processed %d reeval request(s)", requests_processed)

    # ─── Phase 70: Re-score pending approvals ────────────────────────────────

    async def _rescore_pending_approvals(self) -> None:
        """
        Every 30 minutes, re-score symbols that were approved by RM but not yet
        entered by Trader. If fresh ML score has dropped significantly, send a
        WITHDRAW message to cancel the pending entry.

        Thresholds:
          - Swing: withdraw if score < MIN_CONFIDENCE * 0.85 (15% grace on original threshold)
          - All: withdraw if proposal is > 30 minutes old (aligned with Trader TTL)
        """
        import time as _time
        if not self._pending_approvals:
            return
        if not self.model.is_trained:
            return

        now_mono = _time.monotonic()
        MIN_SCORE = 0.45  # fallback if can't read config

        try:
            from app.database.session import get_session as _gs
            from app.database.agent_config import get_agent_config
            _db = _gs()
            try:
                cfg_min = get_agent_config(_db, "strategy.min_confidence")
                if cfg_min:
                    MIN_SCORE = float(cfg_min) * 0.85
            finally:
                _db.close()
        except Exception:
            pass

        to_withdraw = []
        for symbol, sent_at in list(self._pending_approvals.items()):
            age_min = (now_mono - sent_at) / 60.0

            # Stale: over 30 min — aligned with Trader TTL (also 30 min)
            if age_min > INTRADAY_PROPOSAL_STALE_MIN:
                to_withdraw.append((symbol, f"stale_{age_min:.0f}min"))
                continue

            # Re-score with fresh daily bars
            try:
                def _rescore_sym(sym=symbol):
                    bars = self._alpaca.get_bars(sym, timeframe="1D", limit=300)
                    if bars is None or bars.empty:
                        return None
                    feats = self.feature_engineer.engineer_features(sym, bars, fetch_fundamentals=False)
                    if feats is None:
                        return None
                    fn = getattr(self.model, "feature_names", None)
                    import numpy as np
                    x = [[feats.get(f, 0.0) for f in fn]] if fn else [list(feats.values())]
                    x = np.nan_to_num(x)
                    x = self._normalize_for_inference(x, [sym], self.model)
                    _, probs = self.model.predict(x)
                    return float(probs[0])

                score = await asyncio.to_thread(_rescore_sym)
                if score is not None and score < MIN_SCORE:
                    to_withdraw.append((symbol, f"rescore_{score:.3f}_below_{MIN_SCORE:.3f}"))
                else:
                    self.logger.debug("Phase 70: %s still valid (score=%.3f age=%.0fmin)", symbol, score or 0.0, age_min)
            except Exception as exc:
                self.logger.debug("Phase 70 rescore failed for %s: %s", symbol, exc)

        for symbol, reason in to_withdraw:
            self._pending_approvals.pop(symbol, None)
            self.send_message(EXIT_REQUESTS_QUEUE, {
                "symbol": symbol,
                "action": "WITHDRAW",
                "reason": reason,
            })
            self.logger.info("Phase 70: withdrawing pending approval for %s — %s", symbol, reason)
            await self.log_decision("PROPOSAL_WITHDRAWN", reasoning={"symbol": symbol, "reason": reason})

    # ─── Proposal Building ────────────────────────────────────────────────────

    def _check_swing_entry_gates(self, symbol: str, price: float) -> Optional[str]:
        """
        Phase 52 swing entry gates (applied at 09:50 send time).
        Returns a rejection reason string, or None if gates pass.
        """
        # Gap-chase filter: skip if price is >2% above prior-day close
        try:
            daily = self._alpaca.get_bars(symbol, timeframe="1Day", limit=2)
            if daily is not None and len(daily) >= 2:
                prior_close = float(daily.iloc[-2]["close"])
                gap_pct = (price - prior_close) / prior_close
                if gap_pct > 0.02:
                    return f"gap_chase: price {gap_pct:.1%} above prior close (limit 2%)"
        except Exception:
            pass  # data unavailable — fail open

        # Illiquidity guard: skip if bid/ask spread > 0.5% of mid.
        # Spreads > 2% are treated as stale IEX quotes (not real spreads) and ignored.
        try:
            quote = self._alpaca.get_quote(symbol)
            if quote is not None:
                sp = quote["spread_pct"]
                if sp <= 0.02 and sp > 0.005:
                    return f"spread_too_wide: {sp:.3%} (limit 0.5%)"
        except Exception:
            pass

        return None

    def _check_intraday_entry_gates(self, symbol: str, feats: Dict[str, float]) -> Optional[str]:
        """
        Phase 52 intraday entry gates (applied at proposal send time).
        Uses pre-computed features where possible; falls back to a live quote for spread.
        Returns a rejection reason string, or None if gates pass.
        """
        # Momentum confirmation: last bar must be a green (up) bar — consecutive_bars > 0
        consec = feats.get("consecutive_bars", None)
        if consec is not None and consec <= 0:
            return f"no_momentum: consecutive_bars={consec:.0f} (need >0)"

        # Volume confirmation: last bar volume must exceed 20-bar average — volume_surge >= 1.0
        vol_surge = feats.get("volume_surge", None)
        if vol_surge is not None and vol_surge < 1.0:
            return f"low_volume: volume_surge={vol_surge:.2f} (need >=1.0)"

        # Spread guard: skip if bid/ask spread > 0.3% of mid.
        # Spreads > 2% are treated as stale IEX quotes and ignored.
        try:
            quote = self._alpaca.get_quote(symbol)
            if quote is not None:
                sp = quote["spread_pct"]
                if sp <= 0.02 and sp > 0.003:
                    return f"spread_too_wide: {sp:.3%} (limit 0.3%)"
        except Exception:
            pass

        # Phase 53: block entry if negative news in last 30 min
        if news_monitor.has_negative_news(symbol):
            article = news_monitor.get_latest_article(symbol)
            title = (article or {}).get("title", "?")[:60] if article else "?"
            return f"negative_news: '{title}'"

    def _top_features_for(self, symbol: str, strategy: str = "swing", top_n: int = 8) -> Optional[dict]:
        """
        Return the top-N most important feature values for symbol at decision time.
        Uses model's global feature_importances_ to rank, then returns actual values.
        Fails silently — audit trail never blocks trading.
        """
        try:
            cache = self._last_swing_features if strategy == "swing" else self._last_intraday_features
            feat_vals = cache.get(symbol)
            if not feat_vals:
                return None
            model = self.model if strategy == "swing" else self.intraday_model
            feature_names = getattr(model, "feature_names", None)
            inner = getattr(model, "model", None)
            importances = getattr(inner, "feature_importances_", None) if inner else None
            if feature_names and importances is not None and len(feature_names) == len(importances):
                ranked = sorted(
                    zip(feature_names, importances), key=lambda x: x[1], reverse=True
                )
                return {
                    f: round(float(feat_vals.get(f, 0.0)), 6)
                    for f, _ in ranked[:top_n]
                    if f in feat_vals
                }
            # Fallback: just return the top-N keys by absolute value
            return dict(
                sorted(feat_vals.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]
            )
        except Exception:
            return None

        return None

    async def _build_proposals(self, selected: List[tuple]) -> List[Dict[str, Any]]:
        """Build trade proposal dicts for the Risk Manager."""
        try:
            account = self._alpaca.get_account()
            account_value = account["portfolio_value"]
        except Exception:
            account_value = 20_000.0  # fallback

        proposals = []
        for symbol, confidence in selected:
            price = self._alpaca.get_latest_price(symbol)
            if price is None or price <= 0:
                continue

            # Phase 52: swing entry quality gates
            gate_fail = self._check_swing_entry_gates(symbol, price)
            if gate_fail:
                self.logger.info("SWING_ENTRY_GATE_BLOCKED %s: %s", symbol, gate_fail)
                try:
                    from app.database.decision_audit import write_decision
                    write_decision(symbol, "swing", "block",
                                   model_score=float(confidence),
                                   block_reason=f"entry_gate: {str(gate_fail)[:120]}",
                                   price_at_decision=price,
                                   top_features=self._top_features_for(symbol, "swing"))
                except Exception:
                    pass
                continue

            quantity = self._calculate_quantity(
                price, account_value, trade_type="swing", confidence=float(confidence)
            )
            # Regime-aware sizing: scale by regime score (RISK_OFF → 0.3×, RISK_CAUTION → 0.6×)
            _regime_mult, _regime_lbl, _regime_score = self._regime_sizing_multiplier()
            if _regime_mult != 1.0:
                quantity = max(1, int(quantity * _regime_mult))
                self.logger.info(
                    "Regime sizing swing %s: %.1f× (%s score=%.2f)",
                    symbol, _regime_mult, _regime_lbl, _regime_score or 0,
                )
            # Phase 3d: vol-targeting sizing (overrides quantity when enabled)
            from app.config import settings as _cfg
            _swing_vt_mult = 1.0
            if _cfg.vol_targeting_enabled:
                _atr_norm_sw = (self._last_swing_features or {}).get(symbol, {}).get("atr_norm", 0.0)
                quantity, _swing_vt_mult = self._vol_targeting_quantity(
                    price, account_value, quantity, _atr_norm_sw
                )
                if _swing_vt_mult != 1.0:
                    self.logger.info(
                        "Vol-targeting swing %s: qty=%d mult=%.2f (atr_norm=%.4f)",
                        symbol, quantity, _swing_vt_mult, _atr_norm_sw,
                    )
            proposal: Dict[str, Any] = {
                "symbol": symbol,
                "direction": "BUY",
                "quantity": quantity,
                "entry_price": price,
                "confidence": float(confidence),
                "stop_loss": round(price * 0.98, 2),
                "profit_target": round(price * 1.05, 2),
                "source_agent": "portfolio_manager",
                "trade_type": "swing",
                "proposal_uuid": str(uuid.uuid4()),
            }
            # NIS Tier 2: apply news signal overlay (non-blocking)
            news_sig = None
            macro_ctx = None
            try:
                from app.news.intelligence_service import nis
                from app.agents.premarket import premarket_intel
                macro_ctx = premarket_intel.macro_context
                sector = self._get_symbol_sector(symbol)
                news_sig = await asyncio.to_thread(
                    nis.get_stock_signal, symbol, sector, 24, macro_ctx
                )
                if news_sig.action_policy == "block_entry":
                    self.logger.info(
                        "NIS block_entry %s: %s", symbol, news_sig.rationale
                    )
                    try:
                        from app.database.decision_audit import write_decision
                        write_decision(symbol, "swing", "block",
                                       model_score=float(confidence),
                                       block_reason=f"nis_block_entry: {news_sig.rationale[:120]}",
                                       news_signal=news_sig, macro_context=macro_ctx,
                                       price_at_decision=price,
                                       top_features=self._top_features_for(symbol, "swing"))
                    except Exception:
                        pass
                    continue
                if news_sig.sizing_multiplier != 1.0:
                    old_qty = proposal["quantity"]
                    proposal["quantity"] = max(1, int(old_qty * news_sig.sizing_multiplier))
                    self.logger.info(
                        "NIS sizing %s: %.2f× (%d→%d) — %s",
                        symbol, news_sig.sizing_multiplier, old_qty,
                        proposal["quantity"], news_sig.rationale,
                    )
                proposal["news_signal"] = {
                    "action_policy": news_sig.action_policy,
                    "direction_score": news_sig.direction_score,
                    "materiality_score": news_sig.materiality_score,
                    "sizing_multiplier": news_sig.sizing_multiplier,
                    "rationale": news_sig.rationale,
                }
                # Add news_score_overlay to ML confidence
                overlay = news_sig.news_score_overlay()
                if overlay != 0.0:
                    proposal["confidence"] = round(
                        float(min(1.0, max(0.0, float(confidence) + overlay))), 4
                    )
            except Exception as exc:
                self.logger.debug("NIS Tier 2 overlay failed for %s (non-fatal): %s", symbol, exc)

            # Optional AI signal review (non-blocking)
            try:
                from app.ai.claude_client import review_pm_signal
                from app.strategy.regime_detector import regime_detector
                regime = await asyncio.to_thread(regime_detector.get_regime)
                ai_summary = review_pm_signal(
                    symbol=symbol,
                    signal_type="ML_SELECTION",
                    confidence=float(confidence),
                    reasoning={"price": price, "stop": proposal["stop_loss"],
                               "target": proposal["profit_target"]},
                    regime=regime,
                )
                if ai_summary:
                    proposal["ai_review"] = ai_summary
            except Exception:
                pass

            # Phase 61: Decision audit — record every enter decision
            try:
                from app.database.decision_audit import write_decision
                _rm, _rl, _rs = self._regime_sizing_multiplier()
                write_decision(
                    symbol, "swing", "enter",
                    model_score=float(confidence),
                    size_multiplier=round(float(news_sig.sizing_multiplier if news_sig else 1.0), 4),
                    news_signal=news_sig,
                    macro_context=macro_ctx,
                    top_features=self._top_features_for(symbol, "swing"),
                    regime_sizing_mult=_rm,
                    regime_label=_rl,
                    regime_score=_rs,
                    vol_targeting_mult=_swing_vt_mult if _swing_vt_mult != 1.0 else None,
                )
            except Exception:
                pass

            # ProposalLog: persist swing proposal as PENDING (status updated to SENT in _send_swing_proposals)
            try:
                from app.database.models import ProposalLog
                from app.database.session import get_session as _gsp
                _pdb = _gsp()
                try:
                    _swing_ver = getattr(self.model, "version", None)
                    _regime = self._current_regime_ctx or {}
                    _pl_row = ProposalLog(
                        proposal_uuid=proposal["proposal_uuid"],
                        strategy="swing",
                        scan_time=datetime.utcnow(),
                        symbol=symbol,
                        ml_score=round(float(confidence), 4),
                        confidence=round(float(proposal.get("confidence", confidence)), 4),
                        above_threshold=True,
                        model_version=str(_swing_ver) if _swing_ver else None,
                        top_features=self._top_features_for(symbol, "swing"),
                        nis_signal=proposal.get("news_signal"),
                        pm_status="PENDING",
                        pm_decided_at=datetime.utcnow(),
                        direction=proposal.get("direction", "BUY"),
                        entry_price=proposal.get("entry_price"),
                        stop_price=proposal.get("stop_loss"),
                        target_price=proposal.get("profit_target"),
                        quantity=proposal.get("quantity"),
                        proposed_at=datetime.utcnow(),
                        regime_score_at_scan=_regime.get("regime_score"),
                        regime_label_at_scan=_regime.get("regime_label"),
                        regime_trigger_at_scan=_regime.get("trigger"),
                    )
                    _pdb.add(_pl_row)
                    _pdb.commit()
                finally:
                    _pdb.close()
            except Exception as _ple:
                self.logger.debug("ProposalLog swing write failed (non-fatal): %s", _ple)

            proposals.append(proposal)

        return proposals

    def _get_symbol_sector(self, symbol: str) -> str:
        """Return sector for a symbol from WatchlistTicker, or 'Unknown'."""
        try:
            from app.database.session import get_session
            from app.database.models import WatchlistTicker
            db = get_session()
            try:
                row = db.query(WatchlistTicker).filter_by(symbol=symbol).first()
                return row.sector or "Unknown" if row else "Unknown"
            finally:
                db.close()
        except Exception:
            return "Unknown"

    def _fetch_target_upside(self, symbols: List[str]) -> Dict[str, float]:
        """
        Return FMP price-target upside ratio for each symbol.
        upside = (targetConsensus - current_price) / current_price
        Positive = stock trading below consensus, negative = above.
        Returns {} on any failure — boost is optional, never blocks selection.
        """
        result: Dict[str, float] = {}
        try:
            import requests
            from app.config import settings
            key = settings.fmp_api_key
            if not key:
                return result
            # Fetch current prices via Alpaca snapshot
            prices: Dict[str, float] = {}
            for sym in symbols:
                try:
                    bars = self._alpaca.get_bars(sym, timeframe="1D", limit=1)
                    if not bars.empty:
                        prices[sym] = float(bars["close"].iloc[-1])
                except Exception:
                    pass
            # Fetch consensus targets (one request per symbol — cached 24h by FMP)
            base = "https://financialmodelingprep.com/stable"
            for sym in symbols:
                price = prices.get(sym)
                if not price or price <= 0:
                    continue
                try:
                    r = requests.get(
                        f"{base}/price-target-consensus",
                        params={"symbol": sym, "apikey": key},
                        timeout=5,
                    )
                    if r.status_code == 200:
                        data = r.json()
                        if data:
                            target = data[0].get("targetConsensus") or data[0].get("targetMedian")
                            if target and float(target) > 0:
                                result[sym] = (float(target) - price) / price
                except Exception:
                    pass
        except Exception as exc:
            self.logger.debug("Price target upside fetch failed: %s", exc)
        return result

    def _get_deployed_by_type(self) -> Dict[str, float]:
        """
        Return {trade_type: deployed_dollars} for open positions.
        Uses Alpaca positions tagged with trade_type in their client_order_id.
        Falls back to 0 on any error so sizing degrades gracefully.
        """
        deployed = {"swing": 0.0, "intraday": 0.0, "total": 0.0}
        try:
            positions = self._alpaca.get_positions()  # list of position dicts
            for pos in positions:
                market_val = abs(float(pos.get("market_value") or 0))
                deployed["total"] += market_val
                # Trade type tagged in metadata; default to swing for legacy positions
                tt = pos.get("trade_type") or "swing"
                if tt in deployed:
                    deployed[tt] += market_val
        except Exception:
            pass
        return deployed

    def _calculate_quantity(
        self,
        price: float,
        account_value: float,
        trade_type: str = "swing",
        confidence: float = MIN_CONFIDENCE,
    ) -> int:
        """
        Size a position using three layered constraints:
          1. Strategy budget: swing uses 70%, intraday uses 30% of account
          2. Gross exposure cap: total deployed never exceeds 80% of account
          3. Confidence scalar: high-confidence signals get larger allocations

        Base position = budget_for_type × POSITION_RISK_PCT × confidence_scalar
        Then clamp to remaining headroom in both the strategy budget and gross cap.
        """
        risk_pct = POSITION_RISK_PCT
        try:
            from app.database.session import get_session
            from app.database.agent_config import get_agent_config
            _db = get_session()
            try:
                risk_pct = get_agent_config(_db, "pm.position_risk_pct")
            finally:
                _db.close()
        except Exception:
            pass

        # 1. Strategy budget for this trade type
        budget_pct = SWING_BUDGET_PCT if trade_type == "swing" else INTRADAY_BUDGET_PCT
        strategy_budget = account_value * budget_pct

        # 2. How much of each budget and gross cap is already deployed
        deployed = self._get_deployed_by_type()
        type_headroom = max(0.0, strategy_budget - deployed.get(trade_type, 0.0))
        gross_headroom = max(0.0, account_value * GROSS_EXPOSURE_CAP - deployed["total"])
        available = min(type_headroom, gross_headroom)

        # 3. Base size = strategy_budget × risk_pct × confidence_scalar
        scalar = _confidence_scalar(confidence)
        base_dollars = strategy_budget * risk_pct * scalar

        # Clamp to available headroom
        position_dollars = min(base_dollars, available)

        qty = int(position_dollars / price)
        return max(qty, 1)

    # ─── Vol-Targeting Sizing (Phase 3d) ─────────────────────────────────────

    def _vol_targeting_quantity(
        self,
        price: float,
        account_value: float,
        base_quantity: int,
        atr_norm: float,
    ) -> tuple:
        """Return (adjusted_qty, vol_targeting_mult) using ATR-based vol targeting.

        Sizes the position so that one average daily move (ATR) contributes
        vol_target_pct of account equity in P&L.  Bounded by max_position_size_pct
        ceiling and vol_targeting_min_notional floor.

        Returns (base_quantity, 1.0) on any error so live trading never blocks.
        """
        from app.config import settings
        try:
            if price <= 0 or atr_norm <= 0:
                return base_quantity, 1.0

            atr_per_share = atr_norm * price            # $ daily move per share
            target_vol_dollar = account_value * settings.vol_target_pct
            raw_qty = int(target_vol_dollar / atr_per_share)

            # Ceiling: never exceed max_position_size_pct of account
            max_qty = max(1, int(account_value * settings.max_position_size_pct / price))
            qty = min(raw_qty, max_qty)

            # Floor: ensure minimum notional, but don't exceed ceiling
            if qty * price < settings.vol_targeting_min_notional:
                floor_qty = max(1, int(settings.vol_targeting_min_notional / price))
                qty = min(floor_qty, max_qty)

            qty = max(1, qty)
            mult = round(qty / max(base_quantity, 1), 4)
            return qty, mult

        except Exception as exc:
            self.logger.debug("vol_targeting_quantity failed (non-fatal): %s", exc)
            return base_quantity, 1.0

    # ─── Regime Sizing ────────────────────────────────────────────────────────

    def _regime_sizing_multiplier(self) -> tuple:
        """Return (multiplier, label, score) based on current regime context.

        Multipliers are config-driven (REGIME_SIZING_* in .env).
        Returns full size (1.0) when regime model is not yet available so
        early paper trading days are not penalised before the model warms up.
        """
        from app.config import settings
        ctx = self._current_regime_ctx
        if ctx is None:
            return (settings.regime_sizing_unknown, "UNKNOWN", None)
        score = ctx.get("regime_score")
        label = ctx.get("regime_label", "UNKNOWN")
        if score is None or label == "UNKNOWN":
            return (settings.regime_sizing_unknown, "UNKNOWN", score)
        if score >= settings.regime_risk_on_threshold:
            return (settings.regime_sizing_risk_on, "RISK_ON", score)
        if score < settings.regime_risk_off_threshold:
            return (settings.regime_sizing_risk_off, "RISK_OFF", score)
        return (settings.regime_sizing_risk_caution, "RISK_CAUTION", score)

    # ─── Retraining ───────────────────────────────────────────────────────────

    async def _retrain(self):
        """Retrain swing + intraday models with walk-forward gate enforcement.

        If the new model fails the gate, the previous ACTIVE model is restored
        and the new version is marked RETIRED. Models only go live after passing.
        """
        import functools
        import os
        from app.ml.retrain_config import SWING_RETRAIN, INTRADAY_RETRAIN, SWING_GATE, INTRADAY_GATE
        from app.ml.intraday_training import IntradayModelTrainer
        from app.database.session import get_session
        from app.database.models import ModelVersion

        os.environ.setdefault("OMP_NUM_THREADS", str(MAX_THREADS))
        os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(MAX_THREADS))
        loop = asyncio.get_event_loop()

        def _previous_active(strategy: str) -> int | None:
            """Return version number of current ACTIVE model before we overwrite it."""
            db = get_session()
            try:
                row = db.query(ModelVersion).filter_by(
                    model_name=strategy, status="ACTIVE"
                ).order_by(ModelVersion.version.desc()).first()
                return row.version if row else None
            finally:
                db.close()

        def _restore_previous(strategy: str, prev_version: int | None, new_version: int):
            """Retire the failed new model and restore the previous ACTIVE one."""
            if prev_version is None:
                return
            db = get_session()
            try:
                new = db.query(ModelVersion).filter_by(
                    model_name=strategy, version=new_version
                ).first()
                if new:
                    new.status = "RETIRED"
                prev = db.query(ModelVersion).filter_by(
                    model_name=strategy, version=prev_version
                ).first()
                if prev:
                    prev.status = "ACTIVE"
                db.commit()
                self.logger.info("%s v%d gate failed — restored v%d as ACTIVE",
                                 strategy, new_version, prev_version)
            finally:
                db.close()

        swing_ver = intraday_ver = None
        swing_promoted = intraday_promoted = False

        # ── Swing retrain + gate ─────────────────────────────────────────────
        self.logger.info("Swing retrain starting (model_type=%s hpo=%d wf_folds=%d)...",
                         SWING_RETRAIN["model_type"], SWING_RETRAIN["hpo_trials"],
                         SWING_RETRAIN["walk_forward_folds"])
        prev_swing = _previous_active("swing")
        try:
            swing_ver = await loop.run_in_executor(
                None,
                functools.partial(self.trainer.train_model,
                                  fetch_fundamentals=SWING_RETRAIN["fetch_fundamentals"]),
            )
            self.logger.info("Swing v%d trained — running walk-forward gate...", swing_ver)

            # Walk-forward gate
            from scripts.walkforward_tier3 import run_swing_walkforward
            wf = await loop.run_in_executor(
                None,
                functools.partial(run_swing_walkforward,
                                  n_folds=SWING_RETRAIN["walk_forward_folds"],
                                  model_version=swing_ver),
            )
            avg_sh = wf.avg_sharpe
            min_sh = wf.min_sharpe
            gate_ok = (avg_sh >= SWING_GATE["min_avg_sharpe"] and
                       min_sh >= SWING_GATE["min_fold_sharpe"])

            from app.ml.training import ModelTrainer as _SwingTrainer
            _SwingTrainer.record_tier3_result(swing_ver, avg_sh,
                                              [f.sharpe for f in wf.folds], gate_ok)
            if gate_ok:
                swing_promoted = True
                self.logger.info("Swing v%d GATE PASSED (avg=%.3f min=%.3f) — now ACTIVE",
                                 swing_ver, avg_sh, min_sh)
            else:
                _restore_previous("swing", prev_swing, swing_ver)
                self.logger.warning("Swing v%d GATE FAILED (avg=%.3f min=%.3f) — keeping v%d",
                                    swing_ver, avg_sh, min_sh, prev_swing)
                await self.log_decision("RETRAINING_GATE_FAILED", reasoning={
                    "strategy": "swing", "version": swing_ver,
                    "avg_sharpe": avg_sh, "min_sharpe": min_sh,
                    "restored_version": prev_swing,
                })
        except Exception as e:
            self.logger.error("Swing retrain failed: %s", e, exc_info=True)
            await self.log_decision("RETRAINING_FAILED", reasoning={"strategy": "swing", "error": str(e)})

        # ── Intraday retrain + gate ──────────────────────────────────────────
        self.logger.info("Intraday retrain starting...")
        prev_intraday = _previous_active("intraday")
        try:
            intraday_trainer = IntradayModelTrainer()
            intraday_ver = await loop.run_in_executor(
                None,
                functools.partial(intraday_trainer.train_model, **INTRADAY_RETRAIN),
            )
            self.logger.info("Intraday v%d trained — running walk-forward gate...", intraday_ver)

            from scripts.walkforward_tier3 import run_intraday_walkforward
            wf = await loop.run_in_executor(
                None,
                functools.partial(run_intraday_walkforward, n_folds=3,
                                  model_version=intraday_ver),
            )
            avg_sh = wf.avg_sharpe
            min_sh = wf.min_sharpe
            gate_ok = (avg_sh >= INTRADAY_GATE["min_avg_sharpe"] and
                       min_sh >= INTRADAY_GATE["min_fold_sharpe"])

            intraday_trainer.record_tier3_result(intraday_ver, avg_sh,
                                                 [f.sharpe for f in wf.folds], gate_ok)
            if gate_ok:
                intraday_promoted = True
                self.logger.info("Intraday v%d GATE PASSED (avg=%.3f min=%.3f) — now ACTIVE",
                                 intraday_ver, avg_sh, min_sh)
            else:
                _restore_previous("intraday", prev_intraday, intraday_ver)
                self.logger.warning("Intraday v%d GATE FAILED (avg=%.3f min=%.3f) — keeping v%d",
                                    intraday_ver, avg_sh, min_sh, prev_intraday)
                await self.log_decision("RETRAINING_GATE_FAILED", reasoning={
                    "strategy": "intraday", "version": intraday_ver,
                    "avg_sharpe": avg_sh, "min_sharpe": min_sh,
                    "restored_version": prev_intraday,
                })
        except Exception as e:
            self.logger.error("Intraday retrain failed: %s", e, exc_info=True)
            await self.log_decision("RETRAINING_FAILED", reasoning={"strategy": "intraday", "error": str(e)})

        # ── Reload whichever models passed, log outcome ───────────────────────
        self._try_load_model()
        if swing_promoted or intraday_promoted:
            await self.log_decision("MODEL_RETRAINED", reasoning={
                "swing_version": swing_ver if swing_promoted else None,
                "intraday_version": intraday_ver if intraday_promoted else None,
            })

        # Reap leftover joblib/loky worker processes
        try:
            from joblib.externals.loky import get_reusable_executor
            get_reusable_executor().shutdown(wait=False, kill_workers=True)
        except Exception:
            pass

    # ─── Normalization ────────────────────────────────────────────────────────

    def _normalize_for_inference(
        self,
        X: np.ndarray,
        symbols: list,
        model,
    ) -> np.ndarray:
        """Normalize feature matrix for inference.

        Fix 2: if the model carries a TSNormalizerState (trained with rolling
        time-series normalization), use it. Each symbol gets its live feature
        row normalized against its trailing history in the state.

        Falls back to cs_normalize for models trained before Fix 2 (v184 and
        earlier), which used cross-sectional normalization. This ensures
        backwards compatibility during the v184→v185 transition.
        """
        ts_state = getattr(model, "_ts_norm_state", None)
        # An empty TSNormalizerState (n_features=0, no history) is set by LambdaRank
        # training which intentionally skips TSNorm. Treat it as absent so we fall
        # through to cs_normalize rather than getting 0 rows from transform.
        if ts_state is not None and (ts_state.n_features == 0 and not ts_state.history):
            ts_state = None
        if ts_state is not None:
            try:
                from app.ml.ts_normalize import transform as _ts_transform
                import datetime as _dt
                _sym_arr = np.array(symbols)
                # All symbols share the same inference window (today).
                _wid_arr = np.full(len(symbols), _dt.date.today().toordinal(), dtype=int)
                X_norm, keep = _ts_transform(X, _sym_arr, _wid_arr, ts_state)
                if keep.all():
                    return X_norm
                # Partial keep (new symbols with no history): fill gaps with cs_normalize
                X_out = X_norm.copy()
                if not keep.all():
                    X_out[~keep] = cs_normalize(X[~keep]) if keep.sum() > 0 else X[~keep]
                return X_out
            except Exception as exc:
                self.logger.warning(
                    "TSNormalizerState transform failed, falling back to cs_normalize: %s", exc
                )
        # Legacy path: cross-sectional normalization (pre-Fix-2 models)
        return cs_normalize(X)

    # ─── Model Loading ────────────────────────────────────────────────────────

    def _try_load_model(self) -> bool:
        """Attempt to load the latest swing + intraday models from DB."""
        from app.database.models import ModelVersion
        from app.database.session import get_session

        db = get_session()
        swing_loaded = intraday_loaded = False
        try:
            for model_name, model_obj in [
                ("swing", self.model),
                ("intraday", self.intraday_model),
            ]:
                latest = (
                    db.query(ModelVersion)
                    .filter_by(model_name=model_name, status="ACTIVE")
                    .order_by(ModelVersion.version.desc())
                    .first()
                )
                if latest and latest.model_path:
                    try:
                        import pickle as _pickle
                        from pathlib import Path
                        from app.ml.model import LambdaRankModel as _LRModel
                        model_path = Path(latest.model_path)
                        model_dir = str(model_path.parent)
                        version = latest.version
                        # Detect model type by peeking at the pkl file.
                        # v201+ models are LambdaRankModel (self-contained, no separate scaler).
                        # v200 and earlier are PortfolioSelectorModel with separate scaler/meta files.
                        with open(model_path, "rb") as _f:
                            _raw = _pickle.load(_f)
                        if isinstance(_raw, _LRModel):
                            wrapper = _LRModel()
                            wrapper.__dict__.update(_raw.__dict__)
                            # Load TS norm state (same as LambdaRankModel.load)
                            norm_path = model_path.parent / f"swing_norm_v{version}.pkl"
                            if norm_path.exists() and model_name == "swing":
                                try:
                                    from app.ml.ts_normalize import load_state as _ts_load
                                    wrapper._ts_norm_state = _ts_load(str(norm_path))
                                except Exception:
                                    wrapper._ts_norm_state = None
                            wrapper.is_trained = True
                            wrapper.version = version
                        else:
                            wrapper = PortfolioSelectorModel(model_type="xgboost")
                            wrapper.load(model_dir, version, model_name=model_name)
                        loaded = wrapper
                        if model_name == "swing":
                            self.model = loaded
                            swing_loaded = True
                        else:
                            self.intraday_model = loaded
                            intraday_loaded = True
                        self.logger.info(
                            "Loaded %s model v%d (%s)",
                            model_name, latest.version, type(loaded).__name__,
                        )
                    except Exception as exc:
                        self.logger.warning("Could not load %s model: %s", model_name, exc)
                else:
                    self.logger.info("No %s model in DB yet", model_name)
        finally:
            db.close()
        return swing_loaded or intraday_loaded


# Module-level singleton (lazy — no connections at import time)
portfolio_manager = PortfolioManager()
