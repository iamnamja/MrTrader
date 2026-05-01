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
from app.ml.cs_normalize import cs_normalize
from app.ml.model import PortfolioSelectorModel
from app.ml.training import ModelTrainer
from app.utils.constants import SP_500_TICKERS, RUSSELL_1000_TICKERS

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
            hpo_trials=SWING_RETRAIN["hpo_trials"],
            n_workers=SWING_RETRAIN["n_workers"],
        )
        self._analyzed_today: bool = False       # 08:00 pre-market analysis done
        self._selected_today: bool = False       # 09:50 proposals sent
        self._selected_intraday_today: bool = False  # legacy compat (missed-task check)
        self._intraday_windows_run: set = set()  # Phase 51: (hour,min) windows already scanned
        self._intraday_symbol_last_entry: Dict[str, float] = {}  # symbol → monotonic ts of last entry
        self._retrained_today: bool = False
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
        # Adaptive intraday universe — top-N candidates from 9:45 scan reused for later windows
        self._morning_intraday_candidates: List[str] = []
        # SPY % at time of last intraday scan (for adaptive re-scan trigger)
        self._last_scan_spy_pct: float = 0.0
        # Monotonic timestamp of last adaptive re-scan (prevents back-to-back triggers)
        self._last_adaptive_scan_at: float = 0.0

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
        except Exception as e:
            self.logger.warning("Could not restore daily flags from DB: %s", e)
        finally:
            db.close()

    async def run(self):
        self.logger.info("Portfolio Manager started")
        self.status = "running"
        self._try_load_model()
        self._restore_daily_flags()

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
                    self._premarket_run_today = False
                    self._benchmark_recorded_today = False
                    self._weekly_report_generated_today = False
                    self._eod_jobs_run_today = False
                    self._last_swing_features = {}
                    self._morning_intraday_candidates = []
                    self._last_scan_spy_pct = 0.0
                    self._last_adaptive_scan_at = 0.0
                    self._last_intraday_features = {}
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
                        is_weekday
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
                    is_weekday
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
        """Return active tickers from DB watchlist; fall back to SP_500_TICKERS."""
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
            self.logger.debug("Watchlist DB unavailable, using SP_500: %s", exc)
        return list(SP_500_TICKERS)

    # ─── Instrument Selection ─────────────────────────────────────────────────

    def _fetch_swing_features(self) -> Dict[str, Dict[str, float]]:
        """Fetch bars + engineer features concurrently. Run in a thread via asyncio.to_thread."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _fetch_one(symbol: str):
            try:
                bars = self._alpaca.get_bars(symbol, timeframe="1D", limit=300)
                if bars.empty:
                    return symbol, None
                feats = self.feature_engineer.engineer_features(symbol, bars, fetch_fundamentals=False)
                return symbol, feats
            except Exception as e:
                self.logger.debug("Skipping %s: %s", symbol, e)
                return symbol, None

        features_by_symbol: Dict[str, Dict[str, float]] = {}
        symbols = self._get_universe()
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(_fetch_one, s): s for s in symbols}
            for future in as_completed(futures, timeout=300):
                try:
                    symbol, feats = future.result(timeout=30)
                except Exception as e:
                    self.logger.debug("Feature fetch timed out or failed: %s", e)
                    continue
                if feats is not None:
                    features_by_symbol[symbol] = feats
        return features_by_symbol

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

        # Gap 3: write daily summary row to risk_metrics
        try:
            from app.database.daily_summary import write_daily_summary
            await asyncio.to_thread(write_daily_summary)
            self.logger.info("EOD daily summary written to risk_metrics")
        except Exception as exc:
            self.logger.warning("EOD daily summary failed (non-fatal): %s", exc)

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

    async def _analyze_swing_premarket(self):
        """
        08:00 ET: Score all universe stocks using yesterday's daily bars.
        Results cached in self._swing_proposals — not sent to Risk Manager yet.
        Sending is deferred to 09:50 so we enter after the open volatility spike.
        """
        model_ver = getattr(self.model, "version", None)
        self.logger.info("Pre-market swing analysis starting — model v%s", model_ver)

        if not self.model.is_trained:
            self.logger.warning("No trained model — pre-market analysis skipped")
            await self.log_decision("SELECTION_SKIPPED", reasoning={
                "reason": "model not trained",
                "strategy": "swing",
                "model_version": model_ver,
            })
            return

        universe = self._get_universe()
        self.logger.info("Fetching features for %d symbols...", len(universe))
        t0 = _time.monotonic()

        # Run blocking network calls in a thread so the event loop stays free
        try:
            features_by_symbol = await asyncio.wait_for(
                asyncio.to_thread(self._fetch_swing_features), timeout=420
            )
        except asyncio.TimeoutError:
            self.logger.error("Swing feature fetch timed out after 7 minutes — skipping")
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
        model_feature_names = getattr(self.model, "feature_names", None)
        if model_feature_names:
            X = np.array([
                [features_by_symbol[s].get(f, 0.0) for f in model_feature_names]
                for s in symbols
            ])
        else:
            X = np.array([list(features_by_symbol[s].values()) for s in symbols])
        X = np.nan_to_num(X)
        X = cs_normalize(X)

        try:
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

        ranked = sorted(zip(symbols, boosted_probs), key=lambda x: x[1], reverse=True)
        selected = [(sym, prob) for sym, prob in ranked if prob >= min_conf][:top_n]

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
            vix = yf.download("^VIX", period="5d", progress=False, auto_adjust=True)
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

    def _market_regime_allows_entries(self) -> bool:
        """
        PM abstention gate. Returns False on unfavorable broad-market conditions:
          - VIX >= 25 (elevated fear)
          - SPY below its 20-day MA (short-term downtrend)
          - SPY 5-day return <= 0 (negative momentum, Phase 55)
        Any condition alone is enough to abstain. Fails open if data unavailable.
        """
        try:
            import yfinance as yf
            import pandas as pd
            spy = yf.download("SPY", period="30d", progress=False, auto_adjust=True)
            if isinstance(spy.columns, pd.MultiIndex):
                spy.columns = spy.columns.get_level_values(0)
            spy.columns = [c.lower() for c in spy.columns]
            if len(spy) < 20:
                return True
            spy_close = float(spy["close"].iloc[-1])
            spy_ma20 = float(spy["close"].tail(20).mean())
            spy_below_ma = spy_close < spy_ma20
            spy_5d_ret = (spy_close / float(spy["close"].iloc[-6]) - 1.0) if len(spy) >= 6 else 0.0
            spy_momentum_weak = spy_5d_ret <= 0.0

            vix_level = self._fetch_vix_level()
            vix_high = vix_level >= 25.0

            if spy_below_ma or vix_high or spy_momentum_weak:
                self.logger.info(
                    "PM abstention gate ACTIVE — SPY=%.2f MA20=%.2f (below=%s) "
                    "VIX=%.1f (>=25=%s) SPY5d=%.2f%% (weak=%s)",
                    spy_close, spy_ma20, spy_below_ma, vix_level, vix_high,
                    spy_5d_ret * 100, spy_momentum_weak,
                )
                return False
            self.logger.info(
                "PM abstention gate clear — SPY=%.2f MA20=%.2f VIX=%.1f SPY5d=%.2f%%",
                spy_close, spy_ma20, vix_level, spy_5d_ret * 100,
            )
            return True
        except Exception as exc:
            self.logger.warning("PM abstention gate check failed (%s) — failing open", exc)
            return True

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

        # Phase 3-Parallel: PM abstention gate — suppress entries on bad regime days
        regime_ok = await asyncio.to_thread(self._market_regime_allows_entries)
        if not regime_ok:
            self.logger.info("PM abstention gate: suppressing %d swing proposals today",
                             len(self._swing_proposals))
            await self.log_decision(
                "SWING_ABSTAINED",
                reasoning={"reason": "PM abstention gate: VIX>=25 or SPY below 20-day MA"},
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
        for proposal in self._swing_proposals:
            self.send_message(TRADE_PROPOSALS_QUEUE, proposal)
            self._pending_approvals[proposal["symbol"]] = _time.monotonic()
            self.logger.info(
                "Proposal sent: %s @ $%.2f (confidence=%.2f)",
                proposal["symbol"], proposal["entry_price"], proposal["confidence"],
            )

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
            # Market hours — run intraday
            self.logger.info(
                "select_instruments called at %02d:%02d ET — routing to intraday scan",
                now.hour, now.minute,
            )
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

        # PM abstention gate: skip all intraday entries on bad regime days (VIX>=25 or SPY<MA20)
        regime_ok = await asyncio.to_thread(self._market_regime_allows_entries)
        if not regime_ok:
            self.logger.info("PM abstention gate: suppressing all intraday entries today")
            return

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

        def _fetch_one_intraday(symbol: str) -> Optional[tuple]:
            """Return (features_dict, prior_day_range) or None."""
            from app.ml.intraday_features import compute_intraday_features, MIN_BARS
            try:
                bars = self._alpaca.get_bars(symbol, timeframe="5Min", limit=78)
                if bars is None or bars.empty or len(bars) < MIN_BARS:
                    return None
                daily = self._alpaca.get_bars(symbol, timeframe="1Day", limit=25)
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
                )
                if feats is None:
                    return None
                return feats, prior_day_range
            except Exception as exc:
                self.logger.debug("Intraday feature skip %s: %s", symbol, exc)
                return None

        def _fetch_intraday_features() -> tuple:
            """Return (features_by_symbol, prior_ranges_by_symbol)."""
            from concurrent.futures import ThreadPoolExecutor, as_completed
            feat_result: Dict[str, Dict[str, float]] = {}
            range_result: Dict[str, Optional[float]] = {}
            with ThreadPoolExecutor(max_workers=16) as pool:
                futures = {pool.submit(_fetch_one_intraday, s): s for s in scan_universe}
                for future in as_completed(futures, timeout=300):
                    sym = futures[future]
                    try:
                        out = future.result(timeout=30)
                    except Exception:
                        continue
                    if out is not None:
                        feat_result[sym] = out[0]
                        range_result[sym] = out[1]
            return feat_result, range_result

        try:
            features_by_symbol, prior_ranges_by_symbol = await asyncio.wait_for(
                asyncio.to_thread(_fetch_intraday_features), timeout=420
            )
        except asyncio.TimeoutError:
            self.logger.error("Intraday feature fetch timed out after 7 minutes — skipping")
            return

        if not features_by_symbol:
            self.logger.warning("No intraday features computed")
            return

        symbols = list(features_by_symbol.keys())
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
        X = cs_normalize(X)

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

        # Store top-N by score for later scans (10:45, 13:00 use this instead of full universe)
        if not use_morning_candidates:
            self._morning_intraday_candidates = [
                sym for sym, _ in ranked[:INTRADAY_AFTERNOON_CANDIDATES]
            ]
            self.logger.info(
                "Stored %d morning candidates for afternoon scans",
                len(self._morning_intraday_candidates),
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
        ][:TOP_N_INTRADAY]

        if not selected:
            self.logger.info("No intraday candidates above confidence threshold (or all on cooldown)")
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
                continue

            quantity = self._calculate_quantity(
                price, account_value, trade_type="intraday", confidence=float(confidence)
            )
            # Apply SPY caution sizing multiplier (0.5× in down-market caution mode)
            if intraday_size_mult != 1.0:
                quantity = max(1, int(quantity * intraday_size_mult))
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
            self.send_message(TRADE_PROPOSALS_QUEUE, proposal)
            self._pending_approvals[proposal["symbol"]] = _time.monotonic()
            self._intraday_symbol_last_entry[symbol] = _time.monotonic()  # Phase 51 cooldown
            news_monitor.watch(symbol)  # Phase 53: start monitoring for news exits
            self.logger.info(
                "Intraday proposal: %s @ $%.2f (confidence=%.2f)",
                symbol, price, confidence,
            )
            try:
                from app.database.decision_audit import write_decision
                write_decision(
                    symbol, "intraday", "enter",
                    model_score=float(confidence),
                    top_features=self._top_features_for(symbol, "intraday"),
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
            active_trades = db.query(Trade).filter_by(status="ACTIVE", direction="BUY").all()
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
                    _, probs = self.model.predict(x)
                    results[trade.symbol] = {
                        "score": float(probs[0]),
                        "trade_id": trade.id,
                        "entry_price": float(trade.entry_price or 0),
                        "target_price": float(trade.target_price or 0),
                        "atr": (
                            float(trade.target_price or 0) - float(trade.entry_price or 0)
                            if trade.target_price and trade.entry_price else 0.0
                        ),
                    }
                except Exception as exc:
                    self.logger.debug("Re-score failed for %s: %s", trade.symbol, exc)
            return results

        scores = await asyncio.to_thread(_score_positions)

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

        for symbol, info in scores.items():
            score = info["score"]
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
            for sym, info in scores.items():
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
        regime_ok = await asyncio.to_thread(self._market_regime_allows_entries)
        if not regime_ok:
            self.logger.info("30-min scan skipped — PM abstention gate active (VIX/SPY MA)")
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
        X = cs_normalize(X)

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
                continue

            quantity = self._calculate_quantity(
                price, account_value, trade_type="swing", confidence=float(confidence)
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
                write_decision(
                    symbol, "swing", "enter",
                    model_score=float(confidence),
                    size_multiplier=round(float(news_sig.sizing_multiplier if news_sig else 1.0), 4),
                    news_signal=news_sig,
                    macro_context=macro_ctx,
                    top_features=self._top_features_for(symbol, "swing"),
                )
            except Exception:
                pass

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

        os.environ.setdefault("OMP_NUM_THREADS", "24")
        os.environ.setdefault("LOKY_MAX_CPU_COUNT", "24")
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
                        from pathlib import Path
                        model_path = Path(latest.model_path)
                        model_dir = str(model_path.parent)
                        version = latest.version
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
