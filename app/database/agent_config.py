"""
Agent configuration store — runtime-tunable parameters for PM / Risk / Trader.

All values live in the `configuration` DB table under namespaced keys
(e.g. "pm.min_confidence", "risk.max_position_size_pct").

Agents call get_agent_config() at decision time so changes take effect
without a restart.  Hardcoded module constants remain as the fallback
default if the DB row has never been written.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# ── Schema: all tunable parameters ────────────────────────────────────────────
# Each entry: key, default, type, min, max, description, group
CONFIG_SCHEMA: List[Dict[str, Any]] = [
    # Portfolio Manager
    {
        "key": "pm.top_n_stocks",
        "default": 10,
        "type": "int",
        "min": 1,
        "max": 30,
        "description": "Maximum stocks to select per cycle",
        "group": "Portfolio Manager",
    },
    {
        "key": "pm.min_confidence",
        "default": 0.55,
        "type": "float",
        "min": 0.5,
        "max": 0.95,
        "description": "Minimum ML model probability to propose a trade",
        "group": "Portfolio Manager",
    },
    {
        "key": "pm.position_risk_pct",
        "default": 0.02,
        "type": "float",
        "min": 0.005,
        "max": 0.05,
        "description": "Fraction of account to risk per trade (position sizing)",
        "group": "Portfolio Manager",
    },
    # Risk Manager
    {
        "key": "risk.max_position_size_pct",
        "default": 0.05,
        "type": "float",
        "min": 0.01,
        "max": 0.20,
        "description": "Max single position as % of account value",
        "group": "Risk Manager",
    },
    {
        "key": "risk.max_sector_concentration_pct",
        "default": 0.20,
        "type": "float",
        "min": 0.05,
        "max": 0.50,
        "description": "Max sector exposure as % of account value",
        "group": "Risk Manager",
    },
    {
        "key": "risk.max_daily_loss_pct",
        "default": 0.02,
        "type": "float",
        "min": 0.005,
        "max": 0.10,
        "description": "Max daily loss before blocking new trades",
        "group": "Risk Manager",
    },
    {
        "key": "risk.max_account_drawdown_pct",
        "default": 0.05,
        "type": "float",
        "min": 0.01,
        "max": 0.20,
        "description": "Max peak-to-trough drawdown before blocking new trades",
        "group": "Risk Manager",
    },
    {
        "key": "risk.max_open_positions",
        "default": 5,
        "type": "int",
        "min": 1,
        "max": 20,
        "description": "Maximum simultaneous open positions",
        "group": "Risk Manager",
    },
    {
        "key": "pm.exit_threshold",
        "default": 0.35,
        "type": "float",
        "min": 0.10,
        "max": 0.55,
        "description": "Re-score below this threshold triggers PM exit signal",
        "group": "Portfolio Manager",
    },
    {
        "key": "pm.swing_selector",
        "default": "pead_quality_short",
        "type": "str",
        "description": (
            "Swing selection method: 'ml_model' uses LambdaRank/XGBoost scores; "
            "'factor_portfolio' uses momentum+quality composite; "
            "'pead' uses PEAD scorer (EPS surprise, hold-5); "
            "'quality_short' uses QualityShortScorer (shorts-only); "
            "'pead_quality_short' combines PEAD + QualityShort (Phase I default)"
        ),
        "group": "Portfolio Manager",
    },
    {
        "key": "pm.pead_enable_shorts",
        "default": "false",
        "type": "str",
        "description": (
            "Enable PEAD short entries (EPS surprise < -5%). "
            "Requires margin-enabled Alpaca account and short order routing to be wired. "
            "Default false = longs only."
        ),
        "group": "Portfolio Manager",
    },
    # PEAD live config surface — defaults pin the validated +0.546 CPCV config.
    # Exposed so the live PEAD selector is inspectable and cannot silently drift
    # from the backtest (see run_pead_cpcv.py for the source config).
    {
        "key": "pm.pead_long_threshold",
        "default": 0.05,
        "type": "float",
        "min": 0.0,
        "max": 0.5,
        "description": "PEAD long EPS-surprise threshold (validated +0.546 config = 0.05).",
        "group": "Portfolio Manager",
    },
    {
        "key": "pm.pead_short_threshold",
        "default": -0.05,
        "type": "float",
        "min": -0.5,
        "max": 0.0,
        "description": "PEAD short EPS-surprise threshold (validated config = -0.05).",
        "group": "Portfolio Manager",
    },
    {
        "key": "pm.pead_max_days_after",
        "default": 3,
        "type": "int",
        "min": 1,
        "max": 10,
        "description": "Max calendar days after earnings report to still enter PEAD (validated = 3).",
        "group": "Portfolio Manager",
    },
    {
        "key": "pm.pead_max_hold_days",
        "default": 40,
        "type": "int",
        "min": 1,
        "max": 60,
        "description": "PEAD max hold in trading days before forced exit (validated +0.546 run = 40).",
        "group": "Portfolio Manager",
    },
    {
        "key": "pm.pead_size_mult",
        "default": 1.0,
        "type": "float",
        "min": 1.0,
        "max": 10.0,
        "description": "PEAD per-name size multiplier. Alpha-v4 rebaseline: TELEMETRY size (1.0 = no ramp) — PEAD is a weak market-beta satellite, so it runs small alongside the trend sleeve. (Was B4 paper-ramp 3.0; reverted 2026-06-06.) Live-tunable. PAPER ONLY.",
        "group": "Portfolio Manager",
    },
    {
        "key": "pm.pead_max_position_pct",
        "default": 0.05,
        "type": "float",
        "min": 0.01,
        "max": 0.25,
        "description": "Per-name ceiling for PEAD positions as a fraction of account value. Alpha-v4 rebaseline: TELEMETRY size 0.05 = 5% NAV (was B4 paper-ramp 0.10; reverted 2026-06-06 to give the trend sleeve room under the 80% gross cap). Live-tunable. PAPER ONLY.",
        "group": "Portfolio Manager",
    },
    {
        "key": "pm.pead_regime_control",
        "default": "trend",
        "type": "str",
        "description": "B5 regime de-risk control, REPLACES the VIX>30 block when set. 'trend' = block PEAD entries when SPY < its N-day SMA (validated: +0.661 vs +0.546 VIX-block). '' (empty) = keep the VIX>30 block. Fails CLOSED to VIX if SPY data is unavailable.",
        "group": "Portfolio Manager",
    },
    {
        "key": "pm.pead_trend_ma",
        "default": 200,
        "type": "int",
        "min": 50,
        "max": 300,
        "description": "B5 SPY SMA window (trading days) for the trend filter. Validated = 200.",
        "group": "Portfolio Manager",
    },
    {
        "key": "pm.pead_regime_floor",
        "default": 0.5,
        "type": "float",
        "min": 0.0,
        "max": 1.0,
        "description": "B5 regime-scalar floor: block PEAD entries when the regime scalar is below this (trend mode returns 0/1, so any floor in (0,1] gates downtrends).",
        "group": "Portfolio Manager",
    },
    # ── Dead cross-sectional swing ML ranker — live kill switch ──────────────────
    {
        "key": "pm.swing_ml_live_enabled",
        "default": "false",
        "type": "str",
        "description": "MASTER FLAG for the live cross-sectional swing ML ranker (selector='ml_model' premarket fall-through + the 30-min _scan_new_opportunities rescan). 'false' = DORMANT: the validated-null ranker (DECISIONS 2026-06-03; frozen for retraining via SWING_ENABLED=False) proposes no live trades. PEAD / quality_short / factor_portfolio / trend / intraday are unaffected. Set 'true' only to deliberately re-enable the dead ranker.",
        "group": "Portfolio Manager",
    },
    # ── Trend (TSMOM) sleeve — Alpha-v4 Phase 2/live wiring ──────────────────────
    {
        "key": "pm.trend_enabled",
        "default": "false",
        "type": "str",
        "description": "MASTER FLAG for the live TSMOM trend sleeve. 'false' = code deployed but DORMANT (no orders). Set 'true' to go live (weekly ETF rebalance). Safety default OFF so deploy never auto-trades before you flip it.",
        "group": "Portfolio Manager",
    },
    {
        "key": "pm.trend_shadow",
        "default": "true",
        "type": "str",
        "description": "Trend dry-run (SAFETY DEFAULT ON): 'true' = compute + LOG the ETF orders it would place (decision_audit, block_reason='shadow') without sending. 'false' = send real orders. Independent of trend_enabled (enabled gates whether it runs at all).",
        "group": "Portfolio Manager",
    },
    {
        "key": "pm.whole_book_gate_mode",
        "default": "shadow",
        "type": "str",
        "description": "Alpha-v10 R0.5 whole-book risk gate mode. 'shadow' (default) = the gate computes the proposed-book risk-policy-v1 caps (gross-ex-cash, net equity beta, single/book notional) and LOGS + emails what it WOULD block, but does NOT block (the rebalance proceeds as today). 'enforce' = a cap breach HOLDS the rebalance (fail-closed; a missed rebalance, never a bad trade). 'off' = gate not evaluated. FAIL-SAFE: any gate error -> proceed (shadow). Flip to 'enforce' only after a clean shadow window.",
        "group": "Portfolio Manager",
    },
    {
        "key": "pm.crash_governor_enabled",
        "default": "true",
        "type": "str",
        "description": "Alpha-v7 F1b VIX term-structure CRASH GOVERNOR (exposure overlay; owner-approved 2026-06-14). 'true' = de-risk the trend sleeve's budget to crash_governor_derisk_to when VIX>VIX3M (backwardation = acute stress), else full exposure. 'false' = overlay off (multiplier always 1.0). FAIL-SAFE: VIX/VIX3M missing/stale/error -> multiplier 1.0 (today's behavior); the overlay can only REDUCE exposure.",
        "group": "Portfolio Manager",
    },
    {
        "key": "pm.crash_governor_derisk_to",
        "default": 0.5,
        "type": "float",
        "min": 0.0,
        "max": 1.0,
        "description": "Trend exposure multiplier applied while in VIX backwardation (VixTermGovernorConfig.derisk_to). 0.5 = halve exposure; 0.0 = flat. Validated config (2026-06-14): maxDD -13.9%->-12.1%, COVID DD -10.7%->-6.5%, Sharpe ~flat.",
        "group": "Portfolio Manager",
    },
    {
        "key": "pm.crash_governor_ratio_threshold",
        "default": 1.0,
        "type": "float",
        "min": 0.8,
        "max": 1.5,
        "description": "VIX/VIX3M ratio above which the governor de-risks (VixTermGovernorConfig.ratio_threshold). 1.0 = the contango/backwardation boundary (validated; robust neighborhood).",
        "group": "Portfolio Manager",
    },
    {
        "key": "pm.crash_governor_confirm_days",
        "default": 1,
        "type": "int",
        "min": 1,
        "max": 10,
        "description": "Consecutive inverted (VIX>VIX3M) settled closes required before de-risking (VixTermGovernorConfig.confirm_days). 1 = react immediately (validated default).",
        "group": "Portfolio Manager",
    },
    {
        "key": "pm.credit_governor_enabled",
        "default": "false",
        "type": "str",
        "description": "Alpha-v8 G1 CREDIT de-risk overlay (HYG/IEF). 'false' (DEFAULT — owner-gated CANDIDATE, not yet approved) = off (multiplier 1.0). 'true' = de-risk the trend sleeve to credit_governor_derisk_to when HYG/IEF is >credit_governor_band below its credit_governor_lookback-day MA (credit spreads widening). Composes multiplicatively with the VIX governor (product clamped to a 0.25 floor). FAIL-SAFE: missing/stale/error -> 1.0; can only REDUCE exposure. CAVEATS: small tail-insurance effect (marginal dSharpe +0.064, PIT-CONFIRMED 2026-06-22 P0.4 — reproduces +0.0639, NOT a vol-match artifact, all-3-crises); binding caveat is multiplicity (post-hoc L=120/band=0.02 trigger — see DECISIONS 2026-06-14 G1 + 2026-06-22 P0.4); review before enabling.",
        "group": "Portfolio Manager",
    },
    {
        "key": "pm.credit_governor_derisk_to",
        "default": 0.5,
        "type": "float",
        "min": 0.0,
        "max": 1.0,
        "description": "Trend exposure multiplier applied while credit-stressed (CreditGovernorConfig.derisk_to). Validated G1 candidate config = 0.5.",
        "group": "Portfolio Manager",
    },
    {
        "key": "pm.credit_governor_lookback",
        "default": 120,
        "type": "int",
        "min": 20,
        "max": 252,
        "description": "Trailing MA window (days) for the HYG/IEF credit ratio (CreditGovernorConfig.lookback). Validated = 120 (the selective trigger; L60 was over-eager and failed).",
        "group": "Portfolio Manager",
    },
    {
        "key": "pm.credit_governor_band",
        "default": 0.02,
        "type": "float",
        "min": 0.0,
        "max": 0.10,
        "description": "De-risk when HYG/IEF < (1-band)*MA (CreditGovernorConfig.band). Validated = 0.02 (fire on >2% deterioration = stress, not noise).",
        "group": "Portfolio Manager",
    },
    {
        "key": "pm.trend_allocation_pct",
        "default": 0.50,
        "type": "float",
        "min": 0.0,
        "max": 0.80,
        "description": "Trend sleeve budget as a fraction of account value (caps total trend gross). Alpha-v9 P1-2 (2026-06-16): raised 0.25->0.50 after a Kelly/vol-target analysis (scripts/analyze_trend_allocation.py) on the real 19.4y book (100%-gross Sharpe 0.72, ann vol 9.34%): Kelly is not the binding constraint (full Kelly = 7.7x gross), so 0.25 (2.3% standalone vol) far under-deployed the only live edge; 0.50 targets ~4.7% standalone vol / ~-7% maxDD — deeply Kelly-haircut, VIX-governor-protected, under the 0.80 cap. Prior: reconciled 2026-06-12 from 0.40 to 0.25 (Track-B 25% risk-budget framing #451) after the H1 DEMOTE. Revisable as live-paper data accrues (P1-4). See DECISIONS 2026-06-16 (P1-2).",
        "group": "Portfolio Manager",
    },
    {
        "key": "pm.trend_max_position_pct",
        "default": 0.25,
        "type": "float",
        "min": 0.01,
        "max": 0.40,
        "description": "Per-ETF ceiling as a fraction of account value (RM max_pct_override for selector=='trend'). Matches the sleeve's per-name cap (0.25).",
        "group": "Portfolio Manager",
    },
    {
        "key": "pm.trend_universe",
        "default": "SPY,QQQ,IWM,EFA,EEM,TLT,IEF,GLD,DBC,UUP",
        "type": "str",
        "description": "Comma-separated ETF universe for the trend sleeve (multi-asset, liquid, no leveraged/inverse). Matches the validated backtest basket.",
        "group": "Portfolio Manager",
    },
    {
        "key": "pm.trend_rebalance_weekday",
        "default": 0,
        "type": "int",
        "min": 0,
        "max": 4,
        "description": "Weekday for the weekly trend rebalance (0=Mon .. 4=Fri). Weekly cadence matches the validated sleeve.",
        "group": "Portfolio Manager",
    },
    # ── Cash / T-bill sleeve — Alpha-v9 P1-1 (default OFF; deploy idle cash to T-bills) ─
    {
        "key": "pm.cash_enabled",
        "default": "false",
        "type": "str",
        "description": "MASTER FLAG for the T-bill cash sleeve (P1-1). 'false' = code deployed but DORMANT (no orders). 'true' = weekly rebalance that parks idle settled cash (beyond the buffer) into a T-bill ETF so it earns the risk-free rate instead of zero. Safety default OFF.",
        "group": "Portfolio Manager",
    },
    {
        "key": "pm.cash_shadow",
        "default": "true",
        "type": "str",
        "description": "Cash sleeve dry-run (SAFETY DEFAULT ON): 'true' = compute + LOG the T-bill orders it would place (decision_audit block_reason='shadow') without sending. 'false' = send real orders. Independent of pm.cash_enabled.",
        "group": "Portfolio Manager",
    },
    {
        "key": "pm.cash_buffer_pct",
        "default": 0.02,
        "type": "float",
        "min": 0.0,
        "max": 0.20,
        "description": "Liquidity buffer kept as SETTLED CASH (fraction of NAV), never parked in T-bills, so other sleeves have same-day cash for intraday needs. The cash sleeve deploys (settled cash - buffer) into T-bills and SELLS T-bills to refill the buffer when settled cash dips below it (the risk-off path). 0.02 = 2% of NAV.",
        "group": "Portfolio Manager",
    },
    {
        "key": "pm.cash_universe",
        "default": "SGOV,BIL",
        "type": "str",
        "description": "Comma-separated T-bill ETF universe (first = primary buy target). Must be ultra-short Treasury/T-bill ETFs (SGOV, BIL, SHV, ...) — these are EXCLUDED from the 80% risk gross cap as cash-equivalents, so deploying them never starves trend/PEAD.",
        "group": "Portfolio Manager",
    },
    {
        "key": "pm.cash_rebalance_weekday",
        "default": 0,
        "type": "int",
        "min": 0,
        "max": 4,
        "description": "Weekday for the weekly cash rebalance (0=Mon .. 4=Fri). Runs AFTER the trend rebalance (09:50 ET) so the idle remainder is known.",
        "group": "Portfolio Manager",
    },
    # ── IBKR connection — Alpha-v10 P2.2 (READ-ONLY adapter; futures venue, PAPER first) ─
    {
        "key": "ibkr.enabled",
        "default": "false",
        "type": "str",
        "description": "Master flag for the IBKR connection. 'false' (default) = adapter never connects (fully inert). 'true' = the read-only adapter may connect to the running IB Gateway/TWS for verify-on-connect + cross-venue book-state. NO order capability exists yet (R1).",
        "group": "IBKR",
    },
    {
        "key": "ibkr.host",
        "default": "127.0.0.1",
        "type": "str",
        "description": "IB Gateway/TWS API host. localhost (the adapter runs on the same machine; TWS API is set to 'allow localhost only').",
        "group": "IBKR",
    },
    {
        "key": "ibkr.port",
        "default": 7497,
        "type": "int",
        "description": "IB Gateway/TWS API socket port. TWS paper=7497, IB Gateway paper=4002 (live: 7496/4001). Paper-only until R1 go-live.",
        "group": "IBKR",
    },
    {
        "key": "ibkr.client_id",
        "default": 1,
        "type": "int",
        "description": "API client id the adapter uses to identify its connection (any int; distinct per concurrent client).",
        "group": "IBKR",
    },
    # ── Crypto trend LIVE-PAPER tracker — Alpha-v9 P3-1 (report-only; NO capital) ─
    {
        "key": "pm.crypto_paper_enabled",
        "default": "true",
        "type": "str",
        "description": "Crypto-trend LIVE-PAPER tracker (P3-1). 'true' = weekly, recompute the rules-based crypto-trend book on live Alpaca closes and freeze the forward out-of-sample slice (Sharpe-to-date vs the 0.64 backtest). REPORT-ONLY — no orders, no capital, no risk-cap interaction. 'false' = dormant.",
        "group": "Portfolio Manager",
    },
    {
        "key": "pm.crypto_paper_rebalance_weekday",
        "default": 0,
        "type": "int",
        "min": 0,
        "max": 6,
        "description": "Weekday for the weekly crypto live-paper tracker (0=Mon .. 6=Sun). Crypto trades 24/7 so any day works; the tracker just recomputes + snapshots the OOS record.",
        "group": "Portfolio Manager",
    },
    # ── Regime-aware sleeve allocator — Alpha-v4 P3 (gate-controlled, default OFF) ─
    {
        "key": "pm.allocator_enabled",
        "default": "false",
        "type": "str",
        "description": "MASTER FLAG for the live regime-aware sleeve allocator. 'false' (default) = sleeves use their STATIC budgets (pm.trend_allocation_pct / pm.pead_size_mult) exactly as today — zero behavior change. 'true' = the allocator computes effective sleeve weights weekly and both sleeves read them (with fixed-weight fallback).",
        "group": "Portfolio Manager",
    },
    {
        "key": "pm.allocator_scheme",
        "default": "equal",
        "type": "str",
        "description": "Allocator weighting scheme: 'equal' (validated Phase-3 winner), 'vol' (inverse-vol risk parity), or 'regime' (regime-tilted). Keep 'equal' until the book-level gate (scripts/run_book_allocator.py) selects vol/regime — on 2 sleeves equal beats both. Maps to sleeve_allocator.build_book scheme.",
        "group": "Portfolio Manager",
    },
    {
        "key": "pm.allocator_vol_lookback",
        "default": 60,
        "type": "int",
        "min": 20,
        "max": 252,
        "description": "Trailing window (trading days) for the allocator's per-sleeve realized-vol estimate (vol/regime schemes). AllocatorConfig.vol_lookback.",
        "group": "Portfolio Manager",
    },
    {
        "key": "pm.allocator_total_budget_pct",
        "default": 0.80,
        "type": "float",
        "min": 0.0,
        "max": 0.80,
        "description": "Total gross the allocator's sleeve weights (which sum to 1) multiply into — bounded by the global 80% gross cap shared across sleeves. trend effective allocation = allocator_trend_weight * this.",
        "group": "Portfolio Manager",
    },
    {
        "key": "pm.allocator_min_deployed_days",
        "default": 20,
        "type": "int",
        "min": 2,
        "max": 252,
        "description": "Warmup guard: each sleeve needs >= this many deployed (gross>0) daily-return rows before the allocator leaves equal/static weighting. Below it, the allocator falls back to static config.",
        "group": "Portfolio Manager",
    },
    {
        "key": "pm.allocator_stale_days",
        "default": 8,
        "type": "int",
        "min": 1,
        "max": 60,
        "description": "Effective allocator weights older than this many days are treated as STALE -> readers fall back to static config (safety against a wedged recompute).",
        "group": "Portfolio Manager",
    },
    # Allocator OUTPUTS (written by the weekly recompute; read by the sleeve readers).
    {
        "key": "pm.allocator_trend_weight",
        "default": 0.40,
        "type": "float",
        "min": 0.0,
        "max": 1.0,
        "description": "OUTPUT: latest allocator trend-sleeve weight (fraction of the allocator budget). Written by run_allocator; read by trend_sleeve.effective_trend_allocation. Default 0.40 mirrors the static split.",
        "group": "Portfolio Manager",
    },
    {
        "key": "pm.allocator_pead_weight",
        "default": 0.40,
        "type": "float",
        "min": 0.0,
        "max": 1.0,
        "description": "OUTPUT: latest allocator PEAD-sleeve weight (fraction of the allocator budget). Written by run_allocator; read by effective_pead_size_mult.",
        "group": "Portfolio Manager",
    },
    {
        "key": "pm.allocator_last_computed",
        "default": "",
        "type": "str",
        "description": "OUTPUT: ISO timestamp of the last allocator recompute. Used for the staleness fallback (pm.allocator_stale_days).",
        "group": "Portfolio Manager",
    },
    {
        "key": "pm.pead_vix_block_all",
        "default": 30.0,
        "type": "float",
        "min": 10.0,
        "max": 100.0,
        "description": "VIX level above which ALL PEAD entries are blocked (crisis). Validated = 30.",
        "group": "Portfolio Manager",
    },
    {
        "key": "pm.pead_vix_block_short",
        "default": 100.0,
        "type": "float",
        "min": 10.0,
        "max": 100.0,
        "description": "VIX above which PEAD short leg is disabled. Long-only validated = 100 (never).",
        "group": "Portfolio Manager",
    },
    {
        "key": "pm.pead_vix_conf_ref",
        "default": 100.0,
        "type": "float",
        "min": 10.0,
        "max": 100.0,
        "description": "VIX confidence-damping reference. Validated long-only = 100 (no damping).",
        "group": "Portfolio Manager",
    },
    {
        "key": "pm.pead_max_announce_day_move",
        "default": 1.0,
        "type": "float",
        "min": 0.01,
        "max": 1.0,
        "description": "Priced-in filter: skip if announce-day move exceeds this. Validated = 1.0 (OFF).",
        "group": "Portfolio Manager",
    },
    {
        "key": "pm.pead_require_positive_revision",
        "default": "false",
        "type": "str",
        "description": "PEAD earnings-quality gate. Validated +0.546 config = false (OFF).",
        "group": "Portfolio Manager",
    },
    {
        "key": "pm.ls_net_exposure_pct",
        "default": 0.40,
        "type": "float",
        "min": 0.0,
        "max": 1.0,
        "description": (
            "Target net long exposure (long_mkt_val - short_mkt_val) / NAV. "
            "0.40 = 40% net long (directional L/S). 1.0 = long-only."
        ),
        "group": "Portfolio Manager",
    },
    {
        "key": "pm.ls_top_n_long",
        "default": 20,
        "type": "int",
        "min": 5,
        "max": 50,
        "description": "Number of long candidates from factor composite score (top-N).",
        "group": "Portfolio Manager",
    },
    {
        "key": "pm.ls_top_n_short",
        "default": 15,
        "type": "int",
        "min": 5,
        "max": 50,
        "description": "Number of short candidates from factor composite score (bottom-N).",
        "group": "Portfolio Manager",
    },
    {
        "key": "pm.ls_borrow_cost_annual_pct",
        "default": 0.005,
        "type": "float",
        "min": 0.0,
        "max": 0.10,
        "description": "Annualised borrow cost for short positions (deducted daily). 0.005 = 0.5%/yr.",
        "group": "Portfolio Manager",
    },
    {
        "key": "pm.ls_net_exposure_tolerance",
        "default": 0.15,
        "type": "float",
        "min": 0.05,
        "max": 0.40,
        "description": (
            "Allowed deviation from ls_net_exposure_pct before new entries are blocked. "
            "0.15 means entries blocked if net exposure would be outside [target-15%, target+15%]."
        ),
        "group": "Portfolio Manager",
    },
    {
        "key": "pm.ls_max_short_notional_pct",
        "default": 0.75,
        "type": "float",
        "min": 0.10,
        "max": 1.50,
        "description": "Hard cap on total short notional as fraction of NAV. 0.75 = 75% NAV max short.",
        "group": "Portfolio Manager",
    },
    # Trader / Strategy
    {
        "key": "strategy.partial_exit_pct",
        "default": 0.50,
        "type": "float",
        "min": 0.10,
        "max": 0.90,
        "description": "Fraction of position to exit at 1×ATR profit (partial exit)",
        "group": "Strategy",
    },
    {
        "key": "strategy.max_hold_bars",
        "default": 20,
        "type": "int",
        "min": 5,
        "max": 60,
        "description": "Safety-net max hold in daily bars before forced exit",
        "group": "Strategy",
    },
    # Risk Intelligence (Phase 19)
    {
        "key": "risk.max_correlation",
        "default": 0.75,
        "type": "float",
        "min": 0.30,
        "max": 0.99,
        "description": "Max 60-day return correlation with any open position before rejecting entry",
        "group": "Risk Manager",
    },
    {
        "key": "risk.max_portfolio_beta",
        "default": 1.30,
        "type": "float",
        "min": 0.50,
        "max": 3.00,
        "description": "Max portfolio beta vs SPY before blocking high-beta new entries",
        "group": "Risk Manager",
    },
    {
        "key": "risk.high_beta_threshold",
        "default": 1.20,
        "type": "float",
        "min": 0.50,
        "max": 2.50,
        "description": "Beta above which a new position is considered high-beta",
        "group": "Risk Manager",
    },
    {
        "key": "risk.max_factor_concentration",
        "default": 0.60,
        "type": "float",
        "min": 0.20,
        "max": 0.90,
        "description": "Max fraction of portfolio capital in the same sector/factor",
        "group": "Risk Manager",
    },
    # Execution quality
    {
        "key": "risk.max_spread_pct",
        "default": 0.005,
        "type": "float",
        "min": 0.0005,
        "max": 0.02,
        "description": "Max bid-ask spread as fraction of mid-price before rejecting entry",
        "group": "Risk Manager",
    },
    {
        "key": "risk.max_adtv_pct",
        "default": 0.01,
        "type": "float",
        "min": 0.001,
        "max": 0.05,
        "description": "Max trade cost as fraction of 20-day ADTV (liquidity gate)",
        "group": "Risk Manager",
    },
    {
        "key": "risk.max_portfolio_heat_pct",
        "default": 0.06,
        "type": "float",
        "min": 0.02,
        "max": 0.20,
        "description": "Max total portfolio heat (sum of position risks) as fraction of account value",
        "group": "Risk Manager",
    },
    {
        "key": "risk.normal_volatility_atr_ratio",
        "default": 0.02,
        "type": "float",
        "min": 0.005,
        "max": 0.10,
        "description": "Base ATR/price ratio used when computing adaptive stop-loss distance",
        "group": "Risk Manager",
    },
    {
        "key": "risk.stop_loss_base_pct",
        "default": 0.02,
        "type": "float",
        "min": 0.005,
        "max": 0.10,
        "description": "Default stop-loss percentage at normal volatility",
        "group": "Risk Manager",
    },
    {
        "key": "strategy.limit_order_offset_pct",
        "default": 0.001,
        "type": "float",
        "min": 0.0001,
        "max": 0.01,
        "description": "Limit order placed this % below ask for swing entries (10bps default)",
        "group": "Strategy",
    },
    {
        "key": "strategy.limit_order_requote_age_minutes",
        "default": 30,
        "type": "int",
        "min": 5,
        "max": 120,
        "description": "Re-quote a swing limit order if it has been unfilled for this many minutes",
        "group": "Strategy",
    },
    {
        "key": "strategy.limit_order_requote_drift_bps",
        "default": 20.0,
        "type": "float",
        "min": 5.0,
        "max": 100.0,
        "description": "Re-quote a swing limit order if the ask has drifted more than this many bps from our limit",
        "group": "Strategy",
    },
    {
        "key": "strategy.limit_order_max_requotes",
        "default": 3,
        "type": "int",
        "min": 0,
        "max": 10,
        "description": "Maximum number of times a swing limit order may be re-quoted before falling through to escalation/cancel",
        "group": "Strategy",
    },
    {
        "key": "strategy.limit_order_eod_escalation_hour",
        "default": 15,
        "type": "int",
        "min": 14,
        "max": 15,
        "description": "Hour (ET, 24h) at which unfilled swing limits escalate to a marketable limit",
        "group": "Strategy",
    },
    {
        "key": "strategy.limit_order_eod_escalation_minute",
        "default": 15,
        "type": "int",
        "min": 0,
        "max": 59,
        "description": "Minute (ET) at which unfilled swing limits escalate to a marketable limit",
        "group": "Strategy",
    },
    {
        "key": "strategy.limit_order_cancel_hour",
        "default": 15,
        "type": "int",
        "min": 14,
        "max": 16,
        "description": "Hour (ET) after which unfilled swing limit orders are cancelled outright",
        "group": "Strategy",
    },
    {
        "key": "strategy.limit_order_cancel_minute",
        "default": 45,
        "type": "int",
        "min": 0,
        "max": 59,
        "description": "Minute (ET) after which unfilled swing limit orders are cancelled outright",
        "group": "Strategy",
    },
    {
        "key": "strategy.ema_fast",
        "default": 20,
        "type": "int",
        "min": 5,
        "max": 50,
        "description": "Fast EMA period for crossover signal",
        "group": "Strategy",
    },
    {
        "key": "strategy.ema_slow",
        "default": 50,
        "type": "int",
        "min": 20,
        "max": 200,
        "description": "Slow EMA period for crossover signal",
        "group": "Strategy",
    },
    {
        "key": "strategy.rsi_period",
        "default": 14,
        "type": "int",
        "min": 5,
        "max": 30,
        "description": "RSI calculation period",
        "group": "Strategy",
    },
    {
        "key": "strategy.rsi_dip_entry",
        "default": 45,
        "type": "int",
        "min": 20,
        "max": 60,
        "description": "RSI level that triggers dip-and-recovery entry",
        "group": "Strategy",
    },
    {
        "key": "strategy.atr_stop_mult",
        "default": 2.5,
        "type": "float",
        "min": 1.0,
        "max": 5.0,
        "description": "ATR multiplier for stop-loss placement",
        "group": "Strategy",
    },
    {
        "key": "strategy.atr_target_mult",
        "default": 4.0,
        "type": "float",
        "min": 1.5,
        "max": 10.0,
        "description": "ATR multiplier for profit-target placement",
        "group": "Strategy",
    },
    {
        "key": "strategy.trail_activation_pct",
        "default": 0.04,
        "type": "float",
        "min": 0.01,
        "max": 0.15,
        "description": "% gain before trailing stop activates",
        "group": "Strategy",
    },
    {
        "key": "strategy.trail_pct",
        "default": 0.03,
        "type": "float",
        "min": 0.01,
        "max": 0.10,
        "description": "Trailing stop distance below highest close",
        "group": "Strategy",
    },
    # Reconciliation
    {
        "key": "reconcile.ghost_min_age_minutes",
        "default": 5,
        "type": "int",
        "min": 1,
        "max": 60,
        "description": "Minimum trade age in minutes before it can be marked as a ghost",
        "group": "Reconciliation",
    },
    {
        "key": "reconcile.interval_minutes",
        "default": 5,
        "type": "int",
        "min": 1,
        "max": 30,
        "description": "Frequency of periodic position reconciliation during market hours",
        "group": "Reconciliation",
    },
]

_DEFAULTS: Dict[str, Any] = {s["key"]: s["default"] for s in CONFIG_SCHEMA}


def get_agent_config(db, key: str) -> Any:
    """
    Read one config value from DB, falling back to the schema default.
    Coerces the stored value to the declared type.
    """
    from app.database.config_store import get_config
    schema = next((s for s in CONFIG_SCHEMA if s["key"] == key), None)
    raw = get_config(db, f"agent.{key}")
    if raw is None:
        return _DEFAULTS.get(key)
    try:
        if schema and schema["type"] == "int":
            return int(raw)
        if schema and schema["type"] == "float":
            return float(raw)
        return raw
    except (TypeError, ValueError):
        return _DEFAULTS.get(key)


def get_all_agent_config(db) -> Dict[str, Any]:
    """Return all config values as a flat dict (DB values override defaults)."""
    return {s["key"]: get_agent_config(db, s["key"]) for s in CONFIG_SCHEMA}


def set_agent_config(db, key: str, value: Any) -> None:
    """Write one config value, validating against schema bounds."""
    schema = next((s for s in CONFIG_SCHEMA if s["key"] == key), None)
    if schema is None:
        raise ValueError(f"Unknown config key: {key}")

    # Coerce + range-check
    try:
        if schema["type"] == "int":
            value = int(value)
        elif schema["type"] == "float":
            value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid value for {key}: {exc}") from exc

    if "min" in schema and value < schema["min"]:
        raise ValueError(f"{key} must be >= {schema['min']}")
    if "max" in schema and value > schema["max"]:
        raise ValueError(f"{key} must be <= {schema['max']}")

    from app.database.config_store import set_config
    set_config(db, f"agent.{key}", value, description=schema.get("description", ""))
    logger.info("Agent config updated: %s = %s", key, value)
