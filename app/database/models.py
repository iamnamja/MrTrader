from sqlalchemy import Column, Date, Integer, String, Float, DateTime, ForeignKey, JSON, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class Trade(Base):
    """Trade model - represents a trading transaction"""
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), index=True, nullable=False)
    direction = Column(String(10), nullable=False)  # BUY or SELL
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    quantity = Column(Integer, nullable=False)
    # PENDING_FILL | ACTIVE | CLOSED | REJECTED | CANCELLED
    # RECONCILE_GHOST | RECONCILE_SUPERSEDED | FORCE_CLOSED_NO_POSITION
    status = Column(String(30), default="PENDING")
    status_reason = Column(String(255), nullable=True)   # human-readable note on current status
    pnl = Column(Float, nullable=True)
    signal_type = Column(String(20), nullable=True)   # EMA_CROSSOVER | RSI_DIP | ML_RANK | RECONCILED
    trade_type = Column(String(20), nullable=True, default="swing")  # swing | intraday
    stop_price = Column(Float, nullable=True)         # initial ATR stop
    target_price = Column(Float, nullable=True)       # ATR profit target
    highest_price = Column(Float, nullable=True)      # for trailing stop tracking
    bars_held = Column(Integer, default=0)            # bars in position
    alpaca_order_id = Column(String(50), nullable=True, index=True)   # Alpaca order UUID; set after order placed
    proposal_id = Column(String(36), nullable=True, index=True)       # PM-generated UUID; threads through full chain
    # stop_hit | target_hit | time_exit | pm_review | news_exit | manual | kill_switch
    # | eod_intraday | reconcile_ghost_expired | reconcile_superseded
    exit_reason = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    closed_at = Column(DateTime, nullable=True)

    # Relationships
    orders = relationship("Order", back_populates="trade")
    decisions = relationship("AgentDecision", back_populates="trade")

    def __repr__(self):
        return f"<Trade {self.symbol} {self.direction} {self.quantity}@${self.entry_price}>"


class Order(Base):
    """Order model - represents orders placed with broker"""
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, index=True)
    trade_id = Column(Integer, ForeignKey("trades.id"), index=True, nullable=False)
    order_type = Column(String(20), nullable=False)  # ENTRY or EXIT
    order_id = Column(String(50), nullable=True)  # Alpaca order ID
    status = Column(String(20), default="PENDING")  # PENDING, FILLED, FAILED
    filled_price = Column(Float, nullable=True)
    filled_qty = Column(Integer, nullable=True)
    intended_price = Column(Float, nullable=True)   # price at signal time (pre-execution)
    slippage_bps = Column(Float, nullable=True)     # (filled - intended) / intended * 10000
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    # Relationships
    trade = relationship("Trade", back_populates="orders")

    def __repr__(self):
        return f"<Order {self.order_type} {self.status} {self.timestamp}>"


class Position(Base):
    """Position model - represents current open positions"""
    __tablename__ = "positions"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), index=True, nullable=False, unique=True)
    quantity = Column(Integer, nullable=False)
    avg_price = Column(Float, nullable=False)
    current_price = Column(Float, nullable=True)
    pnl_unrealized = Column(Float, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<Position {self.symbol} x{self.quantity} @ ${self.avg_price}>"


class AgentDecision(Base):
    """AgentDecision model - audit trail of agent decisions"""
    __tablename__ = "agent_decisions"

    id = Column(Integer, primary_key=True, index=True)
    agent_name = Column(String(50), nullable=False)  # portfolio_manager, risk_manager, trader
    decision_type = Column(String(50), nullable=False)  # TRADE_PROPOSAL, TRADE_APPROVED, TRADE_REJECTED, EXIT_SIGNAL
    trade_id = Column(Integer, ForeignKey("trades.id"), nullable=True, index=True)
    reasoning = Column(JSON, nullable=True)  # Why this decision?
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    # Relationships
    trade = relationship("Trade", back_populates="decisions")

    def __repr__(self):
        return f"<AgentDecision {self.agent_name} {self.decision_type} {self.timestamp}>"


class RiskMetric(Base):
    """RiskMetric model - daily risk metrics tracking"""
    __tablename__ = "risk_metrics"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(String(10), nullable=False, unique=True, index=True)  # YYYY-MM-DD
    daily_pnl = Column(Float, nullable=True)
    account_pnl = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)
    position_concentration = Column(JSON, nullable=True)  # {symbol: pct}
    sector_concentration = Column(JSON, nullable=True)  # {sector: pct}
    timestamp = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<RiskMetric {self.date} pnl={self.daily_pnl}>"


class ModelVersion(Base):
    """ModelVersion model - track ML model versions and performance"""
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(50), nullable=False)  # portfolio_selector
    version = Column(Integer, nullable=False)  # 1, 2, 3, ...
    training_date = Column(DateTime, default=datetime.utcnow, index=True)
    data_range_start = Column(String(10), nullable=True)  # YYYY-MM-DD
    data_range_end = Column(String(10), nullable=True)  # YYYY-MM-DD
    performance = Column(JSON, nullable=True)  # {sharpe: 1.5, accuracy: 0.65, ...}
    status = Column(String(20), default="ACTIVE")  # TRAINING, ACTIVE, ARCHIVED
    model_path = Column(String(255), nullable=True)  # Path to saved model file

    def __repr__(self):
        return f"<ModelVersion {self.model_name} v{self.version} {self.status}>"


class AuditLog(Base):
    """AuditLog model - comprehensive audit trail"""
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    action = Column(String(100), nullable=False)  # TRADE_EXECUTED, POSITION_CLOSED, ALERT_SENT, etc
    details = Column(JSON, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self):
        return f"<AuditLog {self.action} {self.timestamp}>"


class Configuration(Base):
    """Configuration model - store runtime configuration"""
    __tablename__ = "configuration"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(100), unique=True, nullable=False, index=True)
    value = Column(Text, nullable=False)
    description = Column(String(255), nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<Configuration {self.key}={self.value}>"


class TradeProposal(Base):
    """Persisted record of every PM proposal received by the Risk Manager."""
    __tablename__ = "trade_proposals"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), index=True, nullable=False)
    trade_type = Column(String(20), nullable=False)          # "swing" | "intraday"
    direction = Column(String(10), nullable=False, default="BUY")
    entry_price = Column(Float, nullable=True)
    confidence = Column(Float, nullable=True)
    ml_score = Column(Float, nullable=True)
    status = Column(String(20), nullable=False, default="PENDING")  # PENDING | APPROVED | REJECTED
    reject_reason = Column(String(255), nullable=True)
    source_agent = Column(String(50), nullable=True)
    trade_id = Column(Integer, ForeignKey("trades.id"), nullable=True)  # set when executed
    proposed_at = Column(DateTime, default=datetime.utcnow, index=True)
    decided_at = Column(DateTime, nullable=True)

    def __repr__(self):
        return f"<TradeProposal {self.symbol} {self.trade_type} {self.status}>"


class WatchlistTicker(Base):
    """Dynamic ticker universe for portfolio manager selection."""
    __tablename__ = "watchlist_tickers"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), unique=True, nullable=False, index=True)
    sector = Column(String(50), nullable=True)
    notes = Column(String(255), nullable=True)
    active = Column(Integer, default=1)  # 1=active, 0=disabled
    added_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<WatchlistTicker {self.symbol} active={self.active}>"


class TradingSession(Base):
    """TradingSession model - track trading sessions and approval"""
    __tablename__ = "trading_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_date = Column(String(10), unique=True, nullable=False, index=True)  # YYYY-MM-DD
    mode = Column(String(20), default="paper")  # paper or live
    approved_at = Column(DateTime, nullable=True)
    approved_by = Column(String(100), nullable=True)
    capital = Column(Float, nullable=False)
    daily_pnl = Column(Float, nullable=True)
    session_pnl = Column(Float, nullable=True)
    trades_count = Column(Integer, default=0)
    status = Column(String(20), default="ACTIVE")  # ACTIVE, CLOSED, PAUSED
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)

    def __repr__(self):
        return f"<TradingSession {self.session_date} {self.mode}>"


# ── News Intelligence Service (NIS) tables ────────────────────────────────────

class CalendarEvent(Base):
    """Economic + earnings calendar events with consensus data from Finnhub."""
    __tablename__ = "calendar_events"

    id = Column(String(64), primary_key=True)           # sha256(event_type+event_time)
    event_type = Column(String(50), nullable=False)     # FOMC, NFP, CPI, EARNINGS, etc.
    symbol = Column(String(10), nullable=True)          # null for macro events
    event_time = Column(DateTime, nullable=False, index=True)
    importance = Column(String(10), nullable=False)     # low / medium / high
    source = Column(String(20), nullable=False)         # finnhub / polygon / manual
    estimate = Column(Float, nullable=True)             # consensus estimate
    prior = Column(Float, nullable=True)
    actual = Column(Float, nullable=True)
    currency = Column(String(10), nullable=True)
    country = Column(String(5), nullable=True)
    fetched_at = Column(DateTime, default=datetime.utcnow)
    payload = Column(JSON, nullable=True)               # raw Finnhub response


class MacroSignalCache(Base):
    """Cached Tier 1 macro LLM classifications (keyed by date)."""
    __tablename__ = "macro_signal_cache"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(String(10), nullable=False, index=True)   # YYYY-MM-DD
    prompt_version = Column(String(20), nullable=False)
    risk_level = Column(String(10), nullable=False)         # LOW / MEDIUM / HIGH
    direction = Column(String(10), nullable=False)
    sizing_factor = Column(Float, nullable=False)
    block_new_entries = Column(Boolean, nullable=False, default=False)
    rationale = Column(Text, nullable=True)
    events_payload = Column(JSON, nullable=True)            # events that drove this
    evaluated_at = Column(DateTime, default=datetime.utcnow)


class NewsSignalCache(Base):
    """Cached Tier 2 per-symbol LLM news scores."""
    __tablename__ = "news_signal_cache"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False, index=True)
    cache_key = Column(String(64), nullable=False, unique=True)  # sha256(prompt_ver+headlines)
    prompt_version = Column(String(20), nullable=False)
    direction_score = Column(Float, nullable=False)
    materiality_score = Column(Float, nullable=False)
    downside_risk_score = Column(Float, nullable=False)
    upside_catalyst_score = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    already_priced_in_score = Column(Float, nullable=False)
    action_policy = Column(String(30), nullable=False)
    sizing_multiplier = Column(Float, nullable=False)
    rationale = Column(Text, nullable=True)
    top_headlines = Column(JSON, nullable=True)
    evaluated_at = Column(DateTime, default=datetime.utcnow, index=True)
    as_of_date = Column(String(10), nullable=True, index=True)  # Phase 64: YYYY-MM-DD for point-in-time backfill lookup


class LLMCallLog(Base):
    """Audit log for every LLM call made by the NIS."""
    __tablename__ = "llm_call_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    called_at = Column(DateTime, default=datetime.utcnow, index=True)
    call_type = Column(String(20), nullable=False)      # macro_tier1 / stock_tier2
    symbol = Column(String(10), nullable=True)
    provider = Column(String(20), nullable=False, default="anthropic")
    model_name = Column(String(50), nullable=False)
    prompt_version = Column(String(20), nullable=False)
    input_tokens = Column(Integer, nullable=False)
    output_tokens = Column(Integer, nullable=False)
    latency_ms = Column(Integer, nullable=False)
    cost_usd = Column(Float, nullable=False)
    cache_hit = Column(Boolean, nullable=False, default=False)
    error = Column(Text, nullable=True)


class DecisionAudit(Base):
    """
    Structured record of every PM entry/block decision.
    Filled in at decision time; EOD script back-fills realized outcome columns.
    Query example: SELECT block_reason, AVG(outcome_pnl_pct), COUNT(*)
                   FROM decision_audit WHERE final_decision='block'
                   GROUP BY block_reason
    """
    __tablename__ = "decision_audit"

    id = Column(String(36), primary_key=True)          # UUID
    decided_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    symbol = Column(String(10), nullable=False, index=True)
    strategy = Column(String(20), nullable=False)       # 'swing' | 'intraday'
    model_score = Column(Float, nullable=True)          # ML confidence at decision time
    news_action_policy = Column(String(30), nullable=True)   # NIS Tier 2 action_policy
    news_direction_score = Column(Float, nullable=True)
    news_materiality = Column(Float, nullable=True)
    news_sizing_multiplier = Column(Float, nullable=True)
    news_rationale = Column(Text, nullable=True)
    macro_risk_level = Column(String(10), nullable=True)     # NIS Tier 1 risk level
    macro_sizing_factor = Column(Float, nullable=True)
    final_decision = Column(String(20), nullable=False)  # 'enter'|'block'|'size_down'|'exit_review'
    size_multiplier = Column(Float, nullable=False, default=1.0)
    block_reason = Column(String(255), nullable=True)
    # Top model features at decision time (JSON: {feature_name: value, ...} sorted by importance)
    top_features = Column(JSON, nullable=True)
    # Gate classification — used for calibration reporting
    gate_category = Column(String(20), nullable=True, index=True)  # alpha|quality|risk|structural|scan
    # Stock price at decision time — anchor for counterfactual P&L calculation
    price_at_decision = Column(Float, nullable=True)
    # Intended direction (BUY/SELL) — needed to compute signed counterfactual P&L
    direction = Column(String(5), nullable=True)
    # Back-filled by EOD script (gate_outcomes_backfill job)
    outcome_pnl_pct = Column(Float, nullable=True)      # realized P&L % if entered (APPROVED trades)
    outcome_4h_pct = Column(Float, nullable=True)       # counterfactual: price change 4h after decision
    outcome_1d_pct = Column(Float, nullable=True)       # counterfactual: price change 1d after decision
    outcome_fetched_at = Column(DateTime, nullable=True)  # when backfill last ran for this row


class PendingLimitOrder(Base):
    """Phase 78b — persisted swing limit orders so they survive restarts."""
    __tablename__ = "pending_limit_orders"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False, unique=True, index=True)
    order_id = Column(String(50), nullable=False, index=True)
    trade_id = Column(Integer, ForeignKey("trades.id"), nullable=True, index=True)
    shares = Column(Integer, nullable=False)
    limit_price = Column(Float, nullable=False)
    intended_price = Column(Float, nullable=False)
    stop_price = Column(Float, nullable=True)
    target_price = Column(Float, nullable=True)
    atr = Column(Float, nullable=True)
    trade_type = Column(String(20), nullable=False, default="swing")
    signal_type = Column(String(30), nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class SwingProposalLog(Base):
    """PM-generated swing proposals — persisted at scan time, survives restarts.

    One row per symbol per scan batch. Tracks the full scoring context so
    you can reconstruct why each symbol was proposed or skipped.
    """
    __tablename__ = "swing_proposal_log"

    id = Column(Integer, primary_key=True, index=True)

    # Batch identity — all rows from one pre-market scan share this
    batch_id = Column(String(40), nullable=False, index=True)   # ISO date + scan label e.g. "2026-05-05_premarket"
    scan_time = Column(DateTime, nullable=False, default=datetime.utcnow)
    scan_label = Column(String(30), nullable=False, default="premarket")  # "premarket" | "30min" | "rescan"

    # Symbol + ranking
    symbol = Column(String(10), nullable=False, index=True)
    rank = Column(Integer, nullable=True)           # rank within batch (1 = highest score)
    ml_score = Column(Float, nullable=True)         # raw model confidence

    # Proposal details
    direction = Column(String(5), nullable=False, default="BUY")
    entry_price = Column(Float, nullable=True)
    stop_price = Column(Float, nullable=True)
    target_price = Column(Float, nullable=True)
    quantity = Column(Integer, nullable=True)
    confidence = Column(Float, nullable=True)       # final confidence (after NIS overlay)
    sector = Column(String(50), nullable=True)

    # Market context at scan time
    vix_at_scan = Column(Float, nullable=True)
    spy_price_at_scan = Column(Float, nullable=True)
    spy_ma20_at_scan = Column(Float, nullable=True)
    spy_5d_return_at_scan = Column(Float, nullable=True)
    opportunity_score = Column(Float, nullable=True)  # Phase 88 composite score

    # Gate results (JSON): each gate that passed/blocked this proposal
    gate_results = Column(JSON, nullable=True)      # {"earnings": "pass", "macro": "block: FOMC", ...}
    nis_signal = Column(JSON, nullable=True)        # NIS action_policy, scores, rationale
    ai_review = Column(JSON, nullable=True)         # Claude AI review if run

    # Status lifecycle
    status = Column(String(20), nullable=False, default="PENDING", index=True)
    # PENDING → SENT (forwarded to RM) | BLOCKED (gate/NIS blocked before RM)
    # EXPIRED (not sent — restart/missed window) | SUPPRESSED (opportunity score low)
    status_reason = Column(String(255), nullable=True)

    # RM outcome (filled in after RM decides)
    rm_status = Column(String(20), nullable=True)   # APPROVED | REJECTED | None (not yet sent)
    rm_reason = Column(String(255), nullable=True)
    rm_decided_at = Column(DateTime, nullable=True)

    # Trade linkage
    trade_id = Column(Integer, ForeignKey("trades.id"), nullable=True)

    proposed_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    sent_at = Column(DateTime, nullable=True)

    def __repr__(self):
        return f"<SwingProposalLog {self.symbol} {self.scan_label} {self.status}>"


class IntraProposalLog(Base):
    """PM-generated intraday scan results — one row per symbol per scan window.

    Written at scoring time so you can see what PM considered even if gates blocked
    the entire scan before proposals were forwarded to RM.
    """
    __tablename__ = "intra_proposal_log"

    id = Column(Integer, primary_key=True, index=True)

    scan_time = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    window = Column(String(10), nullable=False)  # "09:45", "10:45", "13:00", "12:12" etc.

    symbol = Column(String(10), nullable=False, index=True)
    rank = Column(Integer, nullable=True)
    ml_score = Column(Float, nullable=True)
    above_threshold = Column(Boolean, nullable=False, default=False)

    # Scan-level gate block (1A/1B/1C) — applies to all symbols in this window
    scan_gate_block = Column(String(100), nullable=True)  # None = not blocked; else reason

    # Per-symbol proposal outcome (only set for symbols that survived scan gates)
    entry_price = Column(Float, nullable=True)
    stop_price = Column(Float, nullable=True)
    target_price = Column(Float, nullable=True)
    quantity = Column(Integer, nullable=True)
    status = Column(String(20), nullable=False, default="SCORED")
    # SCORED | SENT | ENTRY_GATE_BLOCKED | NIS_BLOCKED | COOLDOWN

    nis_signal = Column(JSON, nullable=True)
    entry_gate_reason = Column(String(255), nullable=True)

    # RM outcome (filled after RM decides)
    rm_status = Column(String(20), nullable=True)
    rm_reason = Column(String(255), nullable=True)

    proposed_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    def __repr__(self):
        return f"<IntraProposalLog {self.symbol} {self.window} {self.status}>"


class ProposalLog(Base):
    """Unified PM proposal log — one row per symbol per scan, covers both swing and intraday.

    Lifecycle: PM writes at scan time (pm_status=SCORED/SENT/GATE_BLOCKED).
    RM writes back rm_status after approving/rejecting.
    Trader writes trade_id after placing the order.

    proposal_uuid is the shared key that flows through PM queue message -> RM -> Trader -> Trade.
    """
    __tablename__ = "proposal_log"

    id = Column(Integer, primary_key=True, index=True)

    # ── Identity ──────────────────────────────────────────────────────────────
    proposal_uuid = Column(String(36), nullable=True, index=True)  # PM-generated UUID; may be null for gate-blocked rows
    strategy = Column(String(10), nullable=False, index=True)       # 'swing' | 'intraday'
    batch_id = Column(String(50), nullable=True, index=True)        # groups all rows from one scan

    # ── Trigger context ───────────────────────────────────────────────────────
    triggered_by = Column(String(30), nullable=True)  # 'scheduled' | 'manual_ui' | 'adaptive_rescan' | 'pm_review'
    scan_time = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    scan_window = Column(String(20), nullable=True)   # '09:45' | 'premarket' | '13:00' etc.
    scan_label = Column(String(30), nullable=True)    # human label e.g. 'premarket', '30min_review'

    # ── Symbol ────────────────────────────────────────────────────────────────
    symbol = Column(String(10), nullable=False, index=True)
    rank = Column(Integer, nullable=True)
    sector = Column(String(50), nullable=True)

    # ── ML scoring ────────────────────────────────────────────────────────────
    ml_score = Column(Float, nullable=True)
    confidence = Column(Float, nullable=True)         # final confidence after any NIS adjustment
    above_threshold = Column(Boolean, nullable=True)
    model_version = Column(String(20), nullable=True)  # e.g. 'intraday_v29', 'swing_v38'
    top_features = Column(JSON, nullable=True)        # {feature: value} sorted by importance

    # ── Market context at scan time ───────────────────────────────────────────
    vix_at_scan = Column(Float, nullable=True)
    spy_price_at_scan = Column(Float, nullable=True)
    spy_5d_return_at_scan = Column(Float, nullable=True)
    spy_first_hour_range = Column(Float, nullable=True)
    opportunity_score = Column(Float, nullable=True)

    # ── Gate results ──────────────────────────────────────────────────────────
    # JSON: {"gate_1a": {"value": 0.27, "threshold": 0.20, "result": "pass"}, ...}
    gate_results = Column(JSON, nullable=True)
    scan_gate_block = Column(String(120), nullable=True)  # set when entire scan blocked before per-symbol step
    nis_signal = Column(JSON, nullable=True)              # full NIS Tier 2 response

    # ── PM outcome ────────────────────────────────────────────────────────────
    # SCORED | SENT | SCAN_GATE_BLOCKED | ENTRY_GATE_BLOCKED | NIS_BLOCKED | COOLDOWN | SUPPRESSED
    pm_status = Column(String(30), nullable=False, default="SCORED", index=True)
    pm_status_reason = Column(String(255), nullable=True)
    pm_decided_at = Column(DateTime, nullable=True)

    # ── Proposal details (set when pm_status=SENT) ───────────────────────────
    direction = Column(String(5), nullable=True, default="BUY")
    entry_price = Column(Float, nullable=True)
    stop_price = Column(Float, nullable=True)
    target_price = Column(Float, nullable=True)
    quantity = Column(Integer, nullable=True)

    # ── RM outcome (RM writes back using proposal_uuid) ───────────────────────
    # PENDING | APPROVED | REJECTED | WITHDRAWN
    rm_status = Column(String(20), nullable=True, index=True)
    rm_reason = Column(String(255), nullable=True)
    rm_rule = Column(String(50), nullable=True)       # which RM rule fired e.g. 'max_positions'
    rm_inputs = Column(JSON, nullable=True)           # portfolio state at RM decision time
    rm_decided_at = Column(DateTime, nullable=True)

    # ── Outcome back-fill (EOD script) ────────────────────────────────────────
    outcome_pnl_pct = Column(Float, nullable=True)    # realized P&L % if trade taken
    outcome_4h_pct = Column(Float, nullable=True)     # price change 4h after signal
    outcome_1d_pct = Column(Float, nullable=True)     # price change 1d after signal

    # ── Trade linkage ─────────────────────────────────────────────────────────
    trade_id = Column(Integer, ForeignKey("trades.id"), nullable=True)

    # ── Timestamps ────────────────────────────────────────────────────────────
    proposed_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    sent_to_rm_at = Column(DateTime, nullable=True)

    def __repr__(self):
        return f"<ProposalLog {self.strategy} {self.symbol} {self.pm_status}>"


class ProposalEvent(Base):
    """Append-only lineage log — one row per state change per proposal.

    Answers: what happened to this proposal, in what order, and why?
    Never updated — only inserted. Query by proposal_uuid to reconstruct timeline.
    """
    __tablename__ = "proposal_events"

    id = Column(Integer, primary_key=True, index=True)
    proposal_uuid = Column(String(36), nullable=False, index=True)
    event_time = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    # Who triggered this event
    # 'portfolio_manager' | 'risk_manager' | 'trader' | 'user' | 'reconciler'
    actor = Column(String(30), nullable=False)

    # SCORED | SCAN_GATE_BLOCKED | ENTRY_GATE_BLOCKED | NIS_BLOCKED | COOLDOWN
    # SENT_TO_RM | RM_APPROVED | RM_REJECTED | RM_WITHDRAWN
    # ORDER_PLACED | PARTIALLY_FILLED | FILLED
    # STOP_MOVED | PM_REVIEW_HOLD | PM_REVIEW_EXIT
    # TRADE_CLOSED | KILL_SWITCH | FORCE_CLOSED
    event_type = Column(String(40), nullable=False, index=True)

    # Free-form JSON — whatever is relevant to this specific event
    # e.g. STOP_MOVED: {old_stop, new_stop, reason, current_price}
    # e.g. RM_REJECTED: {rule, portfolio_heat, open_positions, limit}
    # e.g. TRADE_CLOSED: {exit_price, pnl, pnl_pct, exit_reason, bars_held}
    details = Column(JSON, nullable=True)

    def __repr__(self):
        return f"<ProposalEvent {self.proposal_uuid[:8]} {self.event_type} {self.event_time}>"


class NisMacroSnapshot(Base):
    """One row per premarket NIS macro run — survives server restarts.

    The in-memory premarket_intel.macro_context is lost on restart; this table
    lets the API fall back to the most recent snapshot so the UI always shows
    the last-known macro context.
    """
    __tablename__ = "nis_macro_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    snapshot_date = Column(Date, nullable=False, unique=True, index=True)  # one per calendar day
    as_of = Column(DateTime, nullable=False)
    overall_risk = Column(String(10), nullable=False)        # LOW | MEDIUM | HIGH
    block_new_entries = Column(Boolean, nullable=False, default=False)
    global_sizing_factor = Column(Float, nullable=False, default=1.0)
    rationale = Column(Text, nullable=True)
    events_json = Column(JSON, nullable=True)                # full events_today array
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class ScanAbstention(Base):
    """Scan-level gate abstentions — entire intraday scan skipped due to a market gate.

    Unlike per-symbol decision_audit rows, scan gates (gate1a SPY range, gate1c melt-up)
    have no target symbol.  We record SPY price at abstention time and back-fill
    SPY return over the session so we can evaluate whether abstaining was correct.
    """
    __tablename__ = "scan_abstentions"

    id = Column(Integer, primary_key=True, index=True)
    abstained_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    strategy = Column(String(20), nullable=False, default="intraday")
    gate_type = Column(String(50), nullable=False, index=True)  # gate1a_spy_range | gate1c_meltup
    gate_detail = Column(Text, nullable=True)                   # human-readable trigger detail
    # Link to the proposal_log batch that was killed
    proposal_log_batch_id = Column(String(40), nullable=True, index=True)
    # SPY price context
    spy_price_at_abstention = Column(Float, nullable=True)
    spy_first_hour_range_pct = Column(Float, nullable=True)     # gate1a diagnostic
    # Back-filled by gate_outcomes_backfill job
    spy_outcome_4h_pct = Column(Float, nullable=True)           # SPY return from abstention → +4h
    spy_outcome_1d_pct = Column(Float, nullable=True)           # SPY return for full session
    outcome_fetched_at = Column(DateTime, nullable=True)
    # good_abstention = SPY fell (gate was right); bad_abstention = SPY rose (missed gains)
    verdict = Column(String(20), nullable=True)


class ProcessHeartbeat(Base):
    """Phase 83 — PM heartbeat for deadman watchdog."""
    __tablename__ = "process_heartbeat"

    id = Column(Integer, primary_key=True)
    process_name = Column(String(50), nullable=False, unique=True, index=True)
    last_beat = Column(DateTime, nullable=False, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=False, default=datetime.utcnow)
