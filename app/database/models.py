from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON, Text, Boolean
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
    status = Column(String(20), default="PENDING")  # PENDING_FILL, ACTIVE, CLOSED, REJECTED, CANCELLED, RECONCILE_GHOST
    pnl = Column(Float, nullable=True)
    signal_type = Column(String(20), nullable=True)   # EMA_CROSSOVER | RSI_DIP | ML_RANK | RECONCILED
    trade_type = Column(String(20), nullable=True, default="swing")  # swing | intraday
    stop_price = Column(Float, nullable=True)         # initial ATR stop
    target_price = Column(Float, nullable=True)       # ATR profit target
    highest_price = Column(Float, nullable=True)      # for trailing stop tracking
    bars_held = Column(Integer, default=0)            # bars in position
    alpaca_order_id = Column(String(50), nullable=True, index=True)   # Alpaca order UUID; set after order placed
    proposal_id = Column(String(36), nullable=True, index=True)       # PM-generated UUID; threads through full chain
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
    # Back-filled by EOD script
    outcome_pnl_pct = Column(Float, nullable=True)      # realized P&L % if entered
    outcome_4h_pct = Column(Float, nullable=True)       # price change 4h after decision
    outcome_1d_pct = Column(Float, nullable=True)       # price change 1 day after decision


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


class ProcessHeartbeat(Base):
    """Phase 83 — PM heartbeat for deadman watchdog."""
    __tablename__ = "process_heartbeat"

    id = Column(Integer, primary_key=True)
    process_name = Column(String(50), nullable=False, unique=True, index=True)
    last_beat = Column(DateTime, nullable=False, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=False, default=datetime.utcnow)
