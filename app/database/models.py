from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON, Text
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
    status = Column(String(20), default="PENDING")  # PENDING, ACTIVE, CLOSED, REJECTED
    pnl = Column(Float, nullable=True)
    signal_type = Column(String(20), nullable=True)   # EMA_CROSSOVER | RSI_DIP | NONE
    stop_price = Column(Float, nullable=True)         # initial ATR stop
    target_price = Column(Float, nullable=True)       # ATR profit target
    highest_price = Column(Float, nullable=True)      # for trailing stop tracking
    bars_held = Column(Integer, default=0)            # bars in position
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
