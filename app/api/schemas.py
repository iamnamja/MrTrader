"""
Pydantic schemas for dashboard API responses.
"""

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel


class PositionResponse(BaseModel):
    symbol: str
    quantity: int
    avg_price: float
    current_price: Optional[float] = None
    pnl_unrealized: Optional[float] = None
    pnl_unrealized_pct: Optional[float] = None


class TradeResponse(BaseModel):
    id: int
    symbol: str
    direction: str
    entry_price: float
    exit_price: Optional[float] = None
    quantity: int
    pnl: Optional[float] = None
    status: str
    signal_type: Optional[str] = None
    stop_price: Optional[float] = None
    target_price: Optional[float] = None
    created_at: datetime
    closed_at: Optional[datetime] = None


class AgentDecisionResponse(BaseModel):
    id: int
    agent_name: str
    decision_type: str
    trade_id: Optional[int] = None
    reasoning: Optional[Dict[str, Any]] = None
    timestamp: datetime
    symbol: Optional[str] = None


class DashboardSummaryResponse(BaseModel):
    timestamp: datetime
    account_value: Optional[float] = None
    buying_power: Optional[float] = None
    cash: Optional[float] = None
    daily_pnl: float = 0.0
    daily_pnl_pct: Optional[float] = None
    total_pnl: Optional[float] = None
    total_pnl_pct: Optional[float] = None
    open_positions_count: int = 0
    trades_today_count: int = 0
    trading_mode: str
    system_status: str
    win_rate: Optional[float] = None
    max_drawdown_pct: Optional[float] = None


class SystemHealthResponse(BaseModel):
    database: str
    redis: str
    alpaca: str
    overall: str
    timestamp: str


class JobResponse(BaseModel):
    id: str
    name: str
    next_run_time: Optional[str] = None


class DailyMetricResponse(BaseModel):
    date: str
    daily_pnl: Optional[float] = None
    max_drawdown: Optional[float] = None
