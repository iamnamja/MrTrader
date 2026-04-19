export interface Summary {
  // Current API field names
  account_value?: number
  daily_pnl?: number
  daily_pnl_pct?: number
  open_positions_count?: number
  win_rate?: number
  max_drawdown_pct?: number
  buying_power?: number
  cash?: number
  total_pnl?: number
  total_pnl_pct?: number
  trades_today_count?: number
  trading_mode?: string
  system_status?: string
  timestamp?: string
  // Legacy / WS-push field names (keep for backwards compat)
  portfolio_value?: number
  equity?: number
  pnl_today?: number
  pnl_today_pct?: number
  open_positions?: number
}

export interface Health {
  status?: string
  system_status?: string
  trading_mode?: string
  mode?: string
  checks?: Record<string, boolean | string>
  database_ok?: boolean
  redis_ok?: boolean
  alpaca_ok?: boolean
}

export interface Position {
  symbol: string
  qty: number
  avg_entry_price?: number
  current_price?: number
  market_value?: number
  unrealized_pl?: number
  unrealized_plpc?: number
}

export interface Trade {
  symbol: string
  direction: string
  signal_type?: string
  entry_price?: number
  exit_price?: number
  quantity: number
  pnl?: number
  status: string
  created_at?: string
}

export interface Decision {
  timestamp?: string
  agent_name?: string
  decision_type?: string
  symbol?: string
  reasoning?: { symbol?: string }
}

export interface RampStage {
  stage: number
  capital: number
}

export interface LiveStatus {
  stage?: number
  capital?: number
  days_elapsed?: number
  stages?: RampStage[]
  kill_switch_active?: boolean
}

export interface AuditEntry {
  timestamp?: string
  action: string
  details?: Record<string, unknown>
}

export interface WsMessage {
  type: string
  data: Record<string, unknown>
}

export interface ReadinessCheckItem {
  check: string
  passed: boolean
  value: unknown
  detail: string
}

export interface ReadinessReport {
  ready: boolean
  timestamp: string
  summary: string
  blockers: ReadinessCheckItem[]
  warnings: ReadinessCheckItem[]
  passed: ReadinessCheckItem[]
  all_checks: ReadinessCheckItem[]
}

export interface AttributionItem {
  signal_type: string
  count: number
  wins: number
  total_pnl: number
  win_rate: number
  avg_pnl: number
}

export interface MarketStatus {
  is_open: boolean
  current_time_et: string
  weekday: string
  next_event: { event: string; minutes?: number; date?: string; time?: string }
}

export interface OrchestratorStatus {
  running: boolean
  agents: Record<string, string>
  scheduled_jobs: number
  queues: Record<string, number>
  market?: MarketStatus
}

export interface ScheduledJob {
  id: string
  name: string
  next_run_time: string | null
  paused: boolean
}

export interface SessionLogEntry {
  timestamp: string
  level: string
  message: string
  detail: Record<string, unknown>
}

export interface ConfigSchemaEntry {
  key: string
  default: number
  type: 'int' | 'float'
  min: number
  max: number
  description: string
  group: string
}

export interface DriftItem {
  metric: string
  live: number
  target: number
  delta: number
  pct_diff: number
  status: 'ok' | 'warn' | 'alert'
}

export interface PerformanceReview {
  period_days: number
  start_date: string
  end_date: string
  total_trades: number
  wins: number
  win_rate_pct: number
  total_pnl: number
  avg_pnl_per_trade: number
  sharpe_estimate: number | null
  spy_return_pct: number
  alpha_pct: number | null
  by_signal: Record<string, { trades: number; wins: number; total_pnl: number; win_rate: number; avg_pnl: number }>
  backtest_targets: Record<string, number>
  drift: DriftItem[]
  alerts: number
  warnings: number
  overall_status: 'ok' | 'warn' | 'alert'
}

export interface MacroIndicators {
  fed_funds_rate: number | null
  yield_10y: number | null
  yield_spread_10y2y: number | null
  cpi_yoy: number | null
  unemployment_rate: number | null
}

export interface WatchlistTicker {
  id: number
  symbol: string
  sector: string | null
  notes: string | null
  active: boolean
  added_at: string | null
}

export interface RegimeDetail {
  regime: string
  composite_score: number
  vix: number | null
  vix_score: number
  macro_score: number
  vix_weight: number
  macro_weight: number
  macro_indicators: MacroIndicators
  trend_following_active: boolean
  mean_reversion_active: boolean
  position_size_multiplier: number
}

export interface MonitorHealth {
  timestamp: string
  alpaca_connected: boolean
  account_value: number
  buying_power: number
  cash: number
  open_positions: number
  trades_today: number
  pnl_today: number
  pnl_today_pct: number
  max_drawdown_pct: number
  status: 'healthy' | 'warning' | 'critical'
  consecutive_losing_days: number
}

export interface DailySummary {
  date: string
  timestamp: string
  trades_today: number
  pnl_today: number
  pnl_today_pct: number
  account_value: number
  max_drawdown_pct: number
  status: string
  consecutive_losing_days: number
  open_positions: number
}
