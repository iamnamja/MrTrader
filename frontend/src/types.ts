export interface Summary {
  portfolio_value?: number
  equity?: number
  pnl_today?: number
  pnl_today_pct?: number
  buying_power?: number
  open_positions?: number
  win_rate?: number
  max_drawdown_pct?: number
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

export interface MacroIndicators {
  fed_funds_rate: number | null
  yield_10y: number | null
  yield_spread_10y2y: number | null
  cpi_yoy: number | null
  unemployment_rate: number | null
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
