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
