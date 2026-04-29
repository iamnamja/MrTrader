import { useEffect, useRef, useState, useCallback, Component } from 'react'
import type { ReactNode } from 'react'
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from 'recharts'
import { api } from './api'
import type {
  Summary, Health, Position, Trade, Decision, LiveStatus, AuditEntry, WsMessage,
  ReadinessReport, ReadinessCheckItem, AttributionItem,
  MarketStatus, OrchestratorStatus, ScheduledJob, SessionLogEntry,
  RegimeDetail, PerformanceReview, DriftItem, ConfigSchemaEntry,
  NisMacroContext, NisSignal, DecisionAuditRow, GateSummaryRow, DailySummaryRow,
} from './types'

// ── Colours / tokens ──────────────────────────────────────────────────────────
const C = {
  bg: '#0b0e14', surface: '#131720', surface2: '#1c2230', border: '#1e2d40',
  text: '#d1d5db', muted: '#6b7280', green: '#22c55e', red: '#ef4444',
  blue: '#3b82f6', yellow: '#eab308', accent: '#38bdf8',
}

// ── Inline style helpers ───────────────────────────────────────────────────────
const s = {
  card: {
    background: C.surface, border: `1px solid ${C.border}`,
    borderRadius: 6, padding: 16,
  } as React.CSSProperties,
  cardTitle: {
    fontSize: 11, textTransform: 'uppercase' as const, letterSpacing: '.06em',
    color: C.muted, marginBottom: 12,
  } as React.CSSProperties,
  kpi: {
    background: C.surface, border: `1px solid ${C.border}`,
    borderRadius: 6, padding: '14px 16px',
  } as React.CSSProperties,
  th: {
    color: C.muted, fontWeight: 500, textAlign: 'left' as const,
    padding: '6px 8px', borderBottom: `1px solid ${C.border}`,
    fontSize: 10, textTransform: 'uppercase' as const, letterSpacing: '.05em',
  } as React.CSSProperties,
  td: {
    padding: '8px 8px', borderBottom: `1px solid rgba(255,255,255,.04)`,
    fontSize: 12,
  } as React.CSSProperties,
}

// ── Formatters ────────────────────────────────────────────────────────────────
function fmt$(v: number | null | undefined) {
  if (v == null || isNaN(v)) return '—'
  return '$' + Number(v).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
}
function fmtPct(v: number | null | undefined) {
  if (v == null || isNaN(v)) return '—'
  return (v >= 0 ? '+' : '') + v.toFixed(2) + '%'
}
function fmtTs(iso: string | undefined) {
  if (!iso) return '—'
  // DB stores UTC without 'Z'; append it so JS parses as UTC, then display in ET
  const normalized = iso.endsWith('Z') || iso.includes('+') ? iso : iso + 'Z'
  const d = new Date(normalized)
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', timeZone: 'America/New_York' }) + ' ' +
    d.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true, timeZone: 'America/New_York' })
}
function clr(v: number | undefined | null) {
  if (v == null) return C.text
  return v > 0 ? C.green : v < 0 ? C.red : C.text
}

// ── Toast ─────────────────────────────────────────────────────────────────────
interface ToastMsg { id: number; text: string; type: 'success' | 'error' | 'warning' | 'info' }
let _toastId = 0
function useToasts() {
  const [toasts, setToasts] = useState<ToastMsg[]>([])
  const add = useCallback((text: string, type: ToastMsg['type'] = 'info') => {
    const id = ++_toastId
    setToasts(t => [...t, { id, text, type }])
    setTimeout(() => setToasts(t => t.filter(x => x.id !== id)), 3800)
  }, [])
  return { toasts, add }
}

function ToastContainer({ toasts }: { toasts: ToastMsg[] }) {
  const borderColor = (t: ToastMsg) =>
    t.type === 'success' ? C.green : t.type === 'error' ? C.red : t.type === 'warning' ? C.yellow : C.accent
  return (
    <div style={{ position: 'fixed', bottom: 20, right: 20, zIndex: 2000, display: 'flex', flexDirection: 'column', gap: 8 }}>
      {toasts.map(t => (
        <div key={t.id} style={{
          background: C.surface, border: `1px solid ${C.border}`,
          borderLeft: `3px solid ${borderColor(t)}`, borderRadius: 6,
          padding: '10px 16px', fontSize: 12, minWidth: 240, maxWidth: 360,
        }}>{t.text}</div>
      ))}
    </div>
  )
}

// ── KPI Card ──────────────────────────────────────────────────────────────────
function KpiCard({ label, value, sub, color }: { label: string; value: string; sub?: string; color?: string }) {
  return (
    <div style={s.kpi}>
      <div style={{ fontSize: 10, color: C.muted, textTransform: 'uppercase', letterSpacing: '.06em', marginBottom: 6 }}>{label}</div>
      <div style={{ fontSize: 22, fontWeight: 700, color: color ?? C.accent }}>{value}</div>
      {sub && <div style={{ fontSize: 10, color: C.muted, marginTop: 3 }}>{sub}</div>}
    </div>
  )
}

// ── Macro Widget ──────────────────────────────────────────────────────────────
function MacroWidget() {
  const [regime, setRegime] = useState<RegimeDetail | null>(null)
  useEffect(() => {
    api.regimeDetail().then(d => setRegime(d as RegimeDetail)).catch(() => {})
  }, [])

  if (!regime) return (
    <div style={{ ...s.card, display: 'flex', alignItems: 'center', justifyContent: 'center', color: C.muted, fontSize: 12 }}>
      Loading market regime…
    </div>
  )

  const regimeColor = regime.regime === 'LOW' ? C.green : regime.regime === 'HIGH' ? C.red : C.yellow
  const ind = regime.macro_indicators

  const rows: [string, number | null, string][] = [
    ['Fed Funds Rate', ind.fed_funds_rate, '%'],
    ['10Y Yield', ind.yield_10y, '%'],
    ['Yield Spread (10Y-2Y)', ind.yield_spread_10y2y, '%'],
    ['CPI YoY', ind.cpi_yoy, '%'],
    ['Unemployment', ind.unemployment_rate, '%'],
  ]

  return (
    <div style={s.card}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={s.cardTitle}>Market Regime & Macro</div>
          {regime.fetched_at && (() => {
            const ageSec = Math.round((Date.now() - new Date(regime.fetched_at!).getTime()) / 1000)
            const label = ageSec < 60 ? `${ageSec}s ago` : ageSec < 3600 ? `${Math.round(ageSec/60)}m ago` : `${Math.round(ageSec/3600)}h ago`
            // VIX is ~15min delayed by yfinance free tier; FRED data is weekly/monthly
            return <span style={{ fontSize: 10, color: C.muted }}>VIX ~15min delayed · FRED updated {label}</span>
          })()}
        </div>
        <span style={{
          padding: '3px 10px', borderRadius: 10, fontSize: 10, fontWeight: 700,
          background: `${regimeColor}1a`, color: regimeColor, border: `1px solid ${regimeColor}4d`,
          letterSpacing: '.06em',
        }}>{regime.regime}</span>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8, marginBottom: 12 }}>
        <div style={{ textAlign: 'center' as const }}>
          <div style={{ fontSize: 10, color: C.muted, marginBottom: 2 }}>Composite</div>
          <div style={{ fontSize: 18, fontWeight: 700, color: regimeColor }}>{(regime.composite_score * 100).toFixed(1)}</div>
          <div style={{ fontSize: 9, color: C.muted }}>risk score /100</div>
        </div>
        <div style={{ textAlign: 'center' as const }}>
          <div style={{ fontSize: 10, color: C.muted, marginBottom: 2 }}>VIX</div>
          <div style={{ fontSize: 18, fontWeight: 700, color: C.accent }}>{regime.vix != null ? regime.vix.toFixed(1) : '—'}</div>
          <div style={{ fontSize: 9, color: C.muted }}>score {(regime.vix_score * 100).toFixed(0)}/100</div>
        </div>
        <div style={{ textAlign: 'center' as const }}>
          <div style={{ fontSize: 10, color: C.muted, marginBottom: 2 }}>Macro</div>
          <div style={{ fontSize: 18, fontWeight: 700, color: C.blue }}>{(regime.macro_score * 100).toFixed(0)}</div>
          <div style={{ fontSize: 9, color: C.muted }}>risk score /100</div>
        </div>
      </div>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
        <tbody>
          {rows.map(([label, val, unit]) => (
            <tr key={label}>
              <td style={{ ...s.td, color: C.muted, fontSize: 11 }}>{label}</td>
              <td style={{ ...s.td, textAlign: 'right' as const, color: C.text, fontWeight: 600 }}>
                {val != null ? val.toFixed(2) + unit : '—'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div style={{ marginTop: 8, display: 'flex', gap: 8, fontSize: 10, color: C.muted }}>
        <span style={{ color: regime.trend_following_active ? C.green : C.red }}
          title="Trend Following: active when VIX is moderate and SPY is above its 20-day SMA">
          ● Trend {regime.trend_following_active ? 'ON' : 'OFF'}
        </span>
        <span style={{ color: regime.mean_reversion_active ? C.green : C.red }}
          title="Mean Reversion: active when VIX is elevated (>20) or SPY is in a range. OFF = skipping counter-trend entries in current regime.">
          ● MeanRev {regime.mean_reversion_active ? 'ON' : 'OFF'}
        </span>
        <span>Size ×{regime.position_size_multiplier}</span>
      </div>
    </div>
  )
}

// ── Overview Panel ────────────────────────────────────────────────────────────
type ChartRange = '1d' | '1w' | '1m'

function OverviewPanel({ summary, health, decisions, macroCtx }: {
  summary: Summary; health: Health | null
  decisions: Decision[]
  macroCtx: NisMacroContext | null
}) {
  const [positions, setPositions] = useState<import('./types').Position[]>([])
  const [chartRange, setChartRange] = useState<ChartRange>('1d')
  const [equityHistory, setEquityHistory] = useState<{ time: string; pnl: number }[]>([])

  useEffect(() => {
    fetch('/api/dashboard/positions').then(r => r.json()).then((arr: import('./types').Position[]) => {
      setPositions(Array.isArray(arr) ? arr : [])
    }).catch(() => {})
  }, [summary.timestamp])

  useEffect(() => {
    fetch(`/api/dashboard/metrics/equity-history?range=${chartRange}`)
      .then(r => r.json())
      .then(pts => setEquityHistory(Array.isArray(pts) ? pts : []))
      .catch(() => {})
  }, [chartRange, summary.timestamp])

  const mode = health?.trading_mode ?? health?.mode ?? 'paper'
  const status = health?.status ?? health?.system_status ?? 'unknown'
  const checks = health?.checks ?? { database: health?.database_ok, redis: health?.redis_ok, alpaca: health?.alpaca_ok }

  const lastSigLabel = (() => {
    const h = summary.last_signal_age_hours
    if (h == null) return '—'
    if (h < 1) return `${Math.round(h * 60)}m ago`
    return `${h.toFixed(0)}h ago`
  })()

  return (
    <div>
      {/* Health pills */}
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 16, alignItems: 'center' }}>
        <Pill label={mode.toUpperCase()} ok={mode !== 'live'} warn={mode !== 'live'} />
        <Pill label={status.toUpperCase()} ok={status === 'healthy'} />
        {Object.entries(checks ?? {}).map(([k, v]) => {
          const ok = v === true || v === 'ok' || v === 'connected'
          const names: Record<string, string> = { database: 'DB', redis: 'Redis', alpaca: 'Alpaca' }
          return <Pill key={k} label={(names[k] ?? k).toUpperCase()} ok={!!ok} />
        })}
        {summary.timestamp && (() => {
          const raw = summary.timestamp!
          const ts = new Date(raw.endsWith('Z') ? raw : raw + 'Z')
          const ageMs = Date.now() - ts.getTime()
          const ageSec = Math.round(ageMs / 1000)
          const cached = ageMs > 20_000
          const label = ageSec < 60 ? `${ageSec}s ago` : `${Math.round(ageSec/60)}m ago`
          return (
            <span style={{ fontSize: 10, color: cached ? C.yellow : C.green, marginLeft: 4 }}>
              {cached ? '○ cached' : '● live'} · {label}
            </span>
          )
        })()}
      </div>

      {/* Macro Risk Banner */}
      <MacroRiskBanner ctx={macroCtx} />

      {/* KPI strip */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 12, marginBottom: 16 }}>
        <KpiCard label="Account Value" value={fmt$(summary.account_value ?? summary.portfolio_value ?? summary.equity)} sub="Portfolio equity" />
        <KpiCard label="Daily P&L" value={fmt$(summary.daily_pnl ?? summary.pnl_today)}
          sub={(summary.daily_pnl_pct ?? summary.pnl_today_pct) != null ? fmtPct(summary.daily_pnl_pct ?? summary.pnl_today_pct) : undefined}
          color={clr(summary.daily_pnl ?? summary.pnl_today)} />
        <KpiCard label="Total P&L" value={fmt$(summary.total_pnl)}
          sub={summary.total_pnl_pct != null ? fmtPct(summary.total_pnl_pct) : undefined}
          color={clr(summary.total_pnl)} />
        <KpiCard label="Buying Power" value={fmt$(summary.buying_power)} sub={summary.cash != null ? `Cash ${fmt$(summary.cash)}` : '4x margin'} />
        <KpiCard label="Open Positions" value={(summary.open_positions_count ?? summary.open_positions)?.toString() ?? '—'} sub="Active trades" />
        <KpiCard label="Trades Today" value={summary.trades_today_count?.toString() ?? '0'} sub="Orders executed" />
        <KpiCard label="Capital Deployed"
          value={summary.capital_deployed_pct != null ? summary.capital_deployed_pct.toFixed(1) + '%' : '—'}
          sub={summary.capital_deployed != null ? fmt$(summary.capital_deployed) + ' in market' : undefined} />
        <KpiCard label="Win Rate" value={summary.win_rate != null ? summary.win_rate.toFixed(1) + '%' : '—'}
          sub={summary.win_rate != null ? 'All closed trades' : 'No closed trades yet'} />
        <KpiCard label="Max Drawdown" value={summary.max_drawdown_pct != null ? summary.max_drawdown_pct.toFixed(2) + '%' : '—'}
          color={summary.max_drawdown_pct != null ? (summary.max_drawdown_pct > 5 ? C.red : summary.max_drawdown_pct > 3 ? C.yellow : C.green) : undefined}
          sub={summary.max_drawdown_pct != null ? 'From peak equity' : 'No closed trades yet'} />
        <KpiCard label="Last Signal" value={lastSigLabel}
          sub={summary.last_signal_type ?? 'No signal yet'}
          color={summary.last_signal_type ? C.accent : C.muted} />
      </div>

      {/* Equity curve (left) + Open Positions (right) */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 12 }}>
        <div style={s.card}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
            <div style={s.cardTitle}>Equity Curve (P&L)</div>
            <div style={{ display: 'flex', gap: 4 }}>
              {(['1d', '1w', '1m'] as ChartRange[]).map(r => (
                <button key={r} onClick={() => setChartRange(r)} style={{
                  padding: '2px 10px', borderRadius: 4, fontSize: 10, fontWeight: 600,
                  cursor: 'pointer', border: `1px solid ${chartRange === r ? C.accent : C.border}`,
                  background: chartRange === r ? `${C.accent}22` : 'transparent',
                  color: chartRange === r ? C.accent : C.muted,
                }}>{r.toUpperCase()}</button>
              ))}
            </div>
          </div>
          <ResponsiveContainer width="100%" height={210}>
            <LineChart data={equityHistory}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,.04)" />
              <XAxis dataKey="time" tick={{ fill: C.muted, fontSize: 10 }} tickLine={false} axisLine={false} />
              <YAxis tick={{ fill: C.muted, fontSize: 10 }} tickLine={false} axisLine={false}
                tickFormatter={v => '$' + v.toFixed(0)} />
              <Tooltip contentStyle={{ background: C.surface2, border: `1px solid ${C.border}`, borderRadius: 4, fontSize: 11 }}
                labelStyle={{ color: C.muted }} itemStyle={{ color: C.green }}
                formatter={(v: number) => ['$' + v.toFixed(2), 'P&L']} />
              <Line type="monotone" dataKey="pnl" stroke={C.green} strokeWidth={1.5}
                dot={false} fill="rgba(34,197,94,.07)" />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div style={{ ...s.card, display: 'flex', flexDirection: 'column' }}>
          <div style={s.cardTitle}>Open Positions</div>
          {positions.length === 0
            ? <div style={{ color: C.muted, fontSize: 11, padding: '16px 0' }}>No open positions</div>
            : <div style={{ overflowY: 'auto', maxHeight: 260, flex: 1 }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                <thead><tr>
                  {['Symbol', 'Type', 'Qty', 'Value', 'Entry', 'Current', 'Return', 'P&L', 'Stop', 'Target', 'Signal'].map(h =>
                    <th key={h} style={s.th}>{h}</th>)}
                </tr></thead>
                <tbody>
                  {positions.map((p, i) => {
                    const qty = p.quantity ?? p.qty ?? 0
                    const entry = p.avg_price ?? p.avg_entry_price ?? 0
                    const cur = p.current_price ?? 0
                    const marketValue = cur * qty
                    const pnl = p.pnl_unrealized ?? p.unrealized_pl ?? 0
                    const retPct = p.pnl_unrealized_pct ?? p.unrealized_plpc ?? (entry > 0 ? (cur - entry) / entry * 100 : 0)
                    const retColor = retPct > 0 ? C.green : retPct < 0 ? C.red : C.muted
                    const stop = p.stop_price ?? null
                    const target = p.target_price ?? null
                    const sig = p.signal_type ?? null
                    const tradeType = p.trade_type ?? null
                    const typeColor = tradeType === 'intraday' ? C.yellow : tradeType === 'swing' ? C.blue : C.muted
                    return (
                      <tr key={i} style={{ borderBottom: `1px solid ${C.border}` }}>
                        <td style={{ ...s.td, color: C.accent, fontWeight: 700 }}>{p.symbol}</td>
                        <td style={{ ...s.td, color: typeColor, fontSize: 10, fontWeight: 600 }}>{tradeType ?? '—'}</td>
                        <td style={s.td}>{qty}</td>
                        <td style={s.td}>{fmt$(marketValue)}</td>
                        <td style={s.td}>{fmt$(entry)}</td>
                        <td style={s.td}>{fmt$(cur)}</td>
                        <td style={{ ...s.td, color: retColor }}>{retPct > 0 ? '+' : ''}{retPct.toFixed(2)}%</td>
                        <td style={{ ...s.td, color: retColor }}>{fmt$(pnl)}</td>
                        <td style={{ ...s.td, color: C.red }}>{stop != null ? fmt$(stop) : '—'}</td>
                        <td style={{ ...s.td, color: C.green }}>{target != null ? fmt$(target) : '—'}</td>
                        <td style={{ ...s.td, color: C.muted, fontSize: 10 }}>{sig ?? '—'}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          }
        </div>
      </div>

      {/* Macro (left) + Recent Decisions (right) */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
        <MacroWidget />
        <div style={{ ...s.card, display: 'flex', flexDirection: 'column' }}>
          <div style={s.cardTitle}>Recent Decisions</div>
          <div style={{ overflowY: 'auto', maxHeight: 260, flex: 1 }}>
            <DecisionsTable rows={decisions.slice(0, 50)} />
          </div>
        </div>
      </div>
    </div>
  )
}

function Pill({ label, ok, warn }: { label: string; ok: boolean; warn?: boolean }) {
  const color = warn ? C.yellow : ok ? C.green : C.red
  return (
    <span style={{
      padding: '4px 12px', borderRadius: 12, fontSize: 10, fontWeight: 600,
      textTransform: 'uppercase', letterSpacing: '.06em',
      background: `${color}1a`, color, border: `1px solid ${color}4d`,
    }}>{label}</span>
  )
}

// ── Macro Risk Banner ─────────────────────────────────────────────────────────
function MacroRiskBanner({ ctx }: { ctx: NisMacroContext | null }) {
  if (!ctx) return null
  const riskColor = ctx.overall_risk === 'HIGH' ? C.red : ctx.overall_risk === 'MEDIUM' ? C.yellow : C.green
  const sizeStr = ctx.global_sizing_factor !== 1.0 ? ` · sizing ${ctx.global_sizing_factor}×` : ''
  const blockStr = ctx.block_new_entries ? ' · NEW ENTRIES BLOCKED' : ''
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 10, padding: '8px 14px',
      borderRadius: 6, marginBottom: 12,
      background: `${riskColor}10`, border: `1px solid ${riskColor}40`,
    }}>
      <span style={{ fontWeight: 700, fontSize: 11, color: riskColor, whiteSpace: 'nowrap' }}>
        NIS MACRO: {ctx.overall_risk}{blockStr}{sizeStr}
      </span>
      {ctx.rationale && (
        <span style={{ fontSize: 11, color: C.muted, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {ctx.rationale}
        </span>
      )}
      <span style={{ fontSize: 10, color: C.muted, marginLeft: 'auto', whiteSpace: 'nowrap' }}>
        {ctx.events_today.length} event{ctx.events_today.length !== 1 ? 's' : ''} today
      </span>
    </div>
  )
}

// ── PM Decision Audit Table ───────────────────────────────────────────────────
function DecisionAuditTable({ rows }: { rows: DecisionAuditRow[] }) {
  const [expanded, setExpanded] = useState<string | null>(null)
  if (!rows.length) {
    return <div style={{ color: C.muted, textAlign: 'center', padding: 20, fontSize: 11 }}>No PM decisions recorded yet</div>
  }
  return (
    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
      <thead><tr>
        {['Time', 'Symbol', 'Strat', 'Decision', 'Score', 'NIS Policy', 'Block Reason', 'Outcome'].map(h =>
          <th key={h} style={s.th}>{h}</th>)}
      </tr></thead>
      <tbody>
        {rows.map(r => {
          const decColor = r.final_decision === 'enter' ? C.green
            : r.final_decision === 'block' ? C.red
            : r.final_decision === 'exit_review' ? C.yellow : C.muted
          const nisColor = r.news_action_policy === 'block_entry' ? C.red
            : r.news_action_policy?.includes('size_down') ? C.yellow : C.muted
          const isExpanded = expanded === r.id
          return (
            <>
              <tr key={r.id} onClick={() => setExpanded(isExpanded ? null : r.id)}
                style={{ cursor: r.top_features || r.block_reason ? 'pointer' : 'default',
                  background: isExpanded ? `${C.surface2}` : 'transparent' }}>
                <td style={{ ...s.td, color: C.muted, whiteSpace: 'nowrap' }}>{fmtTs(r.decided_at)}</td>
                <td style={{ ...s.td, color: C.accent, fontWeight: 700 }}>{r.symbol}</td>
                <td style={{ ...s.td, color: C.muted, fontSize: 10 }}>{r.strategy}</td>
                <td style={s.td}>
                  <span style={{ padding: '2px 6px', borderRadius: 3, fontSize: 10, fontWeight: 600,
                    background: `${decColor}20`, color: decColor }}>
                    {r.final_decision.toUpperCase()}
                  </span>
                </td>
                <td style={{ ...s.td, color: C.text }}>{r.model_score != null ? r.model_score.toFixed(3) : '—'}</td>
                <td style={{ ...s.td, color: nisColor, fontSize: 10 }}>{r.news_action_policy ?? '—'}</td>
                <td style={{ ...s.td, color: C.muted, fontSize: 10, maxWidth: 180, overflow: 'hidden',
                  textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {r.block_reason ? r.block_reason.split(':')[0] : '—'}
                </td>
                <td style={{ ...s.td, color: r.outcome_pnl_pct != null ? clr(r.outcome_pnl_pct) : C.muted }}>
                  {r.outcome_pnl_pct != null ? (r.outcome_pnl_pct >= 0 ? '+' : '') + (r.outcome_pnl_pct * 100).toFixed(2) + '%' : '—'}
                </td>
              </tr>
              {isExpanded && (r.top_features || r.block_reason) && (
                <tr key={r.id + '_exp'} style={{ background: C.surface2 }}>
                  <td colSpan={8} style={{ padding: '8px 12px', fontSize: 10, color: C.muted }}>
                    {r.block_reason && (
                      <div style={{ marginBottom: 4 }}>
                        <span style={{ color: C.red }}>Block: </span>{r.block_reason}
                      </div>
                    )}
                    {r.top_features && (
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px 16px' }}>
                        <span style={{ color: C.accent, marginRight: 4 }}>Top features:</span>
                        {Object.entries(r.top_features).map(([f, v]) => (
                          <span key={f}>{f}: <span style={{ color: C.text }}>{Number(v).toFixed(4)}</span></span>
                        ))}
                      </div>
                    )}
                  </td>
                </tr>
              )}
            </>
          )
        })}
      </tbody>
    </table>
  )
}

// ── NIS Signals Table ─────────────────────────────────────────────────────────
function NisSignalsTable({ signals }: { signals: NisSignal[] }) {
  if (!signals.length) {
    return <div style={{ color: C.muted, textAlign: 'center', padding: 20, fontSize: 11 }}>No signals cached yet — morning digest runs at 09:00 ET</div>
  }
  return (
    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
      <thead><tr>
        {['Symbol', 'Policy', 'Direction', 'Materiality', 'Confidence', 'Size', 'Age', 'Rationale'].map(h =>
          <th key={h} style={s.th}>{h}</th>)}
      </tr></thead>
      <tbody>
        {signals.map(sig => {
          const polColor = sig.action_policy === 'block_entry' ? C.red
            : sig.action_policy?.includes('size_down') ? C.yellow
            : sig.action_policy?.includes('size_up') ? C.green : C.muted
          const dirColor = sig.direction_score > 0.1 ? C.green : sig.direction_score < -0.1 ? C.red : C.muted
          const ageMin = Math.round(sig.age_seconds / 60)
          const ageStr = ageMin < 60 ? `${ageMin}m` : `${Math.round(ageMin / 60)}h`
          return (
            <tr key={sig.symbol}>
              <td style={{ ...s.td, color: C.accent, fontWeight: 700 }}>{sig.symbol}</td>
              <td style={s.td}>
                <span style={{ padding: '2px 6px', borderRadius: 3, fontSize: 10, fontWeight: 600,
                  background: `${polColor}20`, color: polColor }}>
                  {sig.action_policy}
                </span>
              </td>
              <td style={{ ...s.td, color: dirColor }}>{sig.direction_score >= 0 ? '+' : ''}{sig.direction_score.toFixed(2)}</td>
              <td style={{ ...s.td, color: C.text }}>{sig.materiality_score.toFixed(2)}</td>
              <td style={{ ...s.td, color: C.text }}>{sig.confidence.toFixed(2)}</td>
              <td style={{ ...s.td, color: sig.sizing_multiplier !== 1.0 ? C.yellow : C.muted }}>
                {sig.sizing_multiplier.toFixed(2)}×
              </td>
              <td style={{ ...s.td, color: C.muted }}>{ageStr}</td>
              <td style={{ ...s.td, color: C.muted, fontSize: 10, maxWidth: 220,
                overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {sig.rationale ?? '—'}
              </td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}

// ── Gate Summary Table ────────────────────────────────────────────────────────
function GateSummaryTable({ rows }: { rows: GateSummaryRow[] }) {
  if (!rows.length) {
    return <div style={{ color: C.muted, textAlign: 'center', padding: 20, fontSize: 11 }}>
      No gate summary yet — needs ~2 weeks of data with backfilled outcomes
    </div>
  }
  return (
    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
      <thead><tr>
        {['Block Reason', 'Count', 'Avg Missed P&L', 'Verdict'].map(h =>
          <th key={h} style={s.th}>{h}</th>)}
      </tr></thead>
      <tbody>
        {rows.map((r, i) => {
          const pnl = r.avg_pnl_pct
          const verdict = pnl == null ? '—'
            : pnl < -0.005 ? '✓ Blocked losers'
            : pnl > 0.005 ? '⚠ Blocked winners'
            : 'Neutral'
          const verdictColor = pnl == null ? C.muted : pnl < -0.005 ? C.green : pnl > 0.005 ? C.red : C.muted
          return (
            <tr key={i}>
              <td style={{ ...s.td, color: C.text, fontFamily: 'monospace' }}>{r.block_reason ?? 'unclassified'}</td>
              <td style={{ ...s.td, color: C.accent }}>{r.count}</td>
              <td style={{ ...s.td, color: pnl != null ? clr(pnl) : C.muted }}>
                {pnl != null ? (pnl >= 0 ? '+' : '') + (pnl * 100).toFixed(2) + '%' : '—'}
              </td>
              <td style={{ ...s.td, color: verdictColor, fontWeight: 600 }}>{verdict}</td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}

// ── Positions Panel ────────────────────────────────────────────────────────────
function PositionsPanel({ onRefresh }: { onRefresh: () => void }) {
  const [rows, setRows] = useState<Position[]>([])
  const load = useCallback(async () => {
    try {
      const j = await api.positions() as { data?: Position[] } | Position[]
      setRows((j as { data?: Position[] }).data ?? (j as Position[]) ?? [])
    } catch { /* ignore */ }
  }, [])
  useEffect(() => { load() }, [load])

  const totalUnreal = rows.reduce((s, p) => s + (p.unrealized_pl ?? 0), 0)
  const largest = rows.reduce((m, p) => (p.market_value ?? 0) > (m.market_value ?? 0) ? p : m, rows[0])

  return (
    <div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 12, marginBottom: 16 }}>
        <KpiCard label="Open Positions" value={String(rows.length)} />
        <KpiCard label="Unrealized P&L" value={fmt$(totalUnreal)} color={clr(totalUnreal)} />
        <KpiCard label="Largest Position" value={largest?.symbol ?? '—'} />
      </div>
      <div style={s.card}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
          <div style={s.cardTitle}>Open Positions</div>
          <button onClick={() => { load(); onRefresh() }} style={btnStyle}>Refresh</button>
        </div>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead><tr>
              {['Symbol', 'Type', 'Qty', 'Avg Entry', 'Current', 'Market Value', 'Unreal P&L', 'P&L %', 'Stop', 'Target', 'R:R', 'Signal', 'Entry Date', 'Bars'].map(h => (
                <th key={h} style={s.th}>{h}</th>
              ))}
            </tr></thead>
            <tbody>
              {rows.length === 0
                ? <tr><td colSpan={14} style={{ ...s.td, textAlign: 'center', color: C.muted, padding: 20 }}>No open positions</td></tr>
                : rows.map(p => {
                  const pct = p.unrealized_plpc != null ? p.unrealized_plpc * 100 : (p.pnl_unrealized_pct ?? null)
                  const qty = p.qty ?? p.quantity ?? 0
                  const entry = p.avg_entry_price ?? p.avg_price ?? 0
                  const cur = p.current_price ?? 0
                  const mv = p.market_value ?? cur * qty
                  const pnl = p.unrealized_pl ?? p.pnl_unrealized ?? 0
                  return (
                    <tr key={p.symbol}>
                      <td style={{ ...s.td, color: C.accent, fontWeight: 600 }}>{p.symbol}</td>
                      <td style={{ ...s.td, color: C.muted, fontSize: 10 }}>{p.trade_type ?? '—'}</td>
                      <td style={s.td}>{qty}</td>
                      <td style={s.td}>{fmt$(entry)}</td>
                      <td style={s.td}>{fmt$(cur)}</td>
                      <td style={s.td}>{fmt$(mv)}</td>
                      <td style={{ ...s.td, color: clr(pnl) }}>{fmt$(pnl)}</td>
                      <td style={{ ...s.td, color: clr(pct) }}>{pct != null ? fmtPct(pct) : '—'}</td>
                      <td style={{ ...s.td, color: C.red }}>{p.stop_price != null ? fmt$(p.stop_price) : '—'}</td>
                      <td style={{ ...s.td, color: C.green }}>{p.target_price != null ? fmt$(p.target_price) : '—'}</td>
                      <td style={{ ...s.td, color: C.muted }}>{p.risk_reward != null ? p.risk_reward.toFixed(1) + ':1' : '—'}</td>
                      <td style={{ ...s.td, color: C.blue, fontSize: 10 }}>{p.signal_type ?? '—'}</td>
                      <td style={{ ...s.td, color: C.muted, fontSize: 10 }}>{p.entry_date ?? '—'}</td>
                      <td style={{ ...s.td, color: C.muted }}>{p.bars_held ?? '—'}</td>
                    </tr>
                  )
                })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

// ── Trades Panel ───────────────────────────────────────────────────────────────
type TradesDateFilter = 'today' | '7d' | '30d' | 'all'

function TradesPanel() {
  const [allRows, setAllRows] = useState<Trade[]>([])
  const [statusFilter, setStatusFilter] = useState('')
  const [dateFilter, setDateFilter] = useState<TradesDateFilter>('all')

  const load = useCallback(async (f = statusFilter) => {
    try {
      const j = await api.trades(f || undefined) as { data?: Trade[] } | Trade[]
      setAllRows((j as { data?: Trade[] }).data ?? (j as Trade[]) ?? [])
    } catch { /* ignore */ }
  }, [statusFilter])
  useEffect(() => { load() }, [load])

  const rows = (() => {
    if (dateFilter === 'all') return allRows
    const now = Date.now()
    const cutoff = dateFilter === 'today'
      ? new Date(new Date().toLocaleDateString('en-CA', { timeZone: 'America/New_York' })).getTime()
      : now - (dateFilter === '7d' ? 7 : 30) * 86400_000
    return allRows.filter(t => {
      const ts = t.created_at ? new Date(t.created_at.endsWith('Z') ? t.created_at : t.created_at + 'Z').getTime() : 0
      return ts >= cutoff
    })
  })()

  const closed = rows.filter(t => t.status === 'CLOSED')
  const wins = closed.filter(t => (t.pnl ?? 0) > 0)
  const totalPnl = closed.reduce((s, t) => s + (t.pnl ?? 0), 0)
  const avgPnl = closed.length ? totalPnl / closed.length : 0
  const wr = closed.length ? wins.length / closed.length * 100 : 0

  return (
    <div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 12, marginBottom: 16 }}>
        <KpiCard label="Total Trades" value={String(rows.length)} />
        <KpiCard label="Closed Trades" value={String(closed.length)} />
        <KpiCard label="Win Rate" value={closed.length ? wr.toFixed(1) + '%' : '—'} />
        <KpiCard label="Avg Trade P&L" value={closed.length ? fmt$(avgPnl) : '—'} color={clr(avgPnl)} />
        <KpiCard label="Total P&L" value={fmt$(totalPnl)} color={clr(totalPnl)} />
      </div>
      <div style={s.card}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12, flexWrap: 'wrap', gap: 8 }}>
          <div style={s.cardTitle}>Trade History</div>
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
            {(['today', '7d', '30d', 'all'] as TradesDateFilter[]).map(d => (
              <button key={d} onClick={() => setDateFilter(d)} style={{
                ...btnStyle,
                color: dateFilter === d ? C.accent : C.muted,
                borderColor: dateFilter === d ? C.accent : C.border,
              }}>{d === 'today' ? 'Today' : d === 'all' ? 'All time' : d}</button>
            ))}
            <select value={statusFilter}
              onChange={e => { setStatusFilter(e.target.value); load(e.target.value) }}
              style={{ background: C.surface2, border: `1px solid ${C.border}`, color: C.text, padding: '3px 8px', borderRadius: 4, fontSize: 11, fontFamily: 'inherit' }}>
              <option value="">All status</option>
              <option value="ACTIVE">Active</option>
              <option value="CLOSED">Closed</option>
            </select>
            <button onClick={() => load()} style={btnStyle}>Refresh</button>
          </div>
        </div>
        <div style={{ overflowX: 'auto', maxHeight: 480, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead><tr>
              {['Symbol', 'Type', 'Dir', 'Signal', 'Entry', 'Exit', '% Chg', 'Qty', 'P&L', 'Status', 'Opened', 'Closed'].map(h => (
                <th key={h} style={{ ...s.th, position: 'sticky', top: 0, background: C.surface, zIndex: 1 }}>{h}</th>
              ))}
            </tr></thead>
            <tbody>
              {rows.length === 0
                ? <tr><td colSpan={12} style={{ ...s.td, textAlign: 'center', color: C.muted, padding: 20 }}>No trades found</td></tr>
                : rows.map((t, i) => {
                  const typeColor = t.trade_type === 'intraday' ? C.yellow : t.trade_type === 'swing' ? C.blue : C.muted
                  const pnlPct = t.exit_price && t.entry_price && t.entry_price > 0
                    ? ((t.exit_price - t.entry_price) / t.entry_price * 100) : null
                  return (
                  <tr key={i}>
                    <td style={{ ...s.td, color: C.accent, fontWeight: 600 }}>{t.symbol}</td>
                    <td style={{ ...s.td, color: typeColor, fontSize: 10, fontWeight: 600 }}>{t.trade_type ?? '—'}</td>
                    <td style={s.td}>{t.direction}</td>
                    <td style={{ ...s.td, color: C.blue }}>{t.signal_type ?? '—'}</td>
                    <td style={s.td}>{fmt$(t.entry_price)}</td>
                    <td style={s.td}>{t.exit_price ? fmt$(t.exit_price) : <span style={{ color: C.muted }}>open</span>}</td>
                    <td style={{ ...s.td, color: clr(pnlPct) }}>{pnlPct != null ? fmtPct(pnlPct) : '—'}</td>
                    <td style={s.td}>{t.quantity}</td>
                    <td style={{ ...s.td, color: clr(t.pnl) }}>{t.pnl != null ? fmt$(t.pnl) : '—'}</td>
                    <td style={s.td}><StatusPill status={t.status} /></td>
                    <td style={{ ...s.td, color: C.muted }}>{fmtTs(t.created_at)}</td>
                    <td style={{ ...s.td, color: C.muted }}>{t.closed_at ? fmtTs(t.closed_at) : <span style={{ color: C.muted }}>—</span>}</td>
                  </tr>
                  )
                })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

function StatusPill({ status }: { status: string }) {
  const color = status === 'ACTIVE' ? C.yellow : status === 'CLOSED' ? C.green : C.red
  return (
    <span style={{
      padding: '2px 8px', borderRadius: 10, fontSize: 10, fontWeight: 600,
      background: `${color}1a`, color, border: `1px solid ${color}4d`,
    }}>{status}</span>
  )
}

// ── Signal Monitor Panel ───────────────────────────────────────────────────────
interface SignalRow { time: string; symbol: string; kind: 'buy' | 'sell' | 'signal'; msg: string }

function SignalsPanel({ feed, decisions }: { feed: SignalRow[]; decisions: Decision[] }) {
  const [auditRows, setAuditRows] = useState<DecisionAuditRow[]>([])
  const [strategy, setStrategy] = useState<'all' | 'swing' | 'intraday'>('all')

  const loadAudit = useCallback(async (strat: string) => {
    try {
      const j = await api.decisionAuditRecent(150, strat === 'all' ? undefined : strat) as { decisions?: DecisionAuditRow[] }
      setAuditRows(j.decisions ?? [])
    } catch { /* ignore */ }
  }, [])

  useEffect(() => { loadAudit(strategy) }, [loadAudit, strategy])
  useEffect(() => {
    const id = setInterval(() => loadAudit(strategy), 30000)
    return () => clearInterval(id)
  }, [loadAudit, strategy])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      {/* Live signal feed */}
      <div style={s.card}>
        <div style={s.cardTitle}>Live Signal Feed (WebSocket)</div>
        <div style={{ maxHeight: 200, overflowY: 'auto' }}>
          {feed.length === 0
            ? <div style={{ color: C.muted, textAlign: 'center', padding: 20, fontSize: 11 }}>Waiting for signals via WebSocket...</div>
            : feed.map((row, i) => {
              const color = row.kind === 'buy' ? C.green : row.kind === 'sell' ? C.red : C.blue
              const label = row.kind === 'buy' ? 'ENTRY' : row.kind === 'sell' ? 'EXIT' : 'SIGNAL'
              return (
                <div key={i} style={{ display: 'flex', gap: 10, padding: '6px 0', borderBottom: `1px solid rgba(255,255,255,.04)`, fontSize: 11 }}>
                  <span style={{ color: C.muted, minWidth: 80, whiteSpace: 'nowrap' }}>{row.time}</span>
                  <span style={{ color: C.accent, fontWeight: 600, minWidth: 50 }}>{row.symbol}</span>
                  <span style={{ padding: '1px 6px', borderRadius: 3, fontSize: 10, fontWeight: 600, background: `${color}26`, color }}>{label}</span>
                  <span style={{ color: C.muted, flex: 1 }}>{row.msg}</span>
                </div>
              )
            })}
        </div>
      </div>

      {/* PM Decision Audit — main review table */}
      <div style={s.card}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
          <div style={s.cardTitle}>PM Decision Audit (click row for detail)</div>
          <div style={{ display: 'flex', gap: 6 }}>
            {(['all', 'swing', 'intraday'] as const).map(s2 => (
              <button key={s2} onClick={() => { setStrategy(s2); loadAudit(s2) }} style={{
                padding: '3px 10px', borderRadius: 4, fontSize: 10, fontWeight: 600, cursor: 'pointer',
                border: `1px solid ${strategy === s2 ? C.accent : C.border}`,
                background: strategy === s2 ? `${C.accent}22` : 'transparent',
                color: strategy === s2 ? C.accent : C.muted,
              }}>{s2}</button>
            ))}
            <button onClick={() => loadAudit(strategy)} style={{
              padding: '3px 10px', borderRadius: 4, fontSize: 10, cursor: 'pointer',
              border: `1px solid ${C.border}`, background: 'transparent', color: C.muted,
            }}>↻</button>
          </div>
        </div>
        <div style={{ maxHeight: 480, overflowY: 'auto' }}>
          <DecisionAuditTable rows={auditRows} />
        </div>
      </div>

      {/* Legacy agent decisions (compact) */}
      <div style={s.card}>
        <div style={s.cardTitle}>Agent System Events (Last 50)</div>
        <div style={{ maxHeight: 200, overflowY: 'auto' }}>
          <DecisionsTable rows={decisions} />
        </div>
      </div>
    </div>
  )
}

function decisionColor(type: string | undefined): string {
  const t = (type ?? '').toUpperCase()
  if (t.includes('BUY') || t.includes('APPROVED') || t.includes('OPEN')) return C.green
  if (t.includes('SELL') || t.includes('CLOSE') || t.includes('REJECT') || t.includes('FORCE')) return C.red
  if (t.includes('WARN') || t.includes('RISK')) return C.yellow
  return C.muted
}

function extractSymbol(d: Decision): string {
  if (d.symbol) return d.symbol
  const r = d.reasoning as Record<string, unknown> | undefined
  if (!r) return '—'
  if (typeof r.symbol === 'string') return r.symbol
  if (Array.isArray(r.symbols) && r.symbols.length) return (r.symbols as string[]).join(', ')
  if (typeof r.ticker === 'string') return r.ticker
  return '—'
}

function DecisionsTable({ rows }: { rows: Decision[] }) {
  if (!rows.length) {
    return <div style={{ color: C.muted, textAlign: 'center', padding: 16, fontSize: 11 }}>No decisions yet</div>
  }
  return (
    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
      <thead><tr>
        {['Time', 'Agent', 'Action', 'Symbol'].map(h => <th key={h} style={s.th}>{h}</th>)}
      </tr></thead>
      <tbody>
        {rows.map((d, i) => {
          const color = decisionColor(d.decision_type)
          const sym = extractSymbol(d)
          return (
            <tr key={i}>
              <td style={{ ...s.td, color: C.muted, whiteSpace: 'nowrap' }}>{fmtTs(d.timestamp)}</td>
              <td style={{ ...s.td, color: C.muted }}>{d.agent_name ?? '—'}</td>
              <td style={s.td}>
                <span style={{ padding: '2px 6px', borderRadius: 3, fontSize: 10, fontWeight: 600,
                  background: `${color}22`, color }}>{d.decision_type ?? '—'}</span>
              </td>
              <td style={{ ...s.td, color: C.accent, fontWeight: 600 }}>{sym}</td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}

// ── Capital Ramp Panel ────────────────────────────────────────────────────────
function RampPanel({ toast }: { toast: (msg: string, type?: 'success' | 'error' | 'warning' | 'info') => void }) {
  const [status, setStatus] = useState<LiveStatus>({})
  const [advanceMsg, setAdvanceMsg] = useState('')
  const load = useCallback(async () => {
    try {
      const j = await api.liveStatus() as LiveStatus
      setStatus(j)
    } catch { /* ignore */ }
  }, [])
  useEffect(() => { load() }, [load])

  const maxCap = Math.max(...(status.stages ?? []).map(s => s.capital), 1)

  async function advance() {
    try {
      const j = await api.increaseCapital() as { status?: string; detail?: string }
      setAdvanceMsg(JSON.stringify(j, null, 2))
      if (j.status === 'advanced') { toast('Capital stage advanced!', 'success'); load() }
      else toast(j.detail ?? j.status ?? 'Cannot advance', 'warning')
    } catch { toast('Request failed', 'error') }
  }

  return (
    <div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 12, marginBottom: 16 }}>
        <KpiCard label="Current Stage" value={status.stage != null ? 'Stage ' + status.stage : '—'} />
        <KpiCard label="Approved Capital" value={fmt$(status.capital)} />
        <KpiCard label="Days in Stage" value={status.days_elapsed != null ? status.days_elapsed + 'd' : '—'} />
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
        <div style={s.card}>
          <div style={s.cardTitle}>Stage Progress</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginTop: 8 }}>
            {(status.stages ?? []).map(stage => {
              const pct = (stage.capital / maxCap * 100).toFixed(1)
              const isCurrent = stage.stage === status.stage
              const isDone = (status.stage ?? 0) > stage.stage
              const barColor = isDone ? C.green : isCurrent ? C.accent : C.muted
              return (
                <div key={stage.stage} style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                  <span style={{ minWidth: 70, fontSize: 11, color: isCurrent ? C.accent : C.muted }}>Stage {stage.stage}</span>
                  <div style={{ flex: 1, height: 6, background: C.surface2, borderRadius: 3, overflow: 'hidden' }}>
                    <div style={{ width: pct + '%', height: '100%', background: barColor, borderRadius: 3, transition: 'width .4s' }} />
                  </div>
                  <span style={{ minWidth: 80, textAlign: 'right', fontSize: 11, color: isCurrent ? C.accent : C.muted }}>
                    {fmt$(stage.capital)} {isCurrent ? '←' : ''}
                  </span>
                </div>
              )
            })}
          </div>
        </div>
        <div style={s.card}>
          <div style={s.cardTitle}>Advance Capital Stage</div>
          <div style={{ fontSize: 12, lineHeight: 1.7 }}>
            <div style={{ color: C.muted, marginBottom: 16 }}>Advance only when health criteria are met (drawdown ≤ 3%, daily loss ≤ 2%).</div>
            <button onClick={advance} style={{
              background: 'rgba(59,130,246,.1)', border: `1px solid ${C.blue}`,
              color: C.blue, padding: '8px 20px', borderRadius: 5, cursor: 'pointer',
              fontFamily: 'inherit', fontSize: 12, fontWeight: 600, letterSpacing: '.04em',
            }}>Request Stage Advance</button>
            {advanceMsg && <pre style={{ marginTop: 12, fontSize: 10, color: C.muted, whiteSpace: 'pre-wrap' }}>{advanceMsg}</pre>}
          </div>
        </div>
      </div>
    </div>
  )
}

// ── Kill Switch Panel ─────────────────────────────────────────────────────────
function KillPanel({ toast }: { toast: (msg: string, type?: 'success' | 'error' | 'warning' | 'info') => void }) {
  const [active, setActive] = useState(false)
  const [showModal, setShowModal] = useState(false)
  const [auditLog, setAuditLog] = useState<AuditEntry[]>([])

  const loadStatus = useCallback(async () => {
    try {
      const j = await api.liveStatus() as { kill_switch_active?: boolean }
      setActive(j.kill_switch_active ?? false)
    } catch { /* ignore */ }
    try {
      const j = await api.auditLog() as { logs?: AuditEntry[]; data?: AuditEntry[] } | AuditEntry[]
      setAuditLog((j as { logs?: AuditEntry[] }).logs ?? (j as { data?: AuditEntry[] }).data ?? (j as AuditEntry[]) ?? [])
    } catch { /* ignore */ }
  }, [])
  useEffect(() => { loadStatus() }, [loadStatus])

  async function confirmKill() {
    setShowModal(false)
    try {
      await api.killSwitch('Manual dashboard activation')
      toast('Kill switch activated — all positions closing', 'error')
      setActive(true)
      loadStatus()
    } catch { toast('Kill switch request failed', 'error') }
  }

  async function reset() {
    if (!confirm('Reset kill switch and resume trading?')) return
    try {
      await api.resetKillSwitch()
      toast('Kill switch reset — trading resumed', 'success')
      setActive(false)
      loadStatus()
    } catch { toast('Reset failed', 'error') }
  }

  return (
    <div style={{ maxWidth: 540, margin: '0 auto', paddingTop: 20 }}>
      {/* Status box */}
      <div style={{
        ...s.card,
        textAlign: 'center',
        ...(active ? { borderColor: 'rgba(239,68,68,.4)', background: 'rgba(239,68,68,.07)' } : {}),
        marginBottom: 16,
      }}>
        <div style={{ fontSize: 13, marginBottom: 4 }}>
          Kill switch is <strong style={{ color: active ? C.red : C.text }}>{active ? 'ACTIVE' : 'INACTIVE'}</strong>
        </div>
        <div style={{ color: C.muted, fontSize: 11 }}>
          {active ? 'All new trades are halted.' : 'All trading is proceeding normally.'}
        </div>
      </div>

      {/* Kill button */}
      {!active && (
        <button onClick={() => setShowModal(true)} style={{
          display: 'block', margin: '0 auto 20px', width: '100%',
          background: 'rgba(239,68,68,.1)', border: `2px solid ${C.red}`, color: C.red,
          padding: '16px 40px', borderRadius: 8, fontSize: 16, fontWeight: 700,
          fontFamily: 'inherit', cursor: 'pointer', letterSpacing: '.06em', textTransform: 'uppercase',
        }}>EMERGENCY STOP</button>
      )}
      {active && (
        <button onClick={reset} style={{
          display: 'block', margin: '0 auto 20px', width: '100%',
          background: 'rgba(34,197,94,.08)', border: `1px solid ${C.green}`, color: C.green,
          padding: '12px 40px', borderRadius: 8, fontSize: 14, fontWeight: 600,
          fontFamily: 'inherit', cursor: 'pointer', letterSpacing: '.04em',
        }}>Reset Kill Switch</button>
      )}

      {/* Audit log */}
      <div style={s.card}>
        <div style={s.cardTitle}>Recent Audit Log</div>
        <div style={{ maxHeight: 320, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead><tr>
              {['Time', 'Action', 'Details'].map(h => <th key={h} style={s.th}>{h}</th>)}
            </tr></thead>
            <tbody>
              {auditLog.length === 0
                ? <tr><td colSpan={3} style={{ ...s.td, textAlign: 'center', color: C.muted, padding: 16 }}>No audit entries</td></tr>
                : auditLog.map((a, i) => (
                  <tr key={i}>
                    <td style={{ ...s.td, color: C.muted, whiteSpace: 'nowrap' }}>{fmtTs(a.timestamp)}</td>
                    <td style={{ ...s.td, color: C.yellow }}>{a.action}</td>
                    <td style={{ ...s.td, color: C.muted, fontSize: 10 }}>{JSON.stringify(a.details ?? {}).slice(0, 80)}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Confirm modal */}
      {showModal && (
        <div style={{
          position: 'fixed', inset: 0, background: 'rgba(0,0,0,.75)',
          zIndex: 1000, display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          <div style={{ ...s.card, maxWidth: 440, width: '90%' }}>
            <h3 style={{ color: C.red, marginBottom: 12, fontSize: 15 }}>Confirm Emergency Stop</h3>
            <p style={{ color: C.muted, fontSize: 12, marginBottom: 20, lineHeight: 1.6 }}>
              This will immediately close ALL open positions with market orders and halt all new trades.
              This action is logged and cannot be undone automatically.
            </p>
            <div style={{ display: 'flex', gap: 10, justifyContent: 'flex-end' }}>
              <button onClick={() => setShowModal(false)} style={btnStyle}>Cancel</button>
              <button onClick={confirmKill} style={{
                ...btnStyle, background: 'rgba(239,68,68,.15)', borderColor: C.red, color: C.red,
              }}>STOP TRADING NOW</button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

const btnStyle: React.CSSProperties = {
  background: 'none', border: `1px solid ${C.border}`, color: C.muted,
  padding: '4px 12px', borderRadius: 4, cursor: 'pointer', fontSize: 11,
  fontFamily: 'inherit',
}

// ── Top Bar ────────────────────────────────────────────────────────────────────
function TopBar({ health, wsConnected }: { health: Health | null; wsConnected: boolean }) {
  const [clock, setClock] = useState('')
  useEffect(() => {
    const tick = () => {
      const now = new Date()
      const etStr = now.toLocaleTimeString('en-US', {
        hour: 'numeric', minute: '2-digit', second: '2-digit', hour12: true,
        timeZone: 'America/New_York',
      })
      const dateStr = now.toLocaleDateString('en-US', {
        month: 'short', day: 'numeric', timeZone: 'America/New_York',
      })
      setClock(`${dateStr}  ${etStr} ET`)
    }
    tick()
    const id = setInterval(tick, 1000)
    return () => clearInterval(id)
  }, [])

  const mode = health?.trading_mode ?? health?.mode ?? 'paper'
  const status = health?.status ?? health?.system_status ?? 'unknown'
  const modeColor = mode === 'live' ? C.red : C.yellow
  const statusColor = status === 'healthy' ? C.green : C.red

  return (
    <div style={{
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      padding: '0 20px', height: 48, background: C.surface,
      borderBottom: `1px solid ${C.border}`, position: 'sticky', top: 0, zIndex: 100,
    }}>
      <div style={{ color: C.accent, fontSize: 15, fontWeight: 700, letterSpacing: '.04em' }}>
        MR<span style={{ color: C.green }}>TRADER</span>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <Badge label={mode.toUpperCase()} color={modeColor} />
        <Badge label={status.toUpperCase()} color={statusColor} />
        <span style={{ color: C.muted, fontSize: 11, minWidth: 150, textAlign: 'right' }}>{clock}</span>
        <div title="WebSocket" style={{
          width: 8, height: 8, borderRadius: '50%',
          background: wsConnected ? C.green : C.red,
          boxShadow: wsConnected ? `0 0 6px ${C.green}` : 'none',
          transition: 'background .3s',
        }} />
      </div>
    </div>
  )
}

function Badge({ label, color }: { label: string; color: string }) {
  return (
    <span style={{
      padding: '3px 10px', borderRadius: 4, fontSize: 11, fontWeight: 600,
      textTransform: 'uppercase', letterSpacing: '.06em',
      background: `${color}26`, color, border: `1px solid ${color}4d`,
    }}>{label}</span>
  )
}

// ── Session Control Panel ─────────────────────────────────────────────────────
function SessionPanel({ toast }: { toast: (msg: string, type?: 'success' | 'error' | 'warning' | 'info') => void }) {
  const [orch, setOrch] = useState<OrchestratorStatus | null>(null)
  const [market, setMarket] = useState<MarketStatus | null>(null)
  const [jobs, setJobs] = useState<ScheduledJob[]>([])
  const [log, setLog] = useState<SessionLogEntry[]>([])
  const [busy, setBusy] = useState<string | null>(null)

  const loadAll = useCallback(async () => {
    try {
      const [s, m, j, l] = await Promise.all([
        api.orchStatus() as Promise<OrchestratorStatus>,
        api.marketStatus() as Promise<MarketStatus>,
        api.orchJobs() as Promise<{ jobs: ScheduledJob[] }>,
        api.sessionLog(50) as Promise<{ entries: SessionLogEntry[] }>,
      ])
      setOrch(s); setMarket(m)
      setJobs(j.jobs ?? []); setLog(l.entries ?? [])
    } catch { /* ignore */ }
  }, [])

  useEffect(() => { loadAll(); const id = setInterval(loadAll, 10000); return () => clearInterval(id) }, [loadAll])

  async function act(label: string, fn: () => Promise<unknown>, successMsg: string) {
    setBusy(label)
    try { await fn(); toast(successMsg, 'success'); await loadAll() }
    catch (e) { toast(`${label} failed`, 'error') }
    finally { setBusy(null) }
  }

  const agentColor = (s: string) =>
    s === 'running' ? C.green : s === 'error' ? C.red : s === 'paused' ? C.yellow : C.muted

  const logColor = (level: string) =>
    level === 'ERROR' ? C.red : level === 'WARNING' ? C.yellow : level === 'INFO' ? C.green : C.muted

  const isTrading = orch?.running && !Object.values(orch.agents).every(v => v === 'paused')

  return (
    <div>
      {/* Market + orchestrator header strip */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 12, marginBottom: 16 }}>
        <div style={s.kpi}>
          <div style={{ fontSize: 10, color: C.muted, textTransform: 'uppercase', letterSpacing: '.06em', marginBottom: 6 }}>Market</div>
          <div style={{ fontSize: 18, fontWeight: 700, color: market?.is_open ? C.green : C.muted }}>
            {market ? (market.is_open ? 'OPEN' : 'CLOSED') : '—'}
          </div>
          <div style={{ fontSize: 10, color: C.muted, marginTop: 3 }}>{market?.current_time_et ?? '—'}</div>
        </div>
        <div style={s.kpi}>
          <div style={{ fontSize: 10, color: C.muted, textTransform: 'uppercase', letterSpacing: '.06em', marginBottom: 6 }}>Next Event</div>
          <div style={{ fontSize: 14, fontWeight: 700, color: C.accent }}>
            {market?.next_event.event === 'market_open' ? 'Opens' : 'Closes'}
          </div>
          <div style={{ fontSize: 10, color: C.muted, marginTop: 3 }}>
            {market?.next_event.minutes != null
              ? `in ${market.next_event.minutes} min`
              : `${market?.next_event.date} @ ${market?.next_event.time}`}
          </div>
        </div>
        <div style={s.kpi}>
          <div style={{ fontSize: 10, color: C.muted, textTransform: 'uppercase', letterSpacing: '.06em', marginBottom: 6 }}>Orchestrator</div>
          <div style={{ fontSize: 18, fontWeight: 700, color: orch?.running ? C.green : C.red }}>
            {orch ? (orch.running ? 'RUNNING' : 'STOPPED') : '—'}
          </div>
          <div style={{ fontSize: 10, color: C.muted, marginTop: 3 }}>{orch?.scheduled_jobs ?? 0} scheduled jobs</div>
        </div>
        <div style={s.kpi}>
          <div style={{ fontSize: 10, color: C.muted, textTransform: 'uppercase', letterSpacing: '.06em', marginBottom: 6 }}>Trading</div>
          <div style={{ fontSize: 18, fontWeight: 700, color: isTrading ? C.green : C.yellow }}>
            {isTrading ? 'ACTIVE' : 'PAUSED'}
          </div>
          <div style={{ fontSize: 10, color: C.muted, marginTop: 3 }}>Agent pipeline</div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 12 }}>
        {/* Controls */}
        <div style={s.card}>
          <div style={s.cardTitle}>Controls</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            <CtrlBtn label="Run One Cycle" color={C.blue} busy={busy}
              onClick={() => act('Run One Cycle', api.triggerCycle, 'Cycle started — watch session log')} />
            <CtrlBtn label="Retrain ML Model" color={C.accent} busy={busy}
              onClick={() => act('Retrain ML Model', api.triggerRetraining, 'Retraining started')} />
            {isTrading
              ? <CtrlBtn label="Pause Trading" color={C.yellow} busy={busy}
                  onClick={() => act('Pause Trading', api.pauseTrading, 'Trading paused')} />
              : <CtrlBtn label="Resume Trading" color={C.green} busy={busy}
                  onClick={() => act('Resume Trading', api.resumeTrading, 'Trading resumed')} />
            }
            <button onClick={loadAll} style={{ ...btnStyle, padding: '6px 12px', color: C.muted }}>Refresh</button>
          </div>
        </div>

        {/* Agent health */}
        <div style={s.card}>
          <div style={s.cardTitle}>Agent Status</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {orch && Object.keys(orch.agents).length > 0
              ? Object.entries(orch.agents).map(([name, status]) => (
                <div key={name} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: 12, color: C.text }}>{name.replace(/_/g, ' ')}</span>
                  <span style={{
                    fontSize: 10, fontWeight: 600, padding: '2px 8px', borderRadius: 10,
                    background: `${agentColor(status)}1a`, color: agentColor(status),
                    border: `1px solid ${agentColor(status)}4d`,
                    textTransform: 'uppercase',
                  }}>{status}</span>
                </div>
              ))
              : <div style={{ color: C.muted, fontSize: 11 }}>No agents registered — app may be starting up</div>
            }
            {orch && Object.keys(orch.queues ?? {}).length > 0 && (
              <div style={{ marginTop: 8, paddingTop: 8, borderTop: `1px solid ${C.border}` }}>
                <div style={{ fontSize: 10, color: C.muted, textTransform: 'uppercase', marginBottom: 6 }}>Queue Depths</div>
                {Object.entries(orch.queues).map(([q, n]) => (
                  <div key={q} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 4 }}>
                    <span style={{ color: C.muted }}>{q.replace(/_/g, ' ')}</span>
                    <span style={{ color: n > 0 ? C.yellow : C.muted }}>{n}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Scheduler jobs */}
      <div style={{ ...s.card, marginBottom: 12 }}>
        <div style={s.cardTitle}>Scheduled Jobs</div>
        {jobs.length === 0
          ? <div style={{ color: C.muted, fontSize: 11 }}>No jobs — scheduler may not be running</div>
          : (
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead><tr>
                {['Job', 'Next Run', 'Status', 'Action'].map(h => <th key={h} style={s.th}>{h}</th>)}
              </tr></thead>
              <tbody>
                {jobs.map(j => (
                  <tr key={j.id}>
                    <td style={{ ...s.td, color: C.accent }}>{j.id}</td>
                    <td style={{ ...s.td, color: C.muted }}>{j.next_run_time ? fmtTs(j.next_run_time) : '—'}</td>
                    <td style={s.td}>
                      <span style={{
                        fontSize: 10, fontWeight: 600, padding: '2px 6px', borderRadius: 8,
                        background: j.paused ? `${C.yellow}1a` : `${C.green}1a`,
                        color: j.paused ? C.yellow : C.green,
                        border: `1px solid ${j.paused ? C.yellow : C.green}4d`,
                      }}>{j.paused ? 'PAUSED' : 'ACTIVE'}</span>
                    </td>
                    <td style={s.td}>
                      <button
                        onClick={() => act(j.id, () => j.paused ? api.resumeJob(j.id) : api.pauseJob(j.id), `Job ${j.id} ${j.paused ? 'resumed' : 'paused'}`)}
                        disabled={busy !== null}
                        style={{ ...btnStyle, fontSize: 10, padding: '2px 8px', color: j.paused ? C.green : C.yellow }}
                      >{j.paused ? 'Resume' : 'Pause'}</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
      </div>

      {/* Session log */}
      <div style={s.card}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
          <div style={s.cardTitle}>Session Log</div>
          <button onClick={loadAll} style={{ ...btnStyle, fontSize: 10 }}>Refresh</button>
        </div>
        <div style={{ maxHeight: 300, overflowY: 'auto' }}>
          {log.length === 0
            ? <div style={{ color: C.muted, fontSize: 11, textAlign: 'center', padding: 20 }}>No events yet</div>
            : log.map((e, i) => (
              <div key={i} style={{ display: 'flex', gap: 10, padding: '6px 0', borderBottom: `1px solid rgba(255,255,255,.04)`, fontSize: 11 }}>
                <span style={{ color: C.muted, whiteSpace: 'nowrap', minWidth: 140 }}>{e.timestamp.slice(0, 19)}</span>
                <span style={{ color: logColor(e.level), minWidth: 52, fontWeight: 600 }}>{e.level}</span>
                <span style={{ color: C.text, flex: 1 }}>{e.message}</span>
              </div>
            ))}
        </div>
      </div>
    </div>
  )
}

function CtrlBtn({ label, color, busy, onClick }: {
  label: string; color: string; busy: string | null; onClick: () => void
}) {
  const isMe = busy === label
  return (
    <button onClick={onClick} disabled={busy !== null} style={{
      background: `${color}1a`, border: `1px solid ${color}4d`, color,
      padding: '10px 16px', borderRadius: 6, cursor: busy ? 'not-allowed' : 'pointer',
      fontFamily: 'inherit', fontSize: 12, fontWeight: 600, letterSpacing: '.04em',
      opacity: busy && !isMe ? 0.5 : 1, textAlign: 'left',
    }}>
      {isMe ? '⟳ Working...' : label}
    </button>
  )
}

// ── Readiness Panel ───────────────────────────────────────────────────────────
function ReadinessPanel() {
  const [report, setReport] = useState<ReadinessReport | null>(null)
  const [loading, setLoading] = useState(false)

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const r = await api.readiness() as ReadinessReport
      setReport(r)
    } catch { /* ignore */ } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { load() }, [load])

  const iconFor = (item: ReadinessCheckItem) => {
    if (item.passed) return { sym: '✓', color: C.green }
    if (item.check === 'smtp_configured' || item.check === 'slack_configured')
      return { sym: '⚠', color: C.yellow }
    return { sym: '✗', color: C.red }
  }

  return (
    <div style={{ maxWidth: 720 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <div>
          <div style={{ fontSize: 15, fontWeight: 700, color: C.accent, marginBottom: 4 }}>Live Trading Readiness</div>
          <div style={{ fontSize: 11, color: C.muted }}>Run this checklist before switching TRADING_MODE=live</div>
        </div>
        <button onClick={load} disabled={loading} style={{ ...btnStyle, padding: '6px 16px' }}>
          {loading ? 'Checking...' : 'Re-run checks'}
        </button>
      </div>

      {report && (
        <>
          {/* Summary badge */}
          <div style={{
            ...s.card, marginBottom: 16, textAlign: 'center',
            borderColor: report.ready ? `${C.green}66` : `${C.red}66`,
            background: report.ready ? `${C.green}0d` : `${C.red}0d`,
          }}>
            <div style={{ fontSize: 22, fontWeight: 700, color: report.ready ? C.green : C.red, marginBottom: 4 }}>
              {report.ready ? '✓ READY FOR LIVE TRADING' : '✗ NOT READY'}
            </div>
            <div style={{ fontSize: 11, color: C.muted }}>{report.summary} · {report.timestamp.slice(0, 19)} UTC</div>
          </div>

          {/* All checks */}
          <div style={s.card}>
            <div style={s.cardTitle}>Checks</div>
            {report.all_checks.map((item, i) => {
              const { sym, color } = iconFor(item)
              return (
                <div key={i} style={{
                  display: 'flex', gap: 12, alignItems: 'flex-start',
                  padding: '10px 0', borderBottom: i < report.all_checks.length - 1 ? `1px solid rgba(255,255,255,.04)` : 'none',
                }}>
                  <span style={{ color, fontWeight: 700, fontSize: 14, minWidth: 16 }}>{sym}</span>
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: 12, fontWeight: 600, color: item.passed ? C.text : color, marginBottom: 2 }}>
                      {item.check.replace(/_/g, ' ')}
                      {item.value != null && (
                        <span style={{ fontWeight: 400, color: C.muted, marginLeft: 8 }}>[{String(item.value)}]</span>
                      )}
                    </div>
                    <div style={{ fontSize: 11, color: C.muted }}>{item.detail}</div>
                  </div>
                </div>
              )
            })}
          </div>

          {report.ready && (
            <div style={{ ...s.card, marginTop: 16, borderColor: `${C.green}66` }}>
              <div style={s.cardTitle}>Next Steps to Go Live</div>
              <ol style={{ paddingLeft: 20, fontSize: 12, lineHeight: 2, color: C.muted }}>
                <li>Change <span style={{ color: C.accent }}>TRADING_MODE=live</span> in your <code>.env</code></li>
                <li>Update <span style={{ color: C.accent }}>ALPACA_BASE_URL=https://api.alpaca.markets</span></li>
                <li>Restart the app: <span style={{ color: C.accent }}>make down && make up</span></li>
                <li>Monitor the dashboard closely for the first hour</li>
                <li>Keep Stage 1 capital ($1k) for at least 2 weeks before advancing</li>
              </ol>
            </div>
          )}
        </>
      )}

      {!report && !loading && (
        <div style={{ ...s.card, textAlign: 'center', color: C.muted, padding: 40 }}>
          Click "Re-run checks" to start the readiness check
        </div>
      )}
    </div>
  )
}

// ── Error Boundary ─────────────────────────────────────────────────────────────
class ErrorBoundary extends Component<{ children: ReactNode }, { error: string | null }> {
  constructor(props: { children: ReactNode }) {
    super(props)
    this.state = { error: null }
  }
  static getDerivedStateFromError(e: unknown) {
    return { error: e instanceof Error ? e.message : String(e) }
  }
  render() {
    if (this.state.error) {
      return (
        <div style={{ padding: 32, color: C.red, textAlign: 'center' }}>
          <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 8 }}>Panel failed to render</div>
          <div style={{ fontSize: 12, color: C.muted }}>{this.state.error}</div>
          <button onClick={() => this.setState({ error: null })} style={{ ...btnStyle, marginTop: 16 }}>Retry</button>
        </div>
      )
    }
    return this.props.children
  }
}

// ── Analytics Panel ────────────────────────────────────────────────────────────
function AnalyticsPanel() {
  const [attribution, setAttribution] = useState<AttributionItem[]>([])
  const [days, setDays] = useState(90)
  const [nisCtx, setNisCtx] = useState<NisMacroContext | null>(null)
  const [nisSignals, setNisSignals] = useState<NisSignal[]>([])
  const [gateSummary, setGateSummary] = useState<GateSummaryRow[]>([])
  const [analyticsTab, setAnalyticsTab] = useState<'attribution' | 'nis' | 'gates'>('attribution')

  const load = useCallback(async (d = days) => {
    try {
      const j = await api.signalAttribution(d) as { attribution?: unknown } | AttributionItem[]
      const raw = (j as { attribution?: unknown }).attribution ?? j
      const arr: AttributionItem[] = Array.isArray(raw)
        ? raw
        : Object.entries(raw as Record<string, Record<string, number>>).map(([k, v]) => ({
            signal_type: k,
            count: v.trades ?? v.count ?? 0,
            win_rate: (v.win_rate ?? 0) / 100,
            avg_pnl: v.avg_pnl ?? 0,
            total_pnl: v.total_pnl ?? 0,
          }))
      setAttribution(arr)
    } catch { /* ignore */ }
  }, [days])

  useEffect(() => { load() }, [load])

  useEffect(() => {
    api.nisMacro().then((j: unknown) => {
      const ctx = j as NisMacroContext & { status?: string }
      if (!ctx.status) setNisCtx(ctx)
    }).catch(() => {})
    api.nisSignals().then((j: unknown) => {
      const res = j as { signals?: NisSignal[] }
      setNisSignals(res.signals ?? [])
    }).catch(() => {})
    api.decisionAuditSummary().then((j: unknown) => {
      const res = j as { gate_summary?: GateSummaryRow[] }
      setGateSummary(res.gate_summary ?? [])
    }).catch(() => {})
  }, [])

  const totalPnl = attribution.reduce((s, a) => s + a.total_pnl, 0)

  return (
    <div>
      {/* Sub-tab selector */}
      <div style={{ display: 'flex', gap: 6, marginBottom: 16 }}>
        {([['attribution', 'Signal Attribution'], ['nis', 'NIS Intelligence'], ['gates', 'Gate Calibration']] as const).map(([key, label]) => (
          <button key={key} onClick={() => setAnalyticsTab(key)} style={{
            padding: '6px 16px', borderRadius: 4, fontSize: 12, fontWeight: 600, cursor: 'pointer',
            border: `1px solid ${analyticsTab === key ? C.accent : C.border}`,
            background: analyticsTab === key ? `${C.accent}22` : 'transparent',
            color: analyticsTab === key ? C.accent : C.muted,
          }}>{label}</button>
        ))}
      </div>

      {analyticsTab === 'attribution' && <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <div style={{ fontSize: 15, fontWeight: 700, color: C.accent }}>Signal Attribution</div>
        <div style={{ display: 'flex', gap: 8 }}>
          {[30, 90, 180].map(d => (
            <button key={d} onClick={() => { setDays(d); load(d) }} style={{
              ...btnStyle,
              color: days === d ? C.accent : C.muted,
              borderColor: days === d ? C.accent : C.border,
            }}>{d}d</button>
          ))}
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 16 }}>
        <KpiCard label="Total Signals Traded" value={String(attribution.reduce((s, a) => s + a.count, 0))} />
        <KpiCard label={`Total P&L (${days}d)`} value={fmt$(totalPnl)} color={clr(totalPnl)} />
      </div>

      <div style={s.card}>
        <div style={s.cardTitle}>Performance by Signal Type</div>
        {attribution.length === 0
          ? <div style={{ color: C.muted, textAlign: 'center', padding: 30, fontSize: 11 }}>No data for this period</div>
          : (
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead><tr>
                {['Signal Type', 'Trades', 'Win Rate', 'Avg P&L', 'Total P&L'].map(h => (
                  <th key={h} style={s.th}>{h}</th>
                ))}
              </tr></thead>
              <tbody>
                {attribution.map((a, i) => (
                  <tr key={i}>
                    <td style={{ ...s.td, color: C.blue, fontWeight: 600 }}>{a.signal_type}</td>
                    <td style={s.td}>{a.count}</td>
                    <td style={{ ...s.td, color: clr(a.win_rate - 0.5) }}>{(a.win_rate * 100).toFixed(1)}%</td>
                    <td style={{ ...s.td, color: clr(a.avg_pnl) }}>{fmt$(a.avg_pnl)}</td>
                    <td style={{ ...s.td, color: clr(a.total_pnl) }}>{fmt$(a.total_pnl)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
      </div>
      </div>}

      {analyticsTab === 'nis' && <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        {/* Macro context card */}
        {nisCtx ? (
          <div style={s.card}>
            <div style={s.cardTitle}>Today's NIS Macro Context</div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 12, marginBottom: 12 }}>
              <KpiCard label="Overall Risk" value={nisCtx.overall_risk}
                color={nisCtx.overall_risk === 'HIGH' ? C.red : nisCtx.overall_risk === 'MEDIUM' ? C.yellow : C.green} />
              <KpiCard label="Sizing Factor" value={nisCtx.global_sizing_factor.toFixed(2) + '×'}
                color={nisCtx.global_sizing_factor < 1 ? C.yellow : C.green} />
              <KpiCard label="Block Entries" value={nisCtx.block_new_entries ? 'YES' : 'NO'}
                color={nisCtx.block_new_entries ? C.red : C.green} />
              <KpiCard label="Events Today" value={String(nisCtx.events_today.length)} />
            </div>
            {nisCtx.rationale && (
              <div style={{ fontSize: 11, color: C.muted, padding: '8px 0', borderTop: `1px solid ${C.border}` }}>
                {nisCtx.rationale}
              </div>
            )}
            {nisCtx.events_today.length > 0 && (
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11, marginTop: 8 }}>
                <thead><tr>
                  {['Event', 'Time', 'Risk', 'Direction', 'Sizing', 'Block', 'Summary'].map(h =>
                    <th key={h} style={s.th}>{h}</th>)}
                </tr></thead>
                <tbody>
                  {nisCtx.events_today.map((e, i) => (
                    <tr key={i}>
                      <td style={{ ...s.td, color: C.text, fontWeight: 600 }}>{e.event_type}</td>
                      <td style={{ ...s.td, color: C.muted }}>{e.event_time ?? '—'}</td>
                      <td style={{ ...s.td, color: e.risk_level === 'HIGH' ? C.red : e.risk_level === 'MEDIUM' ? C.yellow : C.green }}>
                        {e.risk_level}
                      </td>
                      <td style={{ ...s.td, color: e.direction === 'BULLISH' ? C.green : e.direction === 'BEARISH' ? C.red : C.muted }}>
                        {e.direction}
                      </td>
                      <td style={{ ...s.td, color: e.sizing_factor < 1 ? C.yellow : C.muted }}>{e.sizing_factor.toFixed(2)}×</td>
                      <td style={{ ...s.td, color: e.block_new_entries ? C.red : C.muted }}>{e.block_new_entries ? 'Yes' : 'No'}</td>
                      <td style={{ ...s.td, color: C.muted, fontSize: 10 }}>{e.consensus_summary ?? e.rationale ?? '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        ) : (
          <div style={{ ...s.card, color: C.muted, textAlign: 'center', padding: 24, fontSize: 11 }}>
            Macro context not yet available — premarket routine runs at 09:00 ET
          </div>
        )}

        {/* Stock signals */}
        <div style={s.card}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
            <div style={s.cardTitle}>Cached Stock Signals ({nisSignals.length})</div>
            <div style={{ display: 'flex', gap: 8, fontSize: 10, color: C.muted }}>
              {nisSignals.filter(s2 => s2.action_policy === 'block_entry').length > 0 &&
                <span style={{ color: C.red }}>🚫 {nisSignals.filter(s2 => s2.action_policy === 'block_entry').length} blocked</span>}
              {nisSignals.filter(s2 => s2.action_policy?.includes('size_down')).length > 0 &&
                <span style={{ color: C.yellow }}>↓ {nisSignals.filter(s2 => s2.action_policy?.includes('size_down')).length} sized down</span>}
            </div>
          </div>
          <div style={{ maxHeight: 420, overflowY: 'auto' }}>
            <NisSignalsTable signals={nisSignals} />
          </div>
        </div>
      </div>}

      {analyticsTab === 'gates' && <div style={s.card}>
        <div style={s.cardTitle}>Gate Calibration — Did each block reason protect us?</div>
        <div style={{ fontSize: 11, color: C.muted, marginBottom: 12 }}>
          Positive avg missed P&amp;L = gate blocked winners (recalibrate). Negative = gate correctly blocked losers. Needs ~2 weeks of backfilled outcomes.
        </div>
        <GateSummaryTable rows={gateSummary} />
      </div>}
    </div>
  )
}

// ── Agent Config Panel ─────────────────────────────────────────────────────────
function ConfigPanel({ toast }: { toast: (m: string, t?: 'success' | 'error' | 'warning' | 'info') => void }) {
  const [schema, setSchema] = useState<ConfigSchemaEntry[]>([])
  const [values, setValues] = useState<Record<string, number>>({})
  const [edits, setEdits] = useState<Record<string, string>>({})
  const [saving, setSaving] = useState<Record<string, boolean>>({})

  const load = useCallback(async () => {
    try {
      const [s, v] = await Promise.all([
        api.configSchema() as Promise<{ schema: ConfigSchemaEntry[] }>,
        api.configValues() as Promise<{ config: Record<string, number> }>,
      ])
      setSchema(s.schema)
      setValues(v.config)
      // Init edits with current values
      const e: Record<string, string> = {}
      for (const [k, val] of Object.entries(v.config)) e[k] = String(val)
      setEdits(e)
    } catch { /* ignore */ }
  }, [])

  useEffect(() => { load() }, [load])

  const save = async (key: string) => {
    const raw = edits[key]
    const numVal = parseFloat(raw)
    if (isNaN(numVal)) { toast(`Invalid value for ${key}`, 'error'); return }
    setSaving(s => ({ ...s, [key]: true }))
    try {
      await api.configUpdate(key, numVal)
      setValues(v => ({ ...v, [key]: numVal }))
      toast(`Saved ${key} = ${numVal}`, 'success')
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e)
      toast(`${key}: ${msg}`, 'error')
      setEdits(ed => ({ ...ed, [key]: String(values[key] ?? '') }))
    } finally {
      setSaving(s => ({ ...s, [key]: false }))
    }
  }

  const reset = async () => {
    try {
      await api.configReset()
      toast('All settings reset to defaults', 'info')
      load()
    } catch { toast('Reset failed', 'error') }
  }

  const isDirty = (key: string) => String(values[key] ?? '') !== edits[key]

  // Group entries
  const groups = schema.reduce<Record<string, ConfigSchemaEntry[]>>((acc, s) => {
    (acc[s.group] ??= []).push(s)
    return acc
  }, {})

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <div style={{ fontSize: 13, color: C.muted }}>
          Changes take effect on the next agent cycle — no restart needed.
        </div>
        <button onClick={reset} style={{ ...btnStyle, borderColor: C.yellow, color: C.yellow }}>
          Reset All to Defaults
        </button>
      </div>

      {Object.entries(groups).map(([group, entries]) => (
        <div key={group} style={{ ...s.card, marginBottom: 12 }}>
          <div style={s.cardTitle}>{group}</div>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead><tr>
              {['Parameter', 'Description', 'Range', 'Value', ''].map(h => (
                <th key={h} style={s.th}>{h}</th>
              ))}
            </tr></thead>
            <tbody>
              {entries.map(entry => {
                const dirty = isDirty(entry.key)
                const isSaving = saving[entry.key]
                return (
                  <tr key={entry.key}>
                    <td style={{ ...s.td, color: C.accent, fontFamily: 'monospace', fontSize: 11 }}>
                      {entry.key}
                    </td>
                    <td style={{ ...s.td, color: C.muted, maxWidth: 280 }}>{entry.description}</td>
                    <td style={{ ...s.td, color: C.muted, whiteSpace: 'nowrap' as const }}>
                      {entry.min} – {entry.max}
                    </td>
                    <td style={s.td}>
                      <input
                        type="number"
                        value={edits[entry.key] ?? String(entry.default)}
                        step={entry.type === 'int' ? 1 : 0.01}
                        min={entry.min}
                        max={entry.max}
                        onChange={e => setEdits(ed => ({ ...ed, [entry.key]: e.target.value }))}
                        onKeyDown={e => e.key === 'Enter' && save(entry.key)}
                        style={{
                          width: 90, background: C.surface2,
                          border: `1px solid ${dirty ? C.yellow : C.border}`,
                          color: dirty ? C.yellow : C.text,
                          borderRadius: 4, padding: '4px 8px', fontSize: 12, outline: 'none',
                        }}
                      />
                      {entry.default !== undefined && (
                        <span style={{ marginLeft: 6, fontSize: 10, color: C.muted }}>
                          def: {entry.default}
                        </span>
                      )}
                    </td>
                    <td style={s.td}>
                      {dirty && (
                        <button
                          onClick={() => save(entry.key)}
                          disabled={isSaving}
                          style={{ ...btnStyle, fontSize: 10, padding: '3px 10px',
                            borderColor: C.green, color: C.green }}
                        >
                          {isSaving ? '…' : 'Save'}
                        </button>
                      )}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      ))}
    </div>
  )
}

// ── Performance Review Panel ───────────────────────────────────────────────────
function PerformanceReviewPanel() {
  const [data, setData] = useState<PerformanceReview | null>(null)
  const [days, setDays] = useState(30)
  const [loading, setLoading] = useState(false)
  const [dailyRows, setDailyRows] = useState<DailySummaryRow[]>([])

  const load = useCallback(async (d: number) => {
    setLoading(true)
    try {
      const j = await api.performanceReview(d) as PerformanceReview
      setData(j)
    } catch { /* ignore */ }
    finally { setLoading(false) }
  }, [])

  useEffect(() => {
    // Fetch daily summary rows from risk_metrics (populated by EOD job)
    fetch('/api/dashboard/analytics/daily-summary?days=30')
      .then(r => r.ok ? r.json() : null)
      .then((j: unknown) => {
        const arr = (j as { rows?: DailySummaryRow[] })?.rows ?? []
        setDailyRows(Array.isArray(arr) ? arr : [])
      })
      .catch(() => {})
  }, [])

  useEffect(() => { load(days) }, [load, days])

  const statusColor = (s: string) =>
    s === 'ok' ? C.green : s === 'warn' ? C.yellow : C.red

  const driftRow = (item: DriftItem) => (
    <tr key={item.metric}>
      <td style={s.td}>{item.metric}</td>
      <td style={{ ...s.td, color: C.accent }}>{item.live.toFixed(2)}</td>
      <td style={{ ...s.td, color: C.muted }}>{item.target.toFixed(2)}</td>
      <td style={{ ...s.td, color: item.delta >= 0 ? C.green : C.red }}>
        {item.delta >= 0 ? '+' : ''}{item.delta.toFixed(2)}
      </td>
      <td style={s.td}>
        <span style={{
          fontSize: 10, padding: '2px 8px', borderRadius: 8, fontWeight: 600,
          background: `${statusColor(item.status)}1a`, color: statusColor(item.status),
          border: `1px solid ${statusColor(item.status)}4d`,
        }}>{item.status.toUpperCase()}</span>
      </td>
    </tr>
  )

  if (!data) return (
    <div style={{ color: C.muted, textAlign: 'center', padding: 40 }}>
      {loading ? 'Loading…' : 'No data'}
    </div>
  )

  const overallColor = statusColor(data.overall_status)

  return (
    <div>
      {/* Controls */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 16, alignItems: 'center' }}>
        {[7, 30, 90].map(d => (
          <button key={d} onClick={() => setDays(d)} style={{
            ...btnStyle,
            borderColor: days === d ? C.accent : undefined,
            color: days === d ? C.accent : undefined,
          }}>{d}d</button>
        ))}
        <button onClick={() => load(days)} disabled={loading} style={btnStyle}>Refresh</button>
        <span style={{ marginLeft: 'auto', fontSize: 11, color: C.muted }}>
          {data.start_date} → {data.end_date}
        </span>
      </div>

      {/* Status banner */}
      <div style={{
        ...s.card, marginBottom: 16,
        borderColor: overallColor,
        background: `${overallColor}08`,
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ fontSize: 13, fontWeight: 700, color: overallColor }}>
            {data.overall_status === 'ok' ? '✓ On track vs backtest'
              : data.overall_status === 'warn' ? '⚠ Minor drift detected'
              : '⚡ Significant drift — review needed'}
          </span>
          <span style={{ fontSize: 11, color: C.muted }}>
            {data.alerts} alert{data.alerts !== 1 ? 's' : ''} · {data.warnings} warning{data.warnings !== 1 ? 's' : ''}
          </span>
        </div>
      </div>

      {/* KPIs */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 12, marginBottom: 16 }}>
        <KpiCard label="Trades" value={String(data.total_trades)} sub={`${data.wins} wins`} />
        <KpiCard label="Win Rate" value={data.win_rate_pct.toFixed(1) + '%'}
          color={data.win_rate_pct >= 50 ? C.green : C.red} sub="target 55%" />
        <KpiCard label="Total P&L" value={fmt$(data.total_pnl)} color={clr(data.total_pnl)} />
        <KpiCard label="Avg P&L / Trade" value={fmt$(data.avg_pnl_per_trade)} color={clr(data.avg_pnl_per_trade)} />
        <KpiCard label="Sharpe (est.)" value={data.sharpe_estimate != null ? data.sharpe_estimate.toFixed(2) : '—'}
          color={data.sharpe_estimate != null ? (data.sharpe_estimate >= 1 ? C.green : C.yellow) : undefined} />
        <KpiCard label="Alpha vs SPY"
          value={data.alpha_pct != null ? (data.alpha_pct >= 0 ? '+' : '') + data.alpha_pct.toFixed(2) + '%' : '—'}
          color={data.alpha_pct != null ? clr(data.alpha_pct) : undefined}
          sub={`SPY: ${data.spy_return_pct >= 0 ? '+' : ''}${data.spy_return_pct.toFixed(2)}%`} />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
        {/* Drift vs backtest */}
        <div style={s.card}>
          <div style={s.cardTitle}>Drift vs Backtest Targets</div>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead><tr>
              {['Metric', 'Live', 'Target', 'Delta', 'Status'].map(h => (
                <th key={h} style={s.th}>{h}</th>
              ))}
            </tr></thead>
            <tbody>
              {data.drift.length === 0
                ? <tr><td colSpan={5} style={{ ...s.td, color: C.muted, textAlign: 'center' }}>No data yet</td></tr>
                : data.drift.map(driftRow)}
            </tbody>
          </table>
        </div>

        {/* Per-signal breakdown */}
        <div style={s.card}>
          <div style={s.cardTitle}>By Signal Type</div>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead><tr>
              {['Signal', 'Trades', 'Win%', 'Avg P&L', 'Total'].map(h => (
                <th key={h} style={s.th}>{h}</th>
              ))}
            </tr></thead>
            <tbody>
              {Object.keys(data.by_signal).length === 0
                ? <tr><td colSpan={5} style={{ ...s.td, color: C.muted, textAlign: 'center' }}>No trades yet</td></tr>
                : Object.entries(data.by_signal)
                    .sort((a, b) => b[1].total_pnl - a[1].total_pnl)
                    .map(([sig, g]) => (
                    <tr key={sig}>
                      <td style={{ ...s.td, color: C.accent }}>{sig}</td>
                      <td style={s.td}>{g.trades}</td>
                      <td style={{ ...s.td, color: g.win_rate >= 50 ? C.green : C.red }}>{g.win_rate}%</td>
                      <td style={{ ...s.td, color: clr(g.avg_pnl) }}>{fmt$(g.avg_pnl)}</td>
                      <td style={{ ...s.td, color: clr(g.total_pnl) }}>{fmt$(g.total_pnl)}</td>
                    </tr>
                  ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Daily Summary Table */}
      <div style={{ ...s.card, marginTop: 12 }}>
        <div style={s.cardTitle}>Daily Summary (EOD job · last 30 days)</div>
        {dailyRows.length === 0 ? (
          <div style={{ color: C.muted, textAlign: 'center', padding: 20, fontSize: 11 }}>
            No daily summary yet — EOD job runs at 16:30 ET
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
              <thead><tr>
                {['Date', 'Total P&L', 'Swing P&L', 'Intraday P&L', 'Trades', 'Win Rate', 'Block Rate', 'NIS Blocks', 'Macro Blocks', 'Corr Blocks'].map(h =>
                  <th key={h} style={s.th}>{h}</th>)}
              </tr></thead>
              <tbody>
                {dailyRows.map((r, i) => (
                  <tr key={i}>
                    <td style={{ ...s.td, color: C.muted, whiteSpace: 'nowrap' }}>{r.date}</td>
                    <td style={{ ...s.td, color: clr(r.daily_pnl), fontWeight: 600 }}>
                      {r.daily_pnl != null ? (r.daily_pnl >= 0 ? '+' : '') + '$' + Math.abs(r.daily_pnl).toFixed(2) : '—'}
                    </td>
                    <td style={{ ...s.td, color: clr(r.swing_pnl) }}>{(r.swing_pnl >= 0 ? '+' : '') + '$' + Math.abs(r.swing_pnl).toFixed(2)}</td>
                    <td style={{ ...s.td, color: clr(r.intraday_pnl) }}>{(r.intraday_pnl >= 0 ? '+' : '') + '$' + Math.abs(r.intraday_pnl).toFixed(2)}</td>
                    <td style={s.td}>{r.swing_trades + r.intraday_trades}</td>
                    <td style={{ ...s.td, color: r.swing_win_rate != null ? clr(r.swing_win_rate - 0.5) : C.muted }}>
                      {r.swing_win_rate != null ? (r.swing_win_rate * 100).toFixed(0) + '%' : '—'}
                    </td>
                    <td style={{ ...s.td, color: r.block_rate > 0.5 ? C.yellow : C.muted }}>{(r.block_rate * 100).toFixed(0)}%</td>
                    <td style={{ ...s.td, color: r.nis_blocks > 0 ? C.yellow : C.muted }}>{r.nis_blocks}</td>
                    <td style={{ ...s.td, color: r.macro_blocks > 0 ? C.yellow : C.muted }}>{r.macro_blocks}</td>
                    <td style={{ ...s.td, color: r.correlation_blocks > 0 ? C.yellow : C.muted }}>{r.correlation_blocks}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}

// ── Watchlist Panel ────────────────────────────────────────────────────────────
function WatchlistPanel({ toast }: { toast: (m: string, t?: 'success' | 'error' | 'warning' | 'info') => void }) {
  const [tickers, setTickers] = useState<import('./types').WatchlistTicker[]>([])
  const [newSym, setNewSym] = useState('')
  const [newSector, setNewSector] = useState('')
  const [loading, setLoading] = useState(false)

  const load = useCallback(async () => {
    try {
      const j = await api.watchlist() as { tickers: import('./types').WatchlistTicker[] }
      setTickers(j.tickers ?? [])
    } catch { /* ignore */ }
  }, [])

  useEffect(() => { load() }, [load])

  const addTicker = async () => {
    const sym = newSym.trim().toUpperCase()
    if (!sym) return
    setLoading(true)
    try {
      await api.watchlistAdd(sym, newSector.trim() || undefined)
      toast(`Added ${sym}`, 'success')
      setNewSym(''); setNewSector('')
      load()
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e)
      toast(`Failed to add ${sym}: ${msg}`, 'error')
    } finally { setLoading(false) }
  }

  const toggleActive = async (sym: string, current: boolean) => {
    try {
      await api.watchlistPatch(sym, { active: !current })
      setTickers(ts => ts.map(t => t.symbol === sym ? { ...t, active: !current } : t))
    } catch { toast(`Failed to update ${sym}`, 'error') }
  }

  const removeTicker = async (sym: string) => {
    try {
      await api.watchlistDelete(sym)
      toast(`Removed ${sym}`, 'info')
      setTickers(ts => ts.filter(t => t.symbol !== sym))
    } catch { toast(`Failed to remove ${sym}`, 'error') }
  }

  const bulkLoad = async () => {
    setLoading(true)
    try {
      const j = await api.watchlistBulk() as { total_added: number }
      toast(`Loaded S&P 100 — ${j.total_added} added`, 'success')
      load()
    } catch { toast('Bulk load failed', 'error') }
    finally { setLoading(false) }
  }

  const active = tickers.filter(t => t.active).length
  const inactive = tickers.length - active

  return (
    <div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 12, marginBottom: 16 }}>
        <KpiCard label="Total Tickers" value={String(tickers.length)} />
        <KpiCard label="Active" value={String(active)} color={C.green} />
        <KpiCard label="Disabled" value={String(inactive)} color={inactive > 0 ? C.yellow : C.muted} />
      </div>

      <div style={s.card}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
          <div style={s.cardTitle}>Add Ticker</div>
          <button onClick={bulkLoad} disabled={loading} style={btnStyle}>Load S&amp;P 100</button>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <input
            value={newSym} onChange={e => setNewSym(e.target.value.toUpperCase())}
            placeholder="Symbol (e.g. TSLA)" maxLength={10}
            style={{ flex: 1, background: C.surface2, border: `1px solid ${C.border}`, color: C.text,
              borderRadius: 4, padding: '6px 10px', fontSize: 12, outline: 'none' }}
            onKeyDown={e => e.key === 'Enter' && addTicker()}
          />
          <input
            value={newSector} onChange={e => setNewSector(e.target.value)}
            placeholder="Sector (optional)"
            style={{ flex: 1, background: C.surface2, border: `1px solid ${C.border}`, color: C.text,
              borderRadius: 4, padding: '6px 10px', fontSize: 12, outline: 'none' }}
          />
          <button onClick={addTicker} disabled={loading || !newSym.trim()} style={btnStyle}>Add</button>
        </div>
      </div>

      <div style={{ ...s.card, marginTop: 12 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
          <div style={s.cardTitle}>Ticker Universe ({tickers.length})</div>
          <button onClick={load} style={btnStyle}>Refresh</button>
        </div>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead><tr>
              {['Symbol', 'Sector', 'Status', 'Added', 'Actions'].map(h => (
                <th key={h} style={s.th}>{h}</th>
              ))}
            </tr></thead>
            <tbody>
              {tickers.length === 0
                ? <tr><td colSpan={5} style={{ ...s.td, textAlign: 'center', color: C.muted, padding: 20 }}>
                    No tickers — click "Load S&P 100" to seed defaults
                  </td></tr>
                : tickers.map(t => (
                  <tr key={t.symbol}>
                    <td style={{ ...s.td, color: C.accent, fontWeight: 700 }}>{t.symbol}</td>
                    <td style={{ ...s.td, color: C.muted }}>{t.sector ?? '—'}</td>
                    <td style={s.td}>
                      <span style={{
                        fontSize: 10, padding: '2px 8px', borderRadius: 8, fontWeight: 600,
                        background: t.active ? `${C.green}1a` : `${C.muted}1a`,
                        color: t.active ? C.green : C.muted,
                        border: `1px solid ${t.active ? C.green : C.muted}4d`,
                      }}>{t.active ? 'Active' : 'Disabled'}</span>
                    </td>
                    <td style={{ ...s.td, color: C.muted, fontSize: 10 }}>
                      {t.added_at ? new Date(t.added_at).toLocaleDateString() : '—'}
                    </td>
                    <td style={s.td}>
                      <div style={{ display: 'flex', gap: 6 }}>
                        <button onClick={() => toggleActive(t.symbol, t.active)}
                          style={{ ...btnStyle, fontSize: 10, padding: '3px 8px' }}>
                          {t.active ? 'Disable' : 'Enable'}
                        </button>
                        <button onClick={() => removeTicker(t.symbol)}
                          style={{ ...btnStyle, fontSize: 10, padding: '3px 8px', borderColor: C.red, color: C.red }}>
                          Remove
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

// ── Monitor Panel ─────────────────────────────────────────────────────────────
function MonitorPanel() {
  const [health, setHealth] = useState<Record<string, unknown> | null>(null)
  const [summary, setSummary] = useState<Record<string, unknown> | null>(null)
  const [history, setHistory] = useState<Record<string, unknown>[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(async () => {
    setLoading(true); setError(null)
    try {
      const [h, s, hist] = await Promise.all([
        api.monitorHealth() as Promise<Record<string, unknown>>,
        api.monitorSummary() as Promise<Record<string, unknown>>,
        api.monitorHistory(7) as Promise<{ history: Record<string, unknown>[] }>,
      ])
      setHealth(h); setSummary(s); setHistory(hist.history ?? [])
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e))
    } finally { setLoading(false) }
  }, [])

  useEffect(() => { load() }, [load])

  const statusColor = (s: unknown) =>
    s === 'critical' ? C.red : s === 'warning' ? C.yellow : C.green

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <h2 style={{ margin: 0, fontSize: 18 }}>Health Monitor</h2>
        <button onClick={load} style={{ ...btnStyle, opacity: loading ? 0.5 : 1 }} disabled={loading}>
          {loading ? 'Refreshing…' : 'Refresh'}
        </button>
      </div>
      {error && <div style={{ color: C.red, marginBottom: 12 }}>{error}</div>}

      {health && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: 12, marginBottom: 20 }}>
          {[
            { label: 'Status', value: String(health.status ?? '—').toUpperCase(), color: statusColor(health.status) },
            { label: 'Account Value', value: `$${Number(health.account_value ?? 0).toLocaleString()}` },
            { label: 'P&L Today', value: `$${Number(health.pnl_today ?? 0).toFixed(2)}` },
            { label: 'P&L Today %', value: `${Number(health.pnl_today_pct ?? 0).toFixed(2)}%` },
            { label: 'Max Drawdown', value: `${Number(health.max_drawdown_pct ?? 0).toFixed(2)}%` },
            { label: 'Open Positions', value: String(health.open_positions ?? 0) },
            { label: 'Trades Today', value: String(health.trades_today ?? 0) },
            { label: 'Losing Streak', value: `${health.consecutive_losing_days ?? 0} days`, color: Number(health.consecutive_losing_days ?? 0) >= 3 ? C.red : Number(health.consecutive_losing_days ?? 0) >= 2 ? C.yellow : undefined },
          ].map(({ label, value, color }) => (
            <div key={label} style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8, padding: '12px 16px' }}>
              <div style={{ fontSize: 11, color: C.muted, marginBottom: 4 }}>{label}</div>
              <div style={{ fontSize: 18, fontWeight: 600, color: color ?? C.text }}>{value}</div>
            </div>
          ))}
        </div>
      )}

      {summary && (summary as { summary?: unknown }).summary !== undefined && (
        <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8, padding: 16, marginBottom: 20 }}>
          <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 8 }}>Last Session Summary</div>
          <pre style={{ margin: 0, fontSize: 12, color: C.muted, whiteSpace: 'pre-wrap' }}>
            {JSON.stringify((summary as { summary?: unknown }).summary, null, 2)}
          </pre>
        </div>
      )}

      {history.length > 0 && (
        <div>
          <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 8 }}>Session History (7 days)</div>
          <div style={{ overflowX: 'auto', overflowY: 'auto', maxHeight: 300 }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ color: C.muted, borderBottom: `1px solid ${C.border}` }}>
                  {['Date', 'P&L', 'P&L %', 'Trades', 'Drawdown %', 'Status', 'Losing Streak'].map(h => (
                    <th key={h} style={{ padding: '6px 12px', textAlign: 'left', fontWeight: 500 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {history.map((row, i) => (
                  <tr key={i} style={{ borderBottom: `1px solid ${C.border}` }}>
                    <td style={{ padding: '6px 12px' }}>{String(row.date ?? '—')}</td>
                    <td style={{ padding: '6px 12px', color: Number(row.pnl_today ?? 0) >= 0 ? C.green : C.red }}>
                      ${Number(row.pnl_today ?? 0).toFixed(2)}
                    </td>
                    <td style={{ padding: '6px 12px' }}>{Number(row.pnl_today_pct ?? 0).toFixed(2)}%</td>
                    <td style={{ padding: '6px 12px' }}>{String(row.trades_today ?? 0)}</td>
                    <td style={{ padding: '6px 12px' }}>{Number(row.max_drawdown_pct ?? 0).toFixed(2)}%</td>
                    <td style={{ padding: '6px 12px', color: statusColor(row.status) }}>
                      {String(row.status ?? '—').toUpperCase()}
                    </td>
                    <td style={{ padding: '6px 12px' }}>{String(row.consecutive_losing_days ?? 0)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}

// ── Agents Panel ──────────────────────────────────────────────────────────────

interface AgentEvent {
  id: number
  agent_name: AgentName | string
  time: string
  decision_type: string
  symbol?: string
  reasoning?: Record<string, unknown>
  expanded: boolean
}

const AGENT_NAMES = ['portfolio_manager', 'risk_manager', 'trader'] as const
type AgentName = typeof AGENT_NAMES[number]

const AGENT_LABELS: Record<AgentName, string> = {
  portfolio_manager: 'Portfolio Manager',
  risk_manager: 'Risk Manager',
  trader: 'Trader',
}

const DECISION_COLOR = (dt: string): string => {
  if (/APPROVED|ENTERED|SELECTED|ANALYSIS|WARMED/.test(dt)) return C.green
  if (/REJECTED|ERROR|FAILED/.test(dt)) return C.red
  if (/EXITED|CLOSED|SKIPPED|PAUSED/.test(dt)) return C.yellow
  return C.accent
}

function ReasoningCard({ type, r }: { type: string; r: Record<string, unknown> }) {
  const row = (label: string, val: unknown) => val == null ? null : (
    <div key={label} style={{ display: 'flex', gap: 8, padding: '2px 0', borderBottom: '1px solid rgba(255,255,255,.04)' }}>
      <span style={{ fontSize: 11, color: C.muted, minWidth: 130, flexShrink: 0 }}>{label}</span>
      <span style={{ fontSize: 11, color: C.text, wordBreak: 'break-word' }}>{String(val)}</span>
    </div>
  )

  // INSTRUMENTS_SELECTED — show ranked stock list
  if (type === 'INSTRUMENTS_SELECTED') {
    const selected = (r.selected as Array<{ symbol: string; confidence: number }>) || []
    return (
      <div style={{ paddingTop: 6 }}>
        {row('Sent at', r.sent_at)}
        <div style={{ fontSize: 11, color: C.muted, margin: '6px 0 4px' }}>Selected stocks (ranked by confidence):</div>
        {selected.map((s, i) => (
          <div key={s.symbol} style={{ display: 'flex', gap: 8, padding: '2px 0' }}>
            <span style={{ fontSize: 11, color: C.muted, minWidth: 20 }}>#{i + 1}</span>
            <span style={{ fontSize: 11, fontWeight: 600, color: C.accent, minWidth: 60 }}>{s.symbol}</span>
            <span style={{ fontSize: 11, color: C.green }}>{(s.confidence * 100).toFixed(1)}% confidence</span>
          </div>
        ))}
        {selected.length === 0 && <div style={{ fontSize: 11, color: C.muted }}>No stocks met confidence threshold</div>}
      </div>
    )
  }

  // SWING_PREMARKET_ANALYSIS — candidates list
  if (type === 'SWING_PREMARKET_ANALYSIS') {
    const candidates = (r.candidates as Array<{ symbol: string; confidence: number }>) || []
    return (
      <div style={{ paddingTop: 6 }}>
        {row('Analyzed', r.universe_size ? `${r.universe_size} stocks` : null)}
        <div style={{ fontSize: 11, color: C.muted, margin: '6px 0 4px' }}>Top candidates:</div>
        {candidates.slice(0, 10).map((s, i) => (
          <div key={s.symbol} style={{ display: 'flex', gap: 8, padding: '2px 0' }}>
            <span style={{ fontSize: 11, color: C.muted, minWidth: 20 }}>#{i + 1}</span>
            <span style={{ fontSize: 11, fontWeight: 600, color: C.accent, minWidth: 60 }}>{s.symbol}</span>
            <span style={{ fontSize: 11, color: C.green }}>{(s.confidence * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>
    )
  }

  // TRADE_APPROVED / TRADE_REJECTED
  if (type === 'TRADE_APPROVED' || type === 'TRADE_REJECTED') {
    const p = (r.proposal as Record<string, unknown>) || r
    return (
      <div style={{ paddingTop: 6 }}>
        {row('Symbol', p.symbol)}
        {row('Direction', p.direction)}
        {row('Entry price', p.entry_price != null ? `$${Number(p.entry_price).toFixed(2)}` : null)}
        {row('Stop loss', p.stop_loss != null ? `$${Number(p.stop_loss).toFixed(2)}` : null)}
        {row('Position size', p.position_size != null ? `$${Number(p.position_size).toFixed(0)}` : null)}
        {row('Confidence', p.confidence != null ? `${(Number(p.confidence) * 100).toFixed(1)}%` : null)}
        {row('Signal type', p.signal_type)}
        {type === 'TRADE_REJECTED' && row('Reason', r.failed_rule ?? r.reason ?? r.message)}
      </div>
    )
  }

  // TRADE_ENTERED
  if (type === 'TRADE_ENTERED') {
    return (
      <div style={{ paddingTop: 6 }}>
        {row('Symbol', r.symbol)}
        {row('Direction', r.direction)}
        {row('Entry price', r.entry_price != null ? `$${Number(r.entry_price).toFixed(2)}` : null)}
        {row('Quantity', r.quantity)}
        {row('Stop loss', r.stop_loss != null ? `$${Number(r.stop_loss).toFixed(2)}` : null)}
        {row('Signal type', r.signal_type)}
        {row('Strategy', r.strategy)}
      </div>
    )
  }

  // TRADE_EXITED / INTRADAY_FORCE_CLOSED
  if (/EXITED|CLOSED/.test(type)) {
    return (
      <div style={{ paddingTop: 6 }}>
        {row('Symbol', r.symbol)}
        {row('Exit price', r.exit_price != null ? `$${Number(r.exit_price).toFixed(2)}` : null)}
        {row('P&L', r.pnl != null ? `$${Number(r.pnl).toFixed(2)}` : null)}
        {row('Reason', r.reason ?? r.exit_reason)}
        {row('Hold time', r.hold_time)}
      </div>
    )
  }

  // MODEL_RETRAINED
  if (type === 'MODEL_RETRAINED') {
    return (
      <div style={{ paddingTop: 6 }}>
        {row('Version', r.version)}
        {row('Sharpe', r.sharpe != null ? Number(r.sharpe).toFixed(2) : null)}
        {row('Accuracy', r.accuracy != null ? `${(Number(r.accuracy) * 100).toFixed(1)}%` : null)}
      </div>
    )
  }

  // Fallback — clean key/value table
  return (
    <div style={{ paddingTop: 6 }}>
      {Object.entries(r).map(([k, v]) =>
        typeof v === 'object' ? null : row(k.replace(/_/g, ' '), v)
      )}
    </div>
  )
}

function AgentFeed({ name, events, onToggle }: {
  name: AgentName
  events: AgentEvent[]
  onToggle: (id: number) => void
}) {
  return (
    <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
      {/* Header */}
      <div style={{ padding: '10px 14px', borderBottom: `1px solid ${C.border}`, display: 'flex', alignItems: 'center', gap: 8 }}>
        <div style={{ width: 8, height: 8, borderRadius: '50%', background: C.green, flexShrink: 0 }} />
        <span style={{ fontWeight: 600, fontSize: 13 }}>{AGENT_LABELS[name]}</span>
        <span style={{ marginLeft: 'auto', fontSize: 11, color: C.muted }}>{events.length} events</span>
      </div>

      {/* Event list */}
      <div style={{ overflowY: 'auto', maxHeight: 'calc(100vh - 220px)', flex: 1 }}>
        {events.length === 0 && (
          <div style={{ padding: 16, color: C.muted, fontSize: 12, textAlign: 'center' }}>
            No decisions recorded yet.<br />
            <span style={{ fontSize: 11 }}>Events appear here once the agent pipeline runs.</span>
          </div>
        )}
        {events.map(ev => (
          <div key={ev.id} style={{ borderBottom: `1px solid rgba(255,255,255,.04)` }}>
            <div
              onClick={() => onToggle(ev.id)}
              style={{ padding: '8px 14px', cursor: 'pointer', display: 'flex', alignItems: 'flex-start', gap: 8 }}
            >
              <span style={{ fontSize: 10, color: C.muted, flexShrink: 0, paddingTop: 2 }}>{ev.time}</span>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
                  <span style={{ fontSize: 11, fontWeight: 600, color: DECISION_COLOR(ev.decision_type) }}>
                    {ev.decision_type}
                  </span>
                  {ev.symbol && (
                    <span style={{ fontSize: 11, background: C.surface2, padding: '1px 6px', borderRadius: 4, color: C.accent }}>
                      {ev.symbol}
                    </span>
                  )}
                </div>
                {/* One-line summary from reasoning */}
                {!ev.expanded && ev.reasoning && (
                  <div style={{ fontSize: 11, color: C.muted, marginTop: 2, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {ev.reasoning.failed_rule
                      ? `✗ ${ev.reasoning.failed_rule}: ${ev.reasoning.message ?? ''}`
                      : ev.reasoning.stop_loss
                      ? `stop=$${Number(ev.reasoning.stop_loss).toFixed(2)}`
                      : ev.reasoning.candidates
                      ? `${(ev.reasoning.candidates as unknown[]).length} candidates`
                      : ev.reasoning.reason
                      ? String(ev.reasoning.reason)
                      : ev.reasoning.exit_price
                      ? `exit=$${Number(ev.reasoning.exit_price).toFixed(2)}  PnL=$${Number(ev.reasoning.pnl ?? 0).toFixed(2)}`
                      : ''}
                  </div>
                )}
              </div>
              <span style={{ fontSize: 10, color: C.muted, flexShrink: 0 }}>{ev.expanded ? '▲' : '▼'}</span>
            </div>
            {ev.expanded && ev.reasoning && (
              <div style={{ padding: '0 14px 10px 36px', background: C.surface2 }}>
                <ReasoningCard type={ev.decision_type} r={ev.reasoning} />
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

function AgentsPanel({ liveEvents }: { liveEvents: AgentEvent[] }) {
  const [historical, setHistorical] = useState<AgentEvent[]>([])
  const [events, setEvents] = useState<AgentEvent[]>([])
  const [loading, setLoading] = useState(true)

  // Load historical decisions on mount — set loading false immediately so panel renders
  useEffect(() => {
    setLoading(false)
    api.decisions(100).then((data: unknown) => {
      const rows = data as Array<{ id: number; agent_name: string; decision_type: string; symbol?: string; reasoning?: Record<string, unknown>; timestamp?: string }>
      const mapped: AgentEvent[] = rows.map(r => ({
        id: r.id,
        agent_name: r.agent_name,
        time: r.timestamp ? fmtTs(r.timestamp) : '—',
        decision_type: r.decision_type,
        symbol: r.symbol,
        reasoning: r.reasoning,
        expanded: false,
      }))
      setHistorical(mapped)
    }).catch((e) => { console.error('Decisions fetch failed:', e) })
  }, [])

  // Merge live + historical, deduplicate by id, newest first
  useEffect(() => {
    const all = [...liveEvents, ...historical]
    const seen = new Set<number>()
    const deduped = all.filter(e => { if (seen.has(e.id)) return false; seen.add(e.id); return true })
    setEvents(deduped)
  }, [liveEvents, historical])

  const toggle = useCallback((id: number) => {
    setEvents(prev => prev.map(e => e.id === id ? { ...e, expanded: !e.expanded } : e))
  }, [])

  const forAgent = (name: AgentName) => events.filter(e => e.agent_name === name)

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <h2 style={{ margin: 0, fontSize: 18 }}>Agent Decisions</h2>
        <span style={{ fontSize: 12, color: C.muted }}>
          {loading ? 'Loading…' : `${events.length} total events · live via WebSocket`}
        </span>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
        {AGENT_NAMES.map(name => (
          <AgentFeed key={name} name={name} events={forAgent(name)} onToggle={toggle} />
        ))}
      </div>
    </div>
  )
}

// ── Main App ──────────────────────────────────────────────────────────────────
const TABS = ['Overview', 'Positions', 'Trades', 'Signal Monitor', 'Capital Ramp', 'Kill Switch', 'Orchestrator', 'Readiness', 'Analytics', 'Watchlist', 'Performance', 'Config', 'Monitor', 'Agents'] as const
type Tab = typeof TABS[number]

export default function App() {
  const [tab, setTab] = useState<Tab>('Overview')
  const [summary, setSummary] = useState<Summary>({})
  const [health, setHealth] = useState<Health | null>(null)
  const [pnlHistory, setPnlHistory] = useState<{ time: string; pnl: number }[]>([])
  const [decisions, setDecisions] = useState<Decision[]>([])
  const [signalFeed, setSignalFeed] = useState<SignalRow[]>([])
  const [agentEvents, setAgentEvents] = useState<AgentEvent[]>([])
  const [wsConnected, setWsConnected] = useState(false)
  const [macroCtx, setMacroCtx] = useState<NisMacroContext | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const { toasts, add: toast } = useToasts()

  // Load summary and health independently — summary renders immediately, health updates the status badge
  const loadSummary = useCallback(async () => {
    // Fire both but don't wait — each updates state as soon as it resolves
    api.summary().then((j: unknown) => {
      const s = j as Summary
      setSummary(s)
      const pnlVal = s.daily_pnl ?? s.pnl_today
      if (pnlVal != null) {
        const now = new Date().toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true })
        setPnlHistory(h => {
          const next = [...h, { time: now, pnl: pnlVal }]
          return next.length > 60 ? next.slice(-60) : next
        })
      }
    }).catch(() => {})
    api.health().then((j: unknown) => setHealth(j as Health)).catch(() => {})
  }, [])

  const loadDecisions = useCallback(async () => {
    try {
      const j = await api.decisions(100) as { data?: Decision[] } | Decision[]
      setDecisions((j as { data?: Decision[] }).data ?? (j as Decision[]) ?? [])
    } catch { /* ignore */ }
  }, [])

  // WebSocket
  useEffect(() => {
    let retryDelay = 3000
    function connect() {
      const proto = location.protocol === 'https:' ? 'wss' : 'ws'
      const ws = new WebSocket(`${proto}://${location.host}/ws`)
      wsRef.current = ws

      ws.onopen = () => {
        setWsConnected(true)
        retryDelay = 3000
        toast('WebSocket connected', 'success')
      }
      ws.onclose = () => {
        setWsConnected(false)
        retryDelay = Math.min(retryDelay * 2, 30000)  // exponential backoff, max 30s
        setTimeout(connect, retryDelay)
      }
      ws.onerror = () => setWsConnected(false)
      ws.onmessage = e => {
        try {
          const msg = JSON.parse(e.data) as WsMessage
          handleWs(msg)
        } catch { /* ignore */ }
      }
    }

    function handleWs(msg: WsMessage) {
      const { type, data } = msg
      if (type === 'portfolio_update') {
        setSummary(s => ({ ...s, ...data }))
      }
      if (type === 'trade_executed' || type === 'trade_closed' || type === 'agent_decision') {
        const kind: SignalRow['kind'] = type === 'trade_executed' ? 'buy' : type === 'trade_closed' ? 'sell' : 'signal'
        const sym = (data.symbol as string) ?? (data.reasoning as { symbol?: string } | undefined)?.symbol ?? '?'
        const msgText = (data.action as string) ?? (data.decision_type as string) ?? (data.message as string) ?? ''
        const now = new Date().toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', second: '2-digit', hour12: true })
        setSignalFeed(f => [{ time: now, symbol: sym, kind, msg: msgText }, ...f].slice(0, 100))

        const nowET = new Date().toLocaleString('en-US', { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit', second: '2-digit', hour12: true, timeZone: 'America/New_York' })
        if (type === 'agent_decision') {
          const ev: AgentEvent = {
            id: Date.now() + Math.random(), // ephemeral id for live events
            agent_name: (data.agent_name as string) ?? 'unknown',
            time: nowET,
            decision_type: (data.decision_type as string) ?? '',
            symbol: data.symbol as string | undefined,
            reasoning: data.reasoning as Record<string, unknown> | undefined,
            expanded: false,
          }
          setAgentEvents(prev => [ev, ...prev].slice(0, 200))
        }
      }
      if (type === 'alert') toast((data.message as string) ?? JSON.stringify(data), 'warning')
    }

    connect()
    return () => { wsRef.current?.close() }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // Initial load + polling
  useEffect(() => {
    loadSummary()
    loadDecisions()
    // Fetch NIS macro context once on load (updated daily by premarket routine)
    api.nisMacro().then((j: unknown) => {
      const ctx = j as NisMacroContext & { status?: string }
      if (!ctx.status) setMacroCtx(ctx)
    }).catch(() => {})
    const id = setInterval(() => { loadSummary(); loadDecisions() }, 10000)
    return () => clearInterval(id)
  }, [loadSummary, loadDecisions])

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', background: C.bg, color: C.text }}>
      <TopBar health={health} wsConnected={wsConnected} />

      {/* Tab bar */}
      <div style={{
        display: 'flex', background: C.surface, borderBottom: `1px solid ${C.border}`,
        padding: '0 20px', overflowX: 'auto',
      }}>
        {TABS.map(t => (
          <div key={t} onClick={() => setTab(t)} style={{
            padding: '10px 18px', cursor: 'pointer', fontSize: 12, fontWeight: 500,
            color: tab === t ? C.accent : C.muted,
            borderBottom: tab === t ? `2px solid ${C.accent}` : '2px solid transparent',
            whiteSpace: 'nowrap', transition: 'color .2s, border-color .2s',
          }}>{t}</div>
        ))}
      </div>

      {/* Panels */}
      <div style={{ flex: 1, padding: '16px 20px', overflowY: 'auto' }}>
        {tab === 'Overview' && (
          <OverviewPanel summary={summary} health={health} decisions={decisions} macroCtx={macroCtx} />
        )}
        {tab === 'Positions' && <PositionsPanel onRefresh={loadSummary} />}
        {tab === 'Trades' && <TradesPanel />}
        {tab === 'Signal Monitor' && <SignalsPanel feed={signalFeed} decisions={decisions} />}
        {tab === 'Capital Ramp' && <RampPanel toast={toast} />}
        {tab === 'Kill Switch' && <KillPanel toast={toast} />}
        {tab === 'Orchestrator' && <SessionPanel toast={toast} />}
        {tab === 'Readiness' && <ReadinessPanel />}
        {tab === 'Analytics' && <ErrorBoundary><AnalyticsPanel /></ErrorBoundary>}
        {tab === 'Watchlist' && <WatchlistPanel toast={toast} />}
        {tab === 'Performance' && <PerformanceReviewPanel />}
        {tab === 'Config' && <ConfigPanel toast={toast} />}
        {tab === 'Monitor' && <MonitorPanel />}
        {tab === 'Agents' && <AgentsPanel liveEvents={agentEvents} />}
      </div>

      <ToastContainer toasts={toasts} />
    </div>
  )
}
