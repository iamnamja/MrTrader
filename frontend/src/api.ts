const BASE = ''

async function get<T>(path: string): Promise<T> {
  const r = await fetch(BASE + path)
  if (!r.ok) throw new Error(`HTTP ${r.status}`)
  return r.json()
}

async function post<T>(path: string, body?: unknown): Promise<T> {
  const r = await fetch(BASE + path, {
    method: 'POST',
    headers: body ? { 'Content-Type': 'application/json' } : {},
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!r.ok) throw new Error(`HTTP ${r.status}`)
  return r.json()
}

export const api = {
  summary: () => get('/api/dashboard/summary'),
  health: () => get('/api/dashboard/health'),
  decisions: (limit = 50) => get(`/api/dashboard/decisions?limit=${limit}`),
  positions: () => get('/api/dashboard/positions'),
  trades: (status?: string) => get(`/api/dashboard/trades${status ? `?status=${status}` : ''}`),
  liveStatus: () => get('/api/dashboard/live/status'),
  auditLog: (limit = 20) => get(`/api/dashboard/live/audit-log?limit=${limit}`),
  killSwitch: (reason: string) => post('/api/dashboard/live/kill-switch', { reason }),
  resetKillSwitch: () => post('/api/dashboard/live/kill-switch/reset'),
  increaseCapital: () => post('/api/dashboard/live/increase-capital'),
  readiness: () => get('/api/dashboard/live/readiness'),
  signalAttribution: (days = 90) => get(`/api/dashboard/analytics/signal-attribution?days=${days}`),
  // Orchestrator
  orchStatus: () => get('/api/orchestrator/status'),
  marketStatus: () => get('/api/orchestrator/market-status'),
  sessionLog: (limit = 50) => get(`/api/orchestrator/session-log?limit=${limit}`),
  orchJobs: () => get('/api/orchestrator/jobs'),
  pauseTrading: () => post('/api/orchestrator/pause-trading'),
  resumeTrading: () => post('/api/orchestrator/resume-trading'),
  triggerCycle: () => post('/api/orchestrator/trigger-cycle'),
  triggerRetraining: () => post('/api/orchestrator/trigger-retraining'),
  triggerIntradayScan: () => post('/api/orchestrator/trigger-intraday-scan'),
  pauseJob: (id: string) => post(`/api/orchestrator/jobs/${id}/pause`),
  resumeJob: (id: string) => post(`/api/orchestrator/jobs/${id}/resume`),
  // Agent config
  configSchema: () => get('/api/config/schema'),
  configValues: () => get('/api/config'),
  configUpdate: (key: string, value: number) => fetch(`/api/config/${key}`, {
    method: 'PUT', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ value }),
  }).then(r => { if (!r.ok) return r.json().then(j => Promise.reject(new Error(j.detail))); return r.json() }),
  configReset: () => post('/api/config/reset'),
  // Performance review
  performanceReview: (days = 30) => get(`/api/dashboard/analytics/performance-review?days=${days}`),
  // Macro
  macroIndicators: () => get('/api/dashboard/analytics/macro'),
  regimeDetail: () => get('/api/dashboard/analytics/regime'),
  // Watchlist
  watchlist: (activeOnly?: boolean) => get(`/api/watchlist${activeOnly ? '?active_only=true' : ''}`),
  watchlistAdd: (symbol: string, sector?: string, notes?: string) =>
    post('/api/watchlist', { symbol, sector, notes }),
  watchlistDelete: (symbol: string) =>
    fetch(`/api/watchlist/${symbol}`, { method: 'DELETE' }).then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json() }),
  watchlistPatch: (symbol: string, patch: { active?: boolean; notes?: string; sector?: string }) =>
    fetch(`/api/watchlist/${symbol}`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(patch) }).then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json() }),
  watchlistBulk: () => post('/api/watchlist/bulk'),
  // Monitor
  monitorHealth: () => get('/api/dashboard/monitor/health'),
  monitorSummary: () => get('/api/dashboard/monitor/summary'),
  monitorRunSummary: () => post('/api/dashboard/monitor/run-summary'),
  monitorHistory: (days = 7) => get(`/api/dashboard/monitor/history?days=${days}`),
  // AI briefing
  aiBriefing: () => get('/api/orchestrator/ai-briefing'),
  // NIS & Decision Audit (Phase 74 + audit hardening)
  nisMacro: () => get('/api/nis/macro'),
  nisSignals: () => get('/api/nis/signals'),
  nisCost: (days = 7) => get(`/api/nis/cost?days=${days}`),
  decisionAuditRecent: (limit = 100, strategy?: string) =>
    get(`/api/decision-audit/recent?limit=${limit}${strategy ? `&strategy=${strategy}` : ''}`),
  decisionAuditSummary: () => get('/api/decision-audit/summary'),
  gateCalibration: () => get('/api/decision-audit/gate-calibration'),
  proposalLog: (days = 3, strategy = '') => get(`/api/dashboard/proposal-log?days=${days}&strategy=${strategy}`),
  swingProposals: (days = 3) => get(`/api/dashboard/proposal-log?days=${days}&strategy=swing`),
  intraProposals: (days = 3) => get(`/api/dashboard/proposal-log?days=${days}&strategy=intraday`),
}
