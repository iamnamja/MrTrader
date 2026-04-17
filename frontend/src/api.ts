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
  positions: () => get('/api/positions'),
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
  pauseJob: (id: string) => post(`/api/orchestrator/jobs/${id}/pause`),
  resumeJob: (id: string) => post(`/api/orchestrator/jobs/${id}/resume`),
  // Macro
  macroIndicators: () => get('/api/dashboard/analytics/macro'),
  regimeDetail: () => get('/api/dashboard/analytics/regime'),
}
