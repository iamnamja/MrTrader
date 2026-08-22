# stop.ps1 - stop everything start.ps1 / serve.ps1 start.
# Usage:  .\stop.ps1
#
# Counterpart to serve.ps1/start.ps1: whatever they launch, this stops. Safe to run when
# parts are already down — every step is a no-op if its target is absent.

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
. (Join-Path $root "_lib.ps1")

Write-Host "=== Stopping MrTrader ===" -ForegroundColor Cyan

# --- API server (uvicorn) and the vite dev server ---
Write-Host "==> Stopping API server (port 8000)..." -ForegroundColor Cyan
Stop-MrtPort -Port 8000
Write-Host "==> Stopping frontend dev server (port 3000)..." -ForegroundColor Cyan
Stop-MrtPort -Port 3000

# --- Dead-man watchdog ---
# serve.ps1 stops this in its `finally`, but only on a CLEAN exit. If that window is closed
# or killed the watchdog is orphaned and keeps emailing [CRITICAL] dead_man_alert about a
# brain that is intentionally down.
Write-Host "==> Stopping dead-man watchdog..." -ForegroundColor Cyan
Stop-MrtDeadManWatchdog

# --- Notification watcher ---
# Stopped LAST so any alert enqueued by the shutdown above still has a live drainer for a
# moment. Note this means a stopped stack sends no email at all — including the daily
# beacon, whose ABSENCE is the alert. That is correct for a deliberate stop; just remember
# that silence while stopped is expected, not a signal.
Write-Host "==> Stopping notification watcher..." -ForegroundColor Cyan
Stop-MrtNotifyWatcher -Root $root

# --- Infrastructure (keep data volumes) ---
Write-Host "==> Stopping Postgres + Redis containers..." -ForegroundColor Cyan
docker stop mrtrader_postgres mrtrader_redis 2>$null | Out-Null

Write-Host ""
Write-Host "All services stopped." -ForegroundColor Green
Write-Host "Restart with .\serve.ps1 (production) or .\start.ps1 (dev)." -ForegroundColor Gray
