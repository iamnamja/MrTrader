# MrTrader startup script — runs infrastructure in Docker, app locally
# Usage: .\start.ps1

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
. (Join-Path $root "_lib.ps1")

Write-Host ""
Write-Host "=== MrTrader Startup ===" -ForegroundColor Cyan

# ── 1. Check Docker is running ─────────────────────────────────────────────────
Write-Host "[1/4] Checking Docker..." -ForegroundColor Yellow
try {
    docker info > $null 2>&1
    Write-Host "      Docker is running." -ForegroundColor Green
} catch {
    Write-Host "      Docker is not running. Please open Docker Desktop and wait for it to start, then re-run this script." -ForegroundColor Red
    exit 1
}

# ── 2. Stop the app container (runs old code) — keep DB + Redis ────────────────
Write-Host "[2/4] Starting infrastructure containers..." -ForegroundColor Yellow
docker stop mrtrader_app 2>$null
docker start mrtrader_postgres mrtrader_redis 2>&1 | Out-Null

# Wait for postgres to be healthy
$attempts = 0
while ($attempts -lt 15) {
    $health = docker inspect --format='{{.State.Health.Status}}' mrtrader_postgres 2>$null
    if ($health -eq "healthy") { break }
    Write-Host "      Waiting for PostgreSQL..." -ForegroundColor Gray
    Start-Sleep 2
    $attempts++
}
Write-Host "      PostgreSQL ready." -ForegroundColor Green
Write-Host "      Redis ready." -ForegroundColor Green

# ── 3. Kill anything holding ports 8000 / 3000 ────────────────────────────────
Write-Host "[3/5] Clearing ports 8000 and 3000..." -ForegroundColor Yellow
Stop-MrtPort -Port 8000
Stop-MrtPort -Port 3000
Write-Host "      Ports 8000 and 3000 are free." -ForegroundColor Green

# ── 4. Notification watcher ───────────────────────────────────────────────────
# Same drainer serve.ps1 starts; stop.ps1 stops it. Idempotent, so re-running is safe.
# Without it, queued mail (including the daily liveness beacon) is never sent.
Write-Host "[4/5] Starting notification watcher..." -ForegroundColor Yellow
Start-MrtNotifyWatcher -Root $root

# ── 5. Start backend + frontend in separate windows ───────────────────────────
Write-Host "[5/5] Starting backend and frontend..." -ForegroundColor Yellow

# Backend — new PowerShell window.
# Invoke the venv interpreter EXPLICITLY instead of relying on Activate.ps1: the previous
# form swallowed activation errors with `2>$null` and then called a bare `uvicorn`, so a
# broken/renamed venv surfaced as "uvicorn is not recognized" in a window that had already
# scrolled away — or, worse, silently ran on whatever interpreter PATH happened to offer.
$pyExe = Get-MrtPython -Root $root
if ($pyExe) {
    $backendCmd = "cd '$root'; Write-Host 'Backend starting...' -ForegroundColor Cyan; " +
                  "& '$pyExe' -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --timeout-graceful-shutdown 30"
} else {
    Write-Host "      WARNING: venv not found - backend will use whatever is on PATH." -ForegroundColor Yellow
    $backendCmd = "cd '$root'; Write-Host 'Backend starting (no venv!)...' -ForegroundColor Yellow; " +
                  "uvicorn app.main:app --host 0.0.0.0 --port 8000 --timeout-graceful-shutdown 30"
}
Start-Process powershell -ArgumentList @("-NoExit", "-Command", $backendCmd) -WindowStyle Normal

Start-Sleep 3

# Frontend — new PowerShell window ($root, not a hardcoded path, so a clone or a renamed
# checkout still works)
$frontendCmd = "cd '$(Join-Path $root 'frontend')'; Write-Host 'Frontend starting...' -ForegroundColor Cyan; npm run dev"
Start-Process powershell -ArgumentList @("-NoExit", "-Command", $frontendCmd) -WindowStyle Normal

Write-Host ""
Write-Host "=== All systems starting ===" -ForegroundColor Cyan
Write-Host "  Backend:   http://localhost:8000" -ForegroundColor White
Write-Host "  Frontend:  http://localhost:3000" -ForegroundColor White
Write-Host "  Dashboard: http://localhost:3000/dashboard" -ForegroundColor White
Write-Host ""
Write-Host "Two new terminal windows have opened for backend and frontend." -ForegroundColor Gray
Write-Host "Wait ~10 seconds then open http://localhost:3000/dashboard" -ForegroundColor Gray
