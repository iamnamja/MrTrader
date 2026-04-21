# MrTrader startup script — runs infrastructure in Docker, app locally
# Usage: .\start.ps1

$ErrorActionPreference = "Stop"

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

# ── 3. Kill anything holding port 8000 ────────────────────────────────────────
Write-Host "[3/4] Clearing port 8000..." -ForegroundColor Yellow
$pids = (netstat -ano | Select-String ":8000.*LISTENING") -replace '.*\s+(\d+)$','$1'
foreach ($p in $pids) {
    if ($p -match '^\d+$') {
        taskkill /PID $p /F 2>$null | Out-Null
    }
}
Write-Host "      Port 8000 is free." -ForegroundColor Green

# ── 3b. Kill anything holding port 3000 ───────────────────────────────────────
Write-Host "      Clearing port 3000..." -ForegroundColor Yellow
$pids3 = (netstat -ano | Select-String ":3000.*LISTENING") -replace '.*\s+(\d+)$','$1'
foreach ($p in $pids3) {
    if ($p -match '^\d+$') {
        taskkill /PID $p /F 2>$null | Out-Null
    }
}
Write-Host "      Port 3000 is free." -ForegroundColor Green

# ── 4. Start backend + frontend in separate windows ───────────────────────────
Write-Host "[4/4] Starting backend and frontend..." -ForegroundColor Yellow

# Backend — new PowerShell window
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "cd 'C:\Projects\MrTrader'; .\.venv\Scripts\Activate.ps1 2>`$null; .\venv\Scripts\Activate.ps1 2>`$null; Write-Host 'Backend starting...' -ForegroundColor Cyan; uvicorn app.main:app --host 0.0.0.0 --port 8000"
) -WindowStyle Normal

Start-Sleep 3

# Frontend — new PowerShell window
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "cd 'C:\Projects\MrTrader\frontend'; Write-Host 'Frontend starting...' -ForegroundColor Cyan; npm run dev"
) -WindowStyle Normal

Write-Host ""
Write-Host "=== All systems starting ===" -ForegroundColor Cyan
Write-Host "  Backend:   http://localhost:8000" -ForegroundColor White
Write-Host "  Frontend:  http://localhost:3000" -ForegroundColor White
Write-Host "  Dashboard: http://localhost:3000/dashboard" -ForegroundColor White
Write-Host ""
Write-Host "Two new terminal windows have opened for backend and frontend." -ForegroundColor Gray
Write-Host "Wait ~10 seconds then open http://localhost:3000/dashboard" -ForegroundColor Gray
