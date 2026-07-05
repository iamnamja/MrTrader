# serve.ps1 - production-style launch: build the frontend, then serve via uvicorn.
# PowerShell 5.1-safe (replaces the `&&` chaining that 5.1 does not support).
# The server only starts if the frontend build succeeds (exit-code check mirrors `&&`).
#
# Usage:  .\serve.ps1
#
# (For dev mode with frontend hot-reload on :3000 + Docker infra, use .\start.ps1 instead.)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

Write-Host "==> Building frontend (production)..." -ForegroundColor Cyan
Set-Location (Join-Path $root "frontend")
npm run build
if ($LASTEXITCODE -ne 0) {
    Write-Host "Frontend build FAILED (exit $LASTEXITCODE) - not starting server." -ForegroundColor Red
    Set-Location $root
    exit $LASTEXITCODE
}

Set-Location $root

# --- Ensure the database is up (Postgres + Redis) ---
# stop.ps1 stops these Docker containers; serve.ps1 only starts uvicorn - so a
# stop.ps1 -> serve.ps1 cycle would otherwise leave Postgres down and the API would
# fail on "connection refused :5432". Starting them here is idempotent (no-op if
# already running) and makes serve.ps1 self-sufficient.
Write-Host "==> Ensuring database is up (Postgres + Redis)..." -ForegroundColor Cyan
docker start mrtrader_postgres mrtrader_redis 2>$null | Out-Null
$pgReady = $false
for ($i = 0; $i -lt 20; $i++) {
    docker exec mrtrader_postgres pg_isready -U mrtrader 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) { $pgReady = $true; break }
    Start-Sleep -Seconds 1
}
if ($pgReady) {
    Write-Host "    Postgres accepting connections." -ForegroundColor Green
} else {
    Write-Host "    WARNING: Postgres not confirmed ready (is Docker running, and do the" -ForegroundColor Yellow
    Write-Host "    mrtrader_postgres/mrtrader_redis containers exist? use .\start.ps1 for" -ForegroundColor Yellow
    Write-Host "    fresh infra). Starting the API anyway - it may fail on DB connect." -ForegroundColor Yellow
}

# --- Launch the EXTERNAL dead-man watchdog (Alpha-v10 H5) alongside the server ---
# It must be a SEPARATE process: an in-process monitor can't report the brain (uvicorn) dying/hanging.
# It reads the brain's 1-min heartbeat file and emails [CRITICAL] dead_man_alert if it goes stale.
#   --start-grace-sec 120  : don't check until the brain has had time to boot + write a fresh
#                            heartbeat, else it would false-alert on the stale file from the last run.
#   alert-only             : no --auto-flatten (no trading authority) — matches the runbook default.
# We stop it in `finally` when uvicorn exits so a clean Ctrl+C shutdown doesn't leave it firing a
# false stale-heartbeat alert ~10 min later. (NOTE: it catches a brain crash/hang, NOT total-machine
# death — it runs on the same box; for power-loss detection use an off-box dead-man's-snitch.)
$watchdog = $null
$pyExe = Join-Path $root "venv\Scripts\python.exe"
if (Test-Path $pyExe) {
    Write-Host "==> Starting dead-man watchdog (alert-only, 120s startup grace)..." -ForegroundColor Cyan
    $env:PYTHONPATH = "."
    $watchdog = Start-Process -FilePath $pyExe `
        -ArgumentList "scripts\dead_man_watchdog.py", "--start-grace-sec", "120" `
        -WorkingDirectory $root -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput "logs\watchdog.out.log" `
        -RedirectStandardError  "logs\watchdog.err.log"
    Write-Host "    watchdog PID $($watchdog.Id) - emails [CRITICAL] dead_man_alert if the brain hangs/dies." -ForegroundColor Green
} else {
    Write-Host "    WARNING: venv python not found ($pyExe) - dead-man watchdog NOT started." -ForegroundColor Yellow
    Write-Host "    Start it manually: `$env:PYTHONPATH='.'; venv\Scripts\python scripts\dead_man_watchdog.py" -ForegroundColor Yellow
}

Write-Host "==> Starting API server on http://0.0.0.0:8000 ..." -ForegroundColor Cyan
# --timeout-graceful-shutdown bounds uvicorn's wait for in-flight work on Ctrl+C;
# the in-process lifespan watchdog (app/main.py) is the hard backstop against hangs.
try {
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --timeout-graceful-shutdown 30
}
finally {
    if ($watchdog -and -not $watchdog.HasExited) {
        Write-Host "==> Stopping dead-man watchdog (PID $($watchdog.Id))..." -ForegroundColor Cyan
        Stop-Process -Id $watchdog.Id -Force -ErrorAction SilentlyContinue
    }
}
