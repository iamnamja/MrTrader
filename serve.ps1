# serve.ps1 - production-style launch: build the frontend, then serve via uvicorn.
# PowerShell 5.1-safe (replaces the `&&` chaining that 5.1 does not support).
# The server only starts if the frontend build succeeds (exit-code check mirrors `&&`).
#
# Usage:  .\serve.ps1
#
# (For dev mode with frontend hot-reload on :3000 + Docker infra, use .\start.ps1 instead.)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
. (Join-Path $root "_lib.ps1")

# --- Resolve the venv interpreter UP FRONT and preflight uvicorn ------------------
# This script must not depend on the venv being ACTIVATED. The VS Code Python extension
# activates it automatically, so a bare `uvicorn` works there and nowhere else — running
# .\serve.ps1 from a plain PowerShell window died with "uvicorn is not recognized" only
# AFTER the frontend build and AFTER the watchdog had been started (which `finally` then
# tore straight back down). Checking here fails in one second with an actionable message
# instead of ~10 seconds in with a half-started stack.
$pyExe = Get-MrtPython -Root $root
$useVenv = [bool]$pyExe
if ($useVenv) {
    # try/catch AND exit-code check: under $ErrorActionPreference='Stop', PowerShell 5.1 turns a
    # native command's STDERR into a terminating error, so a missing module throws a RemoteException
    # instead of just setting $LASTEXITCODE. Without the catch this preflight would die with a stack
    # trace in precisely the case it exists to explain.
    $uvicornOk = $false
    try {
        & $pyExe -m uvicorn --version 2>&1 | Out-Null
        $uvicornOk = ($LASTEXITCODE -eq 0)
    } catch {
        $uvicornOk = $false
    }
    if (-not $uvicornOk) {
        Write-Host "uvicorn is not installed in the venv ($pyExe)." -ForegroundColor Red
        Write-Host "Fix: & '$pyExe' -m pip install -r requirements.txt" -ForegroundColor Yellow
        exit 1
    }
} else {
    Write-Host "WARNING: venv not found at $(Join-Path $root 'venv\Scripts\python.exe')" -ForegroundColor Yellow
    Write-Host "         Falling back to whatever is on PATH. The dead-man watchdog and" -ForegroundColor Yellow
    Write-Host "         notify_watcher will NOT start without the venv." -ForegroundColor Yellow
    if (-not (Get-Command uvicorn -ErrorAction SilentlyContinue)) {
        Write-Host "uvicorn not found on PATH either - cannot start the server." -ForegroundColor Red
        Write-Host "Fix: create the venv, or activate it before running this script." -ForegroundColor Yellow
        exit 1
    }
}

# --- Refuse to start if something already owns port 8000 --------------------------
# Checked here, before ANY work, because uvicorn discovers a port clash far too late: it
# runs the entire lifespan startup first — DB migrations, all three agents, the scheduler,
# position reconciliation — and only then binds. A second .\serve.ps1 therefore boots a
# COMPLETE second trading brain, reconciles positions, and shuts down again, leaving two
# live orchestrators overlapping for a couple hundred milliseconds. Both own schedulers
# that can fire, so that overlap is a real double-act hazard, not a cosmetic one.
#
# Deliberately refuses rather than killing the incumbent (which is what start.ps1's dev
# flow does): silently terminating a running trading server is not something a
# production-style launcher should decide on its own.
if (Test-MrtPortInUse -Port 8000) {
    Write-Host ""
    Write-Host "Port 8000 is already in use - MrTrader looks like it is ALREADY RUNNING." -ForegroundColor Red
    Write-Host "Refusing to start a second instance (two brains would both schedule and trade)." -ForegroundColor Red
    Write-Host ""
    Write-Host "  Stop the running one first:   .\stop.ps1" -ForegroundColor Yellow
    Write-Host "  Then start fresh:             .\serve.ps1" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  Currently listening on :8000 -" -ForegroundColor Gray
    netstat -ano | Select-String ":8000\s.*LISTENING" | ForEach-Object { Write-Host "   $_" -ForegroundColor Gray }
    exit 1
}

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
if ($useVenv) {
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

# --- Start the notification drainer -------------------------------------------------
# Nothing used to start this, so it survived stop.ps1 -> serve.ps1 cycles only by accident:
# stop.ps1 never killed it either. Once killed, nothing brought it back, and the daily
# liveness beacon would enqueue mail that was never sent — a silent failure of the very
# thing meant to detect silence. Idempotent, so re-running serve.ps1 is safe.
#
# Deliberately NOT stopped in the `finally` below (unlike the dead-man watchdog): it owns
# no trading authority, draining a queue while the brain is down is harmless, and stopping
# it is stop.ps1's job. Ctrl+C on this window therefore leaves mail delivery working.
Write-Host "==> Starting notification watcher..." -ForegroundColor Cyan
Start-MrtNotifyWatcher -Root $root -PyExe $pyExe

Write-Host "==> Starting API server on http://0.0.0.0:8000 ..." -ForegroundColor Cyan
# --timeout-graceful-shutdown bounds uvicorn's wait for in-flight work on Ctrl+C;
# the in-process lifespan watchdog (app/main.py) is the hard backstop against hangs.
try {
    # Invoke through the venv interpreter EXPLICITLY (see the preflight above). `-m uvicorn`
    # rather than venv\Scripts\uvicorn.exe keeps $pyExe as the single source of truth and still
    # works if that console-script shim is missing or stale.
    if ($useVenv) {
        & $pyExe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --timeout-graceful-shutdown 30
    } else {
        uvicorn app.main:app --host 0.0.0.0 --port 8000 --timeout-graceful-shutdown 30
    }
}
finally {
    if ($watchdog -and -not $watchdog.HasExited) {
        Write-Host "==> Stopping dead-man watchdog (PID $($watchdog.Id))..." -ForegroundColor Cyan
        Stop-Process -Id $watchdog.Id -Force -ErrorAction SilentlyContinue
    }
}
