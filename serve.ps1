# serve.ps1 — production-style launch: build the frontend, then serve via uvicorn.
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

# ── Ensure the database is up (Postgres + Redis) ──────────────────────────────
# stop.ps1 stops these Docker containers; serve.ps1 only starts uvicorn — so a
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
    Write-Host "    fresh infra). Starting the API anyway — it may fail on DB connect." -ForegroundColor Yellow
}

Write-Host "==> Starting API server on http://0.0.0.0:8000 ..." -ForegroundColor Cyan
# --timeout-graceful-shutdown bounds uvicorn's wait for in-flight work on Ctrl+C;
# the in-process lifespan watchdog (app/main.py) is the hard backstop against hangs.
uvicorn app.main:app --host 0.0.0.0 --port 8000 --timeout-graceful-shutdown 30
