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

Write-Host "==> Starting API server on http://0.0.0.0:8000 ..." -ForegroundColor Cyan
Set-Location $root
uvicorn app.main:app --host 0.0.0.0 --port 8000
