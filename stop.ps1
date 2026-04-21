# MrTrader stop script
Write-Host "=== Stopping MrTrader ===" -ForegroundColor Cyan

# Kill uvicorn
$pids = (netstat -ano | Select-String ":8000.*LISTENING") -replace '.*\s+(\d+)$','$1'
foreach ($p in $pids) {
    if ($p -match '^\d+$') {
        taskkill /PID $p /F 2>$null | Out-Null
    }
}

# Kill vite dev server
$pids2 = (netstat -ano | Select-String ":3000.*LISTENING") -replace '.*\s+(\d+)$','$1'
foreach ($p in $pids2) {
    if ($p -match '^\d+$') {
        taskkill /PID $p /F 2>$null | Out-Null
    }
}

# Stop infrastructure (keep data volumes)
docker stop mrtrader_postgres mrtrader_redis 2>$null | Out-Null

Write-Host "All services stopped." -ForegroundColor Green
