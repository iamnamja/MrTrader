# _lib.ps1 — shared service helpers for start.ps1 / serve.ps1 / stop.ps1.
#
# Dot-source from a sibling script:   . (Join-Path $PSScriptRoot "_lib.ps1")
#
# WHY THIS EXISTS
# ---------------
# The three lifecycle scripts each managed processes their own way and drifted apart.
# serve.ps1 resolved the venv explicitly for the dead-man watchdog but called a bare
# `uvicorn` for the server, so it only ran where something else had already ACTIVATED
# the venv (the VS Code terminal does; a plain PowerShell does not). Separately,
# notify_watcher was started by NOTHING and stopped by NOTHING — it survived
# stop.ps1 -> serve.ps1 cycles by accident, and once killed, nothing brought it back,
# leaving the daily beacon silently enqueuing mail that was never sent.
#
# Every process lookup here matches on the COMMAND LINE rather than a pid file: a
# force-killed process never runs its atexit handler, so a pid file can outlive its
# owner, whereas the command line is the live truth.

# NOTE: deliberately no `Set-StrictMode` here. Dot-sourcing applies it to the CALLING
# script's scope too, so a strictness setting chosen for this file would silently change
# how start/serve/stop behave — turning e.g. an unset $LASTEXITCODE into a hard failure of
# the launcher. A shared library must not reach back and alter its callers' semantics.

function Get-MrtPython {
    <#  Absolute path to the venv interpreter, or $null when there is no venv.
        Callers must NOT assume a bare `python`/`uvicorn` is on PATH — that is only
        true inside an activated shell. #>
    param([Parameter(Mandatory = $true)][string]$Root)
    $py = Join-Path $Root "venv\Scripts\python.exe"
    if (Test-Path $py) { return $py }
    return $null
}

function Get-MrtProcess {
    <#  Python processes whose command line matches $Pattern (-like wildcards).
        Always returns an array, never $null, so .Count is safe. #>
    param([Parameter(Mandatory = $true)][string]$Pattern)
    $procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
             Where-Object { $_.CommandLine -and $_.CommandLine -like $Pattern }
    return @($procs)
}

function Test-MrtNotifyWatcher {
    return (Get-MrtProcess '*notify_watcher*').Count -gt 0
}

function Start-MrtNotifyWatcher {
    <#  Start the notification drainer if it is not already running.

        The app's own lifespan (app/main.py step 10) also starts it and terminates it on a
        CLEAN shutdown, so on a normal boot this is a no-op. It matters when the app does
        NOT come up: the dead-man watchdog reports a dead/hung brain by ENQUEUEING mail, so
        without a live drainer that alert is written and never sent — the app being down is
        exactly when you need the alert most.

        Guarded by a running-check rather than leaning on the script's own singleton lock,
        so a redundant launch cannot disturb the live watcher's log.

        Logs to its OWN file (notify_watcher.launcher.log), NOT the app's notify_watcher.log.
        Sharing one file does not work on Windows: a redirect holds a write handle for the
        child's lifetime, so while a launcher-started watcher was alive the app's own
        open(log, "a") failed with "[Errno 13] Permission denied" — destroying exactly the
        fallback this function exists to provide. Separate files also make it obvious after
        the fact WHICH starter won. This file is truncated per launch (Start-Process cannot
        append); the durable send history lives in the notifications DB's sent_at, and the
        app's own log is untouched. #>
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [string]$PyExe
    )
    if (Test-MrtNotifyWatcher) {
        Write-Host "    notify_watcher already running." -ForegroundColor Green
        return
    }
    if (-not $PyExe) { $PyExe = Get-MrtPython -Root $Root }
    if (-not $PyExe) {
        Write-Host "    WARNING: venv not found - notify_watcher NOT started." -ForegroundColor Yellow
        Write-Host "             Queued emails (incl. the daily beacon) will NOT be sent." -ForegroundColor Yellow
        return
    }

    $logDir = Join-Path $Root "logs"
    if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
    $script = Join-Path $Root "scripts\notify_watcher.py"
    $logPath = Join-Path $logDir "notify_watcher.launcher.log"
    try {
        # Plain Start-Process, no cmd.exe wrapper: writing to our own file removes the need for
        # shell append semantics, and with it a real quoting trap — passing @('/c', $line) as an
        # ARRAY makes PowerShell re-quote the element so cmd mis-parses it and the command
        # silently never runs (no process, no error, no log).
        [void](Start-Process -FilePath $PyExe -ArgumentList $script `
            -WorkingDirectory $Root -WindowStyle Hidden `
            -RedirectStandardOutput $logPath `
            -RedirectStandardError  (Join-Path $logDir "notify_watcher.launcher.err"))

        # POLL rather than sleep a fixed amount: the interpreter has to import the app package
        # and open SQLite before it registers, which routinely takes longer than a flat wait and
        # produced a false "did not stay up" warning while the watcher was in fact coming up.
        $up = $false
        for ($i = 0; $i -lt 25; $i++) {
            Start-Sleep -Milliseconds 200
            if (Test-MrtNotifyWatcher) { $up = $true; break }
        }
        if ($up) {
            Write-Host "    notify_watcher started - drains the email queue (daily beacon, alerts)." -ForegroundColor Green
        } else {
            Write-Host "    WARNING: notify_watcher did not come up within 5s - see $logPath" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "    WARNING: could not start notify_watcher: $($_.Exception.Message)" -ForegroundColor Yellow
    }
}

function Stop-MrtProcessByPattern {
    <#  Force-stop python processes matching $Pattern. Returns how many were stopped. #>
    param(
        [Parameter(Mandatory = $true)][string]$Pattern,
        [string]$Label = "process"
    )
    $procs = Get-MrtProcess $Pattern
    if ($procs.Count -eq 0) { return 0 }
    foreach ($p in $procs) {
        Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
    }
    Write-Host "    stopped $($procs.Count) $Label process(es)." -ForegroundColor Green
    return $procs.Count
}

function Stop-MrtNotifyWatcher {
    param([Parameter(Mandatory = $true)][string]$Root)
    $n = Stop-MrtProcessByPattern -Pattern '*notify_watcher*' -Label "notify_watcher"
    if ($n -eq 0) { return }
    # A forced kill skips the atexit release, so the pid file can outlive its owner.
    # acquire_singleton() already reclaims a stale lock, so this is tidiness, not
    # correctness — and it is only safe once no watcher remains.
    Start-Sleep -Milliseconds 300
    if (-not (Test-MrtNotifyWatcher)) {
        Remove-Item (Join-Path $Root "data\notify_watcher.pid") -Force -ErrorAction SilentlyContinue
    }
}

function Stop-MrtDeadManWatchdog {
    <#  serve.ps1 stops this in its `finally`, but only on a clean exit — if that window
        is killed the watchdog is orphaned and would keep emailing about a brain that is
        intentionally down. #>
    [void](Stop-MrtProcessByPattern -Pattern '*dead_man_watchdog*' -Label "dead-man watchdog")
}

function Test-MrtPortInUse {
    <#  True when something is already LISTENING on $Port.

        Callers must check this BEFORE starting anything. uvicorn runs the whole lifespan
        startup — DB migrations, all three agents, the scheduler, position reconciliation —
        and only THEN binds its socket, so a port clash is discovered far too late: the
        second instance boots a complete trading brain, runs reconciliation, and shuts down
        again. Two live brains for ~200ms is not a theoretical concern when both own
        schedulers that can fire. #>
    param([Parameter(Mandatory = $true)][int]$Port)
    $found = netstat -ano | Select-String ":$Port\s.*LISTENING"
    return ($found.Count -gt 0)
}

function Stop-MrtPort {
    <#  Kill whatever is LISTENING on $Port (uvicorn on 8000, vite on 3000). #>
    param([Parameter(Mandatory = $true)][int]$Port)
    $found = netstat -ano | Select-String ":$Port\s.*LISTENING"
    foreach ($line in $found) {
        $procId = ($line -replace '.*\s+(\d+)\s*$', '$1')
        if ($procId -match '^\d+$') {
            taskkill /PID $procId /F 2>$null | Out-Null
        }
    }
}
