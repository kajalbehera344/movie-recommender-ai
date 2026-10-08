# Starts the FastAPI backend and the Streamlit frontend in separate windows.
# First run also creates the venv and installs dependencies (needs internet, ~3-5 min).
# Usage:  powershell -ExecutionPolicy Bypass -File .\run_demo.ps1

$root = $PSScriptRoot
$py = Join-Path $root "venv\Scripts\python.exe"

function Fail($msg) {
    Write-Host ""
    Write-Host $msg -ForegroundColor Red
    exit 1
}

$envFile = Join-Path $root ".env"
if (-not (Test-Path $envFile) -or (Select-String -Path $envFile -Pattern "your_tmdb_api_key_here" -Quiet)) {
    Fail "Put your TMDB API key in .env first (free key: https://www.themoviedb.org/settings/api)"
}

$busy = @(Get-NetTCPConnection -LocalPort 8000, 8501 -State Listen -ErrorAction SilentlyContinue)
if ($busy) {
    Fail ("Port(s) $(($busy.LocalPort | Sort-Object -Unique) -join ', ') already in use - the demo is probably already running " +
        "in another window (try http://localhost:8501). Close those windows first to restart it.")
}

if (-not (Test-Path $py)) {
    # Some installed library files are ~165 chars deep; Windows' default 260-char path limit breaks pip
    $longPaths = (Get-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -ErrorAction SilentlyContinue).LongPathsEnabled
    if ($longPaths -ne 1 -and $root.Length -gt 90) {
        Fail ("This folder's path is too long for Windows ($($root.Length) characters, max 90):`n  $root`n" +
            "Move the Movie-Recommender folder somewhere shorter, e.g. C:\Movie-Recommender, and run again.")
    }

    # Find a Python 3.10-3.12 (the pinned libraries don't support 3.13 yet)
    $base = $null
    foreach ($v in "3.12", "3.11", "3.10") {
        try { & py "-$v" -c "pass" *> $null; if ($LASTEXITCODE -eq 0) { $base = @("py", "-$v"); break } } catch {}
    }
    if (-not $base) {
        try {
            $ver = & python -c "import sys; print('%d.%d' % sys.version_info[:2])" 2> $null
            if ($ver -in "3.10", "3.11", "3.12") { $base = @("python") }
        } catch {}
    }
    if (-not $base) {
        Fail "Python 3.10, 3.11 or 3.12 is required. Install it from https://www.python.org/downloads/ (tick 'Add Python to PATH')."
    }

    Write-Host "First run: creating virtual environment and installing dependencies (a few minutes)..."
    $baseArgs = @($base | Select-Object -Skip 1)
    & $base[0] @baseArgs -m venv (Join-Path $root "venv")
    & $py -m pip install --upgrade pip -q
    & $py -m pip install -r (Join-Path $root "requirements.txt")
    if ($LASTEXITCODE -ne 0) {
        Remove-Item (Join-Path $root "venv") -Recurse -Force -ErrorAction SilentlyContinue
        Fail "Dependency install failed - see the messages above. Check your internet connection and run again."
    }
}

Write-Host "Starting backend on http://127.0.0.1:8000 ..."
Start-Process powershell -WorkingDirectory $root -ArgumentList "-NoExit", "-Command", "& '$py' -m uvicorn main:app --host 127.0.0.1 --port 8000"

# Wait for the backend to load the pickles before opening the UI
$ready = $false
for ($i = 0; $i -lt 60; $i++) {
    try {
        Invoke-RestMethod "http://127.0.0.1:8000/health" -TimeoutSec 2 | Out-Null
        $ready = $true
        break
    } catch { Start-Sleep -Seconds 1 }
}
if (-not $ready) { Write-Warning "Backend did not respond on /health yet - check its window for errors." }

Write-Host "Starting frontend on http://localhost:8501 ..."
Start-Process powershell -WorkingDirectory $root -ArgumentList "-NoExit", "-Command", "& '$py' -m streamlit run app.py --server.port 8501"
