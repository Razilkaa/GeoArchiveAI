$ErrorActionPreference = "Stop"

$project = "C:\FINAM\Conference"
$sshTarget = "msa-aiad01-ap04"

function Has-ProcessCommand([string]$pattern) {
    return [bool](Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match $pattern })
}

function Test-Http([string]$url) {
    try {
        Invoke-WebRequest $url -UseBasicParsing -TimeoutSec 2 | Out-Null
        return $true
    } catch {
        return $false
    }
}

$serviceLogs = "$project\work\logs"
New-Item -ItemType Directory -Force $serviceLogs | Out-Null

if (-not (Has-ProcessCommand "18080:127\.0\.0\.1:18080")) {
    Start-Process ssh -ArgumentList @(
        "-N", "-L", "18080:127.0.0.1:18080",
        "-o", "ExitOnForwardFailure=yes",
        "-o", "ServerAliveInterval=30",
        $sshTarget
    ) -WindowStyle Hidden
    Start-Sleep -Seconds 2
}

if (-not (Test-Http "http://127.0.0.1:8765/health")) {
    Start-Process python -ArgumentList @(
        "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8765"
    ) -WorkingDirectory "$project\apps\api" -WindowStyle Hidden `
      -RedirectStandardOutput "$serviceLogs\api.stdout.log" `
      -RedirectStandardError "$serviceLogs\api.stderr.log"
}

if (-not (Test-Http "http://127.0.0.1:8501")) {
    Start-Process python -ArgumentList @(
        "-m", "streamlit", "run", "app.py",
        "--server.port", "8501",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false"
    ) -WorkingDirectory "$project\apps\web" -WindowStyle Hidden `
      -RedirectStandardOutput "$serviceLogs\web.stdout.log" `
      -RedirectStandardError "$serviceLogs\web.stderr.log"
}

for ($attempt = 0; $attempt -lt 15; $attempt++) {
    if ((Test-Http "http://127.0.0.1:8765/health") -and (Test-Http "http://127.0.0.1:8501")) {
        break
    }
    Start-Sleep -Seconds 1
}
$services = Invoke-RestMethod "http://127.0.0.1:8765/api/services" -TimeoutSec 10
$services | ConvertTo-Json -Depth 5
