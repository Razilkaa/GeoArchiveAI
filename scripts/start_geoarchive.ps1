$ErrorActionPreference = "Stop"

$project = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$sshTarget = $env:GEOARCHIVE_SSH_TARGET
$apiHost = if ($env:GEOARCHIVE_API_HOST) { $env:GEOARCHIVE_API_HOST } else { "127.0.0.1" }
$apiPort = if ($env:GEOARCHIVE_API_PORT) { $env:GEOARCHIVE_API_PORT } else { "8765" }
$webPort = if ($env:GEOARCHIVE_WEB_PORT) { $env:GEOARCHIVE_WEB_PORT } else { "5173" }
$env:GEOARCHIVE_ROOT = $project
$env:GEOARCHIVE_API_URL = "http://$apiHost`:$apiPort"

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

if ($sshTarget -and -not (Has-ProcessCommand "18080:127\.0\.0\.1:18080")) {
    Start-Process ssh -ArgumentList @(
        "-N", "-L", "18080:127.0.0.1:18080",
        "-o", "ExitOnForwardFailure=yes",
        "-o", "ServerAliveInterval=30",
        $sshTarget
    ) -WindowStyle Hidden
    Start-Sleep -Seconds 2
}

if (-not (Test-Http "http://$apiHost`:$apiPort/health")) {
    Start-Process python -ArgumentList @(
        "-m", "uvicorn", "app.main:app", "--host", $apiHost, "--port", $apiPort
    ) -WorkingDirectory "$project\apps\api" -WindowStyle Hidden `
      -RedirectStandardOutput "$serviceLogs\api.stdout.log" `
      -RedirectStandardError "$serviceLogs\api.stderr.log"
}

if (-not (Test-Http "http://127.0.0.1:$webPort")) {
    if (-not (Test-Path "$project\apps\web\node_modules")) {
        & npm.cmd install --prefix "$project\apps\web"
    }
    Start-Process npm.cmd -ArgumentList @(
        "run", "dev", "--", "--host", "127.0.0.1", "--port", $webPort
    ) -WorkingDirectory "$project\apps\web" -WindowStyle Hidden `
      -RedirectStandardOutput "$serviceLogs\web.stdout.log" `
      -RedirectStandardError "$serviceLogs\web.stderr.log"
}

for ($attempt = 0; $attempt -lt 15; $attempt++) {
    if ((Test-Http "http://$apiHost`:$apiPort/health") -and (Test-Http "http://127.0.0.1:$webPort")) {
        break
    }
    Start-Sleep -Seconds 1
}
$services = Invoke-RestMethod "http://$apiHost`:$apiPort/api/services" -TimeoutSec 10
$services | ConvertTo-Json -Depth 5
