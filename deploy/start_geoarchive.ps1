$ErrorActionPreference = "Stop"

$project = "C:\FINAM\Conference"
$sshTarget = "msa-aiad01-ap04"

function Has-ProcessCommand([string]$pattern) {
    return [bool](Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match $pattern })
}

if (-not (Has-ProcessCommand "18080:127\.0\.0\.1:18080")) {
    Start-Process ssh -ArgumentList @(
        "-N", "-L", "18080:127.0.0.1:18080",
        "-o", "ExitOnForwardFailure=yes",
        "-o", "ServerAliveInterval=30",
        $sshTarget
    ) -WindowStyle Hidden
    Start-Sleep -Seconds 2
}

if (-not (Has-ProcessCommand "uvicorn (api|app\.main):app.*8765")) {
    Start-Process python -ArgumentList @(
        "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8765"
    ) -WorkingDirectory "$project\demo_app\backend" -WindowStyle Hidden
}

if (-not (Has-ProcessCommand "streamlit run app.py.*8501")) {
    Start-Process streamlit -ArgumentList @(
        "run", "app.py", "--server.port", "8501"
    ) -WorkingDirectory "$project\demo_app" -WindowStyle Hidden
}

Start-Sleep -Seconds 2
$services = Invoke-RestMethod "http://127.0.0.1:8765/api/services" -TimeoutSec 10
$services | ConvertTo-Json -Depth 5
