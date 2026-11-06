# Start the human pilot (#35) on this laptop and put it online through ngrok.
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_pilot.ps1 -Domain your-name.ngrok-free.app
#
# Needs: .venv with the server extras, ngrok signed in once (ngrok config add-authtoken ...),
# and the free static domain from https://dashboard.ngrok.com/domains.
# Stop with Ctrl+C: this stops ngrok and the server. Answers already saved stay in pilot_data/.

param(
    [Parameter(Mandatory = $true)][string]$Domain,
    [string]$Items = "server/pilot_items.json",
    [int]$Port = 8000
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not (Test-Path $Items)) { throw "No question bank at $Items. Build the pilot bank first (#35)." }
$ngrok = (Get-Command ngrok -ErrorAction SilentlyContinue).Source
if (-not $ngrok) {
    $ngrok = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Recurse -Filter ngrok.exe |
        Select-Object -First 1 -ExpandProperty FullName
}
if (-not $ngrok) { throw "ngrok not found. Install it: winget install Ngrok.Ngrok" }

$env:ADR_PILOT = "1"
$env:ADR_DATA_DIR = "pilot_data"
$env:ADR_ITEMS = $Items
Remove-Item Env:ADR_DEBUG_STATE, Env:ADR_STUB_MECHANISMS -ErrorAction SilentlyContinue

# 127.0.0.1: only ngrok can reach the server. --no-access-log: behind a tunnel uvicorn would
# print every visitor's real IP address, and the consent screen says we don't record IPs.
$server = Start-Process -PassThru -NoNewWindow ".venv\Scripts\python.exe" `
    "-m uvicorn server.app:app --host 127.0.0.1 --port $Port --no-access-log"
try {
    Start-Sleep -Seconds 3
    if ($server.HasExited) { throw "The server stopped at startup; see the error above." }
    $n = (Get-ChildItem pilot_data -Filter *.jsonl -ErrorAction SilentlyContinue |
        Select-String -List '"finish"').Count
    Write-Host "Server up on http://127.0.0.1:$Port  ($n finished sessions saved so far)"
    Write-Host "Public link: https://$Domain"
    & $ngrok http $Port --url "https://$Domain" --log stdout --log-level warn
} finally {
    if (-not $server.HasExited) { Stop-Process -Id $server.Id }
    Write-Host "Stopped. Saved answers are in pilot_data/."
}
