# Start the human pilot (#35) on this laptop and put it online through ngrok.
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_pilot.ps1 -Domain delicious-huddling-generic.ngrok-free.dev
#
# Needs: .venv with the server extras, ngrok signed in once (ngrok config add-authtoken ...),
# and the free static domain from https://dashboard.ngrok.com/domains.
# Stop with Ctrl+C: this stops ngrok and the server. Answers already saved stay in pilot_data/.

param(
    [Parameter(Mandatory = $true)][string]$Domain,
    [string]$Items = "server/pilot_items.json",
    [string]$DataDir = "pilot_data",
    [int]$Port = 8000
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not (Test-Path $Items)) { throw "No question bank at $Items. Build the pilot bank first (#35)." }
$placeholder = Select-String -Path docs\pilot\consent.md, docs\pilot\debrief.md -Pattern '<[A-Z_]+>'
if ($placeholder) { throw "Fill in the placeholder before anyone sees it: $($placeholder[0].Line.Trim())" }
# Sync services (OneDrive, Dropbox, Google Drive) keep deleted files in their recycle bin and
# version history, so "Remove my answers" would not really remove them.
$full = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($DataDir)
if ($full -match 'OneDrive|Dropbox|Google Drive|iCloud') { throw "$full is in a synced folder; use a local one." }
$ngrok = (Get-Command ngrok -ErrorAction SilentlyContinue).Source
if (-not $ngrok) {
    $ngrok = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Recurse -Filter ngrok.exe |
        Select-Object -First 1 -ExpandProperty FullName
}
if (-not $ngrok) { throw "ngrok not found. Install it: winget install Ngrok.Ngrok" }

$env:ADR_PILOT = "1"
$env:ADR_DATA_DIR = $full
$env:ADR_ITEMS = $Items
Remove-Item Env:ADR_DEBUG_STATE, Env:ADR_STUB_MECHANISMS -ErrorAction SilentlyContinue

# 127.0.0.1: only ngrok can reach the server. --no-access-log: behind a tunnel uvicorn would
# print every visitor's real IP address, and the consent screen says we don't record IPs.
# --inspect=false: otherwise ngrok's inspector (http://127.0.0.1:4040) keeps every request in
# full, answers and visitors' IP addresses included.
$server = Start-Process -PassThru -NoNewWindow ".venv\Scripts\python.exe" `
    "-m uvicorn server.app:app --host 127.0.0.1 --port $Port --no-access-log"
try {
    Start-Sleep -Seconds 3
    if ($server.HasExited) { throw "The server stopped at startup; see the error above." }
    $n = (Get-ChildItem $full -Filter *.jsonl -ErrorAction SilentlyContinue |
        Select-String -List '"finish"').Count
    Write-Host "Server up on http://127.0.0.1:$Port  ($n finished sessions saved so far)"
    Write-Host "Public link: https://$Domain"
    & $ngrok http $Port --url "https://$Domain" --inspect=false --log stdout --log-level warn
} finally {
    if (-not $server.HasExited) { Stop-Process -Id $server.Id }
    Write-Host "Stopped. Saved answers are in $full"
}
