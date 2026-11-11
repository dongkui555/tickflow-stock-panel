param(
    [int]$Port = 8188,
    [string]$QmtDataRoot = $env:QMT_DATA_ROOT
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogDir = Join-Path $ScriptDir "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Python = (Get-Command python -ErrorAction Stop).Source
& $Python -c "import requests, websocket" 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "Python needs requests and websocket-client. Run: python -m pip install requests websocket-client"
}
if ($QmtDataRoot) {
    $QmtDataRoot = (Resolve-Path -LiteralPath $QmtDataRoot -ErrorAction Stop).Path
    & $Python -c "import numpy" 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Exact QMT DAT volumes need numpy in the Windows bridge runtime."
    }
}

$Existing = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($Existing) {
    Write-Host "QMT bridge is already listening on port $Port."
    exit 0
}

$Stdout = Join-Path $LogDir "qmt-bridge.out.log"
$Stderr = Join-Path $LogDir "qmt-bridge.err.log"
$BridgeArgs = @((Join-Path $ScriptDir "qmt_http_api.py"), "--port", $Port)
if ($QmtDataRoot) {
    $BridgeArgs += @("--qmt-data-root", "`"$QmtDataRoot`"")
}
$Process = Start-Process -FilePath $Python `
    -ArgumentList $BridgeArgs `
    -WindowStyle Hidden -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -PassThru

Start-Sleep -Seconds 2
if ($Process.HasExited) {
    throw "QMT bridge failed to start. See $Stderr"
}
Write-Host "QMT bridge started (PID $($Process.Id)): http://127.0.0.1:$Port"
