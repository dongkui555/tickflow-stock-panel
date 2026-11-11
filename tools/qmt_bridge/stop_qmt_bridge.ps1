param(
    [int]$Port = 8188
)

$Connections = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
$Pids = @($Connections | Select-Object -ExpandProperty OwningProcess -Unique)
foreach ($ProcessId in $Pids) {
    Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
}
Write-Host "Stopped $($Pids.Count) QMT bridge process(es) on port $Port."
