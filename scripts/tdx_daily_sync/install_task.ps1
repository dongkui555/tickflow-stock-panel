param(
    [Parameter(Mandatory = $true)][string]$Python,
    [Parameter(Mandatory = $true)][string]$Vipdoc,
    [Parameter(Mandatory = $true)][string]$Logs
)

$ErrorActionPreference = 'Stop'
$name = 'TickStockPanel-TDX-DailySync'
if ((Get-TimeZone).Id -ne 'China Standard Time') {
    throw 'The Windows scheduler must use China Standard Time for the 17:00 trigger.'
}
if (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue) {
    throw "Task already exists. Inspect $name before modifying it."
}
$pythonPath = (Resolve-Path -LiteralPath $Python).Path
$vipdocPath = (Resolve-Path -LiteralPath $Vipdoc).Path
$entry = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot 'client.py')).Path
$logPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Logs)
New-Item -ItemType Directory -Path $logPath -Force | Out-Null
$arguments = '-X utf8 "{0}" --tdx "{1}" --logs "{2}"' -f $entry, $vipdocPath, $logPath
$action = New-ScheduledTaskAction -Execute $pythonPath -Argument $arguments -WorkingDirectory $PSScriptRoot
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At '17:00'
$account = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$principal = New-ScheduledTaskPrincipal -UserId $account -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 2) -Hidden
$task = @{
    TaskName = $name
    Action = $action
    Trigger = $trigger
    Settings = $settings
    Principal = $principal
    Description = '17:00 CST trade-calendar-gated TDX LC1 and professional finance import into Ubuntu Stock Panel.'
}
Register-ScheduledTask @task
