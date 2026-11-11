[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$QmtRoot,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^\d{4}-\d{2}-\d{2}$')]
    [string]$StartDate,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^\d{4}-\d{2}-\d{2}$')]
    [string]$EndDate,

    [string]$DataDir = (Join-Path $PSScriptRoot '..\data'),
    [ValidateSet('BJ', 'SH', 'SZ')]
    [string[]]$Markets = @('SZ'),
    [int]$Workers = 8,
    [string]$Image = 'tick-stock-panel-v022-integrated-app',
    [string]$Audit,
    [switch]$IncludeConflicts,
    [switch]$RepairConflicts,
    [switch]$DryRun,
    [switch]$KeepAppRunning
)

$ErrorActionPreference = 'Stop'

function Invoke-DockerChecked {
    param([string[]]$Arguments)
    & docker @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "docker command failed with exit code ${LASTEXITCODE}: docker $($Arguments -join ' ')"
    }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$dataRoot = (Resolve-Path $DataDir).Path
$qmtRootResolved = (Resolve-Path $QmtRoot).Path
if (-not (Test-Path (Join-Path $qmtRootResolved 'SH\60')) -and
    -not (Test-Path (Join-Path $qmtRootResolved 'SZ\60')) -and
    -not (Test-Path (Join-Path $qmtRootResolved 'BJ\60'))) {
    throw "QMT root does not contain SH\60, SZ\60 or BJ\60: $qmtRootResolved"
}
if ($Workers -lt 1 -or $Workers -gt 16) {
    throw 'Workers must be between 1 and 16.'
}

& docker image inspect $Image *> $null
if ($LASTEXITCODE -ne 0) {
    throw "Docker image not found: $Image. Build the app image before running this maintenance command."
}

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$auditName = "minute_audit_$stamp"
$auditHostDir = Join-Path $dataRoot "import_reports\$auditName"
$auditContainerCsv = "/app/data/import_reports/$auditName/missing_or_partial.csv"
$auditWasCreated = $false
$stoppedApp = $false

try {
    if ($Audit) {
        $auditFull = (Resolve-Path $Audit).Path
        $relative = [IO.Path]::GetRelativePath($dataRoot, $auditFull)
        if ($relative.StartsWith('..') -or [IO.Path]::IsPathRooted($relative)) {
            throw "Audit file must be inside DataDir so Docker can mount it: $auditFull"
        }
        $auditContainerCsv = '/app/data/' + ($relative -replace '\\', '/')
    } else {
        New-Item -ItemType Directory -Force -Path $auditHostDir | Out-Null
        $auditArgs = @(
            'run', '--rm', '--user', '0:0',
            '-v', "$($dataRoot):/app/data",
            '-v', "$($repoRoot)\scripts:/app/scripts:ro",
            $Image,
            '/app/.venv/bin/python', '/app/scripts/audit_market_minute_coverage.py',
            '--data-dir', '/app/data',
            '--start', $StartDate,
            '--end', $EndDate,
            '--output-dir', "/app/data/import_reports/$auditName",
            '--markets'
        ) + @($Markets)
        Write-Host "[minute-k] auditing $StartDate..$EndDate ($($Markets -join ','))"
        Invoke-DockerChecked $auditArgs
        $auditWasCreated = $true
    }

    if (-not $DryRun -and -not $KeepAppRunning) {
        $running = ((& docker inspect --format '{{.State.Running}}' TickFlow_Stock_Panel 2>$null) -eq 'true')
        if ($running) {
            Write-Host '[minute-k] stopping app for atomic offline maintenance'
            & docker-compose stop app
            if ($LASTEXITCODE -ne 0) { throw 'docker-compose stop app failed' }
            $stoppedApp = $true
        }
    }

    $importArgs = @(
        'run', '--rm', '--user', '0:0',
        '-v', "$($dataRoot):/app/data",
        '-v', "$($qmtRootResolved):/qmt:ro",
        '-v', "$($repoRoot)\scripts:/app/scripts:ro",
        $Image,
        '/app/.venv/bin/python', '/app/scripts/minute_k_import.py',
        '--data-dir', '/app/data',
        '--qmt-root', '/qmt',
        '--audit', $auditContainerCsv,
        '--start', $StartDate,
        '--end', $EndDate,
        '--workers', [string]$Workers,
        '--markets'
    ) + @($Markets)
    if ($IncludeConflicts) { $importArgs += '--include-conflicts' }
    if ($RepairConflicts) { $importArgs += '--repair-conflicts' }
    if ($DryRun) { $importArgs += '--dry-run' }

    Write-Host "[minute-k] importing QMT data from $qmtRootResolved"
    Invoke-DockerChecked $importArgs
}
finally {
    if ($stoppedApp) {
        Write-Host '[minute-k] starting app again'
        & docker-compose start app
        if ($LASTEXITCODE -ne 0) { Write-Warning 'docker-compose start app failed; run docker-compose start app manually.' }
    }
}

if ($auditWasCreated) {
    Write-Host "[minute-k] audit report: $auditHostDir"
}
if (-not $DryRun) {
    try {
        $health = Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:3018/health' -TimeoutSec 10
        Write-Host "[minute-k] app health: $($health.StatusCode)"
    } catch {
        Write-Warning "App health check failed: $($_.Exception.Message)"
    }
}
