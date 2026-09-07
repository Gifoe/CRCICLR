param(
    [string]$Python = "D:\nips-temp\TotalP\P2\.conda\gpu-baseline-v1\python.exe",
    [int]$MaxParallel = 5
)

$ErrorActionPreference = "Stop"
$Experiment = Split-Path -Parent $PSScriptRoot
$Logs = Join-Path $Experiment "runtime\logs"
$Output = Join-Path $Logs "scheduled_orchestrator.out"
$ErrorLog = Join-Path $Logs "scheduled_orchestrator.err"
$Status = Join-Path $Experiment "runtime\SCHEDULED_TASK_STATUS.json"
New-Item -ItemType Directory -Force -Path $Logs | Out-Null

$Started = Get-Date
try {
    & (Join-Path $PSScriptRoot "run_all_server.ps1") `
        -Python $Python `
        -MaxParallel $MaxParallel *>&1 |
        Tee-Object -FilePath $Output
    $Payload = [ordered]@{
        status = "COMPLETE"
        started = $Started.ToString("o")
        finished = (Get-Date).ToString("o")
        max_parallel = $MaxParallel
        python = $Python
        ssh_session_independent = $true
    }
    $Payload | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $Status -Encoding UTF8
}
catch {
    $_ | Out-String | Set-Content -LiteralPath $ErrorLog -Encoding UTF8
    $Payload = [ordered]@{
        status = "FAILED"
        started = $Started.ToString("o")
        finished = (Get-Date).ToString("o")
        max_parallel = $MaxParallel
        python = $Python
        ssh_session_independent = $true
        error = ($_ | Out-String)
    }
    $Payload | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $Status -Encoding UTF8
    exit 1
}
