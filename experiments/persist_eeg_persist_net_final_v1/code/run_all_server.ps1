param(
    [string]$Python = "D:\nips-temp\TotalP\P2\.conda\gpu-baseline-v1\python.exe",
    [int]$MaxParallel = 7
)

$ErrorActionPreference = "Stop"
$Experiment = Split-Path -Parent $PSScriptRoot
$Runtime = Join-Path $Experiment "runtime"
$Logs = Join-Path $Runtime "logs"
New-Item -ItemType Directory -Force -Path $Logs | Out-Null

if ($MaxParallel -lt 1 -or $MaxParallel -gt 8) {
    throw "MaxParallel must be in 1..8; the RTX 5090 memory audit authorizes at most 8."
}

foreach ($Fold in 0..4) {
    $SelectionPath = Join-Path $Runtime ("selection\FOLD_{0}.json" -f $Fold)
    if (-not (Test-Path -LiteralPath $SelectionPath)) {
        throw "Missing source-only selection: $SelectionPath"
    }
    $Selection = Get-Content -LiteralPath $SelectionPath -Raw | ConvertFrom-Json
    if ($Selection.engineering_pathology_detected -or
        -not $Selection.outer_outcome_subjects_absent_from_selection -or
        $Selection.target_future_outer_labels_used -or
        $Selection.internal_holdout_accessed -or
        $Selection.outer_test_used) {
        throw "Fold $Fold failed the source-only selection/purity gate."
    }
}

$Tasks = @()
foreach ($Seed in 0..2) {
    foreach ($Fold in 0..4) {
        $Tasks += [pscustomobject]@{ Fold = $Fold; Seed = $Seed }
    }
}

$Running = @()
$Ledger = @()

function Reap-Finished {
    $StillRunning = @()
    foreach ($Entry in $script:Running) {
        $Entry.Process.Refresh()
        if ($Entry.Process.HasExited) {
            $Finished = Get-Date
            $Code = $Entry.Process.ExitCode
            Write-Output ("FINISHED fold={0} seed={1} pid={2} exit={3}" -f $Entry.Fold, $Entry.Seed, $Entry.Process.Id, $Code)
            $script:Ledger += [pscustomobject]@{
                fold = $Entry.Fold
                seed = $Entry.Seed
                pid = $Entry.Process.Id
                exit_code = $Code
                started = $Entry.Started.ToString("o")
                finished = $Finished.ToString("o")
                output = $Entry.Output
                error = $Entry.Error
            }
        }
        else {
            $StillRunning += $Entry
        }
    }
    $script:Running = @($StillRunning)
}

foreach ($Task in $Tasks) {
    while ($Running.Count -ge $MaxParallel) {
        Start-Sleep -Seconds 5
        Reap-Finished
    }
    $Output = Join-Path $Logs ("run_fold_{0}_seed_{1}.out" -f $Task.Fold, $Task.Seed)
    $ErrorLog = Join-Path $Logs ("run_fold_{0}_seed_{1}.err" -f $Task.Fold, $Task.Seed)
    $Started = Get-Date
    $Process = Start-Process -FilePath $Python `
        -ArgumentList @("code\run_experiment.py", "run", "--fold", [string]$Task.Fold, "--seed", [string]$Task.Seed) `
        -WorkingDirectory $Experiment `
        -RedirectStandardOutput $Output `
        -RedirectStandardError $ErrorLog `
        -WindowStyle Hidden `
        -PassThru
    $Running += [pscustomobject]@{
        Fold = $Task.Fold
        Seed = $Task.Seed
        Process = $Process
        Started = $Started
        Output = $Output
        Error = $ErrorLog
    }
    Write-Output ("STARTED fold={0} seed={1} pid={2}" -f $Task.Fold, $Task.Seed, $Process.Id)
}

while ($Running.Count -gt 0) {
    Start-Sleep -Seconds 5
    Reap-Finished
}

$LedgerPath = Join-Path $Runtime "SERVER_ORCHESTRATION.json"
$Payload = [ordered]@{
    max_parallel = $MaxParallel
    python = $Python
    runs = @($Ledger | Sort-Object seed, fold)
    all_exit_zero = -not [bool]($Ledger | Where-Object { $_.exit_code -ne 0 })
    completed = $Ledger.Count
    expected = 15
    outer_results_inspected_during_execution = $false
}
$Payload | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $LedgerPath -Encoding UTF8

if (-not $Payload.all_exit_zero -or $Payload.completed -ne 15) {
    throw "One or more frozen development runs failed; inspect runtime logs."
}

$FinalOut = Join-Path $Logs "finalize_openbmi.out"
$FinalErr = Join-Path $Logs "finalize_openbmi.err"
$Final = Start-Process -FilePath $Python `
    -ArgumentList @("code\finalize_openbmi.py") `
    -WorkingDirectory $Experiment `
    -RedirectStandardOutput $FinalOut `
    -RedirectStandardError $FinalErr `
    -WindowStyle Hidden `
    -Wait `
    -PassThru
if ($Final.ExitCode -ne 0) {
    throw "OpenBMI finalization failed; inspect $FinalErr"
}
Write-Output "OPENBMI_FINALIZATION_COMPLETE"
