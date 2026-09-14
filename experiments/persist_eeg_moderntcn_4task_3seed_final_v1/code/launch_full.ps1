param([int]$Workers = 1)
$ErrorActionPreference = 'Stop'
$python = 'E:/Anaconda/envs/persist_stable_251/python.exe'
$experiment = Split-Path -Parent $PSScriptRoot
$repository = (Resolve-Path (Join-Path $experiment '..\..')).Path
$env:BASELINE_RUNTIME = 'D:/nips-temp/TotalP/P1/baseline3_runtime/moderntcn'
$env:SEVEN_RUNTIME = 'D:/nips-temp/TotalP/P1/seven_backbone_fourtask_3seed_runtime'
$env:SEVEN_REPO = 'D:/nips-temp/TotalP/P1/CRCICLR_BACKBONE_GEN_WORK'
$env:MODERN_REPO = $env:SEVEN_REPO
$env:TASK_GENERALITY_REPO = $env:SEVEN_REPO
$env:FULL_OPENBMI_CACHE = 'D:/nips-temp/TotalP/P1/persist_eeg_stage0_repo_full/outputs/persist_eeg_stage0/cache/openbmi'
$env:FULL_WBCIC_CACHE = 'D:/nips-temp/TotalP/P1/CRCICLR_SOURCE_ONLY_DIAGNOSTIC/experiments/persist_eeg_wbcic_independent_replication_v1/runtime/cache/wbcic_epochs'
$logs = Join-Path $env:BASELINE_RUNTIME 'logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null
Set-Location $PSScriptRoot

& $python -u run_baseline.py --preflight *>> (Join-Path $logs 'preflight.log')
if ($LASTEXITCODE -ne 0) { throw "preflight failed: $LASTEXITCODE" }

$processes = @()
for ($worker = 0; $worker -lt $Workers; $worker++) {
    $out = Join-Path $logs ("worker_{0}.out.log" -f $worker)
    $err = Join-Path $logs ("worker_{0}.err.log" -f $worker)
    $processes += Start-Process -FilePath $python -ArgumentList @('-u','run_baseline.py','--worker-index',"$worker",'--worker-count',"$Workers") -WorkingDirectory $PSScriptRoot -RedirectStandardOutput $out -RedirectStandardError $err -WindowStyle Hidden -PassThru
}
$failed = @()
foreach ($process in $processes) {
    $process.WaitForExit()
    if ($process.ExitCode -ne 0) { $failed += $process.ExitCode }
}
if ($failed.Count -gt 0) { throw "queue workers failed: $($failed -join ',')" }

& $python -u run_baseline.py --aggregate *>> (Join-Path $logs 'aggregate.log')
if ($LASTEXITCODE -ne 0) { throw "aggregate failed: $LASTEXITCODE" }
$lock = Join-Path $experiment 'protocol\HELDOUT_EVALUATION_LOCK.json'
if (!(Test-Path $lock)) {
    & $python -u evaluate_heldout.py --stage prelock *>> (Join-Path $logs 'heldout.log')
    if ($LASTEXITCODE -ne 0) { throw "heldout prelock failed: $LASTEXITCODE" }
}
& $python -u evaluate_heldout.py --stage evaluate *>> (Join-Path $logs 'heldout.log')
if ($LASTEXITCODE -ne 0) { throw "heldout evaluation failed: $LASTEXITCODE" }
Set-Content -LiteralPath (Join-Path $env:BASELINE_RUNTIME 'RUN_COMPLETE.txt') -Value (Get-Date -Format o)
