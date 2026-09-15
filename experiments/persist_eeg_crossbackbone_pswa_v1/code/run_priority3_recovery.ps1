$ErrorActionPreference = 'Stop'
$python = 'E:/Anaconda/envs/persist_stable_251/python.exe'
$code = 'D:/nips-temp/TotalP/P1/CRCICLR_CROSSBACKBONE_PEEH_WORK/experiments/persist_eeg_crossbackbone_pswa_v1/code'
$runtime = 'D:/nips-temp/TotalP/P1/crossbackbone_pswa_runtime'
$env:PSWA_RUNTIME = $runtime
$env:PEEH_REPO = 'D:/nips-temp/TotalP/P1/CRCICLR_CROSSBACKBONE_PEEH_WORK'
$env:PEEH_RUNTIME = 'D:/nips-temp/TotalP/P1/crossbackbone_peeh_runtime'
$env:SEVEN_REPO = 'D:/nips-temp/TotalP/P1/CRCICLR_BACKBONE_GEN_WORK'
$env:SEVEN_RUNTIME = 'D:/nips-temp/TotalP/P1/seven_backbone_fourtask_3seed_runtime'
$env:FULL_OPENBMI_CACHE = 'D:/nips-temp/TotalP/P1/persist_eeg_stage0_repo_full/outputs/persist_eeg_stage0/cache/openbmi'
$env:FULL_WBCIC_CACHE = 'D:/nips-temp/TotalP/P1/CRCICLR_SOURCE_ONLY_DIAGNOSTIC/experiments/persist_eeg_wbcic_independent_replication_v1/runtime/cache/wbcic_epochs'
$env:TRUE_OUTER_WBCIC_CACHE = 'D:/nips-temp/TotalP/P2/wbcic_outer_cache/wbcic_epochs'
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$logDir = Join-Path $runtime 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
Set-Location $code

foreach ($model in @('EEGNet', 'CBraMod', 'TeCh')) {
    $log = Join-Path $logDir ("{0}.out.log" -f $model.ToLowerInvariant())
    $err = Join-Path $logDir ("{0}.err.log" -f $model.ToLowerInvariant())
    foreach ($task in @('OpenBMI_MI', 'OpenBMI_ERP', 'OpenBMI_SSVEP', 'WBCIC_MI')) {
        foreach ($fold in 0..4) {
            $cell = Join-Path $runtime ("cells/{0}/{1}/fold{2}_seed0.json" -f $model.ToLowerInvariant(), $task.ToLowerInvariant(), $fold)
            if (Test-Path -LiteralPath $cell) {
                Add-Content -LiteralPath $log -Value ("SKIP_COMPLETE {0} {1} fold{2}" -f $model, $task, $fold)
                continue
            }
            $ok = $false
            foreach ($attempt in 1..3) {
                $savedPreference = $ErrorActionPreference
                $ErrorActionPreference = 'Continue'
                & $python -u run_pswa_recovery.py --stage run --model $model --task $task --fold $fold 1>> $log 2>> $err
                $cellExitCode = $LASTEXITCODE
                $ErrorActionPreference = $savedPreference
                if ($cellExitCode -eq 0 -and (Test-Path -LiteralPath $cell)) { $ok = $true; break }
                Add-Content -LiteralPath $err -Value ("RETRY {0} {1} fold{2} attempt={3} exit={4}" -f $model, $task, $fold, $attempt, $cellExitCode)
                Start-Sleep -Seconds 3
            }
            if (!$ok) { throw "PSWA recovery failed for $model/$task/fold$fold after 3 isolated attempts" }
        }
    }
    $savedPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $python -u run_pswa_recovery.py --stage aggregate 1>> (Join-Path $logDir 'aggregate.log') 2>> (Join-Path $logDir 'aggregate.err.log')
    $aggregateExitCode = $LASTEXITCODE
    $ErrorActionPreference = $savedPreference
    if ($aggregateExitCode -ne 0) { throw "PSWA aggregate failed after $model, exit=$aggregateExitCode" }
    Set-Content -LiteralPath (Join-Path $runtime ("{0}_COMPLETE.txt" -f $model.ToUpperInvariant())) -Value (Get-Date -Format o)
}
Set-Content -LiteralPath (Join-Path $runtime 'PRIORITY3_COMPLETE.txt') -Value (Get-Date -Format o)
