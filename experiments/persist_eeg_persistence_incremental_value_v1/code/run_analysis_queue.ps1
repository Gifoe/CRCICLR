$ErrorActionPreference = 'Stop'
$env:PEEH_REPO = 'D:\nips-temp\TotalP\P1\CRCICLR_PERSIST_INCREMENTAL_VALUE_V1'
$env:PEEH_RUNTIME = 'D:\nips-temp\TotalP\P1\persist_incremental_value_runtime\seed0_replay'
$env:PSWA_RUNTIME = 'D:\nips-temp\TotalP\P1\persist_incremental_value_runtime\analysis'
$env:SEVEN_RUNTIME = 'D:\nips-temp\TotalP\P1\seven_backbone_fourtask_3seed_runtime'
$env:CUDA_VISIBLE_DEVICES = '0'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
$env:INCREMENTAL_DIRECT_ANALYSIS = '1'
$runtime = 'D:\nips-temp\TotalP\P1\persist_incremental_value_runtime'
$code = 'D:\nips-temp\TotalP\P1\CRCICLR_PERSIST_INCREMENTAL_VALUE_V1\experiments\persist_eeg_persistence_incremental_value_v1\code'
$python = 'E:\Anaconda\envs\persist_stable_251\python.exe'
$log = Join-Path $runtime 'analysis_queue.log'
New-Item -ItemType Directory -Force -Path $runtime | Out-Null

try {
    $amendment = Join-Path (Split-Path $code -Parent) 'protocol\DIRECT_ANALYSIS_AMENDMENT.json'
    if (-not (Test-Path -LiteralPath $amendment)) { throw 'direct-analysis amendment missing' }
    "DIRECT_ANALYSIS_AMENDMENT_SHA256 $((Get-FileHash -LiteralPath $amendment -Algorithm SHA256).Hash) $(Get-Date -Format o)" | Add-Content -LiteralPath $log

    $models = @('EEGNet', 'CBraMod', 'TeCh', 'ModernTCN', 'Medformer', 'EEGConformer', 'FBCNet')
    $tasks = @('OpenBMI_MI', 'OpenBMI_ERP', 'OpenBMI_SSVEP', 'WBCIC_MI')
    foreach ($model in $models) {
        foreach ($task in $tasks) {
            foreach ($fold in 0..4) {
                foreach ($seed in 0..2) {
                    & $python -u (Join-Path $code 'run_cell.py') $model $task $fold $seed *>> $log
                    $status = $LASTEXITCODE
                    if ($status -eq -1073741819) {
                        "NATIVE_RETRY $model $task fold=$fold seed=$seed" | Add-Content -LiteralPath $log
                        & $python -u (Join-Path $code 'run_cell.py') $model $task $fold $seed *>> $log
                        $status = $LASTEXITCODE
                    }
                    if ($status -ne 0) { throw "cell failure $model $task fold=$fold seed=$seed exit=$status" }
                }
            }
        }
    }
    & $python -u (Join-Path $code 'finalize.py') *>> $log
    if ($LASTEXITCODE -ne 0) { throw "finalization failed exit=$LASTEXITCODE" }
    [IO.File]::WriteAllText((Join-Path $runtime 'analysis_queue.exit'), '0')
}
catch {
    $message = ($_ | Out-String)
    $message | Add-Content -LiteralPath $log
    [IO.File]::WriteAllText((Join-Path $runtime 'analysis_queue.exit'), '1')
    throw
}
