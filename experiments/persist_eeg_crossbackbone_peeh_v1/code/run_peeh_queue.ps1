$ErrorActionPreference = 'Continue'
$python = 'E:\Anaconda\envs\persist_stable_251\python.exe'
$code = 'D:\nips-temp\TotalP\P1\CRCICLR_CROSSBACKBONE_PEEH_WORK\experiments\persist_eeg_crossbackbone_peeh_v1\code'
$runtime = 'D:\nips-temp\TotalP\P1\crossbackbone_peeh_runtime'
$env:SEVEN_REPO = 'D:\nips-temp\TotalP\P1\CRCICLR_BACKBONE_GEN_WORK'
$env:SEVEN_RUNTIME = 'D:\nips-temp\TotalP\P1\seven_backbone_fourtask_3seed_runtime'
$env:OFFICIAL_BACKBONE_ROOT = 'D:\nips-temp\TotalP\P1\seven_backbone_fourtask_3seed_runtime\official'
$env:FULL_OPENBMI_CACHE = 'D:\nips-temp\TotalP\P1\persist_eeg_stage0_repo_full\outputs\persist_eeg_stage0\cache\openbmi'
$env:FULL_WBCIC_CACHE = 'D:\nips-temp\TotalP\P1\CRCICLR_SOURCE_ONLY_DIAGNOSTIC\experiments\persist_eeg_wbcic_independent_replication_v1\runtime\cache\wbcic_epochs'
$env:TRUE_OUTER_WBCIC_CACHE = 'D:\nips-temp\TotalP\P2\wbcic_outer_cache\wbcic_epochs'
$env:PEEH_RUNTIME = $runtime
$env:OMP_NUM_THREADS = '8'
$env:MKL_NUM_THREADS = '8'
$env:OPENBLAS_NUM_THREADS = '8'
$logs = Join-Path $runtime 'logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null
Set-Location -LiteralPath $code

function Run-Model([string]$model) {
    $slug = $model.ToLower()
    $out = Join-Path $logs ($slug + '.out.log')
    $err = Join-Path $logs ($slug + '.err.log')
    & $python -u run_crossbackbone_peeh.py --stage run --model $model 1>> $out 2>> $err
    if ($LASTEXITCODE -ne 0) { throw "$model failed: $LASTEXITCODE" }
}

# The three compact frozen backbones run concurrently.  Each process resumes
# at cell granularity.  High-dimensional ModernTCN/Medformer run serially to
# avoid multi-GB representation overlap.
$jobs = @()
foreach ($model in @('EEGNet','CBraMod','TeCh')) {
    $jobs += Start-Job -Name ("PEEH_" + $model) -ScriptBlock ${function:Run-Model} -ArgumentList $model -InitializationScript {
        $python = 'E:\Anaconda\envs\persist_stable_251\python.exe'
        $code = 'D:\nips-temp\TotalP\P1\CRCICLR_CROSSBACKBONE_PEEH_WORK\experiments\persist_eeg_crossbackbone_peeh_v1\code'
        $runtime = 'D:\nips-temp\TotalP\P1\crossbackbone_peeh_runtime'
        $logs = Join-Path $runtime 'logs'
        $env:SEVEN_REPO = 'D:\nips-temp\TotalP\P1\CRCICLR_BACKBONE_GEN_WORK'
        $env:SEVEN_RUNTIME = 'D:\nips-temp\TotalP\P1\seven_backbone_fourtask_3seed_runtime'
        $env:OFFICIAL_BACKBONE_ROOT = 'D:\nips-temp\TotalP\P1\seven_backbone_fourtask_3seed_runtime\official'
        $env:FULL_OPENBMI_CACHE = 'D:\nips-temp\TotalP\P1\persist_eeg_stage0_repo_full\outputs\persist_eeg_stage0\cache\openbmi'
        $env:FULL_WBCIC_CACHE = 'D:\nips-temp\TotalP\P1\CRCICLR_SOURCE_ONLY_DIAGNOSTIC\experiments\persist_eeg_wbcic_independent_replication_v1\runtime\cache\wbcic_epochs'
        $env:TRUE_OUTER_WBCIC_CACHE = 'D:\nips-temp\TotalP\P2\wbcic_outer_cache\wbcic_epochs'
        $env:PEEH_RUNTIME = $runtime
        $env:OMP_NUM_THREADS = '8'; $env:MKL_NUM_THREADS = '8'; $env:OPENBLAS_NUM_THREADS = '8'
        Set-Location -LiteralPath $code
    }
}
$jobs | Wait-Job | Out-Null
$failed = $jobs | Where-Object State -ne 'Completed'
$jobs | Receive-Job | Add-Content -LiteralPath (Join-Path $logs 'job_control.log')
$jobs | Remove-Job
if ($failed) {
    # Preserve the failed cell and continue independent models.  A single
    # protocol-invalid cell must not prevent unrelated frozen analyses from
    # running; final aggregation still remains gated on complete inputs.
    Add-Content -LiteralPath (Join-Path $logs 'job_control.log') -Value "compact backbone queue had failures; continuing independent models"
}

Run-Model 'ModernTCN'
Run-Model 'Medformer'
$cellCount = @(Get-ChildItem -LiteralPath (Join-Path $runtime 'cells') -Recurse -Filter '*.json' -File).Count
if ($cellCount -eq 100) {
    & $python -u run_crossbackbone_peeh.py --stage aggregate 1>> (Join-Path $logs 'aggregate.out.log') 2>> (Join-Path $logs 'aggregate.err.log')
    if ($LASTEXITCODE -ne 0) { throw "aggregate failed: $LASTEXITCODE" }
    Set-Content -LiteralPath (Join-Path $runtime 'RUN_COMPLETE.txt') -Value (Get-Date -Format o)
} else {
    Add-Content -LiteralPath (Join-Path $logs 'job_control.log') -Value "aggregation deferred: $cellCount/100 cells"
}
