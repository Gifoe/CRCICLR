$ErrorActionPreference = 'Continue'
$python = 'E:\Anaconda\envs\persist_stable_251\python.exe'
$exp = 'D:\nips-temp\TotalP\P1\CRCICLR_BASELINE_METRICS_CLOSURE_V1\experiments\persist_eeg_baseline_metrics_closure_v1'
$runner = Join-Path $exp 'code\published_run_pswa_recovery.py'
$lock = Join-Path $exp 'protocol\PSWA_EXTENSION_LOCK.json'
$sidecar = Join-Path $exp 'protocol\PSWA_EXTENSION_LOCK.sha256'
$runtime = 'D:\nips-temp\TotalP\P1\baseline_metrics_closure_v1_pswa_runtime'
$log = Join-Path $runtime 'pswa_extension.log'
$env:PSWA_RUNTIME = $runtime
$env:SEVEN_RUNTIME = 'D:\nips-temp\TotalP\P1\seven_backbone_fourtask_3seed_runtime'
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
New-Item -ItemType Directory -Force -Path $runtime | Out-Null
try {
    if (-not (Test-Path -LiteralPath $lock) -or -not (Test-Path -LiteralPath $sidecar)) {
        throw 'PSWA extension lock missing'
    }
    $recorded = (Get-Content -LiteralPath $sidecar -Raw).Trim()
    if ((Get-FileHash -LiteralPath $lock -Algorithm SHA256).Hash.ToLowerInvariant() -ne $recorded.ToLowerInvariant()) {
        throw 'PSWA extension lock hash mismatch'
    }
    foreach ($model in @('ModernTCN', 'Medformer')) {
        & $python -u $runner --stage run --model $model 2>&1 | Tee-Object -FilePath $log -Append
        if ($LASTEXITCODE -ne 0) { throw "PSWA frozen run failed: $model exit=$LASTEXITCODE" }
    }
    & $python -u $runner --stage aggregate 2>&1 | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) { throw "PSWA aggregate failed: exit=$LASTEXITCODE" }
    [IO.File]::WriteAllText((Join-Path $runtime 'PSWA_COMPLETED.txt'), [DateTime]::UtcNow.ToString('o') + [Environment]::NewLine)
}
catch {
    [IO.File]::WriteAllText((Join-Path $runtime 'PSWA_FAILED.txt'), ($_ | Out-String))
    throw
}
