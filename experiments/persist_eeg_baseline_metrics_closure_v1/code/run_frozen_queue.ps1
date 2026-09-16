$ErrorActionPreference = 'Stop'
$python = 'E:\Anaconda\envs\persist_stable_251\python.exe'
$runner = 'D:\nips-temp\TotalP\P1\CRCICLR_BASELINE_METRICS_CLOSURE_V1\experiments\persist_eeg_baseline_metrics_closure_v1\code\run_frozen_sessions.py'
$runtime = 'D:\nips-temp\TotalP\P1\baseline_metrics_closure_v1_runtime'
$log = Join-Path $runtime 'frozen_inference.log'
$env:CLOSURE_CPU_THREADS = '4'
New-Item -ItemType Directory -Force -Path $runtime | Out-Null
try {
    foreach ($model in @('ModernTCN', 'Medformer', 'EEGNet', 'CBraMod', 'TeCh', 'LiteBN')) {
        & $python -u $runner --stage run --model $model 2>&1 | Tee-Object -FilePath $log -Append
        if ($LASTEXITCODE -ne 0) { throw "Frozen inference failed: $model exit=$LASTEXITCODE" }
    }
    [IO.File]::WriteAllText((Join-Path $runtime 'INFERENCE_COMPLETED.txt'), [DateTime]::UtcNow.ToString('o') + [Environment]::NewLine)
}
catch {
    [IO.File]::WriteAllText((Join-Path $runtime 'INFERENCE_FAILED.txt'), ($_ | Out-String))
    throw
}
