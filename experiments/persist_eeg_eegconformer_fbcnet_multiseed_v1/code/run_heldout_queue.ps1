$ErrorActionPreference = 'Stop'
$experiment = Split-Path -Parent $PSScriptRoot
$repository = Split-Path -Parent (Split-Path -Parent $experiment)
$env:NEW_BASELINE_RUNTIME = 'D:\nips-temp\TotalP\P1\eegconformer_fbcnet_runtime'
$env:NEW_BASELINE_SCRATCH = 'E:\eegconformer_fbcnet_scratch'
$env:NEW_BASELINE_MODEL = 'ALL'
$env:NEW_BASELINE_TASK = 'ALL'
$env:NEW_BASELINE_CPU_THREADS = '16'
$env:PYTHONPATH = $PSScriptRoot
$env:SEVEN_REPO = $repository
$env:SEVEN_RUNTIME = $env:NEW_BASELINE_RUNTIME
$env:MODERN_REPO = $repository
$env:TASK_GENERALITY_REPO = $repository
$env:FULL_OPENBMI_CACHE = 'D:\nips-temp\TotalP\P1\persist_eeg_stage0_repo_full\outputs\persist_eeg_stage0\cache\openbmi'
$env:PERSIST_OPENBMI_CACHE = $env:FULL_OPENBMI_CACHE
$env:FULL_WBCIC_CACHE = 'D:\nips-temp\TotalP\P1\CRCICLR_SOURCE_ONLY_DIAGNOSTIC\experiments\persist_eeg_wbcic_independent_replication_v1\runtime\cache\wbcic_epochs'
$lock = Join-Path $experiment 'protocol\HELDOUT_EVALUATION_LOCK.json'
$sidecar = Join-Path $experiment 'protocol\HELDOUT_EVALUATION_LOCK.sha256'
if (-not (Test-Path -LiteralPath $lock) -or -not (Test-Path -LiteralPath $sidecar)) {
    throw 'Pre-evaluation lock and SHA sidecar are required.'
}
$actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $lock).Hash.ToLowerInvariant()
$expected = (Get-Content -LiteralPath $sidecar -Raw).Trim()
if ($actual -ne $expected) { throw 'Pre-evaluation lock SHA mismatch.' }
$tracked = & git -C $repository ls-files --error-unmatch -- 'experiments/persist_eeg_eegconformer_fbcnet_multiseed_v1/protocol/HELDOUT_EVALUATION_LOCK.json' 2>$null
if ($LASTEXITCODE -ne 0 -or -not $tracked) { throw 'Pre-evaluation lock must be committed.' }
New-Item -ItemType Directory -Force -Path $env:NEW_BASELINE_RUNTIME,$env:NEW_BASELINE_SCRATCH | Out-Null
$log = Join-Path $env:NEW_BASELINE_RUNTIME 'heldout_queue.log'
Add-Content -LiteralPath $log -Value "QUEUE_START $((Get-Date).ToString('o'))" -Encoding UTF8
$ErrorActionPreference = 'Continue'
& 'E:\Anaconda\envs\persist_stable_251\python.exe' -u (Join-Path $PSScriptRoot 'heldout_eval.py') --stage run *>> $log
$code = $LASTEXITCODE
Add-Content -LiteralPath $log -Value "QUEUE_EXIT $code $((Get-Date).ToString('o'))" -Encoding UTF8
Set-Content -LiteralPath (Join-Path $env:NEW_BASELINE_RUNTIME 'heldout_queue_exit.txt') -Value "$code"
exit $code
