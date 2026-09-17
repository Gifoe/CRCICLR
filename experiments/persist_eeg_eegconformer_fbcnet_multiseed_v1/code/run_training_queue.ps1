param(
    [ValidateSet('ALL', 'EEGConformer', 'FBCNet')]
    [string]$Model = 'ALL',
    [ValidateSet('ALL', 'OpenBMI_MI', 'OpenBMI_ERP', 'OpenBMI_SSVEP', 'WBCIC_MI')]
    [string]$Task = 'ALL'
)
$ErrorActionPreference = 'Stop'
$experiment = Split-Path -Parent $PSScriptRoot
$repository = Split-Path -Parent (Split-Path -Parent $experiment)
$env:NEW_BASELINE_RUNTIME = 'D:\nips-temp\TotalP\P1\eegconformer_fbcnet_runtime'
$env:NEW_BASELINE_SCRATCH = 'E:\eegconformer_fbcnet_scratch'
$env:NEW_BASELINE_MODEL = $Model
$env:NEW_BASELINE_TASK = $Task
$env:NEW_BASELINE_CPU_THREADS = '16'
$env:PYTHONPATH = $PSScriptRoot
$env:SEVEN_REPO = $repository
$env:SEVEN_RUNTIME = $env:NEW_BASELINE_RUNTIME
$env:MODERN_REPO = $repository
$env:TASK_GENERALITY_REPO = $repository
$env:FULL_OPENBMI_CACHE = 'D:\nips-temp\TotalP\P1\persist_eeg_stage0_repo_full\outputs\persist_eeg_stage0\cache\openbmi'
$env:PERSIST_OPENBMI_CACHE = $env:FULL_OPENBMI_CACHE
$env:FULL_WBCIC_CACHE = 'D:\nips-temp\TotalP\P1\CRCICLR_SOURCE_ONLY_DIAGNOSTIC\experiments\persist_eeg_wbcic_independent_replication_v1\runtime\cache\wbcic_epochs'
New-Item -ItemType Directory -Force -Path $env:NEW_BASELINE_RUNTIME,$env:NEW_BASELINE_SCRATCH | Out-Null
$shardName = if ($Task -eq 'ALL') { $Model } else { "${Model}_${Task}" }
$log = Join-Path $env:NEW_BASELINE_RUNTIME "train_queue_$shardName.log"
$stamp = (Get-Date).ToString('o')
Add-Content -LiteralPath $log -Value "QUEUE_START $stamp" -Encoding UTF8
$ErrorActionPreference = 'Continue'
& 'E:\Anaconda\envs\persist_stable_251\python.exe' -u (Join-Path $PSScriptRoot 'train_queue.py') *>> $log
$code = $LASTEXITCODE
Add-Content -LiteralPath $log -Value "QUEUE_EXIT $code $((Get-Date).ToString('o'))" -Encoding UTF8
Set-Content -LiteralPath (Join-Path $env:NEW_BASELINE_RUNTIME "train_queue_${shardName}_exit.txt") -Value "$code"
exit $code
