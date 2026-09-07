$ErrorActionPreference = "Continue"
$experiment = "D:\nips-temp\TotalP\P1\CRCICLR_SOURCE_ONLY_DIAGNOSTIC\experiments\persist_eeg_scst_utility_stage1"
$python = "D:\nips-temp\TotalP\P2\.conda\gpu-baseline-v1\python.exe"
$log = Join-Path $experiment "runtime\source_eegnex.log"
$exitFile = Join-Path $experiment "runtime\source_eegnex.exit"
Set-Location (Join-Path $experiment "code")
& $python "train_source_models.py" --models EEGNeX *> $log
$LASTEXITCODE | Set-Content -LiteralPath $exitFile -Encoding ascii
