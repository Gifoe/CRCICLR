$ErrorActionPreference = "Continue"
$experiment = "D:\nips-temp\TotalP\P1\CRCICLR_SOURCE_ONLY_DIAGNOSTIC\experiments\persist_eeg_scst_utility_stage1"
$python = "D:\nips-temp\TotalP\P2\.conda\gpu-baseline-v1\python.exe"
$log = Join-Path $experiment "runtime\source_training.log"
$exitFile = Join-Path $experiment "runtime\source_training.exit"
$stageFile = Join-Path $experiment "runtime\pipeline.stage"
New-Item -ItemType Directory -Force -Path (Join-Path $experiment "runtime") | Out-Null
"SOURCE_TRAINING" | Set-Content -LiteralPath $stageFile -Encoding ascii
Set-Location (Join-Path $experiment "code")
& $python "train_source_models.py" *> $log
$LASTEXITCODE | Set-Content -LiteralPath $exitFile -Encoding ascii
if ($LASTEXITCODE -eq 0) { "SOURCE_TRAINING_COMPLETE" | Set-Content -LiteralPath $stageFile -Encoding ascii } else { "SOURCE_TRAINING_FAILED" | Set-Content -LiteralPath $stageFile -Encoding ascii }
