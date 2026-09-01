$ErrorActionPreference = "Stop"
$exp = "D:\nips-temp\TotalP\P1\CRCICLR_SOURCE_ONLY_DIAGNOSTIC\experiments\persist_eeg_cgr_stack_v2_competence_robust"
$python = "D:\nips-temp\TotalP\P2\.conda\gpu-baseline-v1\python.exe"
$log = Join-Path $exp "runtime\cpv2.log"
$exitFile = Join-Path $exp "runtime\cpv2.exit"
New-Item -ItemType Directory -Force -Path (Join-Path $exp "runtime") | Out-Null
Set-Location -LiteralPath $exp
& $python -u "code\cpv2.py" --phase all *> $log
$rc = $LASTEXITCODE
$rc | Set-Content -LiteralPath $exitFile -Encoding ascii
exit $rc
