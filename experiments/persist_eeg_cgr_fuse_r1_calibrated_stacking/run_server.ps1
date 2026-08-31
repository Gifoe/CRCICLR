$ErrorActionPreference = 'Continue'
$exp = 'D:\nips-temp\TotalP\P1\CRCICLR_SOURCE_ONLY_DIAGNOSTIC\experiments\persist_eeg_cgr_fuse_r1_calibrated_stacking'
$py = 'D:\nips-temp\TotalP\P2\.conda\gpu-baseline-v1\python.exe'
Set-Location -LiteralPath $exp
& $py -u 'code\cgrstack.py' --phase all *> (Join-Path $exp 'run.log')
$rc = $LASTEXITCODE
Set-Content -LiteralPath (Join-Path $exp 'run.exit') -Value $rc
exit $rc
