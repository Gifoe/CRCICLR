$ErrorActionPreference = 'Continue'
$work = 'D:\nips-temp\TotalP\P1\CRCICLR_NESTED_OOF_V1'
$root = Join-Path $work 'experiments\persist_eeg_nested_oof_error_audit_v1'
$py = 'D:\nips-temp\TotalP\P2\.conda\gpu-baseline-v1\python.exe'
$out = Join-Path $root 'task.stdout.log'
$err = Join-Path $root 'task.stderr.log'
$exit = Join-Path $root 'task.exitcode'
Set-Location $work
$env:PERSIST_CUDNN_BENCHMARK = '1'
& $py -u 'experiments\persist_eeg_nested_oof_error_audit_v1\code\run_nested_oof_audit.py' --root 'experiments\persist_eeg_nested_oof_error_audit_v1' --base-root 'D:\nips-temp\TotalP\P1\CRCICLR_GEOSR_FINAL_V1\experiments\persist_eeg_geosr_final_v1' --device cuda:0 1> $out 2> $err
$LASTEXITCODE | Set-Content -LiteralPath $exit -Encoding ascii
