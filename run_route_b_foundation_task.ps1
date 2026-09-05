$ErrorActionPreference = 'Continue'
$work = 'D:\nips-temp\TotalP\P1\CRCICLR_NESTED_OOF_V1'
$root = Join-Path $work 'experiments\persist_eeg_route_b_foundation_screen_v1'
$py = 'D:\nips-temp\TotalP\P2\.conda\gpu-baseline-v1\python.exe'
$out = Join-Path $root 'task.stdout.log'
$err = Join-Path $root 'task.stderr.log'
$exit = Join-Path $root 'task.exitcode'
Set-Location $work
New-Item -ItemType Directory -Force -Path $root | Out-Null
& $py -u 'experiments\persist_eeg_route_b_foundation_screen_v1\code\run_foundation_screen.py' --root $root --base-root 'D:\nips-temp\TotalP\P1\CRCICLR_GEOSR_FINAL_V1' --device cuda:0 1> $out 2> $err
$LASTEXITCODE | Set-Content -LiteralPath $exit -Encoding ascii
