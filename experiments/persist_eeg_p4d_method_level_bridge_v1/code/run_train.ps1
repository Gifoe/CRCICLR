$ErrorActionPreference = 'Stop'
$repo = 'D:\nips-temp\TotalP\P1\CRCICLR_SOURCE_ONLY_DIAGNOSTIC'
$experiment = Join-Path $repo 'experiments\persist_eeg_p4d_method_level_bridge_v1'
$logDirectory = Join-Path $experiment 'runtime\logs'
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
Set-Location -LiteralPath $repo
$stdout = Join-Path $logDirectory 'train_stdout.log'
$stderr = Join-Path $logDirectory 'train_stderr.log'
& 'D:\nips-temp\TotalP\P2\.conda\gpu-baseline-v1\python.exe' 'experiments\persist_eeg_p4d_method_level_bridge_v1\code\train_canonical.py' 1> $stdout 2> $stderr
$exitCode = $LASTEXITCODE
@{
    exit_code = $exitCode
    finished_at = (Get-Date).ToUniversalTime().ToString('o')
} | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $logDirectory 'train_exit.json')
exit $exitCode
