$ErrorActionPreference = 'Stop'
$repo = 'D:\nips-temp\TotalP\P1\CRCICLR_SOURCE_ONLY_DIAGNOSTIC'
$experiment = Join-Path $repo 'experiments\persist_eeg_p4d_method_level_bridge_v1'
$results = Join-Path $experiment 'results'
$logDirectory = Join-Path $experiment 'runtime\logs'
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
$stdout = Join-Path $logDirectory 'post_stdout.log'
$stderr = Join-Path $logDirectory 'post_stderr.log'
$completion = Join-Path $results 'P4D_S6_CANONICAL_TRAINING_COMPLETE.json'
$trainingExit = Join-Path $logDirectory 'train_exit.json'
$deadline = (Get-Date).AddHours(12)
while (-not (Test-Path -LiteralPath $completion)) {
    if (Test-Path -LiteralPath $trainingExit) {
        $training = Get-Content -Raw -LiteralPath $trainingExit | ConvertFrom-Json
        if ([int]$training.exit_code -ne 0) {
            throw "canonical training failed with exit code $($training.exit_code)"
        }
    }
    if ((Get-Date) -gt $deadline) {
        throw 'timed out waiting for canonical S6 training'
    }
    Start-Sleep -Seconds 30
}
Set-Location -LiteralPath $repo
$python = 'D:\nips-temp\TotalP\P2\.conda\gpu-baseline-v1\python.exe'
$code = 'experiments\persist_eeg_p4d_method_level_bridge_v1\code'
& $python "$code\freeze_preoutcome.py" 1> $stdout 2> $stderr
if ($LASTEXITCODE -ne 0) { throw "freeze_preoutcome failed: $LASTEXITCODE" }
& $python "$code\evaluate_canonical.py" 1>> $stdout 2>> $stderr
if ($LASTEXITCODE -ne 0) { throw "evaluate_canonical failed: $LASTEXITCODE" }
& $python "$code\analyze_p4d.py" 1>> $stdout 2>> $stderr
if ($LASTEXITCODE -ne 0) { throw "analyze_p4d failed: $LASTEXITCODE" }
& $python "$code\validate_p4d.py" 1>> $stdout 2>> $stderr
if ($LASTEXITCODE -ne 0) { throw "core validate_p4d failed: $LASTEXITCODE" }
& $python "$code\finalize_p4d.py" 1>> $stdout 2>> $stderr
if ($LASTEXITCODE -ne 0) { throw "finalize_p4d failed: $LASTEXITCODE" }
& $python "$code\validate_p4d.py" '--require-final-report' 1>> $stdout 2>> $stderr
$exitCode = $LASTEXITCODE
@{
    exit_code = $exitCode
    finished_at = (Get-Date).ToUniversalTime().ToString('o')
} | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $logDirectory 'post_exit.json')
exit $exitCode
