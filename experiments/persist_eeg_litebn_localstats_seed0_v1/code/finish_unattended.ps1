# One-shot server-side handoff: never stop an existing training process.
$ErrorActionPreference = 'Stop'
$experimentRoot = 'D:\nips-temp\TotalP\P1\CRCICLR_BACKBONE_GEN_WORK\experiments\persist_eeg_litebn_localstats_seed0_v1'
$runnerPath = "$experimentRoot\code\run_localstats.py"
$completionPath = "$experimentRoot\outputs\FINAL_LOCALSTATS_SEED0_DECISION.md"
$handoffLog = 'D:\nips-temp\P1_logs\localstats_unattended.log'
"Handoff started $(Get-Date -Format o)" | Out-File $handoffLog -Encoding utf8 -Append
do {
    $activeRuns = @(Get-CimInstance Win32_Process -Filter "name = 'python.exe'" | Where-Object { $_.CommandLine -and $_.CommandLine.Contains($runnerPath) })
    if ($activeRuns.Count -gt 0) {
        foreach ($activeRun in $activeRuns) { Wait-Process -Id $activeRun.ProcessId -ErrorAction SilentlyContinue }
    }
} while ($activeRuns.Count -gt 0)
if (-not (Test-Path -LiteralPath $completionPath)) {
    'Original runner exited before completion; attempting one invariant-checked resume.' | Out-File $handoffLog -Encoding utf8 -Append
    & powershell.exe -NoProfile -File "$experimentRoot\code\run_on_server.ps1" >> 'D:\nips-temp\P1_logs\localstats_resume.log' 2>> 'D:\nips-temp\P1_logs\localstats_resume.err'
    if ($LASTEXITCODE -ne 0) { "Resume failed: $LASTEXITCODE" | Out-File $handoffLog -Encoding utf8 -Append; exit $LASTEXITCODE }
}
$verification = & 'E:\Anaconda\envs\persist_stable_251\python.exe' "$experimentRoot\code\verify_outputs.py" "$experimentRoot\outputs"
$verificationExit = $LASTEXITCODE
$verification | Out-File "$experimentRoot\outputs\UNATTENDED_VERIFICATION.json" -Encoding utf8
"Verification exit=$verificationExit $(Get-Date -Format o)" | Out-File $handoffLog -Encoding utf8 -Append
exit $verificationExit
