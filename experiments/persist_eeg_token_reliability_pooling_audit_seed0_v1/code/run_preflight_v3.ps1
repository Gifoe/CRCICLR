$ErrorActionPreference = 'Stop'
$root = 'D:\nips-temp\TotalP\P1\CRCICLR_SEVEN_BACKBONE_HELDOUT_WORK'
$experiment = Join-Path $root 'experiments\persist_eeg_token_reliability_pooling_audit_seed0_v1'
$entry = Join-Path $experiment 'code\run_v2.py'
$expected = 'D82985C3B40E6D81417019C4A2795C88105EAF884402A5D1101EE7D55F7FE476'
$runtime = 'D:\nips-temp\TotalP\P1\token_reliability_pooling_runtime'
$log = Join-Path $runtime 'preflight_fold0_v3.log'
if (Test-Path -LiteralPath $log) { throw 'preflight v3 log exists' }
New-Item -ItemType Directory -Path $runtime -Force | Out-Null
Start-Transcript -Path $log -NoClobber | Out-Null
try {
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $entry).Hash -ne $expected) { throw 'preflight entry hash mismatch' }
    $env:TOKEN_AUDIT_RUNTIME = $runtime
    $env:SEVEN_RUNTIME = 'D:\nips-temp\TotalP\P1\seven_backbone_fourtask_3seed_runtime'
    $env:FULL_OPENBMI_CACHE = 'D:\nips-temp\TotalP\P1\persist_eeg_stage0_repo_full\outputs\persist_eeg_stage0\cache\openbmi'
    $env:OFFICIAL_BACKBONE_ROOT = 'D:\nips-temp\TotalP\P1\seven_backbone_fourtask_3seed_runtime\official'
    $env:MODERN_REPO = $root
    $py = 'E:\Anaconda\envs\persist_stable_251\python.exe'
    $ErrorActionPreference = 'Continue'
    & $py $entry --fold 0 --preflight 2>&1 | ForEach-Object { Write-Host $_ }
    $result = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    Write-Host "PYTHON_EXIT=$result"
    if ($result -ne 0) { throw "preflight failed with exit $result" }
} catch {
    Write-Host "PREFLIGHT_EXCEPTION=$($_.Exception.Message)"
    exit 1
} finally {
    Stop-Transcript | Out-Null
}
