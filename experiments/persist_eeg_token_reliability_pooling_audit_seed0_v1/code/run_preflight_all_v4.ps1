$ErrorActionPreference = 'Stop'
$root = 'D:\nips-temp\TotalP\P1\CRCICLR_SEVEN_BACKBONE_HELDOUT_WORK'
$entry = Join-Path $root 'experiments\persist_eeg_token_reliability_pooling_audit_seed0_v1\code\run_v3.py'
$expected = '73E38D37F4CE0BD38AE9FAAEF776DA72D722F400B9D6EBA604BB63A3961CB386'
$runtime = 'D:\nips-temp\TotalP\P1\token_reliability_pooling_runtime'
$log = Join-Path $runtime 'preflight_all_v4.log'
if (Test-Path -LiteralPath $log) { throw 'preflight all-fold log exists' }
New-Item -ItemType Directory -Path $runtime -Force | Out-Null
Start-Transcript -Path $log -NoClobber | Out-Null
try {
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $entry).Hash -ne $expected) { throw 'entry hash mismatch' }
    $env:TOKEN_AUDIT_RUNTIME = $runtime
    $env:SEVEN_RUNTIME = 'D:\nips-temp\TotalP\P1\seven_backbone_fourtask_3seed_runtime'
    $env:FULL_OPENBMI_CACHE = 'D:\nips-temp\TotalP\P1\persist_eeg_stage0_repo_full\outputs\persist_eeg_stage0\cache\openbmi'
    $env:OFFICIAL_BACKBONE_ROOT = 'D:\nips-temp\TotalP\P1\seven_backbone_fourtask_3seed_runtime\official'
    $env:MODERN_REPO = $root
    $py = 'E:\Anaconda\envs\persist_stable_251\python.exe'
    foreach ($fold in 0..4) {
        $ErrorActionPreference = 'Continue'
        & $py $entry --fold $fold --preflight 2>&1 | ForEach-Object { Write-Host $_ }
        $result = $LASTEXITCODE
        $ErrorActionPreference = 'Stop'
        if ($result -ne 0) { throw "preflight fold $fold failed with exit $result" }
    }
    Write-Host 'ALL_FIVE_PREFLIGHTS_COMPLETE'
} catch {
    Write-Host "PREFLIGHT_EXCEPTION=$($_.Exception.Message)"
    exit 1
} finally {
    Stop-Transcript | Out-Null
}
