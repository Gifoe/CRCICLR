$ErrorActionPreference = 'Stop'
$root = 'D:\nips-temp\TotalP\P1\CRCICLR_SEVEN_BACKBONE_HELDOUT_WORK'
$code = Join-Path $root 'experiments\persist_eeg_token_reliability_pooling_audit_seed0_v1\code'
$entry = Join-Path $code 'finalize_v2.py'
$runtime = 'D:\nips-temp\TotalP\P1\token_reliability_pooling_runtime'
$log = Join-Path $runtime 'final_compact_queue_v2.log'
if (Test-Path -LiteralPath $log) { throw 'final compact V2 queue log already exists' }
if (Test-Path -LiteralPath (Join-Path $runtime 'final_compact_v2')) { throw 'final compact V2 output already exists' }
Start-Transcript -Path $log -NoClobber | Out-Null
try {
    $pins = @{
        'finalize_v2.py' = '19DE0BD914D9378D161F4CC974726629F8D157873E53F56EFF606FA342FE8C6A'
        'aggregate.py' = '0F789C14716A9B6B8BC9DB5C151C5EA7F4C566AAE28B78582C25EE58740A9089'
    }
    foreach ($name in $pins.Keys) {
        if ((Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $code $name)).Hash -ne $pins[$name]) {
            throw "final compact V2 code SHA mismatch: $name"
        }
    }
    $deadline = (Get-Date).AddHours(238)
    while ($true) {
        if ((Get-Date) -gt $deadline) { throw 'OUTER V2 queue wait timeout' }
        $task = Get-ScheduledTask -TaskName 'PERSIST_EEG_TOKEN_RELIABILITY_OUTER_QUEUE_V2'
        $info = Get-ScheduledTaskInfo -TaskName 'PERSIST_EEG_TOKEN_RELIABILITY_OUTER_QUEUE_V2'
        if ($task.State -eq 'Ready') {
            if ($info.LastTaskResult -ne 0) { throw 'OUTER V2 queue failed; refusing final compact aggregation' }
            break
        }
        Start-Sleep -Seconds 60
    }
    foreach ($fold in 0..4) {
        $seal = Join-Path $runtime ("fold$fold\outer_evaluation_v2\OUTER_EVALUATION_SEAL.json")
        if (!(Test-Path -LiteralPath $seal)) { throw "fold$fold OUTER V2 seal missing" }
    }
    $py = 'E:\Anaconda\envs\persist_stable_251\python.exe'
    $ErrorActionPreference = 'Continue'
    & $py $entry --runtime $runtime 2>&1 | ForEach-Object { Write-Host $_ }
    $result = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($result -ne 0) { throw "final compact V2 aggregation failed with exit $result" }
    Write-Host 'FINAL_COMPACT_V2_AGGREGATION_COMPLETE_REVIEW_REQUIRED'
} catch {
    Write-Host "FINAL_COMPACT_V2_QUEUE_FAILURE=$($_.Exception.Message)"
    exit 1
} finally {
    Stop-Transcript | Out-Null
}
