$ErrorActionPreference = 'Stop'
$root = 'D:\nips-temp\TotalP\P1\CRCICLR_SEVEN_BACKBONE_HELDOUT_WORK'
$entry = Join-Path $root 'experiments\persist_eeg_token_reliability_pooling_audit_seed0_v1\code\summarize_train_evidence_v1.py'
$runtime = 'D:\nips-temp\TotalP\P1\token_reliability_pooling_runtime'
$log = Join-Path $runtime 'train_evidence_queue_v1.log'
if (Test-Path -LiteralPath $log) { throw 'TRAIN evidence queue log already exists' }
if (Test-Path -LiteralPath (Join-Path $runtime 'train_evidence_v1')) { throw 'TRAIN evidence output already exists' }
Start-Transcript -Path $log -NoClobber | Out-Null
try {
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $entry).Hash -ne 'B758F51EA57BAF3FE14FE3D41B4C48DD1279B643B4F45FDD8FCF6D6B685E7BF4') {
        throw 'TRAIN evidence source SHA mismatch'
    }
    $deadline = (Get-Date).AddHours(68)
    while ($true) {
        if ((Get-Date) -gt $deadline) { throw 'TRAIN utility queue wait timeout' }
        $task = Get-ScheduledTask -TaskName 'PERSIST_EEG_TOKEN_RELIABILITY_UTILITY_QUEUE_V1'
        $info = Get-ScheduledTaskInfo -TaskName 'PERSIST_EEG_TOKEN_RELIABILITY_UTILITY_QUEUE_V1'
        if ($task.State -eq 'Ready') {
            if ($info.LastTaskResult -ne 0) { throw 'TRAIN utility queue failed; refusing aggregation' }
            break
        }
        Start-Sleep -Seconds 60
    }
    foreach ($fold in 0..4) {
        $seal = Join-Path $runtime ("fold$fold\utility_v1\TRAIN_UTILITY_SEAL.json")
        if (!(Test-Path -LiteralPath $seal)) { throw "fold$fold TRAIN utility seal missing" }
    }
    $py = 'E:\Anaconda\envs\persist_stable_251\python.exe'
    $ErrorActionPreference = 'Continue'
    & $py $entry --runtime $runtime 2>&1 | ForEach-Object { Write-Host $_ }
    $result = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($result -ne 0) { throw "TRAIN evidence aggregation failed with exit $result" }
    Write-Host 'TRAIN_EVIDENCE_AGGREGATION_COMPLETE'
} catch {
    Write-Host "TRAIN_EVIDENCE_QUEUE_FAILURE=$($_.Exception.Message)"
    exit 1
} finally {
    Stop-Transcript | Out-Null
}
