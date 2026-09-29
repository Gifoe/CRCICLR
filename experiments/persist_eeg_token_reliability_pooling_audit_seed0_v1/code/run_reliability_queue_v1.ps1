$ErrorActionPreference = 'Stop'
$root = 'D:\nips-temp\TotalP\P1\CRCICLR_SEVEN_BACKBONE_HELDOUT_WORK'
$code = Join-Path $root 'experiments\persist_eeg_token_reliability_pooling_audit_seed0_v1\code'
$entry = Join-Path $code 'analyze_train.py'
$scores = Join-Path $code 'token_scores.py'
$runtime = 'D:\nips-temp\TotalP\P1\token_reliability_pooling_runtime'
$log = Join-Path $runtime 'reliability_queue_v1.log'
if (Test-Path -LiteralPath $log) { throw 'reliability queue log already exists' }
New-Item -ItemType Directory -Path $runtime -Force | Out-Null
Start-Transcript -Path $log -NoClobber | Out-Null
try {
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $entry).Hash -ne '26E7C150F912BEB51FA665F7BDC2CA663241BF95FD74DEBB3BD8C02262918E40') { throw 'analyzer SHA mismatch' }
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $scores).Hash -ne '772D4BE16BD9CCCDAD71B439AFC36C910D07531A2B04900401C08284A66EDA59') { throw 'score source SHA mismatch' }
    $deadline = (Get-Date).AddHours(68)
    while ($true) {
        if ((Get-Date) -gt $deadline) { throw 'TRAIN extraction wait timeout' }
        $task = Get-ScheduledTask -TaskName 'PERSIST_EEG_TOKEN_RELIABILITY_TRAIN_EXTRACT_QUEUE_V1'
        $info = Get-ScheduledTaskInfo -TaskName 'PERSIST_EEG_TOKEN_RELIABILITY_TRAIN_EXTRACT_QUEUE_V1'
        if ($task.State -eq 'Ready') {
            if ($info.LastTaskResult -ne 0) { throw 'TRAIN extraction queue failed; refusing analysis' }
            break
        }
        Start-Sleep -Seconds 60
    }
    foreach ($fold in 0..4) {
        $source = Join-Path $runtime ("fold$fold\train\EXTRACTION_PROVENANCE.json")
        $out = Join-Path $runtime ("fold$fold\reliability_v1")
        if (!(Test-Path -LiteralPath $source) -or (Test-Path -LiteralPath $out)) { throw "fold$fold source missing or output already exists" }
    }
    $env:TOKEN_AUDIT_RUNTIME = $runtime
    $py = 'E:\Anaconda\envs\persist_stable_251\python.exe'
    foreach ($fold in 0..4) {
        Write-Host "START_TRAIN_RELIABILITY fold=$fold"
        $ErrorActionPreference = 'Continue'
        & $py $entry --fold $fold 2>&1 | ForEach-Object { Write-Host $_ }
        $result = $LASTEXITCODE
        $ErrorActionPreference = 'Stop'
        if ($result -ne 0) { throw "TRAIN reliability fold$fold failed with exit $result" }
    }
    Write-Host 'ALL_FIVE_TRAIN_RELIABILITY_AUDITS_COMPLETE'
} catch {
    Write-Host "RELIABILITY_QUEUE_FAILURE=$($_.Exception.Message)"
    exit 1
} finally {
    Stop-Transcript | Out-Null
}
