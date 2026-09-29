$ErrorActionPreference = 'Stop'
$root = 'D:\nips-temp\TotalP\P1\CRCICLR_SEVEN_BACKBONE_HELDOUT_WORK'
$entry = Join-Path $root 'experiments\persist_eeg_token_reliability_pooling_audit_seed0_v1\code\utility.py'
$runtime = 'D:\nips-temp\TotalP\P1\token_reliability_pooling_runtime'
$log = Join-Path $runtime 'utility_queue_v1.log'
if (Test-Path -LiteralPath $log) { throw 'utility queue log already exists' }
New-Item -ItemType Directory -Path $runtime -Force | Out-Null
Start-Transcript -Path $log -NoClobber | Out-Null
try {
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $entry).Hash -ne '55DE2AE09A27E8838CAADC082F3FF2B09C3D5663A7DB307EC565F39083E90E3C') { throw 'utility source SHA mismatch' }
    $deadline = (Get-Date).AddHours(68)
    while ($true) {
        if ((Get-Date) -gt $deadline) { throw 'reliability queue wait timeout' }
        $task = Get-ScheduledTask -TaskName 'PERSIST_EEG_TOKEN_RELIABILITY_ANALYZE_QUEUE_V1'
        $info = Get-ScheduledTaskInfo -TaskName 'PERSIST_EEG_TOKEN_RELIABILITY_ANALYZE_QUEUE_V1'
        if ($task.State -eq 'Ready') {
            if ($info.LastTaskResult -ne 0) { throw 'reliability queue failed; refusing utility analysis' }
            break
        }
        Start-Sleep -Seconds 60
    }
    foreach ($fold in 0..4) {
        $source = Join-Path $runtime ("fold$fold\train\EXTRACTION_PROVENANCE.json")
        $reliability = Join-Path $runtime ("fold$fold\reliability_v1\TRAIN_RELIABILITY_SEAL.json")
        $out = Join-Path $runtime ("fold$fold\utility_v1")
        if (!(Test-Path -LiteralPath $source) -or !(Test-Path -LiteralPath $reliability) -or (Test-Path -LiteralPath $out)) {
            throw "fold$fold source/reliability missing or utility output exists"
        }
    }
    $env:TOKEN_AUDIT_RUNTIME = $runtime
    $py = 'E:\Anaconda\envs\persist_stable_251\python.exe'
    foreach ($fold in 0..4) {
        $deadline = (Get-Date).AddHours(12)
        while ((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB -lt 24) {
            if ((Get-Date) -gt $deadline) { throw "fold$fold utility RAM gate timeout" }
            Start-Sleep -Seconds 60
        }
        Write-Host "START_TRAIN_UTILITY fold=$fold"
        $ErrorActionPreference = 'Continue'
        & $py $entry --fold $fold 2>&1 | ForEach-Object { Write-Host $_ }
        $result = $LASTEXITCODE
        $ErrorActionPreference = 'Stop'
        if ($result -ne 0) { throw "TRAIN utility fold$fold failed with exit $result" }
    }
    Write-Host 'ALL_FIVE_TRAIN_UTILITY_AUDITS_COMPLETE'
} catch {
    Write-Host "UTILITY_QUEUE_FAILURE=$($_.Exception.Message)"
    exit 1
} finally {
    Stop-Transcript | Out-Null
}
