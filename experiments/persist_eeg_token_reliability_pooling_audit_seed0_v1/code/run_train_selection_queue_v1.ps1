$ErrorActionPreference = 'Stop'
$root = 'D:\nips-temp\TotalP\P1\CRCICLR_SEVEN_BACKBONE_HELDOUT_WORK'
$code = Join-Path $root 'experiments\persist_eeg_token_reliability_pooling_audit_seed0_v1\code'
$entry = Join-Path $code 'select_train_v1.py'
$runtime = 'D:\nips-temp\TotalP\P1\token_reliability_pooling_runtime'
$log = Join-Path $runtime 'train_selection_queue_v1.log'
if (Test-Path -LiteralPath $log) { throw 'TRAIN selection queue log already exists' }
Start-Transcript -Path $log -NoClobber | Out-Null
try {
    $pins = @{
        'select_train_v1.py' = 'A1C6F63A73738FB69C3701F0B317326E78FAC284F07FB6D695079E31F8D1FC66'
        'aggregate.py' = '0F789C14716A9B6B8BC9DB5C151C5EA7F4C566AAE28B78582C25EE58740A9089'
        'token_scores.py' = '772D4BE16BD9CCCDAD71B439AFC36C910D07531A2B04900401C08284A66EDA59'
    }
    foreach ($name in $pins.Keys) {
        if ((Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $code $name)).Hash -ne $pins[$name]) {
            throw "TRAIN selection code SHA mismatch: $name"
        }
    }
    $deadline = (Get-Date).AddHours(68)
    while ($true) {
        if ((Get-Date) -gt $deadline) { throw 'TRAIN evidence queue wait timeout' }
        $task = Get-ScheduledTask -TaskName 'PERSIST_EEG_TOKEN_RELIABILITY_TRAIN_EVIDENCE_QUEUE_V1'
        $info = Get-ScheduledTaskInfo -TaskName 'PERSIST_EEG_TOKEN_RELIABILITY_TRAIN_EVIDENCE_QUEUE_V1'
        if ($task.State -eq 'Ready') {
            if ($info.LastTaskResult -ne 0) { throw 'TRAIN evidence queue failed; refusing pooling selection' }
            break
        }
        Start-Sleep -Seconds 60
    }
    foreach ($fold in 0..4) {
        $out = Join-Path $runtime ("fold$fold\train_selection_v1")
        $seal = Join-Path $runtime ("fold$fold\TRAIN_SELECTION_SEAL.json")
        if ((Test-Path -LiteralPath $out) -or (Test-Path -LiteralPath $seal)) {
            throw "fold$fold TRAIN selection output/seal already exists"
        }
    }
    $env:TOKEN_AUDIT_RUNTIME = $runtime
    $py = 'E:\Anaconda\envs\persist_stable_251\python.exe'
    foreach ($fold in 0..4) {
        $deadline = (Get-Date).AddHours(24)
        while ($true) {
            if ((Get-Date) -gt $deadline) { throw "fold$fold selection resource-gate timeout" }
            $ram = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB
            $gpu = [double]((& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | Select-Object -First 1).Trim())
            if ($ram -ge 40 -and $gpu -ge 16000) { break }
            Start-Sleep -Seconds 60
        }
        Write-Host "START_TRAIN_SELECTION fold=$fold RAM_GB=$ram GPU_FREE_MIB=$gpu"
        $ErrorActionPreference = 'Continue'
        & $py $entry --fold $fold 2>&1 | ForEach-Object { Write-Host $_ }
        $result = $LASTEXITCODE
        $ErrorActionPreference = 'Stop'
        if ($result -ne 0) { throw "TRAIN selection fold$fold failed with exit $result" }
    }
    Write-Host 'ALL_FIVE_TRAIN_SELECTIONS_COMPLETE'
} catch {
    Write-Host "TRAIN_SELECTION_QUEUE_FAILURE=$($_.Exception.Message)"
    exit 1
} finally {
    Stop-Transcript | Out-Null
}
