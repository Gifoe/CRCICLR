$ErrorActionPreference = 'Stop'
$root = 'D:\nips-temp\TotalP\P1\CRCICLR_SEVEN_BACKBONE_HELDOUT_WORK'
$code = Join-Path $root 'experiments\persist_eeg_token_reliability_pooling_audit_seed0_v1\code'
$extract = Join-Path $code 'run_v3.py'
$evaluate = Join-Path $code 'evaluate_outer_v1.py'
$runtime = 'D:\nips-temp\TotalP\P1\token_reliability_pooling_runtime'
$log = Join-Path $runtime 'outer_queue_v1.log'
if (Test-Path -LiteralPath $log) { throw 'OUTER queue log already exists' }
Start-Transcript -Path $log -NoClobber | Out-Null
try {
    $pins = @{
        'run_v3.py' = '73E38D37F4CE0BD38AE9FAAEF776DA72D722F400B9D6EBA604BB63A3961CB386'
        'evaluate_outer_v1.py' = '038AE750CC7F9334D23D7A5AB82934E8918B493763DA2B484144F9A2054D9278'
        'aggregate.py' = '0F789C14716A9B6B8BC9DB5C151C5EA7F4C566AAE28B78582C25EE58740A9089'
    }
    foreach ($name in $pins.Keys) {
        if ((Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $code $name)).Hash -ne $pins[$name]) {
            throw "OUTER code SHA mismatch: $name"
        }
    }
    $deadline = (Get-Date).AddHours(160)
    while ($true) {
        if ((Get-Date) -gt $deadline) { throw 'TRAIN selection queue wait timeout' }
        $task = Get-ScheduledTask -TaskName 'PERSIST_EEG_TOKEN_RELIABILITY_SELECTION_QUEUE_V1'
        $info = Get-ScheduledTaskInfo -TaskName 'PERSIST_EEG_TOKEN_RELIABILITY_SELECTION_QUEUE_V1'
        if ($task.State -eq 'Ready') {
            if ($info.LastTaskResult -ne 0) { throw 'TRAIN selection failed; OUTER must remain closed' }
            break
        }
        Start-Sleep -Seconds 60
    }
    foreach ($fold in 0..4) {
        $seal = Join-Path $runtime ("fold$fold\TRAIN_SELECTION_SEAL.json")
        $out = Join-Path $runtime ("fold$fold\outer")
        $eval = Join-Path $runtime ("fold$fold\outer_evaluation_v1")
        if (!(Test-Path -LiteralPath $seal) -or (Test-Path -LiteralPath $out) -or (Test-Path -LiteralPath $eval)) {
            throw "fold$fold TRAIN seal missing or OUTER output already exists"
        }
    }
    $env:TOKEN_AUDIT_RUNTIME = $runtime
    $env:SEVEN_RUNTIME = 'D:\nips-temp\TotalP\P1\seven_backbone_fourtask_3seed_runtime'
    $env:FULL_OPENBMI_CACHE = 'D:\nips-temp\TotalP\P1\persist_eeg_stage0_repo_full\outputs\persist_eeg_stage0\cache\openbmi'
    $env:OFFICIAL_BACKBONE_ROOT = 'D:\nips-temp\TotalP\P1\seven_backbone_fourtask_3seed_runtime\official'
    $env:MODERN_REPO = $root
    $py = 'E:\Anaconda\envs\persist_stable_251\python.exe'
    foreach ($fold in 0..4) {
        $deadline = (Get-Date).AddHours(60)
        while ($true) {
            if ((Get-Date) -gt $deadline) { throw "fold$fold OUTER resource-gate timeout" }
            $other = @(Get-ScheduledTask | Where-Object {
                $_.State -eq 'Running' -and $_.TaskName -match '^PERSIST_EEG_PC_' -and
                $_.TaskName -notmatch 'TOKEN_RELIABILITY'
            })
            $ram = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB
            $gpuLine = & nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>$null | Select-Object -First 1
            $gpu = 0
            if ($gpuLine -match '^\s*(\d+)') { $gpu = [int]$Matches[1] }
            if ($ram -ge 40 -and $gpu -ge 16000 -and $other.Count -eq 0) { break }
            Start-Sleep -Seconds 60
        }
        Write-Host "START_OUTER_EXTRACTION fold=$fold RAM_GB=$ram GPU_FREE_MIB=$gpu"
        $ErrorActionPreference = 'Continue'
        & $py $extract --fold $fold --role outer --batch 16 2>&1 | ForEach-Object { Write-Host $_ }
        $result = $LASTEXITCODE
        $ErrorActionPreference = 'Stop'
        if ($result -ne 0) { throw "fold$fold OUTER extraction failed with exit $result" }
        Write-Host "START_OUTER_EVALUATION fold=$fold"
        $ErrorActionPreference = 'Continue'
        & $py $evaluate --fold $fold 2>&1 | ForEach-Object { Write-Host $_ }
        $result = $LASTEXITCODE
        $ErrorActionPreference = 'Stop'
        if ($result -ne 0) { throw "fold$fold OUTER evaluation failed with exit $result" }
    }
    Write-Host 'ALL_FIVE_OUTER_EVALUATIONS_COMPLETE'
} catch {
    Write-Host "OUTER_QUEUE_FAILURE=$($_.Exception.Message)"
    exit 1
} finally {
    Stop-Transcript | Out-Null
}
