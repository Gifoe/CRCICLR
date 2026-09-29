$ErrorActionPreference = 'Stop'
$root = 'D:\nips-temp\TotalP\P1\CRCICLR_SEVEN_BACKBONE_HELDOUT_WORK'
$entry = Join-Path $root 'experiments\persist_eeg_token_reliability_pooling_audit_seed0_v1\code\run_v3.py'
$entryHash = '73E38D37F4CE0BD38AE9FAAEF776DA72D722F400B9D6EBA604BB63A3961CB386'
$runtime = 'D:\nips-temp\TotalP\P1\token_reliability_pooling_runtime'
$log = Join-Path $runtime 'train_extract_queue_v1.log'
if (Test-Path -LiteralPath $log) { throw 'queue v1 log already exists' }
New-Item -ItemType Directory -Path $runtime -Force | Out-Null
Start-Transcript -Path $log -NoClobber | Out-Null
try {
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $entry).Hash -ne $entryHash) { throw 'entry SHA mismatch' }
    $preflight = Get-ScheduledTaskInfo -TaskName 'PERSIST_EEG_TOKEN_RELIABILITY_CBRA_MOD_ALL_PREFLIGHT_V4'
    if ($preflight.LastTaskResult -ne 0) { throw 'all-fold preflight has not passed' }
    $env:TOKEN_AUDIT_RUNTIME = $runtime
    $env:SEVEN_RUNTIME = 'D:\nips-temp\TotalP\P1\seven_backbone_fourtask_3seed_runtime'
    $env:FULL_OPENBMI_CACHE = 'D:\nips-temp\TotalP\P1\persist_eeg_stage0_repo_full\outputs\persist_eeg_stage0\cache\openbmi'
    $env:OFFICIAL_BACKBONE_ROOT = 'D:\nips-temp\TotalP\P1\seven_backbone_fourtask_3seed_runtime\official'
    $env:MODERN_REPO = $root
    $py = 'E:\Anaconda\envs\persist_stable_251\python.exe'
    foreach ($fold in 0..4) {
        $out = Join-Path $runtime ("fold$fold\train")
        if (Test-Path -LiteralPath $out) { throw "fold$fold TRAIN output exists; refusing duplicate or partial overwrite" }
        $deadline = (Get-Date).AddHours(60)
        while ($true) {
            if ((Get-Date) -gt $deadline) { throw "fold$fold resource gate timeout" }
            $other = @(Get-ScheduledTask | Where-Object {
                $_.State -eq 'Running' -and $_.TaskName -match '^PERSIST_EEG_PC_' -and
                $_.TaskName -notmatch 'TOKEN_RELIABILITY'
            })
            $freeRam = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB
            $gpuLine = & nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>$null | Select-Object -First 1
            $freeGpu = 0
            if ($gpuLine -match '^\s*(\d+)') { $freeGpu = [int]$Matches[1] }
            if ($freeRam -ge 40 -and $freeGpu -ge 16000 -and $other.Count -eq 0) { break }
            Start-Sleep -Seconds 60
        }
        Write-Host "START_TRAIN_EXTRACTION fold=$fold free_ram_gb=$freeRam free_gpu_mib=$freeGpu"
        $ErrorActionPreference = 'Continue'
        & $py $entry --fold $fold --role train --batch 16 2>&1 | ForEach-Object { Write-Host $_ }
        $result = $LASTEXITCODE
        $ErrorActionPreference = 'Stop'
        if ($result -ne 0) { throw "fold$fold extraction failed with exit $result" }
        $meta = Join-Path $out 'EXTRACTION_PROVENANCE.json'
        if (!(Test-Path -LiteralPath $meta)) { throw "fold$fold extraction missing provenance" }
        Write-Host "COMPLETE_TRAIN_EXTRACTION fold=$fold provenance_sha=$((Get-FileHash -Algorithm SHA256 -LiteralPath $meta).Hash)"
    }
    Write-Host 'ALL_FIVE_TRAIN_EXTRACTIONS_COMPLETE'
} catch {
    Write-Host "TRAIN_QUEUE_FAILURE=$($_.Exception.Message)"
    exit 1
} finally {
    Stop-Transcript | Out-Null
}
