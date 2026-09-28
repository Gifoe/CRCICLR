$ErrorActionPreference = 'Stop'
$base = 'D:\nips-temp\TotalP\P1\pc_differential_adaptation_necessity_runtime'
$python = 'E:\Anaconda\envs\persist_stable_251\python.exe'
$script = Join-Path $base 'code\run_prepare_v1.py'
$codeSha = '047FA3D6E0E181E3739B88BFC309B77925E91190B05D459E1FA26340C8E34420'
$protocolSha = 'FB856216903422091916114DE48BD63A755D94D601AB1346A9E0C082FB31BA5D'
$env:PCDA_RUNTIME = $base
$env:PCDA_SDG_CODE = 'D:\nips-temp\TotalP\P1\shared_decision_geometry_full_runtime\code\run_full_v1.py'
$env:PCDA_EEGNET_PEEH_CODE = Join-Path $base 'code\peeh_eegnet_source.py'
$env:PCDA_CONFORMER_PEEH_CODE = Join-Path $base 'code\peeh_conformer_source.py'
$env:SEVEN_REPO = 'D:\nips-temp\TotalP\P1\CRCICLR_BACKBONE_GEN_WORK'
$env:CONFORMER_REPO = 'D:\nips-temp\TotalP\P1\CRCICLR_EEGCONFORMER_FBCNET_FINAL_V1'
$env:SEVEN_RUNTIME = 'D:\nips-temp\TotalP\P1\seven_backbone_fourtask_3seed_runtime'
$env:NEW_BASELINE_RUNTIME = 'D:\nips-temp\TotalP\P1\eegconformer_fbcnet_runtime'
$env:FULL_OPENBMI_CACHE = 'D:\nips-temp\TotalP\P1\persist_eeg_stage0_repo_full\outputs\persist_eeg_stage0\cache\openbmi'
$queueLog = Join-Path $base 'logs\prepare_queue_v1.log'
$cells = @(
    'EEGNet|OpenBMI_MI|0','EEGNet|OpenBMI_MI|1','EEGNet|OpenBMI_MI|2','EEGNet|OpenBMI_MI|3','EEGNet|OpenBMI_MI|4',
    'EEGNet|OpenBMI_ERP|0','EEGNet|OpenBMI_ERP|1','EEGNet|OpenBMI_ERP|2','EEGNet|OpenBMI_ERP|3','EEGNet|OpenBMI_ERP|4',
    'EEGConformer|OpenBMI_MI|0','EEGConformer|OpenBMI_MI|1','EEGConformer|OpenBMI_MI|2','EEGConformer|OpenBMI_MI|3','EEGConformer|OpenBMI_MI|4'
)
try {
    if (Test-Path -LiteralPath $queueLog) { throw 'Queue log already exists' }
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $script).Hash -ne $codeSha) { throw 'Code SHA mismatch' }
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $base 'protocol\PROTOCOL_LOCK.json')).Hash -ne $protocolSha) { throw 'Protocol SHA mismatch' }
    "START $(Get-Date -Format o)" | Out-File -LiteralPath $queueLog -Encoding utf8
    foreach ($entry in $cells) {
        $parts = $entry.Split('|'); $model = $parts[0]; $task = $parts[1]; $fold = [int]$parts[2]
        $name = "$($model.ToLower())_$($task.ToLower())_fold$($fold)_seed0"
        $npz = Join-Path $base "train\$name.npz"
        $json = Join-Path $base "train\$name.json"
        $log = Join-Path $base "logs\$name.prepare_v1.log"
        if ((Test-Path -LiteralPath $npz) -or (Test-Path -LiteralPath $json) -or (Test-Path -LiteralPath $log)) { throw "Existing result/log: $name" }
        while ($true) {
            $freeRam = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB
            $freeGpu = [int](nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | Select-Object -First 1)
            if ($freeRam -ge 40 -and $freeGpu -ge 16000) { break }
            "WAIT $(Get-Date -Format o) $name RAM=$freeRam GPU=$freeGpu" | Out-File -LiteralPath $queueLog -Append -Encoding utf8
            Start-Sleep -Seconds 30
        }
        "START $(Get-Date -Format o) $name RAM=$freeRam GPU=$freeGpu" | Out-File -LiteralPath $queueLog -Append -Encoding utf8
        "START $(Get-Date -Format o)" | Out-File -LiteralPath $log -Encoding utf8
        $ErrorActionPreference = 'Continue'
        & $python $script prepare --model $model --task $task --fold $fold *>> $log
        $code = $LASTEXITCODE
        $ErrorActionPreference = 'Stop'
        if ($code -ne 0) { throw "Python exited $code for $name" }
        if (-not (Test-Path -LiteralPath $npz) -or -not (Test-Path -LiteralPath $json)) { throw "Missing preparation output for $name" }
        $result = Get-Content -LiteralPath $json -Raw | ConvertFrom-Json
        if ($result.model -ne $model -or $result.task -ne $task -or $result.fold -ne $fold -or $result.final_heldout_eeg_reads -ne 0 -or $result.train_cache_sha256 -ne (Get-FileHash -Algorithm SHA256 -LiteralPath $npz).Hash) { throw "Audit mismatch for $name" }
        "COMPLETE $(Get-Date -Format o) $name SHA=$((Get-FileHash -Algorithm SHA256 -LiteralPath $json).Hash)" | Out-File -LiteralPath $queueLog -Append -Encoding utf8
    }
    "ALL_15_PREPARED $(Get-Date -Format o)" | Out-File -LiteralPath $queueLog -Append -Encoding utf8
} catch {
    "FAIL $(Get-Date -Format o) $_" | Out-File -LiteralPath (Join-Path $base 'logs\prepare_queue_v1.failure.log') -Encoding utf8
    throw
}
