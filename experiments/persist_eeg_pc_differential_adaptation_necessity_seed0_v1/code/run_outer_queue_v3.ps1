$ErrorActionPreference = 'Stop'
$base = 'D:\nips-temp\TotalP\P1\pc_differential_adaptation_necessity_runtime'
$python = 'E:\Anaconda\envs\persist_stable_251\python.exe'
$script = Join-Path $base 'code\outer_v2.py'
$env:PCDA_CORE_CODE = Join-Path $base 'code\run_prepare_v1.py'
$env:PCDA_ANALYSIS_CODE = Join-Path $base 'code\analysis_select_v1.py'
$env:PCDA_RUNTIME = $base
$env:PCDA_SDG_CODE = 'D:\nips-temp\TotalP\P1\shared_decision_geometry_full_runtime\code\run_full_v1.py'
$env:PCDA_EEGNET_PEEH_CODE = Join-Path $base 'code\peeh_eegnet_source.py'
$env:PCDA_CONFORMER_PEEH_CODE = Join-Path $base 'code\peeh_conformer_source.py'
$env:SEVEN_REPO = 'D:\nips-temp\TotalP\P1\CRCICLR_BACKBONE_GEN_WORK'
$env:CONFORMER_REPO = 'D:\nips-temp\TotalP\P1\CRCICLR_EEGCONFORMER_FBCNET_FINAL_V1'
$env:SEVEN_RUNTIME = 'D:\nips-temp\TotalP\P1\seven_backbone_fourtask_3seed_runtime'
$env:NEW_BASELINE_RUNTIME = 'D:\nips-temp\TotalP\P1\eegconformer_fbcnet_runtime'
$env:FULL_OPENBMI_CACHE = 'D:\nips-temp\TotalP\P1\persist_eeg_stage0_repo_full\outputs\persist_eeg_stage0\cache\openbmi'
$codeSha = '94386C13642CC36DD3C3E2188A499CB12E8C4000AA1CDCD6FF0DD654F5BFF311'
$analysisSha = '157FFCBECE8F82D985B45573A427C9D47983CD66C6D8C7BC20A300BB50E5696C'
$protocolSha = 'FB856216903422091916114DE48BD63A755D94D601AB1346A9E0C082FB31BA5D'
$lockSha = 'C245021B69022921F46EA6A1414791B8C01348382AAD04361C479D939BF9F4A5'
$queueLog = Join-Path $base 'logs\outer_queue_v3.log'
$cells = @(
    'EEGNet|OpenBMI_MI|0','EEGNet|OpenBMI_MI|1','EEGNet|OpenBMI_MI|2','EEGNet|OpenBMI_MI|3','EEGNet|OpenBMI_MI|4',
    'EEGNet|OpenBMI_ERP|0','EEGNet|OpenBMI_ERP|1','EEGNet|OpenBMI_ERP|2','EEGNet|OpenBMI_ERP|3','EEGNet|OpenBMI_ERP|4',
    'EEGConformer|OpenBMI_MI|0','EEGConformer|OpenBMI_MI|1','EEGConformer|OpenBMI_MI|2','EEGConformer|OpenBMI_MI|3','EEGConformer|OpenBMI_MI|4'
)
try {
    if (Test-Path -LiteralPath $queueLog) { throw 'OUTER queue log exists' }
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $script).Hash -ne $codeSha) { throw 'OUTER code SHA mismatch' }
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $env:PCDA_ANALYSIS_CODE).Hash -ne $analysisSha) { throw 'TRAIN selection code SHA mismatch' }
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $base 'protocol\PROTOCOL_LOCK.json')).Hash -ne $protocolSha) { throw 'Protocol SHA mismatch' }
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $base 'protocol\ANALYSIS_LOCK.json')).Hash -ne $lockSha) { throw 'Pre-OUTER analysis lock SHA mismatch' }
    if (-not (Select-String -Path (Join-Path $base 'logs\select_queue_v1.log') -Pattern 'ALL_15_TRAIN_SELECTED' -Quiet)) { throw 'TRAIN selection queue not complete' }
    "START $(Get-Date -Format o)" | Out-File -LiteralPath $queueLog -Encoding utf8
    foreach ($entry in $cells) {
        $parts = $entry.Split('|'); $model = $parts[0]; $task = $parts[1]; $fold = [int]$parts[2]
        $name = "$($model.ToLower())_$($task.ToLower())_fold$($fold)_seed0"
        $selected = Join-Path $base "selection\$name.json"
        $target = Join-Path $base "outer\$name.json"
        $log = Join-Path $base "logs\$name.outer_v3.log"
        if (-not (Test-Path -LiteralPath $selected)) { throw "Missing TRAIN selection: $name" }
        if ((Test-Path -LiteralPath $target) -or (Test-Path -LiteralPath $log)) { throw "Existing OUTER result/log: $name" }
        $freeRam = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory * 1KB
        if ($freeRam -lt 40GB) { throw "Heavy launch RAM gate for ${name}: free=$freeRam" }
        $gpu = & nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits
        if (-not $gpu -or [int]($gpu | Select-Object -First 1) -lt 16000) { throw "Heavy launch GPU gate for ${name}: freeMiB=$gpu" }
        "START $(Get-Date -Format o) $name" | Out-File -LiteralPath $queueLog -Append -Encoding utf8
        "START $(Get-Date -Format o)" | Out-File -LiteralPath $log -Encoding utf8
        $ErrorActionPreference = 'Continue'
        & $python $script --model $model --task $task --fold $fold *>> $log
        $code = $LASTEXITCODE
        $ErrorActionPreference = 'Stop'
        if ($code -ne 0) { throw "Python exited $code for $name" }
        if (-not (Test-Path -LiteralPath $target)) { throw "Missing OUTER result for $name" }
        $value = Get-Content -LiteralPath $target -Raw | ConvertFrom-Json
        if ($value.model -ne $model -or $value.task -ne $task -or $value.fold -ne $fold -or
            $value.formal_final_heldout_eeg_reads -ne 0 -or $value.outer_used_for_selection -ne $false -or
            @($value.method_subject_rows).Count -lt 80 -or @($value.random_subject_rows).Count -ne (100 * $value.outer_subject_count)) {
            throw "OUTER audit mismatch for $name"
        }
        "COMPLETE $(Get-Date -Format o) $name SHA=$((Get-FileHash -Algorithm SHA256 -LiteralPath $target).Hash)" | Out-File -LiteralPath $queueLog -Append -Encoding utf8
    }
    "ALL_15_OUTER_COMPLETE $(Get-Date -Format o)" | Out-File -LiteralPath $queueLog -Append -Encoding utf8
} catch {
    "FAIL $(Get-Date -Format o) $_" | Out-File -LiteralPath (Join-Path $base 'logs\outer_queue_v3.failure.log') -Encoding utf8
    throw
}
