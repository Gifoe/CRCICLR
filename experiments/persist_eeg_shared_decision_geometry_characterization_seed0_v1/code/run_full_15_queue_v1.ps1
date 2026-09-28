$ErrorActionPreference = 'Stop'
$base = 'D:\nips-temp\TotalP\P1\shared_decision_geometry_full_runtime'
$python = 'E:\Anaconda\envs\persist_stable_251\python.exe'
$script = Join-Path $base 'code\run_full_v1.py'
$expectedCodeSha = '6C290FFCDAC1ADF83E8BDABD03661F0828B26B6F937360A3DA2A5073F99246AA'
$expectedProtocolSha = '944B5FADDA85DD674B2CD3760C3033DC98253B41C7E59C2C17A203AB58DCD30C'
$env:SDG_RUNTIME = $base
$env:SEVEN_REPO = 'D:\nips-temp\TotalP\P1\CRCICLR_BACKBONE_GEN_WORK'
$env:CONFORMER_REPO = 'D:\nips-temp\TotalP\P1\CRCICLR_EEGCONFORMER_FBCNET_FINAL_V1'
$env:SEVEN_RUNTIME = 'D:\nips-temp\TotalP\P1\seven_backbone_fourtask_3seed_runtime'
$env:NEW_BASELINE_RUNTIME = 'D:\nips-temp\TotalP\P1\eegconformer_fbcnet_runtime'
$env:FULL_OPENBMI_CACHE = 'D:\nips-temp\TotalP\P1\persist_eeg_stage0_repo_full\outputs\persist_eeg_stage0\cache\openbmi'
$queueLog = Join-Path $base 'logs\full_15_queue_v1.log'
$cells = @(
    'EEGNet|OpenBMI_MI|0', 'EEGNet|OpenBMI_MI|1', 'EEGNet|OpenBMI_MI|2', 'EEGNet|OpenBMI_MI|3', 'EEGNet|OpenBMI_MI|4',
    'EEGNet|OpenBMI_ERP|0', 'EEGNet|OpenBMI_ERP|1', 'EEGNet|OpenBMI_ERP|2', 'EEGNet|OpenBMI_ERP|3', 'EEGNet|OpenBMI_ERP|4',
    'EEGConformer|OpenBMI_MI|0', 'EEGConformer|OpenBMI_MI|1', 'EEGConformer|OpenBMI_MI|2', 'EEGConformer|OpenBMI_MI|3', 'EEGConformer|OpenBMI_MI|4'
)
try {
    if (Test-Path -LiteralPath $queueLog) { throw 'Queue log exists; refusing duplicate launch' }
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $script).Hash -ne $expectedCodeSha) { throw 'Code SHA drift' }
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $base 'protocol\PROTOCOL_LOCK.json')).Hash -ne $expectedProtocolSha) { throw 'Protocol SHA drift' }
    "START $(Get-Date -Format o) CODE_SHA=$expectedCodeSha PROTOCOL_SHA=$expectedProtocolSha" | Out-File -LiteralPath $queueLog -Encoding utf8
    foreach ($entry in $cells) {
        $parts = $entry.Split('|')
        $backbone, $task, $fold = $parts[0], $parts[1], [int]$parts[2]
        $name = "$($backbone.ToLower())_$($task.ToLower())_fold$fold"
        $result = Join-Path $base "folds\$name.json"
        $log = Join-Path $base "logs\$name.queue_v1.log"
        if ((Test-Path -LiteralPath $result) -or (Test-Path -LiteralPath $log)) { throw "Duplicate result/log for $name" }
        while ($true) {
            $freeRamGB = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB
            $freeGpuMiB = [int](nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | Select-Object -First 1)
            $other = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' -and $_.CommandLine -match 'shared_decision_geometry_full_runtime.*fold' })
            if ($other.Count -eq 0 -and $freeRamGB -ge 40 -and $freeGpuMiB -ge 16000) { break }
            "WAIT $(Get-Date -Format o) $name RAM_GB=$freeRamGB GPU_MiB=$freeGpuMiB other_sdg=$($other.Count)" | Out-File -LiteralPath $queueLog -Append -Encoding utf8
            Start-Sleep -Seconds 45
        }
        "START $(Get-Date -Format o) $name RAM_GB=$freeRamGB GPU_MiB=$freeGpuMiB" | Out-File -LiteralPath $queueLog -Append -Encoding utf8
        "START $(Get-Date -Format o)" | Out-File -LiteralPath $log -Encoding utf8
        $ErrorActionPreference = 'Continue'
        & $python $script fold --backbone $backbone --task $task --fold $fold *>> $log
        $code = $LASTEXITCODE
        $ErrorActionPreference = 'Stop'
        if ($code -ne 0) { throw "Python failed for $name with exit $code" }
        if (-not (Test-Path -LiteralPath $result)) { throw "Missing result for $name" }
        $value = Get-Content -LiteralPath $result -Raw | ConvertFrom-Json
        if ($value.backbone -ne $backbone -or $value.task -ne $task -or $value.fold -ne $fold -or $value.proof.final_heldout_eeg_reads -ne 0 -or $null -ne $value.proof.cap_per_subject_session_class) {
            throw "Identity, all-trial or heldout audit failure for $name"
        }
        if (@($value.rows.DECISION_SPECTRUM_NULLS).Count -ne 2800) { throw "Null count failure for $name" }
        "COMPLETE $(Get-Date -Format o) $name SHA256=$((Get-FileHash -Algorithm SHA256 -LiteralPath $result).Hash) TRAIN_ROWS=$($value.proof.train_rows)" | Out-File -LiteralPath $queueLog -Append -Encoding utf8
    }
    "ALL_15_FULL_TRIAL_FOLDS_COMPLETE $(Get-Date -Format o)" | Out-File -LiteralPath $queueLog -Append -Encoding utf8
} catch {
    "FAIL $(Get-Date -Format o): $_" | Out-File -LiteralPath (Join-Path $base 'logs\full_15_queue_v1.failure.log') -Encoding utf8
    throw
}
