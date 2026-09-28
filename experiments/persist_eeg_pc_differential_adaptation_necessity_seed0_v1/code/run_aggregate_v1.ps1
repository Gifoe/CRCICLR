$ErrorActionPreference = 'Stop'
$base = 'D:\nips-temp\TotalP\P1\pc_differential_adaptation_necessity_runtime'
$python = 'E:\Anaconda\envs\persist_stable_251\python.exe'
$script = Join-Path $base 'code\aggregate_v1.py'
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
$codeSha = 'ED6A9DF7D43358A9BDEE94CADEC1F220E3A6C7CAA95E1A1B5FBD378CFEBADE9E'
$lockSha = 'C245021B69022921F46EA6A1414791B8C01348382AAD04361C479D939BF9F4A5'
$log = Join-Path $base 'logs\aggregate_v1.log'
try {
    if (Test-Path $log) { throw 'Aggregate log exists' }
    if ((Get-FileHash -Algorithm SHA256 $script).Hash -ne $codeSha) { throw 'Aggregate source SHA drift' }
    if ((Get-FileHash -Algorithm SHA256 (Join-Path $base 'protocol\ANALYSIS_LOCK.json')).Hash -ne $lockSha) { throw 'Analysis lock SHA drift' }
    if (-not (Select-String -Path (Join-Path $base 'logs\outer_queue_v3.log') -Pattern 'ALL_15_OUTER_COMPLETE' -Quiet)) { throw 'OUTER queue incomplete' }
    if (@(Get-ChildItem (Join-Path $base 'outer') -Filter '*.json').Count -ne 15) { throw 'OUTER file count mismatch' }
    if ((Test-Path (Join-Path $base 'outputs')) -and @(Get-ChildItem (Join-Path $base 'outputs') -File).Count -ne 0) { throw 'Outputs already exist' }
    "START $(Get-Date -Format o)" | Out-File -LiteralPath $log -Encoding utf8
    $ErrorActionPreference = 'Continue'
    & $python $script *>> $log
    $exit = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($exit -ne 0) { throw "Python exited $exit" }
    $files = @(Get-ChildItem (Join-Path $base 'outputs') -File)
    if ($files.Count -ne 18) { throw "Output count $($files.Count), expected 18" }
    foreach ($name in @('SOURCE_PROVENANCE.json','REPRESENTATION_PROVENANCE.json','PC_GEOMETRY_AUDIT.json')) {
        if (-not (Test-Path (Join-Path $base "protocol\$name"))) { throw "Missing provenance $name" }
    }
    "ALL_OUTPUTS_COMPLETE $(Get-Date -Format o)" | Out-File -LiteralPath $log -Append -Encoding utf8
} catch {
    "FAIL $(Get-Date -Format o) $_" | Out-File -LiteralPath (Join-Path $base 'logs\aggregate_v1.failure.log') -Encoding utf8
    throw
}
