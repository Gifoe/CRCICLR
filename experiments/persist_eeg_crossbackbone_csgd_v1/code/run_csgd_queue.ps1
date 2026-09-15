$ErrorActionPreference = "Stop"
$repo = "D:\nips-temp\TotalP\P1\CRCICLR_CROSSBACKBONE_CSGD_WORK"
$runtime = "D:\nips-temp\TotalP\P1\crossbackbone_csgd_runtime"
$python = "E:\Anaconda\envs\persist_stable_251\python.exe"
$runner = Join-Path $repo "experiments\persist_eeg_crossbackbone_csgd_v1\code\run_crossbackbone_csgd.py"
$log = Join-Path $runtime "csgd_queue.log"

$env:CSGD_RUNTIME = $runtime
$env:CSGD_CPU_THREADS = "4"
$env:SEVEN_REPO = "D:\nips-temp\TotalP\P1\CRCICLR_BACKBONE_GEN_WORK"
$env:SEVEN_RUNTIME = "D:\nips-temp\TotalP\P1\seven_backbone_fourtask_3seed_runtime"
$env:FULL_OPENBMI_CACHE = "D:\nips-temp\TotalP\P1\persist_eeg_stage0_repo_full\outputs\persist_eeg_stage0\cache\openbmi"
$env:FULL_WBCIC_CACHE = "D:\nips-temp\TotalP\P1\CRCICLR_SOURCE_ONLY_DIAGNOSTIC\experiments\persist_eeg_wbcic_independent_replication_v1\runtime\cache\wbcic_epochs"
$env:TRUE_OUTER_WBCIC_CACHE = "D:\nips-temp\TotalP\P2\wbcic_outer_cache\wbcic_epochs"

New-Item -ItemType Directory -Force -Path $runtime | Out-Null

function Invoke-CSGD([string[]]$Arguments) {
    & $python -u $runner @Arguments 2>&1 | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) {
        throw "CSGD runner failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

try {
    Invoke-CSGD @("--stage", "prelock")
    foreach ($model in @("EEGNet", "LiteBN", "TeCh", "CBraMod", "ModernTCN", "Medformer", "SGN", "TFFormer")) {
        Invoke-CSGD @("--stage", "run", "--model", $model)
    }
    Invoke-CSGD @("--stage", "aggregate")
    [IO.File]::WriteAllText((Join-Path $runtime "COMPLETED.txt"), [DateTime]::UtcNow.ToString("o") + [Environment]::NewLine)
}
catch {
    $message = $_ | Out-String
    [IO.File]::WriteAllText((Join-Path $runtime "FAILED.txt"), $message)
    throw
}
