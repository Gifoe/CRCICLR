$ErrorActionPreference = 'Stop'
$python = 'E:/Anaconda/envs/persist_stable_251/python.exe'
$repo = 'D:/nips-temp/TotalP/P1/CRCICLR_P2_SELECTION_FACTOR_WORK'
$code = Join-Path $repo 'experiments/persist_eeg_p2_selection_factor_eegnet_v1/code'
$runtime = 'D:/nips-temp/TotalP/P1/p2_selection_factor_eegnet_runtime'
$env:P2_RUNTIME = $runtime
$env:PSWA_RUNTIME = 'D:/nips-temp/TotalP/P1/crossbackbone_pswa_runtime'
$env:PEEH_RUNTIME = 'D:/nips-temp/TotalP/P1/crossbackbone_peeh_runtime'
$env:PEEH_REPO = 'D:/nips-temp/TotalP/P1/CRCICLR_CROSSBACKBONE_PEEH_WORK'
$env:SEVEN_REPO = 'D:/nips-temp/TotalP/P1/CRCICLR_BACKBONE_GEN_WORK'
$env:SEVEN_RUNTIME = 'D:/nips-temp/TotalP/P1/seven_backbone_fourtask_3seed_runtime'
$env:FULL_OPENBMI_CACHE = 'D:/nips-temp/TotalP/P1/persist_eeg_stage0_repo_full/outputs/persist_eeg_stage0/cache/openbmi'
$env:FULL_WBCIC_CACHE = 'D:/nips-temp/TotalP/P1/CRCICLR_SOURCE_ONLY_DIAGNOSTIC/experiments/persist_eeg_wbcic_independent_replication_v1/runtime/cache/wbcic_epochs'
$env:TRUE_OUTER_WBCIC_CACHE = 'D:/nips-temp/TotalP/P2/wbcic_outer_cache/wbcic_epochs'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$logDir = Join-Path $runtime 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
Set-Location $code

& $python -u run_p2_selection_factor.py --stage protocol 1>> (Join-Path $logDir 'queue.out.log') 2>> (Join-Path $logDir 'queue.err.log')
if ($LASTEXITCODE -ne 0) { throw 'P2 protocol creation failed' }

$dry = Join-Path $runtime 'cells/openbmi_mi/fold0_seed0.json'
if (!(Test-Path -LiteralPath $dry)) {
    & $python -u run_p2_selection_factor.py --stage dry-run --task OpenBMI_MI --fold 0 1>> (Join-Path $logDir 'queue.out.log') 2>> (Join-Path $logDir 'queue.err.log')
    if ($LASTEXITCODE -ne 0 -or !(Test-Path -LiteralPath $dry)) { throw 'P2 one-cell dry-run failed' }
}
Set-Content -LiteralPath (Join-Path $runtime 'DRY_RUN_PASS.txt') -Value (Get-Date -Format o)

foreach ($task in @('OpenBMI_MI', 'OpenBMI_ERP', 'OpenBMI_SSVEP', 'WBCIC_MI')) {
    foreach ($fold in 0..4) {
        $cell = Join-Path $runtime ("cells/{0}/fold{1}_seed0.json" -f $task.ToLowerInvariant(), $fold)
        if (Test-Path -LiteralPath $cell) { continue }
        $ok = $false
        foreach ($attempt in 1..3) {
            $saved = $ErrorActionPreference
            $ErrorActionPreference = 'Continue'
            & $python -u run_p2_selection_factor.py --stage cell --task $task --fold $fold 1>> (Join-Path $logDir 'queue.out.log') 2>> (Join-Path $logDir 'queue.err.log')
            $exit = $LASTEXITCODE
            $ErrorActionPreference = $saved
            if ($exit -eq 0 -and (Test-Path -LiteralPath $cell)) { $ok = $true; break }
            Add-Content -LiteralPath (Join-Path $logDir 'queue.err.log') -Value ("RETRY {0}/fold{1} attempt={2} exit={3}" -f $task, $fold, $attempt, $exit)
            Start-Sleep -Seconds 3
        }
        if (!$ok) { throw "P2 cell failed: $task/fold$fold" }
    }
}
& $python -u run_p2_selection_factor.py --stage aggregate 1>> (Join-Path $logDir 'queue.out.log') 2>> (Join-Path $logDir 'queue.err.log')
if ($LASTEXITCODE -ne 0) { throw 'P2 aggregation failed' }
Set-Content -LiteralPath (Join-Path $runtime 'COMPLETE.txt') -Value (Get-Date -Format o)
