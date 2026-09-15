$ErrorActionPreference = 'Stop'

$work = 'D:/nips-temp/TotalP/P1/CRCICLR_ALL_WBCIC_TRUE_OUTER_WORK'
$exp = Join-Path $work 'experiments/persist_eeg_all_backbone_wbcic_true_outer_v1'
$runtime = 'D:/nips-temp/TotalP/P1/baseline3_runtime/sgn'
$marker = Join-Path $runtime 'WBCIC_SEED0_TRAINING_COMPLETE.txt'
$python = 'E:/Anaconda/envs/persist_stable_251/python.exe'
$runner = Join-Path $exp 'code/run_all_wbcic_true_outer.py'
$log = 'D:/nips-temp/TotalP/P1/baseline3_runtime/all_wbcic_true_outer.log'

function Invoke-GitPush([int]$Attempts, [int]$DelaySeconds) {
    foreach ($attempt in 1..$Attempts) {
        git -C $work push origin 'codex/persist-eeg-all-backbone-wbcic-true-outer-v1' *>> $log
        if ($LASTEXITCODE -eq 0) { return $true }
        Add-Content -LiteralPath $log -Value "GITHUB_PUSH_RETRY $attempt/$Attempts"
        if ($attempt -lt $Attempts) { Start-Sleep -Seconds $DelaySeconds }
    }
    return $false
}

while (!(Test-Path -LiteralPath $marker)) {
    Start-Sleep -Seconds 30
}

$env:SEVEN_REPO = 'D:/nips-temp/TotalP/P1/CRCICLR_BACKBONE_GEN_WORK'
$env:SEVEN_RUNTIME = 'D:/nips-temp/TotalP/P1/seven_backbone_fourtask_3seed_runtime'
$env:MODERN_REPO = $env:SEVEN_REPO
$env:TASK_GENERALITY_REPO = $env:SEVEN_REPO
$env:OFFICIAL_BACKBONE_ROOT = 'D:/nips-temp/TotalP/P1/seven_backbone_fourtask_3seed_runtime/official'
$env:FULL_WBCIC_CACHE = 'D:/nips-temp/TotalP/P1/CRCICLR_SOURCE_ONLY_DIAGNOSTIC/experiments/persist_eeg_wbcic_independent_replication_v1/runtime/cache/wbcic_epochs'
$env:TRUE_OUTER_WBCIC_CACHE = 'D:/nips-temp/TotalP/P2/wbcic_outer_cache/wbcic_epochs'
$env:MODERNTCN_CODE = 'D:/nips-temp/TotalP/P1/CRCICLR_MODERNTCN_FINAL/experiments/persist_eeg_moderntcn_4task_3seed_final_v1/code'
$env:MODERNTCN_RUNTIME = 'D:/nips-temp/TotalP/P1/baseline3_runtime/moderntcn'
$env:MEDFORMER_CODE = 'D:/nips-temp/TotalP/P1/CRCICLR_MEDFORMER_FINAL/experiments/persist_eeg_medformer_4task_3seed_final_v1/code'
$env:MEDFORMER_RUNTIME = 'D:/nips-temp/TotalP/P1/baseline3_runtime/medformer'
$env:SGN_CODE = 'D:/nips-temp/TotalP/P1/CRCICLR_SGN_FINAL/experiments/persist_eeg_sgn_4task_3seed_final_v1/code'
$env:SGN_RUNTIME = $runtime

Set-Location $exp
$lock = Join-Path $exp 'protocol/TRUE_OUTER_EVALUATION_LOCK.json'
if (!(Test-Path -LiteralPath $lock)) {
    $savedPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $python -u $runner --stage prelock *>> $log
    $prelockExitCode = $LASTEXITCODE
    $ErrorActionPreference = $savedPreference
    if ($prelockExitCode -ne 0) { throw "true-outer prelock failed: $prelockExitCode" }
    git -C $work add -- 'experiments/persist_eeg_all_backbone_wbcic_true_outer_v1/protocol/TRUE_OUTER_EVALUATION_LOCK.json' 'experiments/persist_eeg_all_backbone_wbcic_true_outer_v1/protocol/TRUE_OUTER_EVALUATION_LOCK.sha256'
    git -C $work commit -m 'lock all WBCIC true-outer checkpoints before evaluation'
    if ($LASTEXITCODE -ne 0) { throw 'failed to commit true-outer lock' }
    if (!(Invoke-GitPush 3 10)) {
        Add-Content -LiteralPath $log -Value 'LOCK_COMMITTED_LOCALLY_GITHUB_TEMPORARILY_UNREACHABLE'
    }
}

$savedPreference = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
& $python -u $runner --stage evaluate *>> $log
$evaluationExitCode = $LASTEXITCODE
$ErrorActionPreference = $savedPreference
if ($evaluationExitCode -ne 0) { throw "true-outer evaluation failed: $evaluationExitCode" }

$resultPaths = @(
    'experiments/persist_eeg_all_backbone_wbcic_true_outer_v1/TRUE_OUTER_RESULTS.md',
    'experiments/persist_eeg_all_backbone_wbcic_true_outer_v1/outputs/TRUE_OUTER_REPLICATE_RESULTS.csv',
    'experiments/persist_eeg_all_backbone_wbcic_true_outer_v1/outputs/TRUE_OUTER_SUBJECT_RESULTS.csv',
    'experiments/persist_eeg_all_backbone_wbcic_true_outer_v1/outputs/TRUE_OUTER_MODEL_RESULTS.csv',
    'experiments/persist_eeg_all_backbone_wbcic_true_outer_v1/outputs/RUN_METADATA.json'
)
git -C $work add -- $resultPaths
git -C $work commit -m 'add all-backbone WBCIC true-outer results'
if ($LASTEXITCODE -ne 0) { throw 'failed to commit true-outer results' }
Set-Content -LiteralPath (Join-Path $exp 'TRUE_OUTER_COMPLETE.txt') -Value (Get-Date -Format o)
if (!(Invoke-GitPush 20 60)) {
    throw 'results complete and committed locally, but GitHub remained unreachable after retries'
}
