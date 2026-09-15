$ErrorActionPreference = 'Stop'
$work = 'D:/nips-temp/TotalP/P1/CRCICLR_ALL_WBCIC_TRUE_OUTER_WORK'
$exp = Join-Path $work 'experiments/persist_eeg_all_backbone_wbcic_true_outer_v1'
$python = 'E:/Anaconda/envs/persist_stable_251/python.exe'
$runner = Join-Path $exp 'code/run_all_wbcic_true_outer.py'
$log = 'D:/nips-temp/TotalP/P1/baseline3_runtime/all_wbcic_true_outer.evaluate.log'
$err = 'D:/nips-temp/TotalP/P1/baseline3_runtime/all_wbcic_true_outer.evaluate.err.log'
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
$env:SGN_RUNTIME = 'D:/nips-temp/TotalP/P1/baseline3_runtime/sgn'
Set-Location $exp
$savedPreference = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
& $python -u $runner --stage evaluate 1>> $log 2>> $err
$evaluationExitCode = $LASTEXITCODE
$ErrorActionPreference = $savedPreference
if ($evaluationExitCode -ne 0) { throw "true outer evaluator failed: $evaluationExitCode" }
Set-Content -LiteralPath (Join-Path $exp 'TRUE_OUTER_EVALUATION_FINISHED.txt') -Value (Get-Date -Format o)
