$ErrorActionPreference = 'Stop'
$root = 'D:/nips-temp/TotalP/P1'
$work = "$root/CRCICLR_LITEBN_OUTER_HELDOUT_WORK"
$exp = "$work/experiments/persist_eeg_seven_backbone_litebn_seed0_outer_heldout_v1"
$code = "$exp/code"
$runtime = "$root/seven_backbone_fourtask_3seed_runtime"
$python = 'E:/Anaconda/envs/persist_stable_251/python.exe'
$task = 'PERSIST_LITEBN_SEED0_OUTER_HELDOUT_FINALIZE_V1'

function Write-Status([string]$state, [string]$detail) {
  @{ state=$state; detail=$detail; time=(Get-Date -Format o) } | ConvertTo-Json | Set-Content "$runtime/litebn_seed0_finalize_status.json"
}

$env:SEVEN_RUNTIME = $runtime
$env:SEVEN_REPO = $work
$env:MODERN_REPO = $work
$env:TASK_GENERALITY_REPO = $work
$env:FULL_OPENBMI_CACHE = "$root/persist_eeg_stage0_repo_full/outputs/persist_eeg_stage0/cache/openbmi"
$env:FULL_WBCIC_CACHE = "$root/CRCICLR_SOURCE_ONLY_DIAGNOSTIC/experiments/persist_eeg_wbcic_independent_replication_v1/runtime/cache/wbcic_epochs"
$env:PYTHONFAULTHANDLER = '1'
$env:PYTHONUNBUFFERED = '1'

try {
  Write-Status 'WAITING_FOR_WBCIC' 'waiting for five frozen LiteBN seed0 WBCIC cells'
  while ($true) {
    $s = Get-Content "$runtime/litebn_seed0_outer_heldout_status.json" -Raw | ConvertFrom-Json
    if ($s.state -eq 'COMPLETE') { break }
    if ($s.state -eq 'FAILED') { throw "WBCIC continuation failed at fold $($s.fold), exit $($s.exit_code)" }
    Start-Sleep -Seconds 15
  }
  Write-Status 'OUTER_LOCK' 'aggregating existing outer-development records and creating holdout lock'
  Push-Location $code
  & $python -u litebn_seed0_outer_heldout.py --stage outer-lock
  if ($LASTEXITCODE -ne 0) { throw 'outer-lock stage failed' }
  Pop-Location
  git -C $work add experiments/persist_eeg_seven_backbone_litebn_seed0_outer_heldout_v1/protocol experiments/persist_eeg_seven_backbone_litebn_seed0_outer_heldout_v1/outputs
  git -C $work commit -m 'Lock LiteBN seed0 outer results before heldout evaluation'
  $push = & git -C $work push origin codex/persist-eeg-seven-litebn-seed0-outer-heldout-v1 2>&1
  $push | Set-Content "$runtime/litebn_seed0_github_push_before_heldout.log"
  $pushed = $LASTEXITCODE -eq 0
  # The user explicitly authorized the V8 internal-holdout run. A transient
  # GitHub network failure is recorded; it must not silently be treated as a push.
  $detail = if($pushed){'lock committed and pushed'}else{'lock committed; GitHub push unavailable, recorded'}
  Write-Status 'HELDOUT_EVALUATING' $detail
  Push-Location $code
  & $python -u litebn_seed0_outer_heldout.py --stage heldout
  if ($LASTEXITCODE -ne 0) { throw 'heldout evaluation stage failed' }
  Pop-Location
  git -C $work add experiments/persist_eeg_seven_backbone_litebn_seed0_outer_heldout_v1
  git -C $work commit -m 'Add LiteBN seed0 outer and internal heldout results'
  $push2 = & git -C $work push origin codex/persist-eeg-seven-litebn-seed0-outer-heldout-v1 2>&1
  $push2 | Set-Content "$runtime/litebn_seed0_github_push_results.log"
  if ($LASTEXITCODE -ne 0) { Write-Status 'COMPLETE_LOCAL_PUSH_PENDING' 'results committed locally; GitHub unreachable' }
  else { Write-Status 'COMPLETE_AND_PUSHED' 'results committed and pushed' }
} catch {
  Write-Status 'FAILED' $_.Exception.Message
  throw
}
