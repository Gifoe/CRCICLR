$ErrorActionPreference = 'Stop'
$repo = 'D:\nips-temp\TotalP\P1\CRCICLR_PERSIST_INCREMENTAL_VALUE_V1'
$experiment = Join-Path $repo 'experiments\persist_eeg_pc_mechanism_closure_seed0_v1'
$runtime = 'D:\nips-temp\TotalP\P1\pc_mechanism_closure_runtime'
$entry = Join-Path $experiment 'code\replay_eegnet_v2.py'
$cell = Join-Path $runtime 'cells\eegnet\openbmi_mi\fold3_seed0'
$replay = Join-Path $cell 'replay_v2'
$log = Join-Path $runtime 'scheduled_replay_EEGNet_OpenBMI_MI_f3_v2.log'
$env:PERSIST_SOURCE_REPO = $repo
$env:COUPLING_RUNTIME = 'D:\nips-temp\TotalP\P1\protected_complement_coupling_runtime'
$env:MECHANISM_RUNTIME = $runtime
$env:SEVEN_RUNTIME = 'D:\nips-temp\TotalP\P1\seven_backbone_fourtask_3seed_runtime'
$env:OFFICIAL_BACKBONE_ROOT = Join-Path $env:SEVEN_RUNTIME 'official'
$env:PYTHONUNBUFFERED = '1'
try {
  if ((Test-Path -LiteralPath $log) -or (Test-Path -LiteralPath $replay)) { throw 'replay evidence exists' }
  if ((Get-FileHash -LiteralPath $entry -Algorithm SHA256).Hash.ToLowerInvariant() -ne '944dc5ad03dc536870bd3338d5ee55069149e14b05a466ef175f75b59c18cc3a') { throw 'replay source SHA mismatch' }
  if ((Get-FileHash -LiteralPath (Join-Path $experiment 'protocol\MECHANISM_CLOSURE_ANALYSIS_LOCK.md') -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'ac12d33f7d30b2c846ec4c4fb07f8513bf25e372943a767b134883bf095a7c88') { throw 'analysis lock SHA mismatch' }
  while (-not (Test-Path -LiteralPath (Join-Path $cell 'DIRECTIONAL_V4.json'))) {
    if (Test-Path -LiteralPath (Join-Path $cell 'DIRECTIONAL_V4_FAIL_CLOSED.json')) { throw 'directional prerequisite failed closed' }
    Start-Sleep -Seconds 20
  }
  $directional = Get-Content -LiteralPath (Join-Path $cell 'DIRECTIONAL_V4.json') -Raw | ConvertFrom-Json
  if ($directional.status -ne 'DIRECTIONAL_PHASE_COMPLETE_PENDING_OTHER_REQUIRED_ANALYSES' -or
      $directional.stage_count -ne 5 -or $directional.random_draws_per_stage -ne 20 -or
      $directional.final_heldout_accessed -ne $false) { throw 'directional prerequisite invalid' }
  $gate = Get-Content -LiteralPath (Join-Path $experiment 'gate\GATE_AUDIT.json') -Raw | ConvertFrom-Json
  if ($gate.status -ne 'PASSED' -or $gate.final_heldout_accessed -ne $false) { throw 'upstream gate invalid' }
  while ($true) {
    $freeGb = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB
    $gpuMiB = [int](([string]((& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits) | Select-Object -First 1)).Trim())
    if ($freeGb -ge 40 -and $gpuMiB -ge 16000) { break }
    Start-Sleep -Seconds 30
  }
  "REPLAY_EEGNET_MI_F3_START free_gb=$([Math]::Round($freeGb,2)) gpu_mib=$gpuMiB utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -LiteralPath $log -Encoding utf8
  $python = 'E:\Anaconda\envs\persist_stable_251\python.exe'
  $command = '"' + $python + '" -u "' + $entry + '" --task OpenBMI_MI --fold 3 >> "' + $log + '" 2>&1'
  & cmd.exe /d /c $command
  if ($LASTEXITCODE -ne 0) { throw "replay exited $LASTEXITCODE" }
  $result = Get-Content -LiteralPath (Join-Path $replay 'REPLAY_AUDIT.json') -Raw | ConvertFrom-Json
  if ($result.status -ne 'REPLAY_COMPLETE' -or $result.epochs_saved -ne 60 -or
      $result.final_heldout_accessed -ne $false -or
      $result.trajectory_provenance -ne 'RETRAINED_REPLICA_TRAJECTORY') { throw 'replay result invalid' }
  "REPLAY_EEGNET_MI_F3_END exit=0 utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -LiteralPath $log -Append -Encoding utf8
  exit 0
} catch {
  if (Test-Path -LiteralPath $log) { "REPLAY_EEGNET_MI_F3_EXCEPTION $_" | Out-File -LiteralPath $log -Append -Encoding utf8 }
  exit 1
}
