$ErrorActionPreference = 'Stop'
$repo = 'D:\nips-temp\TotalP\P1\CRCICLR_PERSIST_INCREMENTAL_VALUE_V1'
$experiment = Join-Path $repo 'experiments\persist_eeg_pc_mechanism_closure_seed0_v1'
$runtime = 'D:\nips-temp\TotalP\P1\pc_mechanism_closure_runtime'
$cell = Join-Path $runtime 'cells\eegnet\openbmi_mi\fold3_seed0'
$nativeEntry = Join-Path $experiment 'code\audit_native_pt_eegnet_v3.py'
$mainLog = Join-Path $runtime 'scheduled_endpoint_native_EEGNet_OpenBMI_MI_f3_v4.log'
$env:PERSIST_SOURCE_REPO = $repo
$env:MECHANISM_RUNTIME = $runtime
$env:COUPLING_RUNTIME = 'D:\nips-temp\TotalP\P1\protected_complement_coupling_runtime'
$env:SEVEN_RUNTIME = 'D:\nips-temp\TotalP\P1\seven_backbone_fourtask_3seed_runtime'
$env:OFFICIAL_BACKBONE_ROOT = Join-Path $env:SEVEN_RUNTIME 'official'
$env:PYTHONUNBUFFERED = '1'
$python = 'E:\Anaconda\envs\persist_stable_251\python.exe'

function Wait-Resources {
  param([string]$Phase)
  while ($true) {
    $freeGb = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB
    $gpuMiB = [int](([string]((& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits) | Select-Object -First 1)).Trim())
    if ($freeGb -ge 40 -and $gpuMiB -ge 16000) {
      "$Phase RESOURCE_READY free_gb=$([Math]::Round($freeGb,2)) gpu_mib=$gpuMiB utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -LiteralPath $mainLog -Append -Encoding utf8
      return
    }
    "$Phase RESOURCE_WAIT free_gb=$([Math]::Round($freeGb,2)) gpu_mib=$gpuMiB utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -LiteralPath $mainLog -Append -Encoding utf8
    Start-Sleep -Seconds 30
  }
}

try {
  if (Test-Path -LiteralPath $mainLog) { throw 'v4 pipeline log already exists' }
  if ((Get-FileHash -LiteralPath $nativeEntry -Algorithm SHA256).Hash.ToLowerInvariant() -ne '14ade22a3391d3461fffeb0a3da43d896633b03eedc4c4d89acec90c0af72d5a') { throw 'native P_t source SHA mismatch' }
  if ((Get-FileHash -LiteralPath (Join-Path $experiment 'protocol\MECHANISM_CLOSURE_ANALYSIS_LOCK.md') -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'ac12d33f7d30b2c846ec4c4fb07f8513bf25e372943a767b134883bf095a7c88') { throw 'analysis lock SHA mismatch' }
  if ((Get-FileHash -LiteralPath (Join-Path $experiment 'protocol\FINAL_PROJECTOR_AMENDMENT.md') -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'aec1e604d38f1901bd9eca7e072e30a3d496a28061a726ebbe63133259b0e474') { throw 'amendment SHA mismatch' }
  $replayPath = Join-Path $cell 'replay_v2\REPLAY_AUDIT.json'
  if (-not (Test-Path -LiteralPath $replayPath)) { throw 'replay output missing' }
  $replay = Get-Content -LiteralPath $replayPath -Raw | ConvertFrom-Json
  if ($replay.status -ne 'REPLAY_COMPLETE' -or $replay.epochs_saved -ne 60 -or $replay.trajectory_provenance -ne 'RETRAINED_REPLICA_TRAJECTORY' -or $replay.final_heldout_accessed -ne $false) { throw 'replay prerequisite invalid' }
  $endpointPath = Join-Path $cell 'REPLICA_ENDPOINT_AUDIT_V1.json'
  if (-not (Test-Path -LiteralPath $endpointPath)) { throw 'endpoint audit missing after v3 evidence' }
  $endpointAudit = Get-Content -LiteralPath $endpointPath -Raw | ConvertFrom-Json
  if ($endpointAudit.status -ne 'REPLICA_ENDPOINTS_AUDITED_PENDING_NATIVE_P_AND_MECHANISM' -or
      $endpointAudit.final_heldout_accessed -ne $false -or @($endpointAudit.schedule).Count -ne 6) { throw 'endpoint audit invalid' }
  $epochs = @($endpointAudit.schedule | ForEach-Object { [int]$_.epoch })
  $required = @(6,15,30,45,60)
  if (($epochs | Select-Object -Unique).Count -ne 6 -or
      @($required | Where-Object { $_ -notin $epochs }).Count -ne 0) { throw 'endpoint-derived schedule set invalid' }
  "ENDPOINT_NATIVE_F3_V4_START schedule=$($epochs -join ',') v3_failure_preserved=true utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -LiteralPath $mainLog -Encoding utf8

  foreach ($epoch in $epochs) {
    $tag = '{0:d3}' -f $epoch
    $result = Join-Path $cell "checkpoint_native_pt_v3\epoch_$tag.json"
    $failure = Join-Path $cell "checkpoint_native_pt_v3\epoch_$tag.FAIL_CLOSED.json"
    if (-not (Test-Path -LiteralPath $result)) {
      if (Test-Path -LiteralPath $failure) { throw "native P_t legal epoch $epoch already failed closed" }
      Wait-Resources "NATIVE_PT_E$tag"
      $phaseLog = Join-Path $runtime "scheduled_native_pt_EEGNet_OpenBMI_MI_f3_e${tag}_v4.log"
      if (Test-Path -LiteralPath $phaseLog) { throw "v4 native P_t log already exists at legal epoch $epoch" }
      $cmd = '"' + $python + '" -u "' + $nativeEntry + '" --task OpenBMI_MI --fold 3 --epoch ' + $epoch + ' >> "' + $phaseLog + '" 2>&1'
      & cmd.exe /d /c $cmd
      if ($LASTEXITCODE -ne 0) { throw "native P_t epoch $epoch exited $LASTEXITCODE" }
    }
    $row = Get-Content -LiteralPath $result -Raw | ConvertFrom-Json
    if ($row.status -ne 'NATIVE_CHECKPOINT_PT_COMPLETE_PENDING_BACKTRACE_AND_MECHANISM' -or
        [int]$row.epoch -ne $epoch -or $row.outer_development_accessed -ne $false -or
        $row.final_heldout_accessed -ne $false) { throw "native P_t epoch $epoch invalid" }
    "NATIVE_PT_E${tag}_COMPLETE utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -LiteralPath $mainLog -Append -Encoding utf8
  }
  "ENDPOINT_NATIVE_F3_V4_COMPLETE utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -LiteralPath $mainLog -Append -Encoding utf8
  exit 0
} catch {
  if (Test-Path -LiteralPath $mainLog) { "ENDPOINT_NATIVE_F3_V4_EXCEPTION $_ utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -LiteralPath $mainLog -Append -Encoding utf8 }
  exit 1
}
