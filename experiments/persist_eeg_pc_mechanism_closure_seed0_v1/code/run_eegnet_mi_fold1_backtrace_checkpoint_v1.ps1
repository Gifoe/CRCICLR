<# Wait for native P_t completion, then run final-P backtrace and checkpoint mechanisms. #>
$ErrorActionPreference = 'Stop'
$repo = 'D:\nips-temp\TotalP\P1\CRCICLR_PERSIST_INCREMENTAL_VALUE_V1'
$experiment = Join-Path $repo 'experiments\persist_eeg_pc_mechanism_closure_seed0_v1'
$runtime = 'D:\nips-temp\TotalP\P1\pc_mechanism_closure_runtime'
$cell = Join-Path $runtime 'cells\eegnet\openbmi_mi\fold3_seed0'
$backtraceEntry = Join-Path $experiment 'code\audit_final_p_backtrace_eegnet_v2.py'
$mechanismEntry = Join-Path $experiment 'code\audit_checkpoint_mechanism_eegnet_v2.py'
$mainLog = Join-Path $runtime 'scheduled_backtrace_checkpoint_EEGNet_OpenBMI_MI_f3_v2.log'
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
  if (Test-Path -LiteralPath $mainLog) { throw 'pipeline log already exists' }
  if ((Get-FileHash $backtraceEntry -Algorithm SHA256).Hash.ToLowerInvariant() -ne '8410b848359f5b6b8afa5df1cd30b5b70aefeaadd57ad2ce7f096cdf511d7149') { throw 'backtrace source SHA mismatch' }
  if ((Get-FileHash $mechanismEntry -Algorithm SHA256).Hash.ToLowerInvariant() -ne '097f6c25e200392359d630add0bfa7c36431bc3813234ade48771e481096a520') { throw 'mechanism source SHA mismatch' }
  if ((Get-FileHash (Join-Path $experiment 'protocol\MECHANISM_CLOSURE_ANALYSIS_LOCK.md') -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'ac12d33f7d30b2c846ec4c4fb07f8513bf25e372943a767b134883bf095a7c88') { throw 'analysis lock SHA mismatch' }
  if ((Get-FileHash (Join-Path $experiment 'protocol\FINAL_PROJECTOR_AMENDMENT.md') -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'aec1e604d38f1901bd9eca7e072e30a3d496a28061a726ebbe63133259b0e474') { throw 'amendment SHA mismatch' }
  "BACKTRACE_CHECKPOINT_F3_START utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -LiteralPath $mainLog -Encoding utf8
  $endpointPath = Join-Path $cell 'REPLICA_ENDPOINT_AUDIT_V1.json'
  while (-not (Test-Path $endpointPath)) { Start-Sleep -Seconds 15 }
  $endpoint = Get-Content $endpointPath -Raw | ConvertFrom-Json
  $epochs = @($endpoint.schedule | ForEach-Object { [int]$_.epoch })
  if ($endpoint.status -ne 'REPLICA_ENDPOINTS_AUDITED_PENDING_NATIVE_P_AND_MECHANISM' -or $endpoint.final_heldout_accessed -ne $false -or $epochs.Count -ne 6 -or ($epochs|Select-Object -Unique).Count -ne 6) { throw 'endpoint schedule invalid' }
  while ($true) {
    $valid = 0
    foreach($epoch in $epochs) {
      $tag='{0:d3}' -f $epoch
      $p=Join-Path $cell "checkpoint_native_pt_v3\epoch_$tag.json"
      if(Test-Path $p) {
        $j=Get-Content $p -Raw|ConvertFrom-Json
        if($j.status -eq 'NATIVE_CHECKPOINT_PT_COMPLETE_PENDING_BACKTRACE_AND_MECHANISM' -and [int]$j.epoch -eq $epoch -and $j.final_heldout_accessed -eq $false){$valid++}
      }
    }
    if($valid -eq 6){break}
    "WAIT_NATIVE_PT valid=$valid/6 utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -LiteralPath $mainLog -Append -Encoding utf8
    Start-Sleep -Seconds 15
  }
  $backtrace = Join-Path $cell 'FINAL_P_BACKTRACE_V1.json'
  if(-not (Test-Path $backtrace)) {
    if(Test-Path (Join-Path $cell 'FINAL_P_BACKTRACE_V1.FAIL_CLOSED.json')){throw 'backtrace failed closed'}
    Wait-Resources 'BACKTRACE'
    $phaseLog=Join-Path $runtime 'scheduled_final_p_backtrace_EEGNet_OpenBMI_MI_f3_v2.log'
    if(Test-Path $phaseLog){throw 'backtrace phase log already exists'}
    $cmd='"'+$python+'" -u "'+$backtraceEntry+'" --task OpenBMI_MI --fold 3 >> "'+$phaseLog+'" 2>&1'
    & cmd.exe /d /c $cmd
    if($LASTEXITCODE -ne 0){throw "backtrace exited $LASTEXITCODE"}
  }
  $bj=Get-Content $backtrace -Raw|ConvertFrom-Json
  if($bj.status -ne 'FINAL_P_BACKTRACE_COMPLETE_REPLICA_ONLY' -or $bj.final_heldout_accessed -ne $false -or @($bj.checkpoints).Count -ne 6){throw 'backtrace output invalid'}
  "BACKTRACE_COMPLETE utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -LiteralPath $mainLog -Append -Encoding utf8
  foreach($epoch in $epochs) {
    $tag='{0:d3}' -f $epoch
    $out=Join-Path $cell "checkpoint_mechanism_v1\epoch_$tag.json"
    if(-not (Test-Path $out)) {
      if(Test-Path (Join-Path $cell "checkpoint_mechanism_v1\epoch_$tag.FAIL_CLOSED.json")){throw "mechanism epoch $epoch failed closed"}
      Wait-Resources "MECHANISM_E$tag"
      $phaseLog=Join-Path $runtime "scheduled_checkpoint_mechanism_EEGNet_OpenBMI_MI_f3_e${tag}_v2.log"
      if(Test-Path $phaseLog){throw "mechanism phase log already exists at epoch $epoch"}
      $cmd='"'+$python+'" -u "'+$mechanismEntry+'" --task OpenBMI_MI --fold 3 --epoch '+$epoch+' >> "'+$phaseLog+'" 2>&1'
      & cmd.exe /d /c $cmd
      if($LASTEXITCODE -ne 0){throw "mechanism epoch $epoch exited $LASTEXITCODE"}
    }
    $j=Get-Content $out -Raw|ConvertFrom-Json
    if($j.status -ne 'CHECKPOINT_MECHANISM_COMPLETE_REPLICA_ONLY' -or [int]$j.epoch -ne $epoch -or $j.final_heldout_accessed -ne $false){throw "mechanism epoch $epoch invalid"}
    "MECHANISM_E${tag}_COMPLETE utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -LiteralPath $mainLog -Append -Encoding utf8
  }
  "BACKTRACE_CHECKPOINT_F3_COMPLETE utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -LiteralPath $mainLog -Append -Encoding utf8
  exit 0
} catch {
  if(Test-Path $mainLog){"BACKTRACE_CHECKPOINT_F3_EXCEPTION $_ utc=$([DateTime]::UtcNow.ToString('o'))"|Out-File -LiteralPath $mainLog -Append -Encoding utf8}
  exit 1
}
