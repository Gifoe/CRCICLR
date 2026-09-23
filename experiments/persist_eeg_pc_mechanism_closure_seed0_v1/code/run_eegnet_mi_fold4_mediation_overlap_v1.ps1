$ErrorActionPreference='Stop'
$repo='D:\nips-temp\TotalP\P1\CRCICLR_PERSIST_INCREMENTAL_VALUE_V1'
$exp=Join-Path $repo 'experiments\persist_eeg_pc_mechanism_closure_seed0_v1'
$runtime='D:\nips-temp\TotalP\P1\pc_mechanism_closure_runtime'
$cell=Join-Path $runtime 'cells\eegnet\openbmi_mi\fold4_seed0'
$fold3Cell=Join-Path $runtime 'cells\eegnet\openbmi_mi\fold3_seed0'
$entry=Join-Path $exp 'code\run_mediation_cell_v2.py'
$log=Join-Path $runtime 'scheduled_mediation_cell_EEGNet_OpenBMI_MI_f4_overlap_v1.log'
$env:PERSIST_SOURCE_REPO=$repo
$env:COUPLING_RUNTIME='D:\nips-temp\TotalP\P1\protected_complement_coupling_runtime'
$env:MECHANISM_RUNTIME=$runtime
$env:PYTHONUNBUFFERED='1'
$python='E:\Anaconda\envs\persist_stable_251\python.exe'
$sourceSha='b00479f7cdf6e3ea5b118773e6d3382e6fb29e9b39beb58b4842df10a54e3175'
$analysisSha='ac12d33f7d30b2c846ec4c4fb07f8513bf25e372943a767b134883bf095a7c88'
$amendmentSha='aec1e604d38f1901bd9eca7e072e30a3d496a28061a726ebbe63133259b0e474'

function Get-Resources {
  $free=(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB
  $gpu=[int](([string]((& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)|Select-Object -First 1)).Trim())
  [pscustomobject]@{FreeGB=$free;GpuMiB=$gpu}
}

function Get-Fold3HeavyPid {
  $p=Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'python.exe' -and $_.CommandLine -and
    $_.CommandLine -match '(run_directional_cell_v4\.py|replay_eegnet_v2\.py|audit_checkpoint_mechanism_eegnet_v2\.py)' -and
    $_.CommandLine -match '(--fold\s+3|fold3_seed0)'
  } | Select-Object -First 1
  if($p){return [int]$p.ProcessId}
  return 0
}

try {
  if(Test-Path $log){throw 'overlap mediation log already exists'}
  if((Test-Path (Join-Path $cell 'MEDIATION_V2.json')) -or
     (Test-Path (Join-Path $cell 'MEDIATION_V2_FAIL_CLOSED.json')) -or
     (Test-Path (Join-Path $cell 'mediation_v2'))){throw 'fold4 mediation evidence already exists'}
  if((Get-FileHash $entry -Algorithm SHA256).Hash.ToLowerInvariant() -ne $sourceSha){throw 'mediation source SHA mismatch'}
  if((Get-FileHash (Join-Path $exp 'protocol\MECHANISM_CLOSURE_ANALYSIS_LOCK.md') -Algorithm SHA256).Hash.ToLowerInvariant() -ne $analysisSha){throw 'analysis lock SHA mismatch'}
  if((Get-FileHash (Join-Path $exp 'protocol\FINAL_PROJECTOR_AMENDMENT.md') -Algorithm SHA256).Hash.ToLowerInvariant() -ne $amendmentSha){throw 'amendment SHA mismatch'}

  New-Item -ItemType File -Path $log -ErrorAction Stop | Out-Null
  "F4_OVERLAP_QUEUE_START utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File $log -Append -Encoding utf8
  while($true){
    $r1=Get-Resources
    $pid1=Get-Fold3HeavyPid
    if($pid1 -gt 0 -and $r1.FreeGB -ge 30 -and $r1.GpuMiB -ge 18000){
      Start-Sleep -Seconds 30
      $r2=Get-Resources
      $pid2=Get-Fold3HeavyPid
      if($pid2 -eq $pid1 -and $r2.FreeGB -ge 30 -and $r2.GpuMiB -ge 18000){
        "F4_OVERLAP_GATE heavy_pid=$pid2 free_gb=$([math]::Round($r2.FreeGB,2)) gpu_mib=$($r2.GpuMiB) utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File $log -Append -Encoding utf8
        break
      }
    }
    if(Test-Path (Join-Path $fold3Cell 'CELL_COMPACT_PREVIEW_V3.json')){
      if($r1.FreeGB -ge 40 -and $r1.GpuMiB -ge 16000){
        "F4_SERIAL_FALLBACK_GATE free_gb=$([math]::Round($r1.FreeGB,2)) gpu_mib=$($r1.GpuMiB) utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File $log -Append -Encoding utf8
        break
      }
    }
    Start-Sleep -Seconds 20
  }

  $cmd='"'+$python+'" -u "'+$entry+'" --model EEGNet --task OpenBMI_MI --fold 4 >> "'+$log+'" 2>&1'
  & cmd.exe /d /c $cmd
  if($LASTEXITCODE -ne 0){throw "mediation exited $LASTEXITCODE"}
  $m=Get-Content (Join-Path $cell 'MEDIATION_V2.json') -Raw|ConvertFrom-Json
  if($m.status -ne 'MEDIATION_PHASE_COMPLETE_PENDING_OTHER_REQUIRED_ANALYSES' -or
     $m.model -ne 'EEGNet' -or $m.task -ne 'OpenBMI_MI' -or
     [int]$m.fold -ne 4 -or [int]$m.stage_count -ne 4 -or
     $m.final_heldout_accessed -ne $false){throw 'mediation output invalid'}
  "F4_OVERLAP_MEDIATION_COMPLETE utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File $log -Append -Encoding utf8
  exit 0
} catch {
  if(Test-Path $log){"F4_OVERLAP_MEDIATION_EXCEPTION $_ utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File $log -Append -Encoding utf8}
  exit 1
}
