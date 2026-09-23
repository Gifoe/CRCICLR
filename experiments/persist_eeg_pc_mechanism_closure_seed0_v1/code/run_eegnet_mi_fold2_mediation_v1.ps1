$ErrorActionPreference='Stop'
$repo='D:\nips-temp\TotalP\P1\CRCICLR_PERSIST_INCREMENTAL_VALUE_V1'
$exp=Join-Path $repo 'experiments\persist_eeg_pc_mechanism_closure_seed0_v1'
$runtime='D:\nips-temp\TotalP\P1\pc_mechanism_closure_runtime'
$cell=Join-Path $runtime 'cells\eegnet\openbmi_mi\fold3_seed0'
$entry=Join-Path $exp 'code\run_mediation_cell_v2.py'
$log=Join-Path $runtime 'scheduled_mediation_cell_EEGNet_OpenBMI_MI_f3_v1.log'
$env:PERSIST_SOURCE_REPO=$repo
$env:COUPLING_RUNTIME='D:\nips-temp\TotalP\P1\protected_complement_coupling_runtime'
$env:MECHANISM_RUNTIME=$runtime
$env:PYTHONUNBUFFERED='1'
$python='E:\Anaconda\envs\persist_stable_251\python.exe'
try {
  if(Test-Path $log){throw 'mediation log already exists'}
  if((Test-Path (Join-Path $cell 'MEDIATION_V2.json')) -or (Test-Path (Join-Path $cell 'MEDIATION_V2_FAIL_CLOSED.json')) -or (Test-Path (Join-Path $cell 'mediation_v2'))){throw 'mediation evidence already exists'}
  if((Get-FileHash $entry -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'b00479f7cdf6e3ea5b118773e6d3382e6fb29e9b39beb58b4842df10a54e3175'){throw 'mediation source SHA mismatch'}
  if((Get-FileHash (Join-Path $exp 'protocol\MECHANISM_CLOSURE_ANALYSIS_LOCK.md') -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'ac12d33f7d30b2c846ec4c4fb07f8513bf25e372943a767b134883bf095a7c88'){throw 'analysis lock SHA mismatch'}
  if((Get-FileHash (Join-Path $exp 'protocol\FINAL_PROJECTOR_AMENDMENT.md') -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'aec1e604d38f1901bd9eca7e072e30a3d496a28061a726ebbe63133259b0e474'){throw 'amendment SHA mismatch'}
  while($true){
    $free=(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB
    $gpu=[int](([string]((& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)|Select-Object -First 1)).Trim())
    if($free -ge 40 -and $gpu -ge 16000){break}
    Start-Sleep -Seconds 30
  }
  "MEDIATION_F3_START free_gb=$([math]::Round($free,2)) gpu_mib=$gpu utc=$([DateTime]::UtcNow.ToString('o'))"|Out-File $log -Encoding utf8
  $cmd='"'+$python+'" -u "'+$entry+'" --model EEGNet --task OpenBMI_MI --fold 3 >> "'+$log+'" 2>&1'
  & cmd.exe /d /c $cmd
  if($LASTEXITCODE -ne 0){throw "mediation exited $LASTEXITCODE"}
  $m=Get-Content (Join-Path $cell 'MEDIATION_V2.json') -Raw|ConvertFrom-Json
  if($m.status -ne 'MEDIATION_PHASE_COMPLETE_PENDING_OTHER_REQUIRED_ANALYSES' -or $m.model -ne 'EEGNet' -or $m.task -ne 'OpenBMI_MI' -or [int]$m.fold -ne 3 -or [int]$m.stage_count -ne 4 -or $m.final_heldout_accessed -ne $false){throw 'mediation output invalid'}
  "MEDIATION_F3_COMPLETE utc=$([DateTime]::UtcNow.ToString('o'))"|Out-File $log -Append -Encoding utf8
  exit 0
} catch {
  if(Test-Path $log){"MEDIATION_F3_EXCEPTION $_ utc=$([DateTime]::UtcNow.ToString('o'))"|Out-File $log -Append -Encoding utf8}
  exit 1
}
