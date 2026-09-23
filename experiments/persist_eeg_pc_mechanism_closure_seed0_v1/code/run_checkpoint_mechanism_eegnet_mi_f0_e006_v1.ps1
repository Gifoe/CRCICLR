<# One locked checkpoint-mechanism audit, with separate failure evidence. #>
$ErrorActionPreference='Stop'
$repo='D:\nips-temp\TotalP\P1\CRCICLR_PERSIST_INCREMENTAL_VALUE_V1'
$exp=Join-Path $repo 'experiments\persist_eeg_pc_mechanism_closure_seed0_v1'
$runtime='D:\nips-temp\TotalP\P1\pc_mechanism_closure_runtime'
$cell=Join-Path $runtime 'cells\eegnet\openbmi_mi\fold0_seed0'
$entry=Join-Path $exp 'code\audit_checkpoint_mechanism_eegnet_v1.py'
$result=Join-Path $cell 'checkpoint_mechanism_v1\epoch_006.json'
$failure=Join-Path $cell 'checkpoint_mechanism_v1\epoch_006.FAIL_CLOSED.json'
$log=Join-Path $runtime 'scheduled_checkpoint_mechanism_EEGNet_OpenBMI_MI_f0_e006_v1.log'
$env:PERSIST_SOURCE_REPO=$repo
$env:MECHANISM_RUNTIME=$runtime
$env:COUPLING_RUNTIME='D:\nips-temp\TotalP\P1\protected_complement_coupling_runtime'
$env:SEVEN_RUNTIME='D:\nips-temp\TotalP\P1\seven_backbone_fourtask_3seed_runtime'
$env:OFFICIAL_BACKBONE_ROOT=Join-Path $env:SEVEN_RUNTIME 'official'
$env:PYTHONUNBUFFERED='1'
try {
  if((Test-Path -LiteralPath $log) -or (Test-Path -LiteralPath $result) -or (Test-Path -LiteralPath $failure)) {throw 'checkpoint mechanism terminal evidence exists'}
  if((Get-FileHash -LiteralPath $entry -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'd86affcafcc306c67025f3d241ac5b18818f72261fc9ce0d5ebfcc564de385c0') {throw 'checkpoint mechanism source SHA mismatch'}
  if((Get-FileHash -LiteralPath (Join-Path $exp 'protocol\MECHANISM_CLOSURE_ANALYSIS_LOCK.md') -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'ac12d33f7d30b2c846ec4c4fb07f8513bf25e372943a767b134883bf095a7c88') {throw 'analysis lock SHA mismatch'}
  if((Get-FileHash -LiteralPath (Join-Path $exp 'protocol\FINAL_PROJECTOR_AMENDMENT.md') -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'aec1e604d38f1901bd9eca7e072e30a3d496a28061a726ebbe63133259b0e474') {throw 'final-projector amendment SHA mismatch'}
  if((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB -lt 40) {throw 'insufficient physical free RAM'}
  $gpuFree=[int]((& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | Select-Object -First 1).Trim())
  if($gpuFree -lt 16000) {throw "insufficient GPU free memory: $gpuFree MiB"}
  "CHECKPOINT_MECHANISM_EEGNET_MI_F0_E006_V1_START utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -LiteralPath $log -Encoding utf8
  $python='E:\Anaconda\envs\persist_stable_251\python.exe'
  $command='"'+$python+'" -u "'+$entry+'" --epoch 6 >> "'+$log+'" 2>&1'
  & cmd.exe /d /c $command
  if($LASTEXITCODE -ne 0) {throw "checkpoint mechanism exited $LASTEXITCODE"}
  $row=Get-Content -LiteralPath $result -Raw|ConvertFrom-Json
  if($row.status -ne 'CHECKPOINT_MECHANISM_COMPLETE_REPLICA_ONLY' -or $row.final_heldout_accessed -ne $false -or $row.outer_development_accessed -ne $false) {throw 'checkpoint mechanism result invalid'}
  "CHECKPOINT_MECHANISM_EEGNET_MI_F0_E006_V1_END exit=0 utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -LiteralPath $log -Append -Encoding utf8
  exit 0
} catch {
  "CHECKPOINT_MECHANISM_EEGNET_MI_F0_E006_V1_EXCEPTION $_" | Out-File -LiteralPath $log -Append -Encoding utf8
  exit 1
}
