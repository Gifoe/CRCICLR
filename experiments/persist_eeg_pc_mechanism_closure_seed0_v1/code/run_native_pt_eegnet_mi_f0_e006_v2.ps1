<# One checkpoint-native TRAIN-only P_t discovery, independently provenanced. #>
$ErrorActionPreference='Stop'
$repo='D:\nips-temp\TotalP\P1\CRCICLR_PERSIST_INCREMENTAL_VALUE_V1'
$exp=Join-Path $repo 'experiments\persist_eeg_pc_mechanism_closure_seed0_v1'
$runtime='D:\nips-temp\TotalP\P1\pc_mechanism_closure_runtime'
$cell=Join-Path $runtime 'cells\eegnet\openbmi_mi\fold0_seed0'
$entry=Join-Path $exp 'code\audit_native_pt_eegnet_v2.py'
$result=Join-Path $cell 'checkpoint_native_pt_v2\epoch_006.json'
$failure=Join-Path $cell 'checkpoint_native_pt_v2\epoch_006.FAIL_CLOSED.json'
$endpoint=Join-Path $cell 'REPLICA_ENDPOINT_AUDIT_V1.json'
$log=Join-Path $runtime 'scheduled_native_pt_EEGNet_OpenBMI_MI_f0_e006_v2.log'
$env:PERSIST_SOURCE_REPO=$repo
$env:MECHANISM_RUNTIME=$runtime
$env:COUPLING_RUNTIME='D:\nips-temp\TotalP\P1\protected_complement_coupling_runtime'
$env:SEVEN_RUNTIME='D:\nips-temp\TotalP\P1\seven_backbone_fourtask_3seed_runtime'
$env:OFFICIAL_BACKBONE_ROOT=Join-Path $env:SEVEN_RUNTIME 'official'
$env:PYTHONUNBUFFERED='1'
try {
  if((Test-Path -LiteralPath $log) -or (Test-Path -LiteralPath $result) -or (Test-Path -LiteralPath $failure)) {throw 'native P_t evidence already exists'}
  $audit=Get-Content -LiteralPath $endpoint -Raw|ConvertFrom-Json
  if($audit.status -ne 'REPLICA_ENDPOINTS_AUDITED_PENDING_NATIVE_P_AND_MECHANISM' -or $audit.final_heldout_accessed -ne $false) {throw 'endpoint audit invalid'}
  if((Get-FileHash -LiteralPath $entry -Algorithm SHA256).Hash.ToLowerInvariant() -ne '88a4b8482e155ab11c2ccd222590deca52b3ff7f0824687822d8bbbcfca9466f') {throw 'native P_t source SHA mismatch'}
  if((Get-FileHash -LiteralPath (Join-Path $exp 'protocol\MECHANISM_CLOSURE_ANALYSIS_LOCK.md') -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'ac12d33f7d30b2c846ec4c4fb07f8513bf25e372943a767b134883bf095a7c88') {throw 'analysis lock SHA mismatch'}
  if((Get-FileHash -LiteralPath (Join-Path $exp 'protocol\FINAL_PROJECTOR_AMENDMENT.md') -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'aec1e604d38f1901bd9eca7e072e30a3d496a28061a726ebbe63133259b0e474') {throw 'final-projector amendment SHA mismatch'}
  $freeGB=(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB
  if($freeGB -lt 40) {throw "insufficient physical free RAM before launch: $freeGB GB"}
  "NATIVE_PT_EEGNET_MI_F0_E006_V2_START utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -LiteralPath $log -Encoding utf8
  $python='E:\Anaconda\envs\persist_stable_251\python.exe'
  $command='"'+$python+'" -u "'+$entry+'" --task OpenBMI_MI --fold 0 --epoch 6 >> "'+$log+'" 2>&1'
  & cmd.exe /d /c $command
  if($LASTEXITCODE -ne 0) {throw "native P_t exited $LASTEXITCODE"}
  "NATIVE_PT_EEGNET_MI_F0_E006_V2_END exit=0 utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -LiteralPath $log -Append -Encoding utf8
  exit 0
} catch {
  "NATIVE_PT_EEGNET_MI_F0_E006_V2_EXCEPTION $_" | Out-File -LiteralPath $log -Append -Encoding utf8
  exit 1
}
