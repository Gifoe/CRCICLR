<# Frozen historical final-P target recovered from TRAIN-only replica representations. #>
$ErrorActionPreference='Stop'
$repo='D:\nips-temp\TotalP\P1\CRCICLR_PERSIST_INCREMENTAL_VALUE_V1'
$exp=Join-Path $repo 'experiments\persist_eeg_pc_mechanism_closure_seed0_v1'
$runtime='D:\nips-temp\TotalP\P1\pc_mechanism_closure_runtime'
$cell=Join-Path $runtime 'cells\eegnet\openbmi_mi\fold0_seed0'
$entry=Join-Path $exp 'code\audit_final_p_backtrace_eegnet_v1.py'
$result=Join-Path $cell 'FINAL_P_BACKTRACE_V1.json'
$failure=Join-Path $cell 'FINAL_P_BACKTRACE_V1.FAIL_CLOSED.json'
$log=Join-Path $runtime 'scheduled_final_p_backtrace_EEGNet_OpenBMI_MI_f0_v1.log'
$env:PERSIST_SOURCE_REPO=$repo
$env:MECHANISM_RUNTIME=$runtime
$env:COUPLING_RUNTIME='D:\nips-temp\TotalP\P1\protected_complement_coupling_runtime'
$env:SEVEN_RUNTIME='D:\nips-temp\TotalP\P1\seven_backbone_fourtask_3seed_runtime'
$env:OFFICIAL_BACKBONE_ROOT=Join-Path $env:SEVEN_RUNTIME 'official'
$env:PYTHONUNBUFFERED='1'
try {
  if((Test-Path -LiteralPath $log) -or (Test-Path -LiteralPath $result) -or (Test-Path -LiteralPath $failure)) {throw 'final-P backtrace terminal evidence exists'}
  if((Get-FileHash -LiteralPath $entry -Algorithm SHA256).Hash.ToLowerInvariant() -ne '7d7f2032f554ba375b8ebad91f0adfe0699aad8215db29d38e104d5dac0cdb75') {throw 'backtrace source SHA mismatch'}
  if((Get-FileHash -LiteralPath (Join-Path $exp 'protocol\MECHANISM_CLOSURE_ANALYSIS_LOCK.md') -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'ac12d33f7d30b2c846ec4c4fb07f8513bf25e372943a767b134883bf095a7c88') {throw 'analysis lock SHA mismatch'}
  if((Get-FileHash -LiteralPath (Join-Path $exp 'protocol\FINAL_PROJECTOR_AMENDMENT.md') -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'aec1e604d38f1901bd9eca7e072e30a3d496a28061a726ebbe63133259b0e474') {throw 'final-projector amendment SHA mismatch'}
  foreach($epoch in @(6,15,30,45,49,60)) {
    $tag='{0:d3}' -f $epoch
    $p=Join-Path $cell "checkpoint_native_pt_v3\epoch_$tag.json"
    $row=Get-Content -LiteralPath $p -Raw|ConvertFrom-Json
    if($row.status -ne 'NATIVE_CHECKPOINT_PT_COMPLETE_PENDING_BACKTRACE_AND_MECHANISM' -or $row.final_heldout_accessed -ne $false -or $row.outer_development_accessed -ne $false) {throw "invalid native P_t at epoch $epoch"}
  }
  $freeGB=(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB
  if($freeGB -lt 40) {throw "insufficient physical free RAM: $freeGB GB"}
  $gpuFree=[int]((& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | Select-Object -First 1).Trim())
  if($gpuFree -lt 16000) {throw "insufficient GPU free memory: $gpuFree MiB"}
  "FINAL_P_BACKTRACE_EEGNET_MI_F0_V1_START utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -LiteralPath $log -Encoding utf8
  $python='E:\Anaconda\envs\persist_stable_251\python.exe'
  $command='"'+$python+'" -u "'+$entry+'" >> "'+$log+'" 2>&1'
  & cmd.exe /d /c $command
  if($LASTEXITCODE -ne 0) {throw "final-P backtrace exited $LASTEXITCODE"}
  $row=Get-Content -LiteralPath $result -Raw|ConvertFrom-Json
  if($row.status -ne 'FINAL_P_BACKTRACE_COMPLETE_REPLICA_ONLY' -or $row.final_heldout_accessed -ne $false -or $row.outer_development_accessed -ne $false -or @($row.checkpoints).Count -ne 6) {throw 'final-P backtrace result invalid'}
  "FINAL_P_BACKTRACE_EEGNET_MI_F0_V1_END exit=0 utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -LiteralPath $log -Append -Encoding utf8
  exit 0
} catch {
  "FINAL_P_BACKTRACE_EEGNET_MI_F0_V1_EXCEPTION $_" | Out-File -LiteralPath $log -Append -Encoding utf8
  exit 1
}
