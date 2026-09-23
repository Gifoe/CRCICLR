<# Serial TRAIN-only checkpoint-native P_t discovery for the fixed remaining schedule. #>
$ErrorActionPreference='Stop'
$repo='D:\nips-temp\TotalP\P1\CRCICLR_PERSIST_INCREMENTAL_VALUE_V1'
$exp=Join-Path $repo 'experiments\persist_eeg_pc_mechanism_closure_seed0_v1'
$runtime='D:\nips-temp\TotalP\P1\pc_mechanism_closure_runtime'
$cell=Join-Path $runtime 'cells\eegnet\openbmi_mi\fold0_seed0'
$entry=Join-Path $exp 'code\audit_native_pt_eegnet_v3.py'
$endpoint=Join-Path $cell 'REPLICA_ENDPOINT_AUDIT_V1.json'
$env:PERSIST_SOURCE_REPO=$repo
$env:MECHANISM_RUNTIME=$runtime
$env:COUPLING_RUNTIME='D:\nips-temp\TotalP\P1\protected_complement_coupling_runtime'
$env:SEVEN_RUNTIME='D:\nips-temp\TotalP\P1\seven_backbone_fourtask_3seed_runtime'
$env:OFFICIAL_BACKBONE_ROOT=Join-Path $env:SEVEN_RUNTIME 'official'
$env:PYTHONUNBUFFERED='1'
$python='E:\Anaconda\envs\persist_stable_251\python.exe'
try {
  $audit=Get-Content -LiteralPath $endpoint -Raw|ConvertFrom-Json
  if($audit.status -ne 'REPLICA_ENDPOINTS_AUDITED_PENDING_NATIVE_P_AND_MECHANISM' -or $audit.final_heldout_accessed -ne $false) {throw 'endpoint audit invalid'}
  if((Get-FileHash -LiteralPath $entry -Algorithm SHA256).Hash.ToLowerInvariant() -ne '14ade22a3391d3461fffeb0a3da43d896633b03eedc4c4d89acec90c0af72d5a') {throw 'native P_t source SHA mismatch'}
  if((Get-FileHash -LiteralPath (Join-Path $exp 'protocol\MECHANISM_CLOSURE_ANALYSIS_LOCK.md') -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'ac12d33f7d30b2c846ec4c4fb07f8513bf25e372943a767b134883bf095a7c88') {throw 'analysis lock SHA mismatch'}
  if((Get-FileHash -LiteralPath (Join-Path $exp 'protocol\FINAL_PROJECTOR_AMENDMENT.md') -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'aec1e604d38f1901bd9eca7e072e30a3d496a28061a726ebbe63133259b0e474') {throw 'final-projector amendment SHA mismatch'}
  foreach($epoch in @(15,30,45,49,60)) {
    $tag='{0:d3}' -f $epoch
    $result=Join-Path $cell "checkpoint_native_pt_v3\epoch_$tag.json"
    $failure=Join-Path $cell "checkpoint_native_pt_v3\epoch_$tag.FAIL_CLOSED.json"
    $log=Join-Path $runtime "scheduled_native_pt_EEGNet_OpenBMI_MI_f0_e${tag}_v3.log"
    if((Test-Path -LiteralPath $log) -or (Test-Path -LiteralPath $result) -or (Test-Path -LiteralPath $failure)) {throw "native P_t terminal evidence already exists for epoch $epoch"}
    $freeGB=(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB
    if($freeGB -lt 40) {throw "insufficient physical free RAM before epoch $epoch : $freeGB GB"}
    $gpuFree=[int]((& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | Select-Object -First 1).Trim())
    if($gpuFree -lt 16000) {throw "insufficient GPU free memory before epoch $epoch : $gpuFree MiB"}
    "NATIVE_PT_EEGNET_MI_F0_E${tag}_V3_START utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -LiteralPath $log -Encoding utf8
    $command='"'+$python+'" -u "'+$entry+'" --task OpenBMI_MI --fold 0 --epoch '+$epoch+' >> "'+$log+'" 2>&1'
    & cmd.exe /d /c $command
    if($LASTEXITCODE -ne 0) {throw "native P_t epoch $epoch exited $LASTEXITCODE"}
    $row=Get-Content -LiteralPath $result -Raw|ConvertFrom-Json
    if($row.status -ne 'NATIVE_CHECKPOINT_PT_COMPLETE_PENDING_BACKTRACE_AND_MECHANISM' -or $row.final_heldout_accessed -ne $false -or $row.outer_development_accessed -ne $false -or [int]$row.epoch -ne $epoch) {throw "native P_t epoch $epoch result invalid"}
    "NATIVE_PT_EEGNET_MI_F0_E${tag}_V3_END exit=0 utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -LiteralPath $log -Append -Encoding utf8
  }
  exit 0
} catch {
  $summary=Join-Path $runtime 'scheduled_native_pt_EEGNet_OpenBMI_MI_f0_remaining_v1.FAIL_CLOSED.log'
  "NATIVE_PT_REMAINING_V1_EXCEPTION utc=$([DateTime]::UtcNow.ToString('o')) $_" | Out-File -LiteralPath $summary -Append -Encoding utf8
  exit 1
}
