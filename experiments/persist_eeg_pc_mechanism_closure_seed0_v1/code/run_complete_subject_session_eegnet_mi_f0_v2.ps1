<# Derived G1/G2 V2 after all twenty fixed TRAIN random draws complete. #>
$ErrorActionPreference='Stop'
$repo='D:\nips-temp\TotalP\P1\CRCICLR_PERSIST_INCREMENTAL_VALUE_V1'
$exp=Join-Path $repo 'experiments\persist_eeg_pc_mechanism_closure_seed0_v1'
$runtime='D:\nips-temp\TotalP\P1\pc_mechanism_closure_runtime'
$cell=Join-Path $runtime 'cells\eegnet\openbmi_mi\fold0_seed0'
$entry=Join-Path $exp 'code\complete_subject_session_eegnet_mi_f0_v2.py'
$result=Join-Path $cell 'SUBJECT_SESSION_AUDIT_V2.json'
$failure=Join-Path $cell 'SUBJECT_SESSION_AUDIT_V2.FAIL_CLOSED.json'
$log=Join-Path $runtime 'scheduled_complete_subject_session_EEGNet_OpenBMI_MI_f0_v2.log'
$env:PERSIST_SOURCE_REPO=$repo
$env:MECHANISM_RUNTIME=$runtime
$env:COUPLING_RUNTIME='D:\nips-temp\TotalP\P1\protected_complement_coupling_runtime'
$env:SEVEN_RUNTIME='D:\nips-temp\TotalP\P1\seven_backbone_fourtask_3seed_runtime'
$env:OFFICIAL_BACKBONE_ROOT=Join-Path $env:SEVEN_RUNTIME 'official'
$env:PYTHONUNBUFFERED='1'
try {
  if((Test-Path -LiteralPath $log) -or (Test-Path -LiteralPath $result) -or (Test-Path -LiteralPath $failure)) {throw 'derived V2 terminal evidence exists'}
  if((Get-FileHash -LiteralPath $entry -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'a5cdc7955cda3da07cd5ee422b755ea74cc248813b1ee686d9310a3a3109e4db') {throw 'derived V2 source SHA mismatch'}
  if((Get-FileHash -LiteralPath (Join-Path $exp 'protocol\MECHANISM_CLOSURE_ANALYSIS_LOCK.md') -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'ac12d33f7d30b2c846ec4c4fb07f8513bf25e372943a767b134883bf095a7c88') {throw 'analysis lock SHA mismatch'}
  if((Get-FileHash -LiteralPath (Join-Path $exp 'protocol\FINAL_PROJECTOR_AMENDMENT.md') -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'aec1e604d38f1901bd9eca7e072e30a3d496a28061a726ebbe63133259b0e474') {throw 'projector amendment SHA mismatch'}
  if((Get-ScheduledTask -TaskName 'PERSIST_EEG_PC_MECHANISM_CELL_EEGNET_MI_F0_TRAIN_RANDOM_SUBJECT_V1').State -eq 'Running') {throw 'TRAIN random draw task still running'}
  $draws=@(Get-ChildItem -LiteralPath (Join-Path $cell 'train_random_subject_v1') -Filter 'draw_??.json' -ErrorAction SilentlyContinue)
  if($draws.Count -ne 20) {throw "only $($draws.Count)/20 TRAIN random draw files"}
  if((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB -lt 40) {throw 'insufficient physical free RAM'}
  $gpuFree=[int]((& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | Select-Object -First 1).Trim())
  if($gpuFree -lt 16000) {throw "insufficient GPU free memory: $gpuFree MiB"}
  "SUBJECT_SESSION_V2_START utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -LiteralPath $log -Encoding utf8
  $python='E:\Anaconda\envs\persist_stable_251\python.exe'
  $command='"'+$python+'" -u "'+$entry+'" >> "'+$log+'" 2>&1'
  & cmd.exe /d /c $command
  if($LASTEXITCODE -ne 0) {throw "derived V2 exited $LASTEXITCODE"}
  $row=Get-Content -LiteralPath $result -Raw|ConvertFrom-Json
  if($row.status -ne 'SUBJECT_SESSION_AUDIT_COMPLETE_POST_OUTCOME_DESCRIPTIVE' -or $row.final_heldout_accessed -ne $false -or $row.signature_component_count -ne 16 -or @($row.subject_session_signature_rows).Count -ne 68 -or @($row.subject_session_drift_rows).Count -ne 34) {throw 'derived V2 result invalid'}
  "SUBJECT_SESSION_V2_END exit=0 utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -LiteralPath $log -Append -Encoding utf8
  exit 0
} catch {
  "SUBJECT_SESSION_V2_EXCEPTION $_" | Out-File -LiteralPath $log -Append -Encoding utf8
  exit 1
}
