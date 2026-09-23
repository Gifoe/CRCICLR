<# Compact preview from completed first-cell evidence only. #>
$ErrorActionPreference='Stop'
$repo='D:\nips-temp\TotalP\P1\CRCICLR_PERSIST_INCREMENTAL_VALUE_V1'
$exp=Join-Path $repo 'experiments\persist_eeg_pc_mechanism_closure_seed0_v1'
$runtime='D:\nips-temp\TotalP\P1\pc_mechanism_closure_runtime'
$entry=Join-Path $exp 'code\build_first_cell_compact_preview_v2.py'
$preview=Join-Path $runtime 'compact_preview\eegnet_openbmi_mi_fold0_seed0_v2'
$log=Join-Path $runtime 'scheduled_first_cell_compact_preview_EEGNet_OpenBMI_MI_f0_v2.log'
$env:PERSIST_SOURCE_REPO=$repo
$env:MECHANISM_RUNTIME=$runtime
$env:COUPLING_RUNTIME='D:\nips-temp\TotalP\P1\protected_complement_coupling_runtime'
$env:SEVEN_RUNTIME='D:\nips-temp\TotalP\P1\seven_backbone_fourtask_3seed_runtime'
$env:OFFICIAL_BACKBONE_ROOT=Join-Path $env:SEVEN_RUNTIME 'official'
$env:PYTHONUNBUFFERED='1'
try {
  if((Test-Path -LiteralPath $preview) -or (Test-Path -LiteralPath $log)) {throw 'compact-preview evidence exists'}
  if((Get-FileHash -LiteralPath $entry -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'b3228d7cbc43c22adaa5f752e7e2994ffc9199163907ed86d963dfd286aa50e4') {throw 'compact-preview source SHA mismatch'}
  if((Get-FileHash -LiteralPath (Join-Path $exp 'protocol\MECHANISM_CLOSURE_ANALYSIS_LOCK.md') -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'ac12d33f7d30b2c846ec4c4fb07f8513bf25e372943a767b134883bf095a7c88') {throw 'analysis lock SHA mismatch'}
  if((Get-FileHash -LiteralPath (Join-Path $exp 'protocol\FINAL_PROJECTOR_AMENDMENT.md') -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'aec1e604d38f1901bd9eca7e072e30a3d496a28061a726ebbe63133259b0e474') {throw 'projector amendment SHA mismatch'}
  if((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB -lt 40) {throw 'insufficient physical free RAM'}
  $gpuFree=[int]((& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | Select-Object -First 1).Trim())
  if($gpuFree -lt 16000) {throw "insufficient GPU free memory: $gpuFree MiB"}
  "FIRST_CELL_COMPACT_PREVIEW_START utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -LiteralPath $log -Encoding utf8
  $python='E:\Anaconda\envs\persist_stable_251\python.exe'
  $command='"'+$python+'" -u "'+$entry+'" >> "'+$log+'" 2>&1'
  & cmd.exe /d /c $command
  if($LASTEXITCODE -ne 0) {throw "compact-preview exited $LASTEXITCODE"}
  $row=Get-Content -LiteralPath (Join-Path $preview 'PROVENANCE.json') -Raw|ConvertFrom-Json
  if($row.status -ne 'FIRST_CELL_COMPACT_PREVIEW_COMPLETE_NOT_20_CELL_RESULT' -or $row.final_heldout_accessed -ne $false -or @($row.output_rows.PSObject.Properties).Count -ne 10) {throw 'compact-preview provenance invalid'}
  "FIRST_CELL_COMPACT_PREVIEW_END exit=0 utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -LiteralPath $log -Append -Encoding utf8
  exit 0
} catch {
  "FIRST_CELL_COMPACT_PREVIEW_EXCEPTION $_" | Out-File -LiteralPath $log -Append -Encoding utf8
  exit 1
}
