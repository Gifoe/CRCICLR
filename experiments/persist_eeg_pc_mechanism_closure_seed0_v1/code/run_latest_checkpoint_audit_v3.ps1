<# Verify historical selected/final endpoints with source-specific epoch recipes. #>
$ErrorActionPreference='Stop'
$repo='D:\nips-temp\TotalP\P1\CRCICLR_PERSIST_INCREMENTAL_VALUE_V1'
$runtime='D:\nips-temp\TotalP\P1\pc_mechanism_closure_runtime'
$env:PERSIST_SOURCE_REPO=$repo
$env:COUPLING_RUNTIME='D:\nips-temp\TotalP\P1\protected_complement_coupling_runtime'
$env:PYTHONUNBUFFERED='1'
$python='E:\Anaconda\envs\persist_stable_251\python.exe'
$entry=Join-Path $repo 'experiments\persist_eeg_pc_mechanism_closure_seed0_v1\code\audit_latest_checkpoints_v3.py'
$log=Join-Path $runtime 'scheduled_latest_checkpoint_audit_v3.log'
try {
  "LATEST_AUDIT_V3_START utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -FilePath $log -Append -Encoding utf8
  $command='"'+$python+'" -u "'+$entry+'" >> "'+$log+'" 2>&1'
  & cmd.exe /d /c $command
  if($LASTEXITCODE -ne 0) { throw "latest checkpoint audit v3 exited $LASTEXITCODE" }
  "LATEST_AUDIT_V3_END exit=0 utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -FilePath $log -Append -Encoding utf8
  exit 0
} catch {
  "LATEST_AUDIT_V3_EXCEPTION $_" | Out-File -FilePath $log -Append -Encoding utf8
  exit 1
}
