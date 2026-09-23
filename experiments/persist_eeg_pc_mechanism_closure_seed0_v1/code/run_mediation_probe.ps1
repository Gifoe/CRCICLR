<# Single TRAIN-only engineering probe. Never duplicate or overwrite evidence. #>
$ErrorActionPreference='Stop'
$repo='D:\nips-temp\TotalP\P1\CRCICLR_PERSIST_INCREMENTAL_VALUE_V1'
$runtime='D:\nips-temp\TotalP\P1\pc_mechanism_closure_runtime'
$env:PERSIST_SOURCE_REPO=$repo
$env:COUPLING_RUNTIME='D:\nips-temp\TotalP\P1\protected_complement_coupling_runtime'
$env:PYTHONUNBUFFERED='1'
$python='E:\Anaconda\envs\persist_stable_251\python.exe'
$entry=Join-Path $repo 'experiments\persist_eeg_pc_mechanism_closure_seed0_v1\code\probe_mediation.py'
$log=Join-Path $runtime 'scheduled_mediation_probe.log'
try {
  "MEDIATION_PROBE_START utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -FilePath $log -Append -Encoding utf8
  $command='"'+$python+'" -u "'+$entry+'" >> "'+$log+'" 2>&1'
  & cmd.exe /d /c $command
  if($LASTEXITCODE -ne 0) { throw "mediation probe exited $LASTEXITCODE" }
  "MEDIATION_PROBE_END exit=0 utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -FilePath $log -Append -Encoding utf8
  exit 0
} catch {
  "MEDIATION_PROBE_EXCEPTION $_" | Out-File -FilePath $log -Append -Encoding utf8
  exit 1
}
