<# Fail-closed Phase-2 upstream integrity and checkpoint availability gate. #>
$ErrorActionPreference='Stop'
$repo='D:\nips-temp\TotalP\P1\CRCICLR_PERSIST_INCREMENTAL_VALUE_V1'
$runtime='D:\nips-temp\TotalP\P1\pc_mechanism_closure_runtime'
New-Item -ItemType Directory -Path $runtime -Force | Out-Null
$env:PERSIST_SOURCE_REPO=$repo
$env:COUPLING_RUNTIME='D:\nips-temp\TotalP\P1\protected_complement_coupling_runtime'
$env:PYTHONUNBUFFERED='1'
$python='E:\Anaconda\envs\persist_stable_251\python.exe'
$entry=Join-Path $repo 'experiments\persist_eeg_pc_mechanism_closure_seed0_v1\code\gate_checkpoint_audit.py'
$log=Join-Path $runtime 'scheduled_gate_checkpoint_audit.log'
try {
  "MECHANISM_GATE_START utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -FilePath $log -Append -Encoding utf8
  $command='"'+$python+'" -u "'+$entry+'" >> "'+$log+'" 2>&1'
  & cmd.exe /d /c $command
  if($LASTEXITCODE -ne 0) { throw "gate audit exited $LASTEXITCODE" }
  "MECHANISM_GATE_END exit=0 utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -FilePath $log -Append -Encoding utf8
  exit 0
} catch {
  "MECHANISM_GATE_EXCEPTION $_" | Out-File -FilePath $log -Append -Encoding utf8
  exit 1
}
