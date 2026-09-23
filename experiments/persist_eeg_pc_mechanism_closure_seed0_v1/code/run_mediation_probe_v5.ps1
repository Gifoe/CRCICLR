<# Authorized canonical-final clarification probe; one model per task. #>
param([ValidateSet('EEGNet','EEGConformer')][string]$Model='EEGNet')
$ErrorActionPreference='Stop'
$repo='D:\nips-temp\TotalP\P1\CRCICLR_PERSIST_INCREMENTAL_VALUE_V1'
$runtime='D:\nips-temp\TotalP\P1\pc_mechanism_closure_runtime'
$env:PERSIST_SOURCE_REPO=$repo
$env:COUPLING_RUNTIME='D:\nips-temp\TotalP\P1\protected_complement_coupling_runtime'
$env:PYTHONUNBUFFERED='1'
$python='E:\Anaconda\envs\persist_stable_251\python.exe'
$entry=Join-Path $repo 'experiments\persist_eeg_pc_mechanism_closure_seed0_v1\code\probe_mediation_v5.py'
$log=Join-Path $runtime ("scheduled_mediation_probe_v5_"+$Model+".log")
try {
  "MEDIATION_PROBE_V5_START model=$Model utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -FilePath $log -Append -Encoding utf8
  $command='"'+$python+'" -u "'+$entry+'" --model '+$Model+' --task OpenBMI_MI --fold 0 >> "'+$log+'" 2>&1'
  & cmd.exe /d /c $command
  if($LASTEXITCODE -ne 0) { throw "mediation probe v5 exited $LASTEXITCODE" }
  "MEDIATION_PROBE_V5_END exit=0 utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -FilePath $log -Append -Encoding utf8
  exit 0
} catch {
  "MEDIATION_PROBE_V5_EXCEPTION $_" | Out-File -FilePath $log -Append -Encoding utf8
  exit 1
}
