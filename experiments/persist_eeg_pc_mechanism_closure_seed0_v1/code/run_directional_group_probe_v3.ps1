<# TRAIN-only grouped-forward engineering probe; not a scientific cell. #>
$ErrorActionPreference='Stop'
$repo='D:\nips-temp\TotalP\P1\CRCICLR_PERSIST_INCREMENTAL_VALUE_V1'
$exp=Join-Path $repo 'experiments\persist_eeg_pc_mechanism_closure_seed0_v1'
$runtime='D:\nips-temp\TotalP\P1\pc_mechanism_closure_runtime'
$entry=Join-Path $exp 'code\probe_directional_group_v3.py'
$core=Join-Path $exp 'code\directional_core_v3.py'
$result=Join-Path $exp 'probe\EEGNet_OpenBMI_MI_fold0_directional_group_probe_v3.json'
$log=Join-Path $runtime 'scheduled_directional_group_probe_v3.log'
$env:PERSIST_SOURCE_REPO=$repo
$env:COUPLING_RUNTIME='D:\nips-temp\TotalP\P1\protected_complement_coupling_runtime'
$env:MECHANISM_RUNTIME=$runtime
$env:PYTHONUNBUFFERED='1'
try {
  if((Test-Path -LiteralPath $result) -or (Test-Path -LiteralPath $log)){throw 'group probe evidence exists'}
  if((Get-FileHash -LiteralPath $entry -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'f94b8101f569143778edb80a16d86ebc05e61075a82a8717c024ba6047bfd75b'){throw 'probe source SHA mismatch'}
  if((Get-FileHash -LiteralPath $core -Algorithm SHA256).Hash.ToLowerInvariant() -ne '5f50f0f649039f2cb83c2b951952e04a645908bf5328bfc277b79948e05608af'){throw 'directional core SHA mismatch'}
  "DIRECTIONAL_GROUP_PROBE_V3_START utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -FilePath $log -Encoding utf8
  $python='E:\Anaconda\envs\persist_stable_251\python.exe'
  $command='"'+$python+'" -u "'+$entry+'" >> "'+$log+'" 2>&1'
  & cmd.exe /d /c $command
  if($LASTEXITCODE -ne 0){throw "group probe exited $LASTEXITCODE"}
  "DIRECTIONAL_GROUP_PROBE_V3_END exit=0 utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -FilePath $log -Append -Encoding utf8
  exit 0
} catch {
  "DIRECTIONAL_GROUP_PROBE_V3_EXCEPTION $_" | Out-File -FilePath $log -Append -Encoding utf8
  exit 1
}
