<# TRAIN-only engineering probe, never a scientific cell result. #>
$ErrorActionPreference='Stop'
$repo='D:\nips-temp\TotalP\P1\CRCICLR_PERSIST_INCREMENTAL_VALUE_V1'
$exp=Join-Path $repo 'experiments\persist_eeg_pc_mechanism_closure_seed0_v1'
$runtime='D:\nips-temp\TotalP\P1\pc_mechanism_closure_runtime'
$entry=Join-Path $exp 'code\probe_directional_v1.py'
$core=Join-Path $exp 'code\directional_core_v1.py'
$result=Join-Path $exp 'probe\EEGNet_OpenBMI_MI_fold0_directional_probe_v1.json'
$log=Join-Path $runtime 'scheduled_directional_probe_v1.log'
$env:PERSIST_SOURCE_REPO=$repo
$env:COUPLING_RUNTIME='D:\nips-temp\TotalP\P1\protected_complement_coupling_runtime'
$env:MECHANISM_RUNTIME=$runtime
$env:PYTHONUNBUFFERED='1'
try {
  if((Test-Path -LiteralPath $result) -or (Test-Path -LiteralPath $log)){throw 'directional probe evidence exists'}
  if((Get-FileHash -LiteralPath $entry -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'eab95ec77dbf0f97cff6551a76c30bb36ec54c8ca5cbb278fb88343b37440890'){throw 'probe source SHA mismatch'}
  if((Get-FileHash -LiteralPath $core -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'a4f9c715dae66876aef2316084c803dad8676b53988133e4f458338648287e39'){throw 'directional core SHA mismatch'}
  "DIRECTIONAL_PROBE_START utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -FilePath $log -Encoding utf8
  $python='E:\Anaconda\envs\persist_stable_251\python.exe'
  $command='"'+$python+'" -u "'+$entry+'" >> "'+$log+'" 2>&1'
  & cmd.exe /d /c $command
  if($LASTEXITCODE -ne 0){throw "directional probe exited $LASTEXITCODE"}
  "DIRECTIONAL_PROBE_END exit=0 utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -FilePath $log -Append -Encoding utf8
  exit 0
} catch {
  "DIRECTIONAL_PROBE_EXCEPTION $_" | Out-File -FilePath $log -Append -Encoding utf8
  exit 1
}
