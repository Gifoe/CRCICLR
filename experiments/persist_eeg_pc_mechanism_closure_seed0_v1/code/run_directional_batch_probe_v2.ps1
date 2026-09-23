<# TRAIN-only batch-equivalence engineering probe; not a scientific cell. #>
$ErrorActionPreference='Stop'
$repo='D:\nips-temp\TotalP\P1\CRCICLR_PERSIST_INCREMENTAL_VALUE_V1'
$exp=Join-Path $repo 'experiments\persist_eeg_pc_mechanism_closure_seed0_v1'
$runtime='D:\nips-temp\TotalP\P1\pc_mechanism_closure_runtime'
$entry=Join-Path $exp 'code\probe_directional_batch_v2.py'
$core=Join-Path $exp 'code\directional_core_v2.py'
$result=Join-Path $exp 'probe\EEGNet_OpenBMI_MI_fold0_directional_batch_probe_v2.json'
$log=Join-Path $runtime 'scheduled_directional_batch_probe_v2.log'
$env:PERSIST_SOURCE_REPO=$repo
$env:COUPLING_RUNTIME='D:\nips-temp\TotalP\P1\protected_complement_coupling_runtime'
$env:MECHANISM_RUNTIME=$runtime
$env:PYTHONUNBUFFERED='1'
try {
  if((Test-Path -LiteralPath $result) -or (Test-Path -LiteralPath $log)){throw 'batch probe evidence exists'}
  if((Get-FileHash -LiteralPath $entry -Algorithm SHA256).Hash.ToLowerInvariant() -ne '13119184de332f8a301750f34459b1ea66733e1baf0474183cb090a15525758f'){throw 'probe source SHA mismatch'}
  if((Get-FileHash -LiteralPath $core -Algorithm SHA256).Hash.ToLowerInvariant() -ne '156e076d2381ecb553b947776c6d6e451cd6392bb6365088c338fc2c35f6177b'){throw 'directional core SHA mismatch'}
  "DIRECTIONAL_BATCH_PROBE_V2_START utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -FilePath $log -Encoding utf8
  $python='E:\Anaconda\envs\persist_stable_251\python.exe'
  $command='"'+$python+'" -u "'+$entry+'" >> "'+$log+'" 2>&1'
  & cmd.exe /d /c $command
  if($LASTEXITCODE -ne 0){throw "batch probe exited $LASTEXITCODE"}
  "DIRECTIONAL_BATCH_PROBE_V2_END exit=0 utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -FilePath $log -Append -Encoding utf8
  exit 0
} catch {
  "DIRECTIONAL_BATCH_PROBE_V2_EXCEPTION $_" | Out-File -FilePath $log -Append -Encoding utf8
  exit 1
}
