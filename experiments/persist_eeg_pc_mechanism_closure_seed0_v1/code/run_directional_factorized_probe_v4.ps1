<# TRAIN-only EEGNet affine-prefix equivalence/speed probe; no scientific result. #>
$ErrorActionPreference='Stop'
$repo='D:\nips-temp\TotalP\P1\CRCICLR_PERSIST_INCREMENTAL_VALUE_V1'
$exp=Join-Path $repo 'experiments\persist_eeg_pc_mechanism_closure_seed0_v1'
$runtime='D:\nips-temp\TotalP\P1\pc_mechanism_closure_runtime'
$entry=Join-Path $exp 'code\probe_directional_factorized_v7.py'
$factor=Join-Path $exp 'code\directional_factorized_eegnet_v4.py'
$result=Join-Path $exp 'probe\EEGNet_OpenBMI_MI_fold0_directional_factorized_probe_v7.json'
$log=Join-Path $runtime 'scheduled_directional_factorized_probe_v7.log'
$env:PERSIST_SOURCE_REPO=$repo
$env:COUPLING_RUNTIME='D:\nips-temp\TotalP\P1\protected_complement_coupling_runtime'
$env:MECHANISM_RUNTIME=$runtime
$env:PYTHONUNBUFFERED='1'
try {
  if((Test-Path -LiteralPath $result) -or (Test-Path -LiteralPath $log)){throw 'factorization evidence exists'}
  if((Get-FileHash -LiteralPath $entry -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'ad375bf2db7282a62c4c811a555719e60fa2ab10dc526348e37f499fd9d4bfb2'){throw 'probe source SHA mismatch'}
  if((Get-FileHash -LiteralPath $factor -Algorithm SHA256).Hash.ToLowerInvariant() -ne '03c1f0262885546ea70ebfc7db93471968f9952e2aac62097d13b4317ff91729'){throw 'factorized source SHA mismatch'}
  "DIRECTIONAL_FACTORIZED_PROBE_V7_START utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -FilePath $log -Encoding utf8
  $python='E:\Anaconda\envs\persist_stable_251\python.exe'
  $command='"'+$python+'" -u "'+$entry+'" >> "'+$log+'" 2>&1'
  & cmd.exe /d /c $command
  if($LASTEXITCODE -ne 0){throw "factorized probe exited $LASTEXITCODE"}
  "DIRECTIONAL_FACTORIZED_PROBE_V7_END exit=0 utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -FilePath $log -Append -Encoding utf8
  exit 0
} catch {
  "DIRECTIONAL_FACTORIZED_PROBE_V7_EXCEPTION $_" | Out-File -FilePath $log -Append -Encoding utf8
  exit 1
}
