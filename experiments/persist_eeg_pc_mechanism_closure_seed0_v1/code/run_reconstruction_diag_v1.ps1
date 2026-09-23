<# Preserve v1 failure; run TRAIN-only numerical diagnosis independently. #>
$ErrorActionPreference='Stop'
$repo='D:\nips-temp\TotalP\P1\CRCICLR_PERSIST_INCREMENTAL_VALUE_V1'
$exp=Join-Path $repo 'experiments\persist_eeg_pc_mechanism_closure_seed0_v1'
$runtime='D:\nips-temp\TotalP\P1\pc_mechanism_closure_runtime'
$entry=Join-Path $exp 'code\diagnose_reconstruction_v1.py'
$log=Join-Path $runtime 'scheduled_reconstruction_diag_v1.log'
$result=Join-Path $exp 'probe\EEGNet_OpenBMI_MI_fold0_reconstruction_failure_diagnostic_v1.json'
$env:PERSIST_SOURCE_REPO=$repo
$env:COUPLING_RUNTIME='D:\nips-temp\TotalP\P1\protected_complement_coupling_runtime'
$env:PYTHONUNBUFFERED='1'
try {
  if((Test-Path -LiteralPath $log) -or (Test-Path -LiteralPath $result)){throw 'diagnostic evidence already exists'}
  if((Get-FileHash -LiteralPath $entry -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'ddbfcf5ca9973f900f6025604e45d7d09f9cb8dfdba3918ec5e6a910936828f1'){throw 'diagnostic source SHA mismatch'}
  "RECON_DIAG_START utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -FilePath $log -Encoding utf8
  $python='E:\Anaconda\envs\persist_stable_251\python.exe'
  $command='"'+$python+'" -u "'+$entry+'" >> "'+$log+'" 2>&1'
  & cmd.exe /d /c $command
  if($LASTEXITCODE -ne 0){throw "diagnostic exited $LASTEXITCODE"}
  "RECON_DIAG_END exit=0 utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -FilePath $log -Append -Encoding utf8
  exit 0
} catch {
  "RECON_DIAG_EXCEPTION $_" | Out-File -FilePath $log -Append -Encoding utf8
  exit 1
}
