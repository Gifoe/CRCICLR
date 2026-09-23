<# Preserve v1 failure and diagnosis; audit all frozen TRAIN recipient pairs independently. #>
$ErrorActionPreference='Stop'
$repo='D:\nips-temp\TotalP\P1\CRCICLR_PERSIST_INCREMENTAL_VALUE_V1'
$exp=Join-Path $repo 'experiments\persist_eeg_pc_mechanism_closure_seed0_v1'
$runtime='D:\nips-temp\TotalP\P1\pc_mechanism_closure_runtime'
$entry=Join-Path $exp 'code\diagnose_reconstruction_v2.py'
$log=Join-Path $runtime 'scheduled_reconstruction_diag_v2.log'
$result=Join-Path $exp 'probe\EEGNet_OpenBMI_MI_fold0_reconstruction_failure_diagnostic_v2.json'
$env:PERSIST_SOURCE_REPO=$repo
$env:COUPLING_RUNTIME='D:\nips-temp\TotalP\P1\protected_complement_coupling_runtime'
$env:PYTHONUNBUFFERED='1'
try {
  if((Test-Path -LiteralPath $log) -or (Test-Path -LiteralPath $result)){throw 'diagnostic evidence already exists'}
  if((Get-FileHash -LiteralPath $entry -Algorithm SHA256).Hash.ToLowerInvariant() -ne '7ea60bc2ae1bd671a2216e2eb719f96e3e9425b25b99ad6398877e0c8d099599'){throw 'diagnostic source SHA mismatch'}
  "RECON_DIAG_V2_START utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -FilePath $log -Encoding utf8
  $python='E:\Anaconda\envs\persist_stable_251\python.exe'
  $command='"'+$python+'" -u "'+$entry+'" >> "'+$log+'" 2>&1'
  & cmd.exe /d /c $command
  if($LASTEXITCODE -ne 0){throw "diagnostic exited $LASTEXITCODE"}
  "RECON_DIAG_V2_END exit=0 utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -FilePath $log -Append -Encoding utf8
  exit 0
} catch {
  "RECON_DIAG_V2_EXCEPTION $_" | Out-File -FilePath $log -Append -Encoding utf8
  exit 1
}
