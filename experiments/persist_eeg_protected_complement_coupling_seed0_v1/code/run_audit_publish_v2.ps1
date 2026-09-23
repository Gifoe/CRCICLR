<# Review full outputs and create a distinct compact publication directory. #>
$ErrorActionPreference='Stop'
$repo='D:\nips-temp\TotalP\P1\CRCICLR_PERSIST_INCREMENTAL_VALUE_V1'
$runtime='D:\nips-temp\TotalP\P1\protected_complement_coupling_runtime'
$env:PERSIST_SOURCE_REPO=$repo
$env:COUPLING_RUNTIME=$runtime
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
$env:PYTHONUNBUFFERED='1'
$python='E:\Anaconda\envs\persist_stable_251\python.exe'
$entry=Join-Path $repo 'experiments\persist_eeg_protected_complement_coupling_seed0_v1\code\audit_publish_v2.py'
$log=Join-Path $runtime 'scheduled_audit_publish_v2.log'
try {
  "PUBLISH_AUDIT_START utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -FilePath $log -Append -Encoding utf8
  $command='"'+$python+'" -u "'+$entry+'" >> "'+$log+'" 2>&1'
  & cmd.exe /d /c $command
  if($LASTEXITCODE -ne 0) { throw "publication audit exited $LASTEXITCODE" }
  "PUBLISH_AUDIT_END exit=0 utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -FilePath $log -Append -Encoding utf8
  exit 0
} catch {
  "PUBLISH_AUDIT_EXCEPTION $_" | Out-File -FilePath $log -Append -Encoding utf8
  exit 1
}
