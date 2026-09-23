<# One TRAIN-only engineering probe. Output and log are immutable evidence. #>
$ErrorActionPreference='Stop'
$repo='D:\nips-temp\TotalP\P1\CRCICLR_PERSIST_INCREMENTAL_VALUE_V1'
$runtime='D:\nips-temp\TotalP\P1\pc_mechanism_closure_runtime'
$experiment=Join-Path $repo 'experiments\persist_eeg_pc_mechanism_closure_seed0_v1'
$entry=Join-Path $experiment 'code\probe_sampling_v1.py'
$core=Join-Path $experiment 'code\sampling_core_v1.py'
$log=Join-Path $runtime 'scheduled_sampling_probe_v1.log'
$result=Join-Path $experiment 'probe\EEGNet_OpenBMI_MI_fold0_sampling_v1.json'
$failure=Join-Path $experiment 'probe\EEGNet_OpenBMI_MI_fold0_sampling_v1_FAIL_CLOSED.json'
$env:PERSIST_SOURCE_REPO=$repo
$env:COUPLING_RUNTIME='D:\nips-temp\TotalP\P1\protected_complement_coupling_runtime'
$env:PYTHONUNBUFFERED='1'
$python='E:\Anaconda\envs\persist_stable_251\python.exe'
try {
  if ((Test-Path -LiteralPath $result) -or (Test-Path -LiteralPath $failure) -or (Test-Path -LiteralPath $log)) { throw 'sampling probe evidence already exists' }
  if ((Get-FileHash -LiteralPath $entry -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'c84e7844353aee865456549cc56b0e5d6963f002c5e8364705d4ef151e67133a') { throw 'probe source SHA mismatch' }
  if ((Get-FileHash -LiteralPath $core -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'c8aaae43966a58b7563a1b715d2ca660d53fb353d8ae07dfd6402ae635900880') { throw 'sampling core SHA mismatch' }
  if ((Get-FileHash -LiteralPath (Join-Path $experiment 'protocol\MECHANISM_CLOSURE_ANALYSIS_LOCK.md') -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'ac12d33f7d30b2c846ec4c4fb07f8513bf25e372943a767b134883bf095a7c88') { throw 'analysis lock SHA mismatch' }
  if ((Get-FileHash -LiteralPath (Join-Path $experiment 'protocol\FINAL_PROJECTOR_AMENDMENT.md') -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'aec1e604d38f1901bd9eca7e072e30a3d496a28061a726ebbe63133259b0e474') { throw 'amendment SHA mismatch' }
  "SAMPLING_PROBE_V1_START utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -FilePath $log -Encoding utf8
  $command='"'+$python+'" -u "'+$entry+'" >> "'+$log+'" 2>&1'
  & cmd.exe /d /c $command
  if ($LASTEXITCODE -ne 0) { throw "sampling probe exited $LASTEXITCODE" }
  "SAMPLING_PROBE_V1_END exit=0 utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -FilePath $log -Append -Encoding utf8
  exit 0
} catch {
  "SAMPLING_PROBE_V1_EXCEPTION $_" | Out-File -FilePath $log -Append -Encoding utf8
  exit 1
}
