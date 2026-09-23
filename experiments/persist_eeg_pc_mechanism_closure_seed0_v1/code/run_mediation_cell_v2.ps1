<# Separately provenanced v2 mediation phase after v1 numeric failure. #>
$ErrorActionPreference='Stop'
$repo='D:\nips-temp\TotalP\P1\CRCICLR_PERSIST_INCREMENTAL_VALUE_V1'
$experiment=Join-Path $repo 'experiments\persist_eeg_pc_mechanism_closure_seed0_v1'
$runtime='D:\nips-temp\TotalP\P1\pc_mechanism_closure_runtime'
$entry=Join-Path $experiment 'code\run_mediation_cell_v2.py'
$sampling=Join-Path $experiment 'code\sampling_core.py'
$cellData=Join-Path $experiment 'code\cell_data_v1.py'
$mediation=Join-Path $experiment 'code\mediation_core_v5.py'
$log=Join-Path $runtime 'scheduled_mediation_cell_EEGNet_OpenBMI_MI_f0_v2.log'
$cell=Join-Path $runtime 'cells\eegnet\openbmi_mi\fold0_seed0'
$env:PERSIST_SOURCE_REPO=$repo
$env:COUPLING_RUNTIME='D:\nips-temp\TotalP\P1\protected_complement_coupling_runtime'
$env:MECHANISM_RUNTIME=$runtime
$env:PYTHONUNBUFFERED='1'
$python='E:\Anaconda\envs\persist_stable_251\python.exe'
try {
  if ((Test-Path -LiteralPath $log) -or (Test-Path -LiteralPath (Join-Path $cell 'MEDIATION_V2.json')) -or (Test-Path -LiteralPath (Join-Path $cell 'MEDIATION_V2_FAIL_CLOSED.json'))) { throw 'v2 mediation evidence already exists' }
  if (-not (Test-Path -LiteralPath (Join-Path $cell 'MEDIATION_V1_FAIL_CLOSED.json'))) { throw 'v1 failure evidence missing' }
  $expected=@{
    $entry='b00479f7cdf6e3ea5b118773e6d3382e6fb29e9b39beb58b4842df10a54e3175';
    $sampling='7b8f5b94caca3924de3c714a780987a52337359963f38d5135c78410c8b1a631';
    $cellData='c5e4f90865445bcc1ec313b1ca5b1970cf484a9ef7f44e8aa1bf38a0fd721416';
    $mediation='22853e842a9dfcb5457c9eb96a84b738b4a2a47054b5778dd4e830bca6c27846'
  }
  foreach($path in $expected.Keys) { if((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected[$path]) { throw "source SHA mismatch: $path" } }
  if ((Get-FileHash -LiteralPath (Join-Path $experiment 'protocol\MECHANISM_CLOSURE_ANALYSIS_LOCK.md') -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'ac12d33f7d30b2c846ec4c4fb07f8513bf25e372943a767b134883bf095a7c88') { throw 'analysis lock SHA mismatch' }
  if ((Get-FileHash -LiteralPath (Join-Path $experiment 'protocol\FINAL_PROJECTOR_AMENDMENT.md') -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'aec1e604d38f1901bd9eca7e072e30a3d496a28061a726ebbe63133259b0e474') { throw 'projector amendment SHA mismatch' }
  "MEDIATION_CELL_V2_START utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -FilePath $log -Encoding utf8
  $command='"'+$python+'" -u "'+$entry+'" --model EEGNet --task OpenBMI_MI --fold 0 >> "'+$log+'" 2>&1'
  & cmd.exe /d /c $command
  if($LASTEXITCODE -ne 0){throw "mediation cell exited $LASTEXITCODE"}
  "MEDIATION_CELL_V2_END exit=0 utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -FilePath $log -Append -Encoding utf8
  exit 0
} catch {
  "MEDIATION_CELL_V2_EXCEPTION $_" | Out-File -FilePath $log -Append -Encoding utf8
  exit 1
}
