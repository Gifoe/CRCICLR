<# Production first-cell mediation phase; not a complete mechanism cell. #>
$ErrorActionPreference='Stop'
$repo='D:\nips-temp\TotalP\P1\CRCICLR_PERSIST_INCREMENTAL_VALUE_V1'
$experiment=Join-Path $repo 'experiments\persist_eeg_pc_mechanism_closure_seed0_v1'
$runtime='D:\nips-temp\TotalP\P1\pc_mechanism_closure_runtime'
$entry=Join-Path $experiment 'code\run_mediation_cell_v1.py'
$sampling=Join-Path $experiment 'code\sampling_core.py'
$cellData=Join-Path $experiment 'code\cell_data_v1.py'
$mediation=Join-Path $experiment 'code\mediation_core_v4.py'
$log=Join-Path $runtime 'scheduled_mediation_cell_EEGNet_OpenBMI_MI_f0_v1.log'
$cell=Join-Path $runtime 'cells\eegnet\openbmi_mi\fold0_seed0'
$env:PERSIST_SOURCE_REPO=$repo
$env:COUPLING_RUNTIME='D:\nips-temp\TotalP\P1\protected_complement_coupling_runtime'
$env:MECHANISM_RUNTIME=$runtime
$env:PYTHONUNBUFFERED='1'
$python='E:\Anaconda\envs\persist_stable_251\python.exe'
try {
  if ((Test-Path -LiteralPath $log) -or (Test-Path -LiteralPath (Join-Path $cell 'MEDIATION_V1.json')) -or (Test-Path -LiteralPath (Join-Path $cell 'MEDIATION_V1_FAIL_CLOSED.json'))) { throw 'first-cell mediation evidence already exists' }
  $expected=@{
    $entry='cfd5271e9430ea71cad52aed4cf16cd88e34d3b2214259638b7353a231d20f1c';
    $sampling='7b8f5b94caca3924de3c714a780987a52337359963f38d5135c78410c8b1a631';
    $cellData='c5e4f90865445bcc1ec313b1ca5b1970cf484a9ef7f44e8aa1bf38a0fd721416';
    $mediation='8fad6209ca2845612dcf67cc524a44ba8c76c50a8f4eca6f4f20169ccd9d92e2'
  }
  foreach($path in $expected.Keys) { if((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected[$path]) { throw "source SHA mismatch: $path" } }
  if ((Get-FileHash -LiteralPath (Join-Path $experiment 'protocol\MECHANISM_CLOSURE_ANALYSIS_LOCK.md') -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'ac12d33f7d30b2c846ec4c4fb07f8513bf25e372943a767b134883bf095a7c88') { throw 'analysis lock SHA mismatch' }
  if ((Get-FileHash -LiteralPath (Join-Path $experiment 'protocol\FINAL_PROJECTOR_AMENDMENT.md') -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'aec1e604d38f1901bd9eca7e072e30a3d496a28061a726ebbe63133259b0e474') { throw 'projector amendment SHA mismatch' }
  "MEDIATION_CELL_V1_START utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -FilePath $log -Encoding utf8
  $command='"'+$python+'" -u "'+$entry+'" --model EEGNet --task OpenBMI_MI --fold 0 >> "'+$log+'" 2>&1'
  & cmd.exe /d /c $command
  if($LASTEXITCODE -ne 0){throw "mediation cell exited $LASTEXITCODE"}
  "MEDIATION_CELL_V1_END exit=0 utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -FilePath $log -Append -Encoding utf8
  exit 0
} catch {
  "MEDIATION_CELL_V1_EXCEPTION $_" | Out-File -FilePath $log -Append -Encoding utf8
  exit 1
}
