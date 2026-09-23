<# First-cell directional phase; requires separate mediation and replay phases. #>
$ErrorActionPreference='Stop'
$repo='D:\nips-temp\TotalP\P1\CRCICLR_PERSIST_INCREMENTAL_VALUE_V1'
$exp=Join-Path $repo 'experiments\persist_eeg_pc_mechanism_closure_seed0_v1'
$runtime='D:\nips-temp\TotalP\P1\pc_mechanism_closure_runtime'
$entry=Join-Path $exp 'code\run_directional_cell_v1.py'
$core=Join-Path $exp 'code\directional_core_v1.py'
$sampling=Join-Path $exp 'code\sampling_core.py'
$cellData=Join-Path $exp 'code\cell_data_v1.py'
$cell=Join-Path $runtime 'cells\eegnet\openbmi_mi\fold0_seed0'
$log=Join-Path $runtime 'scheduled_directional_cell_EEGNet_OpenBMI_MI_f0_v1.log'
$env:PERSIST_SOURCE_REPO=$repo
$env:COUPLING_RUNTIME='D:\nips-temp\TotalP\P1\protected_complement_coupling_runtime'
$env:MECHANISM_RUNTIME=$runtime
$env:PYTHONUNBUFFERED='1'
try {
  if((Test-Path -LiteralPath $log) -or (Test-Path -LiteralPath (Join-Path $cell 'DIRECTIONAL_V1.json')) -or (Test-Path -LiteralPath (Join-Path $cell 'DIRECTIONAL_V1_FAIL_CLOSED.json'))){throw 'directional evidence exists'}
  if(-not (Test-Path -LiteralPath (Join-Path $cell 'MEDIATION_V2.json'))){throw 'validated mediation phase is absent'}
  $expected=@{
    $entry='1ed64ef149ac3dcfef13678dd91c1da8fc3c967e9668677f2bdf2ecd8d57915d';
    $core='a4f9c715dae66876aef2316084c803dad8676b53988133e4f458338648287e39';
    $sampling='7b8f5b94caca3924de3c714a780987a52337359963f38d5135c78410c8b1a631';
    $cellData='c5e4f90865445bcc1ec313b1ca5b1970cf484a9ef7f44e8aa1bf38a0fd721416'
  }
  foreach($path in $expected.Keys){if((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected[$path]){throw "source SHA mismatch: $path"}}
  if((Get-FileHash -LiteralPath (Join-Path $exp 'protocol\MECHANISM_CLOSURE_ANALYSIS_LOCK.md') -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'ac12d33f7d30b2c846ec4c4fb07f8513bf25e372943a767b134883bf095a7c88'){throw 'analysis lock SHA mismatch'}
  if((Get-FileHash -LiteralPath (Join-Path $exp 'protocol\FINAL_PROJECTOR_AMENDMENT.md') -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'aec1e604d38f1901bd9eca7e072e30a3d496a28061a726ebbe63133259b0e474'){throw 'projector amendment SHA mismatch'}
  "DIRECTIONAL_CELL_V1_START utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -FilePath $log -Encoding utf8
  $python='E:\Anaconda\envs\persist_stable_251\python.exe'
  $command='"'+$python+'" -u "'+$entry+'" --model EEGNet --task OpenBMI_MI --fold 0 >> "'+$log+'" 2>&1'
  & cmd.exe /d /c $command
  if($LASTEXITCODE -ne 0){throw "directional cell exited $LASTEXITCODE"}
  "DIRECTIONAL_CELL_V1_END exit=0 utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -FilePath $log -Append -Encoding utf8
  exit 0
} catch {
  "DIRECTIONAL_CELL_V1_EXCEPTION $_" | Out-File -FilePath $log -Append -Encoding utf8
  exit 1
}
