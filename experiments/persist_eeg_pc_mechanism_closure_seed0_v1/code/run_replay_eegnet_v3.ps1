<# Independent exact-recipe EEGNet trajectory replay; never writes historical cells. #>
$ErrorActionPreference='Stop'
$repo='D:\nips-temp\TotalP\P1\CRCICLR_PERSIST_INCREMENTAL_VALUE_V1'
$exp=Join-Path $repo 'experiments\persist_eeg_pc_mechanism_closure_seed0_v1'
$runtime='D:\nips-temp\TotalP\P1\pc_mechanism_closure_runtime'
$entry=Join-Path $exp 'code\replay_eegnet_v2.py'
$cell=Join-Path $runtime 'cells\eegnet\openbmi_mi\fold0_seed0'
$replay=Join-Path $cell 'replay_v2'
$log=Join-Path $runtime 'scheduled_replay_EEGNet_OpenBMI_MI_f0_v3.log'
$env:PERSIST_SOURCE_REPO=$repo
$env:COUPLING_RUNTIME='D:\nips-temp\TotalP\P1\protected_complement_coupling_runtime'
$env:MECHANISM_RUNTIME=$runtime
$env:SEVEN_RUNTIME='D:\nips-temp\TotalP\P1\seven_backbone_fourtask_3seed_runtime'
$env:OFFICIAL_BACKBONE_ROOT=Join-Path $env:SEVEN_RUNTIME 'official'
$env:PYTHONUNBUFFERED='1'
try {
  if((Test-Path -LiteralPath $log) -or (Test-Path -LiteralPath $replay)){throw 'replay evidence exists'}
  $preflight=Join-Path $exp 'probe\EEGNet_OpenBMI_MI_fold0_replay_preflight_v2.json'
  if(-not (Test-Path -LiteralPath $preflight)){throw 'historical replay preflight absent'}
  $preflightRecord=Get-Content -Raw -LiteralPath $preflight | ConvertFrom-Json
  if($preflightRecord.status -ne 'REPLAY_PREFLIGHT_PASSED' -or $preflightRecord.final_heldout_accessed -ne $false){throw 'historical replay preflight invalid'}
  if((Get-FileHash -LiteralPath $entry -Algorithm SHA256).Hash.ToLowerInvariant() -ne '944dc5ad03dc536870bd3338d5ee55069149e14b05a466ef175f75b59c18cc3a'){throw 'replay source SHA mismatch'}
  if((Get-FileHash -LiteralPath (Join-Path $exp 'protocol\MECHANISM_CLOSURE_ANALYSIS_LOCK.md') -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'ac12d33f7d30b2c846ec4c4fb07f8513bf25e372943a767b134883bf095a7c88'){throw 'analysis lock SHA mismatch'}
  "REPLAY_EEGNET_MI_F0_START utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -FilePath $log -Encoding utf8
  $python='E:\Anaconda\envs\persist_stable_251\python.exe'
  $command='"'+$python+'" -u "'+$entry+'" --task OpenBMI_MI --fold 0 >> "'+$log+'" 2>&1'
  & cmd.exe /d /c $command
  if($LASTEXITCODE -ne 0){throw "replay exited $LASTEXITCODE"}
  "REPLAY_EEGNET_MI_F0_END exit=0 utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -FilePath $log -Append -Encoding utf8
  exit 0
} catch {
  "REPLAY_EEGNET_MI_F0_EXCEPTION $_" | Out-File -FilePath $log -Append -Encoding utf8
  exit 1
}
