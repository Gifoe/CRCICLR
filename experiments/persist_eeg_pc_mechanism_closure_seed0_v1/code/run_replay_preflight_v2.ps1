<# Read-only historical replay identity preflight; no training or outer outcomes. #>
$ErrorActionPreference='Stop'
$repo='D:\nips-temp\TotalP\P1\CRCICLR_PERSIST_INCREMENTAL_VALUE_V1'
$exp=Join-Path $repo 'experiments\persist_eeg_pc_mechanism_closure_seed0_v1'
$runtime='D:\nips-temp\TotalP\P1\pc_mechanism_closure_runtime'
$entry=Join-Path $exp 'code\probe_replay_eegnet_v2.py'
$result=Join-Path $exp 'probe\EEGNet_OpenBMI_MI_fold0_replay_preflight_v2.json'
$log=Join-Path $runtime 'scheduled_replay_preflight_v2.log'
$env:PERSIST_SOURCE_REPO=$repo
$env:COUPLING_RUNTIME='D:\nips-temp\TotalP\P1\protected_complement_coupling_runtime'
$env:MECHANISM_RUNTIME=$runtime
$env:SEVEN_RUNTIME='D:\nips-temp\TotalP\P1\seven_backbone_fourtask_3seed_runtime'
$env:OFFICIAL_BACKBONE_ROOT=Join-Path $env:SEVEN_RUNTIME 'official'
$env:PYTHONUNBUFFERED='1'
try {
  if((Test-Path -LiteralPath $result) -or (Test-Path -LiteralPath $log)){throw 'replay preflight evidence exists'}
  if((Get-FileHash -LiteralPath $entry -Algorithm SHA256).Hash.ToLowerInvariant() -ne '38695336c26bd7d6320c0da86c6ecff0c729e6f3b852c41c7b07fe2de38635f9'){throw 'preflight source SHA mismatch'}
  "REPLAY_PREFLIGHT_V2_START utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -FilePath $log -Encoding utf8
  $python='E:\Anaconda\envs\persist_stable_251\python.exe'
  $command='"'+$python+'" -u "'+$entry+'" >> "'+$log+'" 2>&1'
  & cmd.exe /d /c $command
  if($LASTEXITCODE -ne 0){throw "replay preflight exited $LASTEXITCODE"}
  "REPLAY_PREFLIGHT_V2_END exit=0 utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -FilePath $log -Append -Encoding utf8
  exit 0
} catch {
  "REPLAY_PREFLIGHT_V2_EXCEPTION $_" | Out-File -FilePath $log -Append -Encoding utf8
  exit 1
}
