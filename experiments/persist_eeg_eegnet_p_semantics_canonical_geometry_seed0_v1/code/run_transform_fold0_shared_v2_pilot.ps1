$ErrorActionPreference='Stop'
$repo='D:\nips-temp\TotalP\P1\CRCICLR_PERSIST_INCREMENTAL_VALUE_V1'
$exp=Join-Path $repo 'experiments\persist_eeg_eegnet_p_semantics_canonical_geometry_seed0_v1'
$runtime='D:\nips-temp\TotalP\P1\p_semantics_canonical_geometry_runtime'
$entry=Join-Path $exp 'code\transform_runner_v2.py'
$log=Join-Path $runtime 'transform_fold0_shared_v2_pilot.log'
$output=Join-Path $runtime 'analysis\fold0_seed0\depth_point_elu_pool2\transform'
$python='E:\Anaconda\envs\persist_stable_251\python.exe'
$env:PERSIST_SOURCE_REPO=$repo
$env:P_SEMANTICS_RUNTIME=$runtime
$env:PYTHONUNBUFFERED='1'
function Log([string]$message){$message|Out-File -LiteralPath $log -Append -Encoding utf8}
try {
  if(Test-Path -LiteralPath $log){throw 'log exists; refuse duplicate'}
  if(Test-Path -LiteralPath $output){throw 'output exists; refuse duplicate'}
  $checks=@{
    'code\transform_runner_v2.py'='4f3c870963ba240529197ef542074bc7d56364c73262e4e06449582477a30174'
    'code\transform_runner.py'='174c340701ccb91efa1c3f33a2e21e82d59c70a00ab89b152dfac4b6d91976fe'
    'code\canonical.py'='a806d525cc10840f88a3eea2e9e60100b2104aae2e1f697c06e98ce2c7582cdc'
    'code\analysis_runner.py'='0a8913b4593c3ebc78819903838efdff15a5c67af3e1d7218256a5aad342002b'
    'code\data_geometry.py'='c3e8e496d2989a3805e1d3edbb8b5c3b39c253a0b6e6d5544ed6f8f8990f27a1'
    'code\semantics.py'='6c1703556ba93ffc23619f7e02acb1eaf7e1fce908cbb45103cfc312f7c015ed'
    'code\decoders.py'='32c6008f4f0e341bf26d34d03154900f5ea41d1a8cce48895721a2018835e0c8'
  }
  foreach($key in $checks.Keys){
    if((Get-FileHash -LiteralPath (Join-Path $exp $key) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $checks[$key]){
      throw "SHA mismatch $key"
    }
  }
  $precedingName='PERSIST_EEG_P_SEMANTICS_EEGNET_MI_F0_EMBEDDING_PERMUTATION_PILOT_V1'
  $precedingAudit=Join-Path $runtime 'analysis\fold0_seed0\embedding_64d\permutations\AUDIT.json'
  "SHARED_TRANSFORM_V2_PILOT_START utc=$([DateTime]::UtcNow.ToString('o'))"|Out-File -LiteralPath $log -Encoding utf8
  while($true){
    $preceding=Get-ScheduledTask -TaskName $precedingName
    $info=Get-ScheduledTaskInfo -TaskName $precedingName
    if($preceding.State -eq 'Ready'){
      if($info.LastTaskResult -ne 0 -or !(Test-Path -LiteralPath $precedingAudit)){throw 'preceding permutation pilot failed'}
      $audit=Get-Content -LiteralPath $precedingAudit -Raw|ConvertFrom-Json
      if($audit.final_heldout_eeg_reads -ne 0){throw 'preceding audit invalid'}
    }
    $ram=(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB
    $gpu=[int]([string]((& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)|Select-Object -First 1)).Trim()
    $heavy=@(Get-CimInstance Win32_Process|Where-Object{
      $_.Name -match '^python(\.exe)?$' -and $_.CommandLine -and
      $_.CommandLine -match 'persist_eeg_pc_mechanism_closure_seed0_v1|analysis_runner.py|transform_runner.py|transform_runner_v2.py|permutation_runner.py|permutation_runner_v2.py'
    })
    if($preceding.State -eq 'Ready' -and $ram -ge 40 -and $gpu -ge 16000 -and $heavy.Count -eq 0){break}
    Log "WAIT_RESOURCE preceding=$($preceding.State) ram_gb=$([math]::Round($ram,2)) gpu_mib=$gpu heavy=$($heavy.Count) utc=$([DateTime]::UtcNow.ToString('o'))"
    Start-Sleep -Seconds 30
  }
  Log "SHARED_TRANSFORM_V2_PILOT_LAUNCH ram_gb=$([math]::Round($ram,2)) gpu_mib=$gpu utc=$([DateTime]::UtcNow.ToString('o'))"
  $cmd='"'+$python+'" -u "'+$entry+'" --fold 0 --stage depth_point_elu_pool2 >> "'+$log+'" 2>&1'
  & cmd.exe /d /c $cmd
  if($LASTEXITCODE -ne 0){throw "python exit=$LASTEXITCODE"}
  $auditPath=Join-Path $output 'AUDIT.json'
  $record=Get-Content -LiteralPath $auditPath -Raw|ConvertFrom-Json
  if($record.schema -ne 'P_SEMANTICS_FOLD_STAGE_TRANSFORM_V2' -or
     $record.final_heldout_eeg_reads -ne 0 -or $record.target_decoder_refit -ne $false -or
     $record.transform_fit_population -ne 'TRAIN_GEOMETRY_ONLY'){
    throw 'transform V2 pilot audit invalid'
  }
  Log "SHARED_TRANSFORM_V2_PILOT_COMPLETE audit_sha=$((Get-FileHash -LiteralPath $auditPath -Algorithm SHA256).Hash.ToLowerInvariant()) utc=$([DateTime]::UtcNow.ToString('o'))"
  exit 0
} catch {
  if(Test-Path -LiteralPath $log){Log "SHARED_TRANSFORM_V2_PILOT_EXCEPTION $($_.Exception.Message) utc=$([DateTime]::UtcNow.ToString('o'))"}
  exit 1
}
