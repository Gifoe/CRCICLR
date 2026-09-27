$ErrorActionPreference='Stop'
$repo='D:\nips-temp\TotalP\P1\CRCICLR_PERSIST_INCREMENTAL_VALUE_V1'
$exp=Join-Path $repo 'experiments\persist_eeg_eegnet_p_semantics_canonical_geometry_seed0_v1'
$runtime='D:\nips-temp\TotalP\P1\p_semantics_canonical_geometry_runtime'
$entry=Join-Path $exp 'code\transform_runner_v3.py'
$log=Join-Path $runtime 'transform_fold0_embedding_v3.log'
$output=Join-Path $runtime 'analysis\fold0_seed0\embedding_64d\transform'
$python='E:\Anaconda\envs\persist_stable_251\python.exe'
$env:PERSIST_SOURCE_REPO=$repo
$env:P_SEMANTICS_RUNTIME=$runtime
$env:PYTHONUNBUFFERED='1'
function Log([string]$m){$m|Out-File -LiteralPath $log -Append -Encoding utf8}
try {
  if(Test-Path -LiteralPath $log){throw 'log exists; refuse duplicate'}
  if(Test-Path -LiteralPath $output){throw 'output exists; refuse duplicate'}
  $checks=@{
    'code\transform_runner_v3.py'='728780435acccae6e3aa9e1712515c8c6c18da524e6105ffc71d84575cae3977'
    'code\canonical.py'='a806d525cc10840f88a3eea2e9e60100b2104aae2e1f697c06e98ce2c7582cdc'
    'code\analysis_runner.py'='0a8913b4593c3ebc78819903838efdff15a5c67af3e1d7218256a5aad342002b'
    'code\data_geometry.py'='c3e8e496d2989a3805e1d3edbb8b5c3b39c253a0b6e6d5544ed6f8f8990f27a1'
    'code\semantics.py'='6c1703556ba93ffc23619f7e02acb1eaf7e1fce908cbb45103cfc312f7c015ed'
    'code\decoders.py'='32c6008f4f0e341bf26d34d03154900f5ea41d1a8cce48895721a2018835e0c8'
    'protocol\PROTOCOL_LOCK.json'='c65bd28a7856ad1c5ba139908b56cf54ac22a16d51f076f3079cb8971d7b7245'
  }
  foreach($key in $checks.Keys){if((Get-FileHash -LiteralPath (Join-Path $exp $key) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $checks[$key]){throw "SHA mismatch $key"}}
  "TRANSFORM_PILOT_START utc=$([DateTime]::UtcNow.ToString('o'))"|Out-File -LiteralPath $log -Encoding utf8
  while($true){
    $preceding=(Get-ScheduledTask -TaskName 'PERSIST_EEG_P_SEMANTICS_REMAINING_DECODERS_V1').State
    $precedingResult=(Get-ScheduledTaskInfo -TaskName 'PERSIST_EEG_P_SEMANTICS_REMAINING_DECODERS_V1').LastTaskResult
    $ram=(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB
    $gpu=[int]([string]((& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)|Select-Object -First 1)).Trim()
    $heavy=@(Get-CimInstance Win32_Process|Where-Object{$_.Name -match '^python(\.exe)?$' -and $_.CommandLine -and $_.CommandLine -match 'persist_eeg_pc_mechanism_closure_seed0_v1|p_semantics_canonical_geometry_runtime|analysis_runner.py|transform_runner|permutation_runner'})
    if($preceding -eq 'Ready' -and $precedingResult -eq 0 -and $ram -ge 40 -and $gpu -ge 16000 -and $heavy.Count -eq 0){break}
    if($preceding -eq 'Ready' -and $precedingResult -ne 0){throw 'preceding decoder queue failed'}
    Log "WAIT_RESOURCE preceding=$preceding ram_gb=$([math]::Round($ram,2)) gpu_mib=$gpu heavy=$($heavy.Count) utc=$([DateTime]::UtcNow.ToString('o'))"
    Start-Sleep -Seconds 30
  }
  if(-not ((Get-Content -LiteralPath (Join-Path $runtime 'remaining_decoders_v1.log') -Tail 2) -match 'REMAINING_DECODERS_COMPLETE')){throw 'preceding decoder queue did not complete successfully'}
  Log "TRANSFORM_PILOT_LAUNCH ram_gb=$([math]::Round($ram,2)) gpu_mib=$gpu utc=$([DateTime]::UtcNow.ToString('o'))"
  $cmd='"'+$python+'" -u "'+$entry+'" --fold 0 --stage embedding_64d >> "'+$log+'" 2>&1'
  & cmd.exe /d /c $cmd
  if($LASTEXITCODE -ne 0){throw "transform python exit=$LASTEXITCODE"}
  $audit=Join-Path $output 'AUDIT.json'
  $j=Get-Content -LiteralPath $audit -Raw|ConvertFrom-Json
  if($j.schema -ne 'P_SEMANTICS_FOLD_STAGE_TRANSFORM_V3' -or $j.final_heldout_eeg_reads -ne 0 -or $j.target_decoder_refit -ne $false -or $j.transform_fit_population -ne 'TRAIN_GEOMETRY_ONLY' -or $j.TRAIN_labels_used_for_unsupervised_rank_CV -ne $false){throw 'transform audit invalid'}
  Log "TRANSFORM_PILOT_COMPLETE audit_sha=$((Get-FileHash -LiteralPath $audit -Algorithm SHA256).Hash.ToLowerInvariant()) utc=$([DateTime]::UtcNow.ToString('o'))"
  exit 0
} catch {if(Test-Path -LiteralPath $log){Log "TRANSFORM_PILOT_EXCEPTION $($_.Exception.Message) utc=$([DateTime]::UtcNow.ToString('o'))"};exit 1}
