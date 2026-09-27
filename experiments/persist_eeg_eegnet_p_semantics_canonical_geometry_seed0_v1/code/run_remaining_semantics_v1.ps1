$ErrorActionPreference='Stop'
$repo='D:\nips-temp\TotalP\P1\CRCICLR_PERSIST_INCREMENTAL_VALUE_V1'
$exp=Join-Path $repo 'experiments\persist_eeg_eegnet_p_semantics_canonical_geometry_seed0_v1'
$runtime='D:\nips-temp\TotalP\P1\p_semantics_canonical_geometry_runtime'
$entry=Join-Path $exp 'code\analysis_runner.py'
$log=Join-Path $runtime 'remaining_semantics_v1.log'
$python='E:\Anaconda\envs\persist_stable_251\python.exe'
$env:PERSIST_SOURCE_REPO=$repo
$env:P_SEMANTICS_RUNTIME=$runtime
$env:PYTHONUNBUFFERED='1'
function Log([string]$m){$m|Out-File -LiteralPath $log -Append -Encoding utf8}
try {
  if(Test-Path -LiteralPath $log){throw 'log exists; refuse duplicate'}
  $checks=@{
    'code\analysis_runner.py'='0a8913b4593c3ebc78819903838efdff15a5c67af3e1d7218256a5aad342002b'
    'code\data_geometry.py'='c3e8e496d2989a3805e1d3edbb8b5c3b39c253a0b6e6d5544ed6f8f8990f27a1'
    'code\semantics.py'='6c1703556ba93ffc23619f7e02acb1eaf7e1fce908cbb45103cfc312f7c015ed'
    'code\decoders.py'='32c6008f4f0e341bf26d34d03154900f5ea41d1a8cce48895721a2018835e0c8'
    'protocol\PROTOCOL_LOCK.json'='c65bd28a7856ad1c5ba139908b56cf54ac22a16d51f076f3079cb8971d7b7245'
  }
  foreach($key in $checks.Keys){if((Get-FileHash -LiteralPath (Join-Path $exp $key) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $checks[$key]){throw "SHA mismatch $key"}}
  $jobs=@()
  foreach($stage in @('embedding_64d','depth_point_elu_pool2','spatial_elu_pool1','temporal_bn')){
    foreach($fold in 0..4){if($stage -eq 'embedding_64d' -and $fold -eq 0){continue};$jobs+=@{stage=$stage;fold=$fold}}
  }
  foreach($job in $jobs){$output=Join-Path $runtime "analysis\fold$($job.fold)_seed0\$($job.stage)\semantics";if(Test-Path -LiteralPath $output){throw "output already exists fold=$($job.fold) stage=$($job.stage)"}}
  "REMAINING_SEMANTICS_START jobs=$($jobs.Count) utc=$([DateTime]::UtcNow.ToString('o'))"|Out-File -LiteralPath $log -Encoding utf8
  foreach($job in $jobs){
    $fold=$job.fold; $stage=$job.stage
    while($true){
      $ram=(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB
      $gpu=[int]([string]((& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)|Select-Object -First 1)).Trim()
      $heavy=@(Get-CimInstance Win32_Process|Where-Object{$_.Name -match '^python(\.exe)?$' -and $_.CommandLine -and $_.CommandLine -match 'persist_eeg_pc_mechanism_closure_seed0_v1|p_semantics_canonical_geometry_runtime'})
      if($ram -ge 40 -and $gpu -ge 16000 -and $heavy.Count -eq 0){break}
      Log "WAIT_RESOURCE fold=$fold stage=$stage ram_gb=$([math]::Round($ram,2)) gpu_mib=$gpu heavy=$($heavy.Count) utc=$([DateTime]::UtcNow.ToString('o'))"
      Start-Sleep -Seconds 20
    }
    Log "SEMANTICS_LAUNCH fold=$fold stage=$stage ram_gb=$([math]::Round($ram,2)) gpu_mib=$gpu utc=$([DateTime]::UtcNow.ToString('o'))"
    $cmd='"'+$python+'" -u "'+$entry+'" semantics --fold '+$fold+' --stage '+$stage+' >> "'+$log+'" 2>&1'
    & cmd.exe /d /c $cmd
    if($LASTEXITCODE -ne 0){throw "analysis failed fold=$fold stage=$stage exit=$LASTEXITCODE"}
    $output=Join-Path $runtime "analysis\fold${fold}_seed0\$stage\semantics"
    $audit=Join-Path $output 'AUDIT.json'
    $j=Get-Content -LiteralPath $audit -Raw|ConvertFrom-Json
    if($j.final_heldout_eeg_reads -ne 0 -or $j.outer_development_fit_rows -ne 0 -or
       $j.checkpoint_validation_fit_rows -ne 0 -or [int]$j.fold -ne $fold -or $j.stage -ne $stage){throw "audit invalid fold=$fold stage=$stage"}
    Log "SEMANTICS_COMPLETE fold=$fold stage=$stage audit_sha=$((Get-FileHash -LiteralPath $audit -Algorithm SHA256).Hash.ToLowerInvariant()) utc=$([DateTime]::UtcNow.ToString('o'))"
  }
  Log "REMAINING_SEMANTICS_COMPLETE utc=$([DateTime]::UtcNow.ToString('o'))"
  exit 0
} catch {if(Test-Path -LiteralPath $log){Log "SEMANTICS_EXCEPTION $($_.Exception.Message) utc=$([DateTime]::UtcNow.ToString('o'))"};exit 1}
