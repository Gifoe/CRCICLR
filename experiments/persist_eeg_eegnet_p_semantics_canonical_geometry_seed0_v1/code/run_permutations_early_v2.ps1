$ErrorActionPreference='Stop'
$repo='D:\nips-temp\TotalP\P1\CRCICLR_PERSIST_INCREMENTAL_VALUE_V1'
$exp=Join-Path $repo 'experiments\persist_eeg_eegnet_p_semantics_canonical_geometry_seed0_v1'
$runtime='D:\nips-temp\TotalP\P1\p_semantics_canonical_geometry_runtime'
$entry=Join-Path $exp 'code\permutation_runner_v2.py'
$log=Join-Path $runtime 'permutations_early_v2.log'
$python='E:\Anaconda\envs\persist_stable_251\python.exe'
$env:PERSIST_SOURCE_REPO=$repo
$env:P_SEMANTICS_RUNTIME=$runtime
$env:PYTHONUNBUFFERED='1'
function Log([string]$message){$message|Out-File -LiteralPath $log -Append -Encoding utf8}
try {
  if(Test-Path -LiteralPath $log){throw 'log exists; refuse duplicate'}
  $checks=@{
    'code\permutation_runner_v2.py'='239761df4fd249940becab171d523297cf8d91442415f98722c2f3c4957dd77d'
    'code\permutations_v2.py'='069618d2db51ec730c5d1363661dcf4d96891ea5c7c2f1930c42e4665fbefbbe'
    'code\permutations.py'='f1a322909bbab24bdbf51f7cbba54e48d989d34c3f9ed2fa151da75e6b0da2c1'
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
  $pilotTask='PERSIST_EEG_P_SEMANTICS_EEGNET_MI_F0_EMBEDDING_PERMUTATION_PILOT_V1'
  $pilotAudit=Join-Path $runtime 'analysis\fold0_seed0\embedding_64d\permutations\AUDIT.json'
  "EARLY_PERMUTATIONS_V2_START utc=$([DateTime]::UtcNow.ToString('o'))"|Out-File -LiteralPath $log -Encoding utf8
  while($true){
    $task=Get-ScheduledTask -TaskName $pilotTask
    $info=Get-ScheduledTaskInfo -TaskName $pilotTask
    if($task.State -eq 'Ready'){
      if($info.LastTaskResult -ne 0 -or !(Test-Path -LiteralPath $pilotAudit)){throw 'embedding permutation pilot failed'}
      $pilot=Get-Content -LiteralPath $pilotAudit -Raw|ConvertFrom-Json
      if($pilot.final_heldout_eeg_reads -ne 0 -or $pilot.session_identity_permutations -ne 200){throw 'embedding pilot audit invalid'}
    }
    $ram=(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB
    $gpu=[int]([string]((& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)|Select-Object -First 1)).Trim()
    $heavy=@(Get-CimInstance Win32_Process|Where-Object{
      $_.Name -match '^python(\.exe)?$' -and $_.CommandLine -and
      $_.CommandLine -match 'persist_eeg_pc_mechanism_closure_seed0_v1|analysis_runner.py|transform_runner.py|permutation_runner.py|permutation_runner_v2.py'
    })
    if($task.State -eq 'Ready' -and $ram -ge 40 -and $gpu -ge 16000 -and $heavy.Count -eq 0){break}
    Log "WAIT_RESOURCE pilot=$($task.State) ram_gb=$([math]::Round($ram,2)) gpu_mib=$gpu heavy=$($heavy.Count) utc=$([DateTime]::UtcNow.ToString('o'))"
    Start-Sleep -Seconds 30
  }
  foreach($stage in @('depth_point_elu_pool2','spatial_elu_pool1','temporal_bn')){
    foreach($fold in 0..4){
      $output=Join-Path $runtime "analysis\fold${fold}_seed0\$stage\permutations"
      if(Test-Path -LiteralPath $output){throw "output exists; refuse duplicate fold=$fold stage=$stage"}
      while($true){
        $ram=(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB
        $gpu=[int]([string]((& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)|Select-Object -First 1)).Trim()
        $heavy=@(Get-CimInstance Win32_Process|Where-Object{
          $_.Name -match '^python(\.exe)?$' -and $_.CommandLine -and
          $_.CommandLine -match 'persist_eeg_pc_mechanism_closure_seed0_v1|analysis_runner.py|transform_runner.py|permutation_runner.py|permutation_runner_v2.py'
        })
        if($ram -ge 40 -and $gpu -ge 16000 -and $heavy.Count -eq 0){break}
        Log "WAIT_RESOURCE fold=$fold stage=$stage ram_gb=$([math]::Round($ram,2)) gpu_mib=$gpu heavy=$($heavy.Count) utc=$([DateTime]::UtcNow.ToString('o'))"
        Start-Sleep -Seconds 30
      }
      Log "LAUNCH fold=$fold stage=$stage ram_gb=$([math]::Round($ram,2)) gpu_mib=$gpu utc=$([DateTime]::UtcNow.ToString('o'))"
      $cmd='"'+$python+'" -u "'+$entry+'" --fold '+$fold+' --stage '+$stage+' >> "'+$log+'" 2>&1'
      & cmd.exe /d /c $cmd
      if($LASTEXITCODE -ne 0){throw "python failed fold=$fold stage=$stage exit=$LASTEXITCODE"}
      $audit=Join-Path $output 'AUDIT.json'
      $record=Get-Content -LiteralPath $audit -Raw|ConvertFrom-Json
      if($record.final_heldout_eeg_reads -ne 0 -or $record.outer_development_fit_rows -ne 0 -or
         $record.session_identity_permutations -ne 200 -or $record.within_cell_trial_label_permutations -ne 200){
        throw "invalid audit fold=$fold stage=$stage"
      }
      Log "COMPLETE fold=$fold stage=$stage audit_sha=$((Get-FileHash -LiteralPath $audit -Algorithm SHA256).Hash.ToLowerInvariant()) utc=$([DateTime]::UtcNow.ToString('o'))"
    }
  }
  Log "EARLY_PERMUTATIONS_V2_ALL_COMPLETE utc=$([DateTime]::UtcNow.ToString('o'))"
  exit 0
} catch {
  if(Test-Path -LiteralPath $log){Log "EARLY_PERMUTATIONS_V2_EXCEPTION $($_.Exception.Message) utc=$([DateTime]::UtcNow.ToString('o'))"}
  exit 1
}
