$ErrorActionPreference='Stop'
$repo='D:\nips-temp\TotalP\P1\CRCICLR_PERSIST_INCREMENTAL_VALUE_V1'
$exp=Join-Path $repo 'experiments\persist_eeg_eegnet_p_semantics_canonical_geometry_seed0_v1'
$runtime='D:\nips-temp\TotalP\P1\p_semantics_canonical_geometry_runtime'
$entry=Join-Path $exp 'code\permutation_runner.py'
$log=Join-Path $runtime 'permutation_fold0_embedding_v1.log'
$output=Join-Path $runtime 'analysis\fold0_seed0\embedding_64d\permutations'
$python='E:\Anaconda\envs\persist_stable_251\python.exe'
$env:PERSIST_SOURCE_REPO=$repo
$env:P_SEMANTICS_RUNTIME=$runtime
$env:PYTHONUNBUFFERED='1'
function Log([string]$m){$m|Out-File -LiteralPath $log -Append -Encoding utf8}
try {
  if(Test-Path -LiteralPath $log){throw 'log exists; refuse duplicate'}
  if(Test-Path -LiteralPath $output){throw 'output exists; refuse duplicate'}
  $checks=@{
    'code\permutation_runner.py'='20f1d620e14039c69081fb198419c993df173449e15968138b380c07af49d074'
    'code\permutations.py'='f1a322909bbab24bdbf51f7cbba54e48d989d34c3f9ed2fa151da75e6b0da2c1'
    'code\analysis_runner.py'='0a8913b4593c3ebc78819903838efdff15a5c67af3e1d7218256a5aad342002b'
    'code\data_geometry.py'='c3e8e496d2989a3805e1d3edbb8b5c3b39c253a0b6e6d5544ed6f8f8990f27a1'
    'code\semantics.py'='6c1703556ba93ffc23619f7e02acb1eaf7e1fce908cbb45103cfc312f7c015ed'
    'code\decoders.py'='32c6008f4f0e341bf26d34d03154900f5ea41d1a8cce48895721a2018835e0c8'
  }
  foreach($key in $checks.Keys){if((Get-FileHash -LiteralPath (Join-Path $exp $key) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $checks[$key]){throw "SHA mismatch $key"}}
  "PERMUTATION_PILOT_START utc=$([DateTime]::UtcNow.ToString('o'))"|Out-File -LiteralPath $log -Encoding utf8
  while($true){
    $preceding=(Get-ScheduledTask -TaskName 'PERSIST_EEG_P_SEMANTICS_EEGNET_MI_F0_EMBEDDING_TRANSFORM_PILOT_V1').State
    $ram=(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB
    $gpu=[int]([string]((& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)|Select-Object -First 1)).Trim()
    $heavy=@(Get-CimInstance Win32_Process|Where-Object{$_.Name -match '^python(\.exe)?$' -and $_.CommandLine -and $_.CommandLine -match 'persist_eeg_pc_mechanism_closure_seed0_v1|analysis_runner.py|transform_runner.py|permutation_runner.py'})
    if($preceding -eq 'Ready' -and $ram -ge 40 -and $gpu -ge 16000 -and $heavy.Count -eq 0){break}
    Log "WAIT_RESOURCE preceding=$preceding ram_gb=$([math]::Round($ram,2)) gpu_mib=$gpu heavy=$($heavy.Count) utc=$([DateTime]::UtcNow.ToString('o'))"
    Start-Sleep -Seconds 30
  }
  Log "PERMUTATION_PILOT_LAUNCH ram_gb=$([math]::Round($ram,2)) gpu_mib=$gpu utc=$([DateTime]::UtcNow.ToString('o'))"
  $cmd='"'+$python+'" -u "'+$entry+'" --fold 0 --stage embedding_64d >> "'+$log+'" 2>&1'
  & cmd.exe /d /c $cmd
  if($LASTEXITCODE -ne 0){throw "permutation python exit=$LASTEXITCODE"}
  $audit=Join-Path $output 'AUDIT.json'
  $j=Get-Content -LiteralPath $audit -Raw|ConvertFrom-Json
  if($j.final_heldout_eeg_reads -ne 0 -or $j.outer_development_fit_rows -ne 0 -or $j.session_identity_permutations -ne 200){throw 'permutation audit invalid'}
  Log "PERMUTATION_PILOT_COMPLETE audit_sha=$((Get-FileHash -LiteralPath $audit -Algorithm SHA256).Hash.ToLowerInvariant()) utc=$([DateTime]::UtcNow.ToString('o'))"
  exit 0
} catch {if(Test-Path -LiteralPath $log){Log "PERMUTATION_PILOT_EXCEPTION $($_.Exception.Message) utc=$([DateTime]::UtcNow.ToString('o'))"};exit 1}
