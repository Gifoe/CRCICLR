$ErrorActionPreference='Stop'
$repo='D:\nips-temp\TotalP\P1\CRCICLR_PERSIST_INCREMENTAL_VALUE_V1'
$exp=Join-Path $repo 'experiments\persist_eeg_eegnet_p_semantics_canonical_geometry_seed0_v1'
$runtime='D:\nips-temp\TotalP\P1\p_semantics_canonical_geometry_runtime'
$log=Join-Path $runtime 'geometry_folds1to4_v1.log'
$entry=Join-Path $exp 'code\run.py'
$data=Join-Path $exp 'code\data_geometry.py'
$python='E:\Anaconda\envs\persist_stable_251\python.exe'
$expectedEntrySha='1ea88ab1a3a4a54d3ca130169d711409e25fd6742657149f79c48b0a671a39a9'
$expectedDataSha='c3e8e496d2989a3805e1d3edbb8b5c3b39c253a0b6e6d5544ed6f8f8990f27a1'
$env:PERSIST_SOURCE_REPO=$repo
$env:P_SEMANTICS_RUNTIME=$runtime
$env:PYTHONUNBUFFERED='1'
function Log([string]$m){$m|Out-File -LiteralPath $log -Append -Encoding utf8}
try {
  if(Test-Path -LiteralPath $log){throw 'log exists; refuse duplicate'}
  if((Get-FileHash -LiteralPath $entry -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedEntrySha){throw 'entry SHA mismatch'}
  if((Get-FileHash -LiteralPath $data -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedDataSha){throw 'data SHA mismatch'}
  foreach($fold in 1..4){if(Test-Path -LiteralPath (Join-Path $runtime "geometry\fold${fold}_seed0")){throw "fold $fold already exists"}}
  "GEOMETRY_FOLDS1TO4_START utc=$([DateTime]::UtcNow.ToString('o'))"|Out-File -LiteralPath $log -Encoding utf8
  foreach($fold in 1..4){
    while($true){
      $ram=(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB
      $gpu=[int]([string]((& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)|Select-Object -First 1)).Trim()
      $heavy=@(Get-CimInstance Win32_Process|Where-Object{$_.Name -match '^python(\.exe)?$' -and $_.CommandLine -and $_.CommandLine -match 'persist_eeg_pc_mechanism_closure_seed0_v1|p_semantics_canonical_geometry_runtime'})
      if($ram -ge 40 -and $gpu -ge 16000 -and $heavy.Count -eq 0){break}
      Log "WAIT_RESOURCE fold=$fold ram_gb=$([math]::Round($ram,2)) gpu_mib=$gpu heavy=$($heavy.Count) utc=$([DateTime]::UtcNow.ToString('o'))"
      Start-Sleep -Seconds 20
    }
    Log "GEOMETRY_LAUNCH fold=$fold ram_gb=$([math]::Round($ram,2)) gpu_mib=$gpu utc=$([DateTime]::UtcNow.ToString('o'))"
    $cmd='"'+$python+'" -u "'+$entry+'" geometry --fold '+$fold+' >> "'+$log+'" 2>&1'
    & cmd.exe /d /c $cmd
    if($LASTEXITCODE -ne 0){throw "geometry fold $fold python exit=$LASTEXITCODE"}
    $result=Join-Path $runtime "geometry\fold${fold}_seed0\GEOMETRY_PROVENANCE.json"
    $j=Get-Content -LiteralPath $result -Raw|ConvertFrom-Json
    if($j.schema -ne 'P_SEMANTICS_EEGNET_INNER_TRAIN_GEOMETRY_V1' -or [int]$j.fold -ne $fold -or
       $j.final_heldout_eeg_reads -ne 0 -or $j.outer_development_eeg_reads -ne 0){throw "fold $fold provenance invalid"}
    Log "GEOMETRY_COMPLETE fold=$fold sha=$((Get-FileHash -LiteralPath $result -Algorithm SHA256).Hash.ToLowerInvariant()) utc=$([DateTime]::UtcNow.ToString('o'))"
  }
  Log "GEOMETRY_FOLDS1TO4_COMPLETE utc=$([DateTime]::UtcNow.ToString('o'))"
  exit 0
} catch {if(Test-Path -LiteralPath $log){Log "GEOMETRY_EXCEPTION $($_.Exception.Message) utc=$([DateTime]::UtcNow.ToString('o'))"};exit 1}
