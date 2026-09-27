$ErrorActionPreference='Stop'
$repo='D:\nips-temp\TotalP\P1\CRCICLR_PERSIST_INCREMENTAL_VALUE_V1'
$exp=Join-Path $repo 'experiments\persist_eeg_eegnet_core_residual_premise_validation_seed0_v1'
$runtime='D:\nips-temp\TotalP\P1\core_residual_premise_runtime'
$entry=Join-Path $exp 'code\run_v2.py'
$python='E:\Anaconda\envs\persist_stable_251\python.exe'
$log=Join-Path $runtime 'evaluate_all_v2.log'
$env:PERSIST_SOURCE_REPO=$repo
$env:P_SEMANTICS_RUNTIME='D:\nips-temp\TotalP\P1\p_semantics_canonical_geometry_runtime'
$env:P_TASKCORE_RUNTIME='D:\nips-temp\TotalP\P1\p_taskcore_specificity_runtime'
$env:TASKCORE_OUTPUT_FOLDER='outputs_v4'
$env:TASKCORE_PARENT_CODE=Join-Path $repo 'experiments\persist_eeg_eegnet_p_taskcore_specificity_closure_seed0_v1\code\run_v4.py'
$env:TASKCORE_PROTOCOL_SNAPSHOT=Join-Path $exp 'protocol\TASKCORE_SOURCE_PROTOCOL_LOCK.json'
$env:CORE_RESIDUAL_RUNTIME=$runtime
$env:PYTHONUNBUFFERED='1'
function Log([string]$line){$line|Out-File -LiteralPath $log -Append -Encoding utf8}
try {
  if(Test-Path -LiteralPath $log){throw 'phase log already exists; refuse duplicate'}
  if(-not (Test-Path -LiteralPath $entry) -or -not (Test-Path -LiteralPath $python)){throw 'entry or python absent'}
  foreach($fold in 0..4){
    $audit=Join-Path $runtime "construction\fold$fold\AUDIT.json"
    if(-not (Test-Path -LiteralPath $audit)){throw "construction audit absent fold$fold"}
  }
  Log "START phase=evaluate entry_sha=$((Get-FileHash -LiteralPath $entry -Algorithm SHA256).Hash.ToLowerInvariant()) utc=$([DateTime]::UtcNow.ToString('o'))"
  foreach($fold in 0..4){
    $out=Join-Path $runtime "evaluation\fold$fold"
    if(Test-Path -LiteralPath $out){throw "phase output exists fold$fold"}
    while($true){
      $ram=(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB
      $gpu=[int]([string]((& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)|Select-Object -First 1)).Trim()
      $heavy=@(Get-CimInstance Win32_Process|Where-Object{$_.Name -match '^python(\.exe)?$' -and $_.CommandLine -and $_.CommandLine -match 'p_semantics_canonical_geometry_runtime|p_taskcore_specificity_runtime|core_residual_premise_runtime|persist_eeg_pc_mechanism_closure_seed0_v1'})
      if($ram -ge 40 -and $gpu -ge 16000 -and $heavy.Count -eq 0){break}
      Log "WAIT fold=$fold ram_gb=$([math]::Round($ram,2)) gpu_mib=$gpu heavy=$($heavy.Count) utc=$([DateTime]::UtcNow.ToString('o'))"
      Start-Sleep -Seconds 30
    }
    $part=Join-Path $runtime "evaluate_fold$($fold)_v2.log"
    if(Test-Path -LiteralPath $part){throw "fold log exists fold$fold"}
    Log "LAUNCH fold=$fold ram_gb=$([math]::Round($ram,2)) gpu_mib=$gpu utc=$([DateTime]::UtcNow.ToString('o'))"
    $command='"'+$python+'" -u "'+$entry+'" evaluate --fold '+$fold+' > "'+$part+'" 2>&1'
    & cmd.exe /d /c $command
    if($LASTEXITCODE -ne 0){throw "fold$fold python exit=$LASTEXITCODE"}
    $audit=Join-Path $out 'AUDIT.json'
    if(-not (Test-Path -LiteralPath $audit)){throw "audit absent fold$fold"}
    $a=Get-Content -LiteralPath $audit -Raw|ConvertFrom-Json
    if($a.final_heldout_eeg_reads -ne 0){throw "heldout gate fold$fold"}
    Log "COMPLETE fold=$fold audit_sha=$((Get-FileHash -LiteralPath $audit -Algorithm SHA256).Hash.ToLowerInvariant()) utc=$([DateTime]::UtcNow.ToString('o'))"
  }
  Log "ALL_COMPLETE phase=evaluate utc=$([DateTime]::UtcNow.ToString('o'))"
  exit 0
} catch {
  if(Test-Path -LiteralPath $log){Log "EXCEPTION $($_.Exception.Message) utc=$([DateTime]::UtcNow.ToString('o'))"}
  exit 1
}
