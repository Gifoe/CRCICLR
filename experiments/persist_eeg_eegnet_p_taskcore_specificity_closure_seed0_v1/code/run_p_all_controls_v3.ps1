$ErrorActionPreference='Stop'
$repo='D:\nips-temp\TotalP\P1\CRCICLR_PERSIST_INCREMENTAL_VALUE_V1'
$exp=Join-Path $repo 'experiments\persist_eeg_eegnet_p_taskcore_specificity_closure_seed0_v1'
$runtime='D:\nips-temp\TotalP\P1\p_taskcore_specificity_runtime'
$entry=Join-Path $exp 'code\run_v3.py'
$python='E:\Anaconda\envs\persist_stable_251\python.exe'
$log=Join-Path $runtime 'p_all_controls_all_v3.log'
$env:PERSIST_SOURCE_REPO=$repo
$env:P_SEMANTICS_RUNTIME='D:\nips-temp\TotalP\P1\p_semantics_canonical_geometry_runtime'
$env:P_TASKCORE_RUNTIME=$runtime
$env:PYTHONUNBUFFERED='1'
function Log([string]$message){$message|Out-File -LiteralPath $log -Append -Encoding utf8}
try {
  if(Test-Path -LiteralPath $log){throw 'log exists; refuse duplicate'}
  if(-not (Test-Path -LiteralPath $python) -or -not (Test-Path -LiteralPath $entry)){throw 'python or entry absent'}
  foreach($fold in 0..4){
    if(-not (Test-Path -LiteralPath (Join-Path $runtime "evaluation\fold$fold\AUDIT.json"))){throw "main evaluation absent fold$fold"}
    if(Test-Path -LiteralPath (Join-Path $runtime "p_all_controls\fold$fold")){throw "P_ALL controls exist fold$fold"}
  }
  Log "START entry_sha=$((Get-FileHash -LiteralPath $entry -Algorithm SHA256).Hash.ToLowerInvariant()) utc=$([DateTime]::UtcNow.ToString('o'))"
  foreach($fold in 0..4){
    while($true){
      $ram=(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB
      $gpu=[int]([string]((& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)|Select-Object -First 1)).Trim()
      $heavy=@(Get-CimInstance Win32_Process|Where-Object{$_.Name -match '^python(\.exe)?$' -and $_.CommandLine -and $_.CommandLine -match 'persist_eeg_pc_mechanism_closure_seed0_v1|p_semantics_canonical_geometry_runtime|p_taskcore_specificity_runtime|persist_eeg_eegnet_p_taskcore_specificity_closure_seed0_v1'})
      if($ram -ge 40 -and $gpu -ge 16000 -and $heavy.Count -eq 0){break}
      Log "WAIT fold=$fold ram_gb=$([math]::Round($ram,2)) gpu_mib=$gpu heavy=$($heavy.Count) utc=$([DateTime]::UtcNow.ToString('o'))"
      Start-Sleep -Seconds 30
    }
    $partLog=Join-Path $runtime "p_all_controls_fold$($fold)_v3.log"
    if(Test-Path -LiteralPath $partLog){throw "fold log exists $fold"}
    Log "LAUNCH fold=$fold ram_gb=$([math]::Round($ram,2)) gpu_mib=$gpu utc=$([DateTime]::UtcNow.ToString('o'))"
    $command='"'+$python+'" -u "'+$entry+'" p-all-controls --fold '+$fold+' > "'+$partLog+'" 2>&1'
    & cmd.exe /d /c $command
    if($LASTEXITCODE -ne 0){throw "fold$fold python exit=$LASTEXITCODE"}
    $audit=Join-Path $runtime "p_all_controls\fold$fold\AUDIT.json"
    if(-not (Test-Path -LiteralPath $audit)){throw "audit absent fold$fold"}
    $a=Get-Content -LiteralPath $audit -Raw | ConvertFrom-Json
    if($a.final_heldout_eeg_reads -ne 0){throw "heldout violation fold$fold"}
    Log "COMPLETE fold=$fold audit_sha=$((Get-FileHash -LiteralPath $audit -Algorithm SHA256).Hash.ToLowerInvariant()) utc=$([DateTime]::UtcNow.ToString('o'))"
  }
  Log "ALL_COMPLETE utc=$([DateTime]::UtcNow.ToString('o'))"
  exit 0
} catch {
  if(Test-Path -LiteralPath $log){Log "EXCEPTION $($_.Exception.Message) utc=$([DateTime]::UtcNow.ToString('o'))"}
  exit 1
}
