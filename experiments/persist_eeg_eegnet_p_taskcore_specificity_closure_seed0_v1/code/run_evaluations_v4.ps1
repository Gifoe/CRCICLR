$ErrorActionPreference='Stop'
$repo='D:\nips-temp\TotalP\P1\CRCICLR_PERSIST_INCREMENTAL_VALUE_V1'
$exp=Join-Path $repo 'experiments\persist_eeg_eegnet_p_taskcore_specificity_closure_seed0_v1'
$runtime='D:\nips-temp\TotalP\P1\p_taskcore_specificity_runtime'
$entry=Join-Path $exp 'code\run_v4.py'
$python='E:\Anaconda\envs\persist_stable_251\python.exe'
$log=Join-Path $runtime 'evaluate_all_v4.log'
$env:PERSIST_SOURCE_REPO=$repo
$env:P_SEMANTICS_RUNTIME='D:\nips-temp\TotalP\P1\p_semantics_canonical_geometry_runtime'
$env:P_TASKCORE_RUNTIME=$runtime
$env:P_TASKCORE_EVAL_FOLDER='evaluation_v4'
$env:PYTHONUNBUFFERED='1'
function Log([string]$message){$message|Out-File -LiteralPath $log -Append -Encoding utf8}
try {
  if(Test-Path -LiteralPath $log){throw 'log exists; refuse duplicate'}
  if(-not (Test-Path -LiteralPath $python) -or -not (Test-Path -LiteralPath $entry)){throw 'python or entry absent'}
  foreach($fold in 0..4){
    $recon=Join-Path $runtime "reconstruction\fold$fold.json"
    if(-not (Test-Path -LiteralPath $recon)){throw "reconstruction absent fold$fold"}
    $j=Get-Content -LiteralPath $recon -Raw | ConvertFrom-Json
    if(-not $j.G_exact_source_geometry -or $j.final_heldout_eeg_reads -ne 0){throw "gate failed fold$fold"}
    if(Test-Path -LiteralPath (Join-Path $runtime "evaluation_v4\fold$fold")){throw "new output exists fold$fold"}
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
    $partLog=Join-Path $runtime "evaluate_fold$($fold)_v4.log"
    if(Test-Path -LiteralPath $partLog){throw "fold log exists $fold"}
    Log "LAUNCH fold=$fold ram_gb=$([math]::Round($ram,2)) gpu_mib=$gpu utc=$([DateTime]::UtcNow.ToString('o'))"
    $command='"'+$python+'" -u "'+$entry+'" evaluate --fold '+$fold+' > "'+$partLog+'" 2>&1'
    & cmd.exe /d /c $command
    if($LASTEXITCODE -ne 0){throw "fold$fold python exit=$LASTEXITCODE"}
    $audit=Join-Path $runtime "evaluation_v4\fold$fold\AUDIT.json"
    if(-not (Test-Path -LiteralPath $audit)){throw "audit absent fold$fold"}
    $a=Get-Content -LiteralPath $audit -Raw | ConvertFrom-Json
    if($a.final_heldout_eeg_reads -ne 0 -or -not $a.G_exact_source_geometry){throw "audit failure fold$fold"}
    Log "COMPLETE fold=$fold audit_sha=$((Get-FileHash -LiteralPath $audit -Algorithm SHA256).Hash.ToLowerInvariant()) utc=$([DateTime]::UtcNow.ToString('o'))"
  }
  Log "ALL_COMPLETE utc=$([DateTime]::UtcNow.ToString('o'))"
  exit 0
} catch {
  if(Test-Path -LiteralPath $log){Log "EXCEPTION $($_.Exception.Message) utc=$([DateTime]::UtcNow.ToString('o'))"}
  exit 1
}
