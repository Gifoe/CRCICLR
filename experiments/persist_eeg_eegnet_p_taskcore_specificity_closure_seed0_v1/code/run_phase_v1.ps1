param(
  [ValidateSet('reconstruct','evaluate')][string]$Phase,
  [ValidateRange(0,4)][int]$Fold
)
$ErrorActionPreference='Stop'
$repo='D:\nips-temp\TotalP\P1\CRCICLR_PERSIST_INCREMENTAL_VALUE_V1'
$exp=Join-Path $repo 'experiments\persist_eeg_eegnet_p_taskcore_specificity_closure_seed0_v1'
$runtime='D:\nips-temp\TotalP\P1\p_taskcore_specificity_runtime'
$sourceRuntime='D:\nips-temp\TotalP\P1\p_semantics_canonical_geometry_runtime'
$python='E:\Anaconda\envs\persist_stable_251\python.exe'
$entry=Join-Path $exp 'code\run.py'
$log=Join-Path $runtime "$($Phase)_fold$Fold`_v1.log"
$output=if($Phase -eq 'reconstruct') {Join-Path $runtime "reconstruction\fold$Fold.json"} else {Join-Path $runtime "evaluation\fold$Fold"}
$env:PERSIST_SOURCE_REPO=$repo
$env:P_SEMANTICS_RUNTIME=$sourceRuntime
$env:P_TASKCORE_RUNTIME=$runtime
$env:PYTHONUNBUFFERED='1'
function Log([string]$message){$message|Out-File -LiteralPath $log -Append -Encoding utf8}
try {
  if(Test-Path -LiteralPath $log){throw 'log exists; refuse duplicate'}
  if(Test-Path -LiteralPath $output){throw 'output exists; refuse duplicate'}
  if(-not (Test-Path -LiteralPath $python)){throw 'python environment absent'}
  if(-not (Test-Path -LiteralPath $entry)){throw 'entry absent'}
  [IO.Directory]::CreateDirectory($runtime)|Out-Null
  Log "START phase=$Phase fold=$Fold entry_sha=$((Get-FileHash -LiteralPath $entry -Algorithm SHA256).Hash.ToLowerInvariant()) utc=$([DateTime]::UtcNow.ToString('o'))"
  while($true){
    $ram=(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB
    $gpu=[int]([string]((& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)|Select-Object -First 1)).Trim()
    $heavy=@(Get-CimInstance Win32_Process|Where-Object{$_.Name -match '^python(\.exe)?$' -and $_.CommandLine -and $_.CommandLine -match 'persist_eeg_pc_mechanism_closure_seed0_v1|p_semantics_canonical_geometry_runtime|p_taskcore_specificity_runtime|persist_eeg_eegnet_p_taskcore_specificity_closure_seed0_v1'})
    if($ram -ge 40 -and $gpu -ge 16000 -and $heavy.Count -eq 0){break}
    Log "WAIT_RESOURCE ram_gb=$([math]::Round($ram,2)) gpu_mib=$gpu heavy=$($heavy.Count) utc=$([DateTime]::UtcNow.ToString('o'))"
    Start-Sleep -Seconds 20
  }
  Log "LAUNCH ram_gb=$([math]::Round($ram,2)) gpu_mib=$gpu utc=$([DateTime]::UtcNow.ToString('o'))"
  $command='"'+$python+'" -u "'+$entry+'" '+$Phase+' --fold '+$Fold+' >> "'+$log+'" 2>&1'
  & cmd.exe /d /c $command
  if($LASTEXITCODE -ne 0){throw "python exit=$LASTEXITCODE"}
  if(-not (Test-Path -LiteralPath $output)){throw 'expected output absent'}
  Log "COMPLETE output=$output utc=$([DateTime]::UtcNow.ToString('o'))"
  exit 0
} catch {
  if(Test-Path -LiteralPath $log){Log "EXCEPTION $($_.Exception.Message) utc=$([DateTime]::UtcNow.ToString('o'))"}
  exit 1
}
