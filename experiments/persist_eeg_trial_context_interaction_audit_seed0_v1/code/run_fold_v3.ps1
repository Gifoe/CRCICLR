param(
  [ValidateSet('train','pairing','outer')][string]$Phase,
  [ValidateRange(0,4)][int]$Fold
)
$ErrorActionPreference='Stop'
$root='D:\nips-temp\TotalP\P1\trial_context_interaction_runtime'
$py='E:\Anaconda\envs\persist_stable_251\python.exe'
$stem="${Phase}_fold${Fold}_v3"
$log=Join-Path $root 'logs'
$stdout=Join-Path $log "$stem.stdout.log"
$stderr=Join-Path $log "$stem.stderr.log"
$status=Join-Path $log "$stem.status.log"
foreach($p in @($stdout,$stderr,$status)){if(Test-Path -LiteralPath $p){throw "existing log: $p"}}
$output=if($Phase -eq 'train'){Join-Path $root "train_lock\fold$Fold.json"}elseif($Phase -eq 'pairing'){Join-Path $root "train_pairing\fold$Fold.json"}else{Join-Path $root "outer\fold$Fold.json"}
if(Test-Path -LiteralPath $output){throw "existing output: $output"}
if($Phase -eq 'outer'){
  foreach($i in 0..4){
    if(!(Test-Path -LiteralPath (Join-Path $root "train_lock\fold$i.json"))){throw "missing TRAIN lock $i"}
    if(!(Test-Path -LiteralPath (Join-Path $root "train_pairing\fold$i.json"))){throw "missing TRAIN pairing null $i"}
  }
}
$running=Get-ScheduledTask | Where-Object {$_.State -eq 'Running' -and $_.TaskName -match 'PERSIST_EEG_TRIAL_CONTEXT' -and $_.TaskName -ne "PERSIST_EEG_TRIAL_CONTEXT_${Phase}_F${Fold}_V3"}
if($running){throw "other interaction task running: $($running.TaskName -join ',')"}
$ram=(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB
$gpu=[int]((& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | Select-Object -First 1).Trim())
if($ram -lt 24 -or $gpu -lt 12000){throw "resource gate ram_gb=$ram gpu_mib=$gpu"}
"START utc=$([DateTime]::UtcNow.ToString('o')) phase=$Phase fold=$Fold ram_gb=$ram gpu_mib=$gpu" | Out-File -LiteralPath $status -Encoding utf8
$env:REL_RUNTIME='D:\nips-temp\TotalP\P1\context_relative_representation_runtime'
$env:INTERACTION_RUNTIME=$root
$script=Join-Path $root 'source\code\run_v3.py'
$p=Start-Process -FilePath $py -ArgumentList @('-u',('"'+$script+'"'),$Phase,"$Fold") -Wait -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr
"EXIT=$($p.ExitCode) utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -LiteralPath $status -Append -Encoding utf8
if($p.ExitCode -ne 0){throw "phase failed exit=$($p.ExitCode) stderr=$stderr"}

