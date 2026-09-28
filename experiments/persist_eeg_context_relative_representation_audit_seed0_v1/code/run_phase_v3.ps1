param(
  [ValidateSet('train','geometry','outer','localization')][string]$Phase,
  [ValidateRange(0,4)][int]$Fold
)
$ErrorActionPreference='Stop'
$root='D:\nips-temp\TotalP\P1\context_relative_representation_runtime'
$py='E:\Anaconda\envs\persist_stable_251\python.exe'
$source=Join-Path $root 'source\code'
$stem="${Phase}_fold${Fold}_v1"
$stdout=Join-Path $root "logs\${stem}.stdout.log"
$stderr=Join-Path $root "logs\${stem}.stderr.log"
$status=Join-Path $root "logs\${stem}.status.log"
foreach($p in @($stdout,$stderr,$status)){if(Test-Path -LiteralPath $p){throw "log exists: $p"}}
if($Phase -eq 'train'){$out=Join-Path $root "train_lock\fold${Fold}.json"}
elseif($Phase -eq 'geometry'){$out=Join-Path $root "geometry\fold${Fold}"}
elseif($Phase -eq 'localization'){$out=Join-Path $root "localization\fold${Fold}"}
else{$out=Join-Path $root "outer\fold${Fold}"}
if(Test-Path -LiteralPath $out){throw "output exists: $out"}
if($Phase -in @('outer','localization')){
  & (Join-Path $root 'source\code\verify_train_lock_v1.ps1') -Root $root | Out-Null
}
$ram=(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB
$gpu=[int]((& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | Select-Object -First 1).Trim())
if($ram -lt 24 -or $gpu -lt 12000){throw "resource gate ram_gb=$ram gpu_mib=$gpu"}
"START phase=$Phase fold=$Fold utc=$([DateTime]::UtcNow.ToString('o')) ram_gb=$ram gpu_mib=$gpu" | Out-File -LiteralPath $status -Encoding utf8
$env:REL_RUNTIME=$root
if($Phase -eq 'geometry'){$script=Join-Path $source 'geometry_v2.py';$args=@('-u',('"'+$script+'"'),'--fold',"$Fold")}
elseif($Phase -eq 'localization'){$script=Join-Path $source 'localization_v1.py';$args=@('-u',('"'+$script+'"'),'--fold',"$Fold")}
else{$script=Join-Path $source 'aggregate.py';$args=@('-u',('"'+$script+'"'),$Phase,'--fold',"$Fold")}
$p=Start-Process -FilePath $py -ArgumentList $args -Wait -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr
"EXIT=$($p.ExitCode) utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -LiteralPath $status -Append -Encoding utf8
if($p.ExitCode -ne 0){throw "phase failed exit=$($p.ExitCode) stderr=$stderr"}
