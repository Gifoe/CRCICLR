param(
  [ValidateRange(0,4)][int]$Fold,
  [ValidateSet('TRAIN_GEOMETRY','OUTER_DEVELOPMENT')][string]$Role
)
$ErrorActionPreference='Stop'
$root='D:\nips-temp\TotalP\P1\context_relative_representation_runtime'
$script=Join-Path $root 'source\code\run.py'
$py='E:\Anaconda\envs\persist_stable_251\python.exe'
$out=Join-Path $root ("features\fold{0}_{1}" -f $Fold,$Role.ToLowerInvariant())
$log=Join-Path $root ("logs\extract_fold{0}_{1}_v1.log" -f $Fold,$Role.ToLowerInvariant())
if(Test-Path -LiteralPath $out){throw "output exists: $out"}
if(Test-Path -LiteralPath $log){throw "log exists: $log"}
$ram=(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB
$gpu=[int]((& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | Select-Object -First 1).Trim())
if($ram -lt 24 -or $gpu -lt 12000){throw "extraction resource gate ram_gb=$ram gpu_mib=$gpu"}
"START fold=$Fold role=$Role ram_gb=$ram gpu_mib=$gpu utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -LiteralPath $log -Encoding utf8
$env:SEVEN_REPO='D:\nips-temp\TotalP\P1\CRCICLR_BACKBONE_GEN_WORK'
$env:REL_RUNTIME=$root
& $py $script --fold $Fold --role $Role *>> $log
$exit=$LASTEXITCODE
"EXIT=$exit utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -LiteralPath $log -Append -Encoding utf8
if($exit -ne 0){throw "extract failed exit=$exit"}
