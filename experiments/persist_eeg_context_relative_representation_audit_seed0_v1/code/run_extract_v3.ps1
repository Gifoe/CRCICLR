param(
  [ValidateRange(0,4)][int]$Fold,
  [ValidateSet('TRAIN_GEOMETRY','OUTER_DEVELOPMENT')][string]$Role
)
$ErrorActionPreference='Stop'
$root='D:\nips-temp\TotalP\P1\context_relative_representation_runtime'
$script=Join-Path $root 'source\code\run_v2.py'
$py='E:\Anaconda\envs\persist_stable_251\python.exe'
$name="fold${Fold}_$($Role.ToLowerInvariant())"
$out=Join-Path $root "features\$name"
$stdout=Join-Path $root "logs\extract_${name}_v3.stdout.log"
$stderr=Join-Path $root "logs\extract_${name}_v3.stderr.log"
$status=Join-Path $root "logs\extract_${name}_v3.status.log"
foreach($p in @($out,$stdout,$stderr,$status)){if(Test-Path -LiteralPath $p){throw "already exists: $p"}}
if($Role -eq 'OUTER_DEVELOPMENT'){
  & (Join-Path $root 'source\code\verify_train_lock_v1.ps1') -Root $root | Out-Null
}
$ram=(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB
$gpu=[int]((& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | Select-Object -First 1).Trim())
if($ram -lt 24 -or $gpu -lt 12000){throw "resource gate ram_gb=$ram gpu_mib=$gpu"}
"START fold=$Fold role=$Role utc=$([DateTime]::UtcNow.ToString('o')) ram_gb=$ram gpu_mib=$gpu" | Out-File -LiteralPath $status -Encoding utf8
$env:SEVEN_REPO='D:\nips-temp\TotalP\P1\CRCICLR_BACKBONE_GEN_WORK'
$env:REL_RUNTIME=$root
$p=Start-Process -FilePath $py -ArgumentList @('-u',('"'+$script+'"'),'--fold',"$Fold",'--role',$Role,'--attempt','v1') -Wait -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr
"EXIT=$($p.ExitCode) utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -LiteralPath $status -Append -Encoding utf8
if($p.ExitCode -ne 0){throw "extraction failed exit=$($p.ExitCode); stderr=$stderr"}
