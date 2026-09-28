$ErrorActionPreference='Stop'
$root='D:\nips-temp\TotalP\P1\context_relative_representation_runtime'
$script=Join-Path $root 'source\code\run_v2.py'
$py='E:\Anaconda\envs\persist_stable_251\python.exe'
$out=Join-Path $root 'features\fold0_train_geometry_v2'
$stdout=Join-Path $root 'logs\extract_fold0_train_geometry_v2.stdout.log'
$stderr=Join-Path $root 'logs\extract_fold0_train_geometry_v2.stderr.log'
$status=Join-Path $root 'logs\extract_fold0_train_geometry_v2.status.log'
foreach($p in @($out,$stdout,$stderr,$status)){if(Test-Path -LiteralPath $p){throw "already exists: $p"}}
$ram=(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB
$gpu=[int]((& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | Select-Object -First 1).Trim())
if($ram -lt 24 -or $gpu -lt 12000){throw "resource gate ram=$ram gpu=$gpu"}
"START utc=$([DateTime]::UtcNow.ToString('o')) ram_gb=$ram gpu_mib=$gpu" | Out-File -LiteralPath $status -Encoding utf8
$env:SEVEN_REPO='D:\nips-temp\TotalP\P1\CRCICLR_BACKBONE_GEN_WORK'
$env:REL_RUNTIME=$root
$p=Start-Process -FilePath $py -ArgumentList @('-u',('"'+$script+'"'),'--fold','0','--role','TRAIN_GEOMETRY','--attempt','v2') -Wait -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr
"EXIT=$($p.ExitCode) utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -LiteralPath $status -Append -Encoding utf8
if($p.ExitCode -ne 0){throw "extraction failed exit=$($p.ExitCode); stderr=$stderr"}
