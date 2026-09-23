$ErrorActionPreference='Stop'
$repo='D:\nips-temp\TotalP\P1\CRCICLR_PERSIST_INCREMENTAL_VALUE_V1'
$exp=Join-Path $repo 'experiments\persist_eeg_pc_mechanism_closure_seed0_v1'
$runtime='D:\nips-temp\TotalP\P1\pc_mechanism_closure_runtime'
$coupling='D:\nips-temp\TotalP\P1\protected_complement_coupling_runtime'
$name='PERSIST_EEG_PC_MECHANISM_CELL_EEGNET_SSVEP_F0_MEDIATION_V1'
$cell=Join-Path $runtime 'cells\eegnet\openbmi_ssvep\fold0_seed0'
$cache=Join-Path $coupling 'stage_cache\EEGNet\OpenBMI_SSVEP\fold0'
$upstream=Join-Path $coupling 'cells\eegnet\openbmi_ssvep\fold0_seed0.json'
$log=Join-Path $runtime 'scheduled_mediation_cell_EEGNet_OpenBMI_SSVEP_f0_v1.log'
$entry=Join-Path $exp 'code\run_mediation_cell_v2.py'
$sampling=Join-Path $exp 'code\sampling_core.py'
$cellData=Join-Path $exp 'code\cell_data_v1.py'
$mediation=Join-Path $exp 'code\mediation_core_v5.py'
$python='E:\Anaconda\envs\persist_stable_251\python.exe'
$analysisSha='ac12d33f7d30b2c846ec4c4fb07f8513bf25e372943a767b134883bf095a7c88'
$amendmentSha='aec1e604d38f1901bd9eca7e072e30a3d496a28061a726ebbe63133259b0e474'
$protocolSha='3ddb5006bf7159b00a0c80fed31856849ccdaa69ecf5060b0161c8673327e15c'
$implementationSha='4aab984969f4ba43128ed738a38ac138d94ee1c9b1252662af91b0c44e3f8eee'

function Get-Resources {
  $free=(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB
  $line=[string]((& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)|Select-Object -First 1)
  $gpu=0
  if(-not [int]::TryParse($line.Trim(),[ref]$gpu)){throw 'GPU query failed'}
  [pscustomobject]@{FreeGB=$free;GpuMiB=$gpu}
}
function Confirm-LocksAndUpstream {
  $gate=Get-Content (Join-Path $exp 'gate\GATE_AUDIT.json') -Raw|ConvertFrom-Json
  if($gate.status -ne 'PASSED' -or $gate.original_cells_complete -ne 20 -or
     $gate.layer_stage_cells_reconstructed -ne 120 -or $gate.final_heldout_accessed -ne $false -or
     $gate.upstream_protocol_sha256 -ne $protocolSha -or $gate.upstream_implementation_sha256 -ne $implementationSha){throw 'upstream gate invalid'}
  foreach($item in @(
    @{Path=(Join-Path $exp 'protocol\MECHANISM_CLOSURE_ANALYSIS_LOCK.md');Sha=$analysisSha},
    @{Path=(Join-Path $exp 'protocol\FINAL_PROJECTOR_AMENDMENT.md');Sha=$amendmentSha}
  )) { if((Get-FileHash $item.Path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $item.Sha){throw "protocol lock hash mismatch: $($item.Path)"} }
  $expected=@{
    $entry='b00479f7cdf6e3ea5b118773e6d3382e6fb29e9b39beb58b4842df10a54e3175';
    $sampling='7b8f5b94caca3924de3c714a780987a52337359963f38d5135c78410c8b1a631';
    $cellData='c5e4f90865445bcc1ec313b1ca5b1970cf484a9ef7f44e8aa1bf38a0fd721416';
    $mediation='22853e842a9dfcb5457c9eb96a84b738b4a2a47054b5778dd4e830bca6c27846'
  }
  foreach($path in $expected.Keys){if((Get-FileHash $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected[$path]){throw "source SHA mismatch: $path"}}
  $prior=Get-Content $upstream -Raw|ConvertFrom-Json
  if($prior.status -ne 'COMPLETE' -or $prior.model -ne 'EEGNet' -or $prior.task -ne 'OpenBMI_SSVEP' -or
     [int]$prior.fold -ne 0 -or [int]$prior.seed -ne 0 -or $prior.final_heldout_accessed -ne $false -or
     $prior.protocol_sha256 -ne $protocolSha -or $prior.implementation_sha256 -ne $implementationSha){throw 'exact frozen cell identity/hash gate failed'}
  if(-not (Test-Path -LiteralPath $cache)){throw 'frozen first-stage cache absent'}
  if(@(Get-ChildItem -LiteralPath $cache -Force).Count -eq 0){throw 'frozen first-stage cache is empty'}
}
function Other-Mechanism-Active {
  $task=@(Get-ScheduledTask | Where-Object {$_.TaskName -like 'PERSIST_EEG_PC_MECHANISM_CELL_*' -and $_.TaskName -ne $name -and $_.State -eq 'Running'})
  $proc=@(Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^python(\.exe)?$' -and $_.CommandLine -and $_.CommandLine -match 'persist_eeg_pc_mechanism_closure_seed0_v1|pc_mechanism_closure_runtime'})
  return ($task.Count -gt 0 -or $proc.Count -gt 0)
}

try {
  if(Test-Path $log){throw 'mediation log already exists'}
  if((Test-Path (Join-Path $cell 'MEDIATION_V2.json')) -or (Test-Path (Join-Path $cell 'MEDIATION_V2_FAIL_CLOSED.json')) -or (Test-Path (Join-Path $cell 'mediation_v2'))){throw 'target cell already has mediation evidence'}
  Confirm-LocksAndUpstream
  New-Item -ItemType File -Path $log -ErrorAction Stop|Out-Null
  "MEDIATION_SSVEP_F0_WAITING utc=$([DateTime]::UtcNow.ToString('o'))"|Out-File $log -Append -Encoding utf8
  while($true){
    Confirm-LocksAndUpstream
    $r=Get-Resources
    $blocked=Other-Mechanism-Active
    if($r.FreeGB -ge 40 -and $r.GpuMiB -ge 16000 -and -not $blocked){
      "MEDIATION_SSVEP_F0_LAUNCH free_gb=$([math]::Round($r.FreeGB,2)) gpu_mib=$($r.GpuMiB) utc=$([DateTime]::UtcNow.ToString('o'))"|Out-File $log -Append -Encoding utf8
      break
    }
    Start-Sleep -Seconds 20
  }
  $env:PERSIST_SOURCE_REPO=$repo
  $env:COUPLING_RUNTIME=$coupling
  $env:MECHANISM_RUNTIME=$runtime
  $env:PYTHONUNBUFFERED='1'
  $cmd='"'+$python+'" -u "'+$entry+'" --model EEGNet --task OpenBMI_SSVEP --fold 0 >> "'+$log+'" 2>&1'
  & cmd.exe /d /c $cmd
  if($LASTEXITCODE -ne 0){throw "mediation process exit=$LASTEXITCODE"}
  $m=Get-Content (Join-Path $cell 'MEDIATION_V2.json') -Raw|ConvertFrom-Json
  if($m.status -ne 'MEDIATION_PHASE_COMPLETE_PENDING_OTHER_REQUIRED_ANALYSES' -or $m.model -ne 'EEGNet' -or
     $m.task -ne 'OpenBMI_SSVEP' -or [int]$m.fold -ne 0 -or [int]$m.stage_count -ne 4 -or
     $m.final_heldout_accessed -ne $false){throw 'mediation output identity/status validation failed'}
  "MEDIATION_SSVEP_F0_COMPLETE utc=$([DateTime]::UtcNow.ToString('o'))"|Out-File $log -Append -Encoding utf8
  exit 0
} catch {
  if(Test-Path $log){"MEDIATION_SSVEP_F0_EXCEPTION $_ utc=$([DateTime]::UtcNow.ToString('o'))"|Out-File $log -Append -Encoding utf8}
  exit 1
}
