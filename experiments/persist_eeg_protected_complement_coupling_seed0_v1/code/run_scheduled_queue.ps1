<# Run one frozen coupling audit operation outside the SSH job object. #>
[CmdletBinding()]
param(
  [Parameter(Mandatory=$true)][ValidateSet('lock','aggregate','cell')][string]$Mode,
  [ValidateSet('EEGNet','EEGConformer')][string]$Model,
  [ValidateSet('OpenBMI_MI','OpenBMI_SSVEP')][string]$Task,
  [int]$Fold=-1
)
$ErrorActionPreference='Stop'
$code=$PSScriptRoot
$runtime='D:\nips-temp\TotalP\P1\protected_complement_coupling_runtime'
$python='E:\Anaconda\envs\persist_stable_251\python.exe'
$env:PERSIST_SOURCE_REPO='D:\nips-temp\TotalP\P1\CRCICLR_PERSIST_INCREMENTAL_VALUE_V1'
$env:COUPLING_RUNTIME=$runtime
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
$env:PYTHONUNBUFFERED='1'
New-Item -ItemType Directory -Force -Path $runtime | Out-Null
if($Mode -eq 'cell' -and ([string]::IsNullOrWhiteSpace($Model) -or [string]::IsNullOrWhiteSpace($Task) -or $Fold -notin 0..4)) { throw 'cell mode requires Model, Task, and Fold 0..4' }
$suffix=if($Mode -eq 'cell') { "cell_${Model}_${Task}_f${Fold}" } else { $Mode }
$log=Join-Path $runtime ("scheduled_coupling_${suffix}.log")
$entry=Join-Path $code 'run_coupling.py'
$arguments=if($Mode -eq 'cell') { "--mode cell --model $Model --task $Task --fold $Fold" } else { "--mode $Mode" }
try {
  "COUPLING_START mode=$Mode utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -FilePath $log -Append -Encoding utf8
  $command='"'+$python+'" -u "'+$entry+'" '+$arguments+' >> "'+$log+'" 2>&1'
  & cmd.exe /d /c $command
  if($LASTEXITCODE -ne 0) { throw "coupling Python exited $LASTEXITCODE" }
  "COUPLING_END mode=$Mode exit=0 utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -FilePath $log -Append -Encoding utf8
  exit 0
} catch {
  "COUPLING_EXCEPTION mode=$Mode $_" | Out-File -FilePath $log -Append -Encoding utf8
  exit 1
}
