<# Runs exactly one frozen arbitration command outside the SSH job object. #>
[CmdletBinding()]
param(
  [Parameter(Mandatory=$true)][ValidateSet('lock','aggregate','probe','cell')][string]$Mode,
  [ValidateSet('EEGNet','EEGConformer')][string]$Model,
  [ValidateSet('OpenBMI_MI','OpenBMI_SSVEP')][string]$Task,
  [int]$Fold=-1
)
$ErrorActionPreference='Stop'
$code=$PSScriptRoot
$runtime='D:\nips-temp\TotalP\P1\protected_arbitration_reliability_runtime'
$python='E:\Anaconda\envs\persist_stable_251\python.exe'
$env:PERSIST_SOURCE_REPO='D:\nips-temp\TotalP\P1\CRCICLR_PERSIST_INCREMENTAL_VALUE_V1'
$env:ARBITRATION_RUNTIME=$runtime
$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1';$env:NUMEXPR_NUM_THREADS='1';$env:PYTHONUNBUFFERED='1'
New-Item -ItemType Directory -Force -Path $runtime | Out-Null
if($Mode -eq 'cell' -and ([string]::IsNullOrWhiteSpace($Model) -or [string]::IsNullOrWhiteSpace($Task) -or $Fold -notin 0..4)){throw 'cell mode requires Model, Task, and Fold 0..4'}
$suffix=if($Mode -eq 'cell'){"cell_{0}_{1}_f{2}" -f $Model,$Task,$Fold}else{$Mode}
$log=Join-Path $runtime ("scheduled_arbitration_{0}.log" -f $suffix)
Push-Location $code
try {
  "ARBITRATION_START mode=$Mode utc=$([DateTime]::UtcNow.ToString('o'))" | Tee-Object -FilePath $log -Append
  $args=if($Mode -eq 'probe'){@('cell','EEGNet','OpenBMI_MI','0')}elseif($Mode -eq 'cell'){@('cell',$Model,$Task,"$Fold")}else{@($Mode)}
  & $python -u (Join-Path $code 'run_arbitration.py') @args *>> $log
  $exitCode=$LASTEXITCODE
  "ARBITRATION_END mode=$Mode exit=$exitCode utc=$([DateTime]::UtcNow.ToString('o'))" | Tee-Object -FilePath $log -Append
  exit $exitCode
} catch { "ARBITRATION_EXCEPTION mode=$Mode $_" | Tee-Object -FilePath $log -Append; exit 1 }
finally { Pop-Location }
