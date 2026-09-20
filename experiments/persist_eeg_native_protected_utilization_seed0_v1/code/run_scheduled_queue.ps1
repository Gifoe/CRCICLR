<# Runs the frozen native Protected audit outside an SSH job object. #>
[CmdletBinding()]
param([Parameter(Mandatory=$true)][ValidateSet('probe','all','aggregate')][string]$Mode)

$ErrorActionPreference = 'Stop'
$code = $PSScriptRoot
$repo = Split-Path -Parent (Split-Path -Parent $code)
$runtime = 'D:\nips-temp\TotalP\P1\native_protected_utilization_runtime'
$python = 'E:\Anaconda\envs\persist_stable_251\python.exe'
$log = Join-Path $runtime ("scheduled_{0}.log" -f $Mode)
New-Item -ItemType Directory -Path $runtime -Force | Out-Null
$env:PERSIST_SOURCE_REPO = 'D:\nips-temp\TotalP\P1\CRCICLR_PERSIST_INCREMENTAL_VALUE_V1'
$env:PEEH_EXTENSION_ROOT = Join-Path $env:PERSIST_SOURCE_REPO 'experiments\persist_eeg_eegconformer_fbcnet_peeh_pswa_v1'
$env:PEEH_EEGNET_ROOT = Join-Path $env:PERSIST_SOURCE_REPO 'experiments\persist_eeg_crossbackbone_peeh_v1'
$env:NATIVE_PROTECTED_RUNTIME = $runtime
$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'; $env:NUMEXPR_NUM_THREADS='1'; $env:PYTHONUNBUFFERED='1'

Push-Location $code
try {
  "NATIVE_AUDIT_START mode=$Mode utc=$([DateTime]::UtcNow.ToString('o'))" | Tee-Object -FilePath $log -Append
  if ($Mode -eq 'probe') { $args = 'cell EEGNet OpenBMI_MI 0' }
  elseif ($Mode -eq 'all') { $args = 'run-all' }
  else { $args = 'aggregate' }
  $command = '"' + $python + '" -u "' + (Join-Path $code 'run_native_protected_utilization.py') + '" ' + $args + ' >> "' + $log + '" 2>&1'
  & cmd.exe /d /c $command
  $exitCode=$LASTEXITCODE
  "NATIVE_AUDIT_END mode=$Mode exit=$exitCode utc=$([DateTime]::UtcNow.ToString('o'))" | Tee-Object -FilePath $log -Append
  exit $exitCode
} catch { "NATIVE_AUDIT_EXCEPTION mode=$Mode $_" | Tee-Object -FilePath $log -Append; exit 1 }
finally { Pop-Location }
