<# Runs one frozen pathway-audit command outside the SSH job object. #>
[CmdletBinding()]
param([Parameter(Mandatory=$true)][ValidateSet('lock','all','aggregate','probe')][string]$Mode)
$ErrorActionPreference='Stop'
$code=$PSScriptRoot
$runtime='D:\nips-temp\TotalP\P1\protected_pathway_mechanism_runtime'
$python='E:\Anaconda\envs\persist_stable_251\python.exe'
$env:PERSIST_SOURCE_REPO='D:\nips-temp\TotalP\P1\CRCICLR_PERSIST_INCREMENTAL_VALUE_V1'
$env:PEEH_EXTENSION_ROOT=Join-Path $env:PERSIST_SOURCE_REPO 'experiments\persist_eeg_eegconformer_fbcnet_peeh_pswa_v1'
$env:PEEH_EEGNET_ROOT=Join-Path $env:PERSIST_SOURCE_REPO 'experiments\persist_eeg_crossbackbone_peeh_v1'
$env:PATHWAY_RUNTIME=$runtime
$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1';$env:NUMEXPR_NUM_THREADS='1';$env:PYTHONUNBUFFERED='1'
New-Item -ItemType Directory -Force -Path $runtime | Out-Null
$log=Join-Path $runtime ("scheduled_{0}.log" -f $Mode)
Push-Location $code
try {
  "PATHWAY_START mode=$Mode utc=$([DateTime]::UtcNow.ToString('o'))" | Tee-Object -FilePath $log -Append
  $arg=if($Mode -eq 'probe'){'cell EEGNet OpenBMI_MI 0'}elseif($Mode -eq 'all'){'run-all'}else{$Mode}
  & cmd.exe /d /c ('"'+$python+'" -u "'+(Join-Path $code 'run_protected_pathway.py')+'" '+$arg+' >> "'+$log+'" 2>&1')
  $exitCode=$LASTEXITCODE; "PATHWAY_END mode=$Mode exit=$exitCode utc=$([DateTime]::UtcNow.ToString('o'))" | Tee-Object -FilePath $log -Append; exit $exitCode
} catch { "PATHWAY_EXCEPTION mode=$Mode $_" | Tee-Object -FilePath $log -Append; exit 1 }
finally { Pop-Location }
