<#
Runs a resumable Experiment 1 queue outside an SSH-session job object.

The wrapper changes process lifetime and logging only.  It does not alter a
checkpoint, data split, selector, rank, statistical unit, or analysis code.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('probe', 'original', 'eegnet', 'ordered')]
    [string]$Mode
)

$ErrorActionPreference = 'Stop'
$code = $PSScriptRoot
$repo = Split-Path -Parent (Split-Path -Parent $code)
$runtime = 'D:\nips-temp\TotalP\P1\persist_incremental_value_runtime\analysis'
$python = 'E:\Anaconda\envs\persist_stable_251\python.exe'
$log = Join-Path $runtime ("scheduled_{0}.log" -f $Mode)

New-Item -ItemType Directory -Path $runtime -Force | Out-Null
$env:INCREMENTAL_DIRECT_ANALYSIS = '1'
$env:CUDA_VISIBLE_DEVICES = '0'
# Preserve the frozen per-cell numerical threading configuration.  Speed comes
# from independent resumable cells, not from changing BLAS reduction order.
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
$env:PYTHONUNBUFFERED = '1'

Push-Location $code
try {
    "SCHEDULED_QUEUE_START mode=$Mode utc=$([DateTime]::UtcNow.ToString('o'))" | Tee-Object -FilePath $log -Append
    switch ($Mode) {
        'probe' {
            & nvidia-smi 2>&1 | Tee-Object -FilePath $log -Append
        }
        'original' {
            $env:INCREMENTAL_ANALYSIS_WORKERS = '3'
            $command = '"' + $python + '" -u "' + (Join-Path $code 'run_analysis_parallel.py') + '" >> "' + $log + '" 2>&1'
            & cmd.exe /d /c $command
        }
        'eegnet' {
            $command = '"' + $python + '" -u "' + (Join-Path $code 'run_pu_u_interpretation_queue.py') + '" --models EEGNet --workers 4 >> "' + $log + '" 2>&1'
            & cmd.exe /d /c $command
        }
        'ordered' {
            $command = '"' + $python + '" -u "' + (Join-Path $code 'run_ordered_pu_u_analysis.py') + '" --workers 3 --poll-seconds 30 --external-running EEGNet >> "' + $log + '" 2>&1'
            & cmd.exe /d /c $command
        }
    }
    $exitCode = $LASTEXITCODE
    "SCHEDULED_QUEUE_END mode=$Mode exit=$exitCode utc=$([DateTime]::UtcNow.ToString('o'))" | Tee-Object -FilePath $log -Append
    exit $exitCode
}
catch {
    "SCHEDULED_QUEUE_EXCEPTION mode=$Mode $_" | Tee-Object -FilePath $log -Append
    exit 1
}
finally {
    Pop-Location
}
