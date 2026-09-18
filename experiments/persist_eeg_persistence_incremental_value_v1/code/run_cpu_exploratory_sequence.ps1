param(
    [ValidateRange(1, 2)]
    [int]$Workers = 2
)

$ErrorActionPreference = 'Stop'
$base = 'D:\chenyu-iclr\BASELINE'
$repo = 'D:\chenyu-iclr\CRCICLR\incremental_interpretation_work'
$code = Join-Path $base 'code'
$env:PERSIST_LOCAL_BASELINE_ROOT = $base
$env:PERSIST_SOURCE_REPO = $repo
$env:PERSIST_ANALYSIS_ROOT = $base
$env:INCREMENTAL_DIRECT_ANALYSIS = '1'
$env:PERSIST_EXECUTION_MODE = 'CPU_EXPLORATORY_NON_EQUIVALENT'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'

function Cell-Count([string]$model) {
    $path = Join-Path $base ("pu_u_interpretation\\" + $model.ToLowerInvariant())
    return @(Get-ChildItem -LiteralPath $path -Recurse -Filter '*.json' -ErrorAction SilentlyContinue).Count
}

function Wait-For-Model([string]$model) {
    while ((Cell-Count $model) -lt 60) {
        $queue = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
            Where-Object { $_.CommandLine -match 'run_pu_u_interpretation_queue.py' -and $_.CommandLine -match $model }
        if (-not $queue) { throw "$model queue stopped before reaching 60 cells" }
        Start-Sleep -Seconds 10
    }
}

Wait-For-Model 'EEGNet'
& python -u (Join-Path $code 'run_pu_u_interpretation.py') aggregate --models EEGNet
if ($LASTEXITCODE -ne 0) { throw 'EEGNet aggregation failed' }
& python -u (Join-Path $code 'run_pu_u_interpretation_queue.py') --models EEGConformer --workers $Workers
if ($LASTEXITCODE -ne 0) { throw 'EEGConformer queue failed' }
& python -u (Join-Path $code 'run_pu_u_interpretation.py') aggregate --models EEGNet EEGConformer
if ($LASTEXITCODE -ne 0) { throw 'combined aggregation failed' }
Write-Output 'CPU_EXPLORATORY_SEQUENCE_COMPLETE'
