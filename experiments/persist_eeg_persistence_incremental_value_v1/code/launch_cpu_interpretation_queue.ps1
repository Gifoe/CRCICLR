param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('EEGNet', 'EEGConformer')]
    [string]$Model,
    [ValidateRange(1, 2)]
    [int]$Workers = 2
)

$ErrorActionPreference = 'Stop'
$base = 'D:\chenyu-iclr\BASELINE'
$repo = 'D:\chenyu-iclr\CRCICLR\incremental_interpretation_work'
$env:PERSIST_LOCAL_BASELINE_ROOT = $base
$env:PERSIST_SOURCE_REPO = $repo
$env:PERSIST_ANALYSIS_ROOT = $base
$env:INCREMENTAL_DIRECT_ANALYSIS = '1'
$env:PERSIST_EXECUTION_MODE = 'CPU_EXPLORATORY_NON_EQUIVALENT'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$log = Join-Path $base "outputs\\cpu_exploratory_${Model}_${stamp}.log"
$err = Join-Path $base "outputs\\cpu_exploratory_${Model}_${stamp}.err.log"
$args = @('-u', (Join-Path $base 'code\\run_pu_u_interpretation_queue.py'), '--models', $Model, '--workers', "$Workers")
$process = Start-Process -FilePath (Get-Command python).Source -ArgumentList $args -RedirectStandardOutput $log -RedirectStandardError $err -PassThru -WindowStyle Hidden
[pscustomobject]@{ model = $Model; pid = $process.Id; log = $log; error_log = $err; workers = $Workers } | ConvertTo-Json -Compress
